"""Входные признаки, не зависящие от климат-поля.

Здесь строятся каналы истории и ковариаты горизонта, которые получают все модели с
многоканальным входом. Нормировки фиксированные и не зависят от окна, поэтому один и
тот же час всегда даёт одни и те же числа. Аномалии относительно климат-поля сюда не
входят: они есть только у модели с полем.

Каналы истории по порядку:

* ``t``, ``p``, ``rh`` - температура, давление и влажность в фиксированной нормировке;
* ``dd`` - дефицит точки росы;
* ``dP3``, ``dP24`` - изменение давления за 3 и 24 ч; канал равен нулю, если
  невалиден текущий час или час с нужным лагом;
* ``sin_d``, ``cos_d`` - фаза истинного солнечного времени;
* ``czp`` - косинус зенитного угла Солнца, ночью ноль;
* ``sin_y``, ``cos_y`` - фаза дня года;
* ``vt``, ``vp``, ``vr`` - маски наличия температуры, давления и влажности.

Невалидный час даёт ноль во всех каналах значений. Значение под нулевой маской не
участвует в расчёте вовсе, поэтому даже бесконечность или нечисло там не отражаются на
входе. На валидных часах результат такой же, как при умножении на маску.
"""
import math

import torch
import torch.nn.functional as F

from mayak.astro import astro_features, dewpoint_c

T_SCALE = 30.0
P_REF = 1013.0
P_SCALE = 50.0
DD_SCALE = 10.0
ELEV_SCALE = 1000.0
DP_SCALES = {3: 3.0, 24: 8.0}
DP_CLAMP = 4.0

HISTORY_CHANNELS = ("t", "p", "rh", "dd", "dP3", "dP24", "sin_d", "cos_d", "czp",
                    "sin_y", "cos_y", "vt", "vp", "vr")
FUTURE_CHANNELS = ("sin_d", "cos_d", "czp", "sin_y", "cos_y")
SITE_FEATURES = ("x", "y", "z", "elev")

N_HISTORY = len(HISTORY_CHANNELS)
N_FUTURE = len(FUTURE_CHANNELS)
N_SITE = len(SITE_FEATURES)


def lag_valid(v, k):
    """Маска часов, у которых валидны и сам час, и час на k раньше.

    Args:
        v: маска наличия, форма (..., L), последняя ось - время.
        k: лаг в часах, не меньше единицы.

    Returns:
        Маска той же формы. Первые k часов окна невалидны: их лаг лежит вне окна.
    """
    vs = F.pad(v, (k, 0))[..., :v.shape[-1]]
    return v * vs


def pressure_tendency(P, vp, k):
    """Изменение давления за k часов в фиксированной нормировке.

    Args:
        P: давление, гПа, форма (..., L).
        vp: маска наличия давления той же формы.
        k: лаг в часах, 3 или 24.

    Returns:
        Изменение давления, делённое на масштаб своего лага и ограниченное по модулю.
        Ноль, если невалиден текущий час или час с лагом.
    """
    ps = F.pad(P, (k, 0))[..., :P.shape[-1]]
    d = ((P - ps) / DP_SCALES[k]).clamp(-DP_CLAMP, DP_CLAMP)
    return torch.where(lag_valid(vp, k) > 0, d, torch.zeros_like(d))


def dewpoint_deficit(x, mask):
    """Дефицит точки росы в градусах Цельсия, не меньше нуля.

    На невалидных часах температура и влажность перед расчётом заменяются нулём, чтобы
    мусор под маской не превращался в бесконечность. На валидных часах значение такое
    же, как без замены. Результат на маску не умножается: это делает вызывающий код.

    Args:
        x: наблюдения, форма (..., L, 3): температура, давление, влажность.
        mask: маски наличия той же формы.

    Returns:
        Дефицит точки росы, форма (..., L).
    """
    T = torch.where(mask[..., 0] > 0, x[..., 0], torch.zeros_like(x[..., 0]))
    RH = torch.where(mask[..., 2] > 0, x[..., 2], torch.zeros_like(x[..., 2]))
    return (T - dewpoint_c(T, RH)).clamp(min=0.0)


def history_channels(x, mask, astro):
    """Каналы истории, не зависящие от климат-поля.

    Args:
        x: наблюдения, форма (B, L, 3): температура, давление, влажность.
        mask: маски наличия, форма (B, L, 3).
        astro: солнечно-календарные признаки часов истории, кортеж из шести тензоров
            формы (B, L): синус и косинус солнечного времени, косинус зенитного угла,
            он же без отрицательной части, синус и косинус фазы года.

    Returns:
        Словарь из имени канала в тензор формы (B, L). Имена и порядок - как в
        ``HISTORY_CHANNELS``.
    """
    T, P, RH = x[..., 0], x[..., 1], x[..., 2]
    vt, vp, vr = mask[..., 0], mask[..., 1], mask[..., 2]
    sin_d, cos_d, _cz, czp, sin_y, cos_y = astro
    zero = torch.zeros_like(T)
    return dict(
        t=torch.where(vt > 0, T / T_SCALE, zero),
        p=torch.where(vp > 0, (P - P_REF) / P_SCALE, zero),
        rh=torch.where(vr > 0, RH / 100.0 - 0.5, zero),
        dd=dewpoint_deficit(x, mask) / DD_SCALE * vt * vr,
        dP3=pressure_tendency(P, vp, 3),
        dP24=pressure_tendency(P, vp, 24),
        sin_d=sin_d, cos_d=cos_d, czp=czp, sin_y=sin_y, cos_y=cos_y,
        vt=vt, vp=vp, vr=vr)


def future_channels(astro):
    """Ковариаты часов горизонта: солнечное время, высота Солнца, фаза года.

    Args:
        astro: солнечно-календарные признаки часов горизонта, кортеж из шести тензоров
            формы (B, H) в том же порядке, что у истории.

    Returns:
        Словарь из имени ковариаты в тензор формы (B, H). Имена и порядок - как в
        ``FUTURE_CHANNELS``.
    """
    sin_d, cos_d, _cz, czp, sin_y, cos_y = astro
    return dict(sin_d=sin_d, cos_d=cos_d, czp=czp, sin_y=sin_y, cos_y=cos_y)


def _astro(batch, doy_key, hour_key, n):
    lat, lon = batch["lat"][:, None], batch["lon"][:, None]
    return astro_features(batch[doy_key][:, -n:], batch[hour_key][:, -n:], lat, lon)


def history_features(batch):
    """Каналы истории батча одним тензором.

    Календарь берётся для последних часов истории, выровненных по правому краю окна,
    по числу часов в наблюдениях.

    Args:
        batch: батч окон с наблюдениями, масками, календарём истории и координатами.

    Returns:
        Тензор формы (B, L, N_HISTORY) в порядке ``HISTORY_CHANNELS``, в типе
        наблюдений.
    """
    x, mask = batch["x_hist"], batch["mask_hist"]
    ch = history_channels(x, mask, _astro(batch, "doy_hist", "hour_hist", x.shape[1]))
    return torch.stack([ch[n] for n in HISTORY_CHANNELS], dim=-1).to(x.dtype)


def future_features(batch):
    """Ковариаты горизонта батча одним тензором.

    Args:
        batch: батч окон с календарём горизонта и координатами.

    Returns:
        Тензор формы (B, H, N_FUTURE) в порядке ``FUTURE_CHANNELS``, в типе наблюдений.
    """
    n = batch["doy_fut"].shape[1]
    ch = future_channels(_astro(batch, "doy_fut", "hour_fut", n))
    return torch.stack([ch[k] for k in FUTURE_CHANNELS], dim=-1).to(batch["x_hist"].dtype)


def site_features(lat, lon, elev):
    """Признаки точки: положение на единичной сфере и высота.

    Точка на сфере не имеет разрыва на линии перемены дат и не сжимается у полюсов,
    в отличие от широты и долготы как чисел.

    Args:
        lat: широта, градусы, форма (B,).
        lon: долгота, градусы, форма (B,).
        elev: высота над уровнем моря, метры, форма (B,).

    Returns:
        Тензор формы (B, N_SITE): три декартовы координаты точки на сфере и высота в
        километрах.
    """
    phi = lat * (math.pi / 180.0)
    lam = lon * (math.pi / 180.0)
    return torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam),
                        torch.sin(phi), elev / ELEV_SCALE], dim=-1)


def batch_site_features(batch):
    """Признаки точки для батча, в типе наблюдений.

    Args:
        batch: батч окон с широтой, долготой и высотой.

    Returns:
        Тензор формы (B, N_SITE).
    """
    return site_features(batch["lat"], batch["lon"], batch["elev"]).to(batch["x_hist"].dtype)


__all__ = ["DD_SCALE", "DP_SCALES", "FUTURE_CHANNELS", "HISTORY_CHANNELS", "N_FUTURE",
           "N_HISTORY", "N_SITE", "P_REF", "P_SCALE", "SITE_FEATURES", "T_SCALE",
           "batch_site_features", "dewpoint_deficit", "future_channels", "future_features",
           "history_channels", "history_features", "lag_valid", "pressure_tendency",
           "site_features"]
