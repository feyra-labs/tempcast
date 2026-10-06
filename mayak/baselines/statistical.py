"""Статистические эталоны: климатология, затухающая персистентность, сезонно-наивный.

У всех трёх интервал нормальный с гармоническим масштабом остатка климатологии станции
на часах горизонта: масштаб меняется по году и суткам. Это тот же масштаб, что нормирует
функцию потерь, поэтому второго эталона разброса не возникает.
"""
import numpy as np

from mayak.constants import H
from mayak.data.splits import time_layout
from mayak.data.store import get_store
from mayak.metrics import ZQ
from mayak.timeaxis import window_calendar

RECENT_HOURS = 24
RECENT_MIN_VALID = 6
DAMPED_VAR_FLOOR = 0.02

def recent_anomaly(xT, mT, clim, t, t0,
                   hours=RECENT_HOURS, min_valid=RECENT_MIN_VALID):
    """Средняя аномалия температуры относительно климатологии перед моментом t.

    Аномалия одного часа несёт ошибку суточного хода климатологии и шум прибора, а при
    пропуске последнего часа не определена. Поэтому берётся среднее по валидным часам
    за последние сутки.

    Args:
        xT: ряд температуры станции.
        mT: маска температуры того же ряда.
        clim: климатология станции.
        t: индекс момента выпуска в ряду; сам час t не входит.
        t0: абсолютный час начала ряда.
        hours: длина усредняемого отрезка в часах.
        min_valid: наименьшее число валидных часов.

    Returns:
        Пара: средняя аномалия и признак того, что валидных часов хватило. Если не
        хватило, аномалия равна нулю.
    """
    kh = np.arange(max(t - hours, 0), t)
    mh = (mT[kh] > 0).astype(np.float64)
    if mh.sum() < min_valid:
        return 0.0, False
    doy, hour = window_calendar(t0, kh)
    a = ((xT[kh] - clim.predict(doy, hour)) * mh).sum() / mh.sum()
    return float(a), True


def fit_climatologies(manifest, force=False):
    """Климатологии всех станций манифеста из кэша, без повторного QC и подгонки.

    Args:
        manifest: путь к манифесту станций.
        force: пересобрать кэш, даже если он актуален.

    Returns:
        Словарь из идентификатора станции в её запись: климатология, ряд, маски, длина,
        начало ряда, координаты, высота и зона.
    """
    return get_store(manifest, rebuild=force).clims()


def quantiles_from_normal(mu, sigma):
    mu = np.asarray(mu, np.float32)
    sigma = np.asarray(sigma, np.float32)
    return mu[..., None] + ZQ * sigma[..., None]


def _horizon_scale(scale, shape):
    """Масштаб интервала на часах горизонта в виде массива той же формы, что медиана.

    Масштаб задаётся на каждый час горизонта каждого окна. Скаляр станции или масштаб
    другой формы не растягивается молча: при числе окон, равном длине горизонта, такой
    масштаб лёг бы вдоль лидов, а не вдоль окон.

    Args:
        scale: масштаб остатка на часах горизонта, °C.
        shape: форма медианы, (B, H).

    Returns:
        Масштаб формы shape, float32.

    Raises:
        ValueError: форма масштаба не совпадает с формой медианы.
    """
    sig = np.asarray(scale, np.float32)
    if sig.shape != tuple(shape):
        raise ValueError(f"масштаб интервала формы {sig.shape}, нужна {tuple(shape)}: эталоны "
                         f"берут масштаб остатка на каждом часе горизонта, а не скаляр станции")
    return sig


def climatology_forecast(mu_clim_fut, scale):
    """Прогноз климатологией станции на часах горизонта.

    Климатология гармоническая: годовые и суточные гармоники среднего и масштаба,
    подогнанные по обучающему окну станции.

    Args:
        mu_clim_fut: климатологическое среднее на часах горизонта, форма (B, H).
        scale: масштаб остатка климатологии на часах горизонта, °C, форма (B, H).

    Returns:
        Пара: медиана формы (B, H) и квантили формы (B, H, nq) нормального
        распределения с масштабом своего часа горизонта.

    Raises:
        ValueError: форма масштаба не совпадает с формой среднего.
    """
    mu = np.asarray(mu_clim_fut, np.float32)
    return mu, quantiles_from_normal(mu, _horizon_scale(scale, mu.shape))


def damped_coefficients(Sxx, Sxy):
    """Коэффициенты затухания аномалии по накопленным суммам, свой для каждого лида.

    Коэффициент - наклон регрессии будущей аномалии на недавнюю без свободного члена,
    ограниченный отрезком от нуля до единицы. Если недавняя аномалия почти всегда
    нулевая, коэффициент равен нулю.

    Args:
        Sxx: сумма квадратов недавней аномалии по валидным парам, форма (H,).
        Sxy: сумма произведений недавней и будущей аномалий, форма (H,).

    Returns:
        Коэффициенты формы (H,), float32.
    """
    Sxx = np.asarray(Sxx, np.float64)
    Sxy = np.asarray(Sxy, np.float64)
    r = np.where(Sxx > 1e-6, Sxy / np.maximum(Sxx, 1e-6), 0.0)
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def fit_damped_persistence(clims, n_windows=20000, seed=0):
    """Коэффициенты затухания по обучающему окну станций.

    Оценка прямая, отдельно для каждого лида, без предположения об экспоненциальном
    затухании. На каждой станции берётся одинаковое число случайных моментов её
    обучающего окна. Пара из недавней и будущей аномалии входит в сумму лида, только
    если будущий час валиден.

    Args:
        clims: записи станций; передаются обучающие станции.
        n_windows: общее число моментов по всем станциям.
        seed: сид выбора моментов.

    Returns:
        Коэффициенты формы (H,).
    """
    rng = np.random.default_rng(seed)
    sids = list(clims.keys())
    Sxx = np.zeros(H)
    Sxy = np.zeros(H)
    per = max(1, n_windows // len(sids))
    for sid in sids:
        s = clims[sid]
        lo, hi = time_layout(s["N"]).span("train")
        if hi - lo < 24 + H + 1:
            continue
        clim = s["clim"]
        t0 = s["t0"]
        xT, mT = s["x"][:, 0], s["mask"][:, 0]
        for _ in range(per):
            t = int(rng.integers(lo + 24, hi - H))
            anom_recent, ok = recent_anomaly(xT, mT, clim, t, t0)
            if not ok:
                continue
            kf = np.arange(t, t + H)
            mf = (mT[kf] > 0).astype(np.float64)
            doy_f, hour_f = window_calendar(t0, kf)
            anom_fut = (xT[kf] - clim.predict(doy_f, hour_f)) * mf
            Sxx += anom_recent ** 2 * mf
            Sxy += anom_recent * anom_fut
    return damped_coefficients(Sxx, Sxy)


def damped_persistence_forecast(a_recent, mu_clim_fut, scale, r, valid=None):
    """Прогноз затухающей персистентностью аномалии.

    Недавняя аномалия прибавляется к климатологии с коэффициентом своего лида. Разброс
    интервала - масштаб остатка климатологии на часе горизонта, уменьшенный так, как
    уменьшается условный разброс аномалии, которая затухает с тем же коэффициентом. Снизу
    доля разброса ограничена, иначе при коэффициенте около единицы интервал схлопывается в
    точку.

    Если в истории окна не хватило валидных часов, аномалия неизвестна. Тогда прогноз
    окна - климатология целиком, вместе с её разбросом: сузить интервал нечем.

    Args:
        a_recent: недавняя аномалия каждого окна, форма (B,).
        mu_clim_fut: климатологическое среднее на часах горизонта, форма (B, H).
        scale: масштаб остатка климатологии на часах горизонта, °C, форма (B, H).
        r: коэффициенты затухания по лидам, форма (H,).
        valid: признак того, что аномалия окна определена, форма (B,); None значит, что
            определена у всех окон.

    Returns:
        Пара: медиана формы (B, H) и квантили формы (B, H, nq).

    Raises:
        ValueError: форма масштаба не совпадает с формой среднего.
    """
    a = np.asarray(a_recent, np.float32)[:, None]
    r = np.broadcast_to(np.asarray(r, np.float32)[None, :], (a.shape[0], len(r)))
    if valid is not None:
        r = np.where(np.asarray(valid, bool)[:, None], r, np.float32(0.0))
    mu = np.asarray(mu_clim_fut, np.float32) + r * a
    sig = _horizon_scale(scale, mu.shape) * np.sqrt(np.clip(1 - r ** 2, DAMPED_VAR_FLOOR, 1.0))
    return mu, quantiles_from_normal(mu, sig)


def seasonal_naive_forecast(x_hist, mask_hist, mu_clim_fut, scale, period=24):
    """Сезонно-наивный прогноз: последнее наблюдение в тот же час суток.

    История выровнена по правому краю окна. Для каждого часа суток берутся последние
    сутки истории, где этот час валиден; если таких суток нет, берётся климатология.

    Args:
        x_hist: наблюдения истории, форма (B, L, 3).
        mask_hist: маски наличия, форма (B, L, 3).
        mu_clim_fut: климатологическое среднее на часах горизонта, форма (B, H).
        scale: масштаб остатка климатологии на часах горизонта, °C, форма (B, H).
        period: период повтора в часах.

    Returns:
        Пара: медиана формы (B, H) и квантили формы (B, H, nq).

    Raises:
        ValueError: форма масштаба не совпадает с формой среднего.
    """
    B, Lh = x_hist.shape[:2]
    D = Lh // period
    T = x_hist[:, Lh - D * period:, 0].reshape(B, D, period)[:, ::-1]
    v = (mask_hist[:, Lh - D * period:, 0] > 0.5).reshape(B, D, period)[:, ::-1]
    has = v.any(axis=1)
    k = v.argmax(axis=1)
    last = np.take_along_axis(T, k[:, None, :], axis=1)[:, 0]
    slot = np.arange(H) % period
    mu = np.where(has[:, slot], last[:, slot], np.asarray(mu_clim_fut, np.float32))
    mu = mu.astype(np.float32)
    return mu, quantiles_from_normal(mu, _horizon_scale(scale, mu.shape))
