"""Входные признаки, не зависящие от климат-поля."""
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
    """Маска часов, у которых валидны и сам час, и час на k раньше."""
    vs = F.pad(v, (k, 0))[..., :v.shape[-1]]
    return v * vs


def pressure_tendency(P, vp, k):
    """Изменение давления за k часов в фиксированной нормировке."""
    ps = F.pad(P, (k, 0))[..., :P.shape[-1]]
    d = ((P - ps) / DP_SCALES[k]).clamp(-DP_CLAMP, DP_CLAMP)
    return torch.where(lag_valid(vp, k) > 0, d, torch.zeros_like(d))


def dewpoint_deficit(x, mask):
    """Дефицит точки росы в градусах Цельсия, не меньше нуля."""
    T = torch.where(mask[..., 0] > 0, x[..., 0], torch.zeros_like(x[..., 0]))
    RH = torch.where(mask[..., 2] > 0, x[..., 2], torch.zeros_like(x[..., 2]))
    return (T - dewpoint_c(T, RH)).clamp(min=0.0)


def history_channels(x, mask, astro):
    """Каналы истории, не зависящие от климат-поля."""
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
    """Ковариаты часов горизонта: солнечное время, высота Солнца, фаза года."""
    sin_d, cos_d, _cz, czp, sin_y, cos_y = astro
    return dict(sin_d=sin_d, cos_d=cos_d, czp=czp, sin_y=sin_y, cos_y=cos_y)


def _astro(batch, doy_key, hour_key, n):
    lat, lon = batch["lat"][:, None], batch["lon"][:, None]
    return astro_features(batch[doy_key][:, -n:], batch[hour_key][:, -n:], lat, lon)


def history_features(batch):
    """Каналы истории батча одним тензором."""
    x, mask = batch["x_hist"], batch["mask_hist"]
    ch = history_channels(x, mask, _astro(batch, "doy_hist", "hour_hist", x.shape[1]))
    return torch.stack([ch[n] for n in HISTORY_CHANNELS], dim=-1).to(x.dtype)


def future_features(batch):
    """Ковариаты горизонта батча одним тензором."""
    n = batch["doy_fut"].shape[1]
    ch = future_channels(_astro(batch, "doy_fut", "hour_fut", n))
    return torch.stack([ch[k] for k in FUTURE_CHANNELS], dim=-1).to(batch["x_hist"].dtype)


def site_features(lat, lon, elev):
    """Признаки точки: положение на единичной сфере и высота."""
    phi = lat * (math.pi / 180.0)
    lam = lon * (math.pi / 180.0)
    return torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam),
                        torch.sin(phi), elev / ELEV_SCALE], dim=-1)


def batch_site_features(batch):
    """Признаки точки для батча, в типе наблюдений."""
    return site_features(batch["lat"], batch["lon"], batch["elev"]).to(batch["x_hist"].dtype)


__all__ = ["DD_SCALE", "DP_SCALES", "FUTURE_CHANNELS", "HISTORY_CHANNELS", "N_FUTURE",
           "N_HISTORY", "N_SITE", "P_REF", "P_SCALE", "SITE_FEATURES", "T_SCALE",
           "batch_site_features", "dewpoint_deficit", "future_channels", "future_features",
           "history_channels", "history_features", "lag_valid", "pressure_tendency",
           "site_features"]
