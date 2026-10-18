"""Статистические эталоны: климатология, затухающая персистентность, сезонно-наивный."""
import numpy as np

from mayak.constants import H
from mayak.data.splits import time_layout
from mayak.metrics import ZQ
from mayak.timeaxis import window_calendar

RECENT_HOURS = 24
RECENT_MIN_VALID = 6
DAMPED_VAR_FLOOR = 0.02

def recent_anomaly(xT, mT, clim, t, t0,
                   hours=RECENT_HOURS, min_valid=RECENT_MIN_VALID):
    """Средняя аномалия температуры относительно климатологии перед моментом t."""
    kh = np.arange(max(t - hours, 0), t)
    mh = (mT[kh] > 0).astype(np.float64)
    if mh.sum() < min_valid:
        return 0.0, False
    doy, hour = window_calendar(t0, kh)
    a = ((xT[kh] - clim.predict(doy, hour)) * mh).sum() / mh.sum()
    return float(a), True


def quantiles_from_normal(mu, sigma):
    mu = np.asarray(mu, np.float32)
    sigma = np.asarray(sigma, np.float32)
    return mu[..., None] + ZQ * sigma[..., None]


def _horizon_scale(scale, shape):
    """Масштаб интервала на часах горизонта в виде массива той же формы, что медиана."""
    sig = np.asarray(scale, np.float32)
    if sig.shape != tuple(shape):
        raise ValueError(f"масштаб интервала формы {sig.shape}, нужна {tuple(shape)}: эталоны "
                         f"берут масштаб остатка на каждом часе горизонта, а не скаляр станции")
    return sig


def climatology_forecast(mu_clim_fut, scale):
    """Прогноз климатологией станции на часах горизонта."""
    mu = np.asarray(mu_clim_fut, np.float32)
    return mu, quantiles_from_normal(mu, _horizon_scale(scale, mu.shape))


def damped_coefficients(Sxx, Sxy):
    """Коэффициенты затухания аномалии по накопленным суммам, свой для каждого лида."""
    Sxx = np.asarray(Sxx, np.float64)
    Sxy = np.asarray(Sxy, np.float64)
    r = np.where(Sxx > 1e-6, Sxy / np.maximum(Sxx, 1e-6), 0.0)
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def fit_damped_persistence(clims, n_windows=20000, seed=0):
    """Коэффициенты затухания по обучающему окну станций."""
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
    """Прогноз затухающей персистентностью аномалии."""
    a = np.asarray(a_recent, np.float32)[:, None]
    r = np.broadcast_to(np.asarray(r, np.float32)[None, :], (a.shape[0], len(r)))
    if valid is not None:
        r = np.where(np.asarray(valid, bool)[:, None], r, np.float32(0.0))
    mu = np.asarray(mu_clim_fut, np.float32) + r * a
    sig = _horizon_scale(scale, mu.shape) * np.sqrt(np.clip(1 - r ** 2, DAMPED_VAR_FLOOR, 1.0))
    return mu, quantiles_from_normal(mu, sig)


def seasonal_naive_forecast(x_hist, mask_hist, mu_clim_fut, scale, period=24):
    """Сезонно-наивный прогноз: последнее наблюдение в тот же час суток."""
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
