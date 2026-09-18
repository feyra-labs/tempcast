"""Метрики прогноза с маской валидности цели."""
import numpy as np

from mayak.constants import QUANTILES

Q = np.array(QUANTILES, np.float32)
I_LO90, I_LO80, I_MED, I_HI80, I_HI90 = 0, 1, 3, 5, 6


def wmean(x, w, axis=None):
    x = np.asarray(x, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), x.shape)
    num = np.where(w > 0, x * w, 0.0).sum(axis=axis)
    den = w.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def pinball_crps(y, q):
    err = y[..., None] - q
    pin = np.maximum(Q * err, (Q - 1) * err)
    return 2.0 * pin.mean(axis=-1)


def inside(y, lo, hi):
    return ((y >= lo) & (y <= hi)).astype(np.float64)


def winkler(y, lo, hi, alpha):
    return ((hi - lo)
            + (2 / alpha) * (lo - y) * (y < lo)
            + (2 / alpha) * (y - hi) * (y > hi))


def skill_per_lead(y, mu, mu_clim, w):
    mse = wmean((mu - y) ** 2, w, axis=0)
    mse_c = wmean((mu_clim - y) ** 2, w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - mse / np.maximum(mse_c, 1e-9)


def skill(y, mu, mu_clim, w):
    mse = wmean((mu - y) ** 2, w)
    mse_c = wmean((mu_clim - y) ** 2, w)
    return float(1.0 - mse / max(float(mse_c), 1e-9))


def coverage(y, q, w, lo=I_LO90, hi=I_HI90):
    return float(wmean(inside(y, q[..., lo], q[..., hi]), w))


def metric_table(y, mu, q, mu_clim, w, leads=(1, 3, 6, 12, 24, 48, 72, 120, 168)):
    crps = pinball_crps(y, q)
    sk = skill_per_lead(y, mu, mu_clim, w)
    out = {}
    for h in leads:
        j = h - 1
        yj, wj = y[:, j], w[:, j]
        e = mu[:, j] - yj
        out[h] = dict(
            MAE=float(wmean(np.abs(e), wj)),
            RMSE=float(np.sqrt(wmean(e ** 2, wj))),
            Skill=float(sk[j]),
            CRPS=float(wmean(crps[:, j], wj)),
            PICP80=float(wmean(inside(yj, q[:, j, I_LO80], q[:, j, I_HI80]), wj)),
            PICP90=float(wmean(inside(yj, q[:, j, I_LO90], q[:, j, I_HI90]), wj)),
            Winkler90=float(wmean(winkler(yj, q[:, j, I_LO90], q[:, j, I_HI90], 0.10), wj)),
            n_valid=int((wj > 0).sum()),
        )
    return out


LEAD_BINS = ((1, 6), (7, 24), (25, 72), (73, 168))


def fit_conformal_shift(y, q, w, lead_bins=LEAD_BINS):
    shift = np.zeros((len(lead_bins), q.shape[-1]), np.float32)
    for bi, (a, b) in enumerate(lead_bins):
        sl = slice(a - 1, b)
        resid = (y[:, sl, None] - q[:, sl, :]).reshape(-1, q.shape[-1])
        resid = resid[np.asarray(w)[:, sl].reshape(-1) > 0]
        if len(resid) == 0:
            raise ValueError(f"в бине лидов {a}-{b} нет ни одного валидного часа")
        for qi, tau in enumerate(QUANTILES):
            shift[bi, qi] = np.quantile(resid[:, qi], tau)
    return shift
