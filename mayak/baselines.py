"""Бейзлайны прогноза: климатология, damped persistence, seasonal-naive,
GRU seq2seq, DLinear"""
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn

from mayak.config import DLinearConfig, GRUConfig
from mayak.constants import H, QUANTILES
from mayak.data.splits import time_bounds
from mayak.data.store import get_store
from mayak.timeaxis import window_calendar

ZQ = norm.ppf(np.array(QUANTILES)).astype(np.float32)

RECENT_HOURS = 24
RECENT_MIN_VALID = 6


def recent_anomaly(xT, mT, clim, t, t0,
                   hours=RECENT_HOURS, min_valid=RECENT_MIN_VALID):
    kh = np.arange(max(t - hours, 0), t)
    mh = (mT[kh] > 0).astype(np.float64)
    if mh.sum() < min_valid:
        return 0.0, False
    doy, hour = window_calendar(t0, kh)
    a = ((xT[kh] - clim.predict(doy, hour)) * mh).sum() / mh.sum()
    return float(a), True


def fit_climatologies(manifest, force=False):
    """Климатологии всех станций манифеста из кэша, без повторного QC и подгонки.
    Возвращает {id: запись станции} с полями clim, x, mask, N, t0, lat, lon, elev, koppen.
    """
    return get_store(manifest, rebuild=force).clims()


def quantiles_from_normal(mu, sigma):
    mu = np.asarray(mu, np.float32)
    sigma = np.asarray(sigma, np.float32)
    return mu[..., None] + ZQ * sigma[..., None]


def climatology_forecast(mu_clim_fut, sigma_clim):
    mu = np.asarray(mu_clim_fut, np.float32)
    sig = np.broadcast_to(np.asarray(sigma_clim, np.float32).reshape(-1, 1), mu.shape)
    return mu, quantiles_from_normal(mu, sig)


def fit_damped_persistence(clims, n_windows=20000, seed=0):
    rng = np.random.default_rng(seed)
    sids = list(clims.keys())
    Sxx = np.zeros(H); Sxy = np.zeros(H)
    per = max(1, n_windows // len(sids))
    for sid in sids:
        s = clims[sid]; lo, hi = time_bounds(s["N"])["train"]
        if hi - lo < 24 + H + 1:
            continue
        clim = s["clim"]; t0 = s["t0"]
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
    r = np.where(Sxx > 1e-6, Sxy / np.maximum(Sxx, 1e-6), 0.0)
    r = np.clip(r, 0.0, 1.0).astype(np.float32)
    return r


def damped_persistence_forecast(a_recent, mu_clim_fut, sigma_clim, r):
    a = np.asarray(a_recent, np.float32)[:, None]
    mu = np.asarray(mu_clim_fut, np.float32) + r[None, :] * a
    sig = np.asarray(sigma_clim, np.float32)[:, None] * np.sqrt(np.clip(1 - r[None, :] ** 2, 0.02, 1.0))
    return mu, quantiles_from_normal(mu, sig)


def seasonal_naive_forecast(x_hist, mask_hist, mu_clim_fut, sigma_clim, period=24):
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
    sig = np.broadcast_to(np.asarray(sigma_clim, np.float32).reshape(-1, 1), mu.shape)
    return mu, quantiles_from_normal(mu, sig)


def _median_centered_offsets(gaps_param, nq, device):
    """Монотонные смещения квантилей (H, nq) с нулём на медиане."""
    gaps = torch.nn.functional.softplus(gaps_param)
    offs = torch.cat([torch.zeros(gaps.shape[0], 1, device=device), torch.cumsum(gaps, -1)], -1)
    return offs - offs[:, nq // 2:nq // 2 + 1]


class GRUSeq2Seq(nn.Module):
    """GRU-кодировщик с прямой головой на весь горизонт (конфиг - GRUConfig)."""
    N_INPUT = 3 + 3 + 3

    def __init__(self, cfg=None):
        super().__init__()
        cfg = GRUConfig() if cfg is None else cfg
        if not isinstance(cfg, GRUConfig):
            cfg = GRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.gru = nn.GRU(input_size=self.N_INPUT, hidden_size=cfg.hidden,
                          num_layers=cfg.layers, batch_first=True)
        self.head_mu = nn.Sequential(nn.Linear(cfg.hidden + 3, cfg.mu_hidden), nn.GELU(),
                                     nn.Linear(cfg.mu_hidden, cfg.horizon))
        self.head_sig = nn.Sequential(nn.Linear(cfg.hidden + 3, cfg.sigma_hidden), nn.GELU(),
                                      nn.Linear(cfg.sigma_hidden, cfg.horizon))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def forward(self, batch):
        x = batch["x_hist"]; m = batch["mask_hist"]

        T, P, RH = x[..., 0], x[..., 1], x[..., 2]
        xn = torch.stack([T / 30.0, (P - 1013.0) / 50.0, (RH - 50.0) / 50.0], dim=-1)

        coord = torch.stack([batch["lat"], batch["lon"], batch["elev"]], -1) / \
            torch.tensor([90.0, 180.0, 1000.0], device=x.device)
        coord_seq = coord[:, None, :].expand(-1, x.shape[1], -1)

        inp = torch.cat([xn * m, m, coord_seq], dim=-1)

        h, _ = self.gru(inp)
        last = h[:, -1]
        feat = torch.cat([last, coord], dim=-1)
        mu = self.head_mu(feat)
        log_sig = self.head_sig(feat).clamp(-2, 4)
        sig = torch.exp(log_sig)
        offs = _median_centered_offsets(self.gaps, self.nq, x.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}


class DLinear(nn.Module):
    """DLinear: разложение скользящим средним и два линейных отображения по времени
    (конфиг - DLinearConfig: длина входа и ядро скользящего среднего)."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = DLinearConfig() if cfg is None else cfg
        if not isinstance(cfg, DLinearConfig):
            cfg = DLinearConfig.from_dict(cfg)
        self.cfg = cfg
        self.k = cfg.kernel
        self.nq = cfg.n_quantiles
        self.input_len = cfg.input_len
        self.lin_trend = nn.Linear(cfg.input_len, cfg.horizon)
        self.lin_resid = nn.Linear(cfg.input_len, cfg.horizon)
        self.log_sig = nn.Parameter(torch.zeros(cfg.horizon))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def forward(self, batch):
        T = (batch["x_hist"][..., 0] * batch["mask_hist"][..., 0])[:, -self.input_len:]
        pad = self.k // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(T[:, None], (pad, pad), mode="replicate"),
            kernel_size=self.k, stride=1)[:, 0]
        resid = T - trend
        mu = self.lin_trend(trend) + self.lin_resid(resid)
        sig = torch.exp(self.log_sig).clamp(0.3, 12)[None].expand_as(mu)
        offs = _median_centered_offsets(self.gaps, self.nq, T.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}
