"""Бейзлайны прогноза: климатология, damped persistence, seasonal-naive,
GRU seq2seq, DLinear"""
import csv, os
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn

from mayak.constants import H, QUANTILES
from mayak.data.qc import run_qc
from mayak.data.climatology import Climatology
from mayak.data.splits import time_bounds

ZQ = norm.ppf(np.array(QUANTILES)).astype(np.float32)


def fit_climatologies(manifest):
    root = os.path.dirname(manifest)
    out = {}
    with open(manifest) as f:
        for r in csv.DictReader(f):
            d = np.load(os.path.join(root, "stations", f"{r['id']}.npz"))
            x, mask = run_qc(d["T"], d["P"], d["RH"], d["valid"])
            N = x.shape[0]; lo, hi = time_bounds(N)["train"]
            t0d, t0h = float(d["t0_doy"]), float(d["t0_hour"])
            k = np.arange(lo, hi)
            doy = (t0d + (t0h + k) / 24.0) % 365.24
            hour = (t0h + k) % 24.0
            clim = Climatology().fit(doy, hour, x[lo:hi, 0], mask[lo:hi, 0])
            out[r["id"]] = dict(clim=clim, lat=float(r["lat"]), lon=float(r["lon"]),
                                elev=float(r["elev"]), koppen=r["koppen"],
                                x=x, mask=mask, N=N, t0d=t0d, t0h=t0h)
    return out


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
    res_sq = np.zeros(H); cnt = 0
    per = max(1, n_windows // len(sids))
    for sid in sids:
        s = clims[sid]; lo, hi = time_bounds(s["N"])["train"]
        if hi - lo < 24 + H + 1:
            continue
        clim = s["clim"]; t0d, t0h = s["t0d"], s["t0h"]
        for _ in range(per):
            t = int(rng.integers(lo + 24, hi - H))
            kh = np.arange(t - 24, t)
            mh = s["mask"][kh, 0]
            if mh.sum() < 6:
                continue
            doy_h = (t0d + (t0h + kh) / 24.0) % 365.24
            hour_h = (t0h + kh) % 24.0
            anom_recent = ((s["x"][kh, 0] - clim.predict(doy_h, hour_h)) * mh).sum() / mh.sum()
            kf = np.arange(t, t + H)
            doy_f = (t0d + (t0h + kf) / 24.0) % 365.24
            hour_f = (t0h + kf) % 24.0
            anom_fut = s["x"][kf, 0] - clim.predict(doy_f, hour_f)
            Sxx += anom_recent ** 2
            Sxy += anom_recent * anom_fut
            cnt += 1
    r = np.where(Sxx > 1e-6, Sxy / np.maximum(Sxx, 1e-6), 0.0)
    r = np.clip(r, 0.0, 1.0).astype(np.float32)
    return r


def damped_persistence_forecast(a_recent, mu_clim_fut, sigma_clim, r):
    a = np.asarray(a_recent, np.float32)[:, None]
    mu = np.asarray(mu_clim_fut, np.float32) + r[None, :] * a
    sig = np.asarray(sigma_clim, np.float32)[:, None] * np.sqrt(np.clip(1 - r[None, :] ** 2, 0.02, 1.0))
    return mu, quantiles_from_normal(mu, sig)


def seasonal_naive_forecast(x_hist, mask_hist, sigma_clim, period=24):
    B = x_hist.shape[0]
    T = x_hist[..., 0]; v = mask_hist[..., 0]
    mu = np.zeros((B, H), np.float32)
    for h in range(1, H + 1):
        idx = 672 - period + ((h - 1) % period)
        col = T[:, idx].copy()
        bad = v[:, idx] < 0.5
        if bad.any():
            last_valid = np.where(v[:, -1] > 0.5, T[:, -1], 0.0)
            col[bad] = last_valid[bad]
        mu[:, h - 1] = col
    sig = np.broadcast_to(np.asarray(sigma_clim, np.float32).reshape(-1, 1), mu.shape)
    return mu, quantiles_from_normal(mu, sig)


class GRUSeq2Seq(nn.Module):
    def __init__(self, hidden=96, nq=len(QUANTILES)):
        super().__init__()
        self.nq = nq
        self.gru = nn.GRU(input_size=3 + 3 + 3, hidden_size=hidden,
                          num_layers=2, batch_first=True)
        self.head_mu = nn.Sequential(nn.Linear(hidden + 3, 256), nn.GELU(),
                                     nn.Linear(256, H))
        self.head_sig = nn.Sequential(nn.Linear(hidden + 3, 128), nn.GELU(),
                                      nn.Linear(128, H))
        self.gaps = nn.Parameter(torch.zeros(H, nq - 1))

    def forward(self, batch):
        x = batch["x_hist"]; m = batch["mask_hist"]
        B = x.shape[0]
        coord = torch.stack([batch["lat"], batch["lon"], batch["elev"]], -1) / \
            torch.tensor([90.0, 180.0, 1000.0], device=x.device)
        coord_seq = coord[:, None, :].expand(-1, x.shape[1], -1)
        inp = torch.cat([x * m, m, coord_seq], dim=-1)
        h, _ = self.gru(inp)
        last = h[:, -1]
        feat = torch.cat([last, coord], dim=-1)
        mu = self.head_mu(feat)
        log_sig = self.head_sig(feat).clamp(-2, 4)
        sig = torch.exp(log_sig)
        gaps = torch.nn.functional.softplus(self.gaps)
        cum = torch.cumsum(gaps, dim=-1)
        offs = torch.cat([torch.zeros(H, 1, device=x.device), cum], dim=-1)
        offs = offs - offs[:, self.nq // 2:self.nq // 2 + 1]
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma_c": sig}


class DLinear(nn.Module):
    def __init__(self, L_in=672, kernel=25, nq=len(QUANTILES)):
        super().__init__()
        self.k = kernel
        self.lin_trend = nn.Linear(L_in, H)
        self.lin_resid = nn.Linear(L_in, H)
        self.log_sig = nn.Parameter(torch.zeros(H))
        self.gaps = nn.Parameter(torch.zeros(H, nq - 1))
        self.nq = nq

    def forward(self, batch):
        T = (batch["x_hist"][..., 0] * batch["mask_hist"][..., 0])
        pad = self.k // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(T[:, None], (pad, pad), mode="replicate"),
            kernel_size=self.k, stride=1)[:, 0]
        resid = T - trend
        mu = self.lin_trend(trend) + self.lin_resid(resid)
        sig = torch.exp(self.log_sig).clamp(0.3, 12)[None].expand_as(mu)
        gaps = torch.nn.functional.softplus(self.gaps)
        offs = torch.cat([torch.zeros(H, 1, device=T.device), torch.cumsum(gaps, -1)], dim=-1)
        offs = offs - offs[:, self.nq // 2:self.nq // 2 + 1]
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma_c": sig}