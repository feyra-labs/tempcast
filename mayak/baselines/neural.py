"""Нейробейзлайны GRU и DLinear и общие части нейробейзлайнов."""
import torch
import torch.nn as nn

from mayak.config import DLinearConfig, GRUConfig
from mayak.features import (N_FUTURE, N_HISTORY, N_SITE, T_SCALE, batch_site_features,
                            future_features, history_features)

N_RECURRENT_INPUT = N_HISTORY + N_SITE
LOG_SIGMA_CLAMP = (-2.0, 4.0)


def median_centered_offsets(gaps_param, nq, device):
    """Монотонные смещения квантилей с нулём на медиане."""
    gaps = torch.nn.functional.softplus(gaps_param)
    offs = torch.cat([torch.zeros(gaps.shape[0], 1, device=device), torch.cumsum(gaps, -1)], -1)
    return offs - offs[:, nq // 2:nq // 2 + 1]


def recurrent_inputs(batch):
    """Вход рекуррентных бейзлайнов на каждый час истории."""
    hist = history_features(batch)
    site = batch_site_features(batch)
    return torch.cat([hist, site[:, None, :].expand(-1, hist.shape[1], -1)], dim=-1)


class LeadHead(nn.Module):
    """Голова прогноза, общая для всех лидов."""

    def __init__(self, d_summary, hidden, horizon, n_quantiles):
        super().__init__()
        self.horizon, self.nq = horizon, n_quantiles
        self.mlp = nn.Sequential(nn.Linear(d_summary + N_FUTURE + N_SITE + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2))
        self.gaps = nn.Parameter(torch.zeros(horizon, n_quantiles - 1))

    def forward(self, summary, batch):
        """Квантили прогноза по сводке истории."""
        B, Hh = summary.shape[0], self.horizon
        dt = summary.dtype
        lead = (torch.arange(Hh, device=summary.device, dtype=dt) + 1) / Hh
        feat = torch.cat([summary[:, None, :].expand(-1, Hh, -1),
                          future_features(batch).to(dt),
                          batch_site_features(batch).to(dt)[:, None, :].expand(-1, Hh, -1),
                          lead[None, :, None].expand(B, -1, -1)], dim=-1)
        o = self.mlp(feat)
        mu = T_SCALE * o[..., 0]
        sig = torch.exp(o[..., 1].clamp(*LOG_SIGMA_CLAMP))
        offs = median_centered_offsets(self.gaps, self.nq, summary.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}


class GRUSeq2Seq(nn.Module):
    """Рекуррентный кодировщик истории и прямая голова на весь горизонт."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = GRUConfig() if cfg is None else cfg
        if not isinstance(cfg, GRUConfig):
            cfg = GRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.gru = nn.GRU(input_size=N_RECURRENT_INPUT, hidden_size=cfg.hidden,
                          num_layers=cfg.layers, batch_first=True)
        self.head = LeadHead(cfg.hidden, cfg.head_hidden, cfg.horizon, self.nq)

    def inputs(self, batch):
        """Вход кодировщика."""
        return recurrent_inputs(batch)

    def forward(self, batch):
        h, _ = self.gru(self.inputs(batch))
        return self.head(h[:, -1], batch)


class DLinear(nn.Module):
    """Разложение ряда температуры на тренд и остаток и два линейных отображения."""

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

    @staticmethod
    def fill_missing(T, valid):
        """Заполнение пропусков средним валидных часов окна."""
        ok = valid > 0
        n = ok.sum(-1, keepdim=True).clamp(min=1).to(T.dtype)
        mean = torch.where(ok, T, torch.zeros_like(T)).sum(-1, keepdim=True) / n
        return torch.where(ok, T, mean.expand_as(T))

    def decompose(self, T):
        """Тренд и остаток ряда."""
        pad = self.k // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(T[:, None], (pad, pad), mode="replicate"),
            kernel_size=self.k, stride=1)[:, 0]
        return trend, T - trend

    def point(self, T):
        """Точечный прогноз по ряду длины входа модели."""
        trend, resid = self.decompose(T)
        return self.lin_trend(trend) + self.lin_resid(resid)

    def forward(self, batch):
        n = self.input_len
        T = self.fill_missing(batch["x_hist"][:, -n:, 0], batch["mask_hist"][:, -n:, 0])
        mu = self.point(T)
        sig = torch.exp(self.log_sig).clamp(0.3, 12)[None].expand_as(mu)
        offs = median_centered_offsets(self.gaps, self.nq, T.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}
