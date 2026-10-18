"""Бейзлайн с линейной рекуррентной памятью без разложения на якорь и аномалию."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.baselines.neural import N_RECURRENT_INPUT, LeadHead, recurrent_inputs
from mayak.config import LRUConfig

OUTPUT_MATRICES = ("C_re", "C_im")
LRU_CHUNK = 32


def lambda_polar(nu_log, theta_log):
    """Скорость затухания и угловая частота собственных чисел из сырых параметров."""
    return torch.exp(nu_log), torch.exp(theta_log)


def lambda_power(nu, theta, p):
    """Собственные числа в степени p, вещественная и мнимая части."""
    p = p[..., None].to(nu.dtype)
    mag = torch.exp(-p * nu)
    return mag * torch.cos(p * theta), mag * torch.sin(p * theta)


def _shift(x, k):
    """Сдвиг по оси времени на k часов в прошлое с нулями в начале."""
    return F.pad(x[:, :-k], (0, 0, k, 0))


def lru_scan_associative(a_re, a_im, u_re, u_im):
    """Линейная рекуррентность с постоянным коэффициентом, параллельный скан."""
    h_re, h_im = u_re, u_im
    L = h_re.shape[1]
    k = 1
    while k < L:
        s_re, s_im = _shift(h_re, k), _shift(h_im, k)
        h_re, h_im = h_re + a_re * s_re - a_im * s_im, h_im + a_re * s_im + a_im * s_re
        a_re, a_im = a_re * a_re - a_im * a_im, 2 * a_re * a_im
        k *= 2
    return h_re, h_im


def lru_scan_chunked(nu, theta, u_re, u_im, chunk=LRU_CHUNK):
    """Та же рекуррентность, развёрнутая блоками фиксированной длины."""
    B, L, N = u_re.shape
    C = min(int(chunk), L)
    pad = (-L) % C
    if pad:
        u_re, u_im = F.pad(u_re, (0, 0, pad, 0)), F.pad(u_im, (0, 0, pad, 0))
    nC = (L + pad) // C
    u_re, u_im = u_re.reshape(B, nC, C, N), u_im.reshape(B, nC, C, N)

    idx = torch.arange(C, device=u_re.device)
    diff = idx[:, None] - idx[None, :]
    k_re, k_im = lambda_power(nu, theta, diff.clamp_min(0))
    low = (diff >= 0).to(k_re.dtype)[..., None]
    k_re, k_im = k_re * low, k_im * low
    y_re = torch.einsum("ijn,bcjn->bcin", k_re, u_re) - torch.einsum("ijn,bcjn->bcin", k_im, u_im)
    y_im = torch.einsum("ijn,bcjn->bcin", k_re, u_im) + torch.einsum("ijn,bcjn->bcin", k_im, u_re)

    if nC > 1:
        aC_re, aC_im = lambda_power(nu, theta, torch.tensor(float(C), device=u_re.device))
        s_re, s_im = lru_scan_associative(aC_re, aC_im, y_re[:, :, -1], y_im[:, :, -1])
        s_re, s_im = _shift(s_re, 1), _shift(s_im, 1)
        p_re, p_im = lambda_power(nu, theta, (idx + 1).to(nu.dtype))
        y_re = y_re + p_re * s_re[:, :, None] - p_im * s_im[:, :, None]
        y_im = y_im + p_re * s_im[:, :, None] + p_im * s_re[:, :, None]
    y_re, y_im = y_re.reshape(B, nC * C, N), y_im.reshape(B, nC * C, N)
    return y_re[:, pad:], y_im[:, pad:]


def lru_recurrent(a_re, a_im, u_re, u_im):
    """Рекуррентность простым циклом по часам - эталон для проверки параллельных развёрток."""
    B, L, N = u_re.shape
    h_re = u_re.new_zeros(B, N)
    h_im = u_re.new_zeros(B, N)
    out_re, out_im = [], []
    for t in range(L):
        h_re, h_im = a_re * h_re - a_im * h_im + u_re[:, t], a_re * h_im + a_im * h_re + u_im[:, t]
        out_re.append(h_re)
        out_im.append(h_im)
    return torch.stack(out_re, 1), torch.stack(out_im, 1)


class LRULayer(nn.Module):
    """Рекуррентное ядро: диагональная комплексная линейная рекуррентность."""

    def __init__(self, d_model, d_state, r_min, r_max, max_phase):
        super().__init__()
        u1, u2 = torch.rand(d_state), torch.rand(d_state)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(max_phase * u2.clamp_min(1e-4))
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        mod = torch.exp(-torch.exp(nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - mod ** 2)))
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(2 * d_model))
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(2 * d_model))
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.randn(d_model))

    def drive(self, x):
        """Собственные числа и комплексный вход рекуррентности."""
        g = torch.exp(self.gamma_log)[:, None]
        nu, theta = lambda_polar(self.nu_log, self.theta_log)
        return nu, theta, x @ (self.B_re * g).T, x @ (self.B_im * g).T

    def states(self, x):
        """Комплексные состояния рекуррентности, блочная развёртка."""
        return lru_scan_chunked(*self.drive(x))

    def forward(self, x):
        h_re, h_im = self.states(x)
        return h_re @ self.C_re.T - h_im @ self.C_im.T + self.D * x


class LRUBlock(nn.Module):
    """Блок стека: нормализация, рекуррентное ядро, нелинейность с затвором, остаток."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lru = LRULayer(cfg.d_model, cfg.d_state, cfg.r_min, cfg.r_max, cfg.max_phase)
        self.out1 = nn.Linear(cfg.d_model, cfg.d_model)
        self.out2 = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        z = self.lru(self.norm(x))
        z = self.drop(F.gelu(z))
        z = self.out1(z) * torch.sigmoid(self.out2(z))
        return x + self.drop(z)


class LRUForecaster(nn.Module):
    """Стек линейных рекуррентных блоков над историей и общая по лидам голова."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = LRUConfig() if cfg is None else cfg
        if not isinstance(cfg, LRUConfig):
            cfg = LRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.embed = nn.Linear(N_RECURRENT_INPUT, cfg.d_model)
        self.blocks = nn.ModuleList([LRUBlock(cfg) for _ in range(cfg.layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = LeadHead(cfg.d_model, cfg.head_hidden, cfg.horizon, self.nq)

    def inputs(self, batch):
        """Вход стека."""
        return recurrent_inputs(batch)

    def encode(self, batch):
        """Выход стека на каждом часе истории."""
        z = self.embed(self.inputs(batch))
        for blk in self.blocks:
            z = blk(z)
        return self.out_norm(z)

    def forward(self, batch):
        return self.head(self.encode(batch)[:, -1], batch)

    def decay_exceptions(self, weight_decay):
        """Отличия от общего правила весового затухания."""
        return [dict(name="lru_output", weight_decay=weight_decay,
                     match=lambda name, by_rule: name.rsplit(".", 1)[-1] in OUTPUT_MATRICES)]
