import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EVIDENCE_FLOOR = 1.0


class LaplaceReadout(nn.Module):
    """Считывание затухающих мод λ = −1/τ + iω по истории признаков энкодера.

    groups      - группы мод (ModeGroup): начальные τ₀ и периоды, ч; ω₀ = 2π / период;
    tau_bounds  - τ = τ_min + (τ_max − τ_min)·sigmoid(raw_tau);
    compression - доказательное сжатие: состояние делится на e + κ, т.е. при малой
                  массе свидетельств e аномалия стягивается к нулю. False - абляция
                  no_compression: деление на max(e, EVIDENCE_FLOOR).
    """

    def __init__(self, width, groups, tau_bounds=(3.0, 240.0), compression=True):
        super().__init__()
        self.compression = compression
        self.tau_lo, self.tau_hi = float(tau_bounds[0]), float(tau_bounds[1])
        tau0 = torch.tensor([t for g in groups for t in g.tau0])
        period = torch.tensor([p for g in groups for p in g.period])
        M = len(tau0)
        self.n_modes = M
        span = self.tau_hi - self.tau_lo
        self.raw_tau = nn.Parameter(
            torch.logit(((tau0 - self.tau_lo) / span).clamp(1e-3, 1 - 1e-3)))

        w = 2 * math.pi
        osc = period > 0
        omega0 = torch.where(osc, w / torch.where(osc, period, torch.ones_like(period)),
                             torch.zeros_like(period))
        self.register_buffer("omega0", omega0)
        self.p_w = nn.Parameter(torch.zeros(M))
        self.p_k = nn.Parameter(torch.full((M,), 2.3))
        self.proj = nn.Linear(width, 2 * M)

    def constants(self):
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * torch.sigmoid(self.raw_tau)
        omega = self.omega0 * torch.exp(0.15 * torch.tanh(self.p_w))
        kappa = 1.0 + F.softplus(self.p_k)
        return tau, omega, kappa

    def normalize(self, n_re, n_im, e, kappa=None):
        """Накопленное состояние (n, e) → амплитуды мод."""
        if self.compression:
            if kappa is None:
                kappa = self.constants()[2]
            den = e + kappa[None, :]
        else:
            den = e.clamp_min(EVIDENCE_FLOOR)
        return n_re / den, n_im / den

    def forward(self, feats, v):
        tau, omega, kappa = self.constants()
        M = self.n_modes
        Lh = feats.shape[1]
        lag = torch.arange(Lh - 1, -1, -1, dtype=feats.dtype, device=feats.device)
        dec = torch.exp(-lag[:, None] / tau[None, :])
        ph = omega[None, :] * lag[:, None]
        kc, ks = dec * torch.cos(ph), dec * torch.sin(ph)
        u = self.proj(feats)
        uc = u[..., :M] * v[..., None]
        us = u[..., M:] * v[..., None]

        n_re = torch.einsum("blm,lm->bm", uc, kc) - torch.einsum("blm,lm->bm", us, ks)
        n_im = torch.einsum("blm,lm->bm", uc, ks) + torch.einsum("blm,lm->bm", us, kc)

        e = torch.einsum("bl,lm->bm", v, dec)
        a_re, a_im = self.normalize(n_re, n_im, e, kappa)
        return a_re, a_im, e

    @torch.no_grad()
    def step(self, state, feat_t, v_t):
        tau, omega, _ = self.constants()
        M = self.n_modes
        rho = torch.exp(-1.0 / tau)
        co, si = torch.cos(omega), torch.sin(omega)
        n_re, n_im, e = state
        u = self.proj(feat_t)
        uc, us = u[..., :M], u[..., M:]
        n_re2 = rho * (co * n_re - si * n_im) + v_t[..., None] * uc
        n_im2 = rho * (si * n_re + co * n_im) + v_t[..., None] * us
        return n_re2, n_im2, rho * e + v_t[..., None]
