import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import M, C_ENC


class LaplaceReadout(nn.Module):

    def __init__(self):
        super().__init__()
        tau0 = torch.tensor([3., 6., 12., 24., 48., 96., 168., 240.,     # R (8)
                             12., 24., 48., 96., 168., 240.,             # D (6)
                             12., 24., 72., 168.,                        # S (4)
                             24., 48., 72., 120., 168., 240.])           # W (6)
        self.raw_tau = nn.Parameter(
            torch.logit(((tau0 - 3.0) / 237.0).clamp(1e-3, 1 - 1e-3)))

        w = 2 * math.pi
        omega0 = torch.cat([
            torch.zeros(8),
            torch.full((6,), w / 24),
            torch.full((4,), w / 12),
            w / torch.tensor([60., 84., 108., 132., 156., 192.]),
        ])
        self.register_buffer("omega0", omega0)
        self.p_w = nn.Parameter(torch.zeros(M))
        self.p_k = nn.Parameter(torch.full((M,), 2.3))
        self.proj = nn.Linear(C_ENC, 2 * M)

    def constants(self):
        tau = 3.0 + 237.0 * torch.sigmoid(self.raw_tau)
        omega = self.omega0 * torch.exp(0.15 * torch.tanh(self.p_w))
        kappa = 1.0 + F.softplus(self.p_k)
        return tau, omega, kappa
    
    def forward(self, feats, v):
        tau, omega, kappa = self.constants()
        Lh = feats.shape[1]
        lag = torch.arange(Lh - 1, -1, -1, dtype=feats.dtype, device=feats.device)
        dec = torch.exp(-lag[:, None] / tau[None, :])
        ph = omega[None, :] * lag[:, None]
        kc, ks = dec * torch.cos(ph), dec * torch.sin(ph)
        u = self.proj(feats)
        uc = u[..., :M] * v[..., None]
        us = u[..., M:] * v[..., None]

        a_re = torch.einsum("blm,lm->bm", uc, kc) - torch.einsum("blm,lm->bm", us, ks)
        a_im = torch.einsum("blm,lm->bm", uc, ks) + torch.einsum("blm,lm->bm", us, kc)

        e = torch.einsum("bl,lm->bm", v, dec)
        den = e + kappa[None, :]
        return a_re / den, a_im / den, e
    
    @torch.no_grad()
    def step(self, state, feat_t, v_t):
        tau, omega, _ = self.constants()
        rho = torch.exp(-1.0 / tau)
        co, si = torch.cos(omega), torch.sin(omega)
        n_re, n_im, e = state
        u = self.proj(feat_t)
        uc, us = u[..., :M], u[..., M:]
        n_re2 = rho * (co * n_re - si * n_im) + v_t[..., None] * uc
        n_im2 = rho * (si * n_re + co * n_im) + v_t[..., None] * us
        return n_re2, n_im2, rho * e + v_t[..., None]