from statistics import NormalDist

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import inv_softplus


def normal_gaps(quantiles, decimals=3):
    """Начальные зазоры между соседними квантилями N(0, 1), от медианы наружу.

    Возвращает (нижние, верхние). z округляются до 1e-3.
    """
    q = list(quantiles)
    m = q.index(0.5)
    z = [round(NormalDist().inv_cdf(x), decimals) for x in q]
    lower = [z[i + 1] - z[i] for i in range(m - 1, -1, -1)]
    upper = [z[i] - z[i - 1] for i in range(m + 1, len(q))]
    return lower, upper


class Heads(nn.Module):
    """Головы: ограниченная поправка r, масштаб интервала и монотонные квантильные смещения."""

    def __init__(self, dz, n_groups, n_sun, quantiles, hidden=48, z_proj=4):
        super().__init__()
        lower, upper = normal_gaps(quantiles)
        self.n_lo, self.n_hi = len(lower), len(upper)
        self.z_proj = z_proj
        self.in_dim = 1 + n_groups + 1 + n_sun + 1 + z_proj + 1 + 1
        self.zproj = nn.Linear(dz, z_proj)
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 + self.n_lo + self.n_hi)
        gaps = torch.tensor(lower + upper)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            self.fc2.bias[1] = 0.9
            self.fc2.bias[2:] = inv_softplus(gaps)

    def forward(self, o, Eg, sun_fut, log_sigma, z, e):
        Bsz, Hn = o.shape
        zp = self.zproj(z)[:, None, :].expand(Bsz, Hn, self.z_proj)
        hn = (torch.arange(1, Hn + 1, dtype=o.dtype, device=o.device) / Hn
              )[None, :, None].expand(Bsz, Hn, 1)
        le = torch.log1p(e.mean(-1))[:, None, None].expand(Bsz, Hn, 1)

        x = torch.cat([o[..., None], Eg, Eg.sum(-1, keepdim=True), sun_fut,
                       log_sigma[..., None], zp, hn, le], dim=-1)
        out = self.fc2(F.gelu(self.fc1(x)))

        r = 0.6 * torch.tanh(out[..., 0])
        ratio = 0.08 + torch.sigmoid(out[..., 1] + 1.5)
        gaps = F.softplus(out[..., 2:])

        lo = torch.flip(torch.cumsum(gaps[..., :self.n_lo], dim=-1), dims=(-1,))
        hi = torch.cumsum(gaps[..., self.n_lo:], dim=-1)
        off = torch.cat([-lo, torch.zeros_like(r)[..., None], hi], dim=-1)
        return r, ratio, off
