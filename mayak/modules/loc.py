import math
import torch
import torch.nn as nn


class LocEncoder(nn.Module):
    """Случайные Фурье-признаки точки на сфере"""
    OUT = 96 + 5 # используется другими модулями как размер входа

    def __init__(self, n_freq: int = 48, f_scale: float = 25.0, f_max: float = 133.0):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        W = torch.randn(3, n_freq, generator=g) * f_scale
        nrm = W.norm(dim=0, keepdim=True)
        W = W * torch.clamp(f_max / nrm, max=1.0)
        self.register_buffer("W", W)

    def forward(self, lat_deg, lon_deg, elev_m):
        phi = lat_deg * math.pi / 180.0
        lam = lon_deg * math.pi / 180.0

        xyz = torch.stack([torch.cos(phi) * torch.cos(lam),
                           torch.cos(phi) * torch.sin(lam),
                           torch.sin(phi)], dim=-1)
        proj = xyz @ self.W
        e = (elev_m / 3000.0).clamp(-0.5, 2.0).unsqueeze(-1)
        return torch.cat([torch.sin(proj), torch.cos(proj), xyz,
                          (lat_deg.abs() / 90.0).unsqueeze(-1), e], dim=-1)