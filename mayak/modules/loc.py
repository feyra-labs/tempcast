import math
import torch
import torch.nn as nn

# Высота в признаках точки: метры, делённые на этот масштаб, с обрезкой в ELEV_CLAMP.
ELEV_SCALE_M = 3000.0
ELEV_CLAMP = (-0.5, 2.0)


class LocEncoder(nn.Module):
    """Признаки точки на сфере: случайные Фурье-признаки и гладкие признаки.

    Порядок признаков: синусы и косинусы проекций точки на случайные частоты, три
    декартовы координаты точки на единичной сфере, модуль широты, делённый на 90, и
    высота в единицах ``ELEV_SCALE_M``. Высота - последний признак: по нему климат-поле
    берёт явный член градиента температуры по высоте.

    Гладкие признаки (координаты на сфере, модуль широты, высота) задают крупномасштабный
    климат и не дают запомнить отдельные точки; Фурье-признаки добавляют детали масштаба
    не меньше ``2π / f_max`` радиан дуги.
    """

    def __init__(self, n_freq: int = 24, f_scale: float = 3.0, f_max: float = 8.0,
                 seed: int = 7):
        super().__init__()
        self.out_dim = 2 * n_freq + 5
        g = torch.Generator().manual_seed(seed)
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
        e = (elev_m / ELEV_SCALE_M).clamp(*ELEV_CLAMP).unsqueeze(-1)
        return torch.cat([torch.sin(proj), torch.cos(proj), xyz,
                          (lat_deg.abs() / 90.0).unsqueeze(-1), e], dim=-1)
