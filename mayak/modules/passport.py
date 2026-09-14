import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import DZ


class Fingerprint(nn.Module):
    """Паспорт станции z ~ N(m, v)"""

    def __init__(self):
        super().__init__()
        self.prior_m = nn.Parameter(torch.zeros(DZ))
        self.prior_s = nn.Parameter(torch.zeros(DZ))

        self.gru = nn.GRU(input_size=7, hidden_size=32, batch_first=True)
        self.obs = nn.Linear(32, 2 * DZ)

    def forward(self, loc, summaries, day_mask, sample: bool):
        """
        loc: (B, 101) координатные признаки
        summaries: (B, 28, 6) суточные сводки (см. daily_summaries в model.py)
        day_mask: (B, 28) есть ли данные в этих сутках (доля валидных часов > 0)
        sample: True на обучении (репараметризация), False на инференсе (берём m)
        """
        B = loc.shape[0]
        m0 = self.prior_m.unsqueeze(0).expand(B, -1)
        v0 = (F.softplus(self.prior_s) + 1e-3).unsqueeze(0).expand(B, -1)

        x = torch.cat([summaries, day_mask.unsqueeze(-1)], dim=-1)
        _, hN = self.gru(x * day_mask.unsqueeze(-1))
        o = self.obs(hN[-1])
        n_days = day_mask.sum(-1, keepdim=True)
        prec1 = F.softplus(o[:, DZ:]) * n_days 
        prec0 = 1.0 / v0

        prec = prec0 + prec1
        m = (prec0 * m0 + prec1 * (m0 + o[:, :DZ])) / prec
        v = 1.0 / prec

        z = m + torch.randn_like(m) * v.sqrt() if sample else m
        kl = 0.5 * (v / v0 + (m - m0) ** 2 / v0 - 1.0 + torch.log(v0 / v)).sum(-1).mean()
        return z, kl
