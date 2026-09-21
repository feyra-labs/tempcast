import torch
import torch.nn as nn
import torch.nn.functional as F


class Fingerprint(nn.Module):
    """Паспорт станции z ~ N(m, v): глобальный прайор + байесовское обновление по сводкам.

    enabled=False - абляция no_passport: z ≡ 0, KL = 0, обучаемых параметров нет.
    """

    def __init__(self, dz=16, hidden=32, n_summary=6, enabled=True):
        super().__init__()
        self.dz = dz
        self.enabled = enabled
        if not enabled:
            return
        self.prior_m = nn.Parameter(torch.zeros(dz))
        self.prior_s = nn.Parameter(torch.zeros(dz))

        self.gru = nn.GRU(input_size=n_summary + 1, hidden_size=hidden, batch_first=True)
        self.obs = nn.Linear(hidden, 2 * dz)

    def forward(self, loc, summaries, day_mask, sample: bool):
        """
        loc: (B, ·) координатные признаки (нужен только размер батча)
        summaries: (B, D, n_summary) суточные сводки, D = L // 24 (MAYAK.daily_summaries)
        day_mask: (B, D) есть ли данные в этих сутках (доля валидных часов > 0)
        sample: True на обучении (репараметризация), False на инференсе (берём m)
        """
        B = loc.shape[0]
        if not self.enabled:
            return loc.new_zeros(B, self.dz), loc.new_zeros(())
        dz = self.dz
        m0 = self.prior_m.unsqueeze(0).expand(B, -1)
        v0 = (F.softplus(self.prior_s) + 1e-3).unsqueeze(0).expand(B, -1)

        x = torch.cat([summaries, day_mask.unsqueeze(-1)], dim=-1)
        _, hN = self.gru(x * day_mask.unsqueeze(-1))
        o = self.obs(hN[-1])
        n_days = day_mask.sum(-1, keepdim=True)
        prec1 = F.softplus(o[:, dz:]) * n_days
        prec0 = 1.0 / v0

        prec = prec0 + prec1
        m = (prec0 * m0 + prec1 * (m0 + o[:, :dz])) / prec
        v = 1.0 / prec

        z = m + torch.randn_like(m) * v.sqrt() if sample else m
        kl = 0.5 * (v / v0 + (m - m0) ** 2 / v0 - 1.0 + torch.log(v0 / v)).sum(-1).mean()
        return z, kl
