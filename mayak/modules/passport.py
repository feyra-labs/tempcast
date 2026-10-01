import torch
import torch.nn as nn
import torch.nn.functional as F


class Fingerprint(nn.Module):
    """Паспорт станции: нормальное распределение, уточняемое по суточным сводкам.

    Общий приор обновляется по-байесовски: каждые сутки с данными добавляют точность,
    среднее сдвигается к наблюдённому.

    Args:
        dz: размер паспорта.
        hidden: ширина рекуррентного кодировщика сводок.
        n_summary: число суточных сводок на сутки.
        enabled: ложь - абляция ``no_passport``: паспорт и штраф расхождения нулевые,
            обучаемых параметров нет.
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
        """Паспорт станции и расхождение апостериорного распределения с приором.

        Args:
            loc: признаки координат, форма (B, ...); нужен только размер батча.
            summaries: суточные сводки, форма (B, D, n_summary), по одной строке на
                сутки истории.
            day_mask: есть ли данные в сутках, форма (B, D).
            sample: истина на обучении - паспорт выбирается случайно из распределения;
                ложь на инференсе - берётся среднее.

        Returns:
            Пара: паспорт формы (B, dz) и среднее по батчу расхождение Кульбака -
            Лейблера с приором.
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
