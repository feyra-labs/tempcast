from statistics import NormalDist

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import inv_softplus

R_MAX = 0.6
R_KAPPA_FLOOR = 1.0
R_KAPPA_RAW_INIT = 2.3


def normal_gaps(quantiles, decimals=3):
    """Начальные зазоры между соседними квантилями стандартного нормального распределения.

    Зазоры идут от медианы наружу, квантили нормального распределения округляются до
    тысячных.

    Args:
        quantiles: уровни квантилей по возрастанию, среди них есть медиана.
        decimals: число знаков после запятой при округлении.

    Returns:
        Пара списков: зазоры ниже медианы и зазоры выше медианы.
    """
    q = list(quantiles)
    m = q.index(0.5)
    z = [round(NormalDist().inv_cdf(x), decimals) for x in q]
    lower = [z[i + 1] - z[i] for i in range(m - 1, -1, -1)]
    upper = [z[i] - z[i - 1] for i in range(m + 1, len(q))]
    return lower, upper


class Heads(nn.Module):
    """Головы прогноза: поправка к аномалии, масштаб интервала и квантильные смещения.

    Поправка ограничена по модулю и умножается на вес доверия к истории. Вес растёт
    со средней по модам массой свидетельств так же, как доказательное сжатие мод:
    масса делится на сумму массы и обучаемого порога. Без истории масса равна нулю,
    вес равен нулю, и медиана совпадает с климат-полем при любых весах голов.
    Масштаб интервала и зазоры между квантилями от веса не зависят: при пустой истории
    интервал по-прежнему учится быть климатологическим.

    Attributes:
        r_kappa: сырой параметр порога веса поправки. Сам порог не меньше одного часа
            свидетельств, чтобы вес не превращался в ступеньку при исчезающе малой массе.
            При инициализации порог около 3.4 ч, как у сжатия мод: поправка набирает
            силу с той же скоростью, что и аномалия.
    """

    def __init__(self, dz, n_groups, n_sun, quantiles, hidden=48, z_proj=4):
        super().__init__()
        lower, upper = normal_gaps(quantiles)
        self.n_lo, self.n_hi = len(lower), len(upper)
        self.z_proj = z_proj
        self.in_dim = 1 + n_groups + 1 + n_sun + 1 + z_proj + 1 + 1
        self.zproj = nn.Linear(dz, z_proj)
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 + self.n_lo + self.n_hi)
        self.r_kappa = nn.Parameter(torch.tensor(R_KAPPA_RAW_INIT))
        gaps = torch.tensor(lower + upper)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            self.fc2.bias[1] = 0.9
            self.fc2.bias[2:] = inv_softplus(gaps)

    def evidence_gate(self, e):
        """Вес поправки по массе свидетельств.

        Args:
            e: масса свидетельств по модам, форма (B, M).

        Returns:
            Вес от нуля до единицы, форма (B,). Он равен нулю ровно тогда, когда в
            истории нет ни одного валидного часа.
        """
        e_mean = e.mean(-1)
        kappa = R_KAPPA_FLOOR + F.softplus(self.r_kappa)
        return e_mean / (e_mean + kappa)

    def forward(self, o, Eg, sun_fut, log_sigma, z, e):
        """Поправка, масштаб интервала и смещения квантилей на всех лидах.

        Args:
            o: аномалия из мод в единицах климатологического разброса, форма (B, H).
            Eg: энергии групп мод на лидах, форма (B, H, число групп).
            sun_fut: солнечные ковариаты на лидах, форма (B, H, n_sun).
            log_sigma: логарифм климатологического разброса на лидах, форма (B, H).
            z: паспорт станции, форма (B, dz).
            e: масса свидетельств по модам, форма (B, M).

        Returns:
            Тройка: поправка (B, H), масштаб интервала (B, H) и смещения квантилей от
            медианы (B, H, число квантилей) с нулём на месте медианы.
        """
        Bsz, Hn = o.shape
        zp = self.zproj(z)[:, None, :].expand(Bsz, Hn, self.z_proj)
        hn = (torch.arange(1, Hn + 1, dtype=o.dtype, device=o.device) / Hn
              )[None, :, None].expand(Bsz, Hn, 1)
        le = torch.log1p(e.mean(-1))[:, None, None].expand(Bsz, Hn, 1)

        x = torch.cat([o[..., None], Eg, Eg.sum(-1, keepdim=True), sun_fut,
                       log_sigma[..., None], zp, hn, le], dim=-1)
        out = self.fc2(F.gelu(self.fc1(x)))

        r = R_MAX * torch.tanh(out[..., 0]) * self.evidence_gate(e)[:, None]
        ratio = 0.08 + torch.sigmoid(out[..., 1] + 1.5)
        gaps = F.softplus(out[..., 2:])

        lo = torch.flip(torch.cumsum(gaps[..., :self.n_lo], dim=-1), dims=(-1,))
        hi = torch.cumsum(gaps[..., self.n_lo:], dim=-1)
        off = torch.cat([-lo, torch.zeros_like(r)[..., None], hi], dim=-1)
        return r, ratio, off

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "fc2.weight" in state_dict and prefix + "r_kappa" not in state_dict:
            raise RuntimeError(
                f"{prefix}r_kappa: в чекпойнте нет порога веса поправки. Веса обучены с "
                f"поправкой, которая не зависит от массы свидетельств и сдвигает медиану "
                f"холодного старта от климат-поля; модель нужно переобучить.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
