import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EVIDENCE_FLOOR = 1.0
OMEGA_TUNE = 0.15


def mode_bounds(tau_bounds, n_modes):
    """Пределы постоянной времени по модам.

    Args:
        tau_bounds: одна пара (τ_min, τ_max), ч, на все моды или по паре на каждую моду.
        n_modes: число мод.

    Returns:
        Пара тензоров формы (n_modes,): нижние и верхние пределы, ч.

    Raises:
        ValueError: пар не одна и не по числу мод.
    """
    b = torch.as_tensor(tau_bounds, dtype=torch.float32)
    if b.dim() == 1:
        b = b[None].expand(n_modes, -1)
    if tuple(b.shape) != (n_modes, 2):
        raise ValueError(f"пределы постоянных времени формы {tuple(b.shape)}: нужна одна пара "
                         f"или {n_modes} пар, по одной на моду")
    return b[:, 0].clone(), b[:, 1].clone()


class LaplaceReadout(nn.Module):
    """Считывание затухающих мод по истории признаков энкодера.

    Каждая мода затухает со своей постоянной времени и вращается со своей частотой.
    Частота моды - начальная частота, умноженная на обучаемый множитель в пределах
    exp(±OMEGA_TUNE). У мод с фиксированной частотой множителя нет: частота равна
    начальной. Формула моды - в описании модели.

    Args:
        width: ширина признаков энкодера.
        groups: группы мод с начальными постоянными времени и периодами, ч; начальная
            частота - полный оборот за период.
        tau_bounds: пределы постоянной времени, ч: одна пара на все моды или по паре на
            каждую моду. Обучаемый параметр через сигмоиду отображается внутрь пределов
            своей моды.
        compression: доказательное сжатие: накопленное состояние делится на массу
            свидетельств плюс обучаемую силу сжатия, поэтому при малой массе аномалия
            стягивается к нулю. Ложь - абляция ``no_compression``: деление только на
            массу свидетельств с нижней границей.
        fixed_freq: флаги мод с фиксированной частотой, по одному на моду; None - таких
            мод нет.

    Raises:
        ValueError: флагов фиксированной частоты не по числу мод.
    """

    def __init__(self, width, groups, tau_bounds=(3.0, 240.0), compression=True,
                 fixed_freq=None):
        super().__init__()
        self.compression = compression
        tau0 = torch.tensor([t for g in groups for t in g.tau0])
        period = torch.tensor([p for g in groups for p in g.period])
        M = len(tau0)
        self.n_modes = M
        lo, hi = mode_bounds(tau_bounds, M)
        fixed = [False] * M if fixed_freq is None else [bool(f) for f in fixed_freq]
        if len(fixed) != M:
            raise ValueError(f"флагов фиксированной частоты {len(fixed)}, а мод {M}")
        # Пределы и флаги следуют из конфига, который лежит в чекпойнте, поэтому в
        # состояние модуля они не пишутся.
        self.register_buffer("tau_lo", lo, persistent=False)
        self.register_buffer("tau_hi", hi, persistent=False)
        self.register_buffer("fixed_freq", torch.tensor(fixed), persistent=False)
        self.raw_tau = nn.Parameter(torch.logit(((tau0 - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)))

        w = 2 * math.pi
        osc = period > 0
        omega0 = torch.where(osc, w / torch.where(osc, period, torch.ones_like(period)),
                             torch.zeros_like(period))
        self.register_buffer("omega0", omega0)
        self.p_w = nn.Parameter(torch.zeros(M))
        self.p_k = nn.Parameter(torch.full((M,), 2.3))
        self.proj = nn.Linear(width, 2 * M)

    def constants(self):
        """Постоянные времени, частоты и силы сжатия мод.

        Returns:
            Тройка тензоров формы (M,): постоянные времени, ч, частоты, рад/ч, - у мод с
            фиксированной частотой начальные, - и силы сжатия.
        """
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * torch.sigmoid(self.raw_tau)
        tuned = self.omega0 * torch.exp(OMEGA_TUNE * torch.tanh(self.p_w))
        omega = torch.where(self.fixed_freq, self.omega0, tuned)
        kappa = 1.0 + F.softplus(self.p_k)
        return tau, omega, kappa

    def normalize(self, n_re, n_im, e, kappa=None):
        """Амплитуды мод по накопленному состоянию.

        Args:
            n_re: действительная часть накопленного состояния, форма (B, M).
            n_im: мнимая часть, форма (B, M).
            e: масса свидетельств, форма (B, M).
            kappa: сила сжатия по модам; None - из текущих параметров.

        Returns:
            Пара: действительные и мнимые амплитуды, форма (B, M).
        """
        if self.compression:
            if kappa is None:
                kappa = self.constants()[2]
            den = e + kappa[None, :]
        else:
            den = e.clamp_min(EVIDENCE_FLOOR)
        return n_re / den, n_im / den

    def project(self, feats):
        """Вклад часов в моды по признакам энкодера.

        Args:
            feats: признаки энкодера, форма (..., width).

        Returns:
            Вклад, форма (..., 2M): сначала действительные части, затем мнимые.
        """
        return self.proj(feats)

    def accumulate(self, u, v):
        """Сумма вкладов часов с затуханием и поворотом фазы к моменту выпуска.

        Последний час последовательности - момент выпуска.

        Args:
            u: вклады часов, форма (B, L, 2M), от старых к новым.
            v: маска наличия температуры, форма (B, L).

        Returns:
            Тройка: действительная и мнимая части накопленного состояния мод и масса
            свидетельств, каждая формы (B, M).
        """
        tau, omega, _ = self.constants()
        M = self.n_modes
        Lh = u.shape[1]
        lag = torch.arange(Lh - 1, -1, -1, dtype=u.dtype, device=u.device)
        dec = torch.exp(-lag[:, None] / tau[None, :])
        ph = omega[None, :] * lag[:, None]
        kc, ks = dec * torch.cos(ph), dec * torch.sin(ph)
        uc = u[..., :M] * v[..., None]
        us = u[..., M:] * v[..., None]

        n_re = torch.einsum("blm,lm->bm", uc, kc) - torch.einsum("blm,lm->bm", us, ks)
        n_im = torch.einsum("blm,lm->bm", uc, ks) + torch.einsum("blm,lm->bm", us, kc)
        e = torch.einsum("bl,lm->bm", v, dec)
        return n_re, n_im, e

    def forward(self, feats, v):
        n_re, n_im, e = self.accumulate(self.project(feats), v)
        a_re, a_im = self.normalize(n_re, n_im, e)
        return a_re, a_im, e
