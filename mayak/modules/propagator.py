import torch
import torch.nn as nn

TAU_SITE = 0.2
OMEGA_SITE = 0.1


class ModalPropagator(nn.Module):
    """Разворачивает амплитуды мод на горизонт: затухание и поворот фазы.

    Паспорт станции подстраивает постоянные времени примерно на двадцать процентов и
    частоты примерно на десять процентов. Подстроенная постоянная времени обрезается
    в допустимый диапазон мод: без обрезки самая медленная мода растягивалась бы за
    верхнюю границу, и аномалия к концу горизонта затухала бы слабее обещанного.

    Выход: аномалия на лидах и энергии групп мод на лидах, размеры групп задаёт конфиг.
    """

    def __init__(self, n_modes, dz, group_sizes, horizon, tau_bounds=(3.0, 240.0)):
        super().__init__()
        if sum(group_sizes) != n_modes:
            raise ValueError(f"группы мод {tuple(group_sizes)} не дают {n_modes} мод")
        self.M = n_modes
        self.group_sizes = tuple(int(g) for g in group_sizes)
        self.horizon = int(horizon)
        self.tau_lo, self.tau_hi = float(tau_bounds[0]), float(tau_bounds[1])
        self.site = nn.Linear(dz, 2 * n_modes)
        nn.init.zeros_(self.site.weight)
        nn.init.zeros_(self.site.bias)
        self.w_re = nn.Parameter(torch.ones(n_modes))
        self.w_im = nn.Parameter(torch.zeros(n_modes))

    def site_constants(self, z, tau, omega):
        """Постоянные времени и частоты мод после подстройки под станцию.

        Args:
            z: паспорт станции, форма (B, dz).
            tau: общие постоянные времени мод в часах, форма (M,).
            omega: общие частоты мод в радианах в час, форма (M,).

        Returns:
            Пара: постоянные времени (B, M), обрезанные в допустимый диапазон, и
            частоты (B, M).
        """
        M = self.M
        d = self.site(z)
        tau_s = (tau[None, :] * torch.exp(TAU_SITE * torch.tanh(d[:, :M]))
                 ).clamp(self.tau_lo, self.tau_hi)
        omg_s = omega[None, :] * torch.exp(OMEGA_SITE * torch.tanh(d[:, M:]))
        return tau_s, omg_s

    def forward(self, a_re, a_im, z, tau, omega):
        tau_s, omg_s = self.site_constants(z, tau, omega)

        h = torch.arange(1, self.horizon + 1, dtype=a_re.dtype, device=a_re.device)
        dec = torch.exp(-h[None, :, None] / tau_s[:, None, :])
        ang = omg_s[:, None, :] * h[None, :, None]
        co, si = torch.cos(ang), torch.sin(ang)

        c_re = dec * (a_re[:, None, :] * co - a_im[:, None, :] * si)
        c_im = dec * (a_re[:, None, :] * si + a_im[:, None, :] * co)
        o = (torch.einsum("bhm,m->bh", c_re, self.w_re)
             + torch.einsum("bhm,m->bh", c_im, self.w_im))

        amp = torch.sqrt(a_re ** 2 + a_im ** 2 + 1e-12)
        Eg = torch.stack(
            [g.sum(-1) for g in (dec * amp[:, None, :]).split(self.group_sizes, dim=-1)],
            dim=-1)
        return o, Eg
