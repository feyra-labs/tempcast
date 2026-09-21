import torch
import torch.nn as nn


class ModalPropagator(nn.Module):
    """Разворачивает амплитуды мод на горизонт: затухание + поворот фазы.

    Паспорт z подстраивает τ (±20 %) и ω (±10 %) под станцию. Выход: аномалия o (B, H)
    и энергии групп мод Eg (B, H, число групп) - размеры групп берутся из конфига.
    """

    def __init__(self, n_modes, dz, group_sizes, horizon):
        super().__init__()
        if sum(group_sizes) != n_modes:
            raise ValueError(f"группы мод {tuple(group_sizes)} не дают {n_modes} мод")
        self.M = n_modes
        self.group_sizes = tuple(int(g) for g in group_sizes)
        self.horizon = int(horizon)
        self.site = nn.Linear(dz, 2 * n_modes)
        nn.init.zeros_(self.site.weight)
        nn.init.zeros_(self.site.bias)
        self.w_re = nn.Parameter(torch.ones(n_modes))
        self.w_im = nn.Parameter(torch.zeros(n_modes))

    def forward(self, a_re, a_im, z, tau, omega):
        M = self.M
        d = self.site(z)
        tau_s = tau[None, :] * torch.exp(0.2 * torch.tanh(d[:, :M]))
        omg_s = omega[None, :] * torch.exp(0.1 * torch.tanh(d[:, M:]))

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
