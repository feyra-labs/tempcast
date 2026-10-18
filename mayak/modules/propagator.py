import torch
import torch.nn as nn

from mayak.modules.readout import mode_bounds

TAU_SITE = 0.2
OMEGA_SITE = 0.1


class ModalPropagator(nn.Module):
    """Разворачивает амплитуды мод на горизонт: затухание и поворот фазы."""

    def __init__(self, n_modes, dz, group_sizes, horizon, tau_bounds=(3.0, 240.0),
                 persistent_modes=None):
        super().__init__()
        if sum(group_sizes) != n_modes:
            raise ValueError(f"группы мод {tuple(group_sizes)} не дают {n_modes} мод")
        self.M = n_modes
        self.group_sizes = tuple(int(g) for g in group_sizes)
        self.horizon = int(horizon)
        lo, hi = mode_bounds(tau_bounds, n_modes)
        self.register_buffer("tau_lo", lo, persistent=False)
        self.register_buffer("tau_hi", hi, persistent=False)
        flags = [False] * n_modes if persistent_modes is None else list(persistent_modes)
        if len(flags) != n_modes:
            raise ValueError(f"флагов квазипостоянных мод {len(flags)}, а мод {n_modes}")
        # Флаги следуют из конфига, в состояние модуля они не пишутся.
        self.register_buffer("persistent_w", torch.tensor([float(bool(f)) for f in flags]),
                             persistent=False)
        self.site = nn.Linear(dz, 2 * n_modes)
        nn.init.zeros_(self.site.weight)
        nn.init.zeros_(self.site.bias)
        self.w_re = nn.Parameter(torch.ones(n_modes))
        self.w_im = nn.Parameter(torch.zeros(n_modes))

    def site_constants(self, z, tau, omega):
        """Постоянные времени и частоты мод после подстройки под станцию."""
        M = self.M
        d = self.site(z)
        tau_s = (tau[None, :] * torch.exp(TAU_SITE * torch.tanh(d[:, :M]))
                 ).clamp(self.tau_lo, self.tau_hi)
        tuned = omega[None, :] * torch.exp(OMEGA_SITE * torch.tanh(d[:, M:]))
        omg_s = torch.where(self.persistent_w > 0, omega[None, :], tuned)
        return tau_s, omg_s

    def forward(self, a_re, a_im, z, tau, omega):
        """Вклады мод на лидах горизонта."""
        tau_s, omg_s = self.site_constants(z, tau, omega)

        h = torch.arange(1, self.horizon + 1, dtype=a_re.dtype, device=a_re.device)
        dec = torch.exp(-h[None, :, None] / tau_s[:, None, :])
        ang = omg_s[:, None, :] * h[None, :, None]
        co, si = torch.cos(ang), torch.sin(ang)

        c_re = dec * (a_re[:, None, :] * co - a_im[:, None, :] * si)
        c_im = dec * (a_re[:, None, :] * si + a_im[:, None, :] * co)
        wp = self.persistent_w
        ww = 1.0 - wp
        o = (torch.einsum("bhm,m->bh", c_re, self.w_re * ww)
             + torch.einsum("bhm,m->bh", c_im, self.w_im * ww))
        o_p = (torch.einsum("bhm,m->bh", c_re, self.w_re * wp)
               + torch.einsum("bhm,m->bh", c_im, self.w_im * wp))

        amp = torch.sqrt(a_re ** 2 + a_im ** 2 + 1e-12)
        Eg = torch.stack(
            [g.sum(-1) for g in (dec * amp[:, None, :]).split(self.group_sizes, dim=-1)],
            dim=-1)
        return o, o_p, Eg
