import torch
import torch.nn as nn

from mayak.constants import M, DZ, GROUPS, H


class ModalPropagator(nn.Module):

    def __init__(self):
        super().__init__()
        self.site = nn.Linear(DZ, 2 * M) 
        nn.init.zeros_(self.site.weight)
        nn.init.zeros_(self.site.bias)
        self.w_re = nn.Parameter(torch.ones(M))
        self.w_im = nn.Parameter(torch.zeros(M))

    def forward(self, a_re, a_im, z, tau, omega):
        d = self.site(z)
        tau_s = tau[None, :] * torch.exp(0.2 * torch.tanh(d[:, :M]))
        omg_s = omega[None, :] * torch.exp(0.1 * torch.tanh(d[:, M:]))

        h = torch.arange(1, H + 1, dtype=a_re.dtype, device=a_re.device)
        dec = torch.exp(-h[None, :, None] / tau_s[:, None, :])
        ang = omg_s[:, None, :] * h[None, :, None]
        co, si = torch.cos(ang), torch.sin(ang)

        c_re = dec * (a_re[:, None, :] * co - a_im[:, None, :] * si)
        c_im = dec * (a_re[:, None, :] * si + a_im[:, None, :] * co)
        o = (torch.einsum("bhm,m->bh", c_re, self.w_re)
             + torch.einsum("bhm,m->bh", c_im, self.w_im))

        amp = torch.sqrt(a_re ** 2 + a_im ** 2 + 1e-12)
        Eg = torch.stack(
            [g.sum(-1) for g in (dec * amp[:, None, :]).split(GROUPS, dim=-1)],
            dim=-1)
        return o, Eg