import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.modules import inv_softplus
from mayak.modules.loc import ELEV_SCALE_M

N_YEAR_BASIS = 7
N_DAY_BASIS = 5
N_DAY_SMOOTH = 3
N_MU = N_YEAR_BASIS * N_DAY_BASIS + 1
N_SIG = N_YEAR_BASIS * N_DAY_SMOOTH
N_DEF = N_YEAR_BASIS * N_DAY_SMOOTH
SIGMA_INIT = 2.2
FILM_STRENGTH = 0.3
LAPSE_INIT = -6.5e-3 * ELEV_SCALE_M


class ClimateField(nn.Module):
    """Имплицитное климат-поле: по признакам координат - коэффициенты гармонического базиса."""

    def __init__(self, loc_dim, dz, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(loc_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head_mu = nn.Linear(hidden, N_MU)
        self.head_scale = nn.Linear(hidden, N_SIG + N_DEF)
        with torch.no_grad():
            for head in (self.head_mu, self.head_scale):
                head.weight.mul_(0.1)
                head.bias.zero_()
            self.head_scale.bias[0] = inv_softplus(torch.tensor(SIGMA_INIT)).item()
        self.lapse = nn.Parameter(torch.tensor(LAPSE_INIT))
        self.film = nn.Linear(dz, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    @staticmethod
    def _basis(sin_y, cos_y, sin_d, cos_d):
        one = torch.ones_like(sin_y)
        s2, c2 = 2 * sin_y * cos_y, 1 - 2 * sin_y ** 2
        s3, c3 = 3 * sin_y - 4 * sin_y ** 3, 4 * cos_y ** 3 - 3 * cos_y
        S = torch.stack([one, sin_y, cos_y, s2, c2, s3, c3], dim=-1)
        d2, e2 = 2 * sin_d * cos_d, 1 - 2 * sin_d ** 2
        D5 = torch.stack([one, sin_d, cos_d, d2, e2], dim=-1)
        b35 = (S.unsqueeze(-1) * D5.unsqueeze(-2)).flatten(-2)
        b21 = (S.unsqueeze(-1) * D5[..., :N_DAY_SMOOTH].unsqueeze(-2)).flatten(-2)
        return b35, b21

    def coefficients(self, loc, z=None):
        """Коэффициенты поля точки."""
        h = self.fc2(F.gelu(self.fc1(loc)))
        c_mu = self.head_mu(F.gelu(h))
        # Постоянный член базиса среднего (первый) плюс градиент по высоте.
        c_mu = torch.cat([c_mu[:, :1] + self.lapse * loc[:, -1:], c_mu[:, 1:]], dim=-1)
        if z is not None:
            gb = torch.tanh(self.film(z))
            k = self.hidden
            h = h * (1 + FILM_STRENGTH * gb[:, :k]) + FILM_STRENGTH * gb[:, k:]
        c = self.head_scale(F.gelu(h))
        return c_mu, c[:, :N_SIG], c[:, N_SIG:]

    def evaluate(self, coefs, astro):
        """Поле в наборе моментов."""
        c_mu, c_sig, c_def = coefs
        sin_d, cos_d, _, czp, sin_y, cos_y = astro
        b35, b21 = self._basis(sin_y, cos_y, sin_d, cos_d)
        mu = (torch.einsum("bi,bki->bk", c_mu[:, :N_MU - 1], b35)
              + c_mu[:, N_MU - 1:N_MU] * czp)
        sigma = (0.8 + F.softplus(torch.einsum("bi,bki->bk", c_sig, b21))).clamp(max=12.0)
        defc = F.softplus(torch.einsum("bi,bki->bk", c_def, b21))
        return mu, sigma, defc
