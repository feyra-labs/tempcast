import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import inv_softplus

N_YEAR_BASIS = 7
N_DAY_BASIS = 5
N_DAY_SMOOTH = 3
N_MU = N_YEAR_BASIS * N_DAY_BASIS + 1
N_SIG = N_YEAR_BASIS * N_DAY_SMOOTH
N_DEF = N_YEAR_BASIS * N_DAY_SMOOTH
SIGMA_INIT = 2.2
FILM_STRENGTH = 0.3


class ClimateField(nn.Module):
    """Имплицитное климат-поле: координаты → коэффициенты гармонического базиса.

    Среднее C - по годовому × суточному базису (N_MU коэффициентов), масштаб σ и
    дефицит точки росы - по сглаженному базису (N_SIG, N_DEF). Паспорт z модулирует
    скрытый слой через FiLM, ограниченный ±FILM_STRENGTH.
    """

    def __init__(self, loc_dim, dz, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(loc_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_MU + N_SIG + N_DEF)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()
            self.head.bias[N_MU] = inv_softplus(torch.tensor(SIGMA_INIT)).item()
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
        h = F.gelu(self.fc1(loc))
        h = self.fc2(h)
        if z is not None:
            gb = torch.tanh(self.film(z))
            k = self.hidden
            h = h * (1 + FILM_STRENGTH * gb[:, :k]) + FILM_STRENGTH * gb[:, k:]
        c = self.head(F.gelu(h))
        return c[:, :N_MU], c[:, N_MU:N_MU + N_SIG], c[:, N_MU + N_SIG:]

    def evaluate(self, coefs, astro):
        """Поле в наборе моментов. astro — кортеж из astro_features"""
        c_mu, c_sig, c_def = coefs
        sin_d, cos_d, _, czp, sin_y, cos_y = astro
        b35, b21 = self._basis(sin_y, cos_y, sin_d, cos_d)
        mu = (torch.einsum("bi,bki->bk", c_mu[:, :N_MU - 1], b35)
              + c_mu[:, N_MU - 1:N_MU] * czp)
        sigma = (0.8 + F.softplus(torch.einsum("bi,bki->bk", c_sig, b21))).clamp(max=12.0)
        defc = F.softplus(torch.einsum("bi,bki->bk", c_def, b21))
        return mu, sigma, defc


class ConstantAnchor(nn.Module):
    """Абляция no_anchor: вместо климат-поля - глобальные обучаемые константы."""

    def __init__(self):
        super().__init__()
        self.anchor_mu = nn.Parameter(torch.zeros(1))
        self.anchor_sig = nn.Parameter(inv_softplus(torch.tensor([SIGMA_INIT])))
        self.anchor_def = nn.Parameter(torch.zeros(1))

    def coefficients(self, loc, z=None):
        B = loc.shape[0]
        return tuple(p.expand(B, 1) for p in (self.anchor_mu, self.anchor_sig, self.anchor_def))

    def evaluate(self, coefs, astro):
        c_mu, c_sig, c_def = coefs
        like = astro[0]
        mu = c_mu.expand_as(like)
        sigma = (0.8 + F.softplus(c_sig)).clamp(max=12.0).expand_as(like)
        defc = F.softplus(c_def).expand_as(like)
        return mu, sigma, defc
