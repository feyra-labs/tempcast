import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import DZ, inv_softplus
from mayak.modules.loc import LocEncoder


class ClimateField(nn.Module):
    """Имплицитное климат-поле: координаты → коэффициенты гармонического базиса"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LocEncoder.OUT, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 36 + 21 + 21)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()
            self.head.bias[36] = inv_softplus(torch.tensor(2.2)).item()
        self.film = nn.Linear(DZ, 2 * 128)
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
        b21 = (S.unsqueeze(-1) * D5[..., :3].unsqueeze(-2)).flatten(-2)
        return b35, b21
    
    def coefficients(self, loc, z=None):
        h = F.gelu(self.fc1(loc))
        h = self.fc2(h)
        if z is not None:
            gb = self.film(z)
            h = h * (1 + gb[:, :128]) + gb[:, 128:]
        c = self.head(F.gelu(h))
        return c[:, :36], c[:, 36:57], c[:, 57:78]

    def evaluate(self, coefs, astro):
        """Поле в наборе моментов. astro — кортеж из astro_features"""
        c_mu, c_sig, c_def = coefs
        sin_d, cos_d, _, czp, sin_y, cos_y = astro
        b35, b21 = self._basis(sin_y, cos_y, sin_d, cos_d)
        mu = torch.einsum("bi,bki->bk", c_mu[:, :35], b35) + c_mu[:, 35:36] * czp
        sigma = (0.8 + F.softplus(torch.einsum("bi,bki->bk", c_sig, b21))).clamp(max=12.0)
        defc = F.softplus(torch.einsum("bi,bki->bk", c_def, b21))
        return mu, sigma, defc