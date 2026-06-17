import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import N_CH, C_ENC


class DSBlock(nn.Module):
    """Depthwise-separable блок с residual-связью."""

    def __init__(self, c, dilation):
        super().__init__()
        self.d = dilation
        self.dw = nn.Conv1d(c, c, 3, dilation=dilation, groups=c)
        self.pw = nn.Conv1d(c, c, 1)
        self.gn = nn.GroupNorm(4, c)

    def forward(self, x):
        h = self.dw(F.pad(x, (2 * self.d, 0)))
        h = F.gelu(self.gn(self.pw(h)))
        return x + h
    
class SynopticEncoder(nn.Module):
    """Каузальный TCN: 12 depthwise-separable блоков, рецептивное поле ≈127 ч."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(N_CH, C_ENC, 1)
        dil = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.blocks = nn.ModuleList(DSBlock(C_ENC, d) for d in dil)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)