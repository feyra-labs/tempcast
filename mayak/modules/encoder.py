import torch.nn as nn
import torch.nn.functional as F


class DSBlock(nn.Module):
    """Depthwise-separable причинный блок с residual-связью.

    Ядро k с дилатацией d видит моменты t, t − d, …, t − (k − 1)·d: слева
    дополняется (k − 1)·d нулями, выход в момент t не зависит от будущего входа.
    """

    def __init__(self, c, dilation, kernel=3, norm_groups=4):
        super().__init__()
        self.d = dilation
        self.pad = (kernel - 1) * dilation
        self.dw = nn.Conv1d(c, c, kernel, dilation=dilation, groups=c)
        self.pw = nn.Conv1d(c, c, 1)
        self.gn = nn.GroupNorm(norm_groups, c)

    def forward(self, x):
        h = self.dw(F.pad(x, (self.pad, 0)))
        h = F.gelu(self.gn(self.pw(h)))
        return x + h


class SynopticEncoder(nn.Module):
    """Каузальный TCN из depthwise-separable блоков.

    Вход (B, n_ch, L) → выход (B, L, width). Рецептивное поле (k − 1)·Σd + 1 ч:
    для ядра 3 и дилатаций (1, 1, 2, 2, …, 32, 32) это 2·126 + 1 = 253 ч
    (ModelConfig.receptive_field).
    """

    def __init__(self, n_ch=13, width=48, dilations=(1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32),
                 kernel=3, norm_groups=4):
        super().__init__()
        self.receptive_field = (kernel - 1) * sum(dilations) + 1
        self.stem = nn.Conv1d(n_ch, width, 1)
        self.blocks = nn.ModuleList(DSBlock(width, d, kernel, norm_groups) for d in dilations)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)
