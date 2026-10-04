"""Синоптический энкодер: строго причинный TCN.

Выход энкодера в данный час зависит только от входа за последние часы в пределах
рецептивного поля. Его длина - сумма дилатаций, умноженная на ядро без единицы, плюс
сам текущий час. Для ядра 3 и дилатаций от 1 до 32, каждая дважды, это 253 ч.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGroupNorm(nn.Module):
    """Групповая нормализация по каналам в пределах одного момента времени.

    Каналы делятся на ``num_groups`` групп; среднее и дисперсия считаются по каналам
    группы в данный момент и не затрагивают ось времени. Обычная групповая нормализация
    по окну усреднила бы и по времени и потеряла бы причинность.

    Args:
        num_groups: число групп каналов.
        num_channels: число каналов, делится на число групп.
        eps: добавка к дисперсии.

    Raises:
        ValueError: каналы не делятся на группы поровну.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"{num_channels} каналов не делятся на {num_groups} групп")
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, L = x.shape
        y = F.group_norm(x.transpose(1, 2).reshape(B * L, C), self.num_groups,
                         self.weight, self.bias, self.eps)
        return y.reshape(B, L, C).transpose(1, 2)


class DSBlock(nn.Module):
    """Depthwise-separable причинный блок с residual-связью.

    Ядро с дилатацией видит текущий час и часы назад с шагом, равным дилатации. Слева
    окно дополняется нулями на длину поля блока - ядро без единицы, умноженное на
    дилатацию, - поэтому выход в данный час не зависит от будущего входа.

    Args:
        c: число каналов.
        dilation: дилатация свёртки.
        kernel: ядро свёртки.
        norm_groups: число групп канальной нормализации.
    """

    def __init__(self, c, dilation, kernel=3, norm_groups=4):
        super().__init__()
        self.d, self.k = dilation, kernel
        self.pad = (kernel - 1) * dilation
        self.dw = nn.Conv1d(c, c, kernel, dilation=dilation, groups=c)
        self.pw = nn.Conv1d(c, c, 1)
        self.norm = ChannelGroupNorm(norm_groups, c)

    def forward(self, x):
        h = self.dw(F.pad(x, (self.pad, 0)))
        return x + F.gelu(self.norm(self.pw(h)))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "gn.weight" in state_dict:
            raise RuntimeError(
                f"{prefix}gn: веса обучены с нормализацией по всей оси времени окна. "
                f"Эта нормализация не причинна и несовместима с потактовым шагом - "
                f"модель нужно переобучить.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class SynopticEncoder(nn.Module):
    """Причинный TCN из depthwise-separable блоков.

    Проход по окну переводит каналы формы (B, n_ch, L) в признаки формы (B, L, width).

    Args:
        n_ch: число входных каналов.
        width: ширина признаков.
        dilations: дилатации блоков по порядку.
        kernel: ядро свёрток.
        norm_groups: число групп канальной нормализации.
    """

    def __init__(self, n_ch=14, width=48, dilations=(1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32),
                 kernel=3, norm_groups=4):
        super().__init__()
        self.width = width
        self.receptive_field = (kernel - 1) * sum(dilations) + 1
        self.stem = nn.Conv1d(n_ch, width, 1)
        self.blocks = nn.ModuleList(DSBlock(width, d, kernel, norm_groups) for d in dilations)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)
