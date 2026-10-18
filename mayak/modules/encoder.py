"""Синоптический энкодер: строго причинный TCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGroupNorm(nn.Module):
    """Групповая нормализация по каналам в пределах одного момента времени."""

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
    """Depthwise-separable причинный блок с residual-связью."""

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


class SynopticEncoder(nn.Module):
    """Причинный TCN из depthwise-separable блоков."""

    def __init__(self, n_ch=14, width=48, dilations=(1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32),
                 kernel=3, norm_groups=4):
        super().__init__()
        self.width = width
        self.stem = nn.Conv1d(n_ch, width, 1)
        self.blocks = nn.ModuleList(DSBlock(width, d, kernel, norm_groups) for d in dilations)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)
