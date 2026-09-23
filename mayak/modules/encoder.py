"""Синоптический энкодер: строго причинный TCN с инкрементальным шагом.

Выход энкодера в момент t зависит только от входа в моменты t − RF + 1 … t, где
RF = (k − 1)·Σd + 1 - рецептивное поле. Для ядра 3 и дилатаций
(1, 1, 2, 2, …, 32, 32) это 2·126 + 1 = 253 ч (ModelConfig.receptive_field).

Два режима с одними и теми же весами:

* ``forward(x)``      - пакетный проход по окну (B, n_ch, L) → (B, L, width);
* ``step(x_t, state)`` - потактовый шаг (B, n_ch) → (B, width). Каждый блок держит
  кольцевой буфер своих (k − 1)·d последних входов; стоимость шага
  O(глубина · ширина²) и не зависит от длины рецептивного поля.
* ``step_shift(x_t, buf)`` - тот же шаг без внутреннего состояния:
  буферы всех блоков приходят одним тензором в хронологическом порядке и
  возвращаются сдвинутыми на час. Кольцо и индекс t живут в хост-коде рантайма.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGroupNorm(nn.Module):
    """Групповая нормализация по каналам в пределах одного момента времени.

    Каналы делятся на ``num_groups`` групп; среднее и дисперсия считаются по каналам
    группы **в данный момент** и не затрагивают ось времени (в отличие от
    nn.GroupNorm на (B, C, L), который усредняет и по L). Вход (B, C) или (B, C, L).
    """

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"{num_channels} каналов не делятся на {num_groups} групп")
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if x.dim() == 2:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        B, C, L = x.shape
        y = F.group_norm(x.transpose(1, 2).reshape(B * L, C), self.num_groups,
                         self.weight, self.bias, self.eps)
        return y.reshape(B, L, C).transpose(1, 2)


class DSBlock(nn.Module):
    """Depthwise-separable причинный блок с residual-связью.

    Ядро k с дилатацией d видит моменты t, t − d, …, t − (k − 1)·d: слева
    дополняется (k − 1)·d нулями, выход в момент t не зависит от будущего входа.
    Потактовый шаг держит кольцевой буфер из (k − 1)·d последних входов блока.
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

    def step(self, x_t, buf, t):
        """Выход блока в момент t по входу x_t (B, C) и буферу входов (B, C, pad)."""
        P, w = self.pad, self.dw.weight[:, 0, :]          # w[:, j] ↔ лаг (k − 1 − j)·d
        acc = self.dw.bias + w[:, -1] * x_t
        for j in range(self.k - 1):
            acc = acc + w[:, j] * buf[:, :, (t - (self.k - 1 - j) * self.d) % P]
        buf[:, :, t % P] = x_t
        h = F.linear(acc, self.pw.weight[:, :, 0], self.pw.bias)
        return x_t + F.gelu(self.norm(h))

    def step_shift(self, x_t, buf):
        win = torch.cat([buf, x_t[..., None]], dim=-1)
        h = self.pw(self.dw(win))[..., 0]
        return x_t + F.gelu(self.norm(h)), win[..., 1:]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "gn.weight" in state_dict:
            raise RuntimeError(
                f"{prefix}gn: веса обучены с nn.GroupNorm по оси времени (до блока 7). "
                f"Эта нормализация не причинна и несовместима с потактовым шагом - "
                f"модель нужно переобучить.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@dataclass
class EncoderState:
    """Состояние потактового энкодера: кольцевые буферы блоков и число шагов t."""
    bufs: list = field(default_factory=list)
    t: int = 0

    @property
    def nbytes(self):
        return int(sum(b.numel() * b.element_size() for b in self.bufs))

    def clone(self):
        return EncoderState([b.clone() for b in self.bufs], self.t)


class SynopticEncoder(nn.Module):
    """Каузальный TCN из depthwise-separable блоков.

    Вход (B, n_ch, L) → выход (B, L, width). Рецептивное поле (k − 1)·Σd + 1 ч:
    для ядра 3 и дилатаций (1, 1, 2, 2, …, 32, 32) это 2·126 + 1 = 253 ч
    (ModelConfig.receptive_field). Потактовый режим - init_state / step / prefill.
    """

    def __init__(self, n_ch=13, width=48, dilations=(1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32),
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

    def init_state(self, batch_size=1):
        """Состояние до первого шага: нулевые буферы ≡ левое дополнение нулями."""
        p = self.stem.weight
        return EncoderState([p.new_zeros(batch_size, self.width, b.pad) for b in self.blocks], 0)

    @property
    def buffer_pads(self):
        """Длины буферов блоков, ч: (k − 1)·d. Их сумма - длина буфера step_shift."""
        return tuple(b.pad for b in self.blocks)

    def ring_to_shift(self, state):
        """EncoderState (кольца) → буфер step_shift (B, width, Σpad), старший час первым."""
        return torch.cat([buf[:, :, torch.arange(state.t - b.pad, state.t) % b.pad]
                          for buf, b in zip(state.bufs, self.blocks)], dim=-1)

    def step_shift(self, x_t, buf):
        """Один час без внутреннего состояния: (B, n_ch), (B, width, Σpad) →
        (признаки (B, width), новый буфер). Эквивалентно step по кольцам."""
        h = F.linear(x_t, self.stem.weight[:, :, 0], self.stem.bias)
        parts = []
        for b, bb in zip(self.blocks, buf.split(self.buffer_pads, dim=-1)):
            h, nb = b.step_shift(h, bb)
            parts.append(nb)
        return h, torch.cat(parts, dim=-1)

    def step(self, x_t, state):
        """Один час: каналы (B, n_ch) → признаки (B, width)."""
        h = F.linear(x_t, self.stem.weight[:, :, 0], self.stem.bias)
        for b, buf in zip(self.blocks, state.bufs):
            h = b.step(h, buf, state.t)
        state.t += 1
        return h

    def prefill(self, x):
        """Пакетный проход по окну (B, n_ch, L) + состояние после его последнего часа.

        Эквивалентно init_state и L вызовам step, но одним проходом: буфер блока
        заполняется последними pad входами этого блока из пакетного прохода.
        """
        B, _, L = x.shape
        state = self.init_state(B)
        if L == 0:
            return x.new_zeros(B, 0, self.width), state
        h = self.stem(x)
        for b, buf in zip(self.blocks, state.bufs):
            s = torch.arange(max(0, L - b.pad), L, device=x.device)
            buf[:, :, s % b.pad] = h[:, :, s]
            h = b(h)
        state.t = L
        return h.transpose(1, 2), state
