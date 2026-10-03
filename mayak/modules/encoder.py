"""Синоптический энкодер: строго причинный TCN с инкрементальным шагом.

Выход энкодера в данный час зависит только от входа за последние часы в пределах
рецептивного поля. Его длина - сумма дилатаций, умноженная на ядро без единицы, плюс
сам текущий час. Для ядра 3 и дилатаций от 1 до 32, каждая дважды, это 253 ч.

Режимы с одними и теми же весами:

* ``forward`` - пакетный проход по окну;
* ``step`` - потактовый шаг. Каждый блок держит кольцевой буфер своих последних
  входов на длину своего поля; стоимость шага зависит от глубины и ширины энкодера,
  но не от длины рецептивного поля;
* ``step_shift`` - тот же шаг без внутреннего состояния: буферы всех блоков приходят
  одним тензором в хронологическом порядке и возвращаются сдвинутыми на час. Кольцо и
  номер шага держит вызывающий.
"""
from dataclasses import dataclass, field

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
        if x.dim() == 2:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        B, C, L = x.shape
        y = F.group_norm(x.transpose(1, 2).reshape(B * L, C), self.num_groups,
                         self.weight, self.bias, self.eps)
        return y.reshape(B, L, C).transpose(1, 2)


class DSBlock(nn.Module):
    """Depthwise-separable причинный блок с residual-связью.

    Ядро с дилатацией видит текущий час и часы назад с шагом, равным дилатации. Слева
    окно дополняется нулями на длину поля блока - ядро без единицы, умноженное на
    дилатацию, - поэтому выход в данный час не зависит от будущего входа. Потактовый шаг
    держит кольцевой буфер последних входов блока той же длины.

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

    def step(self, x_t, buf, t):
        """Выход блока в час номер t по кольцевому буферу прошлых входов.

        Args:
            x_t: вход этого часа, форма (B, C).
            buf: кольцевой буфер прошлых входов блока, форма (B, C, длина поля);
                обновляется на месте.
            t: номер шага от начала потока.

        Returns:
            Выход блока, форма (B, C).
        """
        # столбец веса j отвечает входу на столько дилатаций назад, сколько столбцов
        # до конца ядра
        P, w = self.pad, self.dw.weight[:, 0, :]
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
                f"{prefix}gn: веса обучены с нормализацией по всей оси времени окна. "
                f"Эта нормализация не причинна и несовместима с потактовым шагом - "
                f"модель нужно переобучить.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@dataclass
class EncoderState:
    """Состояние потактового энкодера.

    Attributes:
        bufs: кольцевые буферы блоков.
        t: число сделанных шагов.
    """
    bufs: list = field(default_factory=list)
    t: int = 0

    @property
    def nbytes(self):
        return int(sum(b.numel() * b.element_size() for b in self.bufs))

    def clone(self):
        return EncoderState([b.clone() for b in self.bufs], self.t)


class SynopticEncoder(nn.Module):
    """Причинный TCN из depthwise-separable блоков.

    Пакетный проход переводит каналы формы (B, n_ch, L) в признаки формы
    (B, L, width). Потактовый режим - начальное состояние, шаг и предзаполнение окном.

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

    def forward_with_buffer(self, x):
        """Пакетный проход по отрезку и буфер потактового шага после его последнего часа.

        Буфер каждого блока - последние входы этого блока в хронологическом порядке;
        если отрезок короче буфера, слева стоят нули, как у холодного старта.

        Args:
            x: каналы, форма (B, n_ch, L).

        Returns:
            Пара: признаки формы (B, L, width) и буфер формы (B, width, сумма длин
            буферов блоков).
        """
        h = self.stem(x)
        parts = []
        for b in self.blocks:
            parts.append(F.pad(h, (b.pad, 0))[..., -b.pad:])
            h = b(h)
        return h.transpose(1, 2), torch.cat(parts, dim=-1)

    def init_state(self, batch_size=1):
        """Состояние до первого шага.

        Нулевые буферы дают то же, что дополнение окна нулями слева в пакетном проходе.

        Args:
            batch_size: размер батча.

        Returns:
            Состояние с нулевыми буферами и нулевым числом шагов.
        """
        p = self.stem.weight
        return EncoderState([p.new_zeros(batch_size, self.width, b.pad) for b in self.blocks], 0)

    @property
    def buffer_pads(self):
        """Длины буферов блоков, ч; их сумма - длина общего буфера шага без состояния."""
        return tuple(b.pad for b in self.blocks)

    def ring_to_shift(self, state):
        """Общий буфер шага без состояния из кольцевых буферов.

        Args:
            state: состояние потактового энкодера.

        Returns:
            Буфер формы (B, width, сумма длин буферов), старший час первым.
        """
        return torch.cat([buf[:, :, torch.arange(state.t - b.pad, state.t) % b.pad]
                          for buf, b in zip(state.bufs, self.blocks)], dim=-1)

    def step_shift(self, x_t, buf):
        """Один час без внутреннего состояния; результат тот же, что у шага по кольцам.

        Args:
            x_t: каналы этого часа, форма (B, n_ch).
            buf: общий буфер, форма (B, width, сумма длин буферов), старший час первым.

        Returns:
            Пара: признаки формы (B, width) и буфер, сдвинутый на час.
        """
        h = F.linear(x_t, self.stem.weight[:, :, 0], self.stem.bias)
        parts = []
        for b, bb in zip(self.blocks, buf.split(self.buffer_pads, dim=-1)):
            h, nb = b.step_shift(h, bb)
            parts.append(nb)
        return h, torch.cat(parts, dim=-1)

    def step(self, x_t, state):
        """Один час потока.

        Args:
            x_t: каналы этого часа, форма (B, n_ch).
            state: состояние энкодера; обновляется на месте.

        Returns:
            Признаки формы (B, width).
        """
        h = F.linear(x_t, self.stem.weight[:, :, 0], self.stem.bias)
        for b, buf in zip(self.blocks, state.bufs):
            h = b.step(h, buf, state.t)
        state.t += 1
        return h

    def prefill(self, x):
        """Пакетный проход по окну и состояние после его последнего часа.

        Результат тот же, что у начального состояния и шага на каждый час окна, но за
        один проход: буфер блока заполняется последними входами этого блока из
        пакетного прохода.

        Args:
            x: каналы окна, форма (B, n_ch, L).

        Returns:
            Пара: признаки формы (B, L, width) и состояние после последнего часа.
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
