"""Запись значений так, как их пишет прибор.

Температура записывается целыми градусами Цельсия, влажность целыми процентами,
давление десятыми долями гектопаскаля. Округление к ближайшему значению сетки, ровно
половина округляется к чётному. Правило одно для кэша, обучающих окон, оценки и потока
устройства; совпадение с портом на Rust закреплено общим файлом эталонных случаев.
"""
from __future__ import annotations

import numpy as np

RECORD_SCALE = (1.0, 10.0, 1.0)
T, P, RH = 0, 1, 2


def round_half_even(v):
    """Округление к ближайшему целому, ровно половина к чётному.

    Args:
        v: число или массив.

    Returns:
        Массив float64 той же формы. Нечисловые значения остаются нечисловыми,
        отрицательный ноль становится обычным нулём.
    """
    return np.round(np.asarray(v, np.float64)) + 0.0


def record_channel(v, ch):
    """Значения одного канала на сетке записи прибора.

    Args:
        v: значения канала, любая форма.
        ch: номер канала: 0 температура, 1 давление, 2 влажность.

    Returns:
        Массив float32 той же формы.
    """
    s = RECORD_SCALE[ch]
    return (round_half_even(np.asarray(v, np.float64) * s) / s).astype(np.float32)


def record_values(x):
    """Значения T, P, RH на сетке записи прибора.

    Args:
        x: значения, форма (..., 3).

    Returns:
        Массив float32 той же формы.
    """
    x = np.asarray(x, np.float64)
    s = np.asarray(RECORD_SCALE, np.float64)
    return (round_half_even(x * s) / s).astype(np.float32)


def is_recorded(x, mask=None):
    """Лежат ли значения на сетке записи прибора.

    Args:
        x: значения, форма (..., 3).
        mask: где значение есть, форма (..., 3); None - везде.

    Returns:
        True, если каждое имеющееся значение совпадает со своей записью.
    """
    x = np.asarray(x, np.float32)
    ok = np.ones(x.shape, bool) if mask is None else np.asarray(mask) > 0
    ok &= np.isfinite(x)
    return bool(np.array_equal(record_values(x)[ok], x[ok]))


__all__ = ["RECORD_SCALE", "is_recorded", "record_channel", "record_values",
           "round_half_even"]
