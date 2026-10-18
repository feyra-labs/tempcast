"""Запись значений так, как их пишет прибор."""
from __future__ import annotations

import numpy as np

RECORD_SCALE = (1.0, 10.0, 1.0)
T, P, RH = 0, 1, 2


def round_half_even(v):
    """Округление к ближайшему целому, ровно половина к чётному."""
    return np.round(np.asarray(v, np.float64)) + 0.0


def record_channel(v, ch):
    """Значения одного канала на сетке записи прибора."""
    s = RECORD_SCALE[ch]
    return (round_half_even(np.asarray(v, np.float64) * s) / s).astype(np.float32)


def record_values(x):
    """Значения T, P, RH на сетке записи прибора."""
    x = np.asarray(x, np.float64)
    s = np.asarray(RECORD_SCALE, np.float64)
    return (round_half_even(x * s) / s).astype(np.float32)


__all__ = ["RECORD_SCALE", "record_channel", "record_values", "round_half_even"]
