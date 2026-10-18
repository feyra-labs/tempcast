"""Окно истории и календарь выпуска - общее построение входов модели."""
from __future__ import annotations

import numpy as np

from mayak.data.masking import enforce_invariant
from mayak.timeaxis import window_calendar


def place_history(x, mask, history):
    """История, выровненная по правому краю буфера длины history."""
    x = np.asarray(x, np.float32).reshape(-1, 3)
    n = x.shape[0]
    if n > history:
        raise ValueError(f"история {n} ч длиннее буфера {history} ч")
    x_hist = np.zeros((history, 3), np.float32)
    mask_hist = np.zeros((history, 3), np.float32)
    if n:
        x_hist[history - n:], mask_hist[history - n:] = enforce_invariant(
            x, np.asarray(mask).reshape(-1, 3))
    return x_hist, mask_hist


def issue_calendar(t0, t, history, horizon):
    """Календарь буфера истории и горизонта выпуска."""
    t = int(t)
    doy_h, hour_h = window_calendar(t0, t - history + np.arange(history, dtype=np.int64))
    doy_f, hour_f = window_calendar(t0, t + np.arange(horizon, dtype=np.int64))
    return doy_h, hour_h, doy_f, hour_f


def valid_history_hours(mask_hist):
    """Число часов с валидной температурой во входе модели."""
    m = np.asarray(mask_hist)
    if m.ndim < 2 or m.shape[-1] != 3:
        raise ValueError(f"маска истории формы {m.shape}, нужно (..., L, 3)")
    n = (m[..., 0] > 0).sum(axis=-1, dtype=np.int64)
    return int(n) if n.ndim == 0 else n


__all__ = ["issue_calendar", "place_history", "valid_history_hours"]
