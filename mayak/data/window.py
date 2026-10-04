"""Окно истории и календарь выпуска - общее построение входов модели.

Окно оценки и выпуск устройства строят входы модели этими функциями: история
выравнивается по правому краю буфера наибольшей длины, слева нули, календарь истории и
горизонта считается от одного и того же абсолютного часа.
"""
from __future__ import annotations

import numpy as np

from mayak.data.masking import enforce_invariant
from mayak.timeaxis import window_calendar


def place_history(x, mask, history):
    """История, выровненная по правому краю буфера длины history.

    Args:
        x: значения часов истории, форма (n, 3), от старых к новым; n не больше history.
        mask: маски тех же часов, форма (n, 3).
        history: длина буфера, ч.

    Returns:
        Пара массивов float32 формы (history, 3): значения с нулями на месте
        невалидных часов и маски; слева нули.

    Raises:
        ValueError: часов истории больше длины буфера.
    """
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
    """Календарь буфера истории и горизонта выпуска.

    Args:
        t0: абсолютный час UTC, от которого отсчитывается t.
        t: первый час горизонта относительно t0; история кончается часом раньше.
        history: длина буфера истории, ч.
        horizon: длина горизонта, ч.

    Returns:
        Четыре массива float32: день года и час UTC часов истории, формы (history,), и
        часов горизонта, формы (horizon,).
    """
    t = int(t)
    doy_h, hour_h = window_calendar(t0, t - history + np.arange(history, dtype=np.int64))
    doy_f, hour_f = window_calendar(t0, t + np.arange(horizon, dtype=np.int64))
    return doy_h, hour_h, doy_f, hour_f


__all__ = ["issue_calendar", "place_history"]
