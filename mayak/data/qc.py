"""Контроль качества источника.
ВАЖНО: любое изменение правил требует поднять QC_VERSION — он входит в ключ кэша.
Тест test_qc_fingerprint_pinned ловит изменение правил без смены версии.
"""
from enum import IntFlag

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mayak.data.masking import enforce_invariant

QC_VERSION = "2.0"
CHANNELS = ("T", "P", "RH")
PHYS = {"T": (-90.0, 60.0), "RH": (0.0, 100.0), "P": (300.0, 1100.0)}
MAD_HALF, MAD_THRESH, MAD_MIN_VALID = 6, 6.0, 4


class QCCode(IntFlag):
    OK = 0
    MISSING = 1
    RANGE = 2
    SPIKE = 4


def _median_sorted(s, cnt):
    """Медиана по оси -1 для массива, отсортированного с NaN в конце; cnt — число не-NaN."""
    c = np.maximum(cnt, 1)
    lo = np.take_along_axis(s, ((c - 1) // 2)[:, None], -1)[:, 0]
    hi = np.take_along_axis(s, (c // 2)[:, None], -1)[:, 0]
    return (lo + hi) / 2


def mad_ok(x, valid, half=MAD_HALF, thresh=MAD_THRESH, min_valid=MAD_MIN_VALID, chunk=1 << 16):
    x = np.asarray(x)
    n = len(x)
    xv = np.where(np.asarray(valid) > 0, x, np.nan).astype(x.dtype, copy=False)
    xp = np.pad(xv, (half, half), constant_values=np.nan)
    ok = np.ones(n, dtype=bool)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        w = sliding_window_view(xp[a:b + 2 * half], 2 * half + 1)
        cnt = (~np.isnan(w)).sum(-1)
        s = np.sort(w, axis=-1)
        med = _median_sorted(s, cnt)
        dev = np.sort(np.abs(w - med[:, None]), axis=-1)
        mad = _median_sorted(dev, cnt) + 1e-6
        bad = np.abs(x[a:b] - med) > thresh * 1.4826 * mad
        ok[a:b] = ~((cnt >= min_valid) & bad)
    return ok


def _mad_ok_reference(x, valid, half=MAD_HALF, thresh=MAD_THRESH, min_valid=MAD_MIN_VALID):
    """Только для тестов. Медленная"""
    n = len(x)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi][valid[lo:hi] > 0]
        if len(seg) < min_valid:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-6
        if abs(x[i] - med) > thresh * 1.4826 * mad:
            ok[i] = False
    return ok


def _as_channel_valid(valid, n):
    """Маска источника (N,) или (N, 3) → (N, 3) bool."""
    v = np.asarray(valid)
    if v.ndim == 1:
        v = np.repeat(v[:, None], len(CHANNELS), axis=1)
    if v.shape != (n, len(CHANNELS)):
        raise ValueError(f"маска источника формы {v.shape}, ожидалось ({n},) или ({n}, 3)")
    return v > 0


def qc_station(T, P, RH, valid):
    """Полный QC станции → (x float32 (N,3), mask uint8 (N,3), codes uint8 (N,3))."""
    x = np.stack([T, P, RH], axis=-1).astype(np.float32)
    n = x.shape[0]
    src = _as_channel_valid(valid, n) & np.isfinite(x)
    codes = np.zeros((n, 3), np.uint8)
    codes[~src] |= np.uint8(QCCode.MISSING)
    for j, name in enumerate(CHANNELS):
        lo, hi = PHYS[name]
        with np.errstate(invalid="ignore"):
            phys = (x[:, j] >= lo) & (x[:, j] <= hi)
        codes[src[:, j] & ~phys, j] |= np.uint8(QCCode.RANGE)
        base = src[:, j] & phys
        codes[base & ~mad_ok(x[:, j], base), j] |= np.uint8(QCCode.SPIKE)
    mask = (codes == 0).astype(np.uint8)
    x, _ = enforce_invariant(x, mask)
    return x, mask, codes


def run_qc(T, P, RH, valid):
    x, mask, _ = qc_station(T, P, RH, valid)
    return x, mask.astype(np.float32)
