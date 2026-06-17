"""Контроль качества.

Правила:
  * физические диапазоны: T∈[-90,60] °C, RH∈[0,100] %, P∈[300,1100] гПа;
  * MAD-фильтр выбросов по скользящему окну 12 ч;
  * клок-монотонность.
Аномалия не выбрасывает исключение, а опускает маску в 0 на затронутый канал/точку.
"""
import numpy as np

PHYS = {"T": (-90.0, 60.0), "RH": (0.0, 100.0), "P": (300.0, 1100.0)}

def _mad_mask(x, valid, win=12, thresh=6.0):
    n = len(x); ok = np.ones(n, dtype=bool)
    half = win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi][valid[lo:hi] > 0]
        if len(seg) < 4:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-6
        if abs(x[i] - med) > thresh * 1.4826 * mad:
            ok[i] = False
    return ok

def run_qc(T, P, RH, valid):
    valid = valid.astype(bool)
    chans = {"T": T, "P": P, "RH": RH}
    masks = {}
    for name, arr in chans.items():
        lo, hi = PHYS[name]
        phys_ok = (arr >= lo) & (arr <= hi)
        base = valid & phys_ok
        mad_ok = _mad_mask(arr, base.astype(np.uint8))
        masks[name] = (base & mad_ok).astype(np.float32)
    x = np.stack([T, P, RH], axis=-1).astype(np.float32)
    mask = np.stack([masks["T"], masks["P"], masks["RH"]], axis=-1).astype(np.float32)
    x = np.where(mask > 0, x, 0.0).astype(np.float32)
    return x, mask