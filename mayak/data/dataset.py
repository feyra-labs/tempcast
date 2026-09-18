"""Датасет обучающих окон с куррикулумом холодного старта и аугментациями"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from mayak.constants import L_MAX, H
from mayak.data.masking import (DEFAULT_TARGET_MASK, FilterStats, enforce_invariant,
                                target_window_ok)
from mayak.data.qc import run_qc
from mayak.data.splits import time_bounds


def _calendar(t0_doy, t0_hour, abs_hours):
    doy = (t0_doy + (t0_hour + abs_hours) / 24.0) % 365.24
    hour = (t0_hour + abs_hours) % 24.0
    return doy.astype(np.float32), hour.astype(np.float32)


def slice_history(x, mask, t, L):
    """История длины L, выровненная по правому краю буфера L_MAX. Инвариант соблюдён."""
    x_hist = np.zeros((L_MAX, 3), np.float32)
    mask_hist = np.zeros((L_MAX, 3), np.float32)
    if L > 0:
        src = np.arange(t - L, t)
        x_hist[L_MAX - L:], mask_hist[L_MAX - L:] = enforce_invariant(x[src], mask[src])
    return x_hist, mask_hist


def slice_target(x, mask, t):
    """Цель — температура на [t, t + H) вместе с её маской (1.2).

    Маска цели берётся только из данных (1.6): аугментации её не трогают.
    """
    fut = np.arange(t, t + H)
    return enforce_invariant(x[fut, 0], mask[fut, 0])


def valid_starts(mask_T, lo, hi, step=1, cfg=DEFAULT_TARGET_MASK):
    """Старты t ∈ [lo + 1, hi − H) с шагом step, прошедшие правило 1.5."""
    cand = np.arange(lo + 1, hi - H, step, dtype=np.int64)
    return cand, cand[target_window_ok(mask_T, cand, H, cfg)]


class WindowDataset(Dataset):
    def __init__(self, manifest, split="train", curriculum="full",
                 windows_per_epoch=200_000, seed=0, target_mask=DEFAULT_TARGET_MASK):
        assert split in ("train",)
        assert curriculum in ("full", "L0")
        self.curriculum = curriculum
        self.n = windows_per_epoch
        self.rng = np.random.default_rng(seed)

        root = os.path.dirname(manifest)
        with open(manifest) as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
        assert rows, "нет train-станций — запустите make_splits.py"

        self.st = []
        zone_count = {}
        self.filter_stats = FilterStats()
        for r in rows:
            d = np.load(os.path.join(root, "stations", f"{r['id']}.npz"))
            x, mask = run_qc(d["T"], d["P"], d["RH"], d["valid"])
            N = x.shape[0]
            lo, hi = time_bounds(N)["train"]
            cand, ok = valid_starts(mask[:, 0], lo, hi, cfg=target_mask)
            self.filter_stats.add(len(cand), len(ok))
            if len(ok) == 0:
                continue
            self.st.append(dict(
                lat=float(r["lat"]), lon=float(r["lon"]), elev=float(r["elev"]),
                koppen=r["koppen"], x=x, mask=mask, N=N,
                t0_doy=float(d["t0_doy"]), t0_hour=float(d["t0_hour"]),
                tr=(lo, hi), starts=ok))
            zone_count[r["koppen"]] = zone_count.get(r["koppen"], 0) + 1
        self.filter_stats.report("train")
        assert self.st, "ни у одной train-станции нет окон, прошедших маску цели"

        w = np.array([1.0 / zone_count[s["koppen"]] for s in self.st])
        self.w = w / w.sum()

    def __len__(self):
        return self.n

    def _sample_L(self):
        if self.curriculum == "L0":
            return 0
        u = self.rng.random()
        if u < 0.05:
            return 0
        if u < 0.20:
            return int(self.rng.integers(1, 49))
        if u < 0.45:
            return int(self.rng.integers(2 * 24, 10 * 24 + 1))
        return int(self.rng.integers(10 * 24, L_MAX + 1))

    def __getitem__(self, _idx):
        si = int(self.rng.choice(len(self.st), p=self.w))
        s = self.st[si]
        lo, _hi = s["tr"]
        t = int(s["starts"][self.rng.integers(len(s["starts"]))])

        L = self._sample_L()
        L = min(L, t - lo)
        return self.build(s, t, L)

    def build(self, s, t, L):
        """Окно станции s с началом горизонта t и историей L, с аугментациями.

        Вынесено отдельно, чтобы тесты могли собрать окно в заданной точке.
        """
        k = np.arange(L_MAX)
        abs_h = t - L_MAX + k
        doy_h, hour_h = _calendar(s["t0_doy"], s["t0_hour"], abs_h)
        x_hist, mask_hist = slice_history(s["x"], s["mask"], t, L)

        fut = np.arange(t, t + H)
        doy_f, hour_f = _calendar(s["t0_doy"], s["t0_hour"], fut)
        y, y_mask = slice_target(s["x"], s["mask"], t)

        lat, lon, elev = s["lat"], s["lon"], s["elev"]

        r = self.rng
        if L > 0 and r.random() < 0.3:
            glen = int(r.integers(1, 25))
            gst = int(r.integers(L_MAX - L, L_MAX))
            mask_hist[gst:min(L_MAX, gst + glen)] = 0.0

        if r.random() < 0.1:
            mask_hist[:, 2] = 0.0
        if r.random() < 0.1:
            mask_hist[:, 1] = 0.0

        noise = np.stack([r.normal(0, 0.2, L_MAX),
                          r.normal(0, 0.5, L_MAX),
                          r.normal(0, 2.0, L_MAX)], axis=-1).astype(np.float32)
        x_hist = x_hist + noise * mask_hist

        if L > 0:
            # постоянное смещение прибора — единственная аугментация, которая
            # касается цели (1.6, 9.5); маску цели она не меняет
            off = float(r.uniform(-0.7, 0.7))
            x_hist[:, 0] = x_hist[:, 0] + off * mask_hist[:, 0]
            y = y + off * y_mask
            lat = lat + float(r.uniform(-0.40, 0.40))
            lon = lon + float(r.uniform(-0.40, 0.40))

        x_hist[:, 2] = np.clip(x_hist[:, 2], 0, 100)
        # 1.1 / 1.8: единственная точка восстановления инварианта после аугментаций
        x_hist, mask_hist = enforce_invariant(x_hist, mask_hist)

        return {
            "lat": torch.tensor(lat, dtype=torch.float32),
            "lon": torch.tensor(lon, dtype=torch.float32),
            "elev": torch.tensor(elev, dtype=torch.float32),
            "x_hist": torch.from_numpy(x_hist),
            "mask_hist": torch.from_numpy(mask_hist),
            "doy_hist": torch.from_numpy(doy_h),
            "hour_hist": torch.from_numpy(hour_h),
            "doy_fut": torch.from_numpy(doy_f),
            "hour_fut": torch.from_numpy(hour_f),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
        }


class HoldoutDataset(Dataset):
    """Детерминированные окна для валидации/оценки: фиксированный L, без аугментаций"""

    def __init__(self, manifest, station_split="unseen_val", time_key="calib",
                 every_hours=72, L=L_MAX, max_windows=8000,
                 target_mask=DEFAULT_TARGET_MASK):
        root = os.path.dirname(manifest)
        with open(manifest) as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == station_split]
        self.meta = []
        self.filter_stats = FilterStats()
        for r in rows:
            d = np.load(os.path.join(root, "stations", f"{r['id']}.npz"))
            x, mask = run_qc(d["T"], d["P"], d["RH"], d["valid"])
            N = x.shape[0]
            lo, hi = time_bounds(N)[time_key]
            t0d, t0h = float(d["t0_doy"]), float(d["t0_hour"])
            cand, ok = valid_starts(mask[:, 0], lo, hi, every_hours, cfg=target_mask)
            self.filter_stats.add(len(cand), len(ok))
            for t in ok.tolist():
                self.meta.append(dict(x=x, mask=mask, t=t, t0d=t0d, t0h=t0h,
                                      lat=float(r["lat"]), lon=float(r["lon"]),
                                      elev=float(r["elev"]), koppen=r["koppen"],
                                      split=station_split, L=L))
        self.filter_stats.report(f"{station_split}/{time_key}")
        if len(self.meta) > max_windows:
            step = len(self.meta) // max_windows
            self.meta = self.meta[::step][:max_windows]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        t, L = m["t"], m["L"]
        L = min(L, t)
        k = np.arange(L_MAX)
        abs_h = t - L_MAX + k
        doy_h, hour_h = _calendar(m["t0d"], m["t0h"], abs_h)
        x_hist, mask_hist = slice_history(m["x"], m["mask"], t, L)
        fut = np.arange(t, t + H)
        doy_f, hour_f = _calendar(m["t0d"], m["t0h"], fut)
        y, y_mask = slice_target(m["x"], m["mask"], t)
        return {
            "lat": torch.tensor(m["lat"], dtype=torch.float32),
            "lon": torch.tensor(m["lon"], dtype=torch.float32),
            "elev": torch.tensor(m["elev"], dtype=torch.float32),
            "x_hist": torch.from_numpy(x_hist), "mask_hist": torch.from_numpy(mask_hist),
            "doy_hist": torch.from_numpy(doy_h), "hour_hist": torch.from_numpy(hour_h),
            "doy_fut": torch.from_numpy(doy_f), "hour_fut": torch.from_numpy(hour_f),
            "y": torch.from_numpy(y), "y_mask": torch.from_numpy(y_mask),
        }
