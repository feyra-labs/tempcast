"""Датасет обучающих окон с куррикулумом холодного старта и аугментациями."""
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from mayak.config import ZONE_WEIGHTINGS, AugmentConfig
from mayak.constants import L_MAX, H
from mayak.data.augment import AugWindow, augment_window
from mayak.data.masking import (DEFAULT_TARGET_MASK, FilterStats, enforce_invariant,
                                target_window_ok)
from mayak.data.qc import qc_window
from mayak.data.splits import ROLE_TRAIN, ROLE_VAL, time_bounds
from mayak.data.store import get_store
from mayak.timeaxis import window_calendar
from mayak.zones import normalize_zone

log = logging.getLogger(__name__)

STREAM_SAMPLE, STREAM_AUG = 0, 1


def make_streams(base_seed, worker_id=0, salt=0, aug_seed=None):
    """Два независимых генератора от тройки (сид, воркер, поток).

    salt     — база итератора DataLoader: при непостоянных воркерах она своя на каждую
               эпоху, иначе каждая эпоха повторяла бы предыдущую окно в окно;
    aug_seed — отдельный сид аугментаций (None - тот же, что у потока окон). Поток
               окон зависит только от base_seed: смена aug_seed его не сдвигает.
    """
    def children(seed):
        ss = np.random.SeedSequence(entropy=[int(seed), int(salt)], spawn_key=(int(worker_id),))
        return ss.spawn(2)
    sample, aug = children(base_seed)
    if aug_seed is not None:
        aug = children(aug_seed)[1]
    return np.random.default_rng(sample), np.random.default_rng(aug)


def seed_worker(worker_id):
    info = get_worker_info()
    ds = info.dataset
    if hasattr(ds, "seed_streams"):
        ds.seed_streams(worker_id=info.id, salt=int(info.seed) - int(info.id))


def slice_history(x, mask, t, L):
    """История длины L, выровненная по правому краю буфера L_MAX."""
    x_hist = np.zeros((L_MAX, 3), np.float32)
    mask_hist = np.zeros((L_MAX, 3), np.float32)
    if L > 0:
        src = np.arange(t - L, t)
        x_hist[L_MAX - L:], mask_hist[L_MAX - L:] = enforce_invariant(x[src], mask[src])
    return x_hist, mask_hist


def slice_target(x, mask, t):
    """Цель — температура на [t, t + H) вместе с её маской.

    Маска цели берётся только из данных: аугментации её не трогают.
    """
    fut = np.arange(t, t + H)
    return enforce_invariant(x[fut, 0], mask[fut, 0])


def norm_scale(clim, doy_f, hour_f):
    """Нормировочный масштаб функции потерь на часах горизонта, °C, float32 (H,)."""
    return clim.scale(doy_f, hour_f).astype(np.float32)


def valid_starts(mask_T, lo, hi, step=1, cfg=DEFAULT_TARGET_MASK, history=0):
    """Кандидаты и годные старты t окна [lo, hi)."""
    cand = np.arange(lo + history, hi - H + 1, step, dtype=np.int64)
    return cand, cand[target_window_ok(mask_T, cand, H, cfg)]


def history_len(L, t, lo):
    """Фактическая длина истории: не больше L_MAX и не раньше начала окна сплита."""
    return int(max(0, min(L_MAX if L is None else L, L_MAX, t - lo)))


def zone_weights(zones, mode="inv_sqrt", cap=0.0):
    """Вероятности выбора станций по их полным зонам Кёппена.

    Вес станции n_зоны ** (−a), a = ZONE_WEIGHTINGS[mode]; при cap > 0 вес станции
    ограничен сверху cap · средний вес (итеративно, до сходимости). Сумма = 1.
    """
    if mode not in ZONE_WEIGHTINGS:
        raise ValueError(f"zone_weighting {mode!r}; допустимо {sorted(ZONE_WEIGHTINGS)}")
    zones = [normalize_zone(z) for z in zones]
    count = {}
    for z in zones:
        count[z] = count.get(z, 0) + 1
    w = np.array([count[z] ** -ZONE_WEIGHTINGS[mode] for z in zones], np.float64)
    if cap and cap > 0 and len(w):
        for _ in range(100):
            lim = cap * w.mean()
            if (w <= lim * (1 + 1e-12)).all():
                break
            w = np.minimum(w, lim)
    return w / w.sum()


def zone_distribution(zones, p):
    """{зона: (число станций, суммарная вероятность)} по убыванию вероятности."""
    out = {}
    for z, pi in zip((normalize_zone(z) for z in zones), p):
        n, s = out.get(z, (0, 0.0))
        out[z] = (n + 1, s + float(pi))
    return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))


def footprint(t, L, lo):
    """Часы, которые читает сэмпл: [t − фактическая история, t + H)."""
    return t - history_len(L, t, lo), t + H


class WindowDataset(Dataset):
    def __init__(self, manifest, split="train", curriculum="full",
                 windows_per_epoch=200_000, seed=0, target_mask=DEFAULT_TARGET_MASK,
                 store=None, aug_seed=None, augment=None, cache_root=None, window_qc=True,
                 zone_weighting="inv_sqrt", zone_weight_cap=0.0):
        assert split in ("train",)
        assert curriculum in ("full", "L0")
        self.curriculum = curriculum
        self.n = windows_per_epoch
        self.base_seed = int(seed)
        self.aug_seed = None if aug_seed is None else int(aug_seed)
        self.augment = augment if isinstance(augment, AugmentConfig) else \
            AugmentConfig.from_dict(augment)
        log.info("аугментации: профиль %s, отклонения от профиля %s",
                 self.augment.profile, self.augment.deviations() or "нет")
        self.window_qc = bool(window_qc)
        self.seed_streams(worker_id=0, salt=0)

        store = store or get_store(manifest, cache_root=cache_root)
        self.station_role, self.time_key = ROLE_TRAIN, "train"
        rows = store.by_role(ROLE_TRAIN)
        assert rows, "нет train-станций — запустите make_splits.py"

        self.st = []
        self.filter_stats = FilterStats()
        for r in rows:
            x, mask, N = r["x"], r["mask"], r["N"]
            lo, hi = time_bounds(N)[self.time_key]
            cand, ok = valid_starts(mask[:, 0], lo, hi, cfg=target_mask)
            self.filter_stats.add(len(cand), len(ok))
            if len(ok) == 0:
                continue
            self.st.append(dict(
                id=r["id"], lat=float(r["lat"]), lon=float(r["lon"]), elev=float(r["elev"]),
                koppen=r["koppen"], x=x, mask=mask, N=N, t0=r["t0"], clim=r["clim"],
                qc_elev=r.get("dem_elev") if r.get("dem_elev") is not None else float(r["elev"]),
                tr=(lo, hi), starts=ok))
        self.filter_stats.report("train")
        assert self.st, "ни у одной train-станции нет окон, прошедших маску цели"

        zones = [s["koppen"] for s in self.st]
        self.w = zone_weights(zones, zone_weighting, zone_weight_cap)
        self.zone_report = zone_distribution(zones, self.w)
        log.info("сэмплирование станций: взвешивание %s (cap %s), %d станций в %d зонах; "
                 "доли зон: %s", zone_weighting, zone_weight_cap or "нет", len(zones),
                 len(self.zone_report),
                 ", ".join(f"{z}:{n}ст/{p:.1%}" for z, (n, p) in self.zone_report.items()))

    def seed_streams(self, worker_id=0, salt=0):
        self.rng_sample, self.rng_aug = make_streams(self.base_seed, worker_id, salt,
                                                     aug_seed=self.aug_seed)

    def footprints(self):
        for s in self.st:
            lo = s["tr"][0]
            t = s["starts"]
            yield dict(sid=s["id"], N=s["N"], time_key=self.time_key,
                       lo=t - np.minimum(L_MAX, t - lo), hi=t + H)

    def __len__(self):
        return self.n

    def _sample_L(self):
        r = self.rng_sample
        if self.curriculum == "L0":
            return 0
        u = r.random()
        if u < 0.05:
            return 0
        if u < 0.20:
            return int(r.integers(1, 49))
        if u < 0.45:
            return int(r.integers(2 * 24, 10 * 24 + 1))
        return int(r.integers(10 * 24, L_MAX + 1))

    def __getitem__(self, _idx):
        r = self.rng_sample
        si = int(r.choice(len(self.st), p=self.w))
        s = self.st[si]
        lo, _hi = s["tr"]
        t = int(s["starts"][r.integers(len(s["starts"]))])

        L = history_len(self._sample_L(), t, lo)
        return self.build(s, t, L)

    def build(self, s, t, L, info=None):
        """Окно станции s с началом горизонта t и историей L, с аугментациями.

        info - если передан dict, в него кладутся параметры применённых аугментаций.
        """
        k = np.arange(L_MAX)
        abs_h = t - L_MAX + k
        doy_h, hour_h = window_calendar(s["t0"], abs_h)
        x_hist, mask_hist = slice_history(s["x"], s["mask"], t, L)

        fut = np.arange(t, t + H)
        doy_f, hour_f = window_calendar(s["t0"], fut)
        y, y_mask = slice_target(s["x"], s["mask"], t)
        scale = norm_scale(s["clim"], doy_f, hour_f)

        w = augment_window(AugWindow(x=x_hist, m=mask_hist, y=y, y_mask=y_mask, L=L,
                                     hour=hour_h, lat=s["lat"], lon=s["lon"],
                                     elev=s["elev"], qc_elev=s["qc_elev"]),
                           self.augment, self.rng_aug)
        if info is not None:
            info.update(w.applied)
        lat, lon, elev, y = w.lat, w.lon, w.elev, w.y
        x_hist, mask_hist = enforce_invariant(w.x, w.m)
        if self.window_qc and L > 0:
            mask_hist, _ = qc_window(x_hist, mask_hist, elev=w.qc_elev)
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
            "norm_scale": torch.from_numpy(scale),
        }


class HoldoutDataset(Dataset):
    """Детерминированные окна для валидации/оценки: фиксированный L, без аугментаций"""

    def __init__(self, manifest, station_split=ROLE_VAL, time_key="val",
                 every_hours=72, L=L_MAX, max_windows=8000,
                 target_mask=DEFAULT_TARGET_MASK, store=None):
        store = store or get_store(manifest)
        self.station_role, self.time_key = station_split, time_key
        self.meta = []
        self.filter_stats = FilterStats()
        for r in store.by_role(station_split):
            x, mask, N = r["x"], r["mask"], r["N"]
            lo, hi = time_bounds(N)[time_key]
            cand, ok = valid_starts(mask[:, 0], lo, hi, every_hours, cfg=target_mask,
                                    history=L_MAX)
            self.filter_stats.add(len(cand), len(ok))
            for t in ok.tolist():
                self.meta.append(dict(id=r["id"], N=N, lo=lo, x=x, mask=mask, t=t, t0=r["t0"],
                                      clim=r["clim"],
                                      lat=float(r["lat"]), lon=float(r["lon"]),
                                      elev=float(r["elev"]), koppen=r["koppen"],
                                      split=station_split, L=L))
        self.filter_stats.report(f"{station_split}/{time_key}")
        if len(self.meta) > max_windows:
            step = len(self.meta) // max_windows
            self.meta = self.meta[::step][:max_windows]

    def __len__(self):
        return len(self.meta)

    def footprints(self):
        for m in self.meta:
            lo, hi = footprint(m["t"], m["L"], m["lo"])
            yield dict(sid=m["id"], N=m["N"], time_key=self.time_key,
                       lo=np.array([lo]), hi=np.array([hi]))

    def __getitem__(self, i):
        m = self.meta[i]
        t = m["t"]
        L = history_len(m["L"], t, m["lo"])
        k = np.arange(L_MAX)
        abs_h = t - L_MAX + k
        doy_h, hour_h = window_calendar(m["t0"], abs_h)
        x_hist, mask_hist = slice_history(m["x"], m["mask"], t, L)
        fut = np.arange(t, t + H)
        doy_f, hour_f = window_calendar(m["t0"], fut)
        y, y_mask = slice_target(m["x"], m["mask"], t)
        return {
            "lat": torch.tensor(m["lat"], dtype=torch.float32),
            "lon": torch.tensor(m["lon"], dtype=torch.float32),
            "elev": torch.tensor(m["elev"], dtype=torch.float32),
            "x_hist": torch.from_numpy(x_hist), "mask_hist": torch.from_numpy(mask_hist),
            "doy_hist": torch.from_numpy(doy_h), "hour_hist": torch.from_numpy(hour_h),
            "doy_fut": torch.from_numpy(doy_f), "hour_fut": torch.from_numpy(hour_f),
            "y": torch.from_numpy(y), "y_mask": torch.from_numpy(y_mask),
            "norm_scale": torch.from_numpy(norm_scale(m["clim"], doy_f, hour_f)),
        }
