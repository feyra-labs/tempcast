"""Окна оценки: выбор чекпойнта, калибровка, внутренний и внешний тест."""
import copy
import hashlib
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

from mayak import baselines as BL
from mayak.constants import H, HISTORY_BINS, L_MAX
from mayak.data.dataset import (HISTORY_MIX, block_starts, build_window, footprint, history_len,
                                sample_history_len, slice_context, slice_history,
                                station_qc_elev)
from mayak.data.masking import DEFAULT_TARGET_MASK, FilterStats
from mayak.data.splits import ROLE_TEST, time_layout
from mayak.data.store import read_manifest
from mayak.timeaxis import window_month
from mayak.zones import SEASON_RU, normalize_zone, season_of

HISTORY_GRID = (0, 6, 24, 72, 168, 336, L_MAX)
NOMINAL_HISTORY = L_MAX
PRESSURE_YES, PRESSURE_NO = "есть давление", "нет давления"


def history_label(L):
    """Метка окна, у которого длина истории задана точно."""
    return f"L={int(L)}ч"


def history_bin_label(L):
    """Метка бина длины истории для окна со случайной длиной истории."""
    for lo, hi, name in HISTORY_BINS:
        if lo <= int(L) <= hi:
            return name
    return "прочее"


def history_strata(meta):
    """Метки длины истории окон и порядок этих меток по возрастанию длины."""
    hist = np.asarray(meta["history"], np.int64)
    labels = meta.get("history_label")
    if labels is None:
        labels = [history_bin_label(v) for v in hist.tolist()]
    labels = np.asarray(labels, object).astype(str)
    order = sorted(set(labels.tolist()), key=lambda k: int(hist[labels == k].min()))
    return labels, order


def check_history_grid(grid):
    """Проверяет сетку длин истории стенда оценки."""
    out = tuple(sorted({int(v) for v in grid}))
    if not out:
        raise ValueError("сетка длин истории пуста")
    if out[0] < 0 or out[-1] > L_MAX:
        raise ValueError(f"длины истории {out} вне [0, {L_MAX}]")
    if NOMINAL_HISTORY not in out:
        raise ValueError(f"в сетке {out} нет полной истории {NOMINAL_HISTORY} ч: при ней "
                         f"считаются основные таблицы")
    return out


def stratified_items(per_station, max_windows=None, windows_per_station=None):
    """Стратифицированная подвыборка окон: одинаковое число окон с каждой станции."""
    stations = [sid for sid, ts in per_station.items() if len(ts)]
    if not stations:
        return []
    k = windows_per_station
    if k is None and max_windows:
        k = max(1, int(max_windows) // len(stations))
    out = []
    for sid in stations:
        ts = list(per_station[sid])
        if k is not None and len(ts) > k:
            idx = np.unique(np.linspace(0, len(ts) - 1, k).round().astype(np.int64))
            ts = [ts[i] for i in idx]
        out += [(sid, int(t)) for t in ts]
    return out


def window_history_lengths(items, curriculum, seed):
    """Запрошенная длина истории каждого окна по распределению куррикулума."""
    if curriculum not in HISTORY_MIX:
        raise ValueError(f"неизвестный куррикулум {curriculum!r}; есть {tuple(HISTORY_MIX)}")
    if int(seed) < 0:
        raise ValueError(f"сид набора окон {seed} отрицательный")
    out = np.empty(len(items), np.int64)
    for i, (sid, t) in enumerate(items):
        rng = np.random.default_rng([int(seed), zlib.crc32(str(sid).encode()), int(t)])
        out[i] = sample_history_len(rng, curriculum)
    return out


class EvalSet(Dataset):
    """Окна оценки."""

    def __init__(self, clims, station_splits=(ROLE_TEST,),
                 manifest="data/manifest.csv", time_key="test",
                 every_hours=72, L=None, max_windows=6000, windows_per_station=None,
                 target_mask=DEFAULT_TARGET_MASK, curriculum=None, history_seed=0,
                 train_km=None):
        if L is not None and curriculum is not None:
            raise ValueError("длина истории задаётся либо одним числом, либо куррикулумом")
        split_of = {r["id"]: r.get("split") for r in read_manifest(manifest)}
        self.station_splits, self.time_key = tuple(station_splits), time_key
        self.floor, self.roles = {}, {}
        self.filter_stats = FilterStats()
        per_station = {}
        for sid, s in clims.items():
            sp = split_of.get(sid)
            if sp not in station_splits:
                continue
            layout = time_layout(s["N"])
            self.floor[sid] = layout.history_floor(time_key)
            cand, ok = block_starts(layout, time_key, s["mask"][:, 0], every_hours,
                                    cfg=target_mask)
            self.filter_stats.add(len(cand), len(ok))
            self.roles[sid] = sp
            per_station[sid] = ok.tolist()
        self.filter_stats.report(f"eval/{time_key}")
        self.items = stratified_items(per_station, max_windows, windows_per_station)
        self.clims = clims
        self.L = L
        self.curriculum = curriculum
        self.history_seed = int(history_seed)
        self.train_km = dict(train_km) if train_km else None
        if curriculum is None:
            self.requested = [L] * len(self.items)
        else:
            self.requested = window_history_lengths(self.items, curriculum,
                                                    self.history_seed).tolist()

    def __len__(self):
        return len(self.items)

    def with_history(self, L):
        """Те же окна с одной длиной истории для всех окон."""
        out = copy.copy(self)
        out.L, out.curriculum = int(L), None
        out.requested = [int(L)] * len(self.items)
        out.__dict__.pop("_robustness_meta", None)
        return out

    def hourly(self, n_stations, hours):
        """Те же станции и временное окно, но непрерывный ежечасный выпуск, как у прибора."""
        sids = sorted(self.roles)
        if not sids:
            raise ValueError("ежечасный прогон: в наборе нет станций")
        n = min(int(n_stations), len(sids))
        pick = [sids[i] for i in np.unique(np.linspace(0, len(sids) - 1, n).round()
                                           .astype(np.int64))]
        items = []
        for i, sid in enumerate(pick):
            blocks = time_layout(self.clims[sid]["N"]).blocks[self.time_key]
            lo, hi = max(blocks, key=lambda b: b[1] - b[0])
            n_start = hi - lo - H + 1
            if n_start < 1:
                continue
            k = min(int(hours), n_start)
            off = lo + ((n_start - k) * i) // max(1, len(pick) - 1)
            items += [(sid, int(t)) for t in range(off, off + k)]
        if not items:
            raise ValueError("ежечасный прогон: ни у одной станции блок не вмещает горизонт")
        out = copy.copy(self)
        out.items, out.L, out.curriculum = items, None, None
        out.requested = [None] * len(items)
        out.__dict__.pop("_robustness_meta", None)
        out.__dict__.pop("_attrs", None)
        return out

    def history_length(self, i):
        """Фактическая длина истории окна с учётом самого раннего доступного часа."""
        sid, t = self.items[i]
        return history_len(self.requested[i], t, self.floor[sid])

    def history_spec(self):
        """Правило длины истории набора."""
        return dict(curriculum=self.curriculum, L=self.L,
                    seed=None if self.curriculum is None else self.history_seed)

    def fingerprint(self):
        """Отпечаток набора: станции, начала горизонта и запрошенные длины истории."""
        h = hashlib.sha256()
        for (sid, t), n in zip(self.items, self.requested):
            h.update(f"{sid}\t{t}\t{n}\n".encode())
        return h.hexdigest()[:16]

    def footprints(self):
        for i, (sid, t) in enumerate(self.items):
            lo, hi = footprint(t, self.requested[i], self.floor[sid])
            yield dict(sid=sid, N=self.clims[sid]["N"], time_key=self.time_key,
                       lo=np.array([lo]), t=np.array([t]), hi=np.array([hi]))

    def station_attrs(self):
        if getattr(self, "_attrs", None) is None:
            from mayak.external import station_attributes
            sids = {sid for sid, _t in self.items}
            self._attrs = station_attributes({sid: self.clims[sid] for sid in sids},
                                             train_km=self.train_km)
        return self._attrs

    def window_meta(self):
        """Метки окон для разрезов."""
        label = history_bin_label if self.curriculum is not None else history_label
        sid_a, role, zone, season, hist, hvalid = [], [], [], [], [], []
        has_p, dist, egap = [], [], []
        attrs = self.station_attrs()
        for i, (sid, t) in enumerate(self.items):
            s = self.clims[sid]
            L = self.history_length(i)
            m = s["mask"][t - L:t, 0] if L > 0 else np.zeros(0, np.float32)
            month = int(np.asarray(window_month(s["t0"], [t])).ravel()[0])
            sid_a.append(sid)
            role.append(self.roles[sid])
            zone.append(normalize_zone(s["koppen"]))
            season.append(SEASON_RU[season_of(month, s["lat"])])
            hist.append(L)
            hvalid.append(float((m > 0).mean()) if L > 0 else 0.0)
            mp = s["mask"][t - L:t, 1] if L > 0 else np.zeros(0, np.float32)
            has_p.append(PRESSURE_YES if (mp > 0).any() else PRESSURE_NO)
            dist.append(attrs[sid]["train_distance_label"])
            egap.append(attrs[sid]["elev_gap_label"])
        return dict(station=np.array(sid_a, object), role=np.array(role, object),
                    zone=np.array(zone, object), season=np.array(season, object),
                    history=np.array(hist, np.int64),
                    history_label=np.array([label(v) for v in hist], object),
                    hist_valid=np.array(hvalid, np.float64),
                    has_pressure=np.array(has_p, object), elev_gap=np.array(egap, object),
                    train_distance=np.array(dist, object),
                    t=np.array([t for _sid, t in self.items], np.int64))

    def raw_window(self, i):
        """Сырая история окна до записи и QC."""
        sid, t = self.items[i]
        s = self.clims[sid]
        L = self.history_length(i)
        x, m = slice_history(s["raw"], s["present"], t, L)
        past = slice_context(s["raw"], s["present"], t, L, self.floor[sid])
        return dict(x=x, m=m, past=past, L=L, qc_elev=station_qc_elev(s))

    def __getitem__(self, i):
        """Окно оценки: окно ``build_window`` без аугментаций и поля эталонов."""
        sid, t = self.items[i]
        s = self.clims[sid]
        clim = s["clim"]
        L = self.history_length(i)
        out = build_window(s, t, L, self.floor[sid])
        x_hist, mask_hist = out["x_hist"].numpy(), out["mask_hist"].numpy()
        mu_clim_fut = clim.predict(out["doy_fut"].numpy(), out["hour_fut"].numpy())
        a_recent, a_ok = BL.recent_anomaly(x_hist[:, 0], mask_hist[:, 0], clim, L_MAX,
                                           int(s["t0"]) + int(t) - L_MAX)
        out.update(
            mu_clim_fut=torch.from_numpy(np.asarray(mu_clim_fut, np.float32)),
            a_recent=torch.tensor(a_recent, dtype=torch.float32),
            a_recent_ok=torch.tensor(bool(a_ok)),
            hist_len=torch.tensor(L, dtype=torch.int64),
        )
        return out
