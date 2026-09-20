"""Исполняемый чек-лист антиутече.

Четыре проверки, вызываются перед обучением и перед оценкой:

1. климатология каждой станции подогнана только по её обучающему окну;
2. конформная таблица построена только по калибровочному окну и только
   на валидационных станциях;
3. ни одно окно датасета не выходит за границу своего временного окна
   ни историей, ни горизонтом (а временные окна разделены зазором ≥ H + L_MAX);
4. чекпойнт выбран по метрике на валидационных станциях в валидационном окне.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from mayak.data.splits import (MIN_GAP_HOURS, ROLE_TRAIN, ROLE_VAL, SPLITS_VERSION,
                               TIME_BOUNDS, TIME_KEYS, time_bounds)

log = logging.getLogger(__name__)

SELECTION_KEY = "mayak_selection"          # ключ записи о выборе чекпойнта в .ckpt
SELECTION_TIME_KEY, CONFORMAL_TIME_KEY = "val", "calib"
# какие роли станций допустимы в окнах датасета с данным временным ключом
ALLOWED_ROLES = {"train": {ROLE_TRAIN}, "calib": {ROLE_VAL}}


class LeakageError(RuntimeError):
    """Нарушение чек-листа антиутечек."""


def _fail(msg):
    raise LeakageError(msg)


def _split_state():
    return dict(splits_version=SPLITS_VERSION, time_bounds=dict(TIME_BOUNDS))


def _check_split_state(rec, what):
    if rec.get("splits_version") != SPLITS_VERSION or rec.get("time_bounds") != dict(TIME_BOUNDS):
        _fail(f"{what}: построено при других правилах сплитов "
              f"({rec.get('splits_version')}, {rec.get('time_bounds')}) — "
              f"сейчас ({SPLITS_VERSION}, {dict(TIME_BOUNDS)}); пересоберите")


def _check_station_roles(stations, store, role, what):
    bad = {sid: store.stations[sid]["role"] if sid in store.stations else None
           for sid in stations
           if sid not in store.stations or store.stations[sid]["role"] != role}
    if bad:
        _fail(f"{what}: станции не из роли {role}: {dict(list(bad.items())[:5])}")
    if not stations:
        _fail(f"{what}: пустой список станций")


def check_time_bounds(n_hours):
    """Окна идут по порядку, не пустые и разделены зазором ≥ H + L_MAX."""
    b = time_bounds(n_hours)
    if b["train"][0] < 0 or b["test"][1] > n_hours:
        _fail(f"окна выходят за ряд длины {n_hours}: {b}")
    for k in TIME_KEYS:
        if b[k][0] >= b[k][1]:
            _fail(f"пустое окно {k}: {b[k]}")
    for a, c in zip(TIME_KEYS, TIME_KEYS[1:]):
        gap = b[c][0] - b[a][1]
        if gap < MIN_GAP_HOURS:
            _fail(f"зазор {a}→{c} = {gap} ч < H + L_MAX = {MIN_GAP_HOURS} ч")
    return b


def check_climatology(store, deep=False):
    """Окно подгонки климатологии каждой станции совпадает с её обучающим окном.

    deep=True дополнительно переподгоняет климатологию по обучающему окну и сверяет
    коэффициенты: ловит кэш, в котором записанное окно не соответствует подгонке.
    """
    for sid, s in store.stations.items():
        want = tuple(time_bounds(s["N"])["train"])
        got = tuple(s.get("clim_fit") or ())
        if got != want:
            _fail(f"климатология {sid}: подогнана по окну {got or 'неизвестно'}, "
                  f"а обучающее окно {want}")
        if deep:
            from mayak.data.climatology import Climatology
            from mayak.data.store import CLIM_PARAMS
            from mayak.timeaxis import window_calendar
            lo, hi = want
            d, h = window_calendar(s["t0"], np.arange(lo, hi))
            ref = Climatology(CLIM_PARAMS["n_year"], CLIM_PARAMS["n_day"]).fit(
                d.astype(np.float64), h.astype(np.float64), s["x"][lo:hi, 0],
                s["mask"][lo:hi, 0], min_valid=CLIM_PARAMS["min_valid"])
            if not np.allclose(ref.beta, s["clim"].beta, rtol=1e-6, atol=1e-6):
                _fail(f"климатология {sid}: коэффициенты не воспроизводятся "
                      f"по обучающему окну {want}")


def check_windows(datasets, store=None):
    """Каждое окно каждого датасета лежит внутри своего временного окна.

    Датасет обязан отдавать footprints(): записи dict(sid, N, time_key, lo, hi),
    где [lo, hi) — часы, которые читают окна (история + горизонт). Границы
    пересчитываются заново через time_bounds, а не берутся из датасета.
    """
    n = 0
    for ds in datasets:
        name = type(ds).__name__
        if not hasattr(ds, "footprints"):
            _fail(f"{name}: не умеет перечислять свои окна (нет footprints())")
        for fp in ds.footprints():
            key = fp["time_key"]
            if key not in TIME_KEYS:
                _fail(f"{name}: неизвестное временное окно {key!r}")
            lo, hi = time_bounds(fp["N"])[key]
            flo, fhi = np.asarray(fp["lo"]), np.asarray(fp["hi"])
            bad = np.flatnonzero((flo < lo) | (fhi > hi))
            if bad.size:
                i = int(bad[0])
                _fail(f"{name}: окно станции {fp['sid']} [{int(flo[i])}, {int(fhi[i])}) "
                      f"выходит за окно {key} [{lo}, {hi}) (всего {bad.size} окон)")
            if store is not None and key in ALLOWED_ROLES:
                role = store.stations[fp["sid"]]["role"]
                if role not in ALLOWED_ROLES[key]:
                    _fail(f"{name}: станция {fp['sid']} роли {role} в окне {key} "
                          f"(допустимы {sorted(ALLOWED_ROLES[key])})")
            n += flo.size
    return n


def selection_record(val_ds, monitor):
    """Запись о том, на чём выбирался чекпойнт. Кладётся в .ckpt под SELECTION_KEY."""
    stations = sorted({fp["sid"] for fp in val_ds.footprints()})
    return dict(monitor=monitor, station_role=val_ds.station_role,
                time_key=val_ds.time_key, stations=stations, **_split_state())


def check_selection_record(rec, store, what="чекпойнт"):
    if not rec:
        _fail(f"{what}: нет записи о выборе ({SELECTION_KEY}) — неизвестно, на каких "
              f"данных выбран чекпойнт")
    if not str(rec.get("monitor") or "").startswith("val/"):
        _fail(f"{what}: выбран по метрике {rec.get('monitor')!r}, а не по val/*")
    if rec.get("station_role") != ROLE_VAL or rec.get("time_key") != SELECTION_TIME_KEY:
        _fail(f"{what}: выбран на станциях {rec.get('station_role')!r} в окне "
              f"{rec.get('time_key')!r}; нужно {ROLE_VAL!r} / {SELECTION_TIME_KEY!r}")
    _check_split_state(rec, what)
    _check_station_roles(rec.get("stations", []), store, ROLE_VAL, what)


def check_checkpoint(path, store):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    check_selection_record(ckpt.get(SELECTION_KEY), store, what=f"чекпойнт {path}")


def conformal_meta_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def conformal_record(ds, checkpoint=None):
    stations = sorted({fp["sid"] for fp in ds.footprints()})
    return dict(station_roles=sorted(ds.station_splits), time_key=ds.time_key,
                stations=stations, checkpoint=checkpoint, **_split_state())


def save_conformal(path, shift, record):
    """Таблица поправок + метаданные рядом (<имя>.meta.json)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, shift)
    with open(conformal_meta_path(path), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)


def check_conformal(path, store):
    meta = conformal_meta_path(path)
    what = f"конформная таблица {path}"
    if not os.path.exists(meta):
        _fail(f"{what}: нет метаданных {meta} — неизвестно, на каких данных она подогнана")
    with open(meta) as f:
        rec = json.load(f)
    if rec.get("station_roles") != [ROLE_VAL] or rec.get("time_key") != CONFORMAL_TIME_KEY:
        _fail(f"{what}: подогнана на ролях {rec.get('station_roles')} в окне "
              f"{rec.get('time_key')!r}; нужно [{ROLE_VAL!r}] / {CONFORMAL_TIME_KEY!r}")
    _check_split_state(rec, what)
    _check_station_roles(rec.get("stations", []), store, ROLE_VAL, what)


def run_checklist(store, datasets=(), conformal=None, checkpoints=(), deep=False):
    """Прогнать чек-лист. Возвращает сводку; при нарушении — LeakageError."""
    for s in store.stations.values():
        check_time_bounds(s["N"])
    check_climatology(store, deep=deep)
    n_windows = check_windows(datasets, store)
    if conformal:
        check_conformal(conformal, store)
    for c in checkpoints:
        check_checkpoint(c, store)
    summary = dict(stations=len(store.stations), windows=n_windows,
                   conformal=bool(conformal), checkpoints=len(checkpoints))
    log.info("чек-лист антиутечек пройден: %s", summary)
    return summary
