"""Исполняемый чек-лист антиутече.

Четыре проверки, вызываются перед обучением и перед оценкой:

1. климатология каждой станции (среднее и масштаб остатка) подогнана только
   по её обучающему окну;
2. конформная таблица построена только по калибровочному окну и только
   на валидационных станциях;
3. ни одно окно датасета не выходит за границу своего временного окна
   ни историей, ни горизонтом (а временные окна разделены зазором ≥ H + L_MAX);
4. чекпойнт выбран по метрике на валидационных станциях в валидационном окне;
5. окна внешних станций - только в тестовом временном окне; внешние станции не
   встречаются в основном наборе под другой ролью, в записи о выборе чекпойнта и
   в метаданных конформной таблицы (``check_external``).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from mayak.data.splits import (MIN_GAP_HOURS, ROLE_EXTERNAL, ROLE_TRAIN, ROLE_VAL,
                               SPLITS_VERSION, TIME_BOUNDS, TIME_KEYS, time_bounds)

log = logging.getLogger(__name__)

SELECTION_KEY = "mayak_selection"          # ключ записи о выборе чекпойнта в .ckpt
SELECTION_TIME_KEY, CONFORMAL_TIME_KEY = "val", "calib"
# какие роли станций допустимы в окнах датасета с данным временным ключом
ALLOWED_ROLES = {"train": {ROLE_TRAIN}, "calib": {ROLE_VAL}}
# в каких временных окнах допустимы станции внешнего теста
EXTERNAL_TIME_KEYS = {"test"}


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
            from mayak.data.store import CLIM_PARAMS, new_climatology
            from mayak.timeaxis import window_calendar
            lo, hi = want
            d, h = window_calendar(s["t0"], np.arange(lo, hi))
            ref = new_climatology().fit(
                d.astype(np.float64), h.astype(np.float64), s["x"][lo:hi, 0],
                s["mask"][lo:hi, 0], min_valid=CLIM_PARAMS["min_valid"])
            got = s["clim"]
            same = (np.allclose(ref.beta, got.beta, rtol=1e-6, atol=1e-6)
                    and got.scale_beta is not None
                    and np.allclose(ref.scale_beta, got.scale_beta, rtol=1e-6, atol=1e-6))
            if not same:
                _fail(f"климатология {sid}: коэффициенты среднего или масштаба не "
                      f"воспроизводятся по обучающему окну {want}")


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
            if store is not None:
                role = store.stations[fp["sid"]]["role"]
                if key in ALLOWED_ROLES and role not in ALLOWED_ROLES[key]:
                    _fail(f"{name}: станция {fp['sid']} роли {role} в окне {key} "
                          f"(допустимы {sorted(ALLOWED_ROLES[key])})")
                if role == ROLE_EXTERNAL and key not in EXTERNAL_TIME_KEYS:
                    _fail(f"{name}: станция внешнего теста {fp['sid']} в окне {key}; "
                          f"внешние станции читаются только в {sorted(EXTERNAL_TIME_KEYS)}")
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


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_train_km(store, external_store):
    """{id внешней станции: расстояние до ближайшей обучающей точки, км}."""
    tr = store.by_role(ROLE_TRAIN)
    if not tr:
        return {}
    lat = np.array([s["lat"] for s in tr])
    lon = np.array([s["lon"] for s in tr])
    return {sid: float(_haversine_km(s["lat"], s["lon"], lat, lon).min())
            for sid, s in external_store.stations.items()}


def check_external(store, external_store, checkpoints=(), conformal=None, near_km=10.0):
    """Внешний тест изолирован от обучения, выбора чекпойнта и калибровки.

    store - основной набор (обучение, валидация, калибровка, внутренний тест);
    external_store - набор внешнего теста. Возвращает сводку, в том числе число
    внешних станций ближе near_km к обучающей точке.
    """
    ext = external_store.stations
    if not ext:
        _fail("внешний тест: пустой набор станций")
    wrong = {sid: s["role"] for sid, s in ext.items() if s["role"] != ROLE_EXTERNAL}
    if wrong:
        _fail(f"внешний тест: станции не из роли {ROLE_EXTERNAL}: {dict(list(wrong.items())[:5])}")
    if external_store is not store:
        clash = {sid: store.stations[sid]["role"] for sid in ext
                 if sid in store.stations and store.stations[sid]["role"] != ROLE_EXTERNAL}
        if clash:
            _fail(f"внешний тест: станции есть в основном наборе под другой ролью: "
                  f"{dict(list(clash.items())[:5])}")
    ext_ids = set(ext)
    import torch
    for c in checkpoints:
        rec = torch.load(c, map_location="cpu", weights_only=False).get(SELECTION_KEY) or {}
        hit = sorted(ext_ids & set(rec.get("stations", [])))
        if hit:
            _fail(f"чекпойнт {c}: выбран с участием станций внешнего теста {hit[:5]}")
    if conformal:
        with open(conformal_meta_path(conformal)) as f:
            rec = json.load(f)
        hit = sorted(ext_ids & set(rec.get("stations", [])))
        if hit or ROLE_EXTERNAL in rec.get("station_roles", []):
            _fail(f"конформная таблица {conformal}: подогнана с участием внешнего теста {hit[:5]}")
    dist = nearest_train_km(store, external_store)
    close = sorted(sid for sid, d in dist.items() if d < near_km)
    summary = dict(external_stations=len(ext), checkpoints=len(checkpoints),
                   conformal=bool(conformal), near_train=len(close), near_km=near_km,
                   median_nearest_train_km=float(np.median(list(dist.values()))) if dist else None)
    log.info("внешний тест изолирован: %s", summary)
    if close:
        log.info("внешние станции ближе %.0f км к обучающей точке: %s", near_km, close[:20])
    return summary


def run_checklist(store, datasets=(), conformal=None, checkpoints=(), deep=False,
                  external_store=None):
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
    if external_store is not None:
        summary["external"] = check_external(store, external_store, checkpoints=checkpoints,
                                             conformal=conformal)
    log.info("чек-лист антиутечек пройден: %s", summary)
    return summary
