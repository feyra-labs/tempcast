"""Роли станций и временные окна сплитов.

Временная ось каждой станции режется на четыре непересекающихся окна,
разделённых защитными зазорами:

    [ train ][gap][ val ][gap][ calib ][gap][ test ]

* train — обучение и подгонка всего, что подгоняется по данным станции
  (климатология, damped persistence);
* val   — выбор чекпойнта и любые решения по гиперпараметрам;
* calib — только конформная поправка;
* test  — только тестирование модели.

Роли станций: train (обучение), unseen_val (валидация и калибровка),
unseen_test (финальная оценка). Разбиение — assign_roles.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from mayak.constants import H, L_MAX

log = logging.getLogger(__name__)

SPLITS_VERSION = "2"   # Увеличивать при изменении; входит в ключ кэша.
MIN_GAP_HOURS = H + L_MAX
TIME_BOUNDS = dict(hours_per_year=8766, test_frac=0.3, val_days=60, calib_days=60,
                   gap_hours=MIN_GAP_HOURS)
TIME_KEYS = ("train", "val", "calib", "test")

ROLE_TRAIN, ROLE_VAL, ROLE_TEST = "train", "unseen_val", "unseen_test"
ROLES = (ROLE_TRAIN, ROLE_VAL, ROLE_TEST)


def time_bounds(n_hours, **overrides):
    """Четыре окна ряда длины n_hours → {ключ: (начало, конец)} в индексах часа, [начало, конец).

    test  — последний год, но не больше test_frac ряда (короткие ряды);
    calib — calib_days суток перед test (через зазор);
    val   — val_days суток перед calib (через зазор);
    train — всё, что осталось в начале ряда до зазора перед val.
    """
    cfg = {**TIME_BOUNDS, **overrides}
    gap = int(cfg["gap_hours"])
    if gap < MIN_GAP_HOURS:
        raise ValueError(f"зазор {gap} ч меньше H + L_MAX = {MIN_GAP_HOURS} ч")
    n = int(n_hours)
    test_len = int(min(cfg["hours_per_year"], np.floor(cfg["test_frac"] * n)))
    calib_len, val_len = int(cfg["calib_days"]) * 24, int(cfg["val_days"]) * 24

    test = (n - test_len, n)
    calib = (test[0] - gap - calib_len, test[0] - gap)
    val = (calib[0] - gap - val_len, calib[0] - gap)
    train = (0, val[0] - gap)
    out = dict(train=train, val=val, calib=calib, test=test)
    need = dict(train=H + 1, val=L_MAX + H + 1, calib=L_MAX + H + 1, test=L_MAX + H + 1)
    short = {k: out[k][1] - out[k][0] for k in TIME_KEYS if out[k][1] - out[k][0] < need[k]}
    if short:
        raise ValueError(f"ряд из {n} ч слишком короток для четырёх окон с зазором {gap} ч: "
                         f"длины {short} меньше минимальных {need}")
    return out


def lat_band(lat):
    a = abs(float(lat))
    return "eq" if a < 23.5 else ("mid" if a < 50 else "pol")


def stratum_of(row):
    """Страта станции: полная зона Кёппена × широтный пояс."""
    return str(row["koppen"]).strip(), lat_band(row["lat"])


def _allocate(total, sizes, floor, cap, rng):
    sizes = np.asarray(sizes, np.float64)
    alloc = np.minimum(np.asarray(floor, np.int64), cap).astype(np.int64)
    quota = total * sizes / max(sizes.sum(), 1.0)
    tie = rng.random(len(sizes))
    for _ in range(max(0, int(total) - int(alloc.sum()))):
        room = alloc < cap
        if not room.any():
            break
        deficit = np.where(room, quota - alloc, -np.inf)
        best = np.flatnonzero(deficit == deficit.max())
        alloc[best[np.argmax(tie[best])]] += 1
    return alloc


def assign_roles(rows, n_test=8, val_frac=0.1, n_val=None, seed=0, min_stratum=3):
    """Стратифицированное разбиение станций на три роли.

    Возвращает {id станции: роль}.
    """
    if min_stratum < 3:
        raise ValueError("min_stratum < 3: в страте не хватит станций на три роли")
    rng = np.random.default_rng(seed)
    strata = {}
    for r in sorted(rows, key=lambda r: str(r["id"])):
        strata.setdefault(stratum_of(r), []).append(str(r["id"]))
    keys = sorted(strata)
    groups = [list(rng.permutation(strata[k])) for k in keys]
    sizes = np.array([len(g) for g in groups])
    big = (sizes >= min_stratum).astype(np.int64)

    k_test = _allocate(n_test, sizes, big, sizes - 1, rng)
    n_rest = int(sizes.sum() - k_test.sum())
    n_val = int(round(val_frac * n_rest)) if n_val is None else int(n_val)
    k_val = _allocate(n_val, sizes - k_test, big, sizes - 1 - k_test, rng)

    for name, want, got in (("unseen_test", n_test, k_test.sum()), ("unseen_val", n_val, k_val.sum())):
        if got != want:
            log.warning("роль %s: запрошено %d станций, назначено %d "
                        "(гарантии представительства страт / нехватка станций)", name, want, got)

    roles = {}
    for g, kt, kv in zip(groups, k_test, k_val):
        for i, sid in enumerate(g):
            roles[sid] = ROLE_TEST if i < kt else (ROLE_VAL if i < kt + kv else ROLE_TRAIN)
    return roles


def strata_report(rows, roles):
    """{страта: {роль: число станций}} — для печати и тестов."""
    rep = {}
    for r in rows:
        rep.setdefault(stratum_of(r), Counter())[roles[str(r["id"])]] += 1
    return {k: dict(v) for k, v in sorted(rep.items())}
