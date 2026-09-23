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

Четвёртая роль — external_test: станции реальной сети наблюдений (GHCNh), внешний
тест. Они не участвуют ни в обучении, ни в выборе чекпойнта, ни в конформной
калибровке; из их временных окон читается только test, а train служит одной цели —
подгонке климатологии самой станции (эталон скилла). assign_roles их не трогает.
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
ROLE_EXTERNAL = "external_test"
ALL_ROLES = ROLES + (ROLE_EXTERNAL,)

EXTERNAL_MIN_TRAIN_YEARS = 3
FULL_YEAR_MIN_MONTH_FRAC = 0.3


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


def full_years(mask_T, t0_utc_h, lo, hi, min_month_frac=FULL_YEAR_MIN_MONTH_FRAC,
               hours_per_year=None):
    """Число полных лет в окне [lo, hi) ряда станции.

    Окно режется на годовые блоки по hours_per_year часов, отсчитывая от hi назад
    (неполный остаток в начале отбрасывается). Блок - полный год, если в каждом из
    12 календарных месяцев валидна не меньше min_month_frac часов этого месяца:
    гармоникам годового хода нужен весь сезонный цикл, а не только много часов.
    """
    from mayak.timeaxis import window_month
    hpy = int(hours_per_year or TIME_BOUNDS["hours_per_year"])
    n_blocks = max(0, (int(hi) - int(lo)) // hpy)
    if n_blocks == 0:
        return 0
    a = int(hi) - n_blocks * hpy
    k = np.arange(a, int(hi))
    block = (k - a) // hpy
    month = np.asarray(window_month(t0_utc_h, k), np.int64) - 1
    key = block * 12 + month
    total = np.bincount(key, minlength=n_blocks * 12).reshape(n_blocks, 12)
    valid = np.bincount(key, weights=(np.asarray(mask_T)[a:int(hi)] > 0).astype(np.float64),
                        minlength=n_blocks * 12).reshape(n_blocks, 12)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, valid / np.maximum(total, 1), 0.0)
    return int(np.sum(np.all((total > 0) & (frac >= min_month_frac), axis=1)))


def min_hours_for_train_years(years, **overrides):
    """Минимальная длина ряда, при которой в обучающем окне ≥ years лет."""
    cfg = {**TIME_BOUNDS, **overrides}
    hpy, gap = int(cfg["hours_per_year"]), int(cfg["gap_hours"])
    rest = 3 * gap + (int(cfg["val_days"]) + int(cfg["calib_days"])) * 24
    n = max(1, int(years)) * hpy + rest
    while True:
        try:
            b = time_bounds(n, **overrides)
            if b["train"][1] - b["train"][0] >= int(years) * hpy:
                return n
        except ValueError:
            pass
        n += 24


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

    Станции с ролью external_test в манифесте в разбиении не участвуют и сохраняют
    роль. Возвращает {id станции: роль}.
    """
    if min_stratum < 3:
        raise ValueError("min_stratum < 3: в страте не хватит станций на три роли")
    rng = np.random.default_rng(seed)
    external = sorted(str(r["id"]) for r in rows if str(r.get("split") or "") == ROLE_EXTERNAL)
    rows = [r for r in rows if str(r.get("split") or "") != ROLE_EXTERNAL]
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

    for name, want, got in (("unseen_test", n_test, k_test.sum()),
                            ("unseen_val", n_val, k_val.sum())):
        if got != want:
            log.warning("роль %s: запрошено %d станций, назначено %d "
                        "(гарантии представительства страт / нехватка станций)", name, want, got)

    roles = {sid: ROLE_EXTERNAL for sid in external}
    for g, kt, kv in zip(groups, k_test, k_val):
        for i, sid in enumerate(g):
            roles[sid] = ROLE_TEST if i < kt else (ROLE_VAL if i < kt + kv else ROLE_TRAIN)
    return roles


def strata_report(rows, roles):
    """{страта: {роль: число станций}} — для печати и тестов."""
    rep = {}
    for r in rows:
        if roles[str(r["id"])] == ROLE_EXTERNAL:
            continue
        rep.setdefault(stratum_of(r), Counter())[roles[str(r["id"])]] += 1
    return {k: dict(v) for k, v in sorted(rep.items())}
