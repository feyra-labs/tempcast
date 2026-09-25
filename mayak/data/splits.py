"""Роли станций и временная раскладка ряда.

Ряд каждой станции делится на обучение, год валидации и калибровки и тест:

    [ обучение ][ зазор ][ в | к | в | к | ... ][ зазор ][ тест ]

Роли станций: train (обучение), unseen_val (валидация и калибровка),
unseen_test (финальная оценка). Роль external_test - станции
реальной сети наблюдений: из их ряда читается только тест, а обучение служит
одной цели - климатологии самой станции. Разбиение на роли их не трогает.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from mayak.codehash import code_digests, combined_digest
from mayak.constants import H, L_MAX

log = logging.getLogger(__name__)

MIN_GAP_HOURS = H + L_MAX
TIME_LAYOUT = dict(hours_per_year=8766, test_frac=0.3, n_blocks=12, gap_hours=MIN_GAP_HOURS)
TIME_KEYS = ("train", "val", "calib", "test")
BLOCK_KEYS = ("val", "calib")
MIN_BLOCK_HOURS = H + 1

HISTORY_FORBIDDEN = {"train": ("val", "calib", "test"), "val": ("train", "test"),
                     "calib": ("train", "test"), "test": ("train", "val", "calib")}

LAYOUT_CODE = {
    "mayak.data.splits": ("MIN_GAP_HOURS", "TIME_LAYOUT", "TIME_KEYS", "BLOCK_KEYS",
                          "MIN_BLOCK_HOURS", "HISTORY_FORBIDDEN", "TimeLayout", "time_layout"),
    "mayak.constants": ("H", "L_MAX"),
}

ROLE_TRAIN, ROLE_VAL, ROLE_TEST = "train", "unseen_val", "unseen_test"
ROLES = (ROLE_TRAIN, ROLE_VAL, ROLE_TEST)
ROLE_EXTERNAL = "external_test"
ALL_ROLES = ROLES + (ROLE_EXTERNAL,)

DEFAULT_TEST_FRAC = 0.15
DEFAULT_VAL_FRAC = 0.1
MIN_TRAIN_YEARS = 3
EXTERNAL_MIN_TRAIN_YEARS = MIN_TRAIN_YEARS
FULL_YEAR_MIN_MONTH_FRAC = 0.3


@dataclass(frozen=True)
class TimeLayout:
    """Раскладка ряда одной станции по временным окнам.

    Attributes:
        n_hours: длина ряда, ч.
        blocks: блоки каждого временного окна, по ключу окна. Блок - пара индексов
            часа ряда (начало, конец), конец не входит. У обучения и теста по одному
            блоку, у валидации и калибровки - по половине блоков года перед тестом.
    """

    n_hours: int
    blocks: dict

    def span(self, key):
        """Границы временного окна от начала первого блока до конца последнего.

        Args:
            key: ключ временного окна.

        Returns:
            Пара (начало, конец), конец не входит.
        """
        b = self.blocks[key]
        return b[0][0], b[-1][1]

    def history_floor(self, key):
        """Самый ранний час, который может читать история окна с этим ключом.

        Args:
            key: ключ временного окна.

        Returns:
            Индекс часа ряда.
        """
        first = self.blocks[key][0][0]
        ends = [hi for k in HISTORY_FORBIDDEN[key] for _lo, hi in self.blocks[k] if hi <= first]
        return max(ends, default=0)

    def block_index(self, key, start, end):
        """Номер блока, в котором целиком лежит отрезок часов.

        Args:
            key: ключ временного окна.
            start: начала отрезков, массив индексов часа.
            end: концы отрезков, конец не входит, массив той же формы.

        Returns:
            Массив номеров блоков той же формы; минус один, если отрезок не лежит
            целиком ни в одном блоке.
        """
        start, end = np.asarray(start, np.int64), np.asarray(end, np.int64)
        los = np.array([lo for lo, _hi in self.blocks[key]], np.int64)
        his = np.array([hi for _lo, hi in self.blocks[key]], np.int64)
        idx = np.searchsorted(los, start, side="right") - 1
        safe = np.clip(idx, 0, len(los) - 1)
        ok = (idx >= 0) & (start >= los[safe]) & (end <= his[safe]) & (start < end)
        return np.where(ok, idx, -1)

    def overlaps(self, keys, start, end):
        """Пересекает ли отрезок часов хотя бы один блок перечисленных окон.

        Args:
            keys: ключи временных окон.
            start: начала отрезков, массив индексов часа.
            end: концы отрезков, конец не входит, массив той же формы.

        Returns:
            Булев массив той же формы. Пустой отрезок ничего не пересекает.
        """
        start, end = np.asarray(start, np.int64), np.asarray(end, np.int64)
        hit = np.zeros(np.broadcast(start, end).shape, bool)
        for k in keys:
            for lo, hi in self.blocks[k]:
                hit |= (start < hi) & (end > lo) & (start < end)
        return hit

    def to_dict(self):
        """Раскладка в виде, пригодном для JSON.

        Returns:
            Словарь с длиной ряда и списками блоков по ключам.
        """
        return dict(n_hours=self.n_hours,
                    blocks={k: [list(b) for b in self.blocks[k]] for k in TIME_KEYS})


def time_layout(n_hours, **overrides):
    """Раскладка ряда заданной длины по временным окнам.

    Args:
        n_hours: длина ряда, ч.
        **overrides: замена параметров раскладки по умолчанию.

    Returns:
        Раскладка ряда.

    Raises:
        ValueError: зазор короче горизонта и полной истории вместе, число блоков не
            чётное, либо ряд так короток, что обучение, тест или хотя бы один блок не
            вмещают ни одного окна.
    """
    cfg = {**TIME_LAYOUT, **overrides}
    gap, n_blocks = int(cfg["gap_hours"]), int(cfg["n_blocks"])
    if gap < MIN_GAP_HOURS:
        raise ValueError(f"зазор {gap} ч меньше горизонта и полной истории вместе "
                         f"({MIN_GAP_HOURS} ч)")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError(f"число блоков валидации и калибровки {n_blocks}: нужно чётное, "
                         f"не меньше двух")
    n = int(n_hours)
    year = int(min(int(cfg["hours_per_year"]), np.floor(float(cfg["test_frac"]) * n)))
    test = (n - year, n)
    vc_hi = test[0] - gap
    vc_lo = vc_hi - year
    edges = [vc_lo + (i * year) // n_blocks for i in range(n_blocks + 1)]
    inner = list(zip(edges[:-1], edges[1:]))
    train = (0, vc_lo - gap)
    out = TimeLayout(n_hours=n, blocks=dict(train=(train,), val=tuple(inner[0::2]),
                                            calib=tuple(inner[1::2]), test=(test,)))
    short = {k: min(hi - lo for lo, hi in out.blocks[k]) for k in TIME_KEYS
             if min(hi - lo for lo, hi in out.blocks[k]) < MIN_BLOCK_HOURS}
    if short:
        raise ValueError(f"ряд из {n} ч слишком короток для раскладки с зазором {gap} ч и "
                         f"{n_blocks} блоками: самые короткие блоки {short} меньше "
                         f"{MIN_BLOCK_HOURS} ч")
    return out


def layout_fingerprint():
    """Отпечаток кода временной раскладки ряда.

    Меняется при смысловой правке раскладки, её параметров по умолчанию, горизонта,
    длины истории и правил, куда может заходить история окна. Правка комментариев и
    документации его не меняет.

    Returns:
        Строка из шестнадцатеричных цифр.
    """
    return combined_digest(code_digests(LAYOUT_CODE))


def full_years(mask_T, t0_utc_h, lo, hi, min_month_frac=FULL_YEAR_MIN_MONTH_FRAC,
               hours_per_year=None):
    """Число полных лет в окне [lo, hi) ряда станции.

    Окно режется на годовые блоки по hours_per_year часов, отсчитывая от hi назад
    (неполный остаток в начале отбрасывается). Блок - полный год, если в каждом из
    12 календарных месяцев валидна не меньше min_month_frac часов этого месяца:
    гармоникам годового хода нужен весь сезонный цикл, а не только много часов.
    """
    from mayak.timeaxis import window_month
    hpy = int(hours_per_year or TIME_LAYOUT["hours_per_year"])
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
    """Наименьшая длина ряда, при которой обучение занимает не меньше заданного числа лет.

    Args:
        years: сколько лет должно занимать обучение.
        **overrides: замена параметров раскладки по умолчанию.

    Returns:
        Длина ряда, ч.
    """
    cfg = {**TIME_LAYOUT, **overrides}
    need = max(1, int(years)) * int(cfg["hours_per_year"])
    n = need + 2 * int(cfg["gap_hours"])
    while True:
        try:
            lo, hi = time_layout(n, **overrides).span("train")
            if hi - lo >= need:
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


def assign_roles(rows, n_test=None, test_frac=DEFAULT_TEST_FRAC, val_frac=DEFAULT_VAL_FRAC,
                 n_val=None, seed=0, min_stratum=3, info=None):
    """Стратифицированное разбиение станций на три роли.

    Args:
        rows: строки манифеста с полями id, koppen, lat и, возможно, split.
        n_test: точное число тестовых станций; None - по доле test_frac.
        test_frac: доля тестовых станций среди участвующих в разбиении.
        val_frac: доля валидационных среди станций вне теста.
        n_val: точное число валидационных станций; None - по доле val_frac.
        seed: сид разбиения.
        min_stratum: страты от этого размера обязаны быть во всех ролях.
        info: если передан словарь, в него кладутся запрошенное и назначенное число
            тестовых и валидационных станций.

    Returns:
        Словарь: id станции и её роль.

    Raises:
        ValueError: min_stratum меньше трёх.
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

    want_test = int(round(test_frac * len(rows))) if n_test is None else int(n_test)
    k_test = _allocate(want_test, sizes, big, sizes - 1, rng)
    n_rest = int(sizes.sum() - k_test.sum())
    want_val = int(round(val_frac * n_rest)) if n_val is None else int(n_val)
    k_val = _allocate(want_val, sizes - k_test, big, sizes - 1 - k_test, rng)

    for name, want, got in ((ROLE_TEST, want_test, k_test.sum()),
                            (ROLE_VAL, want_val, k_val.sum())):
        if got != want:
            log.warning("роль %s: запрошено %d станций, назначено %d "
                        "(гарантии представительства страт или нехватка станций)",
                        name, want, got)
    if info is not None:
        info.update(requested={ROLE_TEST: want_test, ROLE_VAL: want_val},
                    assigned={ROLE_TEST: int(k_test.sum()), ROLE_VAL: int(k_val.sum()),
                              ROLE_TRAIN: int(sizes.sum() - k_test.sum() - k_val.sum())})

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
