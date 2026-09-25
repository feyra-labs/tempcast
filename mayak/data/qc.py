"""Единый контроль качества.

Один модуль и один формат выхода для любого источника данных - реанализа, наблюдений
реальной сети, аугментированного обучающего окна и потока рантайма:

* x     - значения float32 (N, 3), каналы CHANNELS = (T, P, RH);
* mask  - по-канальная маска валидности uint8 (N, 3), mask == (codes == 0);
* codes - по-канальные коды причин отбраковки uint8 (N, 3), битовые флаги QCCode.

Инвариант: там, где маска ноль, значение обнулено (enforce_invariant).

Три уровня проверок:

1. поточечные и оконные - одинаковы для офлайн-сборки кэша
и для окна истории после аугментаций;
2. станционные - выполняются один раз при сборке кэша;
3. отбор станций - по длине ряда, доле валидной температуры
   и по результатам станционных проверок.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from enum import IntFlag

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mayak.data.masking import enforce_invariant

CHANNELS = ("T", "P", "RH")
PHYS = {"T": (-90.0, 60.0), "RH": (0.0, 100.0), "P": (300.0, 1100.0)}
MAD_HALF, MAD_THRESH, MAD_MIN_VALID = 6, 6.0, 4
MAD_TO_SD = 1.4826
P_SEA_LEVEL = 1013.25


class QCCode(IntFlag):
    """Коды причин отбраковки."""
    OK = 0
    MISSING = 1
    RANGE = 2
    SPIKE = 4
    SOURCE = 8
    JUMP = 16
    STUCK = 32
    UNITS = 64
    DEWPOINT = 128


QC_CODE_DOC = {
    "MISSING": "нет значения",
    "RANGE": "вне физического диапазона",
    "SPIKE": "выброс относительно скользящей медианы",
    "SOURCE": "помечено источником",
    "JUMP": "аномальный часовой скачок",
    "STUCK": "залипшее значение",
    "UNITS": "подмена единиц",
    "DEWPOINT": "точка росы выше температуры",
}
QC_CODES = tuple(c for c in QCCode if c)


@dataclass(frozen=True)
class QCConfig:
    """Пороги контроля качества. Все поля входят в ключ кэша.

    Кортежи из трёх элементов - по каналам (T, P, RH).
    """
    spike_half: int = MAD_HALF
    spike_thresh: float = MAD_THRESH
    spike_min_valid: int = MAD_MIN_VALID
    jump_half: int = 24
    jump_thresh: float = 8.0
    jump_min_valid: int = 12
    jump_floor: tuple = (8.0, 6.0, 40.0)            # °C/ч, гПа/ч, %/ч
    excursion_max_hours: int = 3                    # скачок туда и обратно за ≤ ч - выброс
    stuck_hours: tuple = (24, 24, 24)               # допустимый срок одного значения, ч
    stuck_max_gap: int = 6                          # пропуск внутри серии не длиннее, ч
    stuck_min_count: int = 4                        # не меньше стольких отсчётов в серии
    rh_sat: float = 99.5                            # «сотня процентов» для RH, %
    rh_sat_hours: int = 72                          # допустимый срок насыщения, ч
    units_half: int = 12                            # полуокно скользящей медианы, ч
    units_min_valid: int = 12
    units_ref_days: int = 45                        # полуокно опорного уровня, сутки
    units_ref_k: float = 3.0                        # в единицах робастного разброса
    units_spread_floor: float = 1.0                 # °C
    units_min_excess: float = 8.0                   # °C: минимальный отрыв от опоры
    units_min_conv: float = -10.0                   # °C: ниже шкалы °F и °C почти совпадают
    slp_min_sep: float = 80.0                       # гПа: станция заметно выше уровня моря
    dewpoint_tol: float = 0.5                       # °C: допуск на округление источника
    day_min_hours: int = 18                         # сутки считаются, если валидно ≥ ч
    solar_min_amp: float = 0.5                      # °C: слабее - проверка не решает
    solar_min_days: int = 30
    solar_lag: tuple = (-2.0, 6.0)                  # ч: максимум T после солнечного полудня
    cp_min_seg_days: int = 180
    cp_min_shift: float = 1.5                       # °C
    cp_min_z: float = 5.0
    dem_tol_m: float = 300.0
    t_median: tuple = (-70.0, 38.0)                 # °C: медиана T станции
    min_hours: int = 8766
    min_valid_frac_T: float = 0.5
    enforce_checks: tuple = ("t_level", "solar_phase", "changepoint", "dem_elevation")

    def __post_init__(self):
        s = object.__setattr__
        for name in ("jump_floor", "solar_lag", "t_median"):
            s(self, name, tuple(float(v) for v in getattr(self, name)))
        s(self, "stuck_hours", tuple(int(v) for v in self.stuck_hours))
        s(self, "enforce_checks", tuple(str(v) for v in self.enforce_checks))
        if len(self.jump_floor) != 3 or len(self.stuck_hours) != 3:
            raise ValueError("jump_floor и stuck_hours — по одному значению на канал T, P, RH")
        unknown = set(self.enforce_checks) - set(STATION_CHECKS)
        if unknown:
            raise ValueError(f"неизвестные станционные проверки {sorted(unknown)}; "
                             f"есть {STATION_CHECKS}")
        if not self.solar_lag[0] < self.solar_lag[1]:
            raise ValueError(f"solar_lag = {self.solar_lag}: нужна пара lo < hi")
        if not 0.0 <= self.min_valid_frac_T <= 1.0:
            raise ValueError("min_valid_frac_T вне [0, 1]")

    def to_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def fingerprint(self):
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


STATION_CHECKS = ("t_level", "solar_phase", "changepoint", "dem_elevation")
DEFAULT_QC = QCConfig()


def _median_sorted(s, cnt):
    """Медиана по оси -1 для массива, отсортированного с NaN в конце; cnt - число не-NaN."""
    c = np.maximum(cnt, 1)
    lo = np.take_along_axis(s, ((c - 1) // 2)[:, None], -1)[:, 0]
    hi = np.take_along_axis(s, (c // 2)[:, None], -1)[:, 0]
    return (lo + hi) / 2


def rolling_median_mad(x, valid, half, chunk=1 << 16, at=None):
    """Скользящие медиана, MAD (+1e-6) и число валидных точек в окне [i − half, i + half].

    Невалидные точки в окно не входят. Там, где в окне нет ни одной валидной точки,
    медиана и MAD - NaN. at - индексы, в которых считать (по умолчанию все);
    результат тогда длины len(at). Возвращает (med, mad, cnt).
    """
    x = np.asarray(x)
    n = len(x)
    xv = np.where(np.asarray(valid) > 0, x, np.nan).astype(x.dtype, copy=False)
    xp = np.pad(xv, (half, half), constant_values=np.nan)
    pos = np.arange(n) if at is None else np.asarray(at, np.int64)
    k = len(pos)
    med = np.empty(k, xv.dtype)
    mad = np.empty(k, xv.dtype)
    cnt = np.empty(k, np.int64)
    view = sliding_window_view(xp, 2 * half + 1)
    for a in range(0, k, chunk):
        b = min(k, a + chunk)
        w = view[a:b] if at is None else view[pos[a:b]]
        c = (~np.isnan(w)).sum(-1)
        s = np.sort(w, axis=-1)
        m = _median_sorted(s, c)
        dev = np.sort(np.abs(w - m[:, None]), axis=-1)
        med[a:b], mad[a:b], cnt[a:b] = m, _median_sorted(dev, c) + 1e-6, c
    return med, mad, cnt


def rolling_median(x, valid, half, min_valid=1, at=None):
    """Скользящая медиана по валидным точкам; NaN, где их меньше min_valid."""
    med, _, cnt = rolling_median_mad(x, valid, half, at=at)
    return np.where(cnt >= min_valid, med, np.nan)


def _window_count(flag, half):
    """Число True в окне [i − half, i + half] для каждого i (кумулятивной суммой)."""
    c = np.concatenate([[0], np.cumsum(np.asarray(flag, np.int64))])
    n = len(flag)
    i = np.arange(n)
    return c[np.minimum(n, i + half + 1)] - c[np.maximum(0, i - half)]


def mad_ok(x, valid, half=MAD_HALF, thresh=MAD_THRESH, min_valid=MAD_MIN_VALID, chunk=1 << 16):
    """True там, где точка не выброс: |x − med| ≤ thresh · 1.4826 · MAD (векторно)."""
    x = np.asarray(x)
    med, mad, cnt = rolling_median_mad(x, valid, half, chunk)
    bad = np.abs(x - med) > thresh * MAD_TO_SD * mad
    return ~((cnt >= min_valid) & bad)


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


def jump_flags(x, ok, half, thresh, min_valid, floor, excursion_max_hours=0):
    """Аномальный часовой скачок → (jump, excursion), bool (N,).

    Приращение d_t = x_t − x_{t−1} определено, только если валидны оба часа (правило
    лагов блока 1). Скачок аномален, если d_t - робастный выброс среди приращений
    в окне ±half ч и одновременно |d_t| > floor. Флаг ставится на час t - первый
    час после скачка.
    """
    x = np.asarray(x, np.float64)
    ok = np.asarray(ok, bool)
    n = len(x)
    jump = np.zeros(n, bool)
    exc = np.zeros(n, bool)
    if n < 2:
        return jump, exc
    dv = np.zeros(n, bool)
    dv[1:] = ok[1:] & ok[:-1]
    d = np.zeros(n)
    d[1:] = np.where(dv[1:], x[1:] - x[:-1], 0.0)
    cand = np.flatnonzero(dv & (np.abs(d) > floor))
    if cand.size == 0:
        return jump, exc
    med, mad, cnt = rolling_median_mad(d, dv, half, at=cand)
    rel = (cnt >= min_valid) & (np.abs(d[cand] - med) > thresh * MAD_TO_SD * mad)
    J = cand[rel]
    used = np.zeros(len(J), bool)
    for i in range(len(J) - 1):
        a, b = J[i], J[i + 1]
        if used[i] or b - a > excursion_max_hours:
            continue
        if d[a] * d[b] < 0 and abs(d[a] + d[b]) <= 0.5 * min(abs(d[a]), abs(d[b])):
            exc[a:b] = True
            used[i] = used[i + 1] = True
    jump[J[~used]] = True
    return jump, exc


def _runs(idx, same, n, min_hours, max_gap, min_count):
    """Серии соседних валидных отсчётов → флаг на отсчётах длинных серий.

    idx  — индексы валидных часов (возрастают);
    same — (len(idx) − 1,) отсчёты k и k+1 принадлежат одной серии по значению.
    Серия рвётся, если значения различаются или пропуск между отсчётами > max_gap.
    Серия длинная, если охватывает ≥ min_hours часов и содержит ≥ min_count отсчётов.
    """
    out = np.zeros(n, bool)
    if idx.size < max(2, min_count):
        return out
    cont = same & (np.diff(idx) <= max_gap)
    start = np.r_[True, ~cont]
    rid = np.cumsum(start) - 1
    first = idx[start]
    last = np.r_[idx[np.flatnonzero(start)[1:] - 1], idx[-1]]
    count = np.bincount(rid)
    long_ = (last - first + 1 >= min_hours) & (count >= min_count)
    out[idx] = long_[rid]
    return out


def stuck_flags(x, ok, min_hours, max_gap=6, min_count=4, below=None):
    """Одно и то же значение дольше min_hours (сравнение точное, после float32).

    below - если задано, в серии участвуют только значения < below (для RH значения
    насыщения обрабатываются отдельным правилом ``saturation_flags``).
    """
    x = np.asarray(x)
    idx = np.flatnonzero(np.asarray(ok, bool))
    v = x[idx]
    same = v[1:] == v[:-1]
    if below is not None:
        el = v < below
        same &= el[1:] & el[:-1]
    return _runs(idx, same, len(x), min_hours, max_gap, min_count)


def saturation_flags(rh, ok, sat, min_hours, max_gap=6, min_count=4):
    """Влажность не ниже sat дольше min_hours - залипание на сотне процентов."""
    rh = np.asarray(rh)
    idx = np.flatnonzero(np.asarray(ok, bool))
    hi = rh[idx] >= sat
    return _runs(idx, hi[1:] & hi[:-1], len(rh), min_hours, max_gap, min_count)


def _daily_reference(x, ok, cfg):
    """Опорный уровень и робастный разброс ряда в масштабе недель → почасовые массивы.

    Суточные медианы (блоки по 24 ч от начала массива; сутки с ≥ units_min_valid
    валидными часами) → скользящая медиана и MAD по ±units_ref_days суток.
    """
    x = np.asarray(x, np.float64)
    n = len(x)
    nd = -(-n // 24)
    xm = np.full(nd * 24, np.nan)
    xm[:n] = np.where(ok, x, np.nan)
    xm = np.sort(xm.reshape(nd, 24), axis=1)
    cnt = np.isfinite(xm).sum(1)
    day = _median_sorted(xm, cnt)
    dv = (cnt >= cfg.units_min_valid) & np.isfinite(day)
    med, mad, c = rolling_median_mad(np.where(dv, day, 0.0), dv, cfg.units_ref_days)
    ref = np.where(c >= 3, med, np.nan)
    spread = np.maximum(MAD_TO_SD * mad, cfg.units_spread_floor)
    hours = np.arange(n) // 24
    return ref[hours], spread[hours]


def fahrenheit_flags(T, ok_raw, ok_ref, cfg=DEFAULT_QC):
    """Участки ряда температуры в °F.

    Суточная скользящая медиана r(t) по сырым значениям резко выше опорного уровня
    ряда, а после перевода (r − 32) / 1.8 ложится на опорный уровень. Опора — медиана
    суточных медиан за ±units_ref_days суток по значениям, прошедшим проверку
    диапазона, поэтому короткий участок в °F её не сдвигает.

    Ограничения: около −40 °C шкалы совпадают, а при отрицательных температурах
    отрыв (0.8·T + 32) мал — такие участки неотличимы от холодной погоды (например,
    от арктического потепления на 10 °C).
    """
    T = np.asarray(T, np.float64)
    ok_raw = np.asarray(ok_raw, bool)
    out = np.zeros(len(T), bool)
    ref, spread = _daily_reference(T, ok_ref, cfg)
    thr = np.maximum(cfg.units_ref_k * spread, cfg.units_min_excess)
    g = (ref + thr)[::24]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gmin = np.nanmin(np.stack([np.r_[g[:1], g[:-1]], g, np.r_[g[1:], g[-1:]]]), axis=0)
    with np.errstate(invalid="ignore"):
        above = ok_raw & (T > gmin[np.arange(len(T)) // 24])
    n_above = _window_count(above, cfg.units_half)
    n_valid = _window_count(ok_raw, cfg.units_half)
    cand = np.flatnonzero(ok_raw & (n_valid >= cfg.units_min_valid) & (2 * n_above >= n_valid))
    if cand.size == 0:
        return out
    r = rolling_median(T, ok_raw, cfg.units_half, cfg.units_min_valid, at=cand)
    conv = (r - 32.0) / 1.8
    with np.errstate(invalid="ignore"):
        high = r - ref[cand] > thr[cand]
        fits = ((np.abs(conv - ref[cand]) <= cfg.units_ref_k * spread[cand])
                & (conv >= cfg.units_min_conv))
    out[cand[high & fits]] = True
    return out


def station_pressure_expected(elev):
    """Давление стандартной атмосферы на высоте elev, гПа."""
    return P_SEA_LEVEL * (1.0 - float(elev) / 44330.0) ** 5.255


def sea_level_pressure_flags(P, ok_raw, elev, cfg=DEFAULT_QC):
    """Давление, приведённое к уровню моря, вместо станционного.

    Решаемо только для станций, у которых стандартное давление ниже уровня моря
    хотя бы на slp_min_sep (≈ 700 м и выше): иначе синоптический размах
    перекрывает разницу. Флаг - суточная медиана ближе к уровню моря, чем
    к ожидаемому станционному давлению.
    """
    n = len(P)
    if elev is None or not np.isfinite(elev):
        return np.zeros(n, bool)
    p_exp = station_pressure_expected(elev)
    sep = P_SEA_LEVEL - p_exp
    if sep < cfg.slp_min_sep:
        return np.zeros(n, bool)
    P = np.asarray(P, np.float64)
    ok_raw = np.asarray(ok_raw, bool)
    out = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        above = ok_raw & (P > p_exp + sep / 2)
    n_above = _window_count(above, cfg.units_half)
    n_valid = _window_count(ok_raw, cfg.units_half)
    cand = np.flatnonzero(ok_raw & (n_valid >= cfg.units_min_valid) & (2 * n_above >= n_valid))
    if cand.size == 0:
        return out
    r = rolling_median(P, ok_raw, cfg.units_half, cfg.units_min_valid, at=cand)
    out[cand[r > p_exp + sep / 2]] = True
    return out


def dewpoint_flags(T, Td, ok_T, cfg=DEFAULT_QC):
    """Точка росы выше температуры больше допуска. Флаг идёт на канал RH."""
    T = np.asarray(T, np.float64)
    Td = np.asarray(Td, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(ok_T, bool) & np.isfinite(Td) & (Td > T + cfg.dewpoint_tol)


def _spike_flags(x, base, cfg):
    """SPIKE по всем каналам одним проходом скользящей медианы.

    Каналы склеиваются в один ряд через разделители из spike_half невалидных точек:
    окно ±spike_half ни одного канала не дотягивается до соседнего, поэтому результат
    побитово равен поканальному ``mad_ok``, а накладные расходы - втрое меньше.
    """
    n, c = x.shape
    h = cfg.spike_half
    xs = np.full((c, n + h), np.nan, x.dtype)
    vs = np.zeros((c, n + h), bool)
    xs[:, :n], vs[:, :n] = x.T, base.T
    ok = mad_ok(xs.ravel()[:-h] if h else xs.ravel(), vs.ravel()[:-h] if h else vs.ravel(),
                h, cfg.spike_thresh, cfg.spike_min_valid)
    ok = np.r_[ok, np.ones(h, bool)].reshape(c, n + h)[:, :n].T
    return base & ~ok


def check_codes(x, src, elev=None, Td=None, cfg=DEFAULT_QC):
    """Поточечные и оконные проверки → коды uint8 (N, 3), без MISSING / SOURCE.

    x   — значения (N, 3);
    src — (N, 3) bool: значение есть и не помечено источником;
    elev — высота станции, м (для проверки давления; None — проверка пропускается);
    Td  — точка росы (N,), если источник даёт её отдельно (реальные наблюдения).
    """
    x = np.asarray(x)
    src = np.asarray(src, bool)
    n = x.shape[0]
    codes = np.zeros((n, 3), np.uint8)
    lo = np.array([PHYS[c][0] for c in CHANNELS], x.dtype)
    hi = np.array([PHYS[c][1] for c in CHANNELS], x.dtype)
    with np.errstate(invalid="ignore"):
        phys = (x >= lo) & (x <= hi)
    codes[src & ~phys] |= np.uint8(QCCode.RANGE)
    base = src & phys
    spikes = _spike_flags(x, base, cfg)
    codes[spikes] |= np.uint8(QCCode.SPIKE)
    for j, name in enumerate(CHANNELS):
        v, b, spike = x[:, j], base[:, j], spikes[:, j]
        jump, exc = jump_flags(v, b & ~spike, cfg.jump_half, cfg.jump_thresh,
                               cfg.jump_min_valid, cfg.jump_floor[j], cfg.excursion_max_hours)
        codes[jump, j] |= np.uint8(QCCode.JUMP)
        codes[exc, j] |= np.uint8(QCCode.SPIKE)
        below = cfg.rh_sat if name == "RH" else None
        stuck = stuck_flags(v, b, cfg.stuck_hours[j], cfg.stuck_max_gap, cfg.stuck_min_count,
                            below=below)
        codes[stuck, j] |= np.uint8(QCCode.STUCK)
    sat = saturation_flags(x[:, 2], base[:, 2], cfg.rh_sat, cfg.rh_sat_hours,
                           cfg.stuck_max_gap, cfg.stuck_min_count)
    codes[sat, 2] |= np.uint8(QCCode.STUCK)
    codes[fahrenheit_flags(x[:, 0], src[:, 0], base[:, 0], cfg), 0] |= np.uint8(QCCode.UNITS)
    codes[sea_level_pressure_flags(x[:, 1], src[:, 1], elev, cfg), 1] |= np.uint8(QCCode.UNITS)
    if Td is not None:
        codes[dewpoint_flags(x[:, 0], Td, base[:, 0], cfg), 2] |= np.uint8(QCCode.DEWPOINT)
    return codes


def _as_channel_valid(valid, n):
    """Маска источника (N,) или (N, 3) → (N, 3) bool."""
    v = np.asarray(valid)
    if v.ndim == 1:
        v = np.repeat(v[:, None], len(CHANNELS), axis=1)
    if v.shape != (n, len(CHANNELS)):
        raise ValueError(f"маска источника формы {v.shape}, ожидалось ({n},) или ({n}, 3)")
    return v > 0


def qc_station(T, P, RH, valid, *, Td=None, flag=None, elev=None, cfg=DEFAULT_QC):
    """Полный QC ряда станции → (x float32 (N,3), mask uint8 (N,3), codes uint8 (N,3)).

    valid - маска наличия от источника, (N,) или (N, 3);
    flag  - штатные флаги источника «подозрительно», (N,) или (N, 3);
    Td    - точка росы (N,), если источник даёт её отдельно;
    elev  - высота станции, м.
    """
    x = np.stack([T, P, RH], axis=-1).astype(np.float32)
    n = x.shape[0]
    src = _as_channel_valid(valid, n) & np.isfinite(x)
    codes = np.zeros((n, 3), np.uint8)
    codes[~src] |= np.uint8(QCCode.MISSING)
    if flag is not None:
        fl = _as_channel_valid(flag, n) & src
        codes[fl] |= np.uint8(QCCode.SOURCE)
        src &= ~fl
    codes |= check_codes(x, src, elev=elev, Td=Td, cfg=cfg)
    mask = (codes == 0).astype(np.uint8)
    x, _ = enforce_invariant(x, mask)
    return x, mask, codes


def run_qc(T, P, RH, valid):
    x, mask, _ = qc_station(T, P, RH, valid)
    return x, mask.astype(np.float32)


def qc_window(x, mask, elev=None, cfg=DEFAULT_QC):
    x = np.asarray(x)
    src = (np.asarray(mask) > 0) & np.isfinite(x)
    codes = check_codes(x, src, elev=elev, cfg=cfg)
    codes[~src] |= np.uint8(QCCode.MISSING)
    return (codes == 0).astype(np.float32), codes


def point_qc(T, P, RH):
    out = np.zeros(3, np.float32)
    mask = np.zeros(3, np.float32)
    for j, (name, val) in enumerate(zip(CHANNELS, (T, P, RH))):
        lo, hi = PHYS[name]
        if val is not None and np.isfinite(val) and lo <= val <= hi:
            out[j] = val
            mask[j] = 1.0
    return out, mask


def code_fractions(codes, mask=None):
    """Доли часов с каждым кодом по каналам (+ доля валидных, если дана маска)."""
    codes = np.asarray(codes)
    n = len(codes)
    out = {}
    for j, ch in enumerate(CHANNELS):
        if mask is not None:
            out[f"{ch}/valid"] = float(np.mean(np.asarray(mask)[:, j] > 0)) if n else 0.0
        for c in QC_CODES:
            out[f"{ch}/{c.name}"] = float(np.mean((codes[:, j] & c) > 0)) if n else 0.0
    return out


def _result(status, value=None, detail=""):
    return dict(status=status, value=None if value is None else float(value), detail=detail)


def _day_index(n, t0):
    abs_h = int(t0) + np.arange(n, dtype=np.int64)
    day = abs_h // 24
    return day - day[0], abs_h % 24, int(day[0])


def check_t_level(T, ok, cfg=DEFAULT_QC):
    """Медиана температуры станции в допустимых пределах.

    Ловит ряд, целиком записанный в °F (у тёплых и умеренных станций медиана
    выходит за 38 °C) и прочие грубые ошибки уровня.
    """
    ok = np.asarray(ok, bool)
    if not ok.any():
        return _result("skip", detail="нет валидной температуры")
    med = float(np.median(np.asarray(T)[ok]))
    lo, hi = cfg.t_median
    if lo <= med <= hi:
        return _result("pass", med)
    return _result("fail", med, f"медиана T {med:.1f} °C вне [{lo:g}, {hi:g}]")


def diurnal_max_utc(T, ok, t0, cfg=DEFAULT_QC):
    """Час UTC максимума первой суточной гармоники аномалии T и её амплитуда.

    Аномалия - отклонение от среднего своих суток UTC (сутки с ≥ day_min_hours
    валидными часами). Возвращает (час максимума UTC, амплитуда °C, число суток).
    """
    T = np.asarray(T, np.float64)
    ok = np.asarray(ok, bool) & np.isfinite(T)
    day, hour, _ = _day_index(len(T), t0)
    cnt = np.bincount(day, weights=ok.astype(np.float64))
    s = np.bincount(day, weights=np.where(ok, T, 0.0))
    good = cnt >= cfg.day_min_hours
    use = ok & good[day]
    n_days = int(good.sum())
    if not use.any():
        return float("nan"), 0.0, n_days
    anom = T[use] - (s / np.maximum(cnt, 1))[day[use]]
    w = 2 * np.pi * hour[use] / 24.0
    a, b = np.sum(anom * np.cos(w)), np.sum(anom * np.sin(w))
    amp = 2.0 * math.hypot(a, b) / use.sum()
    t_max = (math.atan2(b, a) * 24.0 / (2 * np.pi)) % 24.0
    return t_max, amp, n_days


def check_solar_phase(T, ok, t0, lon, cfg=DEFAULT_QC):
    """Фаза суточного хода согласована с солнечной геометрией точки.

    Максимум первой суточной гармоники T переводится в местное солнечное время
    (UTC + lon/15) и сравнивается с солнечным полуднем: запаздывание должно лежать
    в solar_lag. Ловит смещения на несколько часов: ошибку знака долготы у станций
    с |lon| ≳ 30° и местное время вместо UTC у станций с |lon| ≳ 60°. Ошибки
    меньше ~2 ч этой проверкой не видны. При слабом суточном ходе (полярная ночь,
    морской климат) проверка не решает и возвращает skip.
    """
    if lon is None or not np.isfinite(lon):
        return _result("skip", detail="нет долготы")
    t_max, amp, n_days = diurnal_max_utc(T, ok, t0, cfg)
    if n_days < cfg.solar_min_days:
        return _result("skip", detail=f"мало полных суток: {n_days}")
    if amp < cfg.solar_min_amp:
        return _result("skip", amp, f"слабый суточный ход: {amp:.2f} °C")
    lag = (t_max + float(lon) / 15.0 - 12.0 + 12.0) % 24.0 - 12.0
    lo, hi = cfg.solar_lag
    if lo <= lag <= hi:
        return _result("pass", lag)
    return _result("fail", lag, f"максимум T в {lag:+.1f} ч от солнечного полудня, "
                                f"допустимо [{lo:+g}, {hi:+g}]: часовой пояс или знак долготы")


def check_changepoint(T, ok, t0, cfg=DEFAULT_QC):
    """Точка разладки в среднем уровне ряда: замена прибора, переезд станции.

    Остаток от гармонической климатологии ряда → суточные средние → одна точка
    разладки, максимизирующая k(n−k)/n·(m₁ − m₂)² при сегментах не короче
    cp_min_seg_days. Величина сдвига затем переоценивается совместной регрессией
    «гармоники + ступенька в найденной точке». Разладка значима, если |m₂ − m₁| ≥
    cp_min_shift и z ≥ cp_min_z,
    где z учитывает автокорреляцию суточных средних через эффективный объём
    n·(1 − ρ)/(1 + ρ).
    """
    from mayak.data.climatology import Climatology
    from mayak.timeaxis import from_utc_hour, window_calendar

    T = np.asarray(T, np.float64)
    ok = np.asarray(ok, bool) & np.isfinite(T)
    n = len(T)
    if ok.sum() < 2 * cfg.cp_min_seg_days * cfg.day_min_hours:
        return _result("skip", detail="ряд короче двух минимальных сегментов")
    doy, hour = window_calendar(t0, np.arange(n))
    try:
        clim = Climatology().fit(doy.astype(np.float64), hour.astype(np.float64), T,
                                 ok.astype(np.float64))
    except ValueError as e:
        return _result("skip", detail=str(e))
    resid = np.where(ok, T - clim.predict(doy, hour), 0.0)
    day, _, day0 = _day_index(n, t0)
    cnt = np.bincount(day, weights=ok.astype(np.float64))
    s = np.bincount(day, weights=resid)
    good = np.flatnonzero(cnt >= cfg.day_min_hours)
    r = s[good] / cnt[good]
    m = len(r)
    k_min = cfg.cp_min_seg_days
    if m < 2 * k_min:
        return _result("skip", detail=f"мало полных суток: {m}")
    c = np.cumsum(r)
    k = np.arange(k_min, m - k_min + 1)
    m1 = c[k - 1] / k
    m2 = (c[-1] - c[k - 1]) / (m - k)
    i = int(np.argmax(k * (m - k) / m * (m1 - m2) ** 2))
    kb = int(k[i])
    from mayak.data.climatology import _design
    hb = int(np.flatnonzero(day == good[kb])[0])
    step = (np.arange(n) >= hb).astype(np.float64)
    A = np.column_stack([_design(doy.astype(np.float64), hour.astype(np.float64),
                                 clim.n_year, clim.n_day), step])
    coef, *_ = np.linalg.lstsq(A[ok], T[ok], rcond=None)
    shift = float(coef[-1])
    resid2 = np.where(ok, T - A @ coef, 0.0)
    e = np.bincount(day, weights=resid2)[good] / cnt[good]
    sd = float(np.std(e)) + 1e-9
    rho = float(np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], 0.0, 0.99)) if m > 2 else 0.0
    f = (1.0 - rho) / (1.0 + rho)
    z = abs(shift) / (sd * math.sqrt(1.0 / (kb * f) + 1.0 / ((m - kb) * f)))
    when = str(from_utc_hour((day0 + int(good[kb])) * 24).astype("datetime64[D]"))
    if abs(shift) >= cfg.cp_min_shift and z >= cfg.cp_min_z:
        return _result("fail", shift, f"сдвиг уровня {shift:+.2f} °C с {when} (z = {z:.1f})")
    return _result("pass", shift, f"наибольший сдвиг {shift:+.2f} °C с {when} (z = {z:.1f})")


def check_dem_elevation(elev, dem_elev, cfg=DEFAULT_QC):
    """Заявленная высота согласована с высотой из цифровой модели рельефа."""
    if dem_elev is None or elev is None or not (np.isfinite(dem_elev) and np.isfinite(elev)):
        return _result("skip", detail="нет высоты из ЦМР")
    d = float(elev) - float(dem_elev)
    if abs(d) <= cfg.dem_tol_m:
        return _result("pass", d)
    return _result("fail", d, f"заявленная высота отличается от ЦМР на {d:+.0f} м "
                              f"(допуск {cfg.dem_tol_m:g} м)")


def station_checks(x, mask, t0, lon=None, elev=None, dem_elev=None, cfg=DEFAULT_QC,
                   raw_T=None, codes=None):
    """Станционные проверки по ряду после поточечного QC → {имя: результат}.

    Результат: dict(status='pass'|'fail'|'skip', value, detail).
    """
    T, ok = np.asarray(x)[:, 0], np.asarray(mask)[:, 0] > 0
    if raw_T is not None and codes is not None:
        lvl_T = np.asarray(raw_T, np.float64)
        lvl_ok = ((np.asarray(codes)[:, 0] & np.uint8(QCCode.MISSING | QCCode.SOURCE)) == 0) \
            & np.isfinite(lvl_T)
    else:
        lvl_T, lvl_ok = T, ok
    return {
        "t_level": check_t_level(lvl_T, lvl_ok, cfg),
        "solar_phase": check_solar_phase(T, ok, t0, lon, cfg),
        "changepoint": check_changepoint(T, ok, t0, cfg),
        "dem_elevation": check_dem_elevation(elev, dem_elev, cfg),
    }


def station_selection(n, mask, checks, cfg=DEFAULT_QC):
    """Правила отбора станций → список причин [(правило, текст)]; пусто - годна."""
    reasons = []
    if n < cfg.min_hours:
        reasons.append(("length", f"ряд {n} ч короче {cfg.min_hours} ч"))
    mT = np.asarray(mask)[:, 0] > 0
    if n == 0 or not mT.any():
        reasons.append(("no_T", "нет канала температуры"))
    elif mT.mean() < cfg.min_valid_frac_T:
        reasons.append(("valid_T", f"валидной температуры {mT.mean():.1%} "
                                   f"< {cfg.min_valid_frac_T:.0%}"))
    for name in cfg.enforce_checks:
        res = checks.get(name)
        if res is not None and res["status"] == "fail":
            reasons.append((f"check:{name}", f"{name}: {res['detail']}"))
    return reasons
