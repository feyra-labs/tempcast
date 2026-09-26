"""Единый контроль качества.

Один модуль и один формат выхода для любого источника данных: реанализа, наблюдений
реальной сети, обучающего окна и потока устройства.

* x: значения float32, форма (N, 3), каналы T, P, RH;
* mask: поканальная маска валидности uint8, форма (N, 3), единица там, где кодов нет;
* codes: поканальные коды причин отбраковки uint8, форма (N, 3), битовые флаги QCCode.

Там, где маска ноль, значение обнулено.

Оконные проверки работают в двух режимах с одними правилами и одними порогами.

* Центрированный режим видит часы по обе стороны от проверяемого. Им один раз
  чистится источник при сборке кэша: из него берутся маска цели и станционные проверки.
* Причинный режим решает о часе только по этому часу и по прошлым часам, не дальше
  QCConfig.lookback_hours. Решение о часе не меняется, когда приходят новые часы.
  Этим режимом проходит история в обучении, в оценке и на устройстве.

Три уровня проверок:

1. поточечные и оконные;
2. станционные, выполняются один раз при сборке кэша;
3. отбор станций по длине ряда, доле валидной температуры и результатам станционных
   проверок.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntFlag

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mayak.data.masking import enforce_invariant
from mayak.data.recording import record_values

CHANNELS = ("T", "P", "RH")
PHYS = {"T": (-90.0, 60.0), "RH": (0.0, 100.0), "P": (300.0, 1100.0)}
MAD_HALF, MAD_THRESH, MAD_MIN_VALID = 6, 8.0, 4
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
    "JUMP": "аномальный скачок между соседними отчётами",
    "STUCK": "залипшее значение",
    "UNITS": "давление на уровне моря вместо станционного",
    "DEWPOINT": "точка росы выше температуры",
}
QC_CODES = tuple(c for c in QCCode if c)


@dataclass(frozen=True)
class QCConfig:
    """Пороги контроля качества. Все поля входят в ключ кэша.

    Attributes:
        spike_half: полуширина окна проверки выброса, ч.
        spike_thresh: порог выброса в единицах робастного разброса.
        spike_min_valid: наименьшее число отчётов в окне выброса.
        scale_floor: нижняя граница робастного разброса по каналам, больше шага
            записи: полтора градуса, три десятых гПа, два процента. У целочисленного
            ряда медиана отклонений часто равна нулю, без границы любой шаг записи был
            бы выбросом. После ровной ночи прошлое окно не знает об утреннем подъёме,
            граница не даёт принять его за выброс.
        jump_half: полуширина окна, в котором оцениваются приращения, ч.
        jump_thresh: порог скачка в единицах робастного разброса приращений.
        jump_min_valid: наименьшее число приращений в окне.
        jump_floor: наименьший скачок по каналам в единицах за час.
        jump_max_gap: приращение считается между соседними отчётами, если между ними
            не больше стольких часов; оно делится на прошедшее время.
        excursion_max_hours: скачок и обратный скачок в пределах стольких часов
            считаются выбросом, а не сменой уровня.
        stuck_hours: срок одного значения по каналам, ч. Для температуры это срок при
            одновременно стоящей влажности.
        stuck_T_alone_hours: срок одной температуры при меняющейся влажности, ч. Целая
            температура ночью и в пасмурную погоду держится одинаковой много часов.
        stuck_max_gap: пропуск внутри серии одинаковых значений не длиннее, ч.
        stuck_min_count: наименьшее число отчётов в серии.
        rh_sat: влажность не ниже этого значения считается насыщением, %.
        rh_sat_hours: допустимый срок насыщения, ч.
        slp_half: полуширина окна медианы давления, ч. Та же длина, что у окна
            выброса: в причинном режиме, пока медиана не переключилась на приведённое
            давление, его часы ловит проверка выброса, и разрыва между ними нет.
        slp_min_sep: давление на уровне моря отличимо от станционного, если станция
            ниже уровня моря по давлению хотя бы на столько гПа.
        slp_min_valid: наименьшее число отчётов в окне медианы давления.
        dewpoint_tol: допуск на округление точки росы источника, градусы.
        day_min_hours: сутки считаются в станционных проверках, если валидно не меньше.
        solar_min_amp: при более слабом суточном ходе проверка фазы не решает.
        solar_min_days: наименьшее число полных суток для проверки фазы.
        solar_lag: допустимое запаздывание максимума температуры от полудня, ч.
        cp_min_seg_days: наименьший сегмент разладки, сутки.
        cp_min_shift: наименьший значимый сдвиг уровня, градусы.
        cp_min_z: наименьшая значимость сдвига.
        dem_tol_m: допуск расхождения высоты с цифровой моделью рельефа, м.
        t_median: допустимые пределы медианы температуры станции.
        min_hours: наименьшая длина ряда станции, ч.
        min_valid_frac_T: наименьшая доля валидной температуры.
        enforce_checks: станционные проверки, провал которых исключает станцию.
    """
    spike_half: int = MAD_HALF
    spike_thresh: float = MAD_THRESH
    spike_min_valid: int = MAD_MIN_VALID
    scale_floor: tuple = (1.5, 0.3, 2.0)
    jump_half: int = 24
    jump_thresh: float = 8.0
    jump_min_valid: int = 10
    jump_floor: tuple = (8.0, 6.0, 40.0)
    jump_max_gap: int = 6
    excursion_max_hours: int = 3
    stuck_hours: tuple = (24, 24, 24)
    stuck_T_alone_hours: int = 72
    stuck_max_gap: int = 6
    stuck_min_count: int = 4
    rh_sat: float = 99.5
    rh_sat_hours: int = 72
    slp_half: int = 6
    slp_min_sep: float = 80.0
    slp_min_valid: int = 6
    dewpoint_tol: float = 0.5
    day_min_hours: int = 18
    solar_min_amp: float = 0.5
    solar_min_days: int = 30
    solar_lag: tuple = (-2.0, 6.0)
    cp_min_seg_days: int = 180
    cp_min_shift: float = 1.5
    cp_min_z: float = 5.0
    dem_tol_m: float = 300.0
    t_median: tuple = (-70.0, 38.0)
    min_hours: int = 8766
    min_valid_frac_T: float = 0.5
    enforce_checks: tuple = ("t_level", "solar_phase", "changepoint", "dem_elevation")

    def __post_init__(self):
        s = object.__setattr__
        for name in ("scale_floor", "jump_floor", "solar_lag", "t_median"):
            s(self, name, tuple(float(v) for v in getattr(self, name)))
        s(self, "stuck_hours", tuple(int(v) for v in self.stuck_hours))
        s(self, "enforce_checks", tuple(str(v) for v in self.enforce_checks))
        if len(self.jump_floor) != 3 or len(self.stuck_hours) != 3 or len(self.scale_floor) != 3:
            raise ValueError("scale_floor, jump_floor и stuck_hours задаются по одному "
                             "значению на канал T, P, RH")
        if min(self.scale_floor) <= 0:
            raise ValueError(f"scale_floor = {self.scale_floor}: нужны положительные границы")
        if self.jump_max_gap < 1 or self.stuck_max_gap < 1:
            raise ValueError("jump_max_gap и stuck_max_gap не меньше часа")
        shortest = min(self.stuck_hours + (self.stuck_T_alone_hours, self.rh_sat_hours))
        if self.stuck_min_count * self.stuck_max_gap > shortest:
            raise ValueError(f"stuck_min_count {self.stuck_min_count} отчётов через "
                             f"{self.stuck_max_gap} ч не помещаются в кратчайший срок "
                             f"залипания {shortest} ч")
        unknown = set(self.enforce_checks) - set(STATION_CHECKS)
        if unknown:
            raise ValueError(f"неизвестные станционные проверки {sorted(unknown)}; "
                             f"есть {STATION_CHECKS}")
        if not self.solar_lag[0] < self.solar_lag[1]:
            raise ValueError(f"solar_lag = {self.solar_lag}: нужна пара lo < hi")
        if not 0.0 <= self.min_valid_frac_T <= 1.0:
            raise ValueError("min_valid_frac_T вне [0, 1]")

    @property
    def lookback_hours(self):
        """Сколько прошлых часов нужно причинному режиму, чтобы решить о часе.

        Returns:
            Число часов.
        """
        spike = 2 * self.spike_half
        jump = self.excursion_max_hours + 2 * self.jump_half + self.jump_max_gap + spike
        stuck = max(self.stuck_hours + (self.stuck_T_alone_hours, self.rh_sat_hours)) \
            + self.stuck_max_gap
        return max(spike, jump, stuck, 2 * self.slp_half)

    def to_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def fingerprint(self):
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


STATION_CHECKS = ("t_level", "solar_phase", "changepoint", "dem_elevation")
DEFAULT_QC = QCConfig()
QC_MODES = ("centered", "causal")


def window_span(half, causal):
    """Сколько часов окно берёт до и после проверяемого часа.

    Args:
        half: полуширина окна, ч.
        causal: причинный режим.

    Returns:
        Пара чисел часов: до и после.
    """
    return (2 * half, 0) if causal else (half, half)


def _median_sorted(s, cnt):
    """Медиана по последней оси массива, отсортированного с NaN в конце.

    Args:
        s: отсортированный массив, форма (K, W).
        cnt: число конечных значений в каждой строке, форма (K,).

    Returns:
        Медианы, форма (K,).
    """
    c = np.maximum(cnt, 1)
    lo = np.take_along_axis(s, ((c - 1) // 2)[:, None], -1)[:, 0]
    hi = np.take_along_axis(s, (c // 2)[:, None], -1)[:, 0]
    return (lo + hi) / 2


def rolling_median_mad(x, valid, half, chunk=1 << 16, at=None, after=None):
    """Скользящие медиана, медиана абсолютных отклонений и число точек в окне.

    Args:
        x: значения, форма (N,).
        valid: валидность точек, форма (N,).
        half: сколько часов до проверяемого берёт окно.
        chunk: сколько окон обрабатывать за раз.
        at: индексы часов, в которых считать; по умолчанию все.
        after: сколько часов после проверяемого берёт окно; по умолчанию half.

    Returns:
        Тройка массивов длины N или len(at): медиана, медиана отклонений и число
        валидных точек. Там, где в окне нет ни одной точки, медиана и разброс NaN.
    """
    after = half if after is None else int(after)
    x = np.asarray(x, np.float64)
    n = len(x)
    xv = np.where(np.asarray(valid) > 0, x, np.nan)
    xp = np.pad(xv, (half, after), constant_values=np.nan)
    pos = np.arange(n) if at is None else np.asarray(at, np.int64)
    k = len(pos)
    med = np.empty(k)
    mad = np.empty(k)
    cnt = np.empty(k, np.int64)
    view = sliding_window_view(xp, half + after + 1)
    for a in range(0, k, chunk):
        b = min(k, a + chunk)
        w = view[a:b] if at is None else view[pos[a:b]]
        c = (~np.isnan(w)).sum(-1)
        s = np.sort(w, axis=-1)
        m = _median_sorted(s, c)
        dev = np.sort(np.abs(w - m[:, None]), axis=-1)
        med[a:b], mad[a:b], cnt[a:b] = m, _median_sorted(dev, c) + 1e-6, c
    return med, mad, cnt


def rolling_median(x, valid, half, min_valid=1, at=None, after=None):
    """Скользящая медиана по валидным точкам.

    Args:
        x: значения, форма (N,).
        valid: валидность точек, форма (N,).
        half: сколько часов до проверяемого берёт окно.
        min_valid: наименьшее число точек в окне.
        at: индексы часов, в которых считать; по умолчанию все.
        after: сколько часов после проверяемого берёт окно; по умолчанию half.

    Returns:
        Медианы; NaN там, где точек меньше min_valid.
    """
    med, _, cnt = rolling_median_mad(x, valid, half, at=at, after=after)
    return np.where(cnt >= min_valid, med, np.nan)


def _window_count(flag, half, after=None):
    """Число отмеченных часов в окне каждого часа.

    Args:
        flag: отметки, форма (N,).
        half: сколько часов до часа берёт окно.
        after: сколько часов после часа берёт окно; по умолчанию half.

    Returns:
        Целые счётчики, форма (N,).
    """
    after = half if after is None else int(after)
    c = np.concatenate([[0], np.cumsum(np.asarray(flag, np.int64))])
    n = len(flag)
    i = np.arange(n)
    return c[np.minimum(n, i + after + 1)] - c[np.maximum(0, i - half)]


def mad_ok(x, valid, half=MAD_HALF, thresh=MAD_THRESH, min_valid=MAD_MIN_VALID, chunk=1 << 16,
           floor=0.0, causal=False):
    """Какие точки не выбросы относительно скользящей медианы.

    Args:
        x: значения, форма (N,).
        valid: валидность точек, форма (N,).
        half: полуширина окна, ч.
        thresh: порог в единицах разброса.
        min_valid: наименьшее число точек в окне.
        chunk: сколько окон обрабатывать за раз.
        floor: нижняя граница разброса.
        causal: причинный режим окна.

    Returns:
        Булев массив (N,): True там, где точка не выброс.
    """
    x = np.asarray(x, np.float64)
    before, after = window_span(half, causal)
    med, mad, cnt = rolling_median_mad(x, valid, before, chunk, after=after)
    bad = np.abs(x - med) > thresh * np.maximum(MAD_TO_SD * mad, floor)
    return ~((cnt >= min_valid) & bad)


def _mad_ok_reference(x, valid, half=MAD_HALF, thresh=MAD_THRESH, min_valid=MAD_MIN_VALID,
                      floor=0.0, causal=False):
    """Медленный эталон проверки выброса для тестов."""
    x = np.asarray(x, np.float64)
    before, after = window_span(half, causal)
    n = len(x)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - before), min(n, i + after + 1)
        seg = x[lo:hi][valid[lo:hi] > 0]
        if len(seg) < min_valid:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-6
        if abs(x[i] - med) > thresh * max(MAD_TO_SD * mad, floor):
            ok[i] = False
    return ok


def increments(x, ok, max_gap):
    """Приращения между соседними валидными отчётами.

    Args:
        x: значения, форма (N,).
        ok: валидность отчётов, форма (N,).
        max_gap: наибольшее расстояние между отчётами, ч.

    Returns:
        Тройка массивов формы (N,): приращение за час, полное изменение уровня и
        признак того, что приращение определено.
    """
    x = np.asarray(x, np.float64)
    n = len(x)
    rate, level, defined = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    idx = np.flatnonzero(np.asarray(ok, bool))
    if idx.size < 2:
        return rate, level, defined
    gap = np.diff(idx)
    use = gap <= max_gap
    at = idx[1:][use]
    level[at] = (x[idx[1:]] - x[idx[:-1]])[use]
    rate[at] = level[at] / gap[use]
    defined[at] = True
    return rate, level, defined


def _cancels(da, db):
    """Второе изменение уровня возвращает ряд к уровню до первого."""
    return da * db < 0 and abs(da + db) <= 0.5 * min(abs(da), abs(db))


def jump_flags(x, ok, half, thresh, min_valid, floor, excursion_max_hours=0, max_gap=1,
               scale_floor=0.0, causal=False):
    """Аномальные скачки между соседними отчётами.

    Args:
        x: значения, форма (N,).
        ok: валидность отчётов, форма (N,).
        half: полуширина окна приращений, ч.
        thresh: порог в единицах робастного разброса приращений.
        min_valid: наименьшее число приращений в окне.
        floor: наименьший аномальный скачок в единицах за час.
        excursion_max_hours: наибольшая длина выброса туда и обратно, ч.
        max_gap: наибольшее расстояние между соседними отчётами, ч.
        scale_floor: нижняя граница разброса приращений.
        causal: причинный режим окна.

    Returns:
        Пара булевых массивов (N,): скачки и часы выброса туда и обратно.
    """
    ok = np.asarray(ok, bool)
    n = len(ok)
    jump = np.zeros(n, bool)
    exc = np.zeros(n, bool)
    rate, level, dv = increments(x, ok, max_gap)
    cand = np.flatnonzero(dv & (np.abs(rate) > floor))
    if cand.size == 0:
        return jump, exc
    before, after = window_span(half, causal)
    med, mad, cnt = rolling_median_mad(rate, dv, before, at=cand, after=after)
    scale = np.maximum(MAD_TO_SD * mad, scale_floor)
    rel = (cnt >= min_valid) & (np.abs(rate[cand] - med) > thresh * scale)
    J = cand[rel]
    if causal:
        keep = np.ones(len(J), bool)
        for i in range(len(J)):
            b = J[i]
            for k in range(i - 1, -1, -1):
                a = J[k]
                if b - a > excursion_max_hours:
                    break
                if _cancels(level[a], level[b]):
                    keep[i] = False
                    break
        jump[J[keep]] = True
        return jump, exc
    used = np.zeros(len(J), bool)
    for i in range(len(J) - 1):
        a, b = J[i], J[i + 1]
        if used[i] or b - a > excursion_max_hours:
            continue
        if _cancels(level[a], level[b]):
            exc[a:b] = True
            used[i] = used[i + 1] = True
    jump[J[~used]] = True
    return jump, exc & ok


def run_lengths(x, ok, max_gap, causal=False, below=None, at_least=None):
    """Серии соседних отчётов с одинаковым значением.

    Args:
        x: значения, форма (N,).
        ok: валидность отчётов, форма (N,).
        max_gap: наибольший пропуск внутри серии, ч.
        causal: причинный режим.
        below: если задано, в серии участвуют только значения меньше него.
        at_least: если задано, серия - подряд идущие значения не меньше него,
            одинаковыми они быть не обязаны.

    Returns:
        Пара целых массивов (N,): охват серии в часах и число отчётов в ней.
        У невалидных часов нули.
    """
    x = np.asarray(x, np.float64)
    n = len(x)
    span, count = np.zeros(n, np.int64), np.zeros(n, np.int64)
    idx = np.flatnonzero(np.asarray(ok, bool))
    if idx.size == 0:
        return span, count
    v = x[idx]
    if at_least is not None:
        el = v >= at_least
        link = el[1:] & el[:-1]
    else:
        link = v[1:] == v[:-1]
        if below is not None:
            el = v < below
            link &= el[1:] & el[:-1]
    link &= np.diff(idx) <= max_gap
    start = np.r_[True, ~link]
    starts = np.flatnonzero(start)
    rid = np.cumsum(start) - 1
    first_pos = starts[rid]
    if causal:
        last_pos = np.arange(len(idx))
    else:
        last_pos = np.r_[starts[1:] - 1, len(idx) - 1][rid]
    span[idx] = idx[last_pos] - idx[first_pos] + 1
    count[idx] = last_pos - first_pos + 1
    return span, count


def _long(span, count, hours, min_count):
    return (span >= hours) & (count >= min_count)


def stuck_flags(x, ok, min_hours, max_gap=6, min_count=4, below=None, causal=False):
    """Одно и то же значение дольше min_hours часов.

    Args:
        x: значения, форма (N,).
        ok: валидность отчётов, форма (N,).
        min_hours: допустимый срок одного значения, ч.
        max_gap: наибольший пропуск внутри серии, ч.
        min_count: наименьшее число отчётов в серии.
        below: если задано, в серии участвуют только значения меньше него.
        causal: причинный режим.

    Returns:
        Булев массив (N,).
    """
    span, count = run_lengths(x, ok, max_gap, causal, below=below)
    return _long(span, count, min_hours, min_count)


def saturation_flags(rh, ok, sat, min_hours, max_gap=6, min_count=4, causal=False):
    """Влажность не ниже sat дольше min_hours часов: залипание на насыщении.

    Args:
        rh: влажность, форма (N,).
        ok: валидность отчётов, форма (N,).
        sat: порог насыщения, %.
        min_hours: допустимый срок насыщения, ч.
        max_gap: наибольший пропуск внутри серии, ч.
        min_count: наименьшее число отчётов в серии.
        causal: причинный режим.

    Returns:
        Булев массив (N,).
    """
    span, count = run_lengths(rh, ok, max_gap, causal, at_least=sat)
    return _long(span, count, min_hours, min_count)


def stuck_codes(x, base, cfg=DEFAULT_QC, causal=False):
    """Залипание по всем каналам.

    Args:
        x: значения, форма (N, 3).
        base: отчёты, прошедшие проверку диапазона, форма (N, 3).
        cfg: пороги.
        causal: причинный режим.

    Returns:
        Булев массив (N, 3).
    """
    out = np.zeros(base.shape, bool)
    g, c = cfg.stuck_max_gap, cfg.stuck_min_count
    span_t, cnt_t = run_lengths(x[:, 0], base[:, 0], g, causal)
    span_p, cnt_p = run_lengths(x[:, 1], base[:, 1], g, causal)
    span_r, cnt_r = run_lengths(x[:, 2], base[:, 2], g, causal, below=cfg.rh_sat)
    rh_long = _long(span_r, cnt_r, cfg.stuck_hours[0], c)
    out[:, 0] = _long(span_t, cnt_t, cfg.stuck_T_alone_hours, c) \
        | (_long(span_t, cnt_t, cfg.stuck_hours[0], c) & rh_long)
    out[:, 1] = _long(span_p, cnt_p, cfg.stuck_hours[1], c)
    out[:, 2] = _long(span_r, cnt_r, cfg.stuck_hours[2], c)
    out[:, 2] |= saturation_flags(x[:, 2], base[:, 2], cfg.rh_sat, cfg.rh_sat_hours, g, c,
                                  causal)
    return out


def station_pressure_expected(elev):
    """Давление стандартной атмосферы на высоте elev, гПа."""
    return P_SEA_LEVEL * (1.0 - float(elev) / 44330.0) ** 5.255


def sea_level_pressure_flags(P, ok_raw, elev, cfg=DEFAULT_QC, causal=False):
    """Давление, приведённое к уровню моря, вместо станционного.

    Args:
        P: давление, форма (N,).
        ok_raw: значение есть, форма (N,).
        elev: высота станции, м; None - проверка не выполняется.
        cfg: пороги.
        causal: причинный режим.

    Returns:
        Булев массив (N,).
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
    before, after = window_span(cfg.slp_half, causal)
    with np.errstate(invalid="ignore"):
        above = ok_raw & (P > p_exp + sep / 2)
    n_above = _window_count(above, before, after)
    n_valid = _window_count(ok_raw, before, after)
    cand = np.flatnonzero(ok_raw & (n_valid >= cfg.slp_min_valid) & (2 * n_above >= n_valid))
    if cand.size == 0:
        return out
    r = rolling_median(P, ok_raw, before, cfg.slp_min_valid, at=cand, after=after)
    out[cand[r > p_exp + sep / 2]] = True
    return out


def dewpoint_flags(T, Td, ok_T, cfg=DEFAULT_QC):
    """Точка росы выше температуры больше допуска. Флаг идёт на канал RH."""
    T = np.asarray(T, np.float64)
    Td = np.asarray(Td, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(ok_T, bool) & np.isfinite(Td) & (Td > T + cfg.dewpoint_tol)


def _spike_flags(x, base, cfg, causal=False):
    """Выбросы по всем каналам одним проходом скользящей медианы.

    Args:
        x: значения, форма (N, 3).
        base: отчёты, прошедшие проверку диапазона, форма (N, 3).
        cfg: пороги.
        causal: причинный режим.

    Returns:
        Булев массив (N, 3).
    """
    n, c = x.shape
    before, after = window_span(cfg.spike_half, causal)
    sep = max(before, after)
    x64 = np.asarray(x, np.float64)
    xs = np.full((c, n + sep), np.nan)
    vs = np.zeros((c, n + sep), bool)
    fl = np.zeros((c, n + sep))
    xs[:, :n], vs[:, :n] = x64.T, base.T
    fl[:] = np.asarray(cfg.scale_floor, np.float64)[:, None]
    flat = slice(None, -sep) if sep else slice(None)
    xf, vf, ff = xs.ravel()[flat], vs.ravel()[flat], fl.ravel()[flat]
    med, mad, cnt = rolling_median_mad(xf, vf, before, after=after)
    bad = np.abs(xf - med) > cfg.spike_thresh * np.maximum(MAD_TO_SD * mad, ff)
    ok = ~((cnt >= cfg.spike_min_valid) & bad)
    ok = np.r_[ok, np.ones(sep, bool)].reshape(c, n + sep)[:, :n].T
    return base & ~ok


def check_codes(x, src, elev=None, Td=None, cfg=DEFAULT_QC, causal=False):
    """Поточечные и оконные проверки без кодов MISSING и SOURCE.

    Args:
        x: значения, форма (N, 3).
        src: значение есть и не помечено источником, форма (N, 3).
        elev: высота станции для проверки давления, м; None - проверка пропускается.
        Td: точка росы, форма (N,), если источник даёт её отдельно.
        cfg: пороги.
        causal: причинный режим окон.

    Returns:
        Коды uint8, форма (N, 3).
    """
    x = np.asarray(x)
    src = np.asarray(src, bool)
    n = x.shape[0]
    codes = np.zeros((n, 3), np.uint8)
    if n == 0:
        return codes
    lo = np.array([PHYS[c][0] for c in CHANNELS], x.dtype)
    hi = np.array([PHYS[c][1] for c in CHANNELS], x.dtype)
    with np.errstate(invalid="ignore"):
        phys = (x >= lo) & (x <= hi)
    codes[src & ~phys] |= np.uint8(QCCode.RANGE)
    base = src & phys
    spikes = _spike_flags(x, base, cfg, causal)
    codes[spikes] |= np.uint8(QCCode.SPIKE)
    for j in range(3):
        jump, exc = jump_flags(x[:, j], base[:, j] & ~spikes[:, j], cfg.jump_half,
                               cfg.jump_thresh, cfg.jump_min_valid, cfg.jump_floor[j],
                               cfg.excursion_max_hours, cfg.jump_max_gap, cfg.scale_floor[j],
                               causal)
        codes[jump, j] |= np.uint8(QCCode.JUMP)
        codes[exc, j] |= np.uint8(QCCode.SPIKE)
    codes[stuck_codes(x, base, cfg, causal)] |= np.uint8(QCCode.STUCK)
    codes[sea_level_pressure_flags(x[:, 1], src[:, 1], elev, cfg, causal), 1] |= \
        np.uint8(QCCode.UNITS)
    if Td is not None:
        codes[dewpoint_flags(x[:, 0], Td, base[:, 0], cfg), 2] |= np.uint8(QCCode.DEWPOINT)
    return codes


def causal_codes(x, present, elev=None, cfg=DEFAULT_QC, start=0):
    """Причинный QC ряда: то, что увидит устройство.

    Args:
        x: сырые значения, форма (N, 3).
        present: маска наличия значения, форма (N, 3).
        elev: высота станции для проверки давления, м.
        cfg: пороги.
        start: первый час, для которого нужны коды; часы до него служат контекстом.

    Returns:
        Коды uint8 часов с start до конца, форма (N - start, 3), вместе с MISSING.
    """
    x = np.asarray(x, np.float32)
    src = (np.asarray(present) > 0) & np.isfinite(x)
    start = int(start)
    lo = max(0, start - cfg.lookback_hours)
    codes = check_codes(x[lo:], src[lo:], elev=elev, cfg=cfg, causal=True)[start - lo:]
    codes[~src[start:]] |= np.uint8(QCCode.MISSING)
    return codes


def _as_channel_valid(valid, n):
    """Маска источника формы (N,) или (N, 3) в виде булевой (N, 3)."""
    v = np.asarray(valid)
    if v.ndim == 1:
        v = np.repeat(v[:, None], len(CHANNELS), axis=1)
    if v.shape != (n, len(CHANNELS)):
        raise ValueError(f"маска источника формы {v.shape}, ожидалось ({n},) или ({n}, 3)")
    return v > 0


def presence(x, valid):
    """Маска наличия значения от источника.

    Args:
        x: значения, форма (N, 3).
        valid: маска источника, форма (N,) или (N, 3).

    Returns:
        Маска uint8, форма (N, 3): значение есть и конечно.
    """
    x = np.asarray(x)
    return (_as_channel_valid(valid, x.shape[0]) & np.isfinite(x)).astype(np.uint8)


def qc_station(T, P, RH, valid, *, Td=None, flag=None, elev=None, cfg=DEFAULT_QC):
    """Полный центрированный QC ряда станции.

    Args:
        T: температура, форма (N,).
        P: давление, форма (N,).
        RH: влажность, форма (N,).
        valid: маска наличия от источника, форма (N,) или (N, 3).
        Td: точка росы, форма (N,), если источник даёт её отдельно.
        flag: штатные флаги источника «подозрительно», форма (N,) или (N, 3).
        elev: высота станции, м.
        cfg: пороги.

    Returns:
        Тройка: значения float32 (N, 3), маска uint8 (N, 3), коды uint8 (N, 3).
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


def qc_window(x, mask, elev=None, cfg=DEFAULT_QC, past=None):
    """Причинный QC истории окна: тот же, что на устройстве.

    Args:
        x: сырые значения истории, форма (N, 3).
        mask: маска наличия, форма (N, 3).
        elev: высота станции для проверки давления, м.
        cfg: пороги.
        past: пара массивов значений и маски наличия часов прямо перед историей,
            форма (K, 3) каждый; служат только контекстом. None - контекста нет.

    Returns:
        Пара: маска float32 (N, 3) и коды uint8 (N, 3). Значения не меняются.
    """
    x = np.asarray(x, np.float32)
    m = np.asarray(mask)
    k = 0
    if past is not None:
        xp, mp = past
        k = len(xp)
        x = np.concatenate([np.asarray(xp, np.float32), x])
        m = np.concatenate([np.asarray(mp), m])
    codes = causal_codes(x, m, elev=elev, cfg=cfg, start=k)
    return (codes == 0).astype(np.float32), codes


class CausalQC:
    """Причинный QC потока: кольцо сырых часов и коды каждого нового часа.

    Attributes:
        size: длина кольца, ч.
        filled: сколько часов кольца заполнено.
    """

    def __init__(self, elev=None, cfg=DEFAULT_QC):
        self.cfg = cfg
        self.elev = None if elev is None else float(elev)
        self.size = cfg.lookback_hours + 1
        self.reset()

    def reset(self):
        """Кольцо пусто: устройство ничего не знает о прошлом."""
        self.x = np.zeros((self.size, 3), np.float32)
        self.present = np.zeros((self.size, 3), np.uint8)
        self.filled = 0

    def seed(self, x, present):
        """Заполняет кольцо прошлыми часами без расчёта кодов.

        Args:
            x: значения, форма (K, 3), от старых к новым.
            present: маска наличия, форма (K, 3).
        """
        self.reset()
        x = np.asarray(x, np.float32)[-self.size:]
        p = np.asarray(present)[-self.size:]
        k = len(x)
        if k:
            self.x[-k:] = np.where(p > 0, x, 0.0)
            self.present[-k:] = p > 0
        self.filled = k

    def push(self, values):
        """Новый час наблюдений.

        Имеющееся значение сразу записывается так, как его пишет прибор: целые градусы и
        проценты, давление в десятых. Проверки видят уже записанное значение.

        Args:
            values: три значения T, P, RH; None или NaN - значения нет.

        Returns:
            Пара: значения float32 (3,) с нулями на месте отбракованных и коды uint8 (3,).
        """
        v = np.array([np.nan if a is None else float(a) for a in values], np.float32)
        p = np.isfinite(v)
        v = np.where(p, record_values(np.where(p, v, 0.0)), v)
        self.x[:-1], self.present[:-1] = self.x[1:], self.present[1:]
        self.x[-1] = np.where(p, v, 0.0)
        self.present[-1] = p
        self.filled = min(self.size, self.filled + 1)
        n = self.filled
        codes = causal_codes(self.x[-n:], self.present[-n:], self.elev, self.cfg,
                             start=n - 1)[0]
        return np.where(codes == 0, self.x[-1], 0.0).astype(np.float32), codes


def code_fractions(codes, mask=None):
    """Доли часов с каждым кодом по каналам и доля валидных, если дана маска."""
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

    Ловит грубые ошибки уровня источника: ряд в чужих единицах или со сдвигом шкалы.
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
