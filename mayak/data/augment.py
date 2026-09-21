"""Аугментации, имитирующие реальный прибор.

Обучение идёт на реанализе - интерполированном поле без пропусков, квантования,
дрейфа, залипаний, ошибок единиц и нерегулярной отчётности. Здесь всё это
появляется искусственно, чтобы перенос на наблюдения реальной сети состоялся.

Контракт:
* Вход и выход - ``AugWindow``: история (L_MAX, 3), выровненная по правому краю
  (валидная часть - последние ``L`` часов), её маска, цель (H,) и её маска,
  метаданные станции.
* Искажения значений действуют только на валидные точки; после всех
  аугментаций вызывающий код восстанавливает инвариант ``enforce_invariant``
  и прогоняет QC окна.
* Цель не трогается ничем, кроме постоянного смещения температуры. Маска цели не меняется никогда.
* Каждая аугментация берёт случайные числа из своего подпотока, выведенного из
  одного числа основного генератора аугментаций. Отсюда два свойства: окно
  потребляет из ``rng`` ровно одно число, а включение, выключение или смена
  параметров одной аугментации не сдвигает случайные числа остальных - абляция
  одной аугментации не меняет прочие.

Порядок применения (физическая цепочка «истинное значение → прибор → запись →
передача и архив»):

1. ``coords``   - ошибка метаданных станции;
2. ``scale``, ``drift``, ``offset``, ``noise`` - свойства датчика; затем RH
   ограничивается [0, 100] (датчик насыщается);
3. ``spike``, ``stuck``, ``units`` - грубые ошибки записи;
4. ``quantize`` - округление записанного значения;
5. ``dropout``, ``gap``, ``sparse``, ``outage``, ``drop_pressure``,
   ``drop_humidity`` - доступность: только маска.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mayak.config import AUGMENT_PROB_FIELDS, AugmentConfig
from mayak.constants import H, L_MAX
from mayak.data.masking import enforce_invariant
from mayak.data.qc import QC_CODES, P_SEA_LEVEL, QCCode, qc_window, station_pressure_expected

T, P, RH = 0, 1, 2

AUG_ORDER = ("coords", "scale", "drift", "offset", "noise", "spike", "stuck", "units",
             "quantize", "dropout", "gap", "sparse", "outage", "drop_pressure",
             "drop_humidity")
_STREAM_ID = {"coords": 1, "scale": 2, "drift": 3, "offset": 4, "noise": 5, "spike": 6,
              "stuck": 7, "units": 8, "quantize": 9, "dropout": 10, "gap": 11,
              "sparse": 12, "outage": 13, "drop_pressure": 14, "drop_humidity": 15}
assert set(AUG_ORDER) == set(_STREAM_ID) == set(AUGMENT_PROB_FIELDS)

AUG_KIND = {
    "coords": "metadata",
    "scale": "instrument", "drift": "instrument", "offset": "instrument",
    "noise": "instrument", "quantize": "instrument",
    "spike": "gross", "stuck": "gross", "units": "gross",
    "dropout": "availability", "gap": "availability", "sparse": "availability",
    "outage": "availability", "drop_pressure": "availability", "drop_humidity": "availability",
}

EXPECTED_QC = {
    "coords": set(), "scale": set(), "drift": set(), "offset": set(), "noise": set(),
    "quantize": set(),
    "spike": {QCCode.SPIKE, QCCode.RANGE},
    "stuck": {QCCode.STUCK},
    "units": {QCCode.UNITS},
    "dropout": {QCCode.MISSING}, "gap": {QCCode.MISSING}, "sparse": {QCCode.MISSING},
    "outage": {QCCode.MISSING}, "drop_pressure": {QCCode.MISSING},
    "drop_humidity": {QCCode.MISSING},
}
SIDE_QC = {name: set() for name in EXPECTED_QC}
SIDE_QC["spike"] = {QCCode.JUMP}
SIDE_QC["units"] = {QCCode.RANGE, QCCode.JUMP, QCCode.SPIKE}


@dataclass
class AugWindow:
    """Окно в процессе аугментации. Массивы изменяются на месте."""
    x: np.ndarray                  # (L_MAX, 3) float32, история
    m: np.ndarray                  # (L_MAX, 3) float32, маска истории
    y: np.ndarray                  # (H,) float32, цель
    y_mask: np.ndarray             # (H,) float32, маска цели - только читается
    L: int                         # фактическая длина истории
    hour: np.ndarray               # (L_MAX,) час UTC каждого часа истории
    lat: float
    lon: float
    elev: float
    qc_elev: Optional[float] = None
    applied: dict = field(default_factory=dict)

    @property
    def h0(self):
        """Индекс первого часа истории."""
        return L_MAX - self.L

    def valid_rows(self, ch):
        return self.h0 + np.flatnonzero(self.m[self.h0:, ch] > 0)


def aug_streams(key):
    """Подпотоки аугментаций от одного 63-битного ключа окна."""
    return {n: np.random.default_rng([int(key), _STREAM_ID[n]]) for n in AUG_ORDER}


def _sym(r, a):
    return float(r.uniform(-a, a)) if a > 0 else 0.0


def _signed_mag(r, lo, hi):
    """|v| log-равномерно в [lo, hi] (равномерно в [0, hi] при lo = 0), знак случаен."""
    if hi <= 0:
        return 0.0
    mag = float(r.uniform(0.0, hi)) if lo <= 0 else float(np.exp(r.uniform(np.log(lo),
                                                                            np.log(hi))))
    return mag if r.random() < 0.5 else -mag


def _rand_channel(r, w, need=1):
    """Случайный канал, у которого в истории не меньше need валидных часов."""
    ch = [c for c in range(3) if len(w.valid_rows(c)) >= need]
    return int(ch[r.integers(len(ch))]) if ch else None


def _s_coords(r, c, w):
    return dict(dlat=_sym(r, c.coord_jitter_deg), dlon=_sym(r, c.coord_jitter_deg),
                delev=float(r.normal(0.0, c.elev_jitter_m)) if c.elev_jitter_m > 0 else 0.0)


def _s_scale(r, c, w):
    return dict(k=[1.0 + _sym(r, a) for a in c.scale_max])


def _s_drift(r, c, w):
    return dict(amp=[_sym(r, a) for a in c.drift_max],
                walk=bool(r.random() < c.drift_rw_frac), seed=int(r.integers(2 ** 31)))


def _s_offset(r, c, w):
    if w.L == 0 or c.offset_max <= 0:
        return None
    return dict(b=_signed_mag(r, c.offset_min, c.offset_max))


def _s_noise(r, c, w):
    return dict(sd=list(c.noise_sd), seed=int(r.integers(2 ** 31)))


def _s_spike(r, c, w):
    out = []
    for _ in range(int(r.integers(1, c.spike_max_count + 1))):
        ch = _rand_channel(r, w)
        if ch is None:
            break
        rows = w.valid_rows(ch)
        out.append(dict(ch=ch, i=int(rows[r.integers(len(rows))]),
                        d=_signed_mag(r, c.spike_min[ch], c.spike_max[ch])))
    return dict(spikes=out) if out else None


def _s_stuck(r, c, w):
    ch = _rand_channel(r, w, need=2)
    if ch is None:
        return None
    n = int(r.integers(c.stuck_hours[0], c.stuck_hours[1] + 1))
    rows = w.valid_rows(ch)
    return dict(ch=ch, i=int(rows[r.integers(len(rows))]), n=n)


def _s_units(r, c, w):
    ch = P if r.random() < c.units_p_frac else T
    if len(w.valid_rows(ch)) == 0:
        return None
    n = min(w.L, int(r.integers(c.units_hours[0], c.units_hours[1] + 1)))
    i = int(w.h0 + r.integers(0, w.L - n + 1))
    return dict(ch=ch, i=i, n=n)


def _s_quantize(r, c, w):
    return dict(fahrenheit=bool(r.random() < c.quant_f_frac))


def _s_dropout(r, c, w):
    return dict(rate=float(r.uniform(0.0, c.dropout_max_rate)), seed=int(r.integers(2 ** 31)))


def _s_gap(r, c, w):
    gaps = []
    for _ in range(int(r.integers(1, c.gap_max_count + 1))):
        n = int(r.integers(1, c.gap_max_len + 1))
        gaps.append(dict(i=int(r.integers(w.h0, L_MAX)), n=n))
    return dict(gaps=gaps)


def _s_sparse(r, c, w):
    k = int(c.sparse_every[r.integers(len(c.sparse_every))])
    return dict(every=k, phase=int(r.integers(k)))


def _s_outage(r, c, w):
    n = int(r.integers(c.outage_hours[0], c.outage_hours[1] + 1))
    if w.L < n + 2:
        return None
    ch = int(r.integers(3))
    return dict(ch=ch, i=int(w.h0 + r.integers(1, w.L - n)), n=n)


def _s_drop(r, c, w):
    return {}


def _a_coords(w, p):
    w.lat = float(np.clip(w.lat + p["dlat"], -90.0, 90.0))
    w.lon = float((w.lon + p["dlon"] + 180.0) % 360.0 - 180.0)
    w.elev = float(w.elev + p["delev"])
    if w.qc_elev is not None:
        w.qc_elev = float(w.qc_elev + p["delev"])


def _a_scale(w, p):
    w.x[:] = np.where(w.m > 0, w.x * np.asarray(p["k"], np.float32), w.x)


def drift_profile(L, amp, walk, seed):
    """Дрейф на L часов истории: 0 в последнем часе, ±amp (по модулю, в среднем) в первом."""
    if L == 0 or amp == 0:
        return np.zeros(L, np.float32)
    if not walk or L < 2:
        return (amp * (L - 1 - np.arange(L)) / max(L - 1, 1)).astype(np.float32)
    steps = np.random.default_rng(seed).standard_normal(L - 1) * abs(amp) / np.sqrt(L - 1)
    back = np.concatenate([np.cumsum(steps[::-1])[::-1], [0.0]])
    return back.astype(np.float32)


def _a_drift(w, p):
    for ch, a in enumerate(p["amp"]):
        d = drift_profile(w.L, a, p["walk"], p["seed"] + ch)
        w.x[w.h0:, ch] += d * (w.m[w.h0:, ch] > 0)


def _a_offset(w, p):
    b = np.float32(p["b"])
    w.x[:, T] += b * (w.m[:, T] > 0)
    w.y[:] = w.y + b * (w.y_mask > 0)


def _a_noise(w, p):
    g = np.random.default_rng(p["seed"])
    e = g.standard_normal((L_MAX, 3)).astype(np.float32) * np.asarray(p["sd"], np.float32)
    w.x[:] = w.x + e * (w.m > 0)


def _a_spike(w, p):
    for s in p["spikes"]:
        w.x[s["i"], s["ch"]] += np.float32(s["d"])


def _a_stuck(w, p):
    ch, i = p["ch"], p["i"]
    j = min(L_MAX, i + p["n"])
    v = w.x[i, ch]
    seg = w.m[i:j, ch] > 0
    w.x[i:j, ch] = np.where(seg, v, w.x[i:j, ch])


def sea_level_ratio(elev):
    """Множитель «станционное → приведённое к уровню моря» стандартной атмосферы."""
    return P_SEA_LEVEL / station_pressure_expected(max(0.0, float(elev or 0.0)))


def _a_units(w, p):
    ch, i, j = p["ch"], p["i"], p["i"] + p["n"]
    seg = w.m[i:j, ch] > 0
    v = w.x[i:j, ch]
    if ch == T:
        conv = v * np.float32(1.8) + np.float32(32.0)
    else:
        conv = v * np.float32(sea_level_ratio(w.qc_elev if w.qc_elev is not None else w.elev))
    w.x[i:j, ch] = np.where(seg, conv, v)


def quantize_T(v, fahrenheit):
    """Округление записи температуры: целые °F либо шаг 0.1 °C."""
    v = np.asarray(v, np.float64)
    if fahrenheit:
        return ((np.round(v * 1.8 + 32.0) - 32.0) / 1.8).astype(np.float32)
    return (np.round(v * 10.0) / 10.0).astype(np.float32)


def _a_quantize(w, p):
    ok = w.m[:, T] > 0
    w.x[:, T] = np.where(ok, quantize_T(w.x[:, T], p["fahrenheit"]), w.x[:, T])


def _a_dropout(w, p):
    g = np.random.default_rng(p["seed"])
    drop = g.random(w.L) < p["rate"]
    w.m[w.h0:][drop] = 0.0


def _a_gap(w, p):
    for gp in p["gaps"]:
        w.m[gp["i"]:min(L_MAX, gp["i"] + gp["n"])] = 0.0


def _a_sparse(w, p):
    keep = (np.asarray(w.hour).astype(np.int64) % p["every"]) == p["phase"]
    w.m[~keep] = 0.0


def _a_outage(w, p):
    w.m[p["i"]:p["i"] + p["n"], p["ch"]] = 0.0


def _a_drop_pressure(w, p):
    w.m[:, P] = 0.0


def _a_drop_humidity(w, p):
    w.m[:, RH] = 0.0


_SAMPLE = {"coords": _s_coords, "scale": _s_scale, "drift": _s_drift, "offset": _s_offset,
           "noise": _s_noise, "spike": _s_spike, "stuck": _s_stuck, "units": _s_units,
           "quantize": _s_quantize, "dropout": _s_dropout, "gap": _s_gap,
           "sparse": _s_sparse, "outage": _s_outage, "drop_pressure": _s_drop,
           "drop_humidity": _s_drop}
_APPLY = {"coords": _a_coords, "scale": _a_scale, "drift": _a_drift, "offset": _a_offset,
          "noise": _a_noise, "spike": _a_spike, "stuck": _a_stuck, "units": _a_units,
          "quantize": _a_quantize, "dropout": _a_dropout, "gap": _a_gap,
          "sparse": _a_sparse, "outage": _a_outage, "drop_pressure": _a_drop_pressure,
          "drop_humidity": _a_drop_humidity}
_NEEDS_HISTORY = frozenset(AUG_ORDER) - {"coords"}


def apply_one(w, name, params):
    """Применить аугментацию name с заданными параметрами (эталонные окна, тесты)."""
    _APPLY[name](w, params)
    w.applied[name] = params
    return w


def augment_window(w: AugWindow, cfg: AugmentConfig, rng) -> AugWindow:
    """Все аугментации профиля cfg к окну w. Берёт из rng ровно одно число."""
    streams = aug_streams(rng.integers(2 ** 63))
    for name in AUG_ORDER:
        r = streams[name]
        fire = r.random() < getattr(cfg, AUGMENT_PROB_FIELDS[name])
        if not fire or (w.L == 0 and name in _NEEDS_HISTORY):
            continue
        params = _SAMPLE[name](r, cfg, w)
        if params is None:
            continue
        apply_one(w, name, params)
        if name == "noise":
            ok = w.m[:, RH] > 0
            w.x[:, RH] = np.where(ok, np.clip(w.x[:, RH], 0.0, 100.0), w.x[:, RH])
    return w


def clean_history(n=L_MAX, lat=45.0, lon=10.0, elev=200.0, seed=0, t0_hour=0):
    """Правдоподобная чистая история: годовой и суточный ход по солнечному времени,
    синоптический AR(1)-шум, давление по высоте. QC окна на ней ничего не находит.

    Возвращает (x float32 (n, 3), hour (n,) час UTC).
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    hour = (t0_hour + k) % 24
    doy = ((t0_hour + k) // 24 + 150) % 365
    t_sol = hour + lon / 15.0 - 2.5
    season = (8 + 0.2 * abs(lat)) * np.cos(2 * np.pi * (doy - 200) / 365.24)
    diurnal = 5.0 * np.cos(2 * np.pi * (t_sol - 12) / 24)
    rho = np.exp(-1 / 60)
    e = rng.standard_normal(n) * 2.0 * np.sqrt(1 - rho ** 2)
    syn = np.zeros(n)
    for i in range(1, n):
        syn[i] = rho * syn[i - 1] + e[i]
    T_ = 12 + season + diurnal + syn + 0.2 * rng.standard_normal(n)
    P_ = station_pressure_expected(elev) + 3 * syn + 0.3 * rng.standard_normal(n)
    RH_ = np.clip(65 - 2 * diurnal + 3 * rng.standard_normal(n), 5, 99)
    return np.stack([T_, P_, RH_], -1).astype(np.float32), hour.astype(np.float32)


def make_window(x, hour, L=L_MAX, lat=45.0, lon=10.0, elev=200.0, y=None):
    """AugWindow из готовой истории x (L_MAX, 3); цель по умолчанию - константа 10 °C."""
    m = np.zeros((L_MAX, 3), np.float32)
    m[L_MAX - L:] = 1.0
    xx = np.where(m > 0, x, 0.0).astype(np.float32)
    y = np.full(H, 10.0, np.float32) if y is None else np.asarray(y, np.float32).copy()
    return AugWindow(x=xx, m=m, y=y, y_mask=np.ones(H, np.float32), L=L,
                     hour=np.asarray(hour, np.float32), lat=lat, lon=lon, elev=elev,
                     qc_elev=elev)


_H0 = L_MAX - 24 * 28
REFERENCE_CASES = (
    ("spike_T", "spike", dict(spikes=[dict(ch=T, i=400, d=20.0)]), 200.0, T, (400, 401)),
    ("spike_P", "spike", dict(spikes=[dict(ch=P, i=410, d=-25.0)]), 200.0, P, (410, 411)),
    ("stuck_T_36h", "stuck", dict(ch=T, i=300, n=36), 200.0, T, (300, 336)),
    ("stuck_RH_48h", "stuck", dict(ch=RH, i=200, n=48), 200.0, RH, (200, 248)),
    ("units_T_F", "units", dict(ch=T, i=500, n=48), 200.0, T, (500, 548)),
    ("units_P_slp", "units", dict(ch=P, i=250, n=72), 1500.0, P, (250, 322)),
    ("dropout", "dropout", dict(rate=0.2, seed=3), 200.0, None, None),
    ("gap_3d", "gap", dict(gaps=[dict(i=100, n=72)]), 200.0, None, (100, 172)),
    ("sparse_3h", "sparse", dict(every=3, phase=0), 200.0, None, None),
    ("sparse_6h", "sparse", dict(every=6, phase=0), 200.0, None, None),
    ("outage_RH", "outage", dict(ch=RH, i=150, n=120), 200.0, RH, (150, 270)),
    ("drop_pressure", "drop_pressure", {}, 200.0, P, (0, L_MAX)),
    ("drop_humidity", "drop_humidity", {}, 200.0, RH, (0, L_MAX)),
    ("scale", "scale", dict(k=[1.03, 1.0005, 1.05]), 200.0, None, None),
    ("drift_linear", "drift", dict(amp=[2.0, 1.5, 6.0], walk=False, seed=0), 200.0, None, None),
    ("drift_walk", "drift", dict(amp=[2.0, 1.5, 6.0], walk=True, seed=1), 200.0, None, None),
    ("offset", "offset", dict(b=3.0), 200.0, None, None),
    ("noise", "noise", dict(sd=[0.2, 0.3, 2.0], seed=5), 200.0, None, None),
    ("quantize_0.1C", "quantize", dict(fahrenheit=False), 200.0, None, None),
    ("quantize_F", "quantize", dict(fahrenheit=True), 200.0, None, None),
    ("coords", "coords", dict(dlat=0.3, dlon=-0.3, delev=40.0), 200.0, None, None),
)


def reference_windows(seed=0):
    """Эталонные окна: чистое окно и то же окно с каждой аугментацией отдельно."""
    out = []
    for case, name, params, elev, ch, rows in REFERENCE_CASES:
        x, hour = clean_history(lat=45.0, lon=10.0, elev=elev, seed=seed)
        before = make_window(x, hour, elev=elev)
        after = make_window(x, hour, elev=elev)
        apply_one(after, name, dict(params))
        out.append((case, name, before, after, ch, rows))
    return out


def _bits(codes):
    out = 0
    for c in codes:
        out |= int(c)
    return np.uint8(out)


def window_codes(w):
    """Коды QC окна после восстановления инварианта - то, что увидит модель."""
    x, m = enforce_invariant(w.x, w.m)
    return qc_window(x, m, elev=w.qc_elev)[1]


def qc_effect(name, before, after, ch=None, rows=None):
    """Что аугментация name сделала с QC окна.

    hit       - на затронутых часах (rows × ch; None - все) появился хотя бы один код
                из EXPECTED_QC[name]; для «невидимых» аугментаций - None;
    hit_frac  - доля затронутых часов с ожидаемым кодом;
    side_frac - доля часов, валидных до аугментации, с новым кодом вне ожидаемых,
                допустимых побочных (SIDE_QC) и MISSING;
    new       - {код: число часов} новых кодов.
    """
    cb, ca = window_codes(before), window_codes(after)
    new = ca & ~cb
    exp = _bits(EXPECTED_QC[name])
    allowed = exp | _bits(SIDE_QC[name]) | np.uint8(QCCode.MISSING)
    r = slice(None) if rows is None else slice(*rows)
    c = slice(None) if ch is None else ch
    reg = new[r, c]
    hit = bool(((reg & exp) > 0).any()) if exp else None
    hit_frac = float(((reg & exp) > 0).mean()) if exp else None
    valid = before.m > 0
    side = ((new & ~allowed) > 0) & valid
    return dict(hit=hit, hit_frac=hit_frac,
                side_frac=float(side.sum() / max(valid.sum(), 1)),
                new={q.name: int(((new & q) > 0).sum()) for q in QC_CODES
                     if ((new & q) > 0).any()})


__all__ = ["AUG_KIND", "AUG_ORDER", "AugWindow", "EXPECTED_QC", "REFERENCE_CASES", "SIDE_QC",
           "apply_one", "aug_streams", "augment_window", "clean_history", "drift_profile",
           "make_window", "qc_effect", "quantize_T", "reference_windows", "sea_level_ratio",
           "window_codes"]
