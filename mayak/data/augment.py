"""Аугментации обучающих окон, имитирующие реальный прибор."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mayak.astro import dewpoint_from_rh, rh_from_dewpoint
from mayak.config import AUGMENT_PROB_FIELDS, AugmentConfig
from mayak.constants import H, L_MAX
from mayak.data.masking import enforce_invariant
from mayak.data.qc import QC_CODES, QCCode, qc_window, station_pressure_expected
from mayak.data.recording import RECORD_SCALE, record_channel, record_values, round_half_even

T, P, RH = 0, 1, 2

AUG_ORDER = ("offset", "noise", "rh_dewpoint", "stuck", "dropout", "gap", "drop_channel")

_STREAM_ID = {"offset": 4, "noise": 5, "stuck": 7, "dropout": 10, "gap": 11,
              "rh_dewpoint": 16, "drop_channel": 18}
assert set(AUG_ORDER) == set(_STREAM_ID) == set(AUGMENT_PROB_FIELDS)

AUG_KIND = {
    "offset": "instrument", "noise": "instrument", "rh_dewpoint": "record",
    "stuck": "gross",
    "dropout": "availability", "gap": "availability", "drop_channel": "availability",
}

EXPECTED_QC = {
    "offset": set(), "noise": set(), "rh_dewpoint": set(),
    "stuck": {QCCode.STUCK},
    "dropout": {QCCode.MISSING}, "gap": {QCCode.MISSING}, "drop_channel": {QCCode.MISSING},
}

# Варианты ``drop_channel``: каналы, которых нет на всей истории.
DROP_CHANNEL_VARIANTS = ((P,), (RH,), (P, RH))


def station_offset(hour, lon, b, amp=0.0, peak=0.0):
    """Смещение станции в заданные часы."""
    hour = np.asarray(hour, np.float64)
    out = np.full(hour.shape, float(b))
    if amp:
        solar = hour + float(lon) / 15.0
        out += float(amp) * np.cos(2.0 * np.pi * (solar - float(peak)) / 24.0)
    return out.astype(np.float32)


@dataclass
class AugWindow:
    """Окно в процессе аугментации. Массивы изменяются на месте."""
    x: np.ndarray                  # (L_MAX, 3) float32, история
    m: np.ndarray                  # (L_MAX, 3) float32, маска истории
    y: np.ndarray                  # (H,) float32, цель
    y_mask: np.ndarray             # (H,) float32, маска цели - только читается
    L: int                         # фактическая длина истории
    hour: np.ndarray               # (L_MAX,) час UTC каждого часа истории
    hour_fut: np.ndarray           # (H,) час UTC каждого часа горизонта
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

    def add_station_offset(self, b, amp=0.0, peak=0.0, target=True):
        """Прибавить смещение станции к температуре истории и цели."""
        d = station_offset(self.hour, self.lon, b, amp, peak)
        self.x[:, T] += d * (self.m[:, T] > 0)
        if target:
            dy = station_offset(self.hour_fut, self.lon, b, amp, peak)
            self.y[:] = self.y + dy * (self.y_mask > 0)
        return self


_DITHER_STREAM = 17
DITHER_BEFORE = frozenset({"offset", "noise", "rh_dewpoint"})
ON_TARGET = frozenset({"offset"})
DITHER_FRAC = 0.98


def aug_streams(key):
    """Подпотоки аугментаций от одного 63-битного ключа окна."""
    out = {n: np.random.default_rng([int(key), _STREAM_ID[n]]) for n in AUG_ORDER}
    out["dither"] = np.random.default_rng([int(key), _DITHER_STREAM])
    return out


def dither_window(w, rng, target=True):
    """Непрерывные значения на месте записанных прибором."""
    u, uy = _dither_noise(w, rng)
    _add_history(w, u)
    if target:
        _add_target(w, uy)
    return w


def _dither_noise(w, rng):
    half = DITHER_FRAC * 0.5 / np.asarray(RECORD_SCALE, np.float64)
    return (rng.uniform(-1.0, 1.0, w.x.shape) * half,
            rng.uniform(-1.0, 1.0, w.y.shape) * half[T])


def _add_history(w, u):
    w.x[:] = np.where(w.m > 0, w.x + u, w.x).astype(np.float32)


def _add_target(w, uy):
    w.y[:] = np.where(w.y_mask > 0, w.y + uy, w.y).astype(np.float32)


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


def _s_offset(r, c, w):
    b = _signed_mag(r, c.offset_min, c.offset_max)
    amp = _signed_mag(r, 0.0, c.offset_diurnal_max) if r.random() < c.offset_diurnal_frac \
        else 0.0
    if b == 0.0 and amp == 0.0:
        return None
    return dict(b=b, amp=amp, peak=float(r.uniform(*c.offset_peak_hours)))


def _s_noise(r, c, w):
    return dict(sd=list(c.noise_sd), seed=int(r.integers(2 ** 31)))


def _s_stuck(r, c, w):
    ch = _rand_channel(r, w, need=2)
    if ch is None:
        return None
    n = int(r.integers(c.stuck_hours[0], c.stuck_hours[1] + 1))
    rows = w.valid_rows(ch)
    return dict(ch=ch, i=int(rows[r.integers(len(rows))]), n=n)


def _s_dropout(r, c, w):
    return dict(rate=float(r.uniform(0.0, c.dropout_max_rate)), seed=int(r.integers(2 ** 31)))


def _s_gap(r, c, w):
    gaps = []
    for _ in range(int(r.integers(1, c.gap_max_count + 1))):
        n = int(r.integers(1, c.gap_max_len + 1))
        gaps.append(dict(i=int(r.integers(w.h0, L_MAX)), n=n))
    return dict(gaps=gaps)


def _s_drop_channel(r, c, w):
    p = np.asarray(c.drop_channel_weights, np.float64)
    return dict(ch=list(DROP_CHANNEL_VARIANTS[int(r.choice(len(p), p=p / p.sum()))]))


def _s_rh_dewpoint(r, c, w):
    both = (w.m[w.h0:, T] > 0) & (w.m[w.h0:, RH] > 0)
    return {} if both.any() else None


def _a_offset(w, p):
    w.add_station_offset(p["b"], p["amp"], p["peak"])


def _a_noise(w, p):
    g = np.random.default_rng(p["seed"])
    e = g.standard_normal((L_MAX, 3)).astype(np.float32) * np.asarray(p["sd"], np.float32)
    w.x[:] = w.x + e * (w.m > 0)


def _a_stuck(w, p):
    ch, i = p["ch"], p["i"]
    j = min(L_MAX, i + p["n"])
    v = w.x[i, ch]
    seg = w.m[i:j, ch] > 0
    w.x[i:j, ch] = np.where(seg, v, w.x[i:j, ch])


def rh_via_dewpoint(t, rh):
    """Влажность, восстановленная из целых температуры и точки росы."""
    t_rec = round_half_even(t)
    td_rec = round_half_even(dewpoint_from_rh(t, rh))
    return rh_from_dewpoint(t_rec, np.minimum(td_rec, t_rec))


def _a_rh_dewpoint(w, p):
    ok = (w.m[:, T] > 0) & (w.m[:, RH] > 0)
    if ok.any():
        w.x[ok, RH] = rh_via_dewpoint(w.x[ok, T], w.x[ok, RH])


def record_window(w):
    """Запись окна прибором: история и цель на сетке записи."""
    w.x[:] = np.where(w.m > 0, record_values(w.x), w.x)
    w.y[:] = np.where(w.y_mask > 0, record_channel(w.y, T), w.y)
    return w


def _a_dropout(w, p):
    g = np.random.default_rng(p["seed"])
    drop = g.random(w.L) < p["rate"]
    w.m[w.h0:][drop] = 0.0


def _a_gap(w, p):
    for gp in p["gaps"]:
        w.m[gp["i"]:min(L_MAX, gp["i"] + gp["n"])] = 0.0


def _a_drop_channel(w, p):
    w.m[:, list(p["ch"])] = 0.0


_SAMPLE = {"offset": _s_offset, "noise": _s_noise, "rh_dewpoint": _s_rh_dewpoint,
           "stuck": _s_stuck, "dropout": _s_dropout, "gap": _s_gap,
           "drop_channel": _s_drop_channel}
_APPLY = {"offset": _a_offset, "noise": _a_noise, "rh_dewpoint": _a_rh_dewpoint,
          "stuck": _a_stuck, "dropout": _a_dropout, "gap": _a_gap,
          "drop_channel": _a_drop_channel}


def apply_one(w, name, params):
    """Применить одну аугментацию с заданными параметрами."""
    _APPLY[name](w, params)
    w.applied[name] = params
    return w


def augment_window(w: AugWindow, cfg: AugmentConfig, rng) -> AugWindow:
    """Все аугментации профиля к окну. Из генератора берётся ровно одно число."""
    key = rng.integers(2 ** 63)
    streams = aug_streams(key)
    names = AUG_ORDER if w.L > 0 else [n for n in AUG_ORDER if n in ON_TARGET]
    noise, target_done = None, False
    for name in names:
        r = streams[name]
        if not r.random() < getattr(cfg, AUGMENT_PROB_FIELDS[name]):
            continue
        params = _SAMPLE[name](r, cfg, w)
        if params is None:
            continue
        if name in DITHER_BEFORE and noise is None:
            noise = _dither_noise(w, streams["dither"])
            _add_history(w, noise[0])
        if name in ON_TARGET and not target_done:
            _add_target(w, noise[1])
            target_done = True
        apply_one(w, name, params)
        if name == "noise":
            ok = w.m[:, RH] > 0
            w.x[:, RH] = np.where(ok, np.clip(w.x[:, RH], 0.0, 100.0), w.x[:, RH])
    return w


def clean_history(n=L_MAX, lat=45.0, lon=10.0, elev=200.0, seed=0, t0_hour=0):
    """Правдоподобная чистая история, на которой QC окна ничего не находит."""
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
    P_ = station_pressure_expected(elev) + syn + 0.15 * rng.standard_normal(n)
    RH_ = np.clip(65 - 2 * diurnal + 3 * rng.standard_normal(n), 5, 99)
    return record_values(np.stack([T_, P_, RH_], -1)), hour.astype(np.float32)


def make_window(x, hour, L=L_MAX, lat=45.0, lon=10.0, elev=200.0, y=None):
    """Окно аугментаций из готовой истории."""
    m = np.zeros((L_MAX, 3), np.float32)
    m[L_MAX - L:] = 1.0
    xx = np.where(m > 0, x, 0.0).astype(np.float32)
    y = np.full(H, 10.0, np.float32) if y is None else np.asarray(y, np.float32).copy()
    hour = np.asarray(hour, np.float32)
    hour_fut = ((hour[-1] + 1 + np.arange(H)) % 24).astype(np.float32)
    return AugWindow(x=xx, m=m, y=y, y_mask=np.ones(H, np.float32), L=L, hour=hour,
                     hour_fut=hour_fut, lat=lat, lon=lon, elev=elev, qc_elev=elev)


REFERENCE_CASES = (
    ("stuck_T_96h", "stuck", dict(ch=T, i=300, n=96), 200.0, T, (372, 396)),
    ("stuck_RH_48h", "stuck", dict(ch=RH, i=200, n=48), 200.0, RH, (223, 248)),
    ("dropout", "dropout", dict(rate=0.2, seed=3), 200.0, None, None),
    ("gap_3d", "gap", dict(gaps=[dict(i=100, n=72)]), 200.0, None, (100, 172)),
    ("drop_pressure", "drop_channel", dict(ch=[P]), 200.0, P, (0, L_MAX)),
    ("drop_humidity", "drop_channel", dict(ch=[RH]), 200.0, RH, (0, L_MAX)),
    ("drop_both", "drop_channel", dict(ch=[P, RH]), 200.0, [P, RH], (0, L_MAX)),
    ("offset", "offset", dict(b=2.0, amp=1.5, peak=14.0), 200.0, None, None),
    ("noise", "noise", dict(sd=[0.2, 0.3, 2.0], seed=5), 200.0, None, None),
    ("rh_dewpoint", "rh_dewpoint", {}, 200.0, None, None),
)


def reference_windows(seed=0):
    """Эталонные окна: чистое окно и то же окно с каждой аугментацией отдельно."""
    out = []
    for case, name, params, elev, ch, rows in REFERENCE_CASES:
        x, hour = clean_history(lat=45.0, lon=10.0, elev=elev, seed=seed)
        before = make_window(x, hour, elev=elev)
        after = make_window(x, hour, elev=elev)
        if name in DITHER_BEFORE:
            dither_window(after, np.random.default_rng([seed, _DITHER_STREAM]))
        record_window(apply_one(after, name, dict(params)))
        out.append((case, name, before, after, ch, rows))
    return out


def _bits(codes):
    out = 0
    for c in codes:
        out |= int(c)
    return np.uint8(out)


def window_codes(w):
    """Коды QC окна после записи прибором и восстановления инварианта - то, что увидит модель."""
    x, m = enforce_invariant(np.where(w.m > 0, record_values(w.x), w.x), w.m)
    return qc_window(x, m, elev=w.qc_elev)[1]


def qc_effect(name, before, after, ch=None, rows=None):
    """Что аугментация сделала с QC окна."""
    cb, ca = window_codes(before), window_codes(after)
    new = ca & ~cb
    exp = _bits(EXPECTED_QC[name])
    allowed = exp | np.uint8(QCCode.MISSING)
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


__all__ = ["AUG_KIND", "AUG_ORDER", "AugWindow", "DITHER_BEFORE", "DROP_CHANNEL_VARIANTS",
           "EXPECTED_QC", "REFERENCE_CASES", "apply_one", "aug_streams", "augment_window",
           "clean_history", "dither_window", "make_window", "qc_effect", "record_window",
           "reference_windows", "rh_via_dewpoint", "station_offset", "window_codes"]
