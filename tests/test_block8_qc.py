"""Тесты: единый контроль качества для обучения и для реальных наблюдений.

Один формат выхода и одни коды для любого источника;
Поточечные и оконные проверки — каждая порождает свой код на своём канале;
Станционные проверки: фаза суточного хода, разладка, высота, уровень T;
Штатные флаги источника уходят в маску, значение не исправляется;
Отчёт QC — артефакт сборки кэша;
Правила отбора станций;
Те же проверки поверх аугментированной истории обучающего окна.
"""
import csv
import dataclasses
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.constants import L_MAX
from mayak.data import qc as Q
from mayak.data import store as S
from mayak.data.qc import DEFAULT_QC, QCCode, QCConfig
from mayak.timeaxis import to_utc_hour

T0 = int(to_utc_hour(datetime(2015, 1, 1)))
YEAR = 8766


def physical_series(n, lat=45.0, lon=30.0, elev=200.0, seed=0, t0=T0, lag_h=2.5):
    """Правдоподобный почасовой ряд: годовой ход, суточный ход по местному солнечному
    времени с тепловым запаздыванием, синоптический AR(1)-шум, давление по высоте."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    abs_h = t0 + k
    hour = (abs_h % 24).astype(float)
    doy = ((abs_h // 24) % 365).astype(float)
    t_sol = hour + lon / 15.0 - lag_h
    season = (8 + 0.2 * abs(lat)) * np.cos(2 * np.pi * (doy - 200) / 365.24)
    diurnal = 5.0 * np.cos(2 * np.pi * (t_sol - 12) / 24)
    rho = math.exp(-1 / 60)
    e = rng.standard_normal(n) * 2.0 * math.sqrt(1 - rho ** 2)
    syn = np.empty(n)
    syn[0] = 0.0
    for i in range(1, n):
        syn[i] = rho * syn[i - 1] + e[i]
    T = 12 + season + diurnal + syn + 0.2 * rng.standard_normal(n)
    P = Q.station_pressure_expected(elev) + 3 * syn + 0.3 * rng.standard_normal(n)
    RH = np.clip(65 - 2 * diurnal + 3 * rng.standard_normal(n), 5, 99)
    f = lambda a: np.asarray(a, np.float32)
    return f(T), f(P), f(RH)


def run(T, P, RH, valid=None, **kw):
    valid = np.ones(len(T), np.uint8) if valid is None else valid
    return Q.qc_station(T, P, RH, valid, **kw)


def has(codes, idx, ch, code):
    return bool(np.all((codes[idx, ch] & code) > 0))


N = 24 * 60


@pytest.fixture(scope="module")
def clean():
    return physical_series(N, seed=1)


def test_output_contract(clean):
    x, mask, codes = run(*clean, elev=200.0)
    assert x.dtype == np.float32 and mask.dtype == np.uint8 and codes.dtype == np.uint8
    assert x.shape == mask.shape == codes.shape == (N, 3)
    assert np.array_equal(mask, (codes == 0).astype(np.uint8))
    assert np.all(x[mask == 0] == 0)


def test_codes_are_distinct_bits_and_fit_uint8():
    vals = [int(c) for c in Q.QC_CODES]
    assert len(set(vals)) == len(vals) == 8
    assert all(v & (v - 1) == 0 for v in vals) and max(vals) <= 255
    assert set(Q.QC_CODE_DOC) == {c.name for c in Q.QC_CODES}


def test_clean_realistic_series_is_not_flagged():
    for seed, (lat, lon, elev) in enumerate([(45, 30, 200), (-33, 150, 50), (60, -120, 900),
                                             (5, -60, 10), (70, 20, 1500)]):
        T, P, RH = physical_series(3 * 30 * 24, lat=lat, lon=lon, elev=elev, seed=seed)
        _, mask, codes = run(T, P, RH, elev=float(elev))
        for c in (QCCode.JUMP, QCCode.STUCK, QCCode.UNITS, QCCode.RANGE, QCCode.DEWPOINT):
            assert not np.any(codes & c), f"ложный {c.name} на чистом ряду {seed}"
        assert mask.mean() > 0.995


def _inject(kind, T, P, RH):
    T, P, RH = T.copy(), P.copy(), RH.copy()
    extra, ch, sl = {}, 0, None
    if kind == "range_T":
        sl, ch = slice(300, 301), 0
        T[sl] = 75.0
    elif kind == "range_RH":
        sl, ch = slice(310, 312), 2
        RH[sl] = 120.0
    elif kind == "spike":
        sl, ch = slice(400, 401), 0
        T[sl] += 25.0
    elif kind == "jump":
        sl, ch = slice(500, 501), 0
        T[500:] += 15.0
    elif kind == "stuck_T":
        sl, ch = slice(600, 680), 0
        T[sl] = T[600]
    elif kind == "stuck_P":
        sl, ch = slice(700, 730), 1
        P[sl] = P[700]
    elif kind == "stuck_RH":
        sl, ch = slice(800, 830), 2
        RH[sl] = RH[800]
    elif kind == "rh_saturated":
        sl, ch = slice(900, 990), 2
        RH[sl] = 100.0
    elif kind == "fahrenheit":
        sl, ch = slice(1000, 1072), 0
        T[sl] = T[sl] * 1.8 + 32.0
    elif kind == "sea_level_pressure":
        sl, ch = slice(0, len(P)), 1
        P[:] = P + (Q.P_SEA_LEVEL - Q.station_pressure_expected(1500.0))
        extra["elev"] = 1500.0
    elif kind == "dewpoint":
        sl, ch = slice(1100, 1103), 2
        Td = T - 5.0
        Td[sl] = T[sl] + 3.0
        extra["Td"] = Td
    return (T, P, RH), extra, ch, sl


CASES = {
    "range_T": QCCode.RANGE, "range_RH": QCCode.RANGE, "spike": QCCode.SPIKE,
    "jump": QCCode.JUMP, "stuck_T": QCCode.STUCK, "stuck_P": QCCode.STUCK,
    "stuck_RH": QCCode.STUCK, "rh_saturated": QCCode.STUCK, "fahrenheit": QCCode.UNITS,
    "sea_level_pressure": QCCode.UNITS, "dewpoint": QCCode.DEWPOINT,
}


@pytest.mark.parametrize("kind", list(CASES))
def test_each_artifact_yields_its_code_on_its_channel_only(clean, kind):
    series, extra, ch, sl = _inject(kind, *clean)
    extra.setdefault("elev", 200.0)
    x, mask, codes = run(*series, **extra)
    assert has(codes, sl, ch, CASES[kind]), f"{kind}: нет кода {CASES[kind].name}"
    assert np.all(mask[sl, ch] == 0) and np.all(x[sl, ch] == 0)
    _, mask0, _ = run(*clean, elev=extra["elev"])
    for other in {0, 1, 2} - {ch}:
        assert np.array_equal(mask[:, other], mask0[:, other]), \
            f"{kind}: задет чужой канал {Q.CHANNELS[other]}"


def test_stuck_limits_are_respected(clean):
    T, P, RH = (a.copy() for a in clean)
    T[600:620] = T[600]
    T[700:760] = T[700]
    RH[900:950] = 100.0
    _, _, codes = run(T, P, RH, elev=200.0)
    assert not np.any(codes & QCCode.STUCK), "одна температура 60 ч при живой влажности - норма"


def test_temperature_stuck_together_with_humidity(clean):
    """Температура и влажность стоят вместе 30 ч: залипание обоих каналов."""
    T, P, RH = (a.copy() for a in clean)
    T[600:630], RH[600:630] = T[600], RH[600]
    _, _, codes = run(T, P, RH, elev=200.0)
    assert has(codes, slice(600, 630), 0, QCCode.STUCK)
    assert has(codes, slice(600, 630), 2, QCCode.STUCK)


def test_stuck_survives_sparse_reporting(clean):
    """Замёрзший датчик со сводками раз в 3 ч ловится: серия идёт по валидным отсчётам."""
    T, P, RH = (a.copy() for a in clean)
    valid = np.ones((N, 3), np.uint8)
    valid[600:690, 0] = 0
    valid[600:690:3, 0] = 1
    T[600:690] = 1.5
    _, _, codes = run(T, P, RH, valid=valid, elev=200.0)
    assert has(codes, slice(600, 690, 3), 0, QCCode.STUCK)


def test_spike_return_is_not_a_jump(clean):
    T, P, RH = (a.copy() for a in clean)
    T[400] += 25.0
    _, _, codes = run(T, P, RH, elev=200.0)
    assert codes[400, 0] & QCCode.SPIKE and not np.any(codes[:, 0] & QCCode.JUMP)


def test_short_excursion_is_a_spike_and_return_hour_stays_valid(clean):
    """Выброс на 2 ч, не пойманный порогом SPIKE: бракуются оба часа выброса,
    а час возврата к норме остаётся валидным."""
    T, P, RH = (a.copy() for a in clean)
    T[400:402] += 22.0
    _, mask, codes = run(T, P, RH, elev=200.0)
    assert np.all(codes[400:402, 0] & QCCode.SPIKE) and np.all(mask[400:402, 0] == 0)
    assert mask[402, 0] == 1 and mask[399, 0] == 1
    assert not np.any(codes[:, 0] & QCCode.JUMP)


def test_persistent_step_is_a_jump_not_a_spike(clean):
    T, P, RH = (a.copy() for a in clean)
    T[500:] += 15.0
    _, mask, codes = run(T, P, RH, elev=200.0)
    assert codes[500, 0] & QCCode.JUMP and mask[501:520, 0].all()


def test_units_undecidable_near_minus_forty_is_not_flagged():
    """Около −40 °C шкалы совпадают: холодная погода не должна считаться °F."""
    n = 24 * 90
    T, P, RH = physical_series(n, lat=75, lon=20, seed=3)
    T = (T - np.median(T) - 38.0).astype(np.float32)
    T[1000:1100] += 10.0
    _, _, codes = run(T, P, RH, elev=200.0)
    assert not np.any(codes & QCCode.UNITS)


def test_sea_level_pressure_not_decidable_for_low_station(clean):
    T, P, RH = clean
    P = (P + (Q.P_SEA_LEVEL - Q.station_pressure_expected(200.0))).astype(np.float32)
    _, _, codes = run(T, P, RH, elev=200.0)
    assert not np.any(codes & QCCode.UNITS), "200 м: синоптика перекрывает разницу"


def _fahrenheit_brute(T, ok_raw, ok_ref, cfg=DEFAULT_QC):
    T = np.asarray(T, np.float64)
    r = Q.rolling_median(T, ok_raw, cfg.units_half, cfg.units_min_valid)
    ref, spread = Q._daily_reference(T, ok_ref, cfg)
    conv = (r - 32) / 1.8
    with np.errstate(invalid="ignore"):
        high = r - ref > np.maximum(cfg.units_ref_k * spread, cfg.units_min_excess)
        fits = (np.abs(conv - ref) <= cfg.units_ref_k * spread) & (conv >= cfg.units_min_conv)
    return ok_raw & high & fits


def _jump_brute(x, ok, floor, cfg=DEFAULT_QC):
    x = np.asarray(x, np.float64)
    n = len(x)
    dv = np.zeros(n, bool)
    dv[1:] = ok[1:] & ok[:-1]
    d = np.zeros(n)
    d[1:] = np.where(dv[1:], x[1:] - x[:-1], 0)
    rel = ~Q.mad_ok(d, dv, cfg.jump_half, cfg.jump_thresh, cfg.jump_min_valid)
    return dv & rel & (np.abs(d) > floor)


def _slp_brute(P, ok, elev, cfg=DEFAULT_QC):
    p_exp = Q.station_pressure_expected(elev)
    sep = Q.P_SEA_LEVEL - p_exp
    if sep < cfg.slp_min_sep:
        return np.zeros(len(P), bool)
    r = Q.rolling_median(np.asarray(P, np.float64), ok, cfg.slp_half, cfg.units_min_valid)
    with np.errstate(invalid="ignore"):
        return ok & (r > p_exp + sep / 2)


def test_prefiltered_checks_match_brute_force():
    rng = np.random.default_rng(1)
    hits = dict(F=0, J=0, P=0)
    for trial in range(150):
        n = int(rng.integers(30, 1500))
        h = np.arange(n)
        T = 15 + 8 * np.sin(2 * np.pi * (h - 8) / 24) + rng.standard_normal(n) * rng.uniform(.2, 4)
        if rng.random() < 0.7:
            a = int(rng.integers(0, n))
            b = a + int(rng.integers(5, 200))
            T[a:b] = T[a:b] * 1.8 + 32
        if rng.random() < 0.5:
            T[int(rng.integers(1, n)):] += rng.choice([-1, 1]) * rng.uniform(5, 20)
        ok = rng.random(n) < rng.uniform(0.5, 1)
        ok_ref = ok & (T < 60)
        f = Q.fahrenheit_flags(T, ok, ok_ref)
        assert np.array_equal(f, _fahrenheit_brute(T, ok, ok_ref)), trial
        j, _ = Q.jump_flags(T, ok, DEFAULT_QC.jump_half, DEFAULT_QC.jump_thresh,
                            DEFAULT_QC.jump_min_valid, DEFAULT_QC.jump_floor[0])
        assert np.array_equal(j, _jump_brute(T, ok, DEFAULT_QC.jump_floor[0])), trial
        elev = float(rng.uniform(0, 2500))
        P = Q.station_pressure_expected(elev) + 8 * rng.standard_normal(n)
        if rng.random() < 0.5:
            a = int(rng.integers(0, n))
            P[a:] += Q.P_SEA_LEVEL - Q.station_pressure_expected(elev)
        p = Q.sea_level_pressure_flags(P, ok, elev)
        assert np.array_equal(p, _slp_brute(P, ok, elev)), trial
        hits["F"] += f.sum()
        hits["J"] += j.sum()
        hits["P"] += p.sum()
    assert all(v > 0 for v in hits.values()), f"сравнение без срабатываний бессмысленно: {hits}"


@pytest.mark.parametrize("causal", [False, True])
def test_batched_spike_matches_per_channel_mad(causal):
    rng = np.random.default_rng(3)
    cfg = DEFAULT_QC
    for trial in range(100):
        n = int(rng.integers(1, 400))
        x = (rng.standard_normal((n, 3)) * [3, 5, 10] + [10, 1000, 60]).astype(np.float32)
        x[rng.random((n, 3)) < 0.03] += 40
        b = rng.random((n, 3)) < 0.8
        ref = np.stack([b[:, j] & ~Q.mad_ok(x[:, j], b[:, j], cfg.spike_half, cfg.spike_thresh,
                                            cfg.spike_min_valid, floor=cfg.scale_floor[j],
                                            causal=causal)
                        for j in range(3)], -1)
        assert np.array_equal(Q._spike_flags(x, b, cfg, causal), ref), trial


def test_source_flags_go_to_mask_without_correction(clean):
    T, P, RH = clean
    flag = np.zeros((N, 3), np.uint8)
    flag[100:110, 1] = 1
    valid = np.ones((N, 3), np.uint8)
    valid[200, 1] = 0
    x, mask, codes = run(T, P, RH, valid=valid, flag=flag, elev=200.0)
    assert np.all(codes[100:110, 1] == QCCode.SOURCE)
    assert codes[200, 1] == QCCode.MISSING
    assert np.all(mask[100:110, 1] == 0) and np.all(x[100:110, 1] == 0)
    assert np.all(mask[100:110, [0, 2]] == 1)
    assert np.array_equal(x[111:200, 1], P[111:200]), "соседние значения не исправлены"


def test_flagged_value_does_not_poison_window_checks(clean):
    """Помеченный источником выброс не участвует в скользящей медиане соседей."""
    T, P, RH = (a.copy() for a in clean)
    T[300:320] = 55.0
    flag = np.zeros(N, np.uint8)
    flag[300:320] = 1
    _, _, codes = run(T, P, RH, flag=np.stack([flag, 0 * flag, 0 * flag], -1), elev=200.0)
    assert np.all(codes[300:320, 0] == QCCode.SOURCE)
    assert not np.any(codes[280:340, 0] & (QCCode.SPIKE | QCCode.JUMP | QCCode.UNITS))


CAUSAL_DELAY = {"range_T": 0, "spike": 0, "stuck_T": DEFAULT_QC.stuck_T_alone_hours - 1,
                "stuck_RH": DEFAULT_QC.stuck_hours[2] - 1,
                "rh_saturated": DEFAULT_QC.rh_sat_hours - 1, "fahrenheit": DEFAULT_QC.units_half,
                "sea_level_pressure": DEFAULT_QC.units_min_valid}


@pytest.mark.parametrize("kind", list(CAUSAL_DELAY))
def test_window_path_catches_artifact_after_causal_delay(clean, kind):
    """Окно проходит причинный QC: артефакт получает свой код на своём канале, начиная с
    часа, когда его можно распознать по прошлому. Центрированный QC кэша помечает его
    целиком."""
    series, extra, ch, sl = _inject(kind, *clean)
    elev = extra.get("elev", 200.0)
    _, _, c_station = run(*series, elev=elev)
    lo = max(0, sl.start - 400) if sl.stop - sl.start < N else 0
    x = np.stack(series, -1)[lo:lo + L_MAX]
    mask_w, c_win = Q.qc_window(x, np.ones_like(x), elev=elev)
    wsl = slice(sl.start - lo + CAUSAL_DELAY[kind], min(sl.stop, lo + L_MAX) - lo)
    assert has(c_station, sl, ch, CASES[kind])
    assert has(c_win, wsl, ch, CASES[kind]), kind
    assert np.all(mask_w[sl.start - lo:wsl.stop, ch][c_win[sl.start - lo:wsl.stop, ch] > 0] == 0)


def test_window_qc_marks_input_gaps_missing_and_keeps_values(clean):
    x = np.stack(clean, -1)[:L_MAX].copy()
    m = np.ones_like(x)
    m[10:20, 2] = 0
    x0 = x.copy()
    mask, codes = Q.qc_window(x, m, elev=200.0)
    assert np.all(codes[10:20, 2] == QCCode.MISSING) and np.all(mask[10:20, 2] == 0)
    assert np.array_equal(x, x0), "qc_window не меняет значения"
    assert mask.dtype == np.float32


def test_runtime_step_uses_the_causal_qc(clean):
    """Поток устройства получает коды каждого часа той же функцией, что пакет."""
    T, P, RH = (a.copy() for a in clean)
    T[400] += 25.0
    RH[600:640] = RH[600]
    T[600:640] = T[600]
    T[700] = 75.0
    x = np.stack([T, P, RH], -1)[:900]
    present = np.ones_like(x, np.uint8)
    present[50:60, 1] = 0
    ref = Q.causal_codes(x, present, elev=200.0)
    ring = Q.CausalQC(elev=200.0)
    got = np.stack([ring.push([x[k, j] if present[k, j] else None for j in range(3)])[1]
                    for k in range(len(x))])
    assert np.array_equal(got, ref)
    assert np.any(ref & QCCode.STUCK) and np.any(ref & QCCode.RANGE)


LONG = 2 * YEAR


@pytest.fixture(scope="module")
def long_series():
    return physical_series(LONG, lat=45, lon=75, seed=5)


def _checks(T, lon=75.0, t0=T0, elev=None, dem=None):
    ok = np.isfinite(T)
    x = np.stack([T, np.zeros_like(T), np.zeros_like(T)], -1)
    m = np.stack([ok, ok, ok], -1).astype(np.uint8)
    return Q.station_checks(x, m, t0, lon=lon, elev=elev, dem_elev=dem)


def test_station_checks_pass_on_clean_station(long_series):
    c = _checks(long_series[0])
    assert {k: v["status"] for k, v in c.items()} == dict(
        t_level="pass", solar_phase="pass", changepoint="pass", dem_elevation="skip")
    assert 1.0 < c["solar_phase"]["value"] < 4.0


def test_solar_phase_catches_longitude_sign_error(long_series):
    c = _checks(long_series[0], lon=-75.0)
    assert c["solar_phase"]["status"] == "fail" and "долготы" in c["solar_phase"]["detail"]


def test_solar_phase_catches_local_time_stored_as_utc():
    """Метки в местном времени (UTC+8), записанные как UTC: ряд сдвинут на 8 ч."""
    T, _, _ = physical_series(YEAR, lat=30, lon=120, seed=2)
    assert _checks(T, lon=120.0)["solar_phase"]["status"] == "pass"
    shifted = np.roll(T, 8)
    assert _checks(shifted, lon=120.0)["solar_phase"]["status"] == "fail"


def test_solar_phase_is_inconclusive_without_diurnal_cycle():
    rng = np.random.default_rng(0)
    T = (5 + rng.standard_normal(YEAR)).astype(np.float32)
    c = _checks(T)
    assert c["solar_phase"]["status"] == "skip" and "слабый" in c["solar_phase"]["detail"]


def test_diurnal_phase_estimate_is_exact_on_pure_harmonic():
    h = np.arange(24 * 60)
    for peak in (0.0, 5.5, 14.0, 23.0):
        T = 10 + 4 * np.cos(2 * np.pi * (h - peak) / 24)
        t_max, amp, days = Q.diurnal_max_utc(T, np.ones_like(T, bool), 0)
        assert abs((t_max - peak + 12) % 24 - 12) < 1e-6 and abs(amp - 4) < 1e-6


def test_changepoint_catches_level_shift(long_series):
    T = long_series[0].copy()
    T[YEAR:] += 3.0
    c = _checks(T)["changepoint"]
    assert c["status"] == "fail" and 2.5 < c["value"] < 3.5
    when = np.datetime64(c["detail"].split(" с ")[1].split(" ")[0])
    true = np.datetime64("1970-01-01T00") + np.timedelta64(T0 + YEAR, "h")
    assert abs((when - true.astype("datetime64[D]")).astype(int)) <= 10


def test_changepoint_ignores_small_shift(long_series):
    T = long_series[0].copy()
    T[YEAR:] += 0.5
    assert _checks(T)["changepoint"]["status"] == "pass"


def test_dem_elevation_check():
    assert Q.check_dem_elevation(100.0, 150.0)["status"] == "pass"
    assert Q.check_dem_elevation(1500.0, 200.0)["status"] == "fail"
    assert Q.check_dem_elevation(100.0, None)["status"] == "skip"


def test_t_level_catches_whole_series_in_fahrenheit(long_series):
    T = long_series[0] * 1.8 + 32
    assert _checks(T)["t_level"]["status"] == "fail"


def test_t_level_uses_raw_values_not_range_censored(long_series):
    T = (long_series[0] * 1.8 + 32).astype(np.float32)
    P, RH = long_series[1], long_series[2]
    x, mask, codes = run(T, P, RH, elev=200.0)
    assert np.median(x[mask[:, 0] > 0, 0]) < DEFAULT_QC.t_median[1], "предпосылка теста"
    censored = Q.station_checks(x, mask, T0, lon=75.0)["t_level"]
    raw = Q.station_checks(x, mask, T0, lon=75.0, raw_T=T, codes=codes)["t_level"]
    assert censored["status"] == "pass" and raw["status"] == "fail"


def test_station_selection_rules():
    ok = {k: dict(status="pass", value=0, detail="") for k in Q.STATION_CHECKS}
    m = np.ones((YEAR, 3), np.uint8)
    assert Q.station_selection(YEAR, m, ok) == []
    rules = lambda *a: [r for r, _ in Q.station_selection(*a)]
    assert rules(100, m[:100], ok) == ["length"]
    m2 = m.copy()
    m2[:, 0] = 0
    assert rules(YEAR, m2, ok) == ["no_T"]
    m2[: YEAR // 3, 0] = 1
    assert rules(YEAR, m2, ok) == ["valid_T"]
    bad = dict(ok, changepoint=dict(status="fail", value=2, detail="x"))
    assert rules(YEAR, m, bad) == ["check:changepoint"]
    cfg = dataclasses.replace(DEFAULT_QC, enforce_checks=("t_level",))
    assert Q.station_selection(YEAR, m, bad, cfg) == [], "неприменяемая проверка не исключает"


def test_qc_config_validation():
    with pytest.raises(ValueError):
        QCConfig(enforce_checks=("nope",))
    with pytest.raises(ValueError):
        QCConfig(jump_floor=(1.0, 2.0))
    with pytest.raises(ValueError):
        QCConfig(solar_lag=(3.0, 1.0))


def _write(root, sid, T, P, RH, valid=None, **opt):
    n = len(T)
    valid = np.ones((n, 3), np.uint8) if valid is None else valid
    np.savez(root / "stations" / f"{sid}.npz", T=T, P=P, RH=RH, valid=valid,
             t0_utc_h=np.int64(T0), **opt)


def _manifest(root, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return str(root / "manifest.csv")


LEN = int(1.4 * YEAR)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Набор, где каждая станция нарушает ровно одно правило отбора."""
    root = tmp_path_factory.mktemp("qc8")
    (root / "stations").mkdir()
    rows = []

    def add(sid, lon=30.0, elev=200.0, dem="", mutate=None, **opt):
        T, P, RH = physical_series(LEN, lon=lon, elev=elev, seed=len(rows))
        if mutate:
            T, P, RH = mutate(T.copy(), P.copy(), RH.copy())
        _write(root, sid, T, P, RH, **opt)
        rows.append(dict(id=sid, lat=45.0, lon=lon, elev=elev, koppen="Cfb", split="train",
                         dem_elev=dem))

    def shift(T, P, RH):
        T[LEN // 2:] += 4.0
        return T, P, RH

    def fahr(T, P, RH):
        return (T * 1.8 + 32).astype(np.float32), P, RH

    flag = np.zeros((LEN, 3), np.uint8)
    flag[1000:1010, 0] = 1
    Td = physical_series(LEN, seed=0)[0] - 6.0
    Td[2000:2003] += 20.0
    add("good", flag=flag, Td=Td.astype(np.float32))
    add("good2", dem="230")
    add("wrong_lon", lon=-60.0, mutate=lambda T, P, RH: (np.roll(T, -8), P, RH))
    add("moved", mutate=shift)
    add("fahrenheit", mutate=fahr)
    add("bad_dem", dem="1500")
    v = np.ones((LEN, 3), np.uint8)
    v[: int(0.7 * LEN), 0] = 0
    add("sparse_T", valid=v)
    T, P, RH = physical_series(5000)
    _write(root, "short", T, P, RH)
    rows.append(dict(id="short", lat=45.0, lon=30.0, elev=200.0, koppen="Cfb", split="train",
                     dem_elev=""))
    manifest = _manifest(root, rows)
    S._STORES.clear()
    path, _ = S.build_cache(manifest)
    meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
    with open(Path(path) / "qc_report.csv") as f:
        report = {r["id"]: r for r in csv.DictReader(f)}
    yield dict(manifest=manifest, path=path, meta=meta, report=report, root=root)
    S._STORES.clear()


EXPECTED_RULE = {"wrong_lon": "check:solar_phase", "moved": "check:changepoint",
                 "fahrenheit": "check:t_level", "bad_dem": "check:dem_elevation",
                 "sparse_T": "valid_T", "short": "length"}


def test_selection_excludes_exactly_the_bad_stations(built):
    assert set(built["meta"]["excluded"]) == set(EXPECTED_RULE)
    store = S.get_store(built["manifest"])
    assert set(store.stations) == {"good", "good2"}
    assert "сплиты" in built["meta"]["excluded"]["short"]


def test_qc_summary_counts_rules_and_checks(built):
    qc = built["meta"]["qc"]
    assert qc["stations_total"] == 8 and qc["stations_included"] == 2
    for sid, rule in EXPECTED_RULE.items():
        assert qc["excluded_by_rule"].get(rule, 0) >= 1, (sid, rule)
    for name, c in qc["station_checks"].items():
        assert sum(c.values()) == 8, name
    assert qc["station_checks"]["dem_elevation"] == dict(**{"pass": 1, "fail": 1, "skip": 6})
    assert qc["fingerprint"] == DEFAULT_QC.fingerprint() and "version" not in qc


def test_report_has_every_station_with_status_and_reason(built):
    rep = built["report"]
    assert set(rep) == {"good", "good2", *EXPECTED_RULE}
    for sid in EXPECTED_RULE:
        assert rep[sid]["status"] == "excluded" and rep[sid]["reason"]
    assert rep["good"]["status"] == "included" and rep["good"]["reason"] == ""
    assert rep["wrong_lon"]["check/solar_phase"] == "fail"
    assert float(rep["sparse_T"]["T/valid"]) == pytest.approx(0.3, abs=0.01)


def test_report_fractions_match_stored_codes(built):
    store = S.get_store(built["manifest"])
    s = store.stations["good"]
    fr = Q.code_fractions(s["qc"], s["mask"])
    for k, v in fr.items():
        assert float(built["report"]["good"][k]) == pytest.approx(v)
    all_codes = np.concatenate([store.stations[i]["qc"] for i in ("good", "good2")])
    for k, v in Q.code_fractions(all_codes).items():
        assert built["meta"]["qc_total"][k] == pytest.approx(v)


def test_source_flags_and_dewpoint_from_source_file_reach_the_cache(built):
    s = S.get_store(built["manifest"]).stations["good"]
    assert np.all(s["qc"][1000:1010, 0] == QCCode.SOURCE)
    assert np.all(s["qc"][2000:2003, 2] & QCCode.DEWPOINT)
    assert np.all(s["mask"][2000:2003, 0] == 1), "точка росы бракует RH, а не T"
    assert s["qc_checks"]["solar_phase"] == "pass"


def test_cache_key_tracks_what_qc_reads(built):
    m = built["manifest"]
    rows = S.read_manifest(m)
    k0 = S.cache_key(S.key_payload(m, rows))
    for field, val in (("lon", "31.0"), ("elev", "250"), ("dem_elev", "210")):
        changed = [dict(r) for r in rows]
        changed[0][field] = val
        assert S.cache_key(S.key_payload(m, changed)) != k0, field
    changed = [dict(r) for r in rows]
    changed[0]["lat"], changed[0]["koppen"] = "10.0", "Af"
    assert S.cache_key(S.key_payload(m, changed)) == k0, "lat и зона в QC не участвуют"
    cfg = dataclasses.replace(DEFAULT_QC, jump_thresh=9.0)
    assert S.cache_key(S.key_payload(m, rows, cfg)) != k0


def test_full_station_qc_is_fast():
    T, P, RH = physical_series(10 * YEAR, seed=0)
    t = time.perf_counter()
    x, mask, codes = run(T, P, RH, elev=200.0)
    Q.station_checks(x, mask, T0, lon=30.0)
    assert time.perf_counter() - t < 5.0, "10 лет почасовых данных — секунды, не минуты"


@pytest.fixture(scope="module")
def train_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("qc8train")
    (root / "stations").mkdir()
    rows = []
    for i, role in enumerate(["train", "train", "unseen_val", "unseen_test"]):
        T, P, RH = physical_series(LEN, lon=10.0 * i, seed=10 + i)
        _write(root, f"t{i}", T, P, RH)
        rows.append(dict(id=f"t{i}", lat=45.0, lon=10.0 * i, elev=200.0, koppen="Cfb",
                         split=role))
    S._STORES.clear()
    return _manifest(root, rows)


def _items(manifest, n=24, **kw):
    from mayak.data.dataset import WindowDataset
    ds = WindowDataset(manifest, windows_per_epoch=n, seed=0, **kw)
    return [ds[i] for i in range(n)]


def test_window_qc_sees_augmented_artifacts(train_manifest):
    """Грубый шум аугментации выводит T за физический диапазон: QC окна обязан это
    увидеть. Без QC окна те же значения проходят в модель как валидные."""
    aug = dict(noise_sd=(40.0, 0.5, 2.0), gap_prob=0.0, offset_max=0.0)
    on = _items(train_manifest, augment=aug, window_qc=True)
    off = _items(train_manifest, augment=aug, window_qc=False)
    lo, hi = Q.PHYS["T"]
    n_bad_off = n_masked = 0
    for a, b in zip(on, off):
        x, m = a["x_hist"].numpy(), a["mask_hist"].numpy()
        assert np.all(x[m == 0] == 0), "инвариант после QC окна"
        tv = x[:, 0][m[:, 0] > 0]
        assert np.all((tv >= lo) & (tv <= hi))
        assert np.all(m <= b["mask_hist"].numpy()), "QC окна только снимает валидность"
        xb, mb = b["x_hist"].numpy(), b["mask_hist"].numpy()
        tb = xb[:, 0][mb[:, 0] > 0]
        n_bad_off += int(np.sum((tb < lo) | (tb > hi)))
        n_masked += int(np.sum(mb[:, 0]) - np.sum(m[:, 0]))
        assert torch.equal(a["y"], b["y"]) and torch.equal(a["y_mask"], b["y_mask"]), \
            "QC окна не трогает цель"
    assert n_bad_off > 0 and n_masked >= n_bad_off


def test_window_qc_does_not_eat_clean_history(train_manifest):
    aug = dict(profile="base", noise_sd=(0.2, 0.5, 2.0), gap_prob=0.0, offset_max=0.0,
               drop_humidity_prob=0.0, drop_pressure_prob=0.0)
    on = _items(train_manifest, augment=aug, window_qc=True)
    off = _items(train_manifest, augment=aug, window_qc=False)
    kept = sum(a["mask_hist"].sum().item() for a in on)
    total = sum(b["mask_hist"].sum().item() for b in off)
    assert total > 0 and kept / total > 0.995
    for a, b in zip(on, off):
        assert torch.equal(a["x_hist"][a["mask_hist"] > 0], b["x_hist"][a["mask_hist"] > 0])


def test_window_qc_does_not_shift_random_streams(train_manifest):
    on = _items(train_manifest, window_qc=True)
    off = _items(train_manifest, window_qc=False)
    for a, b in zip(on, off):
        for k in ("doy_fut", "y", "y_mask", "lat", "lon", "norm_scale"):
            assert torch.equal(a[k], b[k]), k


def test_data_config_carries_window_qc_to_datamodule(train_manifest):
    from mayak.config import ConfigError, DataConfig
    from mayak.data.datamodule import MayakData
    assert DataConfig().window_qc is True
    with pytest.raises(ConfigError):
        DataConfig(window_qc="yes")
    dm = MayakData(train_manifest, windows_per_epoch=4, num_workers=0,
                   data_config=DataConfig(manifest=train_manifest, window_qc=False))
    dm.setup()
    assert dm.train_ds.window_qc is False
