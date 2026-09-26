"""Тесты: причинный QC истории, общий для обучения, оценки и устройства."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mayak.constants import L_MAX
from mayak.data import qc as Q
from mayak.data.qc import DEFAULT_QC, QCCode, QCConfig
from mayak.data.recording import record_values

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "qc_causal" / "golden.json"


def diurnal_series(n, amp=8.0, base=15.0, syn_sd=3.0, rh_mean=55.0, elev=200.0, seed=0):
    """Ряд с сильным суточным ходом: быстрый подъём утром, медленный спад вечером."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    ph = 2 * np.pi * ((k % 24) - 15) / 24
    d = amp * (np.cos(ph) + 0.3 * np.cos(2 * ph + 0.8)) / 1.3
    rho = math.exp(-1 / 60)
    e = rng.standard_normal(n) * syn_sd * math.sqrt(1 - rho ** 2)
    syn = np.zeros(n)
    for i in range(1, n):
        syn[i] = rho * syn[i - 1] + e[i]
    T = base + d + syn + 0.3 * rng.standard_normal(n)
    P = Q.station_pressure_expected(elev) + syn + 0.15 * rng.standard_normal(n)
    RH = np.clip(rh_mean - 2.5 * d + 3 * rng.standard_normal(n), 3, 100)
    return np.stack([T, P, RH], -1)


recorded = record_values


def with_artifacts(seed, n=1100, step=1):
    rng = np.random.default_rng(seed)
    x = recorded(diurnal_series(n, seed=seed))
    present = np.zeros((n, 3), np.uint8)
    present[::step] = 1
    for _ in range(6):
        i, ch = int(rng.integers(0, n)), int(rng.integers(3))
        x[i, ch] += rng.choice([-1, 1]) * [20, 25, 60][ch]
    a = int(rng.integers(0, n - 100))
    x[a:a + 90, 0], x[a:a + 90, 2] = x[a, 0], x[a, 2]
    a = int(rng.integers(0, n - 10))
    x[a:a + 3, 0] += 15
    present[rng.random(present.shape) < 0.05] = 0
    return x, present


@pytest.mark.parametrize("seed, step", [(0, 1), (1, 2), (2, 3)])
def test_codes_of_an_hour_do_not_change_when_future_hours_arrive(seed, step):
    x, present = with_artifacts(seed, step=step)
    full = Q.causal_codes(x, present, elev=200.0)
    for cut in np.random.default_rng(seed).integers(20, len(x), 12):
        assert np.array_equal(Q.causal_codes(x[:cut], present[:cut], elev=200.0), full[:cut])


@pytest.mark.parametrize("seed, step", [(3, 1), (4, 3)])
def test_stream_equals_batch_hour_by_hour(seed, step):
    x, present = with_artifacts(seed, step=step)
    ref = Q.causal_codes(x, present, elev=200.0)
    ring = Q.CausalQC(elev=200.0)
    assert ring.size == DEFAULT_QC.lookback_hours + 1
    for k in range(len(x)):
        v, codes = ring.push([x[k, j] if present[k, j] else None for j in range(3)])
        assert np.array_equal(codes, ref[k]), k
        assert np.all(v[codes > 0] == 0)
    assert np.any(ref & QCCode.STUCK) and np.any(ref & QCCode.SPIKE)


def test_window_with_context_equals_codes_of_the_whole_series():
    x, present = with_artifacts(5)
    full = Q.causal_codes(x, present, elev=200.0)
    k = 300
    mask, codes = Q.qc_window(x[k:k + L_MAX], present[k:k + L_MAX], elev=200.0,
                              past=(x[:k], present[:k]))
    assert np.array_equal(codes, full[k:k + L_MAX])
    assert np.array_equal(mask, (full[k:k + L_MAX] == 0).astype(np.float32))


def test_lookback_covers_every_rule():
    c = DEFAULT_QC
    assert c.lookback_hours >= c.stuck_T_alone_hours + c.stuck_max_gap
    assert c.lookback_hours >= 2 * c.jump_half + c.jump_max_gap + 2 * c.spike_half
    assert c.lookback_hours >= 2 * c.slp_half
    assert QCConfig(stuck_T_alone_hours=24, rh_sat_hours=24).lookback_hours < c.lookback_hours
    assert c.lookback_hours == 78, "без проверки единиц температуры глубина - трое суток"


def test_reference_vectors_are_reproduced():
    """Пакет и поток на Python воспроизводят эталон, по которому сверяется порт на Rust."""
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cfg = dict(DEFAULT_QC.to_dict(), lookback_hours=DEFAULT_QC.lookback_hours)
    assert doc["config"] == cfg, "пороги изменились: python scripts/make_qc_golden.py"
    assert doc["phys"] == {c: list(v) for c, v in Q.PHYS.items()}
    seen = 0
    for case in doc["cases"]:
        present = np.array([[v is not None for v in row] for row in case["x"]], np.uint8)
        x = np.array([[0.0 if v is None else v for v in row] for row in case["x"]], np.float32)
        want = np.array(case["codes"], np.uint8)
        assert np.array_equal(Q.causal_codes(x, present, elev=case["elev"]), want), case["name"]
        seen |= int(np.bitwise_or.reduce(want.ravel()))
    for c in (QCCode.MISSING, QCCode.RANGE, QCCode.SPIKE, QCCode.JUMP, QCCode.STUCK,
              QCCode.UNITS):
        assert seen & c, f"эталон не покрывает код {c.name}"


def test_clean_integer_record_is_rarely_flagged():
    """Чистые ряды после записи целыми числами: ложные коды - меньше процента часов.

    Самый трудный случай - суточный размах 20 градусов при почасовых отчётах и
    влажность при отчётах раз в 3 часа."""
    for seed, (amp, base, rh) in enumerate([(10, 25, 25), (6, 10, 65), (1, 27, 80),
                                            (0.3, -25, 85)]):
        for step in (1, 2, 3):
            x = recorded(diurnal_series(24 * 120, amp=amp, base=base, rh_mean=rh, seed=seed))
            present = np.zeros(x.shape, np.uint8)
            present[::step] = 1
            codes = Q.causal_codes(x, present, elev=200.0)
            v = present > 0
            for c in (QCCode.STUCK, QCCode.UNITS, QCCode.RANGE):
                assert not np.any(codes[v] & c), (seed, step, c.name)
            for j in range(3):
                frac = float(((codes[:, j] & (QCCode.SPIKE | QCCode.JUMP)) > 0)[v[:, j]].mean())
                limit = 0.01 if step == 1 else 0.015
                assert frac < limit, (seed, step, Q.CHANNELS[j], frac)


def test_morning_warming_after_a_flat_night_is_not_a_spike():
    n = 24 * 20
    T = np.full(n, 5.0)
    for d in range(20):
        T[24 * d + 7:24 * d + 13] = 5.0 + 2.0 * np.arange(1, 7)
        T[24 * d + 13:24 * d + 19] = 17.0 - 2.0 * np.arange(1, 7)
    x = np.stack([T, np.full(n, 1000.0), np.full(n, 60.0) + (np.arange(n) % 5)], -1)
    codes = Q.causal_codes(x.astype(np.float32), np.ones((n, 3)), elev=200.0)
    assert not np.any(codes[:, 0] & QCCode.SPIKE)


def test_a_three_hour_change_is_judged_per_hour():
    """Скачок на 12 градусов между отчётами через 3 ч - не скачок за час."""
    n = 24 * 10
    x = np.zeros((n, 3), np.float32)
    x[:, 0], x[:, 1], x[:, 2] = 10.0, 1000.0, 60.0 + (np.arange(n) % 4)
    x[:, 0] += np.round(np.sin(np.arange(n) / 5.0))
    x[150:, 0] += 12.0
    present = np.zeros((n, 3), np.uint8)
    present[::3] = 1
    codes = Q.causal_codes(x, present, elev=200.0)
    assert not np.any(codes[:, 0] & QCCode.JUMP)
    rate, level, defined = Q.increments(x[:, 0], present[:, 0] > 0, DEFAULT_QC.jump_max_gap)
    assert defined[150] and level[150] == pytest.approx(12.0 + x[150, 0] - 12.0 - x[147, 0])
    assert rate[150] == pytest.approx(level[150] / 3)


def test_integer_step_of_one_degree_is_not_a_spike():
    n = 200
    x = np.zeros((n, 3), np.float32)
    x[:, 0], x[:, 1], x[:, 2] = 3.0, 1000.0, 70.0
    x[100:, 0] = 4.0
    x[50, 2] = 71.0
    x[::7, 1] += 0.1
    codes = Q.causal_codes(x, np.ones((n, 3)), elev=200.0)
    assert not np.any(codes & (QCCode.SPIKE | QCCode.JUMP))


def test_temperature_alone_stays_valid_longer_than_with_humidity():
    n = 400
    x = recorded(diurnal_series(n, seed=7))
    alone = x.copy()
    alone[100:200, 0] = alone[100, 0] + 0.5
    c = Q.causal_codes(alone, np.ones((n, 3)), elev=200.0)
    stuck = np.flatnonzero(c[:, 0] & QCCode.STUCK)
    assert stuck.min() == 100 + DEFAULT_QC.stuck_T_alone_hours - 1
    both = x.copy()
    both[100:140, 0], both[100:140, 2] = both[100, 0] + 0.5, both[100, 2] + 0.5
    c = Q.causal_codes(both, np.ones((n, 3)), elev=200.0)
    stuck = np.flatnonzero(c[:, 0] & QCCode.STUCK)
    assert stuck.min() == 100 + DEFAULT_QC.stuck_hours[0] - 1 and stuck.max() == 139
    assert np.all(c[100 + DEFAULT_QC.stuck_hours[2] - 1:140, 2] & QCCode.STUCK)


def test_return_to_previous_level_is_not_a_jump():
    """Прибор помечает первый скачок, а возврат к прежнему уровню оставляет валидным."""
    n = 300
    rng = np.random.default_rng(0)
    x = np.zeros((n, 3), np.float32)
    x[:, 0] = np.round(10 + 3 * np.sin(np.arange(n) / 4.0) + rng.standard_normal(n))
    x[:, 1], x[:, 2] = 1000.0, 60.0 + (np.arange(n) % 5)
    x[200:202, 0] += 20.0
    cfg = dataclasses.replace(DEFAULT_QC, spike_thresh=50.0)
    c = Q.causal_codes(x, np.ones((n, 3)), elev=200.0, cfg=cfg)
    assert c[200, 0] & QCCode.JUMP
    assert c[202, 0] == 0


def test_temperature_units_are_not_checked_on_device():
    """Температуру в градусах Цельсия обеспечивает владелец прибора: QC её единицы не
    проверяет, код UNITS бывает только у давления."""
    n = 24 * 10
    x = recorded(diurnal_series(n, amp=3, base=5, seed=2))
    x[150:190, 0] = x[150:190, 0] * 1.8 + 32
    codes = Q.causal_codes(x, np.ones((n, 3)), elev=200.0)
    assert not np.any(codes[:, 0] & QCCode.UNITS)


def test_qc_config_rejects_inconsistent_stuck_limits():
    with pytest.raises(ValueError, match="stuck_min_count"):
        QCConfig(stuck_min_count=10)
    with pytest.raises(ValueError, match="scale_floor"):
        QCConfig(scale_floor=(1.0, 0.0, 1.0))


def test_qc_context_follows_device_age():
    from mayak.data.dataset import footprint, qc_context
    look = DEFAULT_QC.lookback_hours
    assert qc_context(10_000, L_MAX, 0) == look
    assert qc_context(10_000, 100, 0) == 0, "прибор включён 100 ч назад"
    assert qc_context(L_MAX + 50, L_MAX, 0) == 50, "контекст не раньше начала ряда"
    assert qc_context(5_000, L_MAX, 5_000 - L_MAX - 30) == 30
    assert qc_context(5_000, L_MAX, 1_000) == look
    assert footprint(10_000, None, 0) == (10_000 - L_MAX - look, 10_000 + 168)


@pytest.fixture(scope="module")
def ext_store(tmp_path_factory):
    """Набор с одной внешней станцией, у которой источник пометил выброс."""
    import csv

    from mayak.data import store as S
    from mayak.timeaxis import to_utc_hour
    from datetime import datetime
    root = tmp_path_factory.mktemp("ext")
    (root / "stations").mkdir()
    n = 24 * 365 * 8
    x = recorded(diurnal_series(n, amp=4, base=10, seed=11))
    flag = np.zeros((n, 3), np.uint8)
    t_spike = n - 1000
    x[t_spike, 0] += 30.0
    flag[t_spike, 0] = 1
    x[n - 800, 0] = x[n - 800, 0] + 0.0
    flag[n - 800, 0] = 1
    valid = np.ones(n, np.uint8)
    valid[n - 700] = 0
    np.savez(root / "stations" / "e0.npz", T=x[:, 0], P=x[:, 1], RH=x[:, 2], valid=valid,
             flag=flag, t0_utc_h=int(to_utc_hour(datetime(2015, 1, 1))))
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader()
        w.writerow(dict(id="e0", lat=45.0, lon=10.0, elev=200.0, koppen="Cfb",
                        split="external_test"))
    return root / "manifest.csv", S.get_store(str(root / "manifest.csv")), t_spike, n


def test_cache_keeps_raw_values_and_presence(ext_store):
    _manifest, store, t_spike, n = ext_store
    s = store.stations["e0"]
    assert s["raw"].shape == s["x"].shape == (n, 3) and s["present"].dtype == np.uint8
    assert s["present"][n - 800, 0] == 1, "флаг источника - не отсутствие значения"
    assert s["present"][n - 700].sum() == 0 and np.all(s["raw"][n - 700] == 0)
    assert s["mask"][n - 800, 0] == 0 and s["qc"][n - 800, 0] & QCCode.SOURCE
    assert s["raw"][t_spike, 0] > s["raw"][t_spike - 1, 0] + 20, "сырое значение не исправлено"
    assert s["x"][t_spike, 0] == 0 and s["mask"][t_spike, 0] == 0


def test_eval_history_is_what_the_device_sees(ext_store):
    """История оценки - сырые значения через причинный QC; флаг источника прибор не
    видит. Цель - маска центрированного QC кэша со штатными флагами."""
    import torch

    from mayak.data.dataset import slice_context, slice_history
    from mayak.evaluate import EvalSet
    manifest, store, t_spike, n = ext_store
    ds = EvalSet(store.clims(), station_splits=("external_test",), manifest=str(manifest),
                 time_key="test", every_hours=24, max_windows=None)
    assert len(ds) > 5
    s = store.stations["e0"]
    for i in range(len(ds)):
        _sid, t = ds.items[i]
        item = ds[i]
        L = int(item["x_hist"].shape[0])
        xr, mr = slice_history(s["raw"], s["present"], t, L)
        past = slice_context(s["raw"], s["present"], t, L, ds.floor["e0"])
        mask, _ = Q.qc_window(xr, mr, elev=200.0, past=past)
        assert torch.equal(item["mask_hist"], torch.from_numpy(mask))
        y_mask = s["mask"][t:t + 168, 0].astype(np.float32)
        assert np.array_equal(item["y_mask"].numpy(), y_mask)
        if t - L_MAX <= n - 800 < t:
            assert item["mask_hist"][n - 800 - (t - L_MAX), 0] == 1, "флаг источника не виден"
