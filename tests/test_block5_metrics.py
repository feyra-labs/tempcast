"""Тесты: знаменатели, макро-оценка, надёжность, значимость."""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mayak.constants import H, QUANTILES
from mayak.data import store as S
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from mayak.metrics import (LEAD_BINS, METRICS, NQ, Evaluation, apply_conformal, breakdown,
                           by_lead, conformal_table, fit_conformal_shift, lead_bin_index,
                           seed_spread, spread)
from mayak.zones import (KG_TIF_CODE, KOPPEN_ZONES, SEASONS, UNKNOWN_ZONE, koppen_group,
                         koppen_id, normalize_zone, season_of, seasons_of)

REPO = Path(__file__).resolve().parents[1]
Q = np.asarray(QUANTILES, np.float64)
N_HOURS = 12_000


def _case(n_windows=60, horizon=H, n_stations=5, seed=0, noise=1.0, clim_noise=3.0):
    """Случайный набор предсказаний с монотонными квантилями и дырявой маской."""
    rng = np.random.default_rng(seed)
    y = rng.normal(10, 5, (n_windows, horizon))
    mu = y + rng.normal(0, noise, (n_windows, horizon))
    mu_clim = y + rng.normal(0, clim_noise, (n_windows, horizon))
    q = mu[..., None] + np.linspace(-3, 3, NQ)
    w = (rng.random((n_windows, horizon)) < 0.7).astype(np.float64)
    w[:, 0] = 1.0
    station = np.array([f"s{i % n_stations}" for i in range(n_windows)], object)
    return Evaluation(y=y, mu=mu, q=q, mu_clim=mu_clim, w=w, station=station)


def _manual_skill(ev, sel, leads=None):
    """Скилл «на бумаге»: числитель и знаменатель по одним и тем же валидным парам."""
    w = ev.w[sel]
    if leads is not None:
        m = np.zeros(ev.horizon, bool)
        m[np.asarray(leads) - 1] = True
        w = w * m[None, :]
    k = w > 0
    se = ((ev.mu[sel] - ev.y[sel]) ** 2)[k]
    se_c = ((ev.mu_clim[sel] - ev.y[sel]) ** 2)[k]
    return 1.0 - se.mean() / se_c.mean()


def _calibrated(n=4000, horizon=8, sigma=2.0, seed=0, n_stations=8):
    """Идеально откалиброванный прогноз: q - истинные квантили распределения y."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 5, (n, horizon))
    from scipy.stats import norm
    q = mu[..., None] + sigma * norm.ppf(Q)[None, None, :]
    y = mu + sigma * rng.standard_normal((n, horizon))
    station = np.array([f"s{i % n_stations}" for i in range(n)], object)
    return Evaluation(y=y, mu=mu, q=q, mu_clim=np.zeros_like(y),
                      w=np.ones((n, horizon)), station=station)


def test_skill_on_subsample_matches_manual():
    ev = _case(seed=1)
    sel = np.zeros(len(ev.y), bool)
    sel[::3] = True
    got = ev.restrict(windows=sel).pooled()["Skill"]
    assert got == pytest.approx(_manual_skill(ev, sel))


def test_skill_on_subsample_and_leads_matches_manual():
    ev = _case(seed=2)
    sel = np.array([s in ("s0", "s3") for s in ev.station])
    leads = [1, 24, 72]
    got = ev.restrict(windows=sel, leads=leads).pooled()["Skill"]
    assert got == pytest.approx(_manual_skill(ev, sel, leads))


def test_global_denominator_would_give_another_number():
    rng = np.random.default_rng(0)
    n, h = 200, 4
    calm = np.arange(n) < n // 2
    y = np.zeros((n, h))
    amp = np.where(calm, 1.0, 10.0)[:, None]
    y = amp * rng.standard_normal((n, h))
    mu = y + 0.5 * amp * rng.standard_normal((n, h))
    mu_clim = y + 1.0 * amp * rng.standard_normal((n, h))
    ev = Evaluation(y=y, mu=mu, q=mu[..., None] + np.linspace(-1, 1, NQ),
                    mu_clim=mu_clim, w=np.ones((n, h)),
                    station=np.array([f"s{i % 4}" for i in range(n)], object))

    honest = ev.restrict(windows=calm).pooled()["Skill"]
    se = ((mu - y) ** 2)[calm].mean()
    se_clim_global = ((mu_clim - y) ** 2).mean()
    cheating = 1.0 - se / se_clim_global

    assert honest == pytest.approx(_manual_skill(ev, calm))
    assert cheating > honest + 0.15, "разница между знаменателями должна быть видна"


def test_by_lead_denominator_is_per_lead():
    ev = _case(seed=3)
    tbl = by_lead(ev, leads=(1, 24, 168))
    for h in (1, 24, 168):
        w = ev.w[:, h - 1] > 0
        se = ((ev.mu[:, h - 1] - ev.y[:, h - 1]) ** 2)[w].mean()
        se_c = ((ev.mu_clim[:, h - 1] - ev.y[:, h - 1]) ** 2)[w].mean()
        assert tbl[h]["pooled"]["Skill"] == pytest.approx(1.0 - se / se_c)


def test_pooled_and_macro_diverge_on_constructed_example():
    h = 2
    big = np.full((90, h), 4.0)
    small = np.full((10, h), 1.0)
    err = np.concatenate([big, small])
    y = np.zeros_like(err)
    mu = err
    station = np.array(["big"] * 90 + ["small"] * 10, object)
    ev = Evaluation(y=y, mu=mu, q=mu[..., None] + np.linspace(-1, 1, NQ),
                    mu_clim=np.full_like(y, 5.0), w=np.ones_like(y), station=station)

    pooled, macro = ev.pooled(), ev.macro()
    assert pooled["MAE"] == pytest.approx((90 * 4 + 10 * 1) / 100)
    assert macro["MAE"] == pytest.approx((4 + 1) / 2)
    assert pooled["MAE"] > macro["MAE"] + 1.0

    # скилл: знаменатель 25 на каждой паре
    assert pooled["Skill"] == pytest.approx(1 - (90 * 16 + 10 * 1) / (100 * 25))
    assert macro["Skill"] == pytest.approx(((1 - 16 / 25) + (1 - 1 / 25)) / 2)


def test_macro_equals_pooled_when_stations_are_identical():
    ev = _case(seed=4, n_stations=1)
    pooled, macro = ev.pooled(), ev.macro()
    for m in METRICS:
        assert pooled[m] == pytest.approx(macro[m])


def test_summary_carries_window_and_station_counts():
    ev = _case(n_windows=60, n_stations=5, seed=5)
    s = ev.summary()
    assert s["n_windows"] == 60 and s["n_stations"] == 5
    sel = np.array([st == "s0" for st in ev.station])
    s0 = ev.restrict(windows=sel).summary()
    assert s0["n_stations"] == 1 and s0["n_windows"] == int(sel.sum())


def test_bootstrap_zero_interval_on_degenerate_data():
    """Все станции одинаковы → любой ресэмпл даёт то же число → интервал точка."""
    h = 4
    one = np.arange(h, dtype=np.float64)
    y = np.tile(one, (12, 1))
    mu = y + 1.0
    ev = Evaluation(y=y, mu=mu, q=mu[..., None] + np.linspace(-1, 1, NQ),
                    mu_clim=y + 2.0, w=np.ones_like(y),
                    station=np.array([f"s{i % 6}" for i in range(12)], object))
    ci = ev.bootstrap_ci(n_boot=200, seed=0)
    point = ev.pooled()
    for m in METRICS:
        lo, hi = ci["pooled"][m]
        assert hi - lo == pytest.approx(0.0, abs=1e-12)
        assert lo == pytest.approx(point[m])
        lo_m, hi_m = ci["macro"][m]
        assert hi_m - lo_m == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_single_station_interval_is_a_point():
    ev = _case(n_stations=1, seed=6)
    ci = ev.bootstrap_ci(n_boot=100, seed=0)
    assert ci["n_stations"] == 1
    for m in METRICS:
        lo, hi = ci["pooled"][m]
        assert hi - lo == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_interval_is_nonzero_and_contains_estimate():
    ev = _case(n_windows=400, n_stations=20, seed=7)
    ci = ev.bootstrap_ci(n_boot=500, seed=0, level=0.90)
    point = ev.pooled()
    assert ci["n_boot"] == 500 and ci["n_stations"] == 20
    for m in ("MAE", "RMSE", "CRPS", "Skill"):
        lo, hi = ci["pooled"][m]
        assert hi > lo, f"{m}: вырожденный интервал на неоднородных станциях"
        assert lo <= point[m] <= hi


def test_bootstrap_is_reproducible_by_seed():
    ev = _case(n_windows=200, n_stations=10, seed=8)
    a = ev.bootstrap_ci(n_boot=100, seed=3)
    b = ev.bootstrap_ci(n_boot=100, seed=3)
    c = ev.bootstrap_ci(n_boot=100, seed=4)
    assert a["pooled"]["MAE"] == b["pooled"]["MAE"]
    assert a["pooled"]["MAE"] != c["pooled"]["MAE"]


def test_zone_and_season_ids_stable_across_processes():
    """Разный PYTHONHASHSEED не должен менять ни одного идентификатора."""
    code = ("import json;"
            "from mayak.zones import koppen_id, season_id, KOPPEN_ZONES;"
            "print(json.dumps([[koppen_id(z) for z in KOPPEN_ZONES] + [koppen_id('UNK')],"
            "[season_id(m, lat) for m in range(1, 13) for lat in (52.0, -33.0)]]))")
    outs = []
    for hashseed in ("0", "1", "12345", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(REPO)}
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                           capture_output=True, text=True, check=True)
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, f"идентификаторы поехали между запусками: {set(outs)}"


def test_koppen_table_is_full_and_injective():
    ids = [koppen_id(z) for z in KOPPEN_ZONES]
    assert len(KOPPEN_ZONES) == 30
    assert sorted(ids) == list(range(30))
    assert koppen_id(UNKNOWN_ZONE) == 30
    assert koppen_id("Cfb") != koppen_id("Cfa"), "полная зона, а не первая буква"
    assert koppen_group("Cfb") == "C" and koppen_group("ET") == "E"


def test_unknown_and_truncated_zone_fall_back_to_unk():
    assert normalize_zone("  Cfb ") == "Cfb"
    assert normalize_zone("C") == UNKNOWN_ZONE
    assert normalize_zone("") == UNKNOWN_ZONE
    assert normalize_zone("Zzz") == UNKNOWN_ZONE


def test_kg_tif_codes_match_zone_order():
    assert KG_TIF_CODE[1] == "Af" and KG_TIF_CODE[30] == "EF"
    assert len(KG_TIF_CODE) == len(KOPPEN_ZONES)
    assert list(KG_TIF_CODE.values()) == list(KOPPEN_ZONES)


@pytest.mark.parametrize("month,north,south", [
    (1, "winter", "summer"), (2, "winter", "summer"), (3, "spring", "autumn"),
    (6, "summer", "winter"), (9, "autumn", "spring"), (12, "winter", "summer"),
])
def test_season_by_calendar_month_and_hemisphere(month, north, south):
    assert season_of(month, 52.0) == north
    assert season_of(month, -33.0) == south


def test_season_has_four_bins_and_no_collisions():
    got = {season_of(m, 52.0) for m in range(1, 13)}
    assert got == set(SEASONS)
    assert len({season_of(m, 52.0) for m in (12, 1, 2)}) == 1
    assert seasons_of([1, 4, 7, 10], 52.0).tolist() == ["winter", "spring", "summer", "autumn"]
    with pytest.raises(ValueError):
        season_of(0, 52.0)


def test_pit_histogram_matches_expectation_on_calibrated_forecast():
    ev = _calibrated(seed=11)
    pit = ev.pit_histogram()
    assert pit["expected"].sum() == pytest.approx(1.0)
    assert pit["observed"].sum() == pytest.approx(1.0)
    assert np.allclose(pit["observed"], pit["expected"], atol=0.01)


def test_pit_histogram_detects_overconfidence():
    """Слишком узкие интервалы → хвостовые бины тяжелее ожидаемого."""
    ev = _calibrated(seed=12)
    narrow = Evaluation(y=ev.y, mu=ev.mu, q=ev.mu[..., None] + 0.3 * (ev.q - ev.mu[..., None]),
                        mu_clim=ev.mu_clim, w=ev.w, station=ev.station)
    pit = narrow.pit_histogram()
    assert pit["observed"][0] > 3 * pit["expected"][0]
    assert pit["observed"][-1] > 3 * pit["expected"][-1]


def test_pit_by_lead_bin_covers_all_bins():
    ev = _calibrated(horizon=H, n=300, seed=13)
    out = ev.pit_by_lead_bin(LEAD_BINS)
    assert list(out) == [f"{a}-{b}" for a, b in LEAD_BINS]
    for p in out.values():
        assert p["n"] > 0 and p["observed"].sum() == pytest.approx(1.0)


def test_reliability_curve_matches_nominal_when_calibrated():
    ev = _calibrated(seed=14)
    rel = ev.reliability()
    assert np.allclose(rel["nominal"], Q)
    assert np.allclose(rel["empirical"], Q, atol=0.012)


def test_sharpness_coverage_is_monotone():
    ev = _calibrated(seed=15)
    rows = ev.sharpness_coverage()
    assert [r["nominal"] for r in rows] == [0.9, 0.8, 0.5]
    for r in rows:
        assert r["coverage"] == pytest.approx(r["nominal"], abs=0.02)
    assert rows[0]["width"] > rows[1]["width"] > rows[2]["width"]
    assert rows[0]["coverage"] > rows[1]["coverage"] > rows[2]["coverage"]


def test_reliability_respects_target_mask():
    """Мусор на невалидных часах не должен влиять ни на PIT, ни на надёжность."""
    ev = _case(seed=16)
    w = ev.w.copy()
    y = np.where(w > 0, ev.y, 1e6)
    dirty = Evaluation(y=y, mu=ev.mu, q=ev.q, mu_clim=ev.mu_clim, w=w, station=ev.station)
    clean = ev.restrict()
    assert np.allclose(dirty.pit_histogram()["observed"], clean.pit_histogram()["observed"])
    assert np.allclose(dirty.reliability()["empirical"], clean.reliability()["empirical"])
    for m in METRICS:
        assert dirty.pooled()[m] == pytest.approx(clean.pooled()[m])


def _reference_conformal(q, shift):
    q = np.array(q, np.float32, copy=True)
    for h in range(q.shape[-2]):
        q[..., h, :] += shift[lead_bin_index(h + 1)]
    return np.maximum.accumulate(q, axis=-1)


def test_apply_conformal_matches_reference_loop():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(0, 3, (17, H, NQ)), axis=-1).astype(np.float32)
    shift = rng.normal(0, 0.5, (len(LEAD_BINS), NQ)).astype(np.float32)
    assert np.allclose(apply_conformal(q, shift), _reference_conformal(q, shift))


def test_apply_conformal_works_on_single_forecast_like_runtime():
    """Рантайм подаёт (H, NQ) — та же функция обязана его принять."""
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(0, 3, (H, NQ)), axis=-1).astype(np.float32)
    shift = rng.normal(0, 0.5, (len(LEAD_BINS), NQ)).astype(np.float32)
    one = apply_conformal(q, shift)
    batch = apply_conformal(q[None], shift)
    assert np.allclose(one, batch[0])
    assert np.allclose(one, _reference_conformal(q, shift))


def test_conformal_keeps_quantiles_monotone():
    rng = np.random.default_rng(2)
    q = np.sort(rng.normal(0, 3, (50, H, NQ)), axis=-1).astype(np.float32)
    shift = rng.normal(0, 2.0, (len(LEAD_BINS), NQ)).astype(np.float32)
    out = apply_conformal(q, shift)
    assert np.all(np.diff(out, axis=-1) >= 0)


def test_conformal_table_rejects_wrong_shape():
    with pytest.raises(ValueError, match="бинов"):
        conformal_table(np.zeros((2, NQ), np.float32))
    with pytest.raises(ValueError, match="поправок формы"):
        conformal_table(np.zeros((len(LEAD_BINS), NQ + 1), np.float32))


def test_all_modules_share_one_conformal_implementation():
    import mayak.evaluate as E
    import mayak.runtime.streaming as R
    from mayak import metrics as M
    calibrate = _load_module(REPO / "scripts" / "calibrate.py", "calibrate_for_test")
    assert E.apply_conformal is M.apply_conformal
    assert R.apply_conformal is M.apply_conformal
    assert calibrate.apply_conformal is M.apply_conformal


def test_with_conformal_sets_median_from_quantiles():
    ev = _case(n_windows=20, seed=17)
    shift = np.zeros((len(LEAD_BINS), NQ), np.float32)
    shift[:, :] = 1.5
    out = ev.with_conformal(shift)
    assert np.allclose(out.mu, out.q[..., 3])
    assert np.allclose(out.q, apply_conformal(ev.q, shift))
    assert ev.with_conformal(None) is ev


def test_fit_conformal_shift_uses_valid_hours_only():
    rng = np.random.default_rng(3)
    n = 400
    q = np.zeros((n, H, NQ), np.float32) + np.linspace(-2, 2, NQ)
    y = rng.normal(0, 1, (n, H)).astype(np.float32)
    w = (rng.random((n, H)) < 0.6).astype(np.float32)
    dirty = np.where(w > 0, y, 1e6).astype(np.float32)
    assert np.allclose(fit_conformal_shift(y, q, w), fit_conformal_shift(dirty, q, w))


def test_breakdown_drops_small_strata_and_counts_rows():
    ev = _case(n_windows=120, n_stations=6, seed=18)
    keys = np.array(["крупная"] * 100 + ["мелкая"] * 20, object)
    rows = breakdown(ev, keys, min_windows=30, min_stations=1)
    assert set(rows) == {"крупная"}
    assert rows["крупная"]["n_windows"] == 100

    rows = breakdown(ev, keys, min_windows=10, min_stations=1)
    assert set(rows) == {"крупная", "мелкая"}
    assert rows["мелкая"]["n_windows"] == 20
    for r in rows.values():
        assert r["n_stations"] >= 1


def test_breakdown_row_matches_direct_restriction():
    ev = _case(n_windows=90, n_stations=5, seed=19)
    keys = np.array([f"g{i % 3}" for i in range(90)], object)
    rows = breakdown(ev, keys, min_windows=1, min_stations=1)
    for g in ("g0", "g1", "g2"):
        sel = keys == g
        assert rows[g]["pooled"]["Skill"] == pytest.approx(_manual_skill(ev, sel))


def test_breakdown_rejects_misaligned_keys():
    ev = _case(n_windows=10, seed=20)
    with pytest.raises(ValueError, match="меток"):
        breakdown(ev, np.array(["a"] * 9, object))


def test_seed_spread_reports_mean_and_range():
    evs = [_case(seed=s, n_windows=80) for s in (21, 22, 23)]
    summaries = [ev.summary() for ev in evs]
    sp = seed_spread(summaries)
    maes = [s["pooled"]["MAE"] for s in summaries]
    assert sp["MAE"]["n"] == 3
    assert sp["MAE"]["mean"] == pytest.approx(float(np.mean(maes)))
    assert sp["MAE"]["min"] == pytest.approx(min(maes))
    assert sp["MAE"]["max"] == pytest.approx(max(maes))
    assert spread([1.0, 1.0])["std"] == pytest.approx(0.0)
    assert spread([])["n"] == 0


def test_stratified_subsample_keeps_every_station():
    from mayak.evaluate import stratified_items
    per_station = {"many": list(range(0, 1000, 10)), "few": [0, 5, 10, 15, 20]}
    items = stratified_items(per_station, max_windows=20)
    got = {sid: sum(1 for s, _ in items if s == sid) for sid in per_station}
    assert got["few"] == 5, "маленькая станция не должна пропадать целиком"
    assert got["many"] == 10, "квота на станцию, а не шаг по общему списку"

    flat = [(s, t) for s, ts in per_station.items() for t in ts]
    step = len(flat) // 20
    old = {sid: sum(1 for s, _ in flat[::step][:20] if s == sid) for sid in per_station}
    assert old["few"] < got["few"]


def test_stratified_subsample_is_spread_over_time():
    from mayak.evaluate import stratified_items
    items = stratified_items({"a": list(range(100))}, windows_per_station=5)
    ts = [t for _s, t in items]
    assert ts == [0, 25, 50, 74, 99]


def test_stratified_subsample_keeps_everything_without_cap():
    from mayak.evaluate import stratified_items
    per_station = {"a": [1, 2, 3], "b": [4, 5]}
    assert len(stratified_items(per_station)) == 5
    assert stratified_items({}) == []
    assert stratified_items({"a": []}) == []


def _write_station(root, sid, seed, n=N_HOURS):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


STATIONS = [("t0", ROLE_TRAIN, 52.0, "Cfb"), ("t1", ROLE_TRAIN, -33.0, "Csb"),
            ("v0", ROLE_VAL, 60.0, "Dfc"), ("x0", ROLE_TEST, 10.0, "Af")]


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data5")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role, lat, zone) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=lat, lon=5.0 * i, elev=100.0, koppen=zone, split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def store(manifest):
    S._STORES.clear()
    return S.get_store(manifest)


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_eval_batch_has_no_unstable_ids(store, manifest):
    from mayak.evaluate import EvalSet
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                 time_key="test")
    item = ds[0]
    for key in ("koppen_id", "season_id", "seen"):
        assert key not in item, f"{key}: поле батча никем не читается и строилось хешем"


def test_window_meta_is_aligned_and_labelled(store, manifest):
    from mayak.evaluate import EvalSet
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                 time_key="test")
    meta = ds.window_meta()
    assert all(len(v) == len(ds) for v in meta.values())
    assert list(meta["station"]) == [sid for sid, _t in ds.items]
    assert set(meta["role"]) <= {ROLE_TRAIN, ROLE_TEST}
    assert set(meta["zone"]) <= set(KOPPEN_ZONES) | {UNKNOWN_ZONE}
    assert (meta["history"] >= 0).all()
    assert ((meta["hist_valid"] >= 0) & (meta["hist_valid"] <= 1)).all()


def test_window_meta_season_flips_for_southern_station(store, manifest):
    """t0 и t1 стоят на одних и тех же часах, но t1 в южном полушарии."""
    from mayak.evaluate import EvalSet
    from mayak.zones import SEASON_RU
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN,), manifest=manifest, time_key="test")
    meta = ds.window_meta()
    by_t = {}
    for sid, t, season in zip(meta["station"], [t for _s, t in ds.items], meta["season"]):
        by_t.setdefault(t, {})[sid] = season
    pairs = [v for v in by_t.values() if len(v) == 2]
    assert pairs, "нужны окна с одинаковым t на обеих станциях"
    flip = {SEASON_RU["winter"]: SEASON_RU["summer"], SEASON_RU["summer"]: SEASON_RU["winter"],
            SEASON_RU["spring"]: SEASON_RU["autumn"], SEASON_RU["autumn"]: SEASON_RU["spring"]}
    for v in pairs:
        assert v["t1"] == flip[v["t0"]]


def test_eval_set_subsample_is_stratified(store, manifest):
    from mayak.evaluate import EvalSet
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                 time_key="test", every_hours=24, windows_per_station=3)
    counts = {}
    for sid, _t in ds.items:
        counts[sid] = counts.get(sid, 0) + 1
    assert set(counts) == {"t0", "t1", "x0"}
    assert set(counts.values()) == {3}


def test_evaluation_from_eval_set_breakdowns_run(store, manifest):
    """Сквозная проверка: окна → метаданные → разрезы, без модели."""
    from mayak.evaluate import EvalSet, all_breakdowns
    from mayak.metrics import Evaluation as Ev
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                 time_key="test", every_hours=24, windows_per_station=20)
    meta = ds.window_meta()
    rng = np.random.default_rng(0)
    n = len(ds)
    y = rng.normal(10, 5, (n, H))
    mu = y + rng.normal(0, 1, (n, H))
    ev = Ev(y=y, mu=mu, q=mu[..., None] + np.linspace(-3, 3, NQ), mu_clim=y + rng.normal(0, 3, (n, H)),
            w=np.ones((n, H)), station=meta["station"])
    out = all_breakdowns(ev, meta, leads=[24], min_windows=1, min_stations=1)
    assert set(out) == {"роль станции", "зона Кёппена", "сезон", "длина истории",
                        "валидность истории"}
    for name, rows in out.items():
        assert rows, f"разрез {name} пуст"
        for r in rows.values():
            assert r["n_windows"] > 0 and r["n_stations"] > 0
            assert np.isfinite(r["pooled"]["MAE"]) and np.isfinite(r["macro"]["MAE"])
