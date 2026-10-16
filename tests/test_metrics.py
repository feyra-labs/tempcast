"""Тесты: знаменатели, макро-оценка, надёжность, значимость, ядро калибровки."""
import csv

import numpy as np
import pytest

from mayak.constants import H, QUANTILES
from mayak.data import store as S
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from mayak.metrics import (ABLATION_CELLS, HISTORY_BINS, I_MED, LEAD_BINS, METRICS, NQ,
                           ACIParams, AdaptiveCalibration, Evaluation, ablation_significance,
                           apply_conformal, breakdown, by_lead, calibrate_forecast,
                           conformal_table, fit_conformal_shift, lead_bin_index, lead_bin_of,
                           metric_table, seed_spread, significance_verdict, spread)
from mayak.zones import KOPPEN_ZONES, UNKNOWN_ZONE, normalize_zone, season_of

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


def test_bootstrap_zero_interval_on_degenerate_data():
    """Все станции одинаковы, поэтому любой ресэмпл даёт то же число и интервал сжат в точку."""
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


def test_unknown_and_truncated_zone_fall_back_to_unk():
    assert normalize_zone("  Cfb ") == "Cfb"
    assert normalize_zone("C") == UNKNOWN_ZONE
    assert normalize_zone("") == UNKNOWN_ZONE
    assert normalize_zone("Zzz") == UNKNOWN_ZONE


@pytest.mark.parametrize("month,north,south", [
    (1, "winter", "summer"), (2, "winter", "summer"), (3, "spring", "autumn"),
    (6, "summer", "winter"), (9, "autumn", "spring"), (12, "winter", "summer"),
])
def test_season_by_calendar_month_and_hemisphere(month, north, south):
    assert season_of(month, 52.0) == north
    assert season_of(month, -33.0) == south


def test_pit_histogram_matches_expectation_on_calibrated_forecast():
    ev = _calibrated(seed=11)
    pit = ev.pit_histogram()
    assert pit["expected"].sum() == pytest.approx(1.0)
    assert pit["observed"].sum() == pytest.approx(1.0)
    assert np.allclose(pit["observed"], pit["expected"], atol=0.01)


def test_pit_histogram_detects_overconfidence():
    """Слишком узкие интервалы делают хвостовые бины тяжелее ожидаемого."""
    ev = _calibrated(seed=12)
    narrow = Evaluation(y=ev.y, mu=ev.mu, q=ev.mu[..., None] + 0.3 * (ev.q - ev.mu[..., None]),
                        mu_clim=ev.mu_clim, w=ev.w, station=ev.station)
    pit = narrow.pit_histogram()
    assert pit["observed"][0] > 3 * pit["expected"][0]
    assert pit["observed"][-1] > 3 * pit["expected"][-1]


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


def _reference_conformal(q, shift, history):
    """Поправка по лидам и бину длины истории, порядок квантилей от медианы наружу, циклом."""
    from mayak.metrics import history_bin_of
    q = np.array(q, np.float32, copy=True)
    hist = np.broadcast_to(np.asarray(history), q.shape[:-2])
    for idx in np.ndindex(*q.shape[:-2]):
        hb = int(history_bin_of(int(hist[idx])))
        for h in range(q.shape[-2]):
            q[idx + (h,)] += shift[lead_bin_index(h + 1), hb]
    for i in range(I_MED - 1, -1, -1):
        q[..., i] = np.minimum(q[..., i], q[..., i + 1])
    for i in range(I_MED + 1, NQ):
        q[..., i] = np.maximum(q[..., i], q[..., i - 1])
    return q


def _median_free_shift(rng, scale):
    """Случайная таблица поправок с нулевой поправкой медианы, как у подгонки."""
    shift = rng.normal(0, scale, (len(LEAD_BINS), len(HISTORY_BINS), NQ)).astype(np.float32)
    shift[..., I_MED] = 0.0
    return shift


def test_apply_conformal_matches_reference_loop():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(0, 3, (17, H, NQ)), axis=-1).astype(np.float32)
    shift = _median_free_shift(rng, 0.5)
    hist = rng.choice([0, 1, 24, 25, 168, 169, 672], 17)
    assert np.allclose(apply_conformal(q, shift, hist), _reference_conformal(q, shift, hist))


def test_apply_conformal_works_on_single_forecast_like_runtime():
    """Рантайм подаёт (H, NQ) и одну длину истории — та же функция обязана его принять."""
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(0, 3, (H, NQ)), axis=-1).astype(np.float32)
    shift = _median_free_shift(rng, 0.5)
    one = apply_conformal(q, shift, 30)
    batch = apply_conformal(q[None], shift, [30])
    assert np.allclose(one, batch[0])
    assert np.allclose(one, _reference_conformal(q, shift, 30))


def test_conformal_row_follows_history_bin():
    rng = np.random.default_rng(4)
    q = np.sort(rng.normal(0, 3, (H, NQ)), axis=-1).astype(np.float32)
    shift = _median_free_shift(rng, 1.0)
    for k, (lo, hi, _name) in enumerate(HISTORY_BINS):
        for L in (lo, hi):
            got = conformal_table(shift, L)
            assert np.array_equal(got, shift[lead_bin_of(H), k]), (k, L)
    assert np.array_equal(conformal_table(shift, 10_000), shift[lead_bin_of(H), -1])
    with pytest.raises(ValueError, match="не задана"):
        apply_conformal(q, shift, None)
    with pytest.raises(ValueError, match="длин истории"):
        apply_conformal(q[None].repeat(3, 0), shift, [1, 2])


def test_conformal_keeps_quantiles_monotone():
    rng = np.random.default_rng(2)
    q = np.sort(rng.normal(0, 3, (50, H, NQ)), axis=-1).astype(np.float32)
    shift = _median_free_shift(rng, 2.0)
    out = apply_conformal(q, shift, np.arange(50) * 13)
    assert np.all(np.diff(out, axis=-1) >= 0)
    assert np.array_equal(out[..., I_MED], q[..., I_MED])


def test_with_conformal_sets_median_from_quantiles():
    ev = _case(n_windows=20, seed=17)
    shift = np.full((len(LEAD_BINS), len(HISTORY_BINS), NQ), 1.5, np.float32)
    shift[..., :I_MED] = -1.5
    shift[..., I_MED] = 0.0
    hist = np.arange(20) * 30
    out = ev.with_conformal(shift, hist)
    assert np.allclose(out.mu, out.q[..., 3])
    assert np.allclose(out.q, apply_conformal(ev.q, shift, hist))
    assert ev.with_conformal(None, hist) is ev


def test_fit_conformal_shift_uses_valid_hours_only():
    rng = np.random.default_rng(3)
    n = 400
    q = np.zeros((n, H, NQ), np.float32) + np.linspace(-2, 2, NQ)
    y = rng.normal(0, 1, (n, H)).astype(np.float32)
    w = (rng.random((n, H)) < 0.6).astype(np.float32)
    hist = rng.choice([0, 12, 100, 672], n)
    dirty = np.where(w > 0, y, 1e6).astype(np.float32)
    assert np.allclose(fit_conformal_shift(y, q, w, hist)[0],
                       fit_conformal_shift(dirty, q, w, hist)[0])


def test_fitted_median_column_is_zero_and_table_keeps_median():
    """Подогнанная таблица не сдвигает медиану; таблица и множитель ACI её не трогают."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 400
    mu = rng.normal(0, 3, (n, H))
    q = (mu[..., None] + norm.ppf(Q)).astype(np.float32)
    y = (mu + 1.5 * rng.standard_normal((n, H))).astype(np.float32)
    hist = rng.choice([0, 5, 24, 100, 168, 300, 672], n)
    shift, rows = fit_conformal_shift(y, q, np.ones((n, H), np.float32), hist)
    assert shift.shape == (len(LEAD_BINS), len(HISTORY_BINS), NQ)
    assert not np.any(shift[..., I_MED])
    assert (shift[..., 0] < 0).all() and (shift[..., -1] > 0).all(), "узкий прогноз расширяется"
    assert not any(r["marginal"] for r in rows) and sum(r["windows"] for r in rows) == n
    out = apply_conformal(q, shift, hist)
    assert np.array_equal(out[..., I_MED], q[..., I_MED])
    for theta in (-0.4, 0.0, 0.6, (0.2, -0.1, 0.0, 0.5)):
        cq, med = calibrate_forecast(q, shift, theta, hist)
        assert np.array_equal(med, q[..., I_MED]) and np.array_equal(cq[..., I_MED], med)


def test_aci_step_follows_miss_indicator():
    """Шаг ACI: промах расширяет на γ(1 − α), попадание сужает на γα, θ во float32 и в границах."""
    p = ACIParams()
    th, miss = p.step(0.0, 2.0)
    assert miss and th == pytest.approx(p.gamma * (1 - p.target), rel=1e-6)
    th, miss = p.step(0.0, 0.5)
    assert not miss and th == pytest.approx(-p.gamma * p.target, rel=1e-6)
    assert th == float(np.float32(th)), "θ живёт во float32, как в состоянии"
    assert p.update(p.theta_max, True) == p.theta_max
    assert p.update(p.theta_min, False) == p.theta_min


def test_aci_hourly_feedback_covers_bin_leads_evenly():
    """Ежечасный выпуск: каждый час - по связи на бин, лиды бина получают её поровну.

    После прогрева в горизонт каждый валидный час обновляет каждый бин ровно один раз, а
    число обратных связей по лидам внутри бина различается не больше чем на единицу.
    В медиане лида записан его номер, нижняя граница интервала на единицу ниже, поэтому
    оценка факта, равного нулю, - это номер лида записи.
    """
    cal = AdaptiveCalibration(ACIParams(), H, LEAD_BINS)
    q = (np.arange(1, H + 1)[:, None] + np.linspace(-1.0, 1.0, NQ)).astype(np.float32)
    t0, n = 1_000_003, H + 500
    seen = np.zeros(H + 1, np.int64)
    for t in range(t0, t0 + n):
        got, before = cal.scores(0.0, t), list(cal.updates)
        cal.feedback(0.0, t)
        if t >= t0 + H:
            assert sorted(b for b, _ in got) == list(range(len(LEAD_BINS))), t
            assert [a - b for a, b in zip(cal.updates, before)] == [1] * len(LEAD_BINS)
            for _, lead in got:
                seen[int(lead)] += 1
        cal.record(t, q)
    for lo, hi in LEAD_BINS:
        c = seen[lo:hi + 1]
        assert c.min() > 0 and c.max() - c.min() <= 1, f"бин {lo}-{hi}: {c.tolist()}"


def test_metrics_match_manual_on_valid_pairs():
    """Метрики лида - по валидным парам; CRPS точечного прогноза равен его MAE."""
    rng = np.random.default_rng(1)
    n = 400
    y = 10 + 5 * rng.standard_normal((n, H))
    mu = y + rng.standard_normal((n, H))
    mu_clim = y + 3 * rng.standard_normal((n, H))
    q = mu[..., None] + np.linspace(-2, 2, NQ)
    w = (rng.random((n, H)) < 0.5).astype(np.float32)
    w[:, 0] = 1
    tbl = metric_table(y, mu, q, mu_clim, w, leads=(24,))[24]
    k = w[:, 23] > 0
    e = mu[k, 23] - y[k, 23]
    assert tbl["MAE"] == pytest.approx(np.abs(e).mean())
    assert tbl["RMSE"] == pytest.approx(np.sqrt((e ** 2).mean()))
    mse_c = ((mu_clim[k, 23] - y[k, 23]) ** 2).mean()
    assert tbl["Skill"] == pytest.approx(1 - (e ** 2).mean() / mse_c)
    lo, hi = q[k, 23, 0], q[k, 23, 6]
    assert tbl["PICP90"] == pytest.approx(((y[k, 23] >= lo) & (y[k, 23] <= hi)).mean())
    assert tbl["n_valid"] == int(k.sum())
    point = metric_table(y, mu, np.repeat(mu[..., None], NQ, -1), mu_clim, w, leads=(24,))[24]
    assert point["CRPS"] == pytest.approx(point["MAE"])


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


def _lead_summaries(skill24):
    """Сводки по лидам таблицы абляций: меняется только Skill@24, остальные ячейки равны."""
    out = {h: {"pooled": {"Skill": 0.5, "CRPS": 1.0, "PICP90": 0.9}}
           for h in {h for _m, h in ABLATION_CELLS}}
    out[24]["pooled"]["Skill"] = skill24
    return out


def test_ablation_significance_needs_twice_seed_range_and_one_sign():
    seeds = [_lead_summaries(v) for v in (0.50, 0.51, 0.49)]  # R = 0.02
    internal = ablation_significance(seeds, {"a": _lead_summaries(0.45),
                                             "b": _lead_summaries(0.47),
                                             "c": _lead_summaries(0.44)})
    assert internal["spread"]["Skill@24"] == pytest.approx(0.02)
    assert internal["rows"]["a"]["Skill@24"]["delta"] == pytest.approx(-0.05)
    assert internal["rows"]["a"]["Skill@24"]["expressed"]
    assert not internal["rows"]["b"]["Skill@24"]["expressed"], "|Δ| > R, но не больше 2R"
    assert not internal["rows"]["a"]["Skill@72"]["expressed"], "Δ = 0 при нулевом размахе"

    external = ablation_significance(seeds, {"a": _lead_summaries(0.56),
                                             "b": _lead_summaries(0.40),
                                             "c": _lead_summaries(0.45)})
    assert external["rows"]["a"]["Skill@24"]["expressed"]
    verdict = significance_verdict(internal, external)
    assert not verdict["a"]["Skill@24"], "выражена на обоих наборах, но знаки Δ разные"
    assert not verdict["b"]["Skill@24"], "на одном наборе не выражена"
    assert verdict["c"]["Skill@24"]
    assert not any(verdict["c"][c] for c in verdict["c"] if c != "Skill@24")
    with pytest.raises(ValueError):
        ablation_significance(seeds[:2], {"a": _lead_summaries(0.45)})


def test_stratified_subsample_keeps_every_station():
    from mayak.data.holdout import stratified_items
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
    from mayak.data.holdout import stratified_items
    items = stratified_items({"a": list(range(100))}, windows_per_station=5)
    ts = [t for _s, t in items]
    assert ts == [0, 25, 50, 74, 99]


def test_stratified_subsample_keeps_everything_without_cap():
    from mayak.data.holdout import stratified_items
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
