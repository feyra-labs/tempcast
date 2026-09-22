"""Тесты: калибровка интервалов.

Что проверяется:
* одна реализация применения калибровки (сплит-конформная таблица + адаптивный
  множитель) на весь проект - оценка, графики и рантайм пользуются ``mayak.metrics``;
* адаптивная калибровка (ACI): сходится к заданному покрытию на синтетическом потоке, в
  том числе после сдвига распределения; не нарушает монотонность квантилей; не
  меняется и не расходится при длинной серии пропусков; ограничена при сплошных
  промахах; рантайм даёт ту же траекторию θ, что офлайн-прогон;
* θ - часть персистентного состояния (+4 Б, формат v3, v2 читается с θ = 0);
* кривая «острота против покрытия» по уже собранным предсказаниям;
* разрезы покрытия с вердиктами и критерий условной поправки на
  сконструированных примерах;
* сохранение предсказаний, конфиг, точки входа.
"""
import csv
import json
import math
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.stats import norm

from mayak.config import (COVERAGE_DIMS_EXTERNAL, COVERAGE_DIMS_INTERNAL, CalibrationConfig,
                          ConfigError)
from mayak.constants import H, QUANTILES
from mayak.data import store as S
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN
from mayak.metrics import (I_HI90, I_LO90, I_MED, LEAD_BINS, NQ, ACIParams, Evaluation,
                           aci_effective_level, aci_run, aci_score, apply_adaptive,
                           apply_conformal, calibrate_forecast, inside, width_at_coverage)
from mayak.runtime.equivalence import feed, future_calendar_after, synthetic_series
from mayak.runtime.streaming import (STATE_HEADER, STATE_HEADERS, STATE_VERSION,
                                     StreamingMayak)

REPO = Path(__file__).resolve().parents[1]
Z = norm.ppf(np.asarray(QUANTILES, np.float64))
LAT, LON, ELEV = 52.37, 4.9, 0.0
DEFAULT_STATE_BYTES = 3352
N_HOURS = 12_000


def _gauss_q(mu, sigma=1.0):
    return np.asarray(mu, np.float64)[..., None] + np.asarray(sigma, np.float64)[..., None] * Z


def _stream(sigma_true, n, seed=0, sigma_pred=1.0):
    """Поток оценок ACI: прогноз N(0, sigma_pred), факт N(0, sigma_true[t])."""
    rng = np.random.default_rng(seed)
    sig = np.broadcast_to(np.asarray(sigma_true, np.float64), (n,))
    y = sig * rng.standard_normal(n)
    q = _gauss_q(np.zeros(n), np.full(n, sigma_pred))
    return aci_score(y, q)


def _dataset(n_st=40, per=12, spacing=72, sigma=None, bias=None, seed=0, horizon=H):
    """Сохранённые «предсказания» с метаданными окон, как у mayak.evaluate.

    Станции s00 … : первая половина - зона Cfb, вторая - BWh. sigma / bias - функции
    зоны → истинный разброс и сдвиг факта относительно прогноза N(mu, 1).
    """
    rng = np.random.default_rng(seed)
    sigma = sigma or (lambda z: 1.0)
    bias = bias or (lambda z: 0.0)
    st = np.repeat([f"s{i:02d}" for i in range(n_st)], per)
    zone = np.where(np.repeat(np.arange(n_st), per) < n_st // 2, "Cfb", "BWh")
    n = len(st)
    sig = np.array([sigma(z) for z in zone])[:, None]
    b = np.array([bias(z) for z in zone])[:, None]
    mu = rng.normal(10, 5, (n, horizon))
    y = mu + b + sig * rng.standard_normal((n, horizon))
    meta = dict(station=st.astype(object),
                role=np.where(rng.random(n) < 0.5, ROLE_TRAIN, ROLE_TEST).astype(object),
                zone=zone.astype(object), season=np.full(n, "зима", object),
                history=rng.choice([0, 12, 100, 500], n).astype(np.int64),
                hist_valid=rng.uniform(0.3, 1.0, n),
                has_pressure=np.full(n, "есть давление", object),
                report_class=np.full(n, "1ч", object), elev_gap=np.full(n, "|Δh| <50 м", object),
                t=np.tile(np.arange(per, dtype=np.int64) * spacing, n_st))
    aux = dict(y=y, y_mask=np.ones_like(y), mu_clim=mu + rng.normal(0, 3, (n, horizon)),
               meta=meta)
    preds = {"МАЯК": dict(mu=mu, q=_gauss_q(mu)),
             "Климатология": dict(mu=aux["mu_clim"], q=_gauss_q(aux["mu_clim"], 3.0))}
    return preds, aux


def _ev(pred, aux):
    return Evaluation(y=aux["y"], mu=pred["mu"], q=pred["q"], mu_clim=aux["mu_clim"],
                      w=aux["y_mask"], station=aux["meta"]["station"])


def _cfg(**kw):
    return CalibrationConfig(**{**dict(bootstrap=200), **kw})


@pytest.fixture(scope="module")
def model():
    from mayak.model import MAYAK
    torch.manual_seed(0)
    return MAYAK().eval()


def _shift(seed=0):
    return np.random.default_rng(seed).normal(0, 0.4, (len(LEAD_BINS), NQ)).astype(np.float32)


def test_apply_adaptive_zero_is_identity_and_keeps_median():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(0, 3, (20, H, NQ)), axis=-1).astype(np.float32)
    q0, mu0 = apply_adaptive(q, 0.0)
    assert q0.dtype == np.float32 and np.array_equal(q0, q)
    assert np.array_equal(mu0, q[..., I_MED])
    for theta in (-1.0, 0.3, 1.2):
        qt, mut = apply_adaptive(q, theta)
        assert np.array_equal(mut, q[..., I_MED]), "медиана поправкой не меняется"
        med = q[..., I_MED:I_MED + 1].astype(np.float64)
        np.testing.assert_allclose(qt - med, math.exp(theta) * (q - med), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="θ"):
        apply_adaptive(q, float("nan"))


@pytest.mark.parametrize("theta", [-5.0, -1.3862944, -0.2, 0.0, 0.7, 1.3862944, 5.0])
def test_adaptive_never_breaks_quantile_order(theta):
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(0, 3, (30, H, NQ)), axis=-1).astype(np.float32)
    q[..., :I_MED] = q[..., I_MED:I_MED + 1]
    out, _ = apply_adaptive(q, theta)
    assert np.isfinite(out).all() and (np.diff(out, axis=-1) >= 0).all()
    messy = rng.normal(0, 3, (30, H, NQ)).astype(np.float32)
    out, _ = calibrate_forecast(messy, _shift(1), theta)
    assert (np.diff(out, axis=-1) >= 0).all()


def test_calibrate_forecast_is_conformal_then_adaptive():
    rng = np.random.default_rng(2)
    q = np.sort(rng.normal(0, 3, (7, H, NQ)), axis=-1).astype(np.float32)
    shift = _shift(2)
    got_q, got_mu = calibrate_forecast(q, shift, 0.4)
    ref_q, ref_mu = apply_adaptive(apply_conformal(q, shift), 0.4)
    assert np.array_equal(got_q, ref_q) and np.array_equal(got_mu, ref_mu)
    same_q, _ = calibrate_forecast(q)
    assert np.array_equal(same_q, q)


def test_evaluation_with_calibration_uses_the_same_function():
    preds, aux = _dataset(n_st=6, per=4)
    ev = _ev(preds["МАЯК"], aux)
    assert ev.with_calibration() is ev and ev.with_conformal(None) is ev
    shift = _shift(3)
    cal = ev.with_calibration(shift, 0.3)
    q, mu = calibrate_forecast(ev.q, shift, 0.3)
    assert np.array_equal(cal.q, q) and np.array_equal(cal.mu, mu)
    wider = ev.with_calibration(None, 0.5).pooled()["PICP90"]
    assert wider > ev.pooled()["PICP90"] > ev.with_calibration(None, -0.5).pooled()["PICP90"]


def test_all_modules_share_one_calibration_implementation():
    import mayak.evaluate as E
    import mayak.runtime.streaming as R
    from mayak import metrics as M
    assert E.calibrate_forecast is M.calibrate_forecast
    for name in ("apply_conformal", "apply_adaptive", "aci_score", "ACIParams"):
        assert getattr(R, name) is getattr(M, name), name
    offenders = []
    for p in (REPO / "mayak").rglob("*.py"):
        if p.name == "metrics.py":
            continue
        src = p.read_text(encoding="utf-8")
        if "maximum.accumulate" in src or "q[:, 3]" in src or "q[:, I_MED]" in src:
            offenders.append(str(p.relative_to(REPO)))
    assert not offenders, f"копии применения поправки вне mayak/metrics.py: {offenders}"


def test_aci_score_agrees_with_adapted_interval():
    rng = np.random.default_rng(4)
    n = 20_000
    q = np.sort(rng.normal(0, 3, (n, NQ)), axis=-1).astype(np.float32)
    y = rng.normal(0, 4, n)
    s = aci_score(y, q)
    for theta in (-0.8, 0.0, 0.6):
        qa, _ = apply_adaptive(q, theta)
        lo, hi = qa[:, I_LO90].astype(np.float64), qa[:, I_HI90].astype(np.float64)
        far = np.minimum(np.abs(y - lo), np.abs(y - hi)) > 1e-4
        assert np.array_equal((s <= math.exp(theta))[far], inside(y, lo, hi)[far].astype(bool))


def test_aci_score_edge_cases():
    q = np.array([0, 0, 0, 0, 1, 2, 3], np.float64)
    assert aci_score(0.0, q) == 0.0
    assert aci_score(1.5, q) == pytest.approx(0.5)
    assert np.isinf(aci_score(-1.0, q))
    assert np.isnan(aci_score(float("nan"), q))


def test_aci_params_validation_and_update_rule():
    p = ACIParams()
    assert p.interval == (I_LO90, I_HI90)
    assert ACIParams(target=0.2).interval == (1, 5)
    for bad in (dict(target=0.07), dict(gamma=0.0), dict(gamma=1.0), dict(max_factor=1.0),
                dict(max_factor=float("inf"))):
        with pytest.raises(ValueError):
            ACIParams(**bad)
    th, miss = p.step(0.0, 2.0)
    assert miss and th == pytest.approx(p.gamma * (1 - p.target), rel=1e-6)
    th, miss = p.step(0.0, 0.5)
    assert not miss and th == pytest.approx(-p.gamma * p.target, rel=1e-6)
    assert th == float(np.float32(th)), "θ живёт во float32, как в состоянии"
    assert p.update(p.theta_max, True) == p.theta_max
    assert p.update(p.theta_min, False) == p.theta_min
    assert aci_effective_level(0.0) == pytest.approx(0.9)
    assert aci_effective_level(0.5) > 0.9 > aci_effective_level(-0.5)


@pytest.mark.parametrize("sigma_true", [1.6, 0.6])
def test_aci_converges_to_target_coverage(sigma_true):
    """Модель ошибается в разбросе в sigma_true раз: θ → ln(sigma_true), покрытие → 90 %."""
    p = ACIParams(gamma=0.01)
    n = 20_000
    r = aci_run(_stream(sigma_true, n, seed=5), p)
    assert r["clipped"] == 0
    miss = r["miss"]
    assert miss.mean() - p.target == pytest.approx(r["theta_end"] / (p.gamma * n), abs=1e-4)
    assert abs(miss.mean() - p.target) < 0.01
    assert abs(miss[n // 2:].mean() - p.target) < 0.015
    assert r["theta_end"] == pytest.approx(math.log(sigma_true), abs=0.15)
    base_cov = 1 - (_stream(sigma_true, n, seed=5) > 1.0).mean()
    assert abs(base_cov - 0.9) > 0.05, "без ACI покрытие далеко от номинала"


def test_aci_recovers_after_distribution_shift():
    p = ACIParams(gamma=0.01)
    sig = np.r_[np.full(10_000, 1.0), np.full(10_000, 2.0)]
    r = aci_run(_stream(sig, len(sig), seed=6), p)
    tail = r["miss"][-5000:]
    assert abs(tail.mean() - p.target) < 0.02
    assert r["theta_end"] == pytest.approx(math.log(2.0), abs=0.2)


def test_aci_frozen_without_feedback_and_bounded_under_constant_misses():
    p = ACIParams(gamma=0.01)
    s = np.r_[_stream(1.3, 3000, seed=7), np.full(50_000, np.nan),
    _stream(1.3, 3000, seed=8)]
    r = aci_run(s, p)
    gap = slice(3000, 53_000)
    assert np.all(r["theta"][gap] == r["theta"][3000]), "без обратной связи θ заморожен"
    assert np.isnan(r["miss"][gap]).all()
    assert np.isfinite(r["theta"]).all()
    bad = aci_run(np.full(5000, np.inf), p)
    assert bad["theta_end"] == p.theta_max and bad["clipped"] > 0
    assert np.all(bad["theta"] <= p.theta_max)
    back = aci_run(_stream(1.0, 20_000, seed=9), p, theta0=bad["theta_end"])
    assert abs(back["theta_end"]) < 0.2, "после восстановления прибора θ возвращается"


def test_runtime_theta_matches_offline_run(model):
    p = ACIParams(gamma=0.05)
    n = 150
    s = synthetic_series(n + 1, seed=3)
    st = StreamingMayak(model, LAT, LON, ELEV, conformal=_shift(4), aci=p)
    scores = []
    for k in range(n):
        if k and st._pending is not None:
            y = float(s["x"][k, 0]) if s["m"][k, 0] > 0 else float("nan")
            scores.append(float(aci_score(y, st._pending["q"][0], p.interval)))
        feed(st, s, k, k + 1)
        q, _ = st.forecast(*future_calendar_after(s, k + 1, H))
        assert np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()
    r = aci_run(scores, p)
    assert st.theta == r["theta_end"] != 0.0
    assert st.aci_updates == int(np.isfinite(scores).sum())
    assert st.aci_misses == int(np.nansum(r["miss"]))
    assert 0.0 <= st.aci_coverage <= 1.0


def test_runtime_forecast_equals_single_implementation(model):
    s = synthetic_series(80, seed=5)
    shift = _shift(5)
    raw = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 80)
    cal = feed(StreamingMayak(model, LAT, LON, ELEV, conformal=shift), s, 0, 80)
    cal.reset_calibration(0.25)
    cal_doy = future_calendar_after(s, 80, H)
    q_raw, _ = raw.forecast(*cal_doy)
    q, mu = cal.forecast(*cal_doy)
    ref_q, ref_mu = calibrate_forecast(q_raw, shift, cal.theta)
    assert np.array_equal(q, ref_q) and np.array_equal(mu, ref_mu)


def test_runtime_without_aci_never_moves_theta(model):
    s = synthetic_series(60, seed=6)
    st = StreamingMayak(model, LAT, LON, ELEV)
    for k in range(60):
        feed(st, s, k, k + 1)
        st.forecast(*future_calendar_after(s, k + 1, H))
    assert st.theta == 0.0 and st.aci_updates == 0 and st._pending is None


def test_runtime_long_gap_freezes_theta(model):
    """Длинная серия пропусков: обратной связи нет, θ не меняется и не расходится."""
    p = ACIParams(gamma=0.05)
    s = synthetic_series(700, seed=7, p_valid=1.0)
    st = feed(StreamingMayak(model, LAT, LON, ELEV, aci=p), s, 0, 48)
    st.reset_calibration(0.3)
    st.forecast(*future_calendar_after(s, 48, H))
    for k in range(48, 600):
        st.step(None, float(s["x"][k, 1]), float(s["x"][k, 2]), s["doy"][k], s["hour"][k])
    assert st.theta == pytest.approx(0.3, abs=1e-7) and st.aci_updates == 0
    feed(st, s, 600, 650)
    assert st.aci_updates == 0, "лиды старого прогноза давно прошли"
    q, _ = st.forecast(*future_calendar_after(s, 650, H))
    assert np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()
    feed(st, s, 650, 670)
    assert st.aci_updates == int((s["m"][650:670, 0] > 0).sum()) > 0


def test_runtime_each_lead_is_checked_once_and_in_order(model):
    p = ACIParams(gamma=0.05)
    s = synthetic_series(300, seed=8, p_valid=1.0)
    st = feed(StreamingMayak(model, LAT, LON, ELEV, aci=p), s, 0, 24)
    st.forecast(*future_calendar_after(s, 24, H))
    valid = s["m"][:, 0] > 0
    feed(st, s, 24, 24 + 30)
    n30 = int(valid[24:54].sum())
    assert st.aci_updates == n30
    st.step(float(s["x"][30, 0]), None, None, s["doy"][30], s["hour"][30])
    assert st.aci_updates == n30
    feed(st, s, 54, 300)
    assert st.aci_updates == int(valid[24:24 + H].sum()), "каждый лид - не больше одного раза"


def test_runtime_constant_misses_hit_the_bound(model):
    """Прибор врёт на много градусов: θ упирается в границу, прогноз остаётся конечным."""
    p = ACIParams(gamma=0.1)
    s = synthetic_series(100, seed=9, p_valid=1.0)
    st = feed(StreamingMayak(model, LAT, LON, ELEV, aci=p), s, 0, 48)
    for k in range(48, 100):
        st.forecast(*future_calendar_after(s, k, H))
        q0 = st._pending["q"][0]
        med, d = float(q0[I_MED]), float(q0[I_HI90] - q0[I_MED])
        T = min(med + 6.0 * d + 0.5, 59.0)
        st.step(T, float(s["x"][k, 1]), float(s["x"][k, 2]), s["doy"][k], s["hour"][k])
    assert st.theta == p.theta_max
    q, _ = st.forecast(*future_calendar_after(s, 100, H))
    assert np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()


def test_state_v3_is_pinned_and_carries_theta(model):
    assert STATE_VERSION == 3 and STATE_HEADER.itemsize == 16
    s = synthetic_series(100, seed=10)
    st = feed(StreamingMayak(model, LAT, LON, ELEV, aci=True), s, 0, 100)
    st.reset_calibration(0.3141)
    raw = st.serialize()
    assert len(raw) == st.state_nbytes == DEFAULT_STATE_BYTES < 4096
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    assert int(hdr["version"]) == 3 and float(hdr["aci_theta"]) == st.theta
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(raw)
    assert back.theta == st.theta and back.serialize() == raw
    cal = future_calendar_after(s, 100, H)
    np.testing.assert_allclose(back.forecast(*cal)[0], st.forecast(*cal)[0], atol=1e-4)


def test_theta_survives_restart_and_reset_keeps_it(model):
    p = ACIParams(gamma=0.05)
    s = synthetic_series(220, seed=11, p_valid=1.0)
    live = StreamingMayak(model, LAT, LON, ELEV, aci=p)
    for k in range(120):
        feed(live, s, k, k + 1)
        live.forecast(*future_calendar_after(s, k + 1, H))
    back = StreamingMayak(model, LAT, LON, ELEV, aci=p)
    back.load_state(live.serialize())
    assert back.theta == live.theta != 0.0
    for st in (live, back):
        for k in range(120, 220):
            feed(st, s, k, k + 1)
            st.forecast(*future_calendar_after(s, k + 1, H))
    assert back.theta == pytest.approx(live.theta, abs=1e-6)
    theta = live.theta
    live.reset()
    assert live.theta == theta and live._pending is None
    live.reset_calibration()
    assert live.theta == 0.0 and live.aci_updates == 0


def test_state_v2_is_read_with_zero_theta(model):
    s = synthetic_series(90, seed=12)
    st = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 90)
    raw = st.serialize()
    hdr3 = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    hdr2 = np.zeros((), STATE_HEADERS[2])
    for name in STATE_HEADERS[2].names:
        hdr2[name] = hdr3[name]
    hdr2["version"] = 2
    v2 = hdr2.tobytes() + raw[STATE_HEADER.itemsize:]
    assert len(v2) == DEFAULT_STATE_BYTES - 4
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.reset_calibration(0.7)
    back.load_state(v2)
    assert back.theta == 0.0
    cal = future_calendar_after(s, 90, H)
    np.testing.assert_allclose(back.forecast(*cal)[0], st.forecast(*cal)[0], atol=1e-4)
    with pytest.raises(ValueError, match="Б"):
        back.load_state(v2 + b"\0")


def test_corrupted_theta_is_rejected(model):
    st = StreamingMayak(model, LAT, LON, ELEV)
    raw = st.serialize()
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0].copy()
    hdr["aci_theta"] = np.nan
    with pytest.raises(ValueError, match="θ"):
        st.load_state(hdr.tobytes() + raw[STATE_HEADER.itemsize:])


def test_sharpness_curve_properties():
    preds, aux = _dataset(n_st=10, per=10, sigma=lambda z: 1.5)
    ev = _ev(preds["МАЯК"], aux)
    curves = ev.sharpness_curve()
    assert list(curves) == ["весь горизонт"] + [f"{a}-{b}" for a, b in LEAD_BINS]
    c = curves["весь горизонт"]
    k = int(np.flatnonzero(np.isclose(c["scale"], 1.0))[0])
    row90 = ev.sharpness_coverage()[0]
    assert c["coverage"][k] == pytest.approx(row90["coverage"])
    assert c["width"][k] == pytest.approx(row90["width"], rel=1e-5)
    assert (np.diff(c["coverage"]) >= 0).all(), "интервалы вложены: покрытие не убывает"
    np.testing.assert_allclose(c["width"], c["scale"] * c["width"][k], rtol=1e-5)


def test_width_at_coverage_compares_models_at_equal_coverage():
    ok, _ = _dataset(n_st=10, per=10, seed=1)
    narrow, _ = _dataset(n_st=10, per=10, seed=1, sigma=lambda z: 1.5)
    _p, aux_ok = _dataset(n_st=10, per=10, seed=1)
    _p, aux_narrow = _dataset(n_st=10, per=10, seed=1, sigma=lambda z: 1.5)
    c_ok = _ev(ok["МАЯК"], aux_ok).sharpness_curve()["весь горизонт"]
    c_narrow = _ev(narrow["МАЯК"], aux_narrow).sharpness_curve()["весь горизонт"]
    w_nominal = 2 * Z[-1]
    assert width_at_coverage(c_ok["coverage"], c_ok["width"], 0.9) == pytest.approx(
        w_nominal, rel=0.03)
    assert width_at_coverage(c_narrow["coverage"], c_narrow["width"], 0.9) == pytest.approx(
        1.5 * w_nominal, rel=0.03)
    assert math.isnan(width_at_coverage([0.1, 0.2], [1.0, 2.0], 0.9))
    assert width_at_coverage([0.8, 1.0], [1.0, 3.0], 0.9) == pytest.approx(2.0)


def test_calibrated_forecast_raises_no_flags():
    from mayak.calibration import conditional_gate, coverage_report
    preds, aux = _dataset(seed=2)
    cfg = _cfg()
    rep = coverage_report(_ev(preds["МАЯК"], aux), aux["meta"], cfg)
    assert rep["overall"]["coverage"] == pytest.approx(0.9, abs=0.01)
    assert set(rep["dims"]) == {"лид", "бин лидов", *COVERAGE_DIMS_INTERNAL}
    for name, rows in rep["dims"].items():
        assert rows, name
        for k, r in rows.items():
            assert not r["off_nominal"] and not r["heterogeneous"], (name, k)
            lo, hi = r["ci"]
            assert lo <= r["coverage"] <= hi
    gate = conditional_gate(rep, cfg)
    assert not any(g["recommended"] for g in gate.values())


def test_zone_specific_miscalibration_is_flagged_and_gated():
    from mayak.calibration import KIND_NARROW, OFF_UNDER, conditional_gate, coverage_report
    preds, aux = _dataset(seed=3, sigma=lambda z: 2.0 if z == "BWh" else 1.0)
    cfg = _cfg()
    rep = coverage_report(_ev(preds["МАЯК"], aux), aux["meta"], cfg)
    zones = rep["dims"]["зона Кёппена"]
    assert zones["BWh"]["off_nominal"] == OFF_UNDER and zones["BWh"]["heterogeneous"]
    assert zones["BWh"]["kind"] == KIND_NARROW
    assert zones["BWh"]["coverage"] == pytest.approx(2 * norm.cdf(Z[-1] / 2) - 1, abs=0.02)
    assert not zones["Cfb"]["off_nominal"] and zones["Cfb"]["heterogeneous"]
    for name in ("роль станции", "длина истории", "валидность истории"):
        assert not any(r["heterogeneous"] for r in rep["dims"][name].values()), name
    gate = conditional_gate(rep, cfg)
    assert gate["зона Кёппена"]["recommended"] and gate["зона Кёппена"]["strata"] == ["BWh", "Cfb"]
    assert not gate["длина истории"]["recommended"]
    assert not gate["роль станции"]["candidate"]
    only_history = conditional_gate(rep, replace(cfg, conditional_dims=("длина истории",)))
    assert not any(g["recommended"] for g in only_history.values()), \
        "зона не в кандидатах - условная поправка по ней не рекомендуется"
    m = rep["matrix"]["зона Кёппена"]["BWh"]
    assert set(m) == {f"{a}-{b}" for a, b in LEAD_BINS}
    assert all(v < 0.8 for v in m.values())


def test_uniform_miscalibration_is_off_nominal_but_not_conditional():
    """Все страты занижены одинаково - это дело маргинальной поправки, а не условной."""
    from mayak.calibration import conditional_gate, coverage_report
    preds, aux = _dataset(seed=4, sigma=lambda z: 1.6)
    cfg = _cfg()
    rep = coverage_report(_ev(preds["МАЯК"], aux), aux["meta"], cfg)
    assert rep["dims"]["зона Кёппена"]["BWh"]["off_nominal"]
    assert rep["dims"]["зона Кёппена"]["Cfb"]["off_nominal"]
    assert not any(g["recommended"] for g in conditional_gate(rep, cfg).values())


def test_one_sided_misses_are_diagnosed_as_shift():
    from mayak.calibration import KIND_ABOVE, KIND_BELOW, coverage_report
    for b, kind in ((2.5, KIND_ABOVE), (-2.5, KIND_BELOW)):
        preds, aux = _dataset(seed=5, bias=lambda z, b=b: b if z == "BWh" else 0.0)
        rep = coverage_report(_ev(preds["МАЯК"], aux), aux["meta"], _cfg(bootstrap=0))
        r = rep["dims"]["зона Кёппена"]["BWh"]
        assert r["kind"] == kind and r["heterogeneous"]
        assert all(math.isnan(x) for x in r["ci"]), "без бутстрапа интервала нет"


def test_small_strata_are_dropped_and_external_dims_used():
    from mayak.calibration import coverage_report, coverage_strata
    preds, aux = _dataset(n_st=10, per=10, seed=6)
    ev = _ev(preds["МАЯК"], aux)
    rep = coverage_report(ev, aux["meta"], _cfg(bootstrap=0, min_windows=60))
    assert rep["dims"]["зона Кёппена"] == {} or all(
        r["n_windows"] >= 60 for r in rep["dims"]["зона Кёппена"].values())
    rep = coverage_report(ev, aux["meta"], _cfg(bootstrap=0, min_stations=6))
    assert rep["dims"]["зона Кёппена"] == {}, "по 5 станций в зоне - меньше порога"
    assert list(coverage_strata(aux["meta"], external=True)) == list(COVERAGE_DIMS_EXTERNAL)
    ext = coverage_report(ev, aux["meta"], _cfg(bootstrap=0), external=True)
    assert "роль станции" not in ext["dims"] and "частота отчётности" in ext["dims"]


def test_served_pairs_follow_time_and_next_issue():
    from mayak.calibration import served_pairs
    station = np.array(["a", "b", "a", "a", "a"], object)
    t = np.array([144, 0, 0, 72, 72])
    win, lead = served_pairs(station, t, H)
    assert len(win) == 72 + 72 + H + H
    a_first = win[:72]
    assert set(a_first.tolist()) == {2} and list(lead[:72]) == list(range(72))
    assert set(win[72:144].tolist()) <= {3, 4} and len(set(win[72:144].tolist())) == 1
    assert list(win[144:144 + H]) == [0] * H and list(lead[144:144 + H]) == list(range(H))
    assert list(win[-H:]) == [1] * H


def test_aci_replay_brings_each_station_to_nominal():
    from mayak.calibration import aci_replay
    rng = np.random.default_rng(7)
    n_st, per = 12, 60
    preds, aux = _dataset(n_st=n_st, per=per, seed=7)
    sig = np.repeat(rng.choice([0.6, 1.0, 1.6], n_st), per)[:, None]
    aux["y"] = preds["МАЯК"]["mu"] + sig * rng.standard_normal(aux["y"].shape)
    ev = _ev(preds["МАЯК"], aux)
    r = aci_replay(ev, aux["meta"], ACIParams(gamma=0.02))
    assert r["overall"]["n"] == n_st * ((per - 1) * 72 + H)
    assert abs(r["overall"]["aci"] - 0.9) < 0.02
    assert r["stations"]["mad_aci"] < 0.5 * r["stations"]["mad_base"]
    assert r["clipped"] == 0 and set(r["by_lead_bin"]) == {f"{a}-{b}" for a, b in LEAD_BINS}
    meta = {k: v for k, v in aux["meta"].items() if k != "t"}
    with pytest.raises(KeyError, match="t"):
        aci_replay(ev, meta, ACIParams())


def test_predictions_roundtrip_and_analysis_on_saved_file(tmp_path):
    from mayak.calibration import analyze, load_predictions, save_predictions
    preds, aux = _dataset(n_st=8, per=8, seed=8)
    shift = _shift(8)
    path = save_predictions(tmp_path / "p" / "internal.npz", preds, aux, shift=shift,
                            info=dict(ckpt="x.ckpt"))
    p2, a2, s2, info = load_predictions(path)
    assert list(p2) == ["МАЯК", "Климатология"] and info == dict(ckpt="x.ckpt")
    assert np.array_equal(s2, shift)
    for n in preds:
        assert np.array_equal(p2[n]["q"], preds[n]["q"].astype(np.float32))
    for k, v in aux["meta"].items():
        assert list(a2["meta"][k]) == list(v), k
    cfg = _cfg(bootstrap=0, min_windows=1, min_stations=1)
    mem = analyze(preds, aux, shift, cfg)
    disk = analyze(p2, a2, s2, cfg, out_dir=str(tmp_path / "out"))
    assert disk["report"]["overall"]["coverage"] == pytest.approx(
        mem["report"]["overall"]["coverage"], abs=1e-6)
    names = {Path(p).name for p in disk["paths"]}
    assert names == {"sharpness_coverage_internal.png", "coverage_strata_internal.png",
                     "calibration_internal.json"}
    data = json.loads((tmp_path / "out" / "calibration_internal.json").read_text("utf-8"))
    assert set(data) >= {"report", "gate", "sharpness", "aci", "config"}
    with pytest.raises(KeyError):
        analyze(preds, aux, None, cfg, model="нет такой")


def test_eval_set_meta_carries_window_time_and_feeds_analysis(tmp_path):
    from mayak.calibration import analyze, load_predictions, save_predictions
    from mayak.evaluate import EvalSet
    root = tmp_path / "data13"
    (root / "stations").mkdir(parents=True)
    rows = []
    for i, (sid, role, lat, zone) in enumerate([("t0", ROLE_TRAIN, 52.0, "Cfb"),
                                                ("t1", ROLE_TRAIN, 45.0, "Cfb"),
                                                ("x0", ROLE_TEST, 10.0, "Af")]):
        rng = np.random.default_rng(i)
        h = np.arange(N_HOURS)
        T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(N_HOURS)
        P = 1000 + 0.2 * rng.standard_normal(N_HOURS)
        RH = 60 + rng.standard_normal(N_HOURS)
        np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32),
                 P=P.astype(np.float32), RH=RH.astype(np.float32),
                 valid=np.ones((N_HOURS, 3), np.uint8), t0_utc_h=np.int64(0))
        rows.append(dict(id=sid, lat=lat, lon=5.0 * i, elev=100.0, koppen=zone, split=role))
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    store = S.get_store(str(manifest))
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=str(manifest),
                 time_key="test", every_hours=24, windows_per_station=10)
    meta = ds.window_meta()
    assert list(meta["t"]) == [t for _sid, t in ds.items]
    n = len(ds)
    rng = np.random.default_rng(0)
    mu = rng.normal(10, 3, (n, H))
    aux = dict(y=mu + rng.standard_normal((n, H)), y_mask=np.ones((n, H)), mu_clim=mu, meta=meta)
    path = save_predictions(tmp_path / "internal.npz", {"МАЯК": dict(mu=mu, q=_gauss_q(mu))}, aux)
    p2, a2, _s, _i = load_predictions(path)
    res = analyze(p2, a2, None, _cfg(bootstrap=0, min_windows=1, min_stations=1))
    assert res["aci"]["overall"]["n"] > 0
    assert res["report"]["dims"]["роль станции"]


def test_cli_runs_on_saved_predictions(tmp_path, capsys):
    from mayak.calibration import main, save_predictions
    preds, aux = _dataset(n_st=8, per=6, seed=9)
    save_predictions(tmp_path / "internal.npz", preds, aux)
    save_predictions(tmp_path / "external.npz", preds, aux)
    main(["--preds", str(tmp_path / "internal.npz"), "--external-preds",
          str(tmp_path / "external.npz"), "--bootstrap", "0", "--out-dir", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "условная конформная поправка" in out and "ВНЕШНИЙ ТЕСТ" in out
    assert (tmp_path / "o" / "calibration_external.json").exists()


@pytest.mark.parametrize("mod, flag", [("mayak.calibration", "--preds"),
                                       ("mayak.evaluate", "--save-preds"),
                                       ("mayak.runtime.run_inference", "--aci")])
def test_entry_points_answer_help(mod, flag, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [mod, "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    out = capsys.readouterr().out
    assert e.value.code == 0 and "usage" in out and flag in out


def test_yaml_matches_dataclass_defaults():
    from mayak.calibration import load_config
    d = yaml.safe_load((REPO / "conf" / "calibration" / "default.yaml").read_text("utf-8"))
    assert CalibrationConfig.from_dict(d) == CalibrationConfig() == load_config()
    aci = CalibrationConfig().aci()
    assert aci == ACIParams(target=0.1, gamma=0.005, max_factor=4.0)
    assert CalibrationConfig(nominal=0.8).aci().interval == (1, 5)


@pytest.mark.parametrize("kw, match", [
    (dict(nominal=0.5), "nominal"), (dict(tolerance=0.0), "tolerance"),
    (dict(conditional_dims=("сезон",)), "conditional_dims"),
    (dict(sharpness_range=(2.0, 4.0)), "sharpness_range"), (dict(aci_gamma=0.0), "aci"),
    (dict(aci_max_factor=1.0), "aci"), (dict(bootstrap=-1), "bootstrap"),
    (dict(ci_level=1.0), "ci_level")])
def test_bad_calibration_config_is_rejected(kw, match):
    with pytest.raises(ConfigError, match=match):
        CalibrationConfig(**kw)
    with pytest.raises(ConfigError, match="неизвестные"):
        CalibrationConfig.from_dict(dict(gamma=0.1))
