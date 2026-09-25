"""Тесты: робастность обученной модели без переобучения."""
import csv
import dataclasses
import json
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.signal import lfilter

from mayak.baselines.statistical import ZQ
from mayak.config import (SCENARIO_INPUT, SCENARIO_INSTRUMENT, SCENARIO_RULES, ConfigError,
                          RobustnessConfig, ScenarioSpec)
from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.augment import P, RH, T, clean_history, make_window
from mayak.data.qc import point_qc
from mayak.data.scenarios import (SCENARIOS, apply_scenario, drift_history, drift_target,
                                  point_qc_mask, scenario_rng)
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "conf"
N_HOURS = 26_400
ALL = tuple(SCENARIO_RULES)
INPUT_ONLY = tuple(n for n, r in SCENARIO_RULES.items() if not r.target)
TARGET = tuple(n for n, r in SCENARIO_RULES.items() if r.target)
MAX_LEVEL = {sc.name: sc.levels[-1] for sc in RobustnessConfig().scenarios}
AVAILABILITY = ("dropout", "gap", "history", "drop_channel")


def _window(L=L_MAX, seed=0, lat=45.0, lon=10.0, holes=0.1):
    """Окно с дырами в истории и в цели; инвариант маски соблюдён."""
    x, hour = clean_history(lat=lat, lon=lon, seed=seed)
    w = make_window(x, hour, L=L, lat=lat, lon=lon,
                    y=10.0 + 3.0 * np.sin(np.arange(H) / 7.0))
    rng = np.random.default_rng(seed + 100)
    w.m[rng.random((L_MAX, 3)) < holes] = 0.0
    w.x[w.m == 0] = 0.0
    w.y_mask[rng.random(H) < 0.2] = 0.0
    w.y[w.y_mask == 0] = 0.0
    return w


def _snap(w):
    return dict(x=w.x.tobytes(), m=w.m.tobytes(), y=w.y.tobytes(), y_mask=w.y_mask.tobytes(),
                L=w.L, lat=w.lat, lon=w.lon, elev=w.elev, qc_elev=w.qc_elev)


def _apply(w, name, level, seed=0, index=0):
    return apply_scenario(w, name, level, scenario_rng(seed, name, index))


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("L, lat, lon", [(L_MAX, 45.0, 10.0), (100, -89.9, 179.95),
                                         (0, 60.0, -180.0)])
def test_zero_level_is_bitwise_identity(name, L, lat, lon):
    """Каждая функция сценария на нуле - тождество сама, без короткого замыкания."""
    w = _window(L=L, lat=lat, lon=lon)
    before = _snap(w)
    SCENARIOS[name].fn(w, 0.0, scenario_rng(0, name, 0), dict(SCENARIO_RULES[name].params))
    assert _snap(w) == before, f"{name}: нулевой уровень изменил окно"


def test_scenario_registry_matches_config_contract():
    assert set(SCENARIOS) == set(SCENARIO_RULES)
    for name, r in SCENARIO_RULES.items():
        assert r.kind in (SCENARIO_INPUT, SCENARIO_INSTRUMENT)
        if r.kind == SCENARIO_INPUT:
            assert not r.target, f"{name}: «отказ входа» не искажает цель"
        if r.variant_of:
            assert SCENARIO_RULES[r.variant_of].kind == SCENARIO_INSTRUMENT
            assert SCENARIO_RULES[r.variant_of].target and not r.target
            assert not r.guard, f"{name}: незамеченное смещение модель знать не может"
    assert {"offset", "drift", "scale"} <= set(TARGET)
    assert set(TARGET) == {"offset", "drift", "scale"}


@pytest.mark.parametrize("name", INPUT_ONLY)
def test_input_failure_never_touches_target(name):
    w = _window()
    y, ym = w.y.copy(), w.y_mask.copy()
    _apply(w, name, MAX_LEVEL[name])
    assert np.array_equal(w.y, y) and np.array_equal(w.y_mask, ym)


def test_offset_is_applied_to_history_and_target_exactly():
    b = 2.5
    w0, w = _window(), _window()
    _apply(w, "offset", b)
    mT = w0.m[:, T] > 0
    np.testing.assert_allclose(w.x[mT, T] - w0.x[mT, T], b, atol=1e-5)
    assert np.array_equal(w.x[~mT, T], w0.x[~mT, T])
    assert np.array_equal(w.x[:, 1:], w0.x[:, 1:])
    ok = w0.y_mask > 0
    np.testing.assert_allclose(w.y[ok] - w0.y[ok], b, atol=1e-5)
    assert np.array_equal(w.y[~ok], w0.y[~ok]) and np.array_equal(w.y_mask, w0.y_mask)
    wi = _window()
    _apply(wi, "offset_input", b)
    assert np.array_equal(wi.x, w.x), "вариант «только вход» искажает историю так же"
    assert np.array_equal(wi.y, w0.y), "... но цель не трогает"


def test_drift_is_continuous_from_history_into_target():
    rate, L = 0.24, 300                    # 0.01 °C/ч
    w0, w = _window(L=L), _window(L=L)
    _apply(w, "drift", rate)
    dh, dt = drift_history(w0, rate), drift_target(w0, rate)
    assert dh[w0.h0] == 0.0, "прибор откалиброван в начале фактической истории"
    np.testing.assert_allclose(dh[-1], rate / 24 * (L - 1), rtol=1e-6)
    np.testing.assert_allclose(dt[0], rate / 24 * L, rtol=1e-6)
    np.testing.assert_allclose(np.diff(np.r_[dh[-1], dt]), rate / 24, rtol=1e-4)
    mT = w0.m[:, T] > 0
    np.testing.assert_allclose(w.x[mT, T] - w0.x[mT, T], dh[mT], atol=1e-4)
    ok = w0.y_mask > 0
    np.testing.assert_allclose(w.y[ok] - w0.y[ok], dt[ok], atol=1e-4)
    wi = _window(L=L)
    _apply(wi, "drift_input", rate)
    assert np.array_equal(wi.x, w.x) and np.array_equal(wi.y, w0.y)


def test_scale_multiplies_history_and_target():
    w0, w = _window(), _window()
    _apply(w, "scale", 0.05)
    mT = w0.m[:, T] > 0
    np.testing.assert_allclose(w.x[mT, T], w0.x[mT, T] * 1.05, rtol=1e-6)
    ok = w0.y_mask > 0
    np.testing.assert_allclose(w.y[ok], w0.y[ok] * 1.05, rtol=1e-6)


def test_apply_scenario_rejects_contract_violations(monkeypatch):
    def touches_target(w, level, rng, p):
        w.y[:] += 1.0

    def touches_target_mask(w, level, rng, p):
        w.y_mask[:] = 1.0

    monkeypatch.setitem(SCENARIOS, "gap", replace(SCENARIOS["gap"], fn=touches_target))
    with pytest.raises(RuntimeError, match="изменил цель"):
        _apply(_window(), "gap", 5)
    monkeypatch.setitem(SCENARIOS, "offset", replace(SCENARIOS["offset"], fn=touches_target_mask))
    with pytest.raises(RuntimeError, match="маску цели"):
        _apply(_window(), "offset", 1.0)


def test_levels_share_random_numbers():
    base = _window()
    mk = {p: _apply(_window(), "dropout", p).m[:, 0] > 0 for p in (0.1, 0.5, 0.9)}
    assert (mk[0.5] <= mk[0.1]).all() and (mk[0.9] <= mk[0.5]).all(), "пропуски вложены"
    assert mk[0.1].sum() > mk[0.5].sum() > mk[0.9].sum()

    n1, n2 = _apply(_window(), "noise", 1.0), _apply(_window(), "noise", 2.0)
    for ch in (T, P):
        v = base.m[:, ch] > 0
        np.testing.assert_allclose(n2.x[v, ch] - base.x[v, ch],
                                   2.0 * (n1.x[v, ch] - base.x[v, ch]), atol=1e-3)
    s1, s2 = _apply(_window(), "spikes", 0.01), _apply(_window(), "spikes", 0.05)
    hit1, hit2 = s1.x != base.x, s2.x != base.x
    assert hit1.any() and (hit1 <= hit2).all(), "выбросы вложены"

    other = _apply(_window(), "dropout", 0.5, index=1).m[:, 0] > 0
    assert not np.array_equal(other, mk[0.5]), "разные окна - разные случайные числа"
    assert np.array_equal(_apply(_window(), "dropout", 0.5).m, _apply(_window(), "dropout", 0.5).m)


def test_freeze_repeats_value_at_failure_only_where_reported():
    n = 72
    w0, w = _window(), _window()
    _apply(w, "freeze", n)
    i0 = L_MAX - n
    for ch in range(3):
        rows = np.flatnonzero(w0.m[:, ch] > 0)
        v = w0.x[rows[rows < i0][-1], ch]
        tail = w0.m[i0:, ch] > 0
        assert np.all(w.x[i0:, ch][tail] == v)
        assert np.all(w.x[i0:, ch][~tail] == 0.0)
    assert np.array_equal(w.m, w0.m), "замёрзший датчик отчитывается там же, где отчитывался"
    assert np.array_equal(w.x[:i0], w0.x[:i0])


def test_history_gap_and_channel_scenarios_only_remove_validity():
    w0 = _window()
    g = _apply(_window(), "gap", 24)
    assert (g.m[-24:] == 0).all() and np.array_equal(g.m[:-24], w0.m[:-24])
    h = _apply(_window(), "history", 600)
    assert h.L == 72 and (h.m[:L_MAX - 72] == 0).all()
    assert np.array_equal(h.m[L_MAX - 72:], w0.m[L_MAX - 72:])
    for lv, gone in ((1, (P,)), (2, (RH,)), (3, (P, RH))):
        d = _apply(_window(), "drop_channel", lv)
        for ch in range(3):
            if ch in gone:
                assert (d.m[:, ch] == 0).all()
            else:
                assert np.array_equal(d.m[:, ch], w0.m[:, ch])


def test_metadata_errors_have_requested_magnitude():
    w0 = _window()
    c = _apply(_window(), "coords", 0.5)
    assert np.hypot(c.lat - w0.lat, c.lon - w0.lon) == pytest.approx(0.5, abs=1e-9)
    e = _apply(_window(), "elev", 250.0)
    assert abs(e.elev - w0.elev) == pytest.approx(250.0)
    assert e.qc_elev - w0.qc_elev == pytest.approx(e.elev - w0.elev)
    edge = _apply(_window(lat=89.9, lon=179.9), "coords", 2.0)
    assert -90.0 <= edge.lat <= 90.0 and -180.0 <= edge.lon < 180.0


def test_point_qc_mask_matches_runtime_point_qc():
    rng = np.random.default_rng(0)
    x = np.stack([rng.uniform(-120, 90, 500), rng.uniform(200, 1200, 500),
                  rng.uniform(-20, 130, 500)], -1).astype(np.float32)
    x[::37, 1] = np.nan
    m = (rng.random((500, 3)) > 0.1).astype(np.float32)
    got = point_qc_mask(x, m)
    ref = np.stack([point_qc(*row)[1] for row in x]) * m
    np.testing.assert_array_equal(got, ref)


def test_yaml_matches_dataclass():
    d = yaml.safe_load((CONF / "robustness" / "default.yaml").read_text(encoding="utf-8"))
    assert set(d) == {f.name for f in dataclasses.fields(RobustnessConfig)}
    assert RobustnessConfig.from_dict(d) == RobustnessConfig()
    from mayak.robustness import load_config
    assert load_config() == RobustnessConfig()
    assert {sc.name for sc in RobustnessConfig().scenarios} == set(SCENARIO_RULES), \
        "каждый сценарий из 12.1 входит в прогон по умолчанию"


def test_config_roundtrip_and_select():
    cfg = RobustnessConfig()
    assert RobustnessConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    sub = cfg.select(["offset", "dropout"])
    assert [sc.name for sc in sub.scenarios] == ["dropout", "offset"]
    with pytest.raises(ConfigError, match="нет сценариев"):
        cfg.select(["nope"])
    assert cfg.scenario("offset_input").guard is False
    assert cfg.scenario("offset").guard is True
    assert ScenarioSpec("offset_input", (0, 1), guard=True).guard is True


@pytest.mark.parametrize("kw, match", [
    (dict(name="nope", levels=(0, 1)), "неизвестный сценарий"),
    (dict(name="offset", levels=(0.5, 1.0)), "первый уровень"),
    (dict(name="offset", levels=(0, 2, 1)), "возрастать"),
    (dict(name="dropout", levels=(0, 1.5)), "больше допустимого"),
    (dict(name="gap", levels=(0, 2.5)), "целые"),
    (dict(name="drop_channel", levels=(0, 4)), "больше допустимого"),
    (dict(name="noise", levels=(0, 1), params=dict(sd=1)), "неизвестные параметры"),
    (dict(name="freeze", levels=(0, 6), params=dict(channels=(5,))), "каналы"),
])
def test_bad_scenario_spec_is_rejected(kw, match):
    with pytest.raises(ConfigError, match=match):
        ScenarioSpec(**kw)


@pytest.mark.parametrize("kw, match", [
    (dict(qc="median"), "qc"), (dict(leads=(0, 24)), "leads"),
    (dict(roles=("external_test",)), "roles"), (dict(time_key="train"), "time_key"),
    (dict(guard_models=()), "guard_models"), (dict(scenarios=()), "scenarios"),
    (dict(scenarios=(dict(name="gap", levels=(0, 1)), dict(name="gap", levels=(0, 2)))),
     "повторяются"),
])
def test_bad_robustness_config_is_rejected(kw, match):
    with pytest.raises(ConfigError, match=match):
        RobustnessConfig(**kw)


STATIONS = [("t0", ROLE_TRAIN, 52.0, "Cfb"), ("t1", ROLE_TRAIN, 48.0, "Dfb"),
            ("v0", ROLE_VAL, 45.0, "Cfa"), ("x0", ROLE_TEST, 50.0, "Cfb"),
            ("x1", ROLE_TEST, 40.0, "Csa"), ("x2", ROLE_TEST, -30.0, "Cfa")]


def _write_station(root, sid, seed, n=N_HOURS):
    """Суточный ход + синоптическая AR(1)-аномалия (память есть - персистентность полезна)
    + редкие пропуски по каналам."""
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    rho = np.exp(-1 / 48)
    syn = lfilter([1.0], [1.0, -rho], rng.standard_normal(n) * 2.0 * np.sqrt(1 - rho ** 2))
    T_ = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + syn + 0.2 * rng.standard_normal(n)
    P_ = 1000 - 2 * syn + 0.2 * rng.standard_normal(n)
    RH_ = np.clip(60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n), 5, 99)
    valid = (rng.random((n, 3)) > 0.03).astype(np.uint8)
    np.savez(root / "stations" / f"{sid}.npz", T=T_.astype(np.float32), P=P_.astype(np.float32),
             RH=RH_.astype(np.float32), valid=valid, t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data12")
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


def _cfg(**kw):
    kw = {**dict(windows_per_station=4, leads=(1, 24), bootstrap=0), **kw}
    return RobustnessConfig(**kw)


@pytest.fixture(scope="module")
def base(store, manifest):
    from mayak.robustness import base_eval_set
    ds = base_eval_set(store.clims(), manifest, _cfg())
    assert len(ds) == 12 and {sid for sid, _t in ds.items} == {"x0", "x1", "x2"}
    return ds


def _rset(base, name, level, qc="point", params=None):
    from mayak.robustness import RobustnessSet
    return RobustnessSet(base, name, level, params, qc=qc)


@pytest.mark.parametrize("qc", ["none", "point"])
@pytest.mark.parametrize("name", ALL)
def test_zero_level_dataset_equals_eval_set(base, name, qc):
    """Нулевой параметр на полном пути (сценарий → инвариант → QC → батч): тот же батч,
    что у EvalSet, до бита - включая пересчитанную по истории недавнюю аномалию."""
    ds = _rset(base, name, 0.0, qc=qc)
    for i in range(len(base)):
        a, b = base[i], ds[i]
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), (name, k, i)


def test_input_failure_keeps_target_on_the_full_path(base):
    for name in INPUT_ONLY:
        ds = _rset(base, name, MAX_LEVEL[name])
        for i in range(len(base)):
            a, b = base[i], ds[i]
            for k in ("y", "y_mask", "mu_clim_fut", "norm_scale", "doy_fut", "hour_fut",
                      "sigma_clim"):
                assert torch.equal(a[k], b[k]), (name, k)


def test_invariant_holds_after_every_scenario_and_level(base):
    for sc in RobustnessConfig().scenarios:
        for level in sc.levels:
            ds = _rset(base, sc.name, level, params=sc.params)
            for i in range(len(base)):
                a, b = base[i], ds[i]
                x, m = b["x_hist"], b["mask_hist"]
                assert torch.all(x[m == 0] == 0), (sc.name, level)
                assert torch.all(m <= a["mask_hist"]), f"{sc.name}: сценарий создал валидность"
                assert torch.equal(b["y_mask"], a["y_mask"])
                assert torch.isfinite(x).all() and torch.isfinite(b["y"]).all()
                assert -90 <= float(b["lat"]) <= 90 and -180 <= float(b["lon"]) < 180


def test_full_loss_of_history_is_cold_start(base, store, manifest):
    """Три пути к полной потере истории дают один и тот же вход - вход холодного старта."""
    from mayak.evaluate import EvalSet
    l0 = EvalSet(store.clims(), station_splits=(ROLE_TEST,), manifest=manifest,
                 time_key="test", every_hours=72, max_windows=None, windows_per_station=4, L=0)
    assert l0.items == base.items
    sets = [_rset(base, "dropout", 1.0), _rset(base, "gap", L_MAX), _rset(base, "history", L_MAX)]
    for i in range(len(base)):
        ref = l0[i]
        for ds in sets:
            b = ds[i]
            assert torch.equal(b["x_hist"], ref["x_hist"])
            assert torch.equal(b["mask_hist"], ref["mask_hist"])
            assert float(b["a_recent"]) == 0.0


def test_window_qc_catches_frozen_sensor_point_qc_does_not(base):
    n = 72
    pt, wq = _rset(base, "freeze", n, qc="point"), _rset(base, "freeze", n, qc="window")
    frac_pt, frac_wq = [], []
    for i in range(len(base)):
        m0 = base[i]["mask_hist"][-n:, 0]
        ok = m0 > 0
        frac_pt.append(float((pt[i]["mask_hist"][-n:, 0][ok] > 0).float().mean()))
        frac_wq.append(float((wq[i]["mask_hist"][-n:, 0][ok] > 0).float().mean()))
    assert min(frac_pt) == 1.0, "поточечный QC рантайма залипание не видит"
    assert max(frac_wq) < 0.5, "оконный QC залипание на 72 ч ловит"

class _Anchored(torch.nn.Module):
    """Модели с известным поведением, выход как у МАЯК (mu, q).

    mode = clim  - ровно климатология (якорь без аномалии);
    mode = mask  - якорь + затухающая аномалия по валидным часам;
    mode = blind - то же, но аномалия - среднее последних 24 ч истории без маски:
                   невалидные часы читаются как 0 °C.
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, b):
        muc = b["mu_clim_fut"]
        if self.mode == "clim":
            a = torch.zeros_like(muc[:, 0])
        elif self.mode == "mask":
            a = b["a_recent"]
        else:
            a = b["x_hist"][:, -24:, 0].mean(1) - muc[:, :24].mean(1)
        h = torch.arange(1, H + 1, dtype=muc.dtype)
        mu = muc + torch.exp(-h / 24.0)[None] * a[:, None]
        q = mu[..., None] + b["sigma_clim"][:, None, None] * torch.as_tensor(ZQ)
        return {"mu": mu, "q": q}


def _short(names, n=3):
    """Сценарии конфига по умолчанию, уровни - 0, середина и максимум."""
    out = []
    for sc in RobustnessConfig().scenarios:
        if sc.name in names:
            lv = sc.levels
            out.append(ScenarioSpec(sc.name, sorted({lv[0], lv[len(lv) // 2], lv[-1]})[:n],
                                    sc.params, None))
    return tuple(out)


def _sweep(base, models, names, **kw):
    from mayak.robustness import robustness_sweep
    cfg = _cfg(scenarios=_short(names), **kw)
    return cfg, robustness_sweep(models, base, cfg, statistical=False)


@pytest.fixture(scope="module")
def clim_rows(base):
    return _sweep(base, {"МАЯК": _Anchored("clim")}, ALL)


def test_climatology_passes_guard_in_every_scenario(clim_rows):
    """Модель, равная своему якорю, - нижняя граница архитектуры: её скилл ровно 0 во
    всех сценариях, в том числе там, где искажена цель. Иначе знаменатель скилла
    посчитан не на тех же парах или эталон искажён вместе с целью."""
    from mayak.robustness import check_skill_guard
    cfg, rows = clim_rows
    assert {r["scenario"] for r in rows} == set(ALL)
    for r in rows:
        assert r["Skill"] == pytest.approx(0.0, abs=1e-9), (r["scenario"], r["level"])
    n = check_skill_guard(rows, cfg.skill_tolerance, cfg.guard_models)
    assert n == sum(1 for r in rows if r["guard"])


def test_guard_catches_mask_blind_model_and_passes_mask_aware(base):
    from mayak.robustness import RobustnessError, check_skill_guard, skill_violations
    cfg, rows = _sweep(base, {"МАЯК": _Anchored("mask"), "слепая": _Anchored("blind")},
                       AVAILABILITY)
    check_skill_guard(rows, cfg.skill_tolerance, ("МАЯК",))
    good = [r for r in rows if r["model"] == "МАЯК" and r["level"] == 0 and r["lead"] == 1]
    assert all(r["Skill"] > 0.2 for r in good), "якорная модель с маской полезна на чистых данных"
    bad = skill_violations(rows, cfg.skill_tolerance, ("слепая",))
    assert bad and all(r["level"] > 0 for r in bad), "нарушение - только под деградацией"
    assert {"dropout", "gap"} <= {r["scenario"] for r in bad}
    with pytest.raises(RobustnessError, match="gap"):
        check_skill_guard(rows, cfg.skill_tolerance, ("слепая",))


def test_guard_is_never_vacuous(clim_rows):
    from mayak.robustness import RobustnessError, check_skill_guard
    _cfg_, rows = clim_rows
    with pytest.raises(RobustnessError, match="пуста"):
        check_skill_guard(rows, 0.05, ("нет такой модели",))
    with pytest.raises(RobustnessError, match="пуста"):
        check_skill_guard([dict(r, guard=False) for r in rows], 0.05, ("МАЯК",))


def test_unnoticed_offset_is_reported_but_not_asserted(base):
    """Незамеченное смещение прибора: скилл падает (модель не может знать о смещении),
    но утверждение его не проверяет. То же смещение как свойство прибора - проверяет."""
    from mayak.robustness import (RobustnessError, check_skill_guard, robustness_sweep,
                                  skill_violations)
    specs = (ScenarioSpec("offset", (0.0, 5.0)), ScenarioSpec("offset_input", (0.0, 5.0)))
    cfg = _cfg(scenarios=specs)
    rows = robustness_sweep({"МАЯК": _Anchored("mask")}, base, cfg, statistical=False)
    inp = [r for r in rows
           if r["scenario"] == "offset_input" and r["level"] == 5 and r["lead"] == 1]
    assert inp[0]["Skill"] < -cfg.skill_tolerance
    assert not skill_violations(rows, cfg.skill_tolerance)
    check_skill_guard(rows, cfg.skill_tolerance)
    cfg_g = _cfg(scenarios=(ScenarioSpec("offset_input", (0.0, 5.0), guard=True),))
    rows_g = robustness_sweep({"МАЯК": _Anchored("mask")}, base, cfg_g, statistical=False)
    with pytest.raises(RobustnessError, match="offset_input"):
        check_skill_guard(rows_g, cfg_g.skill_tolerance)


def test_rows_carry_ci_distortion_and_excess(base):
    from mayak.robustness import robustness_sweep
    cfg = _cfg(scenarios=(ScenarioSpec("offset", (0.0, 1.0, 3.0)),
                          ScenarioSpec("dropout", (0.0, 0.5))), bootstrap=50)
    rows = robustness_sweep({"МАЯК": _Anchored("clim")}, base, cfg, statistical=True,
                            r_damped=np.full(H, 0.5, np.float32))
    assert {r["model"] for r in rows} >= {"МАЯК", "Климатология", "Damped persistence",
                                          "Seasonal-naive 24ч"}
    for r in rows:
        for m in ("MAE", "CRPS", "PICP90", "Skill"):
            assert f"{m}_lo" in r and f"{m}_macro_hi" in r
            if np.isfinite(r[f"{m}_lo"]):
                assert r[f"{m}_lo"] <= r[f"{m}_hi"]
        assert 10 <= r["n_windows"] <= 12 and r["n_stations"] == 3
        if r["scenario"] == "offset":
            assert r["distortion"] == pytest.approx(r["level"], abs=1e-5)
        else:
            assert r["distortion"] == 0.0
        if r["level"] == 0:
            assert r["dMAE"] == 0.0
        assert r["excess"] == pytest.approx(r["dMAE"] - r["distortion"])
    for r in rows:
        if r["model"] == "МАЯК" and r["scenario"] == "offset":
            assert r["excess"] <= 1e-5, "климатология не может расти быстрее искажения"


def test_results_and_plots_are_written(clim_rows, tmp_path):
    from mayak.robustness import plot_all, save_results
    cfg, rows = clim_rows
    pj, pc = save_results(rows, tmp_path, cfg, meta=dict(ckpt="x"))
    blob = json.loads(Path(pj).read_text(encoding="utf-8"),
                      parse_constant=lambda c: pytest.fail(f"{c} в JSON"))
    assert len(blob["rows"]) == len(rows) and blob["violations"] == []
    assert RobustnessConfig.from_dict(blob["config"]) == cfg
    with open(pc, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    paths = plot_all(rows, cfg, str(tmp_path))
    assert all(os.path.getsize(p) > 0 for p in paths)
    names = {Path(p).name for p in paths}
    assert "robustness_internal_offset.png" in names
    assert "robustness_internal_offset_input.png" not in names, "вариант - пунктиром на offset"
    assert any(n.startswith("robustness_internal_summary") for n in names)


def test_mayak_survives_every_scenario_at_max_level(base):
    """Настоящая архитектура на худшем уровне каждого сценария: формы, конечность,
    монотонность квантилей. Обученность здесь не нужна - это проверка пути данных."""
    from mayak.evaluate import gather
    from mayak.model import MAYAK
    torch.manual_seed(0)
    model = MAYAK().eval()
    for sc in RobustnessConfig().scenarios:
        D = gather(model, _rset(base, sc.name, sc.levels[-1], params=sc.params), batch_size=16)
        assert D["q"].shape == (len(base), H, 7), sc.name
        assert np.isfinite(D["mu"]).all() and np.isfinite(D["q"]).all(), sc.name
        assert (np.diff(D["q"], axis=-1) >= -1e-5).all(), sc.name


def test_module_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mayak.robustness", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("mayak.robustness", run_name="__main__", alter_sys=True)
    assert e.value.code == 0 and "usage" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("MAYAK_ROBUSTNESS_CKPT"),
                    reason="нужен обученный чекпойнт: MAYAK_ROBUSTNESS_CKPT, "
                           "MAYAK_ROBUSTNESS_MANIFEST (по умолчанию data/manifest.csv)")
def test_trained_checkpoint_never_falls_below_climatology():
    from mayak.lit import load_model
    from mayak.robustness import (base_eval_set, check_skill_guard, load_config,
                                  robustness_sweep)
    ckpt = os.environ["MAYAK_ROBUSTNESS_CKPT"]
    manifest = os.environ.get("MAYAK_ROBUSTNESS_MANIFEST", "data/manifest.csv")
    cfg = replace(load_config(), bootstrap=0)
    store = S.get_store(manifest)
    base_ds = base_eval_set(store.clims(), manifest, cfg)
    rows = robustness_sweep({"МАЯК": load_model(ckpt)}, base_ds, cfg, statistical=False)
    check_skill_guard(rows, cfg.skill_tolerance, cfg.guard_models)
