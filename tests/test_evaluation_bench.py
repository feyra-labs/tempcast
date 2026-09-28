"""Тесты стенда оценки: сырые выходы, сетка длин истории, одна история для всех."""
import csv
import math

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from mayak.baselines.statistical import ZQ
from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.holdout import (HISTORY_GRID, check_history_grid, history_label,
                                history_strata)
from mayak.data.qc import DEFAULT_QC
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from mayak.metrics import LEAD_BINS, NQ

N_HOURS = 26_400
STATIONS = [("t0", ROLE_TRAIN, 52.0, "Cfb"), ("t1", ROLE_TRAIN, 48.0, "Dfb"),
            ("v0", ROLE_VAL, 45.0, "Cfa"), ("x0", ROLE_TEST, 50.0, "Cfb"),
            ("x1", ROLE_TEST, 40.0, "Csa"), ("x2", ROLE_TEST, -30.0, "Cfa")]
WINDOWS = 6
BOOT = dict(n_boot=40, seed=3, level=0.90)
R_DAMPED = np.full(H, 0.7, np.float32)


def _write_station(root, sid, seed, n=N_HOURS):
    """Суточный ход, синоптическая аномалия с памятью и редкие пропуски."""
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    rho = np.exp(-1 / 48)
    syn = lfilter([1.0], [1.0, -rho], rng.standard_normal(n) * 2.0 * np.sqrt(1 - rho ** 2))
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + syn + 0.2 * rng.standard_normal(n)
    P = 1000 - 2 * syn + 0.2 * rng.standard_normal(n)
    RH = np.clip(60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n), 5, 99)
    valid = (rng.random((n, 3)) > 0.03).astype(np.uint8)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=valid, t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
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


@pytest.fixture(scope="module")
def base(store, manifest):
    from mayak.evaluate import EvalSet
    ds = EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                 time_key="test", every_hours=72, max_windows=None,
                 windows_per_station=WINDOWS)
    assert len(ds) == 5 * WINDOWS
    return ds


class _Anchored(torch.nn.Module):
    """Климатология плюс затухающая средняя аномалия последних суток истории."""

    def forward(self, b):
        muc = b["mu_clim_fut"]
        x, m = b["x_hist"][:, -24:, 0], b["mask_hist"][:, -24:, 0]
        clim_last = muc[:, :24]
        n = m.sum(1).clamp(min=1.0)
        a = ((x - clim_last) * m).sum(1) / n
        h = torch.arange(1, H + 1, dtype=muc.dtype)
        mu = muc + torch.exp(-h / 24.0)[None] * a[:, None]
        q = mu[..., None] + b["sigma_clim"][:, None, None] * torch.as_tensor(ZQ)
        return {"mu": mu, "q": q}


def _models():
    from mayak.model import MAYAK
    torch.manual_seed(0)
    return {"МАЯК": MAYAK().eval(), "Якорь": _Anchored()}


def _shift():
    """Таблица поправок, как у подгонки: интервалы шире, поправка медианы нулевая."""
    rng = np.random.default_rng(11)
    shift = np.sort(rng.normal(0.0, 0.8, (len(LEAD_BINS), NQ)), axis=-1).astype(np.float32)
    shift -= shift[:, 3:4]
    return shift


@pytest.fixture(scope="module")
def results(base):
    from mayak.evaluate import evaluate_set
    named = _models()
    raw = evaluate_set(named, base, r_damped=R_DAMPED, shift=None, bootstrap=BOOT)
    cal = evaluate_set(named, base, r_damped=R_DAMPED, shift=_shift(), bootstrap=BOOT)
    return raw, cal


def _same(a, b):
    """Равенство вложенных результатов бит в бит; NaN равен NaN."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        b = np.asarray(b)
        if a.dtype.kind in "fc":
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_grid_changes_only_the_history(base):
    for L in HISTORY_GRID:
        ds = base.with_history(L)
        assert ds.items == base.items and ds.L == L and ds.curriculum is None
        meta = ds.window_meta()
        assert set(meta["history"].tolist()) == {L}
        assert set(meta["history_label"].tolist()) == {history_label(L)}
        for i in (0, len(ds) - 1):
            a, b = base[i], ds[i]
            for k in ("y", "y_mask", "mu_clim_fut", "sigma_clim", "norm_scale"):
                assert torch.equal(a[k], b[k]), (L, k)
            assert int(b["hist_len"]) == L
            assert torch.all(b["mask_hist"][: L_MAX - L] == 0)
    assert base.L is None, "копия с другой длиной не меняет исходный набор"


def test_history_dependent_baselines_equal_climatology_at_zero_history(base):
    from mayak.evaluate import (CLIMATOLOGY, DAMPED, SEASONAL, collect_predictions,
                                history_baselines, history_free_baselines)
    _p, aux = collect_predictions({}, base.with_history(0))
    clim = history_free_baselines(aux)[CLIMATOLOGY]
    dep = history_baselines(aux, R_DAMPED)
    assert set(dep) == {DAMPED, SEASONAL}
    for name, p in dep.items():
        assert np.array_equal(p["mu"], clim["mu"]), name
        assert np.array_equal(p["q"], clim["q"]), name


def _perturbed(clims, items, cut_of):
    """Копия станций, где часы раньше границы окна заменены мусором."""
    out = {}
    for sid, s in clims.items():
        out[sid] = dict(s)
        for key in ("raw", "x"):
            out[sid][key] = np.array(s[key], copy=True)
    for sid, t in items:
        cut = cut_of(t)
        for key in ("raw", "x"):
            out[sid][key][:cut, 0] += 15.0
            out[sid][key][:cut, 1] -= 40.0
    return out


@pytest.mark.parametrize("L", [24, 168, L_MAX])
def test_baselines_see_only_the_window_history(store, manifest, L):
    """Эталоны видят ту же историю, что модели: часы до начала истории окна (и до
    начала контекста QC при полной истории) на них не влияют, а часы внутри истории
    влияют."""
    from mayak.evaluate import (EvalSet, add_statistical_baselines, collect_predictions)
    clims = store.clims()
    ds = EvalSet(clims, station_splits=(ROLE_TEST,), manifest=manifest, time_key="test",
                 every_hours=72, max_windows=None, windows_per_station=1, L=L)
    ctx = DEFAULT_QC.lookback_hours if L == L_MAX else 0
    far = EvalSet(_perturbed(clims, ds.items, lambda t: t - L - ctx),
                  station_splits=(ROLE_TEST,), manifest=manifest, time_key="test",
                  every_hours=72, max_windows=None, windows_per_station=1, L=L)
    near = EvalSet(_perturbed(clims, ds.items, lambda t: t),
                   station_splits=(ROLE_TEST,), manifest=manifest, time_key="test",
                   every_hours=72, max_windows=None, windows_per_station=1, L=L)
    assert far.items == ds.items == near.items
    runs = []
    for d in (ds, far, near):
        preds, aux = collect_predictions({"Якорь": _Anchored()}, d)
        runs.append((add_statistical_baselines(preds, aux, r_damped=R_DAMPED), aux))
    (p0, a0), (p1, a1), (p2, _a2) = runs
    assert np.array_equal(a0["x_hist"], a1["x_hist"]), "вход моделей не зависит от часов вне окна"
    assert np.array_equal(a0["a_recent"], a1["a_recent"])
    for name in p0:
        assert np.array_equal(p0[name]["mu"], p1[name]["mu"]), name
        assert np.array_equal(p0[name]["q"], p1[name]["q"]), name
    for name in ("Damped persistence", "Seasonal-naive 24ч", "Якорь"):
        assert not np.array_equal(p0[name]["mu"], p2[name]["mu"]), \
            f"{name}: порча внутри истории должна менять прогноз"


def test_history_free_model_is_computed_once_and_equal_across_grid(base, results):
    from mayak.evaluate import CLIMATOLOGY, collect_predictions, history_free_baselines
    raw, _cal = results
    rows = raw["bench"].history[CLIMATOLOGY]
    first = rows[HISTORY_GRID[0]]
    for L in HISTORY_GRID:
        assert rows[L] is first, "строки климатологии не пересчитываются по сетке"
        _p, aux = collect_predictions({}, base.with_history(L))
        again = history_free_baselines(aux)[CLIMATOLOGY]
        ref = raw["bench"].preds[CLIMATOLOGY]
        assert np.array_equal(again["mu"], ref["mu"]) and np.array_equal(again["q"], ref["q"])


def test_history_breakdown_has_every_grid_point(results):
    from mayak.evaluate import BREAKDOWN_LEAD, HISTORY_LEADS
    raw, _cal = results
    labels = [history_label(L) for L in HISTORY_GRID]
    assert set(raw["history"]) >= {"МАЯК", "Якорь", "Климатология", "Damped persistence",
                                   "Seasonal-naive 24ч"}
    for name, by_lead in raw["history"].items():
        assert list(by_lead) == list(HISTORY_LEADS)
        for h, rows in by_lead.items():
            assert list(rows) == labels, (name, h)
            counts = {(s["n_windows"], s["n_stations"]) for s in rows.values()}
            assert len(counts) == 1, "на всей сетке одни и те же окна и пары"
            assert all("ci" in s for s in rows.values())
    assert list(raw["breakdowns"]["длина истории"]) == labels
    assert BREAKDOWN_LEAD in HISTORY_LEADS
    cov = raw["coverage"]["report"]
    assert list(cov["dims"]["длина истории"]) == labels, "покрытие получает реальные бины"
    assert list(cov["matrix"]["длина истории"]) == labels
    assert 0.0 <= cov["history_overall"] <= 1.0


def test_skill_differs_along_the_grid_for_history_models(results):
    raw, _cal = results
    rows = raw["history"]["Якорь"][24]
    sk0 = rows[history_label(0)]["pooled"]["Skill"]
    sk672 = rows[history_label(L_MAX)]["pooled"]["Skill"]
    assert sk0 == pytest.approx(0.0, abs=1e-9), "без истории якорная модель равна климатологии"
    assert sk672 > sk0


def test_conformal_table_does_not_touch_the_comparison(results):
    raw, cal = results
    for key in ("leads", "overall", "history", "breakdowns", "sharpness", "coverage"):
        assert _same(raw[key], cal[key]), key
    for name, p in raw["bench"].preds.items():
        q = cal["bench"].preds[name]
        assert np.array_equal(p["mu"], q["mu"]) and np.array_equal(p["q"], q["q"]), name
    assert raw["calibrated"] is None
    c = cal["calibrated"]
    assert set(c) == {"effect", "report", "gate", "aci"}
    before, after = c["effect"]["весь горизонт"]["PICP90"]
    assert before == raw["coverage"]["report"]["overall"]["coverage"]
    assert after != before, "таблица меняет интервалы в своём разделе"
    assert list(c["report"]["dims"]["длина истории"]) == [history_label(L) for L in HISTORY_GRID]


def test_printing_shows_calibrated_section_only_with_table(results, capsys):
    from mayak.evaluate import print_evaluation
    raw, cal = results
    print_evaluation(raw)
    out_raw = capsys.readouterr().out
    print_evaluation(cal)
    out_cal = capsys.readouterr().out
    assert "после калибровки" not in out_raw and "после калибровки" in out_cal
    assert "Длина истории" in out_raw and history_label(0) in out_raw
    assert out_cal.startswith(out_raw), "сравнительная часть напечатана одинаково"


def test_saved_predictions_feed_calibration_with_history_grid(results, tmp_path):
    from mayak.calibration import analyze, load_config, load_predictions
    from mayak.evaluate import plot_history_curves, save_bench, save_history_table
    from dataclasses import replace
    _raw, cal = results
    shift = _shift()
    nominal, grid_path = save_bench(cal, str(tmp_path), "internal", shift=shift,
                                    info=dict(ckpt="x"))
    preds, aux, s, info = load_predictions(nominal)
    h_preds, h_aux, _s, h_info = load_predictions(grid_path)
    assert list(h_preds) == ["МАЯК"] and set(aux["meta"]["history"].tolist()) == {L_MAX}
    assert info["history_grid"] == list(HISTORY_GRID) == h_info["history_grid"]
    assert len(h_aux["y"]) == len(HISTORY_GRID) * len(aux["y"])
    assert np.array_equal(s, shift)
    cfg = replace(load_config(), bootstrap=0)
    with_table = analyze(preds, aux, s, cfg, history=(h_preds, h_aux))
    without = analyze(preds, aux, None, cfg, history=(h_preds, h_aux))
    labels = [history_label(L) for L in HISTORY_GRID]
    for res in (with_table, without):
        assert list(res["raw"]["report"]["dims"]["длина истории"]) == labels
    assert list(with_table["calibrated"]["report"]["dims"]["длина истории"]) == labels
    assert without["calibrated"] is None
    assert _same(with_table["sharpness"], without["sharpness"]), "кривые остроты - сырые"
    assert _same(with_table["raw"], without["raw"])
    ref = cal["coverage"]["report"]["dims"]["длина истории"]
    got = with_table["raw"]["report"]["dims"]["длина истории"]
    for k in labels:
        assert got[k]["coverage"] == pytest.approx(ref[k]["coverage"], abs=1e-6), k
    assert plot_history_curves(cal["history"], str(tmp_path), "internal").endswith(".png")
    assert save_history_table(cal, str(tmp_path), "internal").endswith(".json")


def test_history_strata_order_by_length():
    meta = dict(history=np.array([672, 0, 24, 6, 24]),
                history_label=np.array([history_label(v) for v in (672, 0, 24, 6, 24)], object))
    labels, order = history_strata(meta)
    assert order == ["L=0ч", "L=6ч", "L=24ч", "L=672ч"]
    _l, bins = history_strata(dict(history=np.array([300, 0, 20, 100])))
    assert bins == ["L=0", "L 1-24ч", "L 25-168ч", f"L 169-{L_MAX}ч"]


@pytest.mark.parametrize("grid, match", [((), "пуста"), ((0, 24), "полной истории"),
                                         ((-1, L_MAX), "вне"), ((0, L_MAX + 1), "вне")])
def test_bad_history_grid_is_rejected(grid, match):
    with pytest.raises(ValueError, match=match):
        check_history_grid(grid)


def test_grid_parsing():
    from mayak.evaluate import parse_grid
    assert parse_grid("672, 0,24,24") == (0, 24, L_MAX)
    assert check_history_grid(HISTORY_GRID) == HISTORY_GRID
