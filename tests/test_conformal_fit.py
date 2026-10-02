"""Тесты: подгонка конформной таблицы МАЯК.

Что проверяется:
* поправка медианы равна нулю по построению, и после применения таблицы и адаптивной
  поправки медиана остаётся медианой модели, в том числе на немонотонном входе;
* таблица разбита по бинам длины истории: на окнах, где ошибка зависит от длины истории,
  она даёт номинальное покрытие в каждом бине, а бин с малым числом окон получает
  маргинальную строку, и это записано в запись о подгонке;
* таблицу со сдвигом медианы и таблицу старого формата без бинов длины истории нельзя
  ни сохранить, ни применить, ни экспортировать;
* калибровочный набор - валидационные станции в блоках калибровки всех сезонов, длины
  истории окон взяты из распределения куррикулума тем же генератором, что у валидации;
* чек-лист отвергает таблицу, подогнанную на одной длине истории, без точности, без
  сведений по бинам длины истории или по другим бинам;
* точность таблицы: потоковый рантайм на fp32 не применяет таблицу int8, экспорт
  требует таблицу своей точности, таблица int8 подгоняется на int8-графах экспорта и
  принимается только теми же графами;
* прогон окон через графы совпадает с пакетной моделью;
* отчёт подгонки: покрытие по сезонам и длине истории до и после таблицы.
"""
import csv
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.config import CalibrationConfig, ConfigError, ModelConfig
from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.dataset import history_probability
from mayak.data.holdout import EvalSet, window_history_lengths
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL, time_layout
from mayak.leakage import (LeakageError, check_conformal, conformal_meta_path,
                           conformal_record, load_conformal, precision_mismatch,
                           save_conformal)
from mayak.metrics import (HISTORY_BINS, I_MED, LEAD_BINS, NQ, Evaluation, apply_adaptive,
                           apply_conformal, calibrate_forecast, conformal_table,
                           fit_conformal_shift, history_bin_of, order_around_median)
from mayak.zones import SEASON_RU

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 35_064            # четыре года: валидация и калибровка занимают целый год
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL), ("v1", ROLE_VAL),
            ("v2", ROLE_VAL), ("x0", ROLE_TEST)]
LAT, LON, ELEV = 52.37, 4.9, 0.0


def _load_calibrate():
    spec = importlib.util.spec_from_file_location("calibrate_under_test",
                                                  REPO / "scripts" / "calibrate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAL = _load_calibrate()


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("conformal_fit")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        rng = np.random.default_rng(i)
        h = np.arange(N_HOURS)
        T = (10 + 8 * np.sin(2 * np.pi * h / 8766) + 5 * np.sin(2 * np.pi * (h - 8) / 24)
             + rng.standard_normal(N_HOURS))
        P = 1000 + 3 * np.sin(2 * np.pi * h / 100) + 0.3 * rng.standard_normal(N_HOURS)
        RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(N_HOURS)
        np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32),
                 P=P.astype(np.float32), RH=RH.astype(np.float32),
                 valid=np.ones((N_HOURS, 3), np.uint8), t0_utc_h=np.int64(0))
        rows.append(dict(id=sid, lat=45.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb",
                         split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def store(manifest):
    S._STORES.clear()
    return S.get_store(manifest)


@pytest.fixture(scope="module")
def cal_set(store, manifest):
    return CAL.calibration_set(store.clims(), manifest, "full",
                               CalibrationConfig(fit_windows_per_station=6))


@pytest.fixture(scope="module")
def model():
    from mayak.runtime.golden import golden_model
    return golden_model(ModelConfig(field_hidden=32, loc_freqs=8, encoder_width=16,
                                    passport_dim=8))


def _gauss(n=400, seed=0, sigma_pred=1.0, sigma_true=1.5):
    from scipy.stats import norm
    from mayak.constants import QUANTILES
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 3, (n, H))
    q = (mu[..., None] + sigma_pred * norm.ppf(np.asarray(QUANTILES))).astype(np.float32)
    y = (mu + np.asarray(sigma_true)[..., None] * rng.standard_normal((n, H))).astype(np.float32)
    return y, q, np.ones((n, H), np.float32)


def _hist(n, seed=0):
    """Длины истории окон из всех бинов."""
    return np.random.default_rng(seed).choice([0, 5, 24, 100, 168, 300, 672], n)


# ------------------------------------------------------------------ медиана на месте

def test_fitted_median_column_is_zero_and_table_keeps_median():
    y, q, w = _gauss()
    hist = _hist(len(y))
    shift, rows = fit_conformal_shift(y, q, w, hist)
    assert shift.shape == (len(LEAD_BINS), len(HISTORY_BINS), NQ)
    assert not np.any(shift[..., I_MED])
    assert (shift[..., 0] < 0).all() and (shift[..., -1] > 0).all(), "узкий прогноз расширяется"
    assert [r["bin"] for r in rows] == [b[2] for b in HISTORY_BINS]
    assert not any(r["marginal"] for r in rows) and sum(r["windows"] for r in rows) == len(y)
    out = apply_conformal(q, shift, hist)
    assert np.array_equal(out[..., I_MED], q[..., I_MED])
    for theta in (-0.4, 0.0, 0.6, (0.2, -0.1, 0.0, 0.5)):
        cq, mu = calibrate_forecast(q, shift, theta, hist)
        assert np.array_equal(mu, q[..., I_MED]) and np.array_equal(cq[..., I_MED], mu)


def test_order_around_median_never_moves_median():
    rng = np.random.default_rng(3)
    messy = rng.normal(0, 3, (200, H, NQ)).astype(np.float32)
    out = order_around_median(messy)
    assert np.array_equal(out[..., I_MED], messy[..., I_MED])
    assert (np.diff(out, axis=-1) >= 0).all()
    ordered = np.sort(messy, axis=-1)
    assert np.array_equal(order_around_median(ordered), ordered)
    shift = rng.normal(0, 2.0, (len(LEAD_BINS), len(HISTORY_BINS), NQ)).astype(np.float32)
    shift[..., I_MED] = 0.0
    cq, mu = calibrate_forecast(messy, shift, 0.5, 100)
    assert np.array_equal(mu, messy[..., I_MED]) and (np.diff(cq, axis=-1) >= 0).all()
    q, mu = apply_adaptive(messy, -0.7)
    assert np.array_equal(mu, messy[..., I_MED])


def test_order_around_median_spreads_missing_values_outwards():
    row = np.array([0.0, np.nan, 0.5, 1.0, 2.0, np.nan, 3.0], np.float32)
    out = order_around_median(row)
    assert np.isnan(out[[0, 1, 5, 6]]).all()
    assert out[2:5].tolist() == [0.5, 1.0, 2.0]


def test_history_table_holds_nominal_in_every_history_bin():
    """Ошибка модели зависит от длины истории: таблица по бинам держит номинал в каждом бине.

    Маргинальная таблица по смеси длин в тех же бинах номинал не держит.
    """
    n = 4000
    hist = _hist(n, seed=5)
    sigma = np.array([2.0, 1.6, 1.2, 0.8])[history_bin_of(hist)]
    y, q, w = _gauss(n=n, seed=5, sigma_true=sigma)
    shift, rows = fit_conformal_shift(y, q, w, hist)
    assert not any(r["marginal"] for r in rows)
    flat = np.repeat(shift[:, :1], len(HISTORY_BINS), axis=1)
    flat[:] = fit_conformal_shift(y, q, w, np.zeros(n, np.int64))[0][:, :1]
    yt, qt, wt = _gauss(n=n, seed=6, sigma_true=sigma)
    station = np.array([f"s{i % 40}" for i in range(n)])
    ev = Evaluation(y=yt, mu=qt[..., I_MED], q=qt, mu_clim=qt[..., I_MED], w=wt, station=station)
    hb = history_bin_of(hist)
    for k in range(len(HISTORY_BINS)):
        own = ev.with_conformal(shift, hist).restrict(windows=hb == k).pooled()["PICP90"]
        marg = ev.with_conformal(flat, hist).restrict(windows=hb == k).pooled()["PICP90"]
        assert abs(own - 0.9) < 0.02, (k, own)
        if k in (0, 3):
            assert abs(marg - 0.9) > 0.04, (k, marg)


def test_sparse_history_bin_gets_marginal_row():
    y, q, w = _gauss(n=300, seed=7)
    hist = np.where(np.arange(300) < 5, 0, 672)
    shift, rows = fit_conformal_shift(y, q, w, hist, min_windows=50)
    own, _ = fit_conformal_shift(y[5:], q[5:], w[5:], np.full(295, 672))
    by_bin = {r["bin"]: r for r in rows}
    assert by_bin["L=0"]["windows"] == 5 and by_bin["L=0"]["marginal"]
    assert by_bin["L 1-24ч"]["windows"] == 0 and by_bin["L 1-24ч"]["marginal"]
    assert not by_bin[HISTORY_BINS[-1][2]]["marginal"]
    full = fit_conformal_shift(y, q, w, np.zeros(300, np.int64))[0][:, 0]
    np.testing.assert_array_equal(shift[:, 0], full)
    np.testing.assert_array_equal(shift[:, 1], full)
    np.testing.assert_array_equal(shift[:, 3], own[:, 3])
    with pytest.raises(ValueError, match="длин истории"):
        fit_conformal_shift(y, q, w, hist[:10])


def test_table_with_median_shift_is_refused_everywhere(tmp_path, cal_set):
    bad = np.zeros((len(LEAD_BINS), len(HISTORY_BINS), NQ), np.float32)
    bad[2, 1, I_MED] = 0.05
    q = np.zeros((1, H, NQ), np.float32)
    for f in (lambda: conformal_table(bad, 0), lambda: apply_conformal(q, bad, 0),
              lambda: calibrate_forecast(q, bad, 0.0, 0)):
        with pytest.raises(ValueError, match="медиан"):
            f()
    with pytest.raises(ValueError, match="медиан"):
        save_conformal(str(tmp_path / "bad.npy"), bad, conformal_record(cal_set))
    np.save(tmp_path / "old.npy", bad)
    with open(conformal_meta_path(str(tmp_path / "old.npy")), "w") as fh:
        json.dump(conformal_record(cal_set), fh)
    with pytest.raises(ValueError, match="медиан"):
        load_conformal(str(tmp_path / "old.npy"))


def test_old_format_table_is_refused_with_clear_error(tmp_path, cal_set, model):
    from mayak.runtime.graphs import export_graphs
    from mayak.runtime.streaming import StreamingMayak
    old = np.zeros((len(LEAD_BINS), NQ), np.float32)
    old[:, 0], old[:, -1] = -0.5, 0.5
    path = str(tmp_path / "v1.npy")
    np.save(path, old)
    with open(conformal_meta_path(path), "w") as fh:
        json.dump(conformal_record(cal_set), fh)
    for f in (lambda: load_conformal(path), lambda: apply_conformal(np.zeros((H, NQ)), old, 0),
              lambda: StreamingMayak(model, LAT, LON, ELEV, conformal=old),
              lambda: export_graphs(model, tmp_path / "e", conformal=old)):
        with pytest.raises(ValueError, match="старого формата"):
            f()


# ------------------------------------------------------------------ калибровочный набор

def test_calibration_set_is_val_stations_in_calib_blocks_of_all_seasons(cal_set):
    assert cal_set.station_splits == (ROLE_VAL,) and cal_set.time_key == "calib"
    assert {sid for sid, _t in cal_set.items} == {"v0", "v1", "v2"}
    lay = time_layout(N_HOURS)
    for _sid, t in cal_set.items:
        assert lay.block_index("calib", np.array([t]), np.array([t + H]))[0] >= 0
    seasons = set(cal_set.window_meta()["season"].tolist())
    assert seasons == set(SEASON_RU.values())


def test_calibration_history_follows_curriculum_generator(cal_set):
    cfg = CalibrationConfig(fit_windows_per_station=6)
    assert cal_set.history_spec() == dict(curriculum="full", L=None, seed=cfg.fit_seed)
    assert cal_set.requested == window_history_lengths(cal_set.items, "full",
                                                       cfg.fit_seed).tolist()
    assert len(set(cal_set.requested)) > 1
    for i in range(len(cal_set)):
        assert int(cal_set[i]["hist_len"]) == cal_set.requested[i]


def test_calibration_history_distribution_matches_curriculum(store, manifest):
    ds = CAL.calibration_set(store.clims(), manifest, "full", CalibrationConfig(fit_every_hours=1))
    lengths = np.asarray(ds.requested)
    n = len(lengths)
    assert n > 2000
    for lo, hi in ((0, 0), (1, 47), (49, 239), (241, L_MAX)):
        p = history_probability("full", lo, hi)
        share = float(((lengths >= lo) & (lengths <= hi)).mean())
        assert abs(share - p) <= 4 * np.sqrt(p * (1 - p) / n), (lo, hi, share, p)


def test_stage_a_checkpoint_calibrates_on_cold_start(store, manifest):
    ds = CAL.calibration_set(store.clims(), manifest, "L0")
    assert set(ds.requested) == {0}


def test_checkpoint_setup_reads_curriculum_and_target_rule(tmp_path):
    from mayak.config import DataConfig
    from mayak.leakage import SELECTION_KEY
    data = DataConfig(target_mask=dict(min_valid_frac=0.7))
    torch.save({SELECTION_KEY: dict(history=dict(curriculum="full", L=None, seed=0)),
                "hyper_parameters": dict(data_config=data.to_dict())}, tmp_path / "a.ckpt")
    assert CAL.checkpoint_setup(str(tmp_path / "a.ckpt")) == ("full", data.target_mask)
    torch.save({SELECTION_KEY: dict(history=dict(curriculum=None, L=672))}, tmp_path / "b.ckpt")
    with pytest.raises(LeakageError, match="куррикулум"):
        CAL.checkpoint_setup(str(tmp_path / "b.ckpt"))


def test_fit_config_fields_are_validated():
    assert CalibrationConfig().fit_windows_per_station is None
    for bad in (dict(fit_every_hours=0), dict(fit_seed=-1), dict(fit_windows_per_station=0)):
        with pytest.raises(ConfigError, match="fit_"):
            CalibrationConfig(**bad)


# ------------------------------------------------------------------ чек-лист

def test_checklist_accepts_curriculum_table_and_rejects_single_length(tmp_path, store,
                                                                       manifest, cal_set):
    shift, fit = fit_conformal_shift(*_gauss(n=50), _hist(50))
    good = str(tmp_path / "good.npy")
    save_conformal(good, shift, conformal_record(cal_set, history_fit=fit))
    check_conformal(good, store)

    full = EvalSet(store.clims(), station_splits=(ROLE_VAL,), manifest=manifest,
                   time_key="calib", every_hours=24, L=L_MAX)
    single = str(tmp_path / "single.npy")
    save_conformal(single, shift, conformal_record(full, history_fit=fit))
    with pytest.raises(LeakageError, match="одной длиной истории"):
        check_conformal(single, store)

    rec = conformal_record(cal_set, history_fit=fit)
    rec.pop("precision")
    untagged = str(tmp_path / "untagged.npy")
    save_conformal(untagged, shift, rec)
    with pytest.raises(LeakageError, match="точность"):
        check_conformal(untagged, store)

    no_fit = str(tmp_path / "no_fit.npy")
    save_conformal(no_fit, shift, conformal_record(cal_set))
    with pytest.raises(LeakageError, match="history_fit"):
        check_conformal(no_fit, store)

    rec = conformal_record(cal_set, history_fit=fit)
    rec["history_bins"] = [[0, 0], [1, 48], [49, 168], [169, L_MAX]]
    other = str(tmp_path / "other_bins.npy")
    save_conformal(other, shift, rec)
    with pytest.raises(LeakageError, match="бинам длины истории"):
        check_conformal(other, store)


# ------------------------------------------------------------------ точность

def test_record_carries_precision_and_int8_needs_graphs(cal_set, tmp_path):
    rec = conformal_record(cal_set)
    assert rec["precision"] == "fp32" and rec["graphs"] is None
    assert rec["history"]["curriculum"] == "full"
    assert rec["history_bins"] == [[b[0], b[1]] for b in HISTORY_BINS]
    assert rec["windows"] == len(cal_set) and rec["windows_digest"] == cal_set.fingerprint()
    with pytest.raises(ValueError, match="int8"):
        conformal_record(cal_set, precision="int8")
    with pytest.raises(ValueError, match="точность"):
        conformal_record(cal_set, precision="fp16")
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"weights")
    assert len(conformal_record(cal_set, checkpoint=str(ck))["checkpoint_digest"]) == 16
    assert precision_mismatch(rec, "fp32") is None
    assert "fp32" in precision_mismatch(rec, "int8")
    assert "не записана" in precision_mismatch({}, "fp32")


def _saved(tmp_path, name, cal_set, precision="fp32", graphs=None, checkpoint=None):
    shift, fit = fit_conformal_shift(*_gauss(n=60, seed=len(name)), _hist(60))
    path = str(tmp_path / f"{name}.npy")
    save_conformal(path, shift, conformal_record(cal_set, checkpoint=checkpoint,
                                                 precision=precision, graphs=graphs,
                                                 history_fit=fit))
    return path, shift


def test_streaming_runtime_does_not_apply_int8_table(model, cal_set, tmp_path, caplog):
    from mayak.runtime.streaming import StreamingMayak
    fp32, shift = _saved(tmp_path, "fp32", cal_set)
    int8, _ = _saved(tmp_path, "int8", cal_set, precision="int8", graphs="0123456789abcdef")
    assert np.array_equal(StreamingMayak(model, LAT, LON, ELEV, conformal=fp32).conformal,
                          shift)
    with caplog.at_level(logging.WARNING):
        st = StreamingMayak(model, LAT, LON, ELEV, conformal=int8)
    assert st.conformal is None
    assert "не применяется" in caplog.text and "int8" in caplog.text


def test_export_requires_table_of_its_precision(model, cal_set, tmp_path):
    from mayak.runtime.graphs import export_graphs
    fp32, _ = _saved(tmp_path, "fp32", cal_set)
    int8, _ = _saved(tmp_path, "int8", cal_set, precision="int8", graphs="0123456789abcdef")
    with pytest.raises(ValueError, match="подогнана на int8"):
        export_graphs(model, tmp_path / "a", conformal=int8)
    with pytest.raises(ValueError, match="подогнана на fp32"):
        export_graphs(model, tmp_path / "b", conformal=fp32, int8=True)
    man = export_graphs(model, tmp_path / "c", conformal=fp32)
    assert man["calibration"]["precision"] == "fp32"
    assert man["calibration"]["conformal"] == "conformal.f32"
    assert export_graphs(model, tmp_path / "d")["calibration"]["precision"] is None


def test_export_checks_the_checkpoint_of_the_table(model, cal_set, tmp_path):
    from mayak.runtime.graphs import export_graphs
    a, b = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    path, _ = _saved(tmp_path, "t", cal_set, checkpoint=str(a))
    export_graphs(model, tmp_path / "ok", conformal=path, checkpoint=str(a))
    with pytest.raises(ValueError, match="другому чекпойнту"):
        export_graphs(model, tmp_path / "bad", conformal=path, checkpoint=str(b))


def test_graph_model_matches_batch_model(model, cal_set):
    from torch.utils.data import DataLoader
    from mayak.runtime.graphs import GraphModel, TorchBackend
    batch = next(iter(DataLoader(cal_set, batch_size=4)))
    with torch.no_grad():
        ref = model(batch)["q"].numpy()
    got = GraphModel(TorchBackend(model), model.cfg)(batch)
    assert got["q"].shape == ref.shape
    assert np.abs(got["q"].numpy() - ref).max() <= 1e-4
    assert torch.equal(got["mu"], got["q"][..., I_MED])


def test_int8_table_is_fitted_on_int8_graphs_of_the_export(model, store, manifest, tmp_path):
    from mayak.runtime.graphs import export_graphs, graphs_digest
    ds = CAL.calibration_set(store.clims(), manifest, "full",
                             CalibrationConfig(fit_windows_per_station=2))
    out = tmp_path / "model"
    export_graphs(model, out, int8=True)
    with pytest.raises(ValueError, match="model-dir"):
        CAL.fit(model, ds, "int8")
    shift, rec, D = CAL.fit(model, ds, "int8", model_dir=str(out))
    assert rec["precision"] == "int8" and rec["graphs"] == graphs_digest(out, "int8")
    assert not np.any(shift[..., I_MED])
    assert D["q"].shape == (len(ds), H, NQ)
    assert len(rec["history_fit"]) == len(HISTORY_BINS) and rec["min_windows"] >= 1
    table = str(tmp_path / "conformal_int8.npy")
    save_conformal(table, shift, rec)
    man = export_graphs(model, out, conformal=table, int8=True)
    assert man["calibration"]["precision"] == "int8"

    import copy
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="других int8-графах"):
        export_graphs(changed, tmp_path / "other", conformal=table, int8=True)


# ------------------------------------------------------------------ отчёт

def test_fit_records_rows_per_history_bin(model, cal_set):
    shift, rec, D = CAL.fit(model, cal_set, min_windows=10)
    meta = cal_set.window_meta()
    assert np.array_equal(D["history"], meta["history"])
    counts = np.bincount(history_bin_of(meta["history"]), minlength=len(HISTORY_BINS))
    assert [r["windows"] for r in rec["history_fit"]] == counts.tolist()
    assert [r["marginal"] for r in rec["history_fit"]] == (counts < 10).tolist()
    assert rec["min_windows"] == 10
    every, _r, _d = CAL.fit(model, cal_set, min_windows=10_000)
    for k in range(len(HISTORY_BINS)):
        np.testing.assert_array_equal(every[:, k], every[:, 0])


def test_fit_report_by_season_and_history(model, cal_set):
    shift, rec, D = CAL.fit(model, cal_set)
    rep = CAL.report(D, cal_set.window_meta(), shift,
                     CalibrationConfig(bootstrap=50, min_windows=1, min_stations=1))
    assert set(rep["dims"]) == {"сезон", "длина истории"}
    assert set(rep["dims"]["сезон"]["raw"]) == set(SEASON_RU.values())
    assert set(rep["dims"]["длина истории"]["calibrated"]) == \
        set(rep["dims"]["длина истории"]["raw"])
    for tag in ("raw", "calibrated"):
        o = rep["overall"][tag]
        lo, hi = o["ci"]
        assert lo <= o["coverage"] <= hi
    ev = Evaluation(y=D["y"], mu=D["mu"], q=D["q"], mu_clim=D["mu_clim"], w=D["y_mask"],
                    station=cal_set.window_meta()["station"])
    assert rep["overall"]["raw"]["coverage"] == pytest.approx(ev.pooled()["PICP90"])
    after = ev.with_conformal(shift, cal_set.window_meta()["history"]).pooled()["PICP90"]
    assert rep["overall"]["calibrated"]["coverage"] == pytest.approx(after)
    assert abs(after - 0.9) < abs(ev.pooled()["PICP90"] - 0.9) + 0.02


def test_calibrate_help_names_precision_and_model_dir(capsys):
    with pytest.raises(SystemExit) as e:
        CAL.main(["--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "--precision" in out and "--model-dir" in out
    with pytest.raises(SystemExit):
        CAL.main(["--ckpt", "x.ckpt", "--precision", "int8"])
    assert "--model-dir" in capsys.readouterr().err
