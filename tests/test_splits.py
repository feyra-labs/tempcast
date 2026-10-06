"""Тесты: временные окна, роли станций, чек-лист антиутечек, чистота обучения."""
import copy
import csv
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.splits import (MIN_GAP_HOURS, ROLE_EXTERNAL, ROLE_TEST, ROLE_TRAIN, ROLE_VAL,
                               ROLES, TIME_KEYS, TimeLayout, assign_roles, strata_report,
                               time_layout)
from mayak.leakage import (SELECTION_KEY, LeakageError, check_checkpoint, check_climatology,
                           check_conformal, check_time_layout, check_windows, conformal_record,
                           run_checklist, save_conformal, selection_record)

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 12_000
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("t2", ROLE_TRAIN),
            ("v0", ROLE_VAL), ("v1", ROLE_VAL), ("x0", ROLE_TEST)]


def _write_station(root, sid, seed, n=N_HOURS):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


def _write_manifest(root, rows):
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data3")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb", split=role))
    return _write_manifest(root, rows)


@pytest.fixture(scope="module")
def store(manifest):
    S._STORES.clear()
    return S.get_store(manifest)


@pytest.fixture(scope="module")
def dm(manifest):
    from mayak.data.datamodule import MayakData
    d = MayakData(manifest=manifest, batch_size=4, windows_per_epoch=64, num_workers=0)
    d.setup("fit")
    return d


def _eval_set(store, manifest, roles, time_key, **kw):
    from mayak.evaluate import EvalSet
    return EvalSet(store.clims(), station_splits=roles, manifest=manifest, time_key=time_key, **kw)


def _clone(store):
    st = copy.copy(store)
    st.stations = {k: dict(v) for k, v in store.stations.items()}
    return st


class _Fake:
    """Датасет, окна которого задаются явно."""

    def __init__(self, fps):
        self.fps = fps

    def footprints(self):
        return iter(self.fps)


def _fp(key, lo, t, sid, n=N_HOURS):
    """След одного окна: история с часа lo, горизонт с часа t."""
    return dict(sid=sid, N=n, time_key=key, lo=np.array([lo]), t=np.array([t]),
                hi=np.array([t + H]))


@pytest.mark.parametrize("n", [8_000, 12_000, 17_531, 87_660, 200_000])
def test_time_layout_ordered_disjoint_and_gapped(n):
    lay = check_time_layout(n)
    assert list(lay.blocks) == list(TIME_KEYS)
    assert lay.span("train")[0] == 0 and lay.span("test")[1] == n
    assert lay.span("val")[0] - lay.span("train")[1] >= MIN_GAP_HOURS == H + L_MAX
    assert lay.span("test")[0] - lay.span("calib")[1] >= MIN_GAP_HOURS
    for k in TIME_KEYS:
        for lo, hi in lay.blocks[k]:
            assert hi - lo > H, "блок не вмещает ни одной цели"


def test_layout_check_catches_broken_layout(monkeypatch):
    import mayak.leakage as LK
    good = time_layout(N_HOURS)
    val, calib = good.blocks["val"], good.blocks["calib"]
    swapped = TimeLayout(N_HOURS, dict(good.blocks, val=calib, calib=val))
    monkeypatch.setattr(LK, "time_layout", lambda n: swapped)
    with pytest.raises(LeakageError, match="не чередуются"):
        check_time_layout(N_HOURS)
    lo, hi = good.span("train")
    close = TimeLayout(N_HOURS, dict(good.blocks, train=((lo, hi + 100),)))
    monkeypatch.setattr(LK, "time_layout", lambda n: close)
    with pytest.raises(LeakageError, match="зазор между обучением"):
        check_time_layout(N_HOURS)


def test_time_layout_rejects_small_gap_odd_blocks_and_short_series():
    with pytest.raises(ValueError, match="зазор"):
        time_layout(50_000, gap_hours=MIN_GAP_HOURS - 1)
    with pytest.raises(ValueError, match="чётное"):
        time_layout(50_000, n_blocks=7)
    with pytest.raises(ValueError, match="короток"):
        time_layout(5_000)


def test_long_series_keeps_last_year_as_test():
    lay = time_layout(10 * 8766)
    assert lay.span("test") == (10 * 8766 - 8766, 10 * 8766)


def test_cache_records_climatology_fit_window(store):
    for s in store.stations.values():
        assert s["clim_fit"] == time_layout(s["N"]).span("train")


def test_sampled_train_windows_stay_in_train_window_with_gap(dm):
    ds, seen = dm.train_ds, []
    ds.build = lambda s, t, L: seen.append((s["id"], s["N"], t, L))
    for i in range(3000):
        ds[i]
    del ds.build
    assert {sid for sid, *_ in seen} <= {sid for sid, r in STATIONS if r == ROLE_TRAIN}
    assert any(L == 0 for *_, L in seen) and any(L == L_MAX for *_, L in seen)
    for _sid, N, t, L in seen:
        lay = time_layout(N)
        lo, hi = lay.span("train")
        assert lo <= t - L and t + H <= hi
        assert lay.span("val")[0] - (t + H) >= H + L_MAX, "окно ближе зазора к валидации"


@pytest.mark.parametrize("L", [0, 24, L_MAX, "full"])
def test_validation_items_match_declared_footprints(manifest, store, L):
    from mayak.data.dataset import qc_context
    from mayak.data.qc import DEFAULT_QC
    kw = dict(curriculum=L) if L == "full" else dict(L=L)
    ds = _eval_set(store, manifest, (ROLE_VAL,), "val", every_hours=48, **kw)
    fps = list(ds.footprints())
    assert len(ds) == len(fps) > 0
    for i, fp in enumerate(fps):
        sid, t = ds.items[i]
        floor, N = ds.floor[sid], store.stations[sid]["N"]
        n = ds.requested[i]
        L_real = int(ds[i]["mask_hist"][:, 0].sum())
        assert L_real == n == int(ds[i]["hist_len"]), \
            "окно оценки обрезало историю или QC отбраковал чистый ряд"
        ctx = qc_context(t, n, floor)
        assert (int(fp["lo"][0]), int(fp["t"][0]), int(fp["hi"][0])) == (t - n - ctx, t, t + H)
        assert ctx == (min(DEFAULT_QC.lookback_hours, t - n - floor) if n == L_MAX else 0)
        lay = time_layout(N)
        assert lay.block_index("val", [t], [t + H])[0] >= 0
        assert not lay.overlaps(("train", "test"), [t - L_MAX], [t])[0]


def test_eval_windows_do_not_depend_on_history_length(store, manifest):
    sets = [_eval_set(store, manifest, (ROLE_TRAIN, ROLE_TEST), "test", every_hours=48, L=L)
            for L in (None, 0, 24, L_MAX)]
    assert all(s.items == sets[0].items for s in sets)
    assert len(sets[0]) > 0
    for i in range(0, len(sets[0]), 7):
        assert int(sets[0][i]["mask_hist"][:, 0].sum()) == L_MAX
    check_windows(sets, store)


def test_datamodule_uses_role_and_window_contract(dm):
    assert dm.val_ds.station_splits == (ROLE_VAL,) and dm.val_ds.time_key == "val"
    assert dm.val_ds.history_spec() == dict(curriculum="full", L=None, seed=0)
    assert dm.train_ds.time_key == "train"
    assert {sid for sid, _t in dm.val_ds.items} == {"v0", "v1"}


def _rows(sizes, seed=0):
    """Страты заданных размеров: (зона, пояс) различаются, координаты случайны."""
    rng = np.random.default_rng(seed)
    zones = ["Af", "BWh", "Cfb", "Dfb", "ET", "Cfa", "BSk", "Csa"]
    rows = []
    for j, n in enumerate(sizes):
        for i in range(n):
            rows.append(dict(id=f"z{j}_{i:03d}", koppen=zones[j],
                             lat=round(float(rng.uniform(30, 45)), 3), lon=0.0))
    return rows


def test_roles_cover_every_station_exactly_once():
    rows = _rows([12, 9, 7, 5, 2, 1])
    roles = assign_roles(rows, n_test=6, val_frac=0.15, seed=3)
    assert set(roles) == {r["id"] for r in rows}
    assert set(roles.values()) <= set(ROLES)
    assert sum(v == ROLE_TEST for v in roles.values()) >= 6


def test_roles_reproducible_by_seed_and_independent_of_row_order():
    rows = _rows([10, 8, 6, 4, 3])
    a = assign_roles(rows, n_test=5, seed=7)
    rev = assign_roles(list(reversed(rows)), n_test=5, seed=7)
    perm = [rows[i] for i in np.random.default_rng(1).permutation(len(rows))]
    assert a == rev == assign_roles(perm, n_test=5, seed=7)
    assert a != assign_roles(rows, n_test=5, seed=8)


def test_small_test_count_does_not_zero_strata():
    rows = _rows([8] * 5)
    roles = assign_roles(rows, n_test=2, val_frac=0.1, seed=0)
    for key, cnt in strata_report(rows, roles).items():
        assert all(cnt.get(r, 0) >= 1 for r in ROLES), (key, cnt)


def test_every_stratum_keeps_a_training_station():
    rows = _rows([30, 2, 2, 2, 1, 1])
    for seed in range(20):
        roles = assign_roles(rows, n_test=8, val_frac=0.3, seed=seed)
        for key, cnt in strata_report(rows, roles).items():
            assert cnt.get(ROLE_TRAIN, 0) >= 1, (seed, key, cnt)


def test_checklist_passes_on_clean_pipeline(store, dm, manifest):
    sets = [dm.train_ds, dm.val_ds,
            _eval_set(store, manifest, (ROLE_VAL,), "calib", every_hours=24),
            _eval_set(store, manifest, (ROLE_TRAIN, ROLE_TEST), "test")]
    summary = run_checklist(store, datasets=sets, deep=True)
    assert summary["windows"] > 0


def test_checklist_catches_climatology_fit_outside_train_window(store):
    bad = _clone(store)
    lo, hi = time_layout(N_HOURS).span("test")
    bad.stations["x0"]["clim_fit"] = (lo, hi)
    with pytest.raises(LeakageError, match="климатология x0"):
        run_checklist(bad)


def test_deep_check_catches_coefficients_fit_on_test_window(store):
    from mayak.data.climatology import Climatology
    from mayak.timeaxis import window_calendar
    bad = _clone(store)
    s = bad.stations["t0"]
    lo, hi = time_layout(s["N"]).span("test")
    d, h = window_calendar(0, np.arange(lo, hi))
    s["clim"] = Climatology().fit(d.astype(float), h.astype(float), s["x"][lo:hi, 0] + 1.0,
                                  s["mask"][lo:hi, 0])
    check_climatology(bad)
    with pytest.raises(LeakageError, match="не воспроизводятся"):
        check_climatology(bad, deep=True)


def test_checklist_catches_target_crossing_block_boundary(store, manifest):
    ds = _eval_set(store, manifest, (ROLE_VAL,), "val", every_hours=72)
    check_windows([ds], store)
    lo, hi = time_layout(N_HOURS).blocks["val"][0]
    sid, _t = ds.items[0]
    ds.items[0] = (sid, hi - H + 1)
    with pytest.raises(LeakageError, match="не лежит целиком в одном блоке окна val"):
        check_windows([ds], store)
    ds.items[0] = (sid, lo - 1)
    with pytest.raises(LeakageError, match="блоке окна val"):
        check_windows([ds], store)


@pytest.mark.parametrize("key, sid, forbidden", [("val", "v0", "train"), ("calib", "v0", "train"),
                                                 ("test", "x0", "calib")])
def test_checklist_catches_evaluation_history_in_forbidden_window(store, key, sid, forbidden):
    lay = time_layout(N_HOURS)
    t = lay.blocks[key][0][0]
    if key != "test":
        check_windows([_Fake([_fp(key, t - L_MAX, t, sid)])], store)
    bad = _fp(key, lay.span(forbidden)[1] - 1, t, sid)
    with pytest.raises(LeakageError, match=f"заходит в окно {forbidden}"):
        check_windows([_Fake([bad])], store)


def test_checklist_catches_foreign_station_in_train_or_calib_window(store):
    """Станция читается только в окнах своей роли; внешняя - только в окне теста."""
    lay = time_layout(N_HOURS)
    lo, _ = lay.span("train")
    with pytest.raises(LeakageError, match="роли unseen_test в окне train"):
        check_windows([_Fake([_fp("train", lo, lo, "x0")])], store)
    clo, _ = lay.blocks["calib"][0]
    with pytest.raises(LeakageError, match="окне calib"):
        check_windows([_Fake([_fp("calib", clo, clo, "t0")])], store)
    ext = _clone(store)
    ext.stations["x0"]["role"] = ROLE_EXTERNAL
    for key in ("train", "val", "calib"):
        t = lay.blocks[key][0][0]
        with pytest.raises(LeakageError, match="x0"):
            check_windows([_Fake([_fp(key, t, t, "x0")])], ext)
    t = lay.span("test")[0]
    check_windows([_Fake([_fp("test", t - L_MAX, t, "x0")])], ext)


def _load_calibrate():
    spec = importlib.util.spec_from_file_location("calibrate_script",
                                                  REPO / "scripts" / "calibrate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibration_set_is_val_stations_in_calib_window(store, manifest):
    ds = _load_calibrate().calibration_set(store.clims(), manifest)
    assert ds.station_splits == (ROLE_VAL,) and ds.time_key == "calib"
    assert {sid for sid, _t in ds.items} == {"v0", "v1"}


def test_conformal_check(store, manifest, tmp_path):
    from mayak.constants import HISTORY_BINS
    shift = np.zeros((4, len(HISTORY_BINS), 7), np.float32)
    fit = [dict(bin=b[2], lo=b[0], hi=b[1], windows=0, marginal=True) for b in HISTORY_BINS]
    good = _load_calibrate().calibration_set(store.clims(), manifest)
    p = str(tmp_path / "ok.npy")
    save_conformal(p, shift, conformal_record(good, history_fit=fit))
    check_conformal(p, store)

    cases = {
        "all_roles": _eval_set(store, manifest, (ROLE_TRAIN, ROLE_TEST), "calib"),
        "val_window": _eval_set(store, manifest, (ROLE_VAL,), "val"),
        "test_window": _eval_set(store, manifest, (ROLE_VAL,), "test"),
    }
    for name, ds in cases.items():
        p = str(tmp_path / f"{name}.npy")
        save_conformal(p, shift, conformal_record(ds, history_fit=fit))
        with pytest.raises(LeakageError):
            check_conformal(p, store)

    np.save(tmp_path / "bare.npy", shift)
    with pytest.raises(LeakageError, match="нет метаданных"):
        check_conformal(str(tmp_path / "bare.npy"), store)

    moved = _clone(store)
    moved.stations["v1"]["role"] = ROLE_TRAIN
    with pytest.raises(LeakageError, match="не из роли"):
        check_conformal(str(tmp_path / "ok.npy"), moved)


def _ckpt(tmp_path, name, record):
    p = tmp_path / f"{name}.ckpt"
    torch.save({} if record is None else {SELECTION_KEY: record}, p)
    return str(p)


def test_checkpoint_selection_check(store, manifest, dm, tmp_path):
    check_checkpoint(_ckpt(tmp_path, "ok", selection_record(dm.val_ds, "val/loss")), store)

    on_test = _eval_set(store, manifest, (ROLE_TEST,), "test", curriculum="full")
    on_calib = _eval_set(store, manifest, (ROLE_VAL,), "calib", curriculum="full")
    fixed_L = _eval_set(store, manifest, (ROLE_VAL,), "val", L=L_MAX)
    bad = {
        "no_record": None,
        "train_metric": selection_record(dm.val_ds, "train/loss"),
        "no_monitor": selection_record(dm.val_ds, None),
        "test_window": selection_record(on_test, "val/loss"),
        "calib_window": selection_record(on_calib, "val/loss"),
        "fixed_history": selection_record(fixed_L, "val/loss"),
        "old_splits": dict(selection_record(dm.val_ds, "val/loss"), layout_code="0" * 16),
    }
    for name, rec in bad.items():
        with pytest.raises(LeakageError):
            check_checkpoint(_ckpt(tmp_path, name, rec), store)


def test_trainer_checkpoint_carries_selection_record(dm, store, tmp_path):
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from mayak.lit import LitForecaster, SelectionProvenance
    torch.manual_seed(0)
    ck = ModelCheckpoint(dirpath=str(tmp_path), monitor="val/loss", mode="min", filename="best")
    trainer = L.Trainer(max_steps=2, accelerator="cpu", devices=1, logger=False,
                        val_check_interval=1, check_val_every_n_epoch=None,
                        limit_val_batches=1, enable_progress_bar=False,
                        enable_model_summary=False, callbacks=[ck, SelectionProvenance()])
    trainer.fit(LitForecaster(arch="dlinear", total_steps=2), datamodule=dm)
    rec = torch.load(ck.best_model_path, map_location="cpu", weights_only=False)[SELECTION_KEY]
    assert rec["station_role"] == ROLE_VAL and rec["time_key"] == "val"
    assert rec["monitor"] == "val/loss" and rec["stations"] == ["v0", "v1"]
    assert rec["history"] == dict(curriculum="full", L=None, seed=0)
    assert rec["windows"] == len(dm.val_ds) and rec["windows_digest"] == dm.val_ds.fingerprint()
    assert "val/loss" in rec["scores"]
    run_checklist(store, checkpoints=[ck.best_model_path])


TRAINING_CODE = ["scripts/run.py", "mayak/data/datamodule.py", "mayak/lit.py"]


@pytest.mark.parametrize("rel", TRAINING_CODE)
def test_training_code_never_mentions_test_window(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    hits = re.findall(r"\btest\b|unseen_test|ROLE_TEST", src)
    assert not hits, f"{rel}: обращения к тестовой роли/окну: {hits}"


def test_stage_a_diagnostics_use_validation_window():
    src = (REPO / "scripts" / "diagnose_stage_a.py").read_text(encoding="utf-8")
    hits = re.findall(r"\btest\b|unseen_test|ROLE_TEST|EvalSet\(", src)
    assert not hits, f"диагностика этапа A строит свои окна или читает тестовое окно: {hits}"
    assert "validation_set(" in src and "train_stations_set(" in src
