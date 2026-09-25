"""Тесты: офлайн-кэш, векторный QC, сиды воркеров, календарь, точки входа."""
import ast
import csv
import importlib
import json
import os
import pkgutil
import runpy
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mayak
from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.qc import QCCode, _mad_ok_reference, mad_ok, qc_station, run_qc
from mayak.timeaxis import (future_calendar, legacy_t0, to_hourly_grid,
                            to_utc_hour, utc_to_doy_hour, window_calendar)

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 12_000
T0 = int(to_utc_hour(datetime(2019, 12, 30, 5)))       # ряд пересекает високосный 2020


def _series(n, seed):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * h / 24) + 0.3 * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    return T.astype(np.float32), P.astype(np.float32), RH.astype(np.float32)


def _write_station(root, sid, seed, n=N_HOURS, t0=T0, valid=None):
    T, P, RH = _series(n, seed)
    valid = np.ones((n, 3), np.uint8) if valid is None else valid
    np.savez(Path(root) / "stations" / f"{sid}.npz", T=T, P=P, RH=RH, valid=valid,
             t0_utc_h=np.int64(t0))


def _write_manifest(root, rows):
    with open(Path(root) / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader()
        w.writerows(rows)
    return str(Path(root) / "manifest.csv")


SPLITS = ["train", "train", "train", "unseen_val"]


def _make_dataset(root):
    (Path(root) / "stations").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, split in enumerate(SPLITS):
        _write_station(root, f"s{i}", seed=i)
        rows.append(dict(id=f"s{i}", lat=40.0 + i, lon=10.0 * i, elev=100.0,
                         koppen="Cfb", split=split))
    return _write_manifest(root, rows)


@pytest.fixture
def manifest(tmp_path):
    return _make_dataset(tmp_path / "data")


@pytest.fixture(autouse=True)
def _fresh_process_memo():
    S._STORES.clear()
    yield
    S._STORES.clear()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mad_vectorized_matches_reference(dtype):
    rng = np.random.default_rng(0)
    for trial in range(150):
        n = int(rng.integers(1, 300))
        x = (rng.standard_normal(n) * rng.uniform(0.1, 5)).astype(dtype)
        if trial % 3 == 0:
            x = np.round(x).astype(dtype)
        x[rng.random(n) < 0.03] += dtype(40)
        v = (rng.random(n) < rng.uniform(0.2, 1.0)).astype(np.uint8)
        x[(v == 0) & (rng.random(n) < 0.3)] = np.nan
        got = mad_ok(x, v, chunk=int(rng.integers(1, 50)))
        assert np.array_equal(got, _mad_ok_reference(x, v)), f"trial {trial}"


def test_qc_per_channel_mask_and_codes():
    n = 300
    T, P, RH = _series(n, 0)
    valid = np.ones((n, 3), np.uint8)
    valid[10, 1] = 0
    RH[20] = 130.0
    T[40] += 30.0
    x, mask, codes = qc_station(T, P, RH, valid)
    assert x.dtype == np.float32 and mask.dtype == np.uint8 and codes.dtype == np.uint8
    assert codes[10, 1] == QCCode.MISSING and mask[10, 0] == mask[10, 2] == 1
    assert codes[20, 2] == QCCode.RANGE and mask[20, 0] == mask[20, 1] == 1
    assert codes[40, 0] & QCCode.SPIKE and mask[40, 1] == mask[40, 2] == 1
    assert np.array_equal(mask, (codes == 0).astype(np.uint8))
    assert np.all(x[mask == 0] == 0)


def test_qc_1d_valid_is_broadcast():
    T, P, RH = _series(200, 1)
    v1 = np.ones(200, np.uint8)
    v1[50:60] = 0
    a = qc_station(T, P, RH, v1)
    b = qc_station(T, P, RH, np.repeat(v1[:, None], 3, 1))
    assert all(np.array_equal(u, w) for u, w in zip(a, b))
    xr, mr = run_qc(T, P, RH, v1)
    assert np.array_equal(xr, a[0]) and np.array_equal(mr, a[1].astype(np.float32))


def test_cache_build_then_hit(manifest):
    p1, built1 = S.build_cache(manifest)
    p2, built2 = S.build_cache(manifest)
    assert built1 and not built2 and p1 == p2
    root = Path(p1).parent
    assert not [d for d in os.listdir(root) if d.startswith(".tmp")], "остался недостроенный кэш"
    for f in ("x.npy", "mask.npy", "qc.npy", "clim_beta.npy", "index.json", "meta.json",
              "qc_report.csv"):
        assert (Path(p1) / f).exists()


def test_cache_content_equals_direct_qc(manifest):
    from mayak.data.climatology import Climatology
    from mayak.data.splits import time_layout
    store = S.get_store(manifest)
    for sid, s in store.stations.items():
        src = S.read_source(S.source_path(manifest, sid))
        x, mask, codes = qc_station(src["T"], src["P"], src["RH"], src["valid"])
        assert s["x"].dtype == np.float32 and s["mask"].dtype == np.uint8
        assert np.array_equal(s["x"], x) and np.array_equal(s["mask"], mask)
        assert np.array_equal(s["qc"], codes) and s["t0"] == T0
        lo, hi = time_layout(s["N"]).span("train")
        d, h = window_calendar(T0, np.arange(lo, hi))
        ref = Climatology().fit(d.astype(np.float64), h.astype(np.float64),
                                x[lo:hi, 0], mask[lo:hi, 0])
        assert np.allclose(s["clim"].beta, ref.beta) and s["clim"].sigma == pytest.approx(ref.sigma)


def _key(manifest):
    return S.cache_key(S.key_payload(manifest))


def test_cache_key_depends_on_content_not_path(manifest, tmp_path, monkeypatch):
    k0 = _key(manifest)
    root = Path(manifest).parent

    moved = tmp_path / "elsewhere"
    shutil.copytree(root, moved)
    assert _key(str(moved / "manifest.csv")) == k0

    rows = S.read_manifest(manifest)
    rows[0]["split"], rows[0]["lat"] = "unseen_test", "12.5"
    _write_manifest(root, rows)
    assert _key(manifest) == k0

    _write_station(root, "s0", seed=99)
    k1 = _key(manifest)
    assert k1 != k0

    _write_station(root, "s9", seed=9)
    _write_manifest(root, rows + [dict(rows[0], id="s9")])
    k2 = _key(manifest)
    assert k2 not in (k0, k1)

    monkeypatch.setattr(S, "TIME_LAYOUT", dict(S.TIME_LAYOUT, n_blocks=6))
    assert _key(manifest) != k2


def test_rebuild_after_source_change_is_picked_up(manifest):
    s_old = S.get_store(manifest)
    _write_station(Path(manifest).parent, "s1", seed=123)
    s_new = S.get_store(manifest)
    assert s_new.key != s_old.key
    assert not np.array_equal(s_new.stations["s1"]["x"], s_old.stations["s1"]["x"])


def test_one_store_per_process(manifest):
    from mayak import baselines as BL
    from mayak.data.dataset import HoldoutDataset, WindowDataset
    store = S.get_store(manifest)
    assert S.get_store(manifest) is store
    tr = WindowDataset(manifest, windows_per_epoch=8)
    va = HoldoutDataset(manifest, station_split="unseen_val", time_key="calib", every_hours=48)
    clims = BL.fit_climatologies(manifest)
    assert clims is store.stations
    assert np.shares_memory(tr.st[0]["x"], store.stations["s0"]["x"])
    assert np.shares_memory(va.meta[0]["x"], store.stations["s3"]["x"])


def test_station_without_climatology_is_excluded_and_reported(manifest):
    root = Path(manifest).parent
    valid = np.zeros((N_HOURS, 3), np.uint8)
    valid[-3000:] = 1
    _write_station(root, "s2", seed=2, valid=valid)
    path, _ = S.build_cache(manifest)
    meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
    assert "s2" in meta["excluded"]
    assert "s2" not in S.get_store(manifest).stations


def test_legacy_source_format_is_read(tmp_path):
    (tmp_path / "stations").mkdir()
    T, P, RH = _series(100, 0)
    np.savez(tmp_path / "stations" / "old.npz", T=T, P=P, RH=RH,
             valid=np.ones(100, np.uint8), t0_doy=np.float32(10.75), t0_hour=np.float32(18))
    src = S.read_source(tmp_path / "stations" / "old.npz")
    d, h = window_calendar(src["t0"], [0])
    assert (d[0], h[0]) == (10 + 18 / 24, 18.0)


def _pandas_ref(ts):
    ts = pd.DatetimeIndex(ts)
    return (ts.dayofyear - 1 + ts.hour / 24).to_numpy(), ts.hour.to_numpy().astype(float)


def test_calendar_matches_independent_reference():
    rng = np.random.default_rng(0)
    base = int(to_utc_hour(datetime(1999, 1, 1)))
    hrs = base + rng.integers(0, 30 * 8784, 5000)
    d, h = window_calendar(0, hrs)
    rd, rh = _pandas_ref(pd.to_datetime(hrs, unit="h"))
    assert np.allclose(d, rd, atol=1e-4) and np.array_equal(h, rh)
    assert np.allclose(d - np.floor(d), h / 24, atol=1e-4)


@pytest.mark.parametrize("ts, doy, hour", [
    (datetime(2021, 1, 1, 0), 0.0, 0.0),
    (datetime(2021, 1, 1, 6), 0.25, 6.0),
    (datetime(2020, 12, 31, 23), 365 + 23 / 24, 23.0),
    (datetime(2021, 12, 31, 23), 364 + 23 / 24, 23.0),
    (datetime(2021, 3, 1, 12, tzinfo=timezone(timedelta(hours=3))), 59 + 9 / 24, 9.0),
])
def test_calendar_fixed_points(ts, doy, hour):
    assert utc_to_doy_hour(ts) == pytest.approx((doy, hour))


def test_all_calendar_paths_agree(manifest):
    """Сборщик (сетка) → кэш → датасет → рантайм дают один и тот же календарь."""
    from mayak.data.dataset import WindowDataset
    times = pd.date_range("2020-02-27 22:00", periods=100, freq="h", tz="UTC")
    t0, _ = to_hourly_grid(times, {"T": np.zeros(100)})
    assert t0 == int(to_utc_hour(times[0]))

    ds = WindowDataset(manifest, windows_per_epoch=4)
    s = ds.st[0]
    t = int(s["starts"][len(s["starts"]) // 2])
    item = ds.build(s, t, 48)
    first_hist = datetime(1970, 1, 1) + timedelta(hours=s["t0"] + t - L_MAX)
    ref_d, ref_h = _pandas_ref(pd.date_range(first_hist, periods=L_MAX + H, freq="h"))
    got_d = np.concatenate([np.asarray(item["doy_hist"]), np.asarray(item["doy_fut"])])
    got_h = np.concatenate([np.asarray(item["hour_hist"]), np.asarray(item["hour_fut"])])
    assert np.allclose(got_d, ref_d, atol=1e-4) and np.array_equal(got_h, ref_h)

    last_obs = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(hours=s["t0"] + t - 1)
    rt_d, rt_h = utc_to_doy_hour(last_obs)
    assert (rt_d, rt_h) == pytest.approx((float(item["doy_hist"][-1]),
                                          float(item["hour_hist"][-1])))
    fd, fh = future_calendar(last_obs, H)
    assert np.allclose(fd, np.asarray(item["doy_fut"]), atol=1e-4)
    assert np.array_equal(fh, np.asarray(item["hour_fut"]))


def test_legacy_t0_does_not_double_fraction():
    assert legacy_t0(10.75, 18) == legacy_t0(10.0, 18)


def test_hourly_grid_marks_gaps_without_interpolation():
    times = pd.to_datetime(["2020-01-01 03:00", "2020-01-01 00:00", "2020-01-01 01:00"], utc=True)
    t0, cols = to_hourly_grid(times, {"T": [3.0, 0.0, 1.0]})
    assert t0 == int(to_utc_hour(times[1]))
    assert np.array_equal(cols["T"][[0, 1, 3]], [0.0, 1.0, 3.0]) and np.isnan(cols["T"][2])


@pytest.mark.parametrize("bad", [
    ["2020-01-01 00:00", "2020-01-01 00:30"],
    ["2020-01-01 00:00", "2020-01-01 00:00"],
])
def test_hourly_grid_fails_loudly(bad):
    with pytest.raises(ValueError):
        to_hourly_grid(pd.to_datetime(bad, utc=True), {"T": np.zeros(len(bad))})


def test_make_synth_writes_new_format(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_synth.py", "--out", str(tmp_path), "--n-stations", "2",
                                      "--years", "1"])
    runpy.run_path(str(REPO / "scripts" / "make_synth.py"), run_name="__main__")
    with np.load(tmp_path / "stations" / "S000.npz") as d:
        assert "t0_utc_h" in d and "t0_doy" not in d
        assert d["valid"].shape == (d["T"].shape[0], 3)


def _loader(manifest, seed, workers, persistent=True):
    import torch
    from torch.utils.data import DataLoader
    from mayak.data.dataset import WindowDataset, seed_worker
    ds = WindowDataset(manifest, windows_per_epoch=16, seed=seed)
    return DataLoader(ds, batch_size=4, num_workers=workers, worker_init_fn=seed_worker,
                      persistent_workers=persistent and workers > 0,
                      generator=torch.Generator().manual_seed(seed),
                      multiprocessing_context="fork" if workers else None)


def _batches(dl):
    return [{k: v.clone() for k, v in b.items()} for b in dl]


def _same(a, b):
    import torch
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.skipif(sys.platform != "linux", reason="нужен fork")
def test_workers_produce_different_windows(manifest):
    b = _batches(_loader(manifest, seed=0, workers=2))
    y0, y1 = b[0]["y"].numpy(), b[1]["y"].numpy()
    assert not any(np.array_equal(u, w) for u in y0 for w in y1)


@pytest.mark.skipif(sys.platform != "linux", reason="нужен fork")
def test_window_stream_reproducible(manifest):
    a = _batches(_loader(manifest, seed=7, workers=2))
    b = _batches(_loader(manifest, seed=7, workers=2))
    c = _batches(_loader(manifest, seed=8, workers=2))
    assert all(_same(u, w) for u, w in zip(a, b))
    assert not all(_same(u, w) for u, w in zip(a, c))


@pytest.mark.skipif(sys.platform != "linux", reason="нужен fork")
def test_non_persistent_workers_do_not_repeat_epochs(manifest):
    dl = _loader(manifest, seed=0, workers=2, persistent=False)
    e1, e2 = _batches(dl), _batches(dl)
    assert not all(_same(u, w) for u, w in zip(e1, e2))


def test_augmentation_stream_does_not_shift_sampling(manifest):
    from mayak.data.dataset import WindowDataset, make_streams
    ds = WindowDataset(manifest, windows_per_epoch=8, seed=0)
    ref = [np.asarray(ds[i]["y_mask"]) for i in range(8)]
    ds.seed_streams(0, 0)
    _, ds.rng_aug = make_streams(12345)
    got = [np.asarray(ds[i]["y_mask"]) for i in range(8)]
    assert all(np.array_equal(u, w) for u, w in zip(ref, got))


def _all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(mayak.__path__, "mayak."))


@pytest.mark.parametrize("mod", _all_modules())
def test_module_imports(mod):
    importlib.import_module(mod)


ENTRY_SCRIPTS = sorted(p.name for p in (REPO / "scripts").glob("*.py"))


def _entry_modules():
    out = []
    for path in sorted((REPO / "mayak").rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"), str(path))
        if not any(isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
                   for n in tree.body):
            continue
        out.append(".".join(path.relative_to(REPO).with_suffix("").parts))
    return out


ENTRY_MODULES = _entry_modules()


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_script_help(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")
    assert e.value.code == 0 and "usage" in capsys.readouterr().out


@pytest.mark.parametrize("mod", ENTRY_MODULES)
def test_module_help(mod, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [mod, "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    assert e.value.code == 0 and "usage" in capsys.readouterr().out
