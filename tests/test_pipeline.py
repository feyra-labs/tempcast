"""Тесты: офлайн-кэш, векторный QC, сиды воркеров, календарь, точки входа."""
import csv
import importlib
import pkgutil
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mayak
from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.qc import DEFAULT_QC, MAD_TO_SD, QCCode, _spike_flags, qc_station
from mayak.data.recording import record_values
from mayak.data.window import issue_calendar
from mayak.timeaxis import to_hourly_grid, to_utc_hour, window_calendar

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


def _spike_reference(x, valid, half, thresh, min_valid, floor, causal):
    """Медленный эталон проверки выброса одного канала: медиана и MAD окна циклом по часам.

    Returns:
        Булев массив (N,): True там, где точка не выброс.
    """
    x = np.asarray(x, np.float64)
    before, after = (2 * half, 0) if causal else (half, half)
    n = len(x)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - before), min(n, i + after + 1)
        seg = x[lo:hi][valid[lo:hi] > 0]
        if len(seg) < min_valid:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-6
        if abs(x[i] - med) > thresh * max(MAD_TO_SD * mad, floor):
            ok[i] = False
    return ok


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spike_flags_match_reference(dtype):
    """Выбросы рабочего QC совпадают с эталоном по каждому каналу в обоих режимах окна."""
    cfg = DEFAULT_QC
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(1, 300))
        x = (rng.standard_normal((n, 3)) * rng.uniform(0.1, 5, 3)).astype(dtype)
        if trial % 3 == 0:
            x = np.round(x).astype(dtype)
        x[rng.random((n, 3)) < 0.03] += dtype(40)
        base = rng.random((n, 3)) < rng.uniform(0.2, 1.0)
        x[~base & (rng.random((n, 3)) < 0.3)] = np.nan
        for causal in (False, True):
            got = _spike_flags(x, base, cfg, causal)
            for j in range(3):
                ok = _spike_reference(x[:, j], base[:, j], cfg.spike_half, cfg.spike_thresh,
                                      cfg.spike_min_valid, cfg.scale_floor[j], causal)
                assert np.array_equal(got[:, j], base[:, j] & ~ok), \
                    f"trial {trial}, causal={causal}, канал {j}"


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


def test_cache_content_equals_direct_qc(manifest):
    from mayak.data.climatology import Climatology
    from mayak.data.splits import time_layout
    store = S.get_store(manifest)
    for sid, s in store.stations.items():
        src = S.read_source(S.source_path(manifest, sid))
        rec = record_values(np.stack([src["T"], src["P"], src["RH"]], -1))
        x, mask, codes = qc_station(rec[:, 0], rec[:, 1], rec[:, 2], src["valid"])
        assert s["x"].dtype == np.float32 and s["mask"].dtype == np.uint8
        assert np.array_equal(s["x"], x) and np.array_equal(s["mask"], mask)
        assert np.array_equal(s["qc"], codes) and s["t0"] == T0
        lo, hi = time_layout(s["N"]).span("train")
        d, h = window_calendar(T0, np.arange(lo, hi))
        ref = Climatology().fit(d.astype(np.float64), h.astype(np.float64),
                                x[lo:hi, 0], mask[lo:hi, 0])
        assert np.allclose(s["clim"].beta, ref.beta) and s["clim"].sigma == pytest.approx(ref.sigma)


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
    d, h = window_calendar(0, to_utc_hour(ts))
    assert (float(d), float(h)) == pytest.approx((doy, hour))


def test_all_calendar_paths_agree(manifest):
    """Сетка сборщика, датасет и устройство дают один и тот же календарь."""
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

    last = s["t0"] + t - 1
    device = issue_calendar(0, last + 1, L_MAX, H)
    for k, v in zip(("doy_hist", "hour_hist", "doy_fut", "hour_fut"), device):
        np.testing.assert_array_equal(np.asarray(item[k]), v, err_msg=k)


def test_hourly_grid_marks_gaps_without_interpolation():
    times = pd.to_datetime(["2020-01-01 03:00", "2020-01-01 00:00", "2020-01-01 01:00"], utc=True)
    t0, cols = to_hourly_grid(times, {"T": [3.0, 0.0, 1.0]})
    assert t0 == int(to_utc_hour(times[1]))
    assert np.array_equal(cols["T"][[0, 1, 3]], [0.0, 1.0, 3.0]) and np.isnan(cols["T"][2])


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


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_script_help(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")
    assert e.value.code == 0 and "usage" in capsys.readouterr().out
