"""Тесты: запись значений прибором - целые градусы и проценты, давление в десятых."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mayak.config import AUGMENT_PROB_FIELDS, AugmentConfig
from mayak.constants import L_MAX
from mayak.data import store as S
from mayak.data.augment import clean_history
from mayak.data.qc import CausalQC, QCCode
from mayak.data.recording import (is_recorded, record_channel, record_values,
                                  round_half_even)
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "tests" / "data" / "recording" / "cases.json"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_halves_go_to_even_including_negative():
    v = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    assert round_half_even(v).tolist() == [-4.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0]
    assert np.signbit(round_half_even(-0.4)) == np.False_, "без отрицательного нуля"
    assert np.isnan(round_half_even(np.nan))


def test_channels_have_their_own_grid():
    x = np.array([[12.5, 1013.25, 60.5], [-7.5, 1013.35, 99.5], [0.49, 999.95, 100.5]],
                 np.float32)
    r = record_values(x)
    np.testing.assert_array_equal(r[:, 0], [12.0, -8.0, 0.0])
    np.testing.assert_array_equal(r[:, 2], [60.0, 100.0, 100.0])
    np.testing.assert_array_equal(r[:, 1], np.float32([1013.2, 1013.3, 1000.0]))
    for j in range(3):
        np.testing.assert_array_equal(record_channel(x[:, j], j), r[:, j])


def test_record_is_idempotent_and_close():
    x = np.random.default_rng(0).normal([10, 950, 60], [15, 40, 25], (5000, 3))
    r = record_values(x)
    np.testing.assert_array_equal(record_values(r), r)
    err = np.abs(r - x)
    assert err[:, 0].max() <= 0.5 + 1e-5 and err[:, 2].max() <= 0.5 + 1e-5
    assert err[:, 1].max() <= 0.05 + 1e-3
    assert 0.27 < np.sqrt(np.mean((r[:, 0] - x[:, 0]) ** 2)) < 0.31, \
        "шум округления до целого - около 0.29 градуса"
    assert is_recorded(r) and not is_recorded(x)


def test_reference_cases_are_the_python_rule():
    """Общий с Rust файл случаев совпадает с правилом на Python и с ручной проверкой."""
    doc = json.loads(CASES.read_text(encoding="utf-8"))
    assert doc["cases"] == _script("make_recording_cases").cases(), \
        "файл устарел: python scripts/make_recording_cases.py"
    got = {(c["channel"], c["x"]): c["recorded"] for c in doc["cases"]}
    for key, want in {("T", -2.5): -2.0, ("T", 2.5): 2.0, ("T", -0.5): 0.0, ("T", 0.5): 0.0,
                      ("T", -3.5): -4.0, ("T", 60.5): 60.0, ("RH", 99.5): 100.0,
                      ("RH", 100.5): 100.0, ("RH", 0.5): 0.0,
                      ("P", 1013.25): float(np.float32(1013.2))}.items():
        assert got[key] == want, key
    halves = [c for c in doc["cases"] if c["channel"] == "T" and c["x"] % 1 == 0.5]
    assert any(c["x"] < 0 for c in halves) and len(halves) >= 8


def test_device_decides_on_the_recorded_value():
    qc = CausalQC(elev=200.0)
    v, codes = qc.push((12.5, 1001.25, 60.5))
    assert codes.tolist() == [0, 0, 0]
    np.testing.assert_array_equal(v, np.float32([12.0, 1001.2, 60.0]))
    _, codes = qc.push((60.4, 1100.04, 100.4))
    assert codes.tolist() == [0, 0, 0], "после записи значения в пределах диапазонов"
    _, codes = qc.push((60.6, 1100.06, 100.6))
    assert all(c & QCCode.RANGE for c in codes)
    _, codes = qc.push((None, float("nan"), 50.0))
    assert codes[0] & QCCode.MISSING and codes[1] & QCCode.MISSING


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("recording")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate([("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL),
                                     ("x0", ROLE_TEST)]):
        x, _ = clean_history(n=12_000, lat=45.0, lon=10.0, elev=200.0, seed=i)
        x = x + np.random.default_rng(i).uniform(-0.5, 0.5, x.shape).astype(np.float32)
        assert not is_recorded(x), "источник с дробными значениями"
        np.savez(root / "stations" / f"{sid}.npz", T=x[:, 0], P=x[:, 1], RH=x[:, 2],
                 valid=np.ones((len(x), 3), np.uint8), t0_utc_h=np.int64(0))
        rows.append(dict(id=sid, lat=45.0, lon=10.0, elev=200.0, koppen="Cfb", split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    return str(path)


def test_cache_holds_recorded_values(manifest):
    store = S.get_store(manifest)
    assert store.stations
    for s in store.stations.values():
        for key, mask in (("raw", "present"), ("x", "mask")):
            x, m = np.asarray(s[key]), np.asarray(s[mask])
            assert is_recorded(x, m), (s["id"], key)
            t = x[:, 0][m[:, 0] > 0]
            rh = x[:, 2][m[:, 2] > 0]
            p = x[:, 1][m[:, 1] > 0].astype(np.float64)
            assert np.array_equal(t, np.round(t)) and np.array_equal(rh, np.round(rh))
            np.testing.assert_allclose(p * 10, np.round(p * 10), atol=1e-3)


def test_recording_code_is_part_of_cache_key():
    assert "mayak.data.recording" in S.CACHE_CODE


@pytest.mark.parametrize("L", [0, 1, 48, L_MAX])
def test_training_window_is_recorded_after_every_augmentation(manifest, L):
    from mayak.data.dataset import WindowDataset
    all_on = AugmentConfig.from_profile("aggressive",
                                        **{f: 1.0 for f in AUGMENT_PROB_FIELDS.values()})
    ds = WindowDataset(manifest, windows_per_epoch=4, seed=0, augment=all_on)
    for i in range(12):
        s = ds.st[i % len(ds.st)]
        t = int(s["starts"][(13 * i) % len(s["starts"])])
        info = {}
        it = ds.build(s, t, L, info)
        x, m = it["x_hist"].numpy(), it["mask_hist"].numpy()
        assert is_recorded(x, m), sorted(info)
        y, ym = it["y"].numpy(), it["y_mask"].numpy()
        assert np.array_equal(y[ym > 0], np.round(y[ym > 0])), sorted(info)
