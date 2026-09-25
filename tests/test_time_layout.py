"""Тесты раскладки ряда по временным окнам, правила окна и числа станций по ролям."""
import csv
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mayak.constants import H, L_MAX
from mayak.data import store as S
from mayak.data.dataset import block_starts, history_len
from mayak.data.splits import (BLOCK_KEYS, HISTORY_FORBIDDEN, MIN_GAP_HOURS, MIN_TRAIN_YEARS,
                               ROLE_EXTERNAL, ROLE_TEST, ROLE_TRAIN, ROLE_VAL, ROLES, TIME_KEYS,
                               TIME_LAYOUT, TimeLayout, assign_roles, min_hours_for_train_years,
                               strata_report, time_layout)
from mayak.leakage import LeakageError, check_time_layout, check_windows, run_checklist
from mayak.timeaxis import to_utc_hour, window_month
from mayak.zones import SEASONS, season_of

REPO = Path(__file__).resolve().parents[1]
YEAR = TIME_LAYOUT["hours_per_year"]
TEN_YEARS = 10 * YEAR
N_HOURS = 12_000
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL), ("v1", ROLE_VAL),
            ("x0", ROLE_TEST)]


def _write_station(root, sid, seed, n=N_HOURS):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("layout")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb",
                         split=role))
    return _write_manifest(root / "manifest.csv", rows)


@pytest.fixture(scope="module")
def store(manifest):
    S._STORES.clear()
    return S.get_store(manifest)


class _Fake:
    """Датасет, окна которого заданы явно."""

    def __init__(self, fps):
        self.fps = fps

    def footprints(self):
        return iter(self.fps)


def _fp(key, lo, t, sid="v0", n=N_HOURS):
    return dict(sid=sid, N=n, time_key=key, lo=np.array([lo]), t=np.array([t]),
                hi=np.array([t + H]))


@pytest.mark.parametrize("n", [N_HOURS, 26_400, TEN_YEARS, TEN_YEARS + 13])
def test_twelve_alternating_contiguous_blocks(n):
    lay = check_time_layout(n)
    assert len(lay.blocks["val"]) == len(lay.blocks["calib"]) == 6
    assert len(lay.blocks["train"]) == len(lay.blocks["test"]) == 1
    inner = sorted((lo, hi, k) for k in BLOCK_KEYS for lo, hi in lay.blocks[k])
    assert [k for *_b, k in inner] == ["val", "calib"] * 6
    for (_a, hi, _k), (lo, _b, _k2) in zip(inner, inner[1:]):
        assert hi == lo, "между блоками валидации и калибровки нет зазора"
    assert inner[0][0] - lay.span("train")[1] == MIN_GAP_HOURS
    assert lay.span("test")[0] - inner[-1][1] == MIN_GAP_HOURS
    assert {hi - lo for lo, hi, _k in inner} <= {(inner[-1][1] - inner[0][0]) // 12,
                                                (inner[-1][1] - inner[0][0]) // 12 + 1}


def test_ten_year_series_gives_year_blocks_and_long_training():
    lay = time_layout(TEN_YEARS)
    assert lay.span("test") == (TEN_YEARS - YEAR, TEN_YEARS)
    assert lay.span("calib")[1] - lay.span("val")[0] == YEAR
    lo, hi = lay.span("train")
    assert 7.8 < (hi - lo) / YEAR < 7.85


@pytest.mark.parametrize("lat", [50.0, -35.0])
@pytest.mark.parametrize("start", [datetime(2014, m, 1, tzinfo=timezone.utc)
                                   for m in range(1, 13)])
def test_every_season_in_both_roles(start, lat):
    """Середина хотя бы одного блока каждой роли приходится на каждый сезон."""
    t0 = int(to_utc_hour(start))
    lay = time_layout(TEN_YEARS)
    for key in BLOCK_KEYS:
        mids = [(lo + hi) // 2 for lo, hi in lay.blocks[key]]
        months = np.asarray(window_month(t0, mids)).ravel()
        assert {season_of(m, lat) for m in months} == set(SEASONS), (key, start)


def test_history_floor_keeps_evaluation_history_out_of_train_and_test():
    lay = time_layout(TEN_YEARS)
    assert lay.history_floor("train") == 0
    assert lay.history_floor("val") == lay.history_floor("calib") == lay.span("train")[1]
    assert lay.history_floor("test") == lay.span("calib")[1]
    for key in ("val", "calib", "test"):
        first = lay.blocks[key][0][0]
        assert first - lay.history_floor(key) >= L_MAX, "зазор короче полной истории"


def test_validation_windows_grow_about_fivefold():
    lay = time_layout(TEN_YEARS)
    ones = np.ones(TEN_YEARS, np.float32)
    _, new = block_starts(lay, "val", ones, step=72)
    old = len(range(0, 1440 - L_MAX - H + 1, 72))
    assert (old, len(new)) == (9, 48)


def test_training_years_requirement_follows_layout():
    n = min_hours_for_train_years(MIN_TRAIN_YEARS)
    lo, hi = time_layout(n).span("train")
    assert hi - lo >= MIN_TRAIN_YEARS * YEAR
    lo, hi = time_layout(n - 24).span("train")
    assert hi - lo < MIN_TRAIN_YEARS * YEAR
    assert TEN_YEARS > n


def test_block_index_and_overlaps():
    lay = TimeLayout(n_hours=100, blocks=dict(train=((0, 10),), val=((20, 30), (40, 50)),
                                              calib=((30, 40), (50, 60)), test=((80, 100),)))
    assert lay.block_index("val", [20, 25, 41, 29], [30, 31, 50, 29]).tolist() == [0, -1, 1, -1]
    assert lay.overlaps(("train", "test"), [5, 10, 70, 79], [15, 20, 80, 81]).tolist() == \
        [True, False, False, True]


def _datasets(manifest, store, L=None):
    from mayak.data.dataset import HoldoutDataset
    from mayak.evaluate import EvalSet
    kw = {} if L is None else dict(L=L)
    return [HoldoutDataset(manifest, time_key="val", every_hours=24, store=store, **kw),
            EvalSet(store.clims(), station_splits=(ROLE_VAL,), manifest=manifest,
                    time_key="calib", every_hours=24, max_windows=None, **kw),
            EvalSet(store.clims(), station_splits=(ROLE_TRAIN, ROLE_TEST), manifest=manifest,
                    time_key="test", every_hours=24, max_windows=None, **kw)]


def test_targets_inside_blocks_and_history_outside_train_and_test(manifest, store):
    """Проверка по раскладке напрямую, без чек-листа."""
    for ds in _datasets(manifest, store):
        for fp in ds.footprints():
            lay = time_layout(fp["N"])
            assert (lay.block_index(fp["time_key"], fp["t"], fp["hi"]) >= 0).all()
            assert (fp["t"] - fp["lo"] == L_MAX).all(), "окно оценки обрезало историю"
            forbidden = HISTORY_FORBIDDEN[fp["time_key"]]
            assert "train" in forbidden
            assert fp["time_key"] == "test" or "test" in forbidden
            assert not lay.overlaps(forbidden, fp["lo"], fp["t"]).any()
    run_checklist(store, datasets=_datasets(manifest, store))


def test_evaluation_history_leaves_its_block(manifest, store):
    """Правило действительно работает: у окон в начале блока история выходит из блока."""
    for ds in _datasets(manifest, store)[:2]:
        lay = time_layout(N_HOURS)
        firsts = {lo for lo, _hi in lay.blocks[ds.time_key]}
        starts = {int(t): int(lo) for fp in ds.footprints() for lo, t in zip(fp["lo"], fp["t"])}
        assert firsts <= set(starts), "первый час каждого блока служит началом горизонта"
        for t in firsts:
            assert starts[t] == t - L_MAX < t, "история окна начинается до его блока"


def test_test_windows_start_at_first_test_hour_with_full_history(manifest, store):
    ds = _datasets(manifest, store)[2]
    lo, _hi = time_layout(N_HOURS).span("test")
    firsts = [i for i, (_sid, t) in enumerate(ds.items) if t == lo]
    assert firsts, "тестовые окна начинаются с первого часа теста"
    for i in firsts:
        assert int(ds[i]["mask_hist"][:, 0].sum()) == L_MAX


def test_history_length_respects_floor():
    assert history_len(None, 1_000, 900) == 100
    assert history_len(L_MAX, 5_000, 0) == L_MAX


@pytest.mark.parametrize("key", ["val", "calib"])
def test_checklist_catches_evaluation_history_reaching_train(store, key):
    lay = time_layout(N_HOURS)
    t = lay.blocks[key][0][0]
    sid = "v0"
    check_windows([_Fake([_fp(key, t - L_MAX, t, sid)])], store)
    bad = _fp(key, lay.span("train")[1] - 1, t, sid)
    with pytest.raises(LeakageError, match="заходит в окно train"):
        check_windows([_Fake([bad])], store)


def test_checklist_catches_test_history_reaching_calibration(store):
    lay = time_layout(N_HOURS)
    t = lay.span("test")[0]
    with pytest.raises(LeakageError, match="заходит в окно calib"):
        check_windows([_Fake([_fp("test", lay.span("calib")[1] - 1, t, "x0")])], store)


def test_checklist_catches_history_before_series_start(store):
    with pytest.raises(LeakageError, match="выходит за ряд"):
        check_windows([_Fake([_fp("train", -5, 100, "t0")])], store)


def test_checklist_requires_horizon_start_and_full_horizon(store):
    lay = time_layout(N_HOURS)
    t = lay.blocks["val"][0][0]
    fp = _fp("val", t - 10, t)
    del fp["t"]
    with pytest.raises(LeakageError, match="нет поля t"):
        check_windows([_Fake([fp])], store)
    short = dict(_fp("val", t - 10, t), hi=np.array([t + H - 1]))
    with pytest.raises(LeakageError, match="длина цели"):
        check_windows([_Fake([short])], store)


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


def test_old_split_records_are_rejected(store):
    from mayak.leakage import check_selection_record
    rec = dict(monitor="val/loss", station_role=ROLE_VAL, time_key="val", stations=["v0"],
               time_layout=dict(TIME_LAYOUT))
    with pytest.raises(LeakageError, match="других правилах сплитов"):
        check_selection_record(rec, store)


def _rows(n, zones=("Af", "BWh", "Cfb", "Dfb", "ET"), seed=0):
    rng = np.random.default_rng(seed)
    return [dict(id=f"p{i:04d}", koppen=zones[i % len(zones)],
                 lat=round(float(rng.uniform(30, 45)), 3), lon=0.0) for i in range(n)]


def test_default_role_counts_follow_fractions():
    rows = _rows(350)
    info = {}
    roles = assign_roles(rows, info=info)
    count = {r: sum(v == r for v in roles.values()) for r in ROLES}
    assert info["requested"] == {ROLE_TEST: 52, ROLE_VAL: 30}
    assert count[ROLE_TEST] == 52 and count[ROLE_VAL] == 30
    assert info["assigned"] == count
    for key, cnt in strata_report(rows, roles).items():
        assert all(cnt.get(r, 0) >= 1 for r in ROLES), key


def test_exact_counts_still_override_fractions():
    roles = assign_roles(_rows(60), n_test=7, n_val=6)
    assert sum(v == ROLE_TEST for v in roles.values()) == 7
    assert sum(v == ROLE_VAL for v in roles.values()) == 6


def _make_splits(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["make_splits.py", *argv])
    try:
        runpy.run_path(str(REPO / "scripts" / "make_splits.py"), run_name="__main__")
    except SystemExit as e:
        return e.code or 0
    return 0


def _station_dir(tmp_path, lengths, external=()):
    (tmp_path / "stations").mkdir()
    rows = []
    for i, n in enumerate(lengths):
        sid = f"s{i:02d}"
        np.savez(tmp_path / "stations" / f"{sid}.npz", T=np.zeros(n, np.float32))
        rows.append(dict(id=sid, lat=45.0 + i % 3, lon=0.0, elev=0.0, koppen="Cfb", split=""))
    rows += [dict(id=e, lat=45.0, lon=0.0, elev=0.0, koppen="Cfb", split=ROLE_EXTERNAL)
             for e in external]
    return _write_manifest(tmp_path / "manifest.csv", rows)


def test_make_splits_refuses_series_too_short_for_training(tmp_path, monkeypatch):
    need = min_hours_for_train_years(MIN_TRAIN_YEARS)
    m = _station_dir(tmp_path, [need] * 9 + [need - 24], external=("ext",))
    code = _make_splits(monkeypatch, ["--manifest", m])
    assert code not in (0, None) and "s09" in str(code) and "--min-train-years 0" in str(code)
    assert all(r["split"] == "" for r in S.read_manifest(m) if r["id"] != "ext")
    assert _make_splits(monkeypatch, ["--manifest", m, "--min-train-years", "0"]) == 0


def test_make_splits_writes_report(tmp_path, monkeypatch):
    m = _station_dir(tmp_path, [TEN_YEARS] * 20, external=("ext",))
    assert _make_splits(monkeypatch, ["--manifest", m, "--seed", "3"]) == 0
    rows = S.read_manifest(m)
    rep = json.loads((tmp_path / "splits_report.json").read_text("utf-8"))
    assert rep["requested"] == {ROLE_TEST: 3, ROLE_VAL: 2}
    assert rep["assigned"] == {r: sum(x["split"] == r for x in rows)
                               for r in {x["split"] for x in rows}}
    assert rep["assigned"][ROLE_EXTERNAL] == 1
    assert rep["series"]["median_hours"] == TEN_YEARS
    assert rep["layout_of_median"]["train_years"] == pytest.approx(7.81, abs=0.01)
    assert rep["layout_of_median"]["block_hours"] == [730, 731]
    assert set(rep["layout_of_median"]["blocks"]) == set(TIME_KEYS)


def test_demo_data_pass_default_splits(tmp_path, monkeypatch):
    """Демо-данные по умолчанию достаточно длинные для правила трёх лет обучения."""
    monkeypatch.setattr(sys, "argv", ["make_synth.py", "--out", str(tmp_path), "--n-stations", "6"])
    runpy.run_path(str(REPO / "scripts" / "make_synth.py"), run_name="__main__")
    m = str(tmp_path / "manifest.csv")
    assert _make_splits(monkeypatch, ["--manifest", m]) == 0
    rep = json.loads((tmp_path / "splits_report.json").read_text("utf-8"))
    assert rep["layout_of_median"]["train_years"] >= MIN_TRAIN_YEARS
