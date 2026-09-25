"""Тесты: наблюдения реальной сети (GHCNh) как независимый внешний тест."""
import copy
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mayak.constants import H, L_MAX, MAGNUS_A, MAGNUS_B
from mayak.data import ghcnh as G
from mayak.data import store as S
from mayak.data.qc import QCCode, station_pressure_expected
from mayak.data.splits import (EXTERNAL_MIN_TRAIN_YEARS, ROLE_EXTERNAL, ROLE_TEST, ROLE_TRAIN,
                               ROLE_VAL, assign_roles, full_years,
                               min_hours_for_train_years, time_layout)
from mayak.leakage import (SELECTION_KEY, LeakageError, check_external, check_windows,
                           conformal_meta_path, run_checklist)
from mayak.metrics import NQ, Evaluation
from mayak.timeaxis import to_utc_hour

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data" / "ghcnh"
BY_YEAR = DATA / "GHCNh_TSX0000TEST_2020.psv"
POR = DATA / "GHCNh_TSX0000TEST_por.psv"
YEAR = 8766
NAN = np.nan


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXP_T = [5.0, 5.6, 6.1, 6.3, NAN, 7.0, 7.4, NAN, NAN, 8.0, NAN, NAN, 10.0]
EXP_P = [1000.0, 999.8, 999.5, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 998.0]
EXP_TD = [1.0, 1.1, NAN, 2.0, NAN, 8.0, 3.1, NAN, NAN, NAN, NAN, NAN, 4.0]
EXP_TD_T = [5.0, 5.6, NAN, 6.3, NAN, 7.0, 7.4, NAN, NAN, NAN, NAN, NAN, 10.0]
EXP_FLAG_T = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
EXP_FLAG_RH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


@pytest.fixture(scope="module")
def sample():
    return G.hourly_station(G.read_ghcnh(BY_YEAR))


def _eq_nan(a, b):
    """Сравнение float32-ряда с ожиданием (точность float32, NaN = NaN)."""
    np.testing.assert_allclose(np.asarray(a, np.float64),
                               np.asarray(b, np.float32).astype(np.float64), rtol=0,
                               atol=1e-4, equal_nan=True)


def test_sample_parses_to_expected_values_masks_and_flags(sample):
    assert sample.t0_utc_h == int(to_utc_hour(datetime(2020, 1, 1)))
    assert sample.n == 13
    _eq_nan(sample.T, EXP_T)
    _eq_nan(sample.P, EXP_P)
    _eq_nan(sample.Td, EXP_TD)
    np.testing.assert_array_equal(sample.valid[:, 0], np.isfinite(EXP_T))
    np.testing.assert_array_equal(sample.valid[:, 1], np.isfinite(EXP_P))
    np.testing.assert_array_equal(sample.valid[:, 2], np.isfinite(EXP_TD))
    np.testing.assert_array_equal(sample.flag[:, 0], EXP_FLAG_T)
    np.testing.assert_array_equal(sample.flag[:, 1], 0)
    np.testing.assert_array_equal(sample.flag[:, 2], EXP_FLAG_RH)
    for arr, v in ((sample.T, sample.valid[:, 0]), (sample.P, sample.valid[:, 1]),
                   (sample.RH, sample.valid[:, 2])):
        assert np.all(np.isnan(arr[v == 0])), "дыра - это NaN и valid=0, а не число"


def test_only_station_pressure_and_own_humidity_are_read(sample):
    """В образце SLP = 1013.2, альтиметр = 1013.5, готовая RH = 55: ни одно не читается."""
    p = sample.P[sample.valid[:, 1] > 0]
    assert not np.any(np.isclose(p, 1013.2)) and not np.any(np.isclose(p, 1013.5))
    rh = sample.RH[sample.valid[:, 2] > 0]
    assert not np.any(np.isclose(rh, 55.0))
    cols = G.wanted_columns()
    for name in G.IGNORED_ELEMENTS:
        assert not any(c.startswith(name) for c in cols), name


def test_por_and_by_year_layouts_parse_identically():
    """POR: время раздельными колонками, другой регистр имён, мусорная последняя строка."""
    a, b = G.read_ghcnh(BY_YEAR), G.read_ghcnh(POR)
    pd.testing.assert_frame_equal(a, b)


def test_report_types_off_fixed_stations_are_dropped():
    obs = G.read_ghcnh(BY_YEAR)
    t = obs.set_index("minute")
    m7 = int(to_utc_hour(datetime(2020, 1, 1, 7))) * 60
    assert np.isnan(t.loc[m7, "T"]), "FM-13 (судно) не значение станции"
    assert G.norm_report_type("FM-15") == G.norm_report_type("FM15") == "FM15"
    assert G.norm_report_type("AUTO_4-USA") == "AUTO"


@pytest.mark.parametrize("code, source, flagged", [
    ("1", "223", False), ("5", "343", False), ("4", "343", False), ("0", "313", False),
    ("2", "343", True), ("3", "343", True), ("6", "345", True), ("7", "346", True),
    ("5", "220", True), ("4", "221", True), ("2", "348", True), ("1", "347", False),
    ("A", "343", False), ("C", "343", False), ("C", "220", True),
    ("U", "343", True), ("I", "313", True), ("L", "343", True), ("o", "999", True),
    ("", "343", False), ("9", "220", False), ("nan", "", False), ("5.0", "343.0", False),
])
def test_quality_policy_is_source_dependent(code, source, flagged):
    assert G.quality_flagged(code, source) is flagged
    assert bool(G.quality_flags([code], [source])[0]) is flagged


def _obs(rows):
    """[(минуты от 00:00 1.1.2020, T, Td, P, флаг T)] → таблица наблюдений."""
    base = int(to_utc_hour(datetime(2020, 1, 1))) * 60
    df = pd.DataFrame(rows, columns=["m", "T", "Td", "P", "fT"])
    return pd.DataFrame(dict(minute=base + df["m"], T=df["T"], Td=df["Td"], P=df["P"],
                             flag_T=df["fT"].astype(bool), flag_Td=False, flag_P=False))


def test_hourly_grid_does_not_interpolate():
    s = G.hourly_station(_obs([(0, 1.0, NAN, NAN, 0), (5 * 60, 6.0, NAN, NAN, 0)]))
    assert s.n == 6
    _eq_nan(s.T, [1.0, NAN, NAN, NAN, NAN, 6.0])
    np.testing.assert_array_equal(s.valid[:, 0], [1, 0, 0, 0, 0, 1])
    assert not s.valid[:, 1:].any()


def test_hourly_grid_tie_breaks_prefer_clean_then_earlier():
    s = G.hourly_station(_obs([(55, 1.0, NAN, NAN, 0), (65, 2.0, NAN, NAN, 0),
                               (120, 3.0, NAN, NAN, 1), (132, 4.0, NAN, NAN, 0)]))
    assert s.t0_utc_h == int(to_utc_hour(datetime(2020, 1, 1, 1)))
    _eq_nan(s.T, [1.0, 4.0])
    assert s.flag[:, 0].sum() == 0
    only_flagged = G.hourly_station(_obs([(0, 3.0, NAN, NAN, 1)]))
    assert only_flagged.valid[0, 0] == 1 and only_flagged.flag[0, 0] == 1, \
        "помеченное значение не удаляется - оно уходит в маску через QC"


@pytest.mark.parametrize("tol", [-1, 30, 45])
def test_tolerance_must_keep_reports_to_one_hour(tol):
    with pytest.raises(ValueError):
        G.hourly_station(_obs([(0, 1.0, NAN, NAN, 0)]), tol_minutes=tol)


def test_tolerance_bounds_are_inclusive():
    s = G.hourly_station(_obs([(0, 0.0, NAN, NAN, 0), (60 + 10, 1.0, NAN, NAN, 0),
                               (180 - 11, 2.0, NAN, NAN, 0)]), tol_minutes=10)
    _eq_nan(s.T, [0.0, 1.0])


@pytest.mark.parametrize("every, cls", [(1, "1ч"), (3, "3ч"), (6, "6ч"), (4, "иное")])
def test_sparse_reporting_leaves_holes_and_is_classified(every, cls):
    rows = [(60 * k, 5.0 + 0.1 * k, NAN, NAN, 0) for k in range(0, 240, every)]
    s = G.hourly_station(_obs(rows))
    assert s.valid[:, 0].sum() == len(rows)
    assert G.report_step(s.valid[:, 0]) == every
    assert G.report_class(G.report_step(s.valid[:, 0])) == cls


def test_humidity_uses_the_models_formula_and_constants(sample):
    from mayak import astro
    assert G.rh_from_dewpoint is astro.rh_from_dewpoint, "одна реализация на проект"
    ok = sample.valid[:, 2] > 0
    T = np.asarray(EXP_TD_T)[ok]
    Td = np.asarray(EXP_TD)[ok]
    _eq_nan(sample.RH[ok], astro.rh_from_dewpoint(T, Td))
    g = lambda t: MAGNUS_A * t / (MAGNUS_B + t)
    manual = np.clip(100 * np.exp(g(Td) - g(T)), 0, 100)
    _eq_nan(sample.RH[ok], manual)


def test_humidity_round_trips_through_model_dewpoint():
    from mayak.astro import dewpoint_c
    rng = np.random.default_rng(0)
    T = rng.uniform(-30, 40, 500)
    Td = T - rng.uniform(0.1, 25, 500)
    obs = pd.DataFrame(dict(minute=np.arange(500) * 60, T=T, Td=Td, P=NAN, flag_T=False,
                            flag_Td=False, flag_P=False))
    s = G.hourly_station(obs)
    back = dewpoint_c(torch.tensor(s.T, dtype=torch.float64),
                      torch.tensor(s.RH, dtype=torch.float64)).numpy()
    ok = s.RH >= 1.0
    np.testing.assert_allclose(back[ok], Td[ok], atol=0.02)


def test_humidity_flag_follows_temperature_and_dewpoint():
    obs = _obs([(0, 5.0, 1.0, NAN, 1), (60, 5.0, 1.0, NAN, 0)])
    obs.loc[1, "flag_Td"] = True
    s = G.hourly_station(obs)
    np.testing.assert_array_equal(s.flag[:, 0], [1, 0])
    np.testing.assert_array_equal(s.flag[:, 2], [1, 1])


def test_station_list_fixed_width_and_csv(tmp_path):
    fw = G.read_station_list(DATA / "station-list-sample.txt")
    assert list(fw["id"]) == ["TSX0000TEST", "USW00094789", "ASN00009999"]
    assert fw.loc[1, "lat"] == pytest.approx(40.6392)
    assert fw.loc[1, "lon"] == pytest.approx(-73.7639)
    assert np.isnan(fw.loc[2, "elev"]), "−999.9 - пропуск высоты"
    assert fw.loc[1, "icao"] == "KJFK" and fw.loc[1, "wmo_id"] == "74486"
    p = tmp_path / "list.csv"
    p.write_text("GHCN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME,GSN,(US)HCN_(US)CRN,WMO_ID,ICAO\r\n"
                 "AAI0000TNCA,12.5014,-70.0152,18.3,,REINA BEATRIX INTL,,,,\r\n")
    c = G.read_station_list(p)
    assert c.loc[0, "id"] == "AAI0000TNCA" and c.loc[0, "elev"] == pytest.approx(18.3)


T_START = datetime(2014, 1, 1)
LON = 5.0
STATION_ELEV = 120.0


def _synthetic(n, seed, lon=LON):
    rng = np.random.default_rng(seed)
    abs_h = int(to_utc_hour(T_START)) + np.arange(n)
    hour = (abs_h % 24).astype(float)
    doy = ((abs_h // 24) % 365).astype(float)
    season = 9 * np.cos(2 * np.pi * (doy - 200) / 365.24)
    diurnal = 5 * np.cos(2 * np.pi * (hour + lon / 15 - 2.5 - 12) / 24)
    rho = math.exp(-1 / 60)
    e = rng.standard_normal(n) * 2.0 * math.sqrt(1 - rho ** 2)
    syn = np.zeros(n)
    for i in range(1, n):
        syn[i] = rho * syn[i - 1] + e[i]
    T = np.round(10 + season + diurnal + syn + 0.2 * rng.standard_normal(n), 1)
    Td = np.round(T - 4 - np.abs(rng.normal(0, 2, n)), 1)
    P = np.round(station_pressure_expected(STATION_ELEV) + 3 * syn
                 + 0.3 * rng.standard_normal(n), 1)
    return abs_h, T, Td, P


def _write_psv(root, sid, n, seed, drop=None, flag_hours=()):
    """Годовые PSV-файлы GHCNh синтетической станции (отчёты ровно в :00)."""
    abs_h, T, Td, P = _synthetic(n, seed)
    keep = np.ones(n, bool) if drop is None else ~drop(abs_h)
    ts = pd.to_datetime(abs_h * 3600, unit="s")
    q = np.where(np.isin(np.arange(n), list(flag_hours)), "3", "5")
    df = pd.DataFrame({"STATION": sid, "Station_name": sid,
                       "DATE": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                       "LATITUDE": "50.0", "LONGITUDE": str(LON), "ELEVATION": str(STATION_ELEV),
                       "temperature": T, "temperature_Quality_Code": q,
                       "temperature_Report_Type": "FM15", "temperature_Source_Code": "343",
                       "dew_point_temperature": Td, "dew_point_temperature_Quality_Code": "5",
                       "dew_point_temperature_Report_Type": "FM15",
                       "dew_point_temperature_Source_Code": "343",
                       "station_level_pressure": P, "station_level_pressure_Quality_Code": "4",
                       "station_level_pressure_Report_Type": "FM15",
                       "station_level_pressure_Source_Code": "343",
                       "sea_level_pressure": 1013.0})[keep]
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    for y, g in df.groupby(ts[keep].year):
        g.to_csv(d / f"GHCNh_{sid}_{y}.psv", sep="|", index=False)
    return abs_h, T, P


N_LONG = int(5.2 * YEAR)
N_SHORT = int(4.0 * YEAR)
FLAG_HOURS = (100, 101, 102)
GAP = (2000, 2010)


def _no_february(abs_h):
    return pd.to_datetime(abs_h * 3600, unit="s").month.to_numpy() == 2


@pytest.fixture(scope="module")
def external(tmp_path_factory):
    raw = tmp_path_factory.mktemp("ghcnh_raw")
    out = tmp_path_factory.mktemp("ghcnh")
    gap = lambda h: (np.arange(len(h)) >= GAP[0]) & (np.arange(len(h)) < GAP[1])
    _, T_long, _ = _write_psv(raw, "EXT_LONG", N_LONG, 1, drop=gap, flag_hours=FLAG_HOURS)
    _write_psv(raw, "EXT_LONG2", N_LONG, 2)
    _write_psv(raw, "EXT_SHORT", N_SHORT, 3)
    _write_psv(raw, "EXT_GAPPY", N_LONG, 4, drop=_no_february)
    stations = pd.DataFrame(dict(id=["EXT_LONG", "EXT_LONG2", "EXT_SHORT", "EXT_GAPPY", "EXT_NONE"],
                                 lat=50.0, lon=LON, elev=STATION_ELEV, name="x"))
    dem = lambda lat, lon: 100.0
    koppen = lambda lat, lon: "Cfb"
    rows, report = G.build_external_dataset(raw, out, stations, dem, koppen, min_train_years=1)
    manifest = str(out / "manifest.csv")
    S._STORES.clear()
    path, _ = S.build_cache(manifest)
    meta = json.loads((Path(path) / "meta.json").read_text())
    yield dict(raw=raw, out=out, manifest=manifest, rows=rows, report=report, meta=meta,
               path=path, T_long=T_long)
    S._STORES.clear()


def test_min_hours_for_train_years_matches_layout():
    for years in (1, 3, 5):
        n = min_hours_for_train_years(years)
        tr = time_layout(n).span("train")
        assert tr[1] - tr[0] >= years * YEAR
        tr_prev = time_layout(n - 24).span("train")
        assert tr_prev[1] - tr_prev[0] < years * YEAR


def test_full_years_requires_every_month():
    t0 = int(to_utc_hour(T_START))
    n = 4 * YEAR
    full = np.ones(n)
    assert full_years(full, t0, 0, n) == 4
    hole = full.copy()
    hole[_no_february(t0 + np.arange(n))] = 0
    assert full_years(hole, t0, 0, n) == 0, "год без февраля - неполный, сколько бы часов ни было"
    sparse = (np.arange(n) % 3 == 0).astype(float)
    assert full_years(sparse, t0, 0, n) == 4, "трёхчасовая отчётность - полный год"
    assert full_years(full, t0, 0, YEAR - 1) == 0


def test_builder_writes_store_contract_and_manifest(external):
    rows = {r["id"]: r for r in external["rows"]}
    assert set(rows) == {"EXT_LONG", "EXT_LONG2", "EXT_SHORT", "EXT_GAPPY"}
    rep = {r["id"]: r for r in external["report"]}
    assert rep["EXT_NONE"]["status"] == "excluded" and "нет скачанных" in rep["EXT_NONE"]["reason"]
    r = rows["EXT_LONG"]
    assert r["split"] == ROLE_EXTERNAL and r["koppen"] == "Cfb"
    assert r["elev"] == 100.0 and r["dem_elev"] == 100.0, "высота модели - из ЦМР"
    assert r["station_elev"] == STATION_ELEV and r["report_every"] == 1
    src = S.read_source(S.source_path(external["manifest"], "EXT_LONG"))
    assert set(src) >= {"T", "P", "RH", "valid", "t0", "flag", "Td"}
    assert src["t0"] == int(to_utc_hour(T_START))
    np.testing.assert_array_equal(src["valid"][GAP[0]:GAP[1]], 0)
    np.testing.assert_array_equal(src["flag"][list(FLAG_HOURS), 0], 1)
    ok = src["valid"][:, 0] > 0
    np.testing.assert_allclose(src["T"][ok], external["T_long"][ok], atol=1e-4)


def test_cache_requires_three_full_train_years(external):
    ex = external["meta"]["excluded"]
    assert "EXT_SHORT" in ex and "полных лет" in ex["EXT_SHORT"]
    assert "EXT_GAPPY" in ex and "полных лет" in ex["EXT_GAPPY"]
    assert external["meta"]["qc"]["excluded_by_rule"].get("clim_years") == 2
    store = S.get_store(external["manifest"])
    assert set(store.stations) == {"EXT_LONG", "EXT_LONG2"}
    for s in store.stations.values():
        lo, hi = time_layout(s["N"]).span("train")
        assert full_years(s["mask"][:, 0], s["t0"], lo, hi) >= EXTERNAL_MIN_TRAIN_YEARS
        assert s["role"] == ROLE_EXTERNAL and s["station_elev"] == STATION_ELEV


def test_source_flags_become_SOURCE_code_and_gaps_MISSING(external):
    s = S.get_store(external["manifest"]).stations["EXT_LONG"]
    assert np.all(s["qc"][list(FLAG_HOURS), 0] & QCCode.SOURCE)
    assert np.all(s["mask"][list(FLAG_HOURS), 0] == 0)
    assert np.all(s["mask"][list(FLAG_HOURS), 1] == 1), "флаг T не бракует давление"
    assert np.all(s["qc"][GAP[0]:GAP[1]] & QCCode.MISSING)
    assert np.all(s["x"][s["mask"] == 0] == 0), "инвариант хранения"
    assert s["qc_checks"]["dem_elevation"] == "pass", "станционная высота сверена с ЦМР"


def test_external_role_enters_cache_key_only_for_external_rows(external):
    m = external["manifest"]
    rows = S.read_manifest(m)
    k0 = S.cache_key(S.key_payload(m, rows))
    as_test = [dict(r, split=ROLE_TEST) for r in rows]
    assert S.cache_key(S.key_payload(m, as_test)) != k0, "правило 3 лет меняет отбор"
    as_val = [dict(r, split=ROLE_VAL) for r in rows]
    assert S.cache_key(S.key_payload(m, as_val)) == S.cache_key(S.key_payload(m, as_test))
    assert S.qc_meta(dict(lon="5", elev="100", station_elev="120"))["elev"] == 120.0
    assert S.qc_meta(dict(lon="5", elev="100"))["elev"] == 100.0


def test_assign_roles_leaves_external_stations_alone():
    rows = [dict(id=f"s{i:02d}", lat=45.0, koppen="Cfb", split="") for i in range(20)]
    rows += [dict(id=f"e{i}", lat=45.0, koppen="Cfb", split=ROLE_EXTERNAL) for i in range(5)]
    roles = assign_roles(rows, n_test=3, val_frac=0.2, seed=0)
    assert all(roles[f"e{i}"] == ROLE_EXTERNAL for i in range(5))
    inner = [roles[f"s{i:02d}"] for i in range(20)]
    assert inner.count(ROLE_TEST) == 3 and ROLE_EXTERNAL not in inner
    again = assign_roles([r for r in rows if r["split"] != ROLE_EXTERNAL], n_test=3, val_frac=0.2)
    assert {k: v for k, v in roles.items() if k.startswith("s")} == again, \
        "внешние станции не сдвигают разбиение остальных"


def test_make_splits_script_keeps_external_rows(tmp_path, monkeypatch):
    rows = [dict(id=f"s{i:02d}", lat=45.0, lon=0, elev=0, koppen="Cfb", split="")
            for i in range(12)] + [dict(id="ext", lat=45.0, lon=0, elev=0, koppen="Cfb",
                                        split=ROLE_EXTERNAL)]
    p = tmp_path / "manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(sys, "argv", ["make_splits.py", "--manifest", str(p), "--n-test", "3",
                                      "--min-train-years", "0"])
    _load_script("make_splits").main()
    out = {r["id"]: r["split"] for r in S.read_manifest(str(p))}
    assert out["ext"] == ROLE_EXTERNAL


MAIN = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL), ("x0", ROLE_TEST)]


@pytest.fixture(scope="module")
def main_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("main10")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(MAIN):
        rng = np.random.default_rng(i)
        n = 12_000
        h = np.arange(n)
        T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(n)
        P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
        RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
        np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32),
                 P=P.astype(np.float32), RH=RH.astype(np.float32),
                 valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))
        rows.append(dict(id=sid, lat=50.0 + 0.02 * i, lon=LON + 0.02 * i, elev=100.0,
                         koppen="Cfb", split=role))
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    return S.get_store(str(root / "manifest.csv"))


class _Fake:
    def __init__(self, fps):
        self.fps = fps

    def footprints(self):
        return iter(self.fps)


@pytest.mark.parametrize("key", ["train", "val", "calib"])
def test_external_stations_are_read_only_in_test_window(external, key):
    store = S.get_store(external["manifest"])
    s = store.stations["EXT_LONG"]
    lay = time_layout(s["N"])
    t = lay.blocks[key][0][0]
    fp = dict(sid="EXT_LONG", N=s["N"], time_key=key, lo=np.array([t]), t=np.array([t]),
              hi=np.array([t + H]))
    with pytest.raises(LeakageError, match="EXT_LONG"):
        check_windows([_Fake([fp])], store)
    t = lay.span("test")[0]
    ok = dict(fp, time_key="test", lo=np.array([t - L_MAX]), t=np.array([t]),
              hi=np.array([t + H]))
    check_windows([_Fake([ok])], store)


def test_check_external_passes_on_clean_pipeline(main_store, external):
    ext = S.get_store(external["manifest"])
    rep = check_external(main_store, ext)
    assert rep["external_stations"] == 2
    assert rep["near_train"] == 2, "обучающие точки в паре км: диагностика, не ошибка"
    summary = run_checklist(ext, external_store=ext)
    assert summary["external"]["external_stations"] == 2


def _clone(store):
    st = copy.copy(store)
    st.stations = {k: dict(v) for k, v in store.stations.items()}
    return st


def test_check_external_rejects_contamination(main_store, external, tmp_path):
    ext = S.get_store(external["manifest"])
    wrong_role = _clone(ext)
    wrong_role.stations["EXT_LONG"]["role"] = ROLE_TEST
    with pytest.raises(LeakageError):
        check_external(main_store, wrong_role)
    overlap = _clone(main_store)
    overlap.stations["EXT_LONG"] = dict(main_store.stations["t0"], id="EXT_LONG", role=ROLE_TRAIN)
    with pytest.raises(LeakageError):
        check_external(overlap, ext)
    ck = tmp_path / "bad.ckpt"
    torch.save({SELECTION_KEY: dict(stations=["v0", "EXT_LONG"])}, ck)
    with pytest.raises(LeakageError):
        check_external(main_store, ext, checkpoints=[str(ck)])
    good = tmp_path / "good.ckpt"
    torch.save({SELECTION_KEY: dict(stations=["v0"])}, good)
    check_external(main_store, ext, checkpoints=[str(good)])
    conf = tmp_path / "conformal.npy"
    np.save(conf, np.zeros((4, NQ)))
    Path(conformal_meta_path(str(conf))).write_text(json.dumps(
        dict(station_roles=[ROLE_VAL], stations=["v0", "EXT_LONG2"])))
    with pytest.raises(LeakageError):
        check_external(main_store, ext, conformal=str(conf))


def test_training_datasets_never_see_external_role(external):
    from mayak.data.dataset import HoldoutDataset, WindowDataset
    with pytest.raises(AssertionError):
        WindowDataset(external["manifest"], windows_per_epoch=4,
                      store=S.get_store(external["manifest"]))
    ds = HoldoutDataset(external["manifest"], store=S.get_store(external["manifest"]))
    assert len(ds) == 0, "валидация берёт только unseen_val"


def test_eval_set_meta_has_external_slices(external):
    from mayak.evaluate import EvalSet, external_breakdowns, PRESSURE_YES
    store = S.get_store(external["manifest"])
    ds = EvalSet(store.clims(), station_splits=(ROLE_EXTERNAL,), manifest=external["manifest"],
                 time_key="test", every_hours=48)
    assert len(ds) > 0
    meta = ds.window_meta()
    assert all(len(v) == len(ds) for v in meta.values())
    assert set(meta["report_class"]) == {"1ч"}
    assert set(meta["elev_gap"]) == {"|Δh| <50 м"}, "120 − 100 = 20 м"
    assert set(meta["has_pressure"]) == {PRESSURE_YES}
    run_checklist(store, datasets=[ds])
    n = len(ds)
    rng = np.random.default_rng(0)
    y = rng.normal(10, 5, (n, H))
    mu = y + rng.normal(0, 1, (n, H))
    ev = Evaluation(y=y, mu=mu, q=mu[..., None] + np.linspace(-3, 3, NQ),
                    mu_clim=y + rng.normal(0, 3, (n, H)), w=np.ones((n, H)),
                    station=meta["station"])
    out = external_breakdowns(ev, meta, leads=[24], min_windows=1, min_stations=1)
    assert set(out) == {"валидность истории", "частота отчётности", "Δ высоты станция−ЦМР",
                        "канал давления"}
    assert all(rows for rows in out.values())


def test_elevation_gap_labels():
    from mayak.external import elev_gap, elev_gap_label
    assert elev_gap(dict(station_elev=None, dem_elev=5.0)) is None
    assert elev_gap_label(None) == "нет данных"
    assert elev_gap_label(-60.0) == "|Δh| 50-150 м"
    assert elev_gap_label(400.0) == "|Δh| ≥300 м"


def _ev(n_st, per, bias, seed):
    rng = np.random.default_rng(seed)
    n = n_st * per
    y = rng.normal(10, 4, (n, H))
    mu = y + bias + rng.normal(0, 1, (n, H))
    q = mu[..., None] + np.linspace(-2, 2, NQ)
    station = np.repeat([f"s{seed}_{i}" for i in range(n_st)], per)
    return Evaluation(y=y, mu=mu, q=q, mu_clim=y + rng.normal(0, 3, (n, H)),
                      w=np.ones((n, H)), station=station)


def test_transfer_table_compares_same_leads_on_common_zones():
    from mayak.external import transfer_table
    ev_i, ev_e = _ev(6, 10, 0.0, 1), _ev(6, 10, 1.5, 2)
    z_i = np.repeat(["Cfb", "Cfb", "Dfb", "Dfb", "Af", "Af"], 10)
    z_e = np.repeat(["Cfa", "Cfb", "Dfc", "Dfb", "BWh", "BWh"], 10)
    tbl = transfer_table(ev_i, z_i, ev_e, z_e, leads=(1, 24), level="group", n_boot=200)
    assert tbl["zones"] == ["C", "D"], "A только внутри, B только снаружи"
    tbl_full = transfer_table(ev_i, z_i, ev_e, z_e, leads=(24,), level="full", min_stations=1,
                              n_boot=0)
    assert tbl_full["zones"] == ["Cfb", "Dfb"]
    r = tbl["rows"][24]
    common = ["Cfa", "Cfb", "Dfb", "Dfc"]
    sel_i, sel_e = np.isin(z_i, common), np.isin(z_e, common)
    ref_i = ev_i.restrict(windows=sel_i, leads=[24]).pooled()
    ref_e = ev_e.restrict(windows=sel_e, leads=[24]).pooled()
    for m in ("MAE", "Skill", "CRPS"):
        assert r["internal"]["pooled"][m] == pytest.approx(ref_i[m])
        assert r["diff"]["pooled"][m] == pytest.approx(ref_e[m] - ref_i[m])
    lo, hi = r["diff_ci"]["pooled"]["MAE"]
    assert lo < r["diff"]["pooled"]["MAE"] < hi and lo > 0, "смещение 1.5 °C видно в разнице MAE"
    assert r["internal"]["n_stations"] == 4 and r["external"]["n_stations"] == 4


def test_bootstrap_ci_is_quantiles_of_bootstrap_samples():
    from mayak.metrics import quantile_ci
    ev = _ev(5, 8, 0.0, 3)
    pooled, macro = ev.bootstrap_samples(n_boot=300, seed=7)
    ci = ev.bootstrap_ci(n_boot=300, seed=7)
    assert ci["pooled"] == quantile_ci(pooled) and ci["macro"] == quantile_ci(macro)


def test_zone_weights_modes_and_cap():
    from mayak.data.dataset import zone_weights
    zones = ["Cfb"] * 16 + ["ET"]
    inv = zone_weights(zones, "inv")
    assert inv[-1] == pytest.approx(0.5), "«inv»: одна станция = целая зона"
    sq = zone_weights(zones, "inv_sqrt")
    assert sq[-1] / sq[0] == pytest.approx(4.0) and sq[-1] == pytest.approx(4 / 20)
    uni = zone_weights(zones, "uniform")
    np.testing.assert_allclose(uni, 1 / 17)
    capped = zone_weights(zones, "inv", cap=2.0)
    assert capped.max() <= 2.0 * capped.mean() * (1 + 1e-9)
    assert capped.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        zone_weights(zones, "log")


def test_zone_weights_key_on_normalized_full_zone():
    from mayak.data.dataset import zone_distribution, zone_weights
    zones = ["Cfb", " Cfb ", "Cfa", "???", ""]
    w = zone_weights(zones, "inv")
    assert w[0] == pytest.approx(w[1]) and w[0] == pytest.approx(w[2] / 2)
    rep = zone_distribution(zones, w)
    assert rep["UNK"][0] == 2 and rep["Cfb"][0] == 2 and rep["Cfa"][0] == 1
    assert sum(p for _n, p in rep.values()) == pytest.approx(1.0)


def test_data_config_validates_zone_weighting_and_reaches_dataset(main_store):
    from mayak.config import ConfigError, DataConfig
    from mayak.data.dataset import WindowDataset
    assert DataConfig().zone_weighting == "inv_sqrt"
    for bad in (dict(zone_weighting="log"), dict(zone_weight_cap=0.5)):
        with pytest.raises(ConfigError):
            DataConfig(**bad)
    ds = WindowDataset(None, windows_per_epoch=4, store=main_store, zone_weighting="inv",
                       zone_weight_cap=0.0)
    assert ds.zone_report == {"Cfb": (2, pytest.approx(1.0))}


def test_raster_sampler_and_koppen_reader(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from mayak.data.rasters import RasterSampler, koppen_reader
    arr = np.array([[15, 0], [29, 99]], np.uint8)
    p = tmp_path / "kg.tif"
    with rasterio.open(p, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(0, 2, 1, 1), nodata=0) as ds:
        ds.write(arr, 1)
    kg = koppen_reader(p)
    assert kg(1.5, 0.5) == "Cfb" and kg(0.5, 0.5) == "ET"
    assert kg(1.5, 1.5) == "UNK" and kg(0.5, 1.5) == "UNK" and kg(10, 10) == "UNK"
    s = RasterSampler(p)
    assert s(1.5, 0.5) == 15.0 and s(1.5, 1.5) is None and s(-5, 0) is None


class _Resp:
    def __init__(self, body, status=200, fail_after=None):
        self.body, self.status, self.pos, self.fail_after = body, status, 0, fail_after
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n=-1):
        if n < 0:
            n = len(self.body)
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise ConnectionResetError("обрыв")
        chunk = self.body[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_resumes_after_break_and_memoizes_404(tmp_path, monkeypatch):
    import urllib.error
    fetch = _load_script("fetch_ghcnh")
    monkeypatch.setattr(fetch, "CHUNK", 10)
    body = bytes(range(256)) * 2
    calls = []

    def opener(req, timeout):
        rng = req.get_header("Range")
        calls.append(rng)
        if "missing" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        if rng is None:
            return _Resp(body, 200, fail_after=100)
        start = int(rng.split("=")[1].rstrip("-"))
        return _Resp(body[start:], 206)

    dest = str(tmp_path / "st" / "f.psv")
    assert fetch.download("https://x/f.psv", dest, open_url=opener, sleep=lambda s: None) == "done"
    assert Path(dest).read_bytes() == body
    assert calls == [None, "bytes=100-"], "докачка с места обрыва, а не заново"
    assert fetch.download("https://x/f.psv", dest, open_url=opener) == "cached"
    assert len(calls) == 2
    miss = str(tmp_path / "st" / "missing.psv")
    assert fetch.download("https://x/missing.psv", miss, open_url=opener) == "absent"
    assert fetch.download("https://x/missing.psv", miss, open_url=opener) == "absent"
    assert len(calls) == 3, "404 запомнен и не перезапрашивается"

    def always_down(req, timeout):
        raise urllib.error.URLError("сеть недоступна")
    with pytest.raises(ConnectionError):
        fetch.download("https://x/g.psv", str(tmp_path / "g.psv"), retries=2,
                       open_url=always_down, sleep=lambda s: None)


def test_file_urls_and_station_selection():
    fetch = _load_script("fetch_ghcnh")
    urls = fetch.file_urls("https://h/access/", "USW00094789", "by-year", [2019, 2020], "parquet")
    assert urls[0] == ("https://h/access/by-year/2019/parquet/GHCNh_USW00094789_2019.parquet",
                       "GHCNh_USW00094789_2019.parquet")
    assert fetch.file_urls("https://h/access", "X", "por")[0][0] == \
        "https://h/access/by-station/GHCNh_X_por.psv"
    assert fetch.parse_years("2018-2020,2015") == [2015, 2018, 2019, 2020]
    df = G.read_station_list(DATA / "station-list-sample.txt")
    assert list(fetch.select_stations(df, bbox=(30, 60, -80, 10))["id"]) == \
        ["TSX0000TEST", "USW00094789"]
    assert list(fetch.select_stations(df, countries=["US"])["id"]) == ["USW00094789"]
    assert list(fetch.select_stations(df, networks=["N"])["id"]) == ["ASN00009999"]
    assert len(fetch.select_stations(df, max_stations=2, seed=1)) == 2


def test_open_meteo_elevation_batches_and_caches(tmp_path):
    mk = _load_script("make_ghcnh")
    seen = []

    def opener(url, timeout):
        seen.append(url)
        n = url.split("latitude=")[1].split("&")[0].count("%2C") + 1
        return _Resp(json.dumps({"elevation": [10.0 * (i + 1) for i in range(n)]}).encode())

    dem = mk.OpenMeteoElevation(str(tmp_path / "dem.json"), open_url=opener, batch=2, pause=0)
    pts = [(50.0, 5.0), (51.0, 6.0), (52.0, 7.0)]
    dem.prefetch(pts)
    assert len(seen) == 2, "по 2 точки на запрос"
    assert dem(51.0, 6.0) == 20.0 and dem(52.0, 7.0) == 10.0
    again = mk.OpenMeteoElevation(str(tmp_path / "dem.json"), open_url=opener)
    again.prefetch(pts)
    assert len(seen) == 2, "ответы кэшируются на диске"
