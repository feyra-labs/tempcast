"""Тесты сборки обучающего набора: выбор точек, скачивание ERA5, сборка станций."""
import csv
import importlib.util
import json
import math
import runpy
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mayak.data import era5 as E
from mayak.data import points as PT
from mayak.data import store as S
from mayak.data.rasters import koppen_reader
from mayak.timeaxis import to_utc_hour
from mayak.zones import KG_TIF_CODE, KOPPEN_ID

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "tests" / "data" / "open_meteo" / "archive_era5_2points.json"
CODE = {z: i + 1 for z, i in KOPPEN_ID.items() if z in KG_TIF_CODE.values()}
TINY_ZONE = "Csc"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zone_array():
    """Карта зон с шагом 1 градус: три материка, остров из одной ячейки, Антарктида."""
    codes = np.zeros((180, 360), np.uint8)

    def put(zone, lat_top, lat_bottom, lon_left, lon_right):
        r0, r1 = 90 - lat_top, 90 - lat_bottom
        c0, c1 = lon_left + 180, lon_right + 180
        codes[r0:r1, c0:c1] = CODE[zone]

    for zone, top in (("Dfc", 60), ("Dfb", 50), ("Cfa", 40), ("BWh", 30)):
        put(zone, top, top - 10, -120, -60)
    for zone, top in (("Aw", 20), ("Af", 0), ("Csb", -20)):
        put(zone, top, top - 20, 10, 50)
    for zone, top in (("ET", 70), ("Dwc", 60), ("BSk", 50)):
        put(zone, top, top - 10, 60, 140)
    put(TINY_ZONE, 36, 35, 0, 1)
    put("EF", -60, -90, -180, 180)
    return codes


@pytest.fixture(scope="module")
def koppen_tif(tmp_path_factory):
    import rasterio
    from affine import Affine

    path = tmp_path_factory.mktemp("koppen") / "koppen.tif"
    codes = _zone_array()
    with rasterio.open(path, "w", driver="GTiff", height=180, width=360, count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=Affine(1.0, 0.0, -180.0, 0.0, -1.0, 90.0)) as ds:
        ds.write(codes, 1)
    return path


@pytest.fixture(scope="module")
def grid(koppen_tif):
    return PT.ZoneGrid.from_raster(koppen_tif, 1.0)


def _zones_with_land(include_antarctica=False):
    codes = _zone_array()
    lat = 89.5 - np.arange(180)
    if not include_antarctica:
        codes = codes[lat >= PT.ANTARCTICA_LAT]
    zones, cells = np.unique(codes[codes > 0], return_counts=True)
    return {KG_TIF_CODE[int(z)]: int(n) for z, n in zip(zones, cells)}


def test_grid_matches_raster(grid):
    assert grid.shape == (180, 360)
    assert np.array_equal(grid.codes, _zone_array())
    assert grid.code_at(35.5, 0.5) == CODE[TINY_ZONE]
    assert grid.code_at(0.0, -30.0) == 0


def test_lattice_is_even_on_the_sphere():
    lat, lon = PT.fibonacci_lattice(2000)
    assert lat.shape == lon.shape == (2000,)
    assert np.all(np.abs(lat) < 90) and np.all((lon >= -180) & (lon < 180))
    north = np.mean(lat > 0)
    band = np.mean(np.abs(lat) < 30)
    assert abs(north - 0.5) < 0.01
    assert abs(band - 0.5) < 0.01


def test_points_are_reproducible(grid):
    a = PT.select_points(grid, n_points=40, seed=3).rows()
    b = PT.select_points(grid, n_points=40, seed=3).rows()
    c = PT.select_points(grid, n_points=40, seed=4).rows()
    assert a == b
    assert a != c


def test_points_count_land_zones_and_spacing(grid, koppen_tif):
    sel = PT.select_points(grid, n_points=40, min_per_zone=3, min_dist_km=150, seed=0)
    rows = sel.rows()
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == 40

    zone_of = koppen_reader(koppen_tif)
    for r in rows:
        assert zone_of(r["lat"], r["lon"]) == r["koppen"], "точка не на суше или не в своей зоне"
        assert r["lat"] >= PT.ANTARCTICA_LAT

    have = {}
    for r in rows:
        have[r["koppen"]] = have.get(r["koppen"], 0) + 1
    for zone, cells in _zones_with_land().items():
        assert have.get(zone, 0) >= min(3, cells), f"зона {zone} недопредставлена"
    assert "EF" not in have
    assert sel.short == {CODE[TINY_ZONE]: 1}
    assert sel.relaxed == []
    assert PT.min_pairwise_km(sel.lat, sel.lon) >= 150.0


def test_antarctica_can_be_included(grid):
    sel = PT.select_points(grid, n_points=40, include_antarctica=True, seed=0)
    zones = {r["koppen"] for r in sel.rows()}
    assert "EF" in zones
    assert set(_zones_with_land(include_antarctica=True)) <= zones


def test_empty_zone_gets_one_point_even_if_too_close(grid):
    sel = PT.select_points(grid, n_points=4, min_per_zone=1, min_dist_km=30000, seed=0,
                           max_evals=20)
    zones = {r["koppen"] for r in sel.rows()}
    assert set(_zones_with_land()) <= zones
    assert sel.relaxed, "без ослабления расстояния зоны остались бы пустыми"


def test_make_points_script(koppen_tif, tmp_path, monkeypatch, capsys):
    out = tmp_path / "data" / "points.csv"
    argv = ["make_points", "--koppen", str(koppen_tif), "--out", str(out), "--n-points", "40",
            "--grid-step", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(REPO / "scripts" / "make_points.py"), run_name="__main__")
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(PT.POINTS_FIELDS)
    assert len(rows) == 40
    first = out.read_bytes()
    runpy.run_path(str(REPO / "scripts" / "make_points.py"), run_name="__main__")
    assert out.read_bytes() == first
    assert "fetch_era5" in capsys.readouterr().out


def test_default_period():
    assert E.default_period("2023-12-31", 10) == (date(2014, 1, 1),
                                                  date(2023, 12, 31))
    assert E.default_period("2024-02-29", 1) == (date(2023, 3, 1),
                                                 date(2024, 2, 29))
    assert E.default_period() == (date(2016, 1, 1), date(2025, 12, 31))


def test_request_fixes_model_timezone_and_variables():
    sig = E.request_signature("2014-01-01", "2023-12-31")
    q = E.request_params([52.35, -33.95], [4.95, 18.45], sig)
    assert q["models"] == "era5"
    assert q["timezone"] == "GMT" and q["timeformat"] == "unixtime"
    assert q["hourly"] == "temperature_2m,surface_pressure,relative_humidity_2m"
    assert q["latitude"] == "52.3500,-33.9500" and q["longitude"] == "4.9500,18.4500"
    assert "elevation" not in q, "высоту должен выбрать API по своей модели рельефа"


def test_parse_recorded_sample():
    payload = json.loads(SAMPLE.read_text("utf-8"))
    locs = E.parse_response(payload, len(payload))
    assert len(locs) == len(payload) == 2
    for raw, loc in zip(payload, locs):
        assert loc["cell_lat"] == raw["latitude"] and loc["cell_lon"] == raw["longitude"]
        assert loc["elevation"] == pytest.approx(raw["elevation"])
        assert np.all(np.diff(loc["hours"]) == 1)
        assert loc["hours"][0] * 3600 == raw["hourly"]["time"][0]
        for ch, var in E.CHANNELS:
            src = raw["hourly"][var]
            assert loc[ch].dtype == np.float32 and loc[ch].size == len(src)
            assert int(np.isnan(loc[ch]).sum()) == sum(v is None for v in src)
            ok = [i for i, v in enumerate(src) if v is not None]
            np.testing.assert_allclose(loc[ch][ok], [src[i] for i in ok], rtol=0, atol=1e-4)


def _one_location(**over):
    loc = dict(latitude=10.0, longitude=20.0, elevation=5.0, utc_offset_seconds=0,
               hourly_units={"temperature_2m": "°C"},
               hourly={"time": ["2020-01-01T00:00", "2020-01-01T01:00"],
                       "temperature_2m_era5": [1.0, None], "surface_pressure_era5": [1000, 999],
                       "relative_humidity_2m_era5": [50, 51]})
    loc.update(over)
    return loc


def test_parse_iso_time_and_model_suffix():
    (loc,) = E.parse_response(_one_location(), 1)
    assert loc["hours"][0] == to_utc_hour(datetime(2020, 1, 1))
    assert loc["T"][0] == 1.0 and math.isnan(loc["T"][1])
    assert loc["P"].tolist() == [1000.0, 999.0]


@pytest.mark.parametrize("payload, n, match", [
    ({"error": True, "reason": "Parameter out of range"}, 1, "out of range"),
    (_one_location(), 2, "точек"),
    (_one_location(utc_offset_seconds=3600), 1, "UTC"),
    (_one_location(hourly_units={"temperature_2m": "°F"}), 1, "единицы"),
    (_one_location(hourly={"time": [0, 3600]}), 1, "temperature_2m"),
])
def test_parse_rejects_bad_answers(payload, n, match):
    with pytest.raises(E.ApiError, match=match):
        E.parse_response(payload, n)


def _hours(sig):
    start = datetime.fromisoformat(sig["start_date"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(sig["end_date"]).replace(tzinfo=timezone.utc)
    n = int((end - start).total_seconds() // 3600) + 24
    return int(start.timestamp()), n


def _synthetic_location(lat, lon, t0, n, seed):
    """Правдоподобные ряды точки в том же виде, что записанный образец ответа."""
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = (15 - 0.3 * abs(lat) + 8 * np.cos(2 * np.pi * (h / 8766.0 - 0.55))
         + 5 * np.cos(2 * np.pi * (h + lon / 15.0 - 14.5) / 24) + 1.5 * rng.standard_normal(n))
    P = 1005 + 6 * np.sin(2 * np.pi * h / 150) + 0.3 * rng.standard_normal(n)
    RH = np.clip(65 - 15 * np.cos(2 * np.pi * (h + lon / 15.0 - 14.5) / 24)
                 + 4 * rng.standard_normal(n), 5, 100)
    return {"latitude": round(lat * 4) / 4, "longitude": round(lon * 4) / 4, "elevation": 120.0,
            "utc_offset_seconds": 0, "timezone": "GMT",
            "hourly_units": {"temperature_2m": "°C", "surface_pressure": "hPa",
                             "relative_humidity_2m": "%"},
            "hourly": {"time": (t0 + 3600 * h).tolist(),
                       "temperature_2m": np.round(T, 1).tolist(),
                       "surface_pressure": np.round(P, 1).tolist(),
                       "relative_humidity_2m": np.round(RH).tolist()}}


def _write_points(path, n=6):
    rows = [dict(id=f"p{i:04d}", lat=round(-30 + 11.3 * i, 4), lon=round(-100 + 23.7 * i, 4),
                 koppen="Cfb") for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PT.POINTS_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _read_points(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_downloaded(data, points, sig, skip=()):
    """Файлы точек и параметры скачивания в том виде, в каком их оставляет скачивание."""
    fe = _load_script("fetch_era5")
    t0, n = _hours(sig)
    done = 0
    for i, p in enumerate(points):
        if p["id"] in skip:
            continue
        loc = _synthetic_location(float(p["lat"]), float(p["lon"]), t0, n, seed=i)
        E.write_raw(E.raw_path(data / "era5", p["id"]),
                    E.make_raw(p, sig, loc, "2026-09-24T00:00:00+00:00"))
        done += 1
    counts = dict(cached=0, done=done, failed=0, pending=len(points) - done)
    fe.write_meta(str(data / "fetch_meta.json"), sig, E.ARCHIVE_URL, str(data / "points.csv"),
                  len(points), counts)


SIG = E.request_signature("2021-01-01", "2021-03-31")


def test_raw_file_roundtrip(tmp_path):
    points = _write_points(tmp_path / "points.csv", n=1)
    _write_downloaded(tmp_path, points, SIG)
    record = E.read_raw(E.raw_path(tmp_path / "era5", points[0]["id"]))
    assert record["request"] == SIG and record["id"] == points[0]["id"]
    assert E.raw_matches(record, points[0], SIG) == ""
    (loc,) = E.parse_response([record["response"]], 1)
    assert loc["T"].size == 90 * 24
    assert not list((tmp_path / "era5").glob("*.part")), "запись атомарна"


def test_rerun_skips_downloaded_points(tmp_path):
    fe = _load_script("fetch_era5")
    points = _write_points(tmp_path / "points.csv", n=4)
    done, todo = fe.classify(points, str(tmp_path / "era5"), SIG)
    assert done == [] and todo == points
    _write_downloaded(tmp_path, points, SIG, skip={"p0002"})
    done, todo = fe.classify(points, str(tmp_path / "era5"), SIG)
    assert [p["id"] for p in done] == ["p0000", "p0001", "p0003"]
    assert [p["id"] for p in todo] == ["p0002"], "качается только недостающее"


def test_other_parameters_or_points_are_refused(tmp_path):
    fe = _load_script("fetch_era5")
    points = _write_points(tmp_path / "points.csv", n=2)
    _write_downloaded(tmp_path, points, SIG)
    other = E.request_signature("2021-01-01", "2021-02-28")
    with pytest.raises(ValueError, match="end_date"):
        fe.classify(points, str(tmp_path / "era5"), other)
    moved = [dict(points[0], lat=points[0]["lat"] + 1.0), points[1]]
    with pytest.raises(ValueError, match="координаты"):
        fe.classify(moved, str(tmp_path / "era5"), SIG)
    done, todo = fe.classify(points, str(tmp_path / "era5"), other, refetch=True)
    assert done == [] and len(todo) == 2


def test_fetch_meta_describes_download(tmp_path):
    points = _write_points(tmp_path / "points.csv", n=3)
    _write_downloaded(tmp_path, points, SIG, skip={"p0001"})
    meta = json.loads((tmp_path / "fetch_meta.json").read_text("utf-8"))
    assert meta["endpoint"] == "/v1/archive" and meta["model"] == "era5"
    assert meta["variables"] == list(E.VARIABLES) and meta["timezone"] == "GMT"
    assert (meta["start_date"], meta["end_date"]) == ("2021-01-01", "2021-03-31")
    assert meta["request"] == SIG
    assert meta["downloaded"] == 2 and meta["pending"] == 1
    assert meta["script"]["name"] == "fetch_era5" and meta["script"]["version"]
    assert meta["first_fetch_utc"] <= meta["last_fetch_utc"]


def _run_script(monkeypatch, name, argv):
    monkeypatch.setattr(sys, "argv", [name, *argv])
    try:
        runpy.run_path(str(REPO / "scripts" / f"{name}.py"), run_name="__main__")
    except SystemExit as e:
        return e.code or 0
    return 0


def test_make_era5_needs_all_points(tmp_path, monkeypatch):
    points = _write_points(tmp_path / "points.csv", n=3)
    _write_downloaded(tmp_path, points, SIG, skip={"p0002"})
    code = _run_script(monkeypatch, "make_era5", ["--data", str(tmp_path)])
    assert code not in (0, None) and "не скачано 1 из 3" in str(code)
    assert _run_script(monkeypatch, "make_era5", ["--data", str(tmp_path),
                                                  "--allow-missing"]) == 0
    assert len(S.read_manifest(tmp_path / "manifest.csv")) == 2


def test_pipeline_from_points_to_cache(koppen_tif, tmp_path, monkeypatch):
    data = tmp_path / "data"
    assert _run_script(monkeypatch, "make_points", [
        "--koppen", str(koppen_tif), "--out", str(data / "points.csv"), "--n-points", "12",
        "--min-per-zone", "1", "--grid-step", "1"]) == 0
    points = _read_points(data / "points.csv")
    sig = E.request_signature("2019-01-01", "2020-05-31")
    _write_downloaded(data, points, sig)
    _, todo = _load_script("fetch_era5").classify(points, str(data / "era5"), sig)
    assert todo == [], "повторное скачивание ничего не запросило бы"

    assert _run_script(monkeypatch, "make_era5", ["--data", str(data)]) == 0
    manifest = data / "manifest.csv"
    with open(manifest, newline="") as f:
        header = next(csv.reader(f))
    assert header == ["id", "lat", "lon", "elev", "koppen", "cell_lat", "cell_lon"]
    rows = S.read_manifest(manifest)
    assert len(rows) == 12
    by_id = {p["id"]: p for p in points}
    for r in rows:
        assert r["koppen"] == by_id[r["id"]]["koppen"]
        assert float(r["elev"]) == 120.0
        src = S.read_source(S.source_path(manifest, r["id"]))
        assert src["t0"] == to_utc_hour(datetime(2019, 1, 1))
        assert src["T"].size == src["valid"].shape[0] == 517 * 24
        assert src["valid"].all()

    assert _run_script(monkeypatch, "make_splits", ["--manifest", str(manifest),
                                                    "--n-test", "2",
                                                    "--min-train-years", "0"]) == 0
    S._STORES.clear()
    path, built = S.build_cache(str(manifest), jobs=1)
    assert built
    with open(Path(path) / "meta.json") as f:
        meta = json.load(f)
    assert meta["qc"]["stations_total"] == 12
    assert meta["qc"]["stations_included"] == 12
    _, built = S.build_cache(str(manifest), jobs=1)
    assert not built
