"""Почасовые ряды реанализа ERA5 из архивного API Open-Meteo."""
from __future__ import annotations

import gzip
import json
import math
import os
from datetime import date, timedelta

import numpy as np

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
TIMEZONE = "GMT"
TIMEFORMAT = "unixtime"
CELL_SELECTION = "land"
CHANNELS = (("T", "temperature_2m"), ("P", "surface_pressure"), ("RH", "relative_humidity_2m"))
VARIABLES = tuple(v for _, v in CHANNELS)
EXPECTED_UNITS = {"temperature_2m": ("°C",), "surface_pressure": ("hPa",),
                  "relative_humidity_2m": ("%",)}

DEFAULT_END = "2025-12-31"
DEFAULT_YEARS = 10

RAW_FORMAT = 1
RAW_SUFFIX = ".json.gz"


class ApiError(RuntimeError):
    """API отказал в запросе по существу: неверный параметр или нет данных."""


def parse_date(value):
    """Дата из строки ISO или готового объекта даты."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def default_period(end=DEFAULT_END, years=DEFAULT_YEARS):
    """Период скачивания: заданное число лет, заканчивающихся указанным днём."""
    end = parse_date(end)
    try:
        before = end.replace(year=end.year - int(years))
    except ValueError:
        before = end.replace(year=end.year - int(years), day=28)
    return before + timedelta(days=1), end


def request_signature(start, end):
    """Всё, что определяет содержимое скачанных рядов, кроме координат."""
    return dict(endpoint="/v1/archive", model=MODEL, variables=list(VARIABLES),
                start_date=parse_date(start).isoformat(), end_date=parse_date(end).isoformat(),
                timezone=TIMEZONE, timeformat=TIMEFORMAT, cell_selection=CELL_SELECTION,
                elevation="api-dem-90m")


def request_params(lats, lons, signature):
    """Параметры запроса к архивному API для нескольких точек сразу."""
    return {"latitude": ",".join(f"{float(v):.4f}" for v in lats),
            "longitude": ",".join(f"{float(v):.4f}" for v in lons),
            "start_date": signature["start_date"], "end_date": signature["end_date"],
            "hourly": ",".join(signature["variables"]), "models": signature["model"],
            "timezone": signature["timezone"], "timeformat": signature["timeformat"],
            "cell_selection": signature["cell_selection"]}


def _hours_of(times):
    if len(times) == 0:
        return np.zeros(0, np.int64)
    if isinstance(times[0], str):
        t = np.asarray(times, dtype="datetime64[s]")
        sec = (t - np.datetime64("1970-01-01T00:00:00", "s")).astype(np.int64)
    else:
        sec = np.asarray(times, dtype=np.int64)
    if np.any(sec % 3600):
        raise ApiError("метки времени ответа не лежат на целых часах")
    return sec // 3600


def _series(hourly, var, n):
    """Ряд переменной из ответа, пустые значения становятся NaN."""
    key = var if var in hourly else f"{var}_{MODEL}"
    if key not in hourly:
        raise ApiError(f"в ответе нет переменной {var}")
    vals = hourly[key]
    if len(vals) != n:
        raise ApiError(f"длина ряда {var} ({len(vals)}) не совпадает с числом часов ({n})")
    arr = np.array(vals, dtype=object)
    arr[np.equal(arr, None)] = math.nan
    return arr.astype(np.float32)


def parse_location(loc):
    """Разбор ответа API для одной точки."""
    if loc.get("error"):
        raise ApiError(str(loc.get("reason", "ошибка без описания")))
    if int(loc.get("utc_offset_seconds", 0)) != 0:
        raise ApiError(f"ответ не в UTC: смещение {loc.get('utc_offset_seconds')} с")
    units = loc.get("hourly_units") or {}
    for var, allowed in EXPECTED_UNITS.items():
        unit = units.get(var, units.get(f"{var}_{MODEL}"))
        if unit is not None and unit not in allowed:
            raise ApiError(f"единицы {var}: {unit}, ожидались {allowed[0]}")
    hourly = loc.get("hourly")
    if not hourly or "time" not in hourly:
        raise ApiError("в ответе нет почасовых рядов")
    hours = _hours_of(hourly["time"])
    out = dict(cell_lat=float(loc["latitude"]), cell_lon=float(loc["longitude"]), hours=hours)
    elev = loc.get("elevation")
    out["elevation"] = None if elev is None or not math.isfinite(float(elev)) else float(elev)
    for ch, var in CHANNELS:
        out[ch] = _series(hourly, var, hours.size)
    return out


def parse_response(payload, n_expected):
    """Разбор ответа API на запрос по нескольким точкам."""
    if isinstance(payload, dict) and payload.get("error"):
        raise ApiError(str(payload.get("reason", "ошибка без описания")))
    locs = payload if isinstance(payload, list) else [payload]
    if len(locs) != int(n_expected):
        raise ApiError(f"в ответе {len(locs)} точек, в запросе {n_expected}")
    return [parse_location(loc) for loc in locs]


def raw_path(raw_dir, point_id):
    """Путь к файлу скачанной точки."""
    return os.path.join(raw_dir, f"{point_id}{RAW_SUFFIX}")


def make_raw(point, signature, response, fetched_utc):
    """Запись скачанной точки: запрошенные координаты, подпись запроса и ответ API."""
    return dict(format=RAW_FORMAT, id=str(point["id"]), lat=float(point["lat"]),
                lon=float(point["lon"]), request=dict(signature), fetched_utc=fetched_utc,
                response=response)


def write_raw(path, record):
    """Атомарная запись файла точки: неполный файл на диске не остаётся."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    os.replace(tmp, path)


def read_raw(path):
    """Чтение файла скачанной точки."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.load(f)
    if record.get("format") != RAW_FORMAT:
        raise ValueError(f"{path}: формат файла {record.get('format')}, ожидался {RAW_FORMAT}")
    return record


def raw_matches(record, point, signature):
    """Совпадает ли скачанный файл с точкой и параметрами текущего запроса."""
    if str(record.get("id")) != str(point["id"]):
        return f"id {record.get('id')} вместо {point['id']}"
    if (round(float(record["lat"]), 4), round(float(record["lon"]), 4)) != \
            (round(float(point["lat"]), 4), round(float(point["lon"]), 4)):
        return (f"координаты {record['lat']}, {record['lon']} вместо "
                f"{point['lat']}, {point['lon']}")
    if record.get("request") != signature:
        diff = sorted(k for k in set(signature) | set(record.get("request") or {})
                      if (record.get("request") or {}).get(k) != signature.get(k))
        return f"другие параметры запроса: {', '.join(diff)}"
    return ""


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по поверхности Земли между парами точек."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


__all__ = ["ARCHIVE_URL", "ApiError", "CHANNELS", "DEFAULT_END", "DEFAULT_YEARS", "MODEL",
           "RAW_SUFFIX", "VARIABLES", "default_period", "haversine_km",
           "make_raw", "parse_date", "parse_location", "parse_response", "raw_matches",
           "raw_path", "read_raw", "request_params", "request_signature", "write_raw"]
