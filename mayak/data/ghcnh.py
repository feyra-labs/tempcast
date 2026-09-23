"""Разбор почасовых наблюдений NOAA/NCEI GHCNh в контракт источника станции.

Источник: Global Historical Climatology Network hourly (GHCNh), документация
``ghcnh_DOCUMENTATION.pdf`` (версия формата 1.1.0). Файлы двух видов:

* по станции за весь период (``by-station/GHCNh_<id>_por.psv``) - время в колонках
  ``Year, Month, Day, Hour, Minute``;
* по станции за год (``by-year/<Y>/psv|parquet/GHCNh_<id>_<Y>.*``) - одна колонка
  ``DATE`` в ISO (UTC).

Регистр имён колонок между вариантами и версиями различается, поэтому имена
приводятся к нижнему регистру. У каждой переменной шесть полей: значение,
``_Measurement_Code``, ``_Quality_Code``, ``_Report_Type``, ``_Source_Code``,
``_Source_Station_ID``.

Читаются только три переменные: ``temperature``, ``dew_point_temperature`` и
``station_level_pressure`` с их кодами качества, типом отчёта и источником.
Давление - строго станционное: ``sea_level_pressure`` и ``altimeter`` приведены к
уровню моря и дали бы каналу разностей давления другой масштаб и суточный ход.
Готовая ``relative_humidity`` не используется: влажность восстанавливается из T и
Td той же формулой, что внутри модели (``mayak.astro.rh_from_dewpoint``).

Выход (``hourly_station``) - ровно контракт исходного файла станции для
``mayak.data.store``: T, P, RH (N,), Td (N,), valid (N, 3), flag (N, 3), t0_utc_h.
Интерполяции нет: час без наблюдения в пределах допуска - дыра (valid = 0).
Помеченное источником значение остаётся в ряду с flag = 1 и уходит в маску через
код QC ``SOURCE`` - не удаляется и не исправляется.
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mayak.astro import rh_from_dewpoint

GHCNH_PARSER_VERSION = "1"   # Увеличивать при изменении разбора; пишется в файл станции.

ELEMENTS = {"T": "temperature", "Td": "dew_point_temperature", "P": "station_level_pressure"}
ATTRS = ("quality_code", "report_type", "source_code")
IGNORED_ELEMENTS = ("sea_level_pressure", "altimeter", "relative_humidity")

MISSING_SENTINELS = (-9999.0, -999.9, -999.0, 9999.0, 99999.0)
EXCLUDED_REPORT_TYPES = frozenset({"FM13", "FM14", "FM18"})

# ---------------------------------------------------------------------------
# Политика кодов качества источника.
#
# Смысл кода зависит от источника (Source_Code):
#   * общие флаги QC GHCNh - буквы (L, o, F, U, D, d, W, K, C, T, S, h, V, w, N, E,
#     p, H): значение не прошло проверку;
#   * наследуемые коды источников 313-315, 322, 335, 343-346: 0/1/4/5 - прошло,
#     2/3/6/7 - подозрительно/ошибка; буквы A (подозрительно, но принято) и
#     C (целые °C от AWOS, приняты как валидные) - прошло; U/P/I/R/M - значение
#     заменено, вставлено или исправлено - это уже не показание прибора;
#   * наследуемые коды источников 220-223, 347, 348: 0/1 - прошло, 2 - подозрительно,
#     3 - ошибка, 4 - вычислено, 5 - удалено (!), 9 - нет кода.
# Для неизвестного источника: цифры 2/3/6/7 и любые буквы - флаг (консервативно).
# «9» и пустое поле - кода нет.
# ---------------------------------------------------------------------------
QUALITY_POLICY_VERSION = "1"
LEGACY_A_SOURCES = frozenset({"313", "314", "315", "322", "335", "343", "344", "345", "346"})
LEGACY_B_SOURCES = frozenset({"220", "221", "222", "223", "347", "348"})
BAD_DIGITS = frozenset({"2", "3", "6", "7"})
BAD_DIGITS_B = frozenset({"2", "3", "4", "5"})
ACCEPTED_LETTERS_A = frozenset({"A", "C"})
NO_CODE = frozenset({"", "9", "NAN", "NONE"})


def _norm_code(v) -> str:
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def quality_flagged(code, source) -> bool:
    """Код качества источника → True, если значение помечено подозрительным."""
    c, s = _norm_code(code), _norm_code(source)
    if c.upper() in NO_CODE:
        return False
    if c.isdigit():
        return c in (BAD_DIGITS_B if s in LEGACY_B_SOURCES else BAD_DIGITS)
    if s in LEGACY_A_SOURCES and c in ACCEPTED_LETTERS_A:
        return False
    return True


def quality_flags(codes, sources) -> np.ndarray:
    """Векторная версия ``quality_flagged`` (по уникальным парам код × источник)."""
    codes = pd.Series(codes, dtype="string").fillna("")
    sources = pd.Series(sources, dtype="string").fillna("")
    pair = codes.str.strip() + "|" + sources.str.strip()
    uniq, inv = np.unique(pair.to_numpy(dtype=object).astype(str), return_inverse=True)
    lut = np.array([quality_flagged(*u.split("|", 1)) for u in uniq], bool)
    return lut[inv] if len(uniq) else np.zeros(len(codes), bool)


def norm_report_type(rt) -> str:
    """'FM-15', 'FM15', 'AUTO_4-USA' → 'FM15', 'FM15', 'AUTO'."""
    return str(rt).strip().upper().split("_")[0].replace("-", "")


_ID_COLS = ("station", "station_id", "station_name", "latitude", "longitude", "elevation",
            "date", "year", "month", "day", "hour", "minute")


def wanted_columns():
    cols = set(_ID_COLS)
    for name in ELEMENTS.values():
        cols.add(name)
        cols.update(f"{name}_{a}" for a in ATTRS)
    return cols


def _read_raw(path):
    want = wanted_columns()
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
        df.columns = [str(c).lower() for c in df.columns]
        return df[[c for c in df.columns if c in want]].astype(str)
    return pd.read_csv(path, sep="|", dtype=str, keep_default_na=False,
                       usecols=lambda c: str(c).strip().lower() in want, low_memory=False) \
        .rename(columns=lambda c: str(c).strip().lower())


def _times(df):
    """Метки времени наблюдений → минуты от эпохи (int64); нечитаемые строки → -1."""
    if "date" in df.columns:
        t = pd.to_datetime(df["date"].str.strip(), utc=True, errors="coerce")
    else:
        parts = {k: pd.to_numeric(df[k], errors="coerce")
                 for k in ("year", "month", "day", "hour", "minute") if k in df.columns}
        if len(parts) < 4:
            raise ValueError("нет ни колонки DATE, ни колонок Year/Month/Day/Hour[/Minute]")
        parts.setdefault("minute", pd.Series(0, index=df.index))
        t = pd.to_datetime(pd.DataFrame(parts), utc=True, errors="coerce")
    ok = t.notna().to_numpy()
    ns = t.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return np.where(ok, ns // 60_000_000_000, -1)


def _values(s):
    v = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
    bad = np.isin(v, MISSING_SENTINELS)
    return np.where(bad, np.nan, v)


def read_ghcnh(paths) -> pd.DataFrame:
    """Файлы GHCNh одной станции → таблица наблюдений.

    Колонки: minute (минуты UTC от эпохи), T, Td, P (NaN - нет значения),
    flag_T, flag_Td, flag_P (bool - помечено источником), station, lat, lon, elev.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    frames = []
    for p in paths:
        df = _read_raw(p)
        if not len(df):
            continue
        out = pd.DataFrame({"minute": _times(df)})
        for key, name in ELEMENTS.items():
            if name not in df.columns:
                out[key] = np.nan
                out[f"flag_{key}"] = False
                continue
            v = _values(df[name])
            rt = df.get(f"{name}_report_type", pd.Series("", index=df.index)).map(norm_report_type)
            v[rt.isin(EXCLUDED_REPORT_TYPES).to_numpy()] = np.nan
            out[key] = v
            out[f"flag_{key}"] = quality_flags(
                df.get(f"{name}_quality_code", pd.Series("", index=df.index)),
                df.get(f"{name}_source_code", pd.Series("", index=df.index)))
        sid = df.get("station", df.get("station_id"))
        out["station"] = sid.str.strip().to_numpy() if sid is not None else ""
        for src, dst in (("latitude", "lat"), ("longitude", "lon"), ("elevation", "elev")):
            out[dst] = _values(df[src]) if src in df.columns else np.nan
        frames.append(out[out["minute"] >= 0])
    if not frames:
        return pd.DataFrame(columns=["minute", "T", "Td", "P", "flag_T", "flag_Td", "flag_P",
                                     "station", "lat", "lon", "elev"])
    return pd.concat(frames, ignore_index=True).sort_values("minute", kind="stable") \
        .reset_index(drop=True)


def nearest_to_hour(minute, present, flagged, tol_minutes):
    """Для каждого целого часа - индекс отчёта, ближайшего к нему в пределах допуска."""
    minute = np.asarray(minute, np.int64)
    idx = np.flatnonzero(np.asarray(present, bool))
    if idx.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    m = minute[idx]
    hour = np.floor_divide(m + 30, 60)
    dist = np.abs(m - 60 * hour)
    keep = dist <= tol_minutes
    idx, m, hour, dist = idx[keep], m[keep], hour[keep], dist[keep]
    fl = np.asarray(flagged, bool)[idx]
    order = np.lexsort((m, dist, fl, hour))
    h_sorted = hour[order]
    first = np.r_[True, h_sorted[1:] != h_sorted[:-1]]
    return h_sorted[first], idx[order][first]


@dataclass
class HourlySeries:
    """Почасовой ряд станции в контракте исходного файла ``mayak.data.store``."""
    t0_utc_h: int
    T: np.ndarray
    P: np.ndarray
    RH: np.ndarray
    Td: np.ndarray
    valid: np.ndarray
    flag: np.ndarray

    @property
    def n(self):
        return len(self.T)

    def to_npz_dict(self):
        return dict(T=self.T, P=self.P, RH=self.RH, Td=self.Td, valid=self.valid,
                    flag=self.flag, t0_utc_h=np.int64(self.t0_utc_h),
                    source=np.array("ghcnh"), parser_version=np.array(GHCNH_PARSER_VERSION),
                    quality_policy=np.array(QUALITY_POLICY_VERSION))


def hourly_station(obs: pd.DataFrame, tol_minutes=20) -> HourlySeries:
    """Таблица наблюдений (``read_ghcnh``) → почасовой ряд без интерполяции.

    Каналы выбираются независимо (по-канальная маска). RH считается в каждом
    отчёте, где есть и T, и Td, и выбирается как единое целое вместе с Td этого
    отчёта: T и Td для влажности всегда из одного отчёта. RH помечен, если помечена
    T или Td этого отчёта.
    """
    if not 0 <= tol_minutes < 30:
        raise ValueError("допуск должен лежать в [0, 30) мин - иначе отчёт попадёт в два часа")
    minute = obs["minute"].to_numpy(np.int64)
    T, Td, P = (obs[k].to_numpy(np.float64) for k in ("T", "Td", "P"))
    fT, fTd, fP = (obs[f"flag_{k}"].to_numpy(bool) for k in ("T", "Td", "P"))
    has_rh = np.isfinite(T) & np.isfinite(Td)
    rh = np.full(len(T), np.nan)
    if has_rh.any():
        rh[has_rh] = rh_from_dewpoint(T[has_rh], Td[has_rh])
    f_rh = fT | fTd

    picks = {name: nearest_to_hour(minute, np.isfinite(v), f, tol_minutes)
             for name, v, f in (("T", T, fT), ("P", P, fP), ("RH", rh, f_rh))}
    hours = np.concatenate([h for h, _ in picks.values()])
    if hours.size == 0:
        raise ValueError("нет ни одного наблюдения в пределах допуска от целого часа")
    t0 = int(hours.min())
    n = int(hours.max()) - t0 + 1

    out = {k: np.full(n, np.nan, np.float32) for k in ("T", "P", "RH", "Td")}
    valid = np.zeros((n, 3), np.uint8)
    flag = np.zeros((n, 3), np.uint8)
    for j, (name, src, f) in enumerate((("T", T, fT), ("P", P, fP), ("RH", rh, f_rh))):
        h, i = picks[name]
        pos = h - t0
        out[name][pos] = src[i]
        valid[pos, j] = 1
        flag[pos, j] = f[i]
        if name == "RH":
            out["Td"][pos] = Td[i]
    return HourlySeries(t0_utc_h=t0, T=out["T"], P=out["P"], RH=out["RH"], Td=out["Td"],
                        valid=valid, flag=flag)


def report_step(mask_T) -> int:
    """Типичный шаг отчётности станции, ч: мода интервалов между валидными часами T.

    0 - меньше двух валидных часов.
    """
    idx = np.flatnonzero(np.asarray(mask_T) > 0)
    if idx.size < 2:
        return 0
    d = np.diff(idx)
    vals, cnt = np.unique(d, return_counts=True)
    return int(vals[np.argmax(cnt)])


REPORT_CLASSES = {1: "1ч", 3: "3ч", 6: "6ч"}


def report_class(step) -> str:
    """Шаг отчётности → класс «1ч» / «3ч» / «6ч» / «иное»."""
    return REPORT_CLASSES.get(int(step), "иное")


STATION_LIST_FWF = dict(colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71),
                                  (72, 75), (76, 79), (80, 85), (86, 90)],
                        names=["id", "lat", "lon", "elev", "state", "name", "gsn", "hcn_crn",
                               "wmo_id", "icao"])


def read_station_list(path_or_text) -> pd.DataFrame:
    """``ghcnh-station-list.txt`` (фиксированная ширина) или ``.csv`` → таблица.

    Колонки: id, lat, lon, elev (NaN вместо −999.9), state, name, wmo_id, icao.
    """
    text = path_or_text
    if os.path.exists(str(path_or_text)):
        with open(path_or_text, encoding="utf-8", errors="replace") as f:
            text = f.read()
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.upper().startswith("GHCN_ID,") or first.upper().startswith("ID,"):
        rows = list(csv.reader(io.StringIO(text)))
        df = pd.DataFrame(rows[1:])
        names = ["id", "lat", "lon", "elev", "state", "name", "gsn", "hcn_crn", "wmo_id", "icao"]
        df = df.iloc[:, :len(names)]
        df.columns = names[:df.shape[1]]
    else:
        df = pd.read_fwf(io.StringIO(text), header=None, dtype=str, **STATION_LIST_FWF)
    df = df.fillna("")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    for c in ("lat", "lon", "elev"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["elev"] <= -999, "elev"] = np.nan
    return df[df["id"].str.len() == 11].reset_index(drop=True)


MANIFEST_FIELDS = ("id", "lat", "lon", "elev", "station_elev", "dem_elev", "koppen", "split",
                   "report_every", "name")


def station_files(raw_dir, sid):
    d = os.path.join(raw_dir, sid)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith(f"GHCNh_{sid}_") and f.endswith((".psv", ".parquet")))


def build_external_dataset(raw_dir, out_dir, stations, dem, koppen, tol_minutes=20,
                           min_train_years=None, min_valid_frac_T=0.5):
    """Скачанные файлы GHCNh → каталог набора внешнего теста.

    stations - таблица ``read_station_list`` (id, lat, lon, elev, name);
    dem      - функция (lat, lon) → высота из ЦМР, м, или None;
    koppen   - функция (lat, lon) → полная зона Кёппена.
    Пишет ``<out>/stations/<id>.npz`` (контракт ``mayak.data.store``),
    ``<out>/manifest.csv`` (роль external_test; elev - из ЦМР, station_elev - из
    списка станций) и ``<out>/selection_report.csv`` с причиной по каждой станции.
    Предотбор здесь грубый (длина ряда и доля валидной T); окончательный отбор -
    правила QC при сборке кэша, в том числе «≥ min_train_years полных лет в
    обучающем окне» для роли external_test.
    """
    from mayak.data.splits import EXTERNAL_MIN_TRAIN_YEARS, ROLE_EXTERNAL, \
        min_hours_for_train_years
    years = EXTERNAL_MIN_TRAIN_YEARS if min_train_years is None else int(min_train_years)
    need_hours = min_hours_for_train_years(years)
    os.makedirs(os.path.join(out_dir, "stations"), exist_ok=True)
    rows, report = [], []
    for st in stations.itertuples(index=False):
        sid = st.id
        rec = dict(id=sid, status="excluded", reason="", n_hours=0, T_valid=0.0,
                   P_valid=0.0, RH_valid=0.0, report_every=0, n_files=0)
        files = station_files(raw_dir, sid)
        rec["n_files"] = len(files)
        try:
            if not files:
                raise ValueError("нет скачанных файлов")
            series = hourly_station(read_ghcnh(files), tol_minutes=tol_minutes)
            v = series.valid
            rec.update(n_hours=series.n, T_valid=float(v[:, 0].mean()),
                       P_valid=float(v[:, 1].mean()), RH_valid=float(v[:, 2].mean()),
                       report_every=report_step(v[:, 0]))
            if series.n < need_hours:
                raise ValueError(f"ряд {series.n} ч < {need_hours} ч "
                                 f"(≥ {years} лет в обучающем окне)")
            if rec["T_valid"] < min_valid_frac_T:
                raise ValueError(f"валидной T {rec['T_valid']:.1%} < {min_valid_frac_T:.0%}")
            h = dem(st.lat, st.lon)
            if h is None or not np.isfinite(h):
                raise ValueError("нет высоты из ЦМР в точке станции")
        except ValueError as e:
            rec["reason"] = str(e)
            report.append(rec)
            continue
        np.savez_compressed(os.path.join(out_dir, "stations", f"{sid}.npz"),
                            **series.to_npz_dict())
        st_elev = st.elev if np.isfinite(st.elev) else ""
        rows.append(dict(id=sid, lat=round(float(st.lat), 4), lon=round(float(st.lon), 4),
                         elev=round(float(h), 1), station_elev=st_elev,
                         dem_elev=round(float(h), 1), koppen=koppen(st.lat, st.lon),
                         split=ROLE_EXTERNAL, report_every=rec["report_every"],
                         name=getattr(st, "name", "")))
        rec["status"] = "included"
        report.append(rec)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "selection_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]) if report else ["id"])
        w.writeheader()
        w.writerows(report)
    return rows, report


__all__ = ["build_external_dataset", "station_files", "ELEMENTS", "GHCNH_PARSER_VERSION",
           "HourlySeries", "IGNORED_ELEMENTS",
           "QUALITY_POLICY_VERSION", "hourly_station", "nearest_to_hour", "quality_flagged",
           "quality_flags", "read_ghcnh", "read_station_list", "report_class", "report_step"]
