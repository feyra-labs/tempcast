"""Внешний тест на наблюдениях реальной сети: атрибуты станций и сопоставление с внутренним.

* атрибуты станции для разрезов, специфичных для внешнего теста: разность «заявленная
  высота − высота из ЦМР» и расстояние до ближайшей обучающей точки;
* transfer_table — прямое сопоставление «внутренний тест против внешнего» по
  одинаковым лидам на общих зонах, с интервалом для разности: станции обоих наборов
  ресэмплируются независимо (блочный бутстрап по станциям, как везде в проекте).
"""
from __future__ import annotations

import numpy as np

from mayak.data.era5 import haversine_km
from mayak.data.splits import ROLE_TRAIN
from mayak.metrics import METRICS, quantile_ci
from mayak.zones import koppen_group, normalize_zone

ELEV_GAP_BINS = ((0.0, 50.0, "|Δh| <50 м"), (50.0, 150.0, "|Δh| 50-150 м"),
                 (150.0, 300.0, "|Δh| 150-300 м"), (300.0, np.inf, "|Δh| ≥300 м"))
TRAIN_DISTANCE_BINS = ((0.0, 25.0, "<25 км"), (25.0, 100.0, "25-100 км"),
                       (100.0, 300.0, "100-300 км"), (300.0, np.inf, "≥300 км"))
TRAIN_DISTANCE_ORDER = tuple(name for _lo, _hi, name in TRAIN_DISTANCE_BINS)
NO_DATA = "нет данных"
TRANSFER_METRICS = ("Skill", "MAE", "CRPS", "PICP90")
TRANSFER_LEADS = (1, 6, 24, 72, 168)


def elev_gap(s):
    st = s.get("station_elev")
    dem = s.get("dem_elev")
    if st is None or dem is None:
        return None
    return float(st) - float(dem)


def elev_gap_label(gap):
    if gap is None or not np.isfinite(gap):
        return NO_DATA
    a = abs(float(gap))
    for lo, hi, name in ELEV_GAP_BINS:
        if lo <= a < hi:
            return name
    return NO_DATA


def nearest_train_km(store, external_store):
    """Расстояние от каждой внешней станции до ближайшей обучающей точки.

    Args:
        store: основной набор; обучающие точки - его станции с ролью обучения.
        external_store: набор внешнего теста.

    Returns:
        Словарь из id внешней станции в расстояние по поверхности Земли, км. Пустой,
        если в основном наборе нет обучающих станций.
    """
    train = store.by_role(ROLE_TRAIN)
    if not train:
        return {}
    lat = np.array([s["lat"] for s in train], np.float64)
    lon = np.array([s["lon"] for s in train], np.float64)
    return {sid: float(haversine_km(s["lat"], s["lon"], lat, lon).min())
            for sid, s in external_store.stations.items()}


def train_distance_label(km):
    """Бин расстояния до ближайшей обучающей точки.

    Args:
        km: расстояние, км; None, если оно не посчитано.

    Returns:
        Имя бина или пометка об отсутствии данных.
    """
    if km is None or not np.isfinite(km):
        return NO_DATA
    for lo, hi, name in TRAIN_DISTANCE_BINS:
        if lo <= float(km) < hi:
            return name
    return NO_DATA


def station_attributes(stations, train_km=None):
    """Атрибуты станций для разрезов внешнего теста.

    Args:
        stations: словарь из id станции в её запись.
        train_km: словарь из id станции в расстояние до ближайшей обучающей точки,
            км; None, если расстояния не посчитаны. Тогда метка расстояния у всех
            станций - пометка об отсутствии данных.

    Returns:
        Словарь из id станции в словарь: разность высот и её бин, расстояние до
        обучающей точки и его бин.
    """
    train_km = train_km or {}
    out = {}
    for sid, s in stations.items():
        g = elev_gap(s)
        km = train_km.get(sid)
        out[sid] = dict(elev_gap=g, elev_gap_label=elev_gap_label(g), train_km=km,
                        train_distance_label=train_distance_label(km))
    return out


def zone_keys(zones, level="group"):
    """Метки зон окон для сопоставления: крупная группа A-E либо полная зона."""
    if level == "group":
        return np.array([koppen_group(z) for z in zones], object)
    if level == "full":
        return np.array([normalize_zone(z) for z in zones], object)
    raise ValueError(f"level {level!r}: 'group' или 'full'")


def _n_stations(ev, sel):
    return ev.restrict(windows=sel).counts()["n_stations"]


def common_zones(ev_int, z_int, ev_ext, z_ext, min_stations=2):
    """Зоны, в которых не меньше min_stations станций и во внутреннем, и во внешнем наборе."""
    out = []
    for z in sorted(set(z_int.tolist()) & set(z_ext.tolist())):
        if (_n_stations(ev_int, z_int == z) >= min_stations
                and _n_stations(ev_ext, z_ext == z) >= min_stations):
            out.append(z)
    return out


def transfer_table(ev_int, zones_int, ev_ext, zones_ext, leads=TRANSFER_LEADS, level="group",
                   min_stations=2, n_boot=1000, seed=0, ci_level=0.90):
    """Внутренний тест против внешнего на одинаковых лидах и общих зонах."""
    zi, ze = zone_keys(zones_int, level), zone_keys(zones_ext, level)
    zones = common_zones(ev_int, zi, ev_ext, ze, min_stations)
    rows = {}
    if not zones:
        return dict(zones=[], level=level, rows=rows)
    si, se = np.isin(zi, zones), np.isin(ze, zones)
    for h in leads:
        a = ev_int.restrict(windows=si, leads=[h])
        b = ev_ext.restrict(windows=se, leads=[h])
        sa, sb = a.summary(), b.summary()
        diff = {agg: {m: sb[agg][m] - sa[agg][m] for m in METRICS} for agg in ("pooled", "macro")}
        row = dict(internal=sa, external=sb, diff=diff)
        if n_boot:
            ba = a.bootstrap_samples(n_boot=n_boot, seed=seed)
            bb = b.bootstrap_samples(n_boot=n_boot, seed=seed + 1)
            if ba is not None and bb is not None:
                row["diff_ci"] = {agg: quantile_ci({m: bb[i][m] - ba[i][m] for m in METRICS},
                                                   ci_level)
                                  for i, agg in enumerate(("pooled", "macro"))}
        rows[int(h)] = row
    return dict(zones=zones, level=level, rows=rows)


def print_transfer(tbl, metrics=TRANSFER_METRICS, title=""):
    if title:
        print(title)
    if not tbl["zones"]:
        print("  (нет общих зон с достаточным числом станций в обоих наборах)")
        return
    print(f"  общие зоны ({tbl['level']}): {', '.join(tbl['zones'])}")
    for agg, name in (("pooled", "пул"), ("macro", "макро")):
        print(f"  --- {name} ---")
        head = f"{'лид,ч':>6}" + "".join(f" {m + ' внутр':>13} {m + ' внеш':>12} {'Δ':>9}"
                                         f" {'ДИ Δ':>19}" for m in metrics)
        print(head)
        for h, r in tbl["rows"].items():
            line = f"{h:>6}"
            for m in metrics:
                pct = m in ("Skill", "PICP90")
                f = (lambda v: f"{v:+.1%}") if pct else (lambda v: f"{v:+.2f}")
                ci = r.get("diff_ci", {}).get(agg, {}).get(m, (np.nan, np.nan))
                ci_s = f"[{f(ci[0])};{f(ci[1])}]" if np.isfinite(ci[0]) else "—"
                line += (f" {f(r['internal'][agg][m]):>13} {f(r['external'][agg][m]):>12}"
                         f" {f(r['diff'][agg][m]):>9} {ci_s:>19}")
            print(line)


__all__ = ["ELEV_GAP_BINS", "NO_DATA", "TRAIN_DISTANCE_BINS", "TRAIN_DISTANCE_ORDER",
           "common_zones", "elev_gap", "elev_gap_label", "nearest_train_km", "print_transfer",
           "station_attributes", "train_distance_label", "transfer_table", "zone_keys"]
