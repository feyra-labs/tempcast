"""Внешний тест на наблюдениях реальной сети: атрибуты станций и сопоставление с внутренним.

* атрибуты станции для разрезов, специфичных для внешнего теста: шаг отчётности
  (по обучающему окну станции), разность «заявленная высота − высота из ЦМР»;
* transfer_table — прямое сопоставление «внутренний тест против внешнего» по
  одинаковым лидам на общих зонах, с интервалом для разности: станции обоих наборов
  ресэмплируются независимо (блочный бутстрап по станциям, как везде в проекте).
"""
from __future__ import annotations

import numpy as np

from mayak.data.ghcnh import report_class, report_step
from mayak.data.splits import time_layout
from mayak.metrics import METRICS, quantile_ci
from mayak.zones import koppen_group, normalize_zone

ELEV_GAP_BINS = ((0.0, 50.0, "|Δh| <50 м"), (50.0, 150.0, "|Δh| 50-150 м"),
                 (150.0, 300.0, "|Δh| 150-300 м"), (300.0, np.inf, "|Δh| ≥300 м"))
NO_DATA = "нет данных"
TRANSFER_METRICS = ("Skill", "MAE", "CRPS", "PICP90")
TRANSFER_LEADS = (1, 6, 24, 72, 168)


def station_report_class(s):
    """Класс шага отчётности станции по маске T в её обучающем окне."""
    lo, hi = time_layout(s["N"]).span("train")
    return report_class(report_step(s["mask"][lo:hi, 0]))


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


def station_attributes(stations):
    """{id: dict(report_class, elev_gap, elev_gap_label)} для набора станций."""
    out = {}
    for sid, s in stations.items():
        g = elev_gap(s)
        out[sid] = dict(report_class=station_report_class(s), elev_gap=g,
                        elev_gap_label=elev_gap_label(g))
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


__all__ = ["ELEV_GAP_BINS", "common_zones", "elev_gap", "elev_gap_label", "print_transfer",
           "station_attributes", "station_report_class", "transfer_table", "zone_keys"]
