"""Выбор обучающих точек на суше: решётка Фибоначчи и гарантия покрытия зон Кёппена.

Выход - таблица id, lat, lon, koppen. Один и тот же сид и те же параметры дают
одну и ту же таблицу.
"""
import argparse
import csv
import logging
import os
from collections import Counter

from mayak.data.points import (DEFAULT_GRID_STEP, DEFAULT_MIN_DIST_KM, DEFAULT_MIN_PER_ZONE,
                               DEFAULT_N_POINTS, DEFAULT_SEED, POINTS_FIELDS, ZoneGrid,
                               min_pairwise_km, select_points)
from mayak.zones import KG_TIF_CODE


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--koppen", default="Beck_KG_V1_present_0p0083.tif",
                    help="растр зон Кёппена с кодами 1..30")
    ap.add_argument("--out", default="data/points.csv")
    ap.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, help="сколько точек всего")
    ap.add_argument("--min-per-zone", type=int, default=DEFAULT_MIN_PER_ZONE,
                    help="минимум точек в каждой зоне, где есть суша")
    ap.add_argument("--min-dist-km", type=float, default=DEFAULT_MIN_DIST_KM,
                    help="минимальное расстояние между точками, км")
    ap.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP,
                    help="шаг сетки зон, градусы; точки ставятся в центры её ячеек")
    ap.add_argument("--include-antarctica", action="store_true",
                    help="брать сушу южнее 60 градусов южной широты")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    grid = ZoneGrid.from_raster(args.koppen, args.grid_step)
    sel = select_points(grid, args.n_points, args.min_per_zone, args.min_dist_km,
                        args.include_antarctica, args.seed)
    rows = sel.rows()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINTS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_source = Counter(sel.source)
    print(f"точек: {len(rows)} (узлов решётки на сфере {sel.n_lattice}; из решётки "
          f"{by_source.get('lattice', 0)}, добрано ради зон {by_source.get('zone', 0)})")
    print(f"наименьшее расстояние между точками: "
          f"{min_pairwise_km(sel.lat, sel.lon):.0f} км")
    counts = sel.zone_counts()
    print("точек по зонам: " + ", ".join(f"{KG_TIF_CODE[z]} {n}" for z, n in counts.items()))
    print(f"таблица: {args.out}")
    print(f"далее: python scripts/fetch_era5.py --points {args.out} --out "
          f"{os.path.dirname(args.out) or '.'}")


if __name__ == "__main__":
    main()
