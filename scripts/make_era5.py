"""Сборка станций обучающего набора из скачанных рядов ERA5.

Читает таблицу точек, параметры скачивания и файлы точек, приводит ряды к полной
почасовой сетке UTC без интерполяции и пишет файл на каждую станцию и манифест.
Высота станции берётся из ответа API: это высота цифровой модели рельефа, к
которой API привёл условия в точке. Зона Кёппена берётся из таблицы точек.
Рядом с манифестом пишется запись о том, каким кодом собраны файлы станций:
сборка кэша откажется работать, если этот код потом изменится, а набор не
пересоберут.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

from mayak.data.era5 import (CHANNELS, MODEL, haversine_km, parse_location, raw_matches,
                             raw_path, read_raw)
from mayak.data.store import command_line, write_source_build
from mayak.timeaxis import from_utc_hour, to_hourly_grid

MANIFEST_FIELDS = ("id", "lat", "lon", "elev", "koppen", "cell_lat", "cell_lon")
BUILD_CODE = {"mayak.data.era5": None, "mayak.timeaxis": None, "scripts/make_era5.py": None}


def build_station(record):
    """Почасовой ряд станции и строка манифеста из скачанной точки.

    Args:
        record: запись скачанной точки.

    Returns:
        Кортеж из начала ряда в часах от эпохи Unix, словаря рядов T, P, RH на
        полной почасовой сетке с NaN в пропусках и разобранного ответа API.

    Raises:
        ValueError: ряд нельзя привести к почасовой сетке или нет высоты.
    """
    loc = parse_location(record["response"])
    if loc["elevation"] is None:
        raise ValueError("в ответе API нет высоты точки")
    t0, cols = to_hourly_grid(from_utc_hour(loc["hours"]), {ch: loc[ch] for ch, _ in CHANNELS})
    return t0, cols, loc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data", help="корень набора данных")
    ap.add_argument("--points", default=None, help="по умолчанию <data>/points.csv")
    ap.add_argument("--out", default=None, help="куда писать станции и манифест; по "
                                                "умолчанию --data")
    ap.add_argument("--allow-missing", action="store_true",
                    help="собрать набор из того, что скачано, вместо ошибки")
    args = ap.parse_args()

    points_path = args.points or os.path.join(args.data, "points.csv")
    out_dir = args.out or args.data
    raw_dir = os.path.join(args.data, "era5")
    meta_path = os.path.join(args.data, "fetch_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"нет {meta_path}: сначала скачайте ряды "
                 f"(python scripts/fetch_era5.py --out {args.data})")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    signature = meta["request"]
    if signature.get("model") != MODEL:
        sys.exit(f"{meta_path}: модель {signature.get('model')}, ожидалась {MODEL}")
    with open(points_path, newline="") as f:
        points = list(csv.DictReader(f))

    stations_dir = os.path.join(out_dir, "stations")
    os.makedirs(stations_dir, exist_ok=True)
    rows, missing, broken = [], [], []
    for p in points:
        path = raw_path(raw_dir, p["id"])
        if not os.path.exists(path):
            missing.append(p["id"])
            continue
        try:
            record = read_raw(path)
            why = raw_matches(record, p, signature)
            if why:
                raise ValueError(why)
            t0, cols, loc = build_station(record)
        except (OSError, ValueError, KeyError) as e:
            broken.append(f"{p['id']}: {e}")
            continue
        valid = np.isfinite(np.stack([cols[ch] for ch, _ in CHANNELS], axis=-1))
        np.savez_compressed(os.path.join(stations_dir, f"{p['id']}.npz"),
                            **{ch: cols[ch] for ch, _ in CHANNELS},
                            valid=valid.astype(np.uint8), t0_utc_h=np.int64(t0))
        rows.append(dict(id=p["id"], lat=round(float(p["lat"]), 4),
                         lon=round(float(p["lon"]), 4), elev=round(loc["elevation"], 1),
                         koppen=p["koppen"], cell_lat=round(loc["cell_lat"], 4),
                         cell_lon=round(loc["cell_lon"], 4)))

    if broken:
        sys.exit(f"{len(broken)} файлов точек не годятся: " + "; ".join(broken[:5])
                 + ". Перекачайте их: python scripts/fetch_era5.py --refetch")
    if missing and not args.allow_missing:
        sys.exit(f"не скачано {len(missing)} из {len(points)} точек "
                 f"({', '.join(missing[:5])}{', …' if len(missing) > 5 else ''}). "
                 "Запустите python scripts/fetch_era5.py ещё раз или добавьте --allow-missing.")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_source_build(out_dir, "make_era5",
                       command_line(["python", "scripts/make_era5.py", *sys.argv[1:]]), BUILD_CODE)

    if rows:
        shift = haversine_km(np.array([r["lat"] for r in rows]), np.array([r["lon"] for r in rows]),
                             np.array([r["cell_lat"] for r in rows]),
                             np.array([r["cell_lon"] for r in rows]))
        print(f"смещение до центра ячейки реанализа, км: медиана {np.median(shift):.1f}, "
              f"максимум {np.max(shift):.1f}")
    print(f"станций: {len(rows)} из {len(points)}" + (f", пропущено {len(missing)}"
                                                      if missing else ""))
    print(f"период: {signature['start_date']}..{signature['end_date']}, модель {MODEL}")
    print(f"манифест: {manifest_path}")
    print(f"далее: python scripts/make_splits.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
