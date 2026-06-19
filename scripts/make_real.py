import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


KG_MAP = {
     1: "Af",
     2: "Am",
     3: "Aw",
     4: "BWh",
     5: "BWk",
     6: "BSh",
     7: "BSk",
     8: "Csa",
     9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}


def get_koppen_reader(tif_path):
    ds = rasterio.open(tif_path)
    band = ds.read(1)

    def get_koppen(lat, lon):
        try:
            row, col = ds.index(lon, lat)

            if (
                row < 0
                or row >= band.shape[0]
                or col < 0
                or col >= band.shape[1]
            ):
                return "UNK"

            code = int(band[row, col])

            if code <= 0:
                return "UNK"

            return KG_MAP.get(code, "UNK")

        except Exception:
            return "UNK"

    return get_koppen


def compute_t0(ts):
    doy = (
        ts.dayofyear
        - 1
        + ts.hour / 24.0
        + ts.minute / 1440.0
        + ts.second / 86400.0
    )

    hour = (
        ts.hour
        + ts.minute / 60.0
        + ts.second / 3600.0
    )

    return np.float32(doy), np.float32(hour)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parquet-dir",
        default="datacheckpoints"
    )

    parser.add_argument(
        "--koppen",
        default="Beck_KG_V1_present_0p0083.tif"
    )

    parser.add_argument(
        "--out",
        default="real_dataset"
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    stations_dir = out_dir / "stations"

    stations_dir.mkdir(parents=True, exist_ok=True)

    print("Loading parquet files...")

    parquet_files = sorted(
        glob.glob(str(Path(args.parquet_dir) / "batch_*.parquet"))
    )

    if not parquet_files:
        raise RuntimeError("Не найдено ни одного batch_*.parquet")

    dfs = []

    for fn in parquet_files:
        print("  ", Path(fn).name)
        dfs.append(pd.read_parquet(fn))

    df = pd.concat(dfs, ignore_index=True)

    print(f"Rows: {len(df):,}")

    df["time"] = pd.to_datetime(df["time"], utc=True)

    get_koppen = get_koppen_reader(args.koppen)

    manifest_rows = []

    grouped = df.groupby("point_id")

    print(f"Stations: {len(grouped)}")

    for idx, (point_id, g) in enumerate(grouped):

        g = g.sort_values("time")

        lat = float(g["latitude"].iloc[0])
        lon = float(g["longitude"].iloc[0])
        elev = float(g["elevation_m"].iloc[0])

        T = g["temperature_2m"].astype(np.float32).to_numpy()

        P = g["surface_pressure"].astype(np.float32).to_numpy()

        RH = g["relative_humidity_2m"].astype(np.float32).to_numpy()

        valid = (
            np.isfinite(T)
            & np.isfinite(P)
            & np.isfinite(RH)
        ).astype(np.uint8)

        first_ts = g["time"].iloc[0]

        t0_doy, t0_hour = compute_t0(first_ts)

        sid = str(point_id)

        np.savez_compressed(
            stations_dir / f"{sid}.npz",
            T=T,
            P=P,
            RH=RH,
            valid=valid,
            t0_doy=t0_doy,
            t0_hour=t0_hour,
        )

        koppen = get_koppen(lat, lon)

        manifest_rows.append(
            {
                "id": sid,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "elev": round(elev, 1),
                "koppen": koppen[0] if koppen != "UNK" else "UNK",
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"{idx + 1} stations processed")

    manifest_path = out_dir / "manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "lat",
                "lon",
                "elev",
                "koppen",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Done")
    print(f"Stations: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"NPZ files: {stations_dir}")


if __name__ == "__main__":
    main()