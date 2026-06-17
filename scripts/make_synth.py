"""Генератор синтетического метео-датасета.
Запуск:
    python scripts/make_synth.py --out data --n-stations 40 --years 2 --seed 1
"""
import argparse, csv, math, os
import numpy as np

def koppen_for(lat):
    a = abs(lat)
    if a < 15:  return "Af"
    if a < 30:  return "BWh"
    if a < 45:  return "Cfb"
    if a < 60:  return "Dfb"
    return "ET"

def make_station(rng, lat, lon, elev, n_hours, t0_doy, t0_hour):
    h = np.arange(n_hours, dtype=np.float64)
    doy = (t0_doy + (t0_hour + h) / 24.0) % 365.24
    hour = (t0_hour + h) % 24.0
    phi = math.radians(lat)

    annual_mean = 27.0 - 0.55 * abs(lat) - 0.0065 * elev
    annual_amp  = 2.0 + 0.35 * abs(lat)
    season_phase = 0.0 if lat >= 0 else math.pi
    seasonal = annual_amp * np.cos(2 * math.pi * (doy - 200) / 365.24 + season_phase)

    decl = -23.44 * np.cos(2 * math.pi * (doy + 10) / 365.24)
    cos_zen = (np.sin(phi) * np.sin(np.radians(decl))
               + np.cos(phi) * np.cos(np.radians(decl)) * np.cos(np.radians(15 * (hour - 12))))
    daily_amp = 3.5 + 4.0 * np.clip(np.cos(phi), 0.1, 1.0)
    diurnal = daily_amp * np.clip(cos_zen, -0.3, 1.0)

    rho = math.exp(-1.0 / 60.0)
    innov = rng.standard_normal(n_hours) * 2.4
    synoptic = np.empty(n_hours)
    synoptic[0] = innov[0]
    for k in range(1, n_hours):
        synoptic[k] = rho * synoptic[k - 1] + math.sqrt(1 - rho * rho) * innov[k]

    micro = rng.standard_normal() * 1.2
    T = annual_mean + seasonal + diurnal + synoptic + micro

    P0 = 1013.25 * (1 - elev / 44330.0) ** 5.255
    P = P0 + 0.6 * synoptic + rng.standard_normal(n_hours) * 1.5

    RH_base = 70 - 0.25 * abs(lat) + 8 * np.cos(2 * math.pi * (doy - 30) / 365.24)
    RH = RH_base - 1.8 * diurnal + rng.standard_normal(n_hours) * 4.0
    RH = np.clip(RH, 3, 100)

    valid = np.ones(n_hours, dtype=np.uint8)
    return (T.astype(np.float32), P.astype(np.float32), RH.astype(np.float32), valid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-stations", type=int, default=40)
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "stations"), exist_ok=True)
    n_hours = int(args.years * 365.24 * 24)

    rows = []
    for i in range(args.n_stations):
        lat = float(rng.uniform(-70, 75))
        lon = float(rng.uniform(-180, 180))
        elev = float(max(0.0, rng.uniform(-30, 2500) if rng.random() > 0.2 else rng.uniform(0, 200)))
        t0_doy = float(rng.uniform(0, 365))
        t0_hour = float(rng.integers(0, 24))
        sid = f"S{i:03d}"
        T, P, RH, valid = make_station(rng, lat, lon, elev, n_hours, t0_doy, t0_hour)
        np.savez_compressed(os.path.join(args.out, "stations", f"{sid}.npz"),
                            T=T, P=P, RH=RH, valid=valid,
                            t0_doy=np.float32(t0_doy), t0_hour=np.float32(t0_hour))
        rows.append(dict(id=sid, lat=round(lat, 4), lon=round(lon, 4),
                         elev=round(elev, 1), koppen=koppen_for(lat)))

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen"])
        w.writeheader(); w.writerows(rows)
    print(f"Готово: {args.n_stations} станций × {n_hours} ч в {args.out}/stations/, манифест {args.out}/manifest.csv")

if __name__ == "__main__":
    main()