"""Стратифицированное разбиение (зона Кёппена, широтный пояс)
станций на train / unseen_val / unseen_test.

Проставляет колонку split в manifest.csv.

Запуск:
    python scripts/make_splits.py --manifest data/manifest.csv \
        --n-test 8 --val-frac 0.1 --seed 0
"""
import argparse, csv
import numpy as np

def lat_band(lat):
    a = abs(float(lat))
    return "eq" if a < 23.5 else ("mid" if a < 50 else "pol")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    strata = {}
    for r in rows:
        key = (r["koppen"], lat_band(r["lat"]))
        strata.setdefault(key, []).append(r)

    for r in rows:
        r["split"] = "train"
    n_total = len(rows)
    for key, group in strata.items():
        rng.shuffle(group)
        k_test = max(0, round(args.n_test * len(group) / n_total))
        for r in group[:k_test]:
            r["split"] = "unseen_test"
        rest = group[k_test:]
        k_val = max(0, round(args.val_frac * len(rest)))
        for r in rest[:k_val]:
            r["split"] = "unseen_val"

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lat", "lon", "elev", "koppen", "split"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print("Сплиты:", dict(Counter(r["split"] for r in rows)))

if __name__ == "__main__":
    main()