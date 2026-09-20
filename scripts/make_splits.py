"""Стратифицированное разбиение станций на роли train / unseen_val / unseen_test.

Страта — (полная зона Кёппена, широтный пояс). В каждой страте размера
≥ --min-stratum есть станции всех трёх ролей; округление долей не обнуляет
мелкие страты; разбиение воспроизводится по --seed и не зависит от порядка строк.
Проставляет колонку split в manifest.csv, остальные колонки сохраняются.

Запуск:
    python scripts/make_splits.py --manifest data/manifest.csv \\
        --n-test 8 --val-frac 0.1 --seed 0
"""
import argparse
import csv
import logging
from collections import Counter

from mayak.data.splits import ROLES, assign_roles, strata_report
from mayak.data.store import read_manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="доля валидационных среди станций вне теста")
    ap.add_argument("--n-val", type=int, default=None, help="точное число (вместо --val-frac)")
    ap.add_argument("--min-stratum", type=int, default=3,
                    help="страты от этого размера обязаны быть во всех ролях")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    roles = assign_roles(rows, n_test=args.n_test, val_frac=args.val_frac, n_val=args.n_val,
                         seed=args.seed, min_stratum=args.min_stratum)
    fields = list(rows[0].keys()) + ([] if "split" in rows[0] else ["split"])
    for r in rows:
        r["split"] = roles[r["id"]]
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("Роли:", dict(Counter(r["split"] for r in rows)))
    print(f"{'страта':>16} " + " ".join(f"{r:>12}" for r in ROLES))
    for (kop, band), cnt in strata_report(rows, roles).items():
        print(f"{kop + '/' + band:>16} " + " ".join(f"{cnt.get(r, 0):>12}" for r in ROLES))


if __name__ == "__main__":
    main()
