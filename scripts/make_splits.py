"""Стратифицированное разбиение станций на роли train, unseen_val и unseen_test.

Страта - полная зона Кёппена и широтный пояс. В каждой страте размера не меньше
--min-stratum есть станции всех трёх ролей; округление долей не обнуляет мелкие
страты; разбиение воспроизводится по --seed и не зависит от порядка строк.
Проставляет колонку split в манифесте, остальные колонки сохраняются. Рядом с
манифестом пишется отчёт о разбиении: запрошенное и назначенное число станций по
ролям, страты, длины рядов и раскладка ряда типичной длины.
"""
import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter

import numpy as np

from mayak.data.splits import (DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC, MIN_TRAIN_YEARS,
                               ROLE_EXTERNAL, ROLES, TIME_LAYOUT, assign_roles,
                               layout_fingerprint, min_hours_for_train_years, strata_report,
                               time_layout)
from mayak.data.store import read_manifest, source_path

REPORT_NAME = "splits_report.json"


def series_hours(manifest, rows):
    """Длина ряда каждой станции по её исходному файлу.

    Args:
        manifest: путь к манифесту; файлы станций лежат в каталоге stations рядом с ним.
        rows: строки манифеста.

    Returns:
        Словарь: id станции и длина её ряда, ч. Станции без файла отсутствуют.
    """
    out = {}
    for r in rows:
        path = source_path(manifest, r["id"])
        if os.path.exists(path):
            with np.load(path) as d:
                out[r["id"]] = int(d["T"].shape[0])
    return out


def check_lengths(rows, hours, years):
    """Станции, у которых обучение по раскладке короче заданного числа лет.

    Args:
        rows: строки манифеста без станций внешнего теста.
        hours: длины рядов по id станции.
        years: сколько лет должно занимать обучение.

    Returns:
        Пара: нужная длина ряда, ч, и список (id, длина) станций короче неё, включая
        станции без исходного файла с длиной ноль.
    """
    need = min_hours_for_train_years(years)
    short = [(r["id"], hours.get(r["id"], 0)) for r in rows if hours.get(r["id"], 0) < need]
    return need, short


def layout_summary(n_hours):
    """Раскладка ряда заданной длины с длинами окон в годах, для отчёта.

    Args:
        n_hours: длина ряда, ч.

    Returns:
        Словарь для JSON.
    """
    lay = time_layout(n_hours)
    hpy = TIME_LAYOUT["hours_per_year"]
    lo, hi = lay.span("train")
    return dict(lay.to_dict(), train_years=round((hi - lo) / hpy, 2),
                test_years=round((lay.span("test")[1] - lay.span("test")[0]) / hpy, 2),
                block_hours=sorted({hi - lo for k in ("val", "calib")
                                    for lo, hi in lay.blocks[k]}))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                    help="доля тестовых станций")
    ap.add_argument("--n-test", type=int, default=None, help="точное число (вместо --test-frac)")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help="доля валидационных среди станций вне теста")
    ap.add_argument("--n-val", type=int, default=None, help="точное число (вместо --val-frac)")
    ap.add_argument("--min-stratum", type=int, default=3,
                    help="страты от этого размера обязаны быть во всех ролях")
    ap.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS,
                    help="сколько лет должно занимать обучение у каждой станции; "
                         "0 - не проверять длину рядов")
    ap.add_argument("--report", default=None,
                    help=f"куда писать отчёт; по умолчанию {REPORT_NAME} рядом с манифестом")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    internal = [r for r in rows if str(r.get("split") or "") != ROLE_EXTERNAL]
    hours = series_hours(args.manifest, internal)
    need = None
    if args.min_train_years > 0:
        need, short = check_lengths(internal, hours, args.min_train_years)
        if short:
            listed = ", ".join(f"{sid} ({n} ч)" for sid, n in short[:10])
            sys.exit(f"у {len(short)} станций ряд короче {need} ч, нужных для "
                     f"{args.min_train_years} лет обучения: {listed}. Скачайте более "
                     f"длинный период (fetch_era5.py --years) или задайте "
                     f"--min-train-years 0")

    info = {}
    roles = assign_roles(rows, n_test=args.n_test, test_frac=args.test_frac,
                         val_frac=args.val_frac, n_val=args.n_val, seed=args.seed,
                         min_stratum=args.min_stratum, info=info)
    fields = list(rows[0].keys()) + ([] if "split" in rows[0] else ["split"])
    for r in rows:
        r["split"] = roles[r["id"]]
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = dict(Counter(r["split"] for r in rows))
    strata = strata_report(rows, roles)
    lengths = sorted(hours.values())
    report = dict(
        manifest=os.path.abspath(args.manifest), seed=args.seed,
        params=dict(test_frac=args.test_frac, n_test=args.n_test, val_frac=args.val_frac,
                    n_val=args.n_val, min_stratum=args.min_stratum,
                    min_train_years=args.min_train_years),
        requested=info["requested"], assigned=counts,
        strata=[dict(koppen=kop, band=band, **{r: cnt.get(r, 0) for r in ROLES})
                for (kop, band), cnt in strata.items()],
        time_layout=dict(TIME_LAYOUT),
        layout_code=layout_fingerprint(),
        series=dict(stations=len(lengths), required_hours=need,
                    min_hours=lengths[0] if lengths else None,
                    median_hours=int(np.median(lengths)) if lengths else None,
                    max_hours=lengths[-1] if lengths else None))
    if lengths:
        report["layout_of_median"] = layout_summary(int(np.median(lengths)))
    path = args.report or os.path.join(os.path.dirname(os.path.abspath(args.manifest)),
                                       REPORT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("роли, запрошено:", info["requested"])
    print("роли, назначено:", counts)
    print(f"{'страта':>16} " + " ".join(f"{r:>12}" for r in ROLES))
    for (kop, band), cnt in strata.items():
        print(f"{kop + '/' + band:>16} " + " ".join(f"{cnt.get(r, 0):>12}" for r in ROLES))
    if lengths:
        lay = report["layout_of_median"]
        print(f"ряд типичной длины {report['series']['median_hours']} ч: обучение "
              f"{lay['train_years']} года, тест {lay['test_years']} года, блоки валидации и "
              f"калибровки по {lay['block_hours']} ч")
    print(f"отчёт: {path}")


if __name__ == "__main__":
    main()
