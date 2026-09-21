"""Сплит-конформная поправка квантилей.

Подгоняется строго на валидационных станциях в калибровочном окне. Рядом с
таблицей сохраняются метаданные (<out>.meta.json): по ним чек-лист антиутечек
проверяет, на каких данных она построена. Тестовое окно здесь не используется,
эффект поправки на тесте печатает mayak/evaluate.py (--conformal).

Запуск:
    python scripts/calibrate.py --ckpt runs/mayak/stageB/best.ckpt --out runs/conformal.npy
"""
import argparse

import numpy as np

from mayak.data.splits import ROLE_VAL
from mayak.data.store import get_store
from mayak.evaluate import EvalSet, gather
from mayak.leakage import CONFORMAL_TIME_KEY, conformal_record, run_checklist, save_conformal
from mayak.metrics import LEAD_BINS, apply_conformal, coverage, fit_conformal_shift


def calibration_set(clims, manifest="data/manifest.csv"):
    return EvalSet(clims, station_splits=(ROLE_VAL,), manifest=manifest,
                   time_key=CONFORMAL_TIME_KEY, every_hours=24)


def fit_conformal(model, clims, manifest="data/manifest.csv"):
    D = gather(model, calibration_set(clims, manifest))
    return fit_conformal_shift(D["y"], D["q"], D["y_mask"], LEAD_BINS)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mayak.lit import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/conformal.npy")
    args = ap.parse_args()

    store = get_store(args.manifest)
    clims = store.clims()
    ds = calibration_set(clims, args.manifest)
    run_checklist(store, datasets=[ds], checkpoints=[args.ckpt])

    model = load_model(args.ckpt)
    D = gather(model, ds)
    shift = fit_conformal_shift(D["y"], D["q"], D["y_mask"], LEAD_BINS)
    save_conformal(args.out, shift, conformal_record(ds, checkpoint=args.ckpt))
    print("Таблица поправок (бины лидов × квантили), °C:")
    print(np.round(shift, 3))

    before = coverage(D["y"], D["q"], D["y_mask"])
    after = coverage(D["y"], apply_conformal(D["q"], shift), D["y_mask"])
    print(f"PICP-90 на калибровочном окне (в выборке): до {before:.1%}  →  после {after:.1%}")


if __name__ == "__main__":
    main()
