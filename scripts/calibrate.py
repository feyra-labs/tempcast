"""Сплит-конформная поправка квантилей.

Подгоняется строго на валидационных станциях в калибровочном окне. Рядом с
таблицей сохраняются метаданные (<out>.meta.json): по ним чек-лист антиутечек
проверяет, на каких данных она построена. Тестовое окно здесь не используется,
эффект поправки на тесте печатает mayak/evaluate.py (--conformal).

Запуск:
    python scripts/calibrate.py --ckpt runs/stageB/best.ckpt --out runs/conformal.npy
"""
import argparse

import numpy as np

from mayak.constants import H
from mayak.data.splits import ROLE_VAL
from mayak.data.store import get_store
from mayak.evaluate import EvalSet, gather
from mayak.leakage import CONFORMAL_TIME_KEY, conformal_record, run_checklist, save_conformal
from mayak.metrics import coverage, fit_conformal_shift

LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]


def lead_bin_index(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1


def calibration_set(clims, manifest="data/manifest.csv"):
    return EvalSet(clims, station_splits=(ROLE_VAL,), manifest=manifest,
                   time_key=CONFORMAL_TIME_KEY, every_hours=24)


def fit_conformal(model, clims, manifest="data/manifest.csv"):
    D = gather(model, calibration_set(clims, manifest))
    return fit_conformal_shift(D["y"], D["q"], D["y_mask"], LEAD_BINS)


def apply_conformal(q, shift):
    q = np.array(q, np.float32, copy=True)
    for h in range(H):
        bi = lead_bin_index(h + 1)
        q[..., h, :] += shift[bi]
    q = np.maximum.accumulate(q, axis=-1)
    return q


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mayak.lit import LitMayak
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/conformal.npy")
    args = ap.parse_args()

    store = get_store(args.manifest)
    clims = store.clims()
    ds = calibration_set(clims, args.manifest)
    run_checklist(store, datasets=[ds], checkpoints=[args.ckpt])

    lit = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu")
    D = gather(lit.model, ds)
    shift = fit_conformal_shift(D["y"], D["q"], D["y_mask"], LEAD_BINS)
    save_conformal(args.out, shift, conformal_record(ds, checkpoint=args.ckpt))
    print("Таблица поправок (бины лидов × квантили), °C:")
    print(np.round(shift, 3))

    before = coverage(D["y"], D["q"], D["y_mask"])
    after = coverage(D["y"], apply_conformal(D["q"], shift), D["y_mask"])
    print(f"PICP-90 на калибровочном окне (в выборке): до {before:.1%}  →  после {after:.1%}")


if __name__ == "__main__":
    main()
