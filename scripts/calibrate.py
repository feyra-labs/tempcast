import argparse
import numpy as np

from mayak.constants import H
from mayak import baselines as BL
from mayak.evaluate import EvalSet, gather
from mayak.metrics import coverage, fit_conformal_shift

LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]


def lead_bin_index(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1


def fit_conformal(model, clims, manifest="data/manifest.csv"):
    ds = EvalSet(clims, manifest=manifest, time_key="calib", every_hours=24)
    D = gather(model, ds)
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
    lit = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu")

    clims = BL.fit_climatologies(args.manifest)
    shift = fit_conformal(lit.model, clims, args.manifest)
    np.save(args.out, shift)
    print("Таблица поправок (бины лидов × квантили), °C:")
    print(np.round(shift, 3))

    ds = EvalSet(clims, manifest=args.manifest, time_key="test", every_hours=72)
    D = gather(lit.model, ds)
    before = coverage(D["y"], D["q"], D["y_mask"])
    after = coverage(D["y"], apply_conformal(D["q"], shift), D["y_mask"])
    print(f"PICP-90 на тесте: до {before:.1%}  →  после {after:.1%}  (цель 86–94%)")


if __name__ == "__main__":
    main()
