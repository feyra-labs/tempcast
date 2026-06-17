import argparse
import numpy as np

from mayak.constants import QUANTILES, H
from mayak import baselines as BL
from mayak.evaluate import EvalSet, gather

LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]


def lead_bin_index(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1


def fit_conformal(model, clims, manifest="data/manifest.csv"):
    ds = EvalSet(clims, manifest=manifest, time_key="calib", every_hours=24)
    D = gather(model, ds)
    y, q = D["y"], D["q"]
    nb = len(LEAD_BINS)
    shift = np.zeros((nb, len(QUANTILES)), np.float32)
    for bi, (a, b) in enumerate(LEAD_BINS):
        sl = slice(a - 1, b)
        resid = (y[:, sl, None] - q[:, sl, :]).reshape(-1, len(QUANTILES))
        for qi, tau in enumerate(QUANTILES):
            shift[bi, qi] = np.quantile(resid[:, qi], tau)
    return shift


def apply_conformal(q, shift):
    q = np.array(q, np.float32, copy=True)
    for h in range(H):
        bi = lead_bin_index(h + 1)
        q[..., h, :] += shift[bi]
    q = np.maximum.accumulate(q, axis=-1)
    return q


def coverage90(y, q):
    lo, hi = q[..., 0], q[..., 6]
    return float(((y >= lo) & (y <= hi)).mean())


def main():
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
    before = coverage90(D["y"], D["q"])
    after = coverage90(D["y"], apply_conformal(D["q"], shift))
    print(f"PICP-90 на тесте: до {before:.1%}  →  после {after:.1%}  (цель 86–94%)")


if __name__ == "__main__":
    main()