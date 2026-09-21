"""Диагностика климат-поля после этапа A.

Работает только на валидационном окне: на валидационных станциях (обобщение поля)
и на обучающих (для сравнения — это окно в обучении не участвовало).

Запуск:
    python scripts/diagnose_stage_a.py --ckpt runs/mayak/stageA/best.ckpt
"""
import argparse
import logging

from torch.utils.data import DataLoader

from mayak.data.splits import ROLE_TRAIN, ROLE_VAL
from mayak.data.store import get_store
from mayak.evaluate import EvalSet, l0_decompose, pure_field_check, stage_a_field_check
from mayak.leakage import run_checklist

TIME_KEY = "val"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mayak.lit import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="чекпойнт этапа A (runs/mayak/stageA/best.ckpt)")
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()

    store = get_store(args.manifest)
    clims = store.clims()
    ds_z = EvalSet(clims, station_splits=(ROLE_VAL,), manifest=args.manifest,
                   time_key=TIME_KEY, L=0)
    run_checklist(store, datasets=[ds_z], checkpoints=[args.ckpt])

    model = load_model(args.ckpt)
    for role in (ROLE_VAL, ROLE_TRAIN):
        stage_a_field_check(model, clims, args.manifest, station_split=role, time_key=TIME_KEY)
        pure_field_check(model, clims, args.manifest, station_split=role, time_key=TIME_KEY)
        l0_decompose(model, clims, args.manifest, station_split=role, time_key=TIME_KEY)

    w = model.loc.W.norm(dim=0)
    print("W норма  макс:", float(w.max()), " среднее:", float(w.mean()))
    zs = [model(b)["z"].abs().mean().item() for b in DataLoader(ds_z, batch_size=128)]
    print(f"|z| при L=0 ({ROLE_VAL}, окно {TIME_KEY}):", sum(zs) / max(len(zs), 1))


if __name__ == "__main__":
    main()
