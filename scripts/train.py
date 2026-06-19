"""Обучение МАЯК в два этапа (A: L=0, B: полный куррикулум).

Запуск отладка:
    python scripts/train.py --steps-a 200 --steps-b 1000 --batch 32 \
        --windows 2000 --workers 0 --accelerator cpu

Запуск полный:
    python scripts/train.py --steps-a 10000 --steps-b 200000 --batch 256 \
        --windows 200000 --workers 8 --accelerator gpu --precision bf16-mixed
"""
import argparse
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from mayak.data.datamodule import MayakData
from mayak.evaluate_ext import coldstart_L0_check
from mayak.lit import LitMayak


def run_stage(lit, manifest, curriculum, max_steps, args, tag, val_L=672):
    dm = MayakData(manifest=manifest, curriculum=curriculum,
                   batch_size=args.batch, windows_per_epoch=args.windows,
                   num_workers=args.workers, val_L=val_L)
    lit.hparams.total_steps = max_steps
    ckpt = ModelCheckpoint(dirpath=f"runs/{tag}", monitor="val/loss",
                           save_top_k=1, mode="min", filename="best")
    trainer = L.Trainer(
        max_steps=max_steps, accelerator=args.accelerator, devices=1,
        precision=args.precision, gradient_clip_val=1.0,
        val_check_interval=args.val_every, check_val_every_n_epoch=None,
        limit_val_batches=args.val_batches, logger=CSVLogger("runs", name=tag),
        log_every_n_steps=20, callbacks=[ckpt, LearningRateMonitor("step"), EarlyStopping(monitor="val/loss", patience=5, mode="min")],
        enable_progress_bar=True)
    trainer.fit(lit, datamodule=dm)
    return ckpt.best_model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--steps-a", type=int, default=10_000)
    ap.add_argument("--steps-b", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--windows", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    args = ap.parse_args()

    # lit = LitMayak()
    # print(">>> Этап A: только L=0 (стабилизация климат-поля)")
    # best_a = run_stage(lit, args.manifest, "L0", args.steps_a, args,
    #                    tag="stageA", val_L=0)
    # from mayak import baselines as BL
    # from mayak.eval import stage_a_field_check, pure_field_check, l0_decompose
    # from torch.utils.data import DataLoader
    # from mayak.eval import EvalSet
    # clims = BL.fit_climatologies(args.manifest)
    # lit = LitMayak.load_from_checkpoint(best_a)
    lit = LitMayak.load_from_checkpoint('runs/stageA/best.ckpt', map_location="cpu")
    # stage_a_field_check(lit.model, clims, args.manifest, station_split="unseen_val")
    # stage_a_field_check(lit.model, clims, args.manifest, station_split="train")
    # pure_field_check(lit.model, clims, args.manifest, station_split="unseen_val")
    # pure_field_check(lit.model, clims, args.manifest, station_split="train")
    # l0_decompose(lit.model, clims, args.manifest, station_split="unseen_val")
    # l0_decompose(lit.model, clims, args.manifest, station_split="train")

    # w = lit.model.loc.W.norm(dim=0)
    # print("W норма  макс:", float(w.max()), " среднее:", float(w.mean()))

    # ds = EvalSet(clims, station_splits=("unseen_val",), manifest=args.manifest, time_key="test", L=0)
    # zs = [lit.model(b)["z"].abs().mean().item() for b in DataLoader(ds, batch_size=128)]
    # print("|z| при L=0 (unseen):", sum(zs)/len(zs)) 

    print(">>> Этап B: полный куррикулум")
    best = run_stage(lit, args.manifest, "full", args.steps_b, args, tag="stageB", val_L=672)
    print("Лучшая модель этапа B:", best)


if __name__ == "__main__":
    main()