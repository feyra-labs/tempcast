"""Обучение МАЯК в два этапа (A: L=0, B: полный куррикулум).

Скрипт только запускает этапы и печатает пути чекпойнтов. Чекпойнты выбираются
по val/loss на валидационных станциях в валидационном окне. Перед обучением
прогоняется чек-лист антиутечек (mayak.leakage). Диагностика поля после этапа A —
отдельный скрипт scripts/diagnose_stage_a.py.

Запуск отладка:
    python scripts/train.py --steps-a 200 --steps-b 1000 --batch 32 \\
        --windows 2000 --workers 0 --accelerator cpu

Запуск полный:
    python scripts/train.py --steps-a 10000 --steps-b 200000 --batch 256 \\
        --windows 200000 --workers 8 --accelerator gpu
"""
import argparse

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from mayak.data.datamodule import MayakData
from mayak.leakage import run_checklist
from mayak.lit import LitMayak, SelectionProvenance

MONITOR = "val/loss"


def run_stage(lit, manifest, curriculum, max_steps, args, tag, val_L=672):
    dm = MayakData(manifest=manifest, curriculum=curriculum,
                   batch_size=args.batch, windows_per_epoch=args.windows,
                   num_workers=args.workers, val_L=val_L)
    dm.setup("fit")
    run_checklist(dm.store, datasets=[dm.train_ds, dm.val_ds])
    lit.hparams.total_steps = max_steps
    ckpt = ModelCheckpoint(dirpath=f"runs/{tag}", monitor=MONITOR,
                           save_top_k=1, mode="min", filename="best")
    trainer = L.Trainer(
        max_steps=max_steps, accelerator=args.accelerator, devices=1,
        precision=args.precision, gradient_clip_val=1.0,
        val_check_interval=args.val_every, check_val_every_n_epoch=None,
        limit_val_batches=args.val_batches, logger=CSVLogger("runs", name=tag),
        log_every_n_steps=20,
        callbacks=[ckpt, SelectionProvenance(), LearningRateMonitor("step"),
                   EarlyStopping(monitor=MONITOR, patience=5, mode="min")],
        enable_progress_bar=True)
    trainer.fit(lit, datamodule=dm)
    return ckpt.best_model_path


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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

    print(">>> Этап A: только L=0 (стабилизация климат-поля)")
    best_a = run_stage(LitMayak(), args.manifest, "L0", args.steps_a, args,
                       tag="stageA", val_L=0)
    print("Лучшая модель этапа A:", best_a)
    print(f"    диагностика поля: python scripts/diagnose_stage_a.py --ckpt {best_a}")

    print(">>> Этап B: полный куррикулум")
    lit = LitMayak.load_from_checkpoint(best_a)
    best_b = run_stage(lit, args.manifest, "full", args.steps_b, args, tag="stageB", val_L=672)
    print("Лучшая модель этапа B:", best_b)


if __name__ == "__main__":
    main()
