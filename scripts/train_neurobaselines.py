"""Обучение нейробейзлайнов (GRU seq2seq, DLinear)

Запуск полный:
    python scripts/train_baselines.py --models gru dlinear --steps 200000 \
        --batch 256 --windows 200000 --workers 8 --accelerator gpu --precision bf16-mixed
Отладка:
    python scripts/train_baselines.py --models gru dlinear --steps 1000 \
        --batch 32 --windows 2000 --workers 0 --accelerator cpu --precision 32 --val-every 200
"""
import argparse
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch

from mayak.data.datamodule import MayakData
from mayak.lit import LitBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gru", "dlinear"])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--windows", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    args = ap.parse_args()

    for name in args.models:
        print(f">>> Обучение бейзлайна: {name}")
        lit = LitBaseline(model_name=name, total_steps=args.steps)
        dm = MayakData(manifest=args.manifest, curriculum="full",
                       batch_size=args.batch, windows_per_epoch=args.windows,
                       num_workers=args.workers)
        print('>>> Обучение...')
        ckpt = ModelCheckpoint(dirpath=f"runs/baseline_{name}", monitor="val/loss",
                               save_top_k=1, mode="min", filename="best")
        trainer = L.Trainer(
            max_steps=args.steps, accelerator=args.accelerator, devices=1,
            precision=args.precision, gradient_clip_val=1.0,
            val_check_interval=args.val_every, check_val_every_n_epoch=None,
            limit_val_batches=args.val_batches, log_every_n_steps=20,
            logger=CSVLogger("runs", name=f"baseline_{name}"),
            callbacks=[ckpt, LearningRateMonitor("step"), EarlyStopping(monitor="val/loss", patience=5, mode="min")])
        trainer.fit(lit, datamodule=dm)
        print(f"    лучший чекпойнт: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()