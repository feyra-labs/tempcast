"""
Параметр curriculum переключает этапы обучения:
  'L0'   — этап A (только холодный старт);
  'full' — этап B (полный куррикулум длины истории).
"""
import pytorch_lightning as L
from torch.utils.data import DataLoader

from mayak.data.dataset import WindowDataset, HoldoutDataset


class MayakData(L.LightningDataModule):
    def __init__(self, manifest="data/manifest.csv", curriculum="full",
                 batch_size=256, windows_per_epoch=200_000,
                 num_workers=4, seed=0, val_L=672):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        self.train_ds = WindowDataset(h.manifest, split="train",
                                      curriculum=h.curriculum,
                                      windows_per_epoch=h.windows_per_epoch,
                                      seed=h.seed)
        self.val_ds = HoldoutDataset(h.manifest, station_split="unseen_val",
                                     time_key="calib", every_hours=72, L=h.val_L)

    def train_dataloader(self):
        h = self.hparams
        return DataLoader(self.train_ds, batch_size=h.batch_size, shuffle=False,
                          num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        h = self.hparams
        return DataLoader(self.val_ds, batch_size=h.batch_size, shuffle=False,
                          num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0)