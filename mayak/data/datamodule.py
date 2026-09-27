"""Данные одного этапа обучения: поток обучающих окон и набор валидации.

Параметр curriculum задаёт этап и распределение длины истории:
  'L0'   — этап A, только холодный старт;
  'full' — этап B, полный куррикулум длины истории.

Валидация берёт окна валидационных станций в валидационном окне, с каждой станции
одинаковое число. Длина истории каждого окна выбирается из того же распределения, что
при обучении этапа, генератором с фиксированным сидом, поэтому набор один и тот же для
всех архитектур и всех повторов.
"""
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from mayak.config import DataConfig
from mayak.data.dataset import WindowDataset, seed_worker
from mayak.data.holdout import EvalSet
from mayak.data.splits import ROLE_VAL
from mayak.data.store import get_store


class MayakData(L.LightningDataModule):
    def __init__(self, manifest="data/manifest.csv", curriculum="full",
                 batch_size=256, windows_per_epoch=200_000,
                 num_workers=4, seed=0, aug_seed=None, data_config=None):
        super().__init__()
        if isinstance(data_config, DataConfig):
            data_config = data_config.to_dict()
        self.save_hyperparameters()
        self.data_cfg = DataConfig.from_dict(data_config) if data_config else DataConfig()

    def setup(self, stage=None):
        if getattr(self, "train_ds", None) is not None:
            return
        h, c = self.hparams, self.data_cfg
        store = get_store(h.manifest, cache_root=c.cache_root)
        self.train_ds = WindowDataset(h.manifest, split="train",
                                      curriculum=h.curriculum,
                                      windows_per_epoch=h.windows_per_epoch,
                                      seed=h.seed, aug_seed=h.aug_seed, store=store,
                                      target_mask=c.target_mask, augment=c.augment,
                                      window_qc=c.window_qc,
                                      zone_weighting=c.zone_weighting,
                                      zone_weight_cap=c.zone_weight_cap)
        self.store = store
        self.val_ds = EvalSet(store.clims(), station_splits=(ROLE_VAL,), manifest=h.manifest,
                              time_key="val", every_hours=c.val_every_hours, max_windows=None,
                              windows_per_station=c.val_windows_per_station,
                              target_mask=c.target_mask, curriculum=h.curriculum,
                              history_seed=c.val_seed)

    def train_dataloader(self):
        h = self.hparams
        return DataLoader(self.train_ds, batch_size=h.batch_size, shuffle=False,
                          num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0, drop_last=True,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(h.seed))

    def val_dataloader(self):
        h = self.hparams
        return DataLoader(self.val_ds, batch_size=h.batch_size, shuffle=False,
                          num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0)
