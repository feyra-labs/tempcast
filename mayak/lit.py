"""LightningModule для МАЯК"""
import copy
import torch
import pytorch_lightning as L

from mayak.model import MAYAK
from mayak.loss import mayak_loss, pinball_loss
from mayak.baselines import GRUSeq2Seq, DLinear


NO_WD_SUFFIX = ("raw_tau", "p_w", "p_k")
LEADS = [1, 3, 6, 12, 24, 48, 72, 120, 168]


class EMA:
    """Экспоненциальное скользящее среднее весов модели"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def store_and_apply(self, model):
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


class LitMayak(L.LightningModule):
    def __init__(self, lr=3e-3, weight_decay=1e-2, total_steps=200_000,
                 ema_decay=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAYAK()
        self.ema = None

    def on_train_start(self):
        if self.ema is None:
            self.ema = EMA(self.model, self.hparams.ema_decay)

    def training_step(self, batch, _):
        out = self.model(batch)
        loss = mayak_loss(out, batch["y"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["y"].shape[0])
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_start(self):
        if self.ema is not None:
            self.ema.store_and_apply(self.model)

    def on_validation_end(self):
        if self.ema is not None:
            self.ema.restore(self.model)

    def validation_step(self, batch, _):
        out = self.model(batch)
        y = batch["y"]
        loss = mayak_loss(out, y)
        self.log("val/loss", loss, prog_bar=True, batch_size=y.shape[0])
        med = out["q"][..., 3]
        for h in LEADS:
            mae = (med[:, h - 1] - y[:, h - 1]).abs().mean()
            self.log(f"val/mae_{h}h", mae, batch_size=y.shape[0])
        return loss

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(NO_WD_SUFFIX):
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.total_steps)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
    
class LitBaseline(L.LightningModule):
    """Обучение нейробейзлайнов GRU/DLinear"""
    def __init__(self, model_name="gru", lr=3e-3, weight_decay=1e-2,
                 total_steps=200_000, ema_decay=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.model = {"gru": GRUSeq2Seq, "dlinear": DLinear}[model_name]()
        self.ema = None

    def on_train_start(self):
        if self.ema is None:
            self.ema = EMA(self.model, self.hparams.ema_decay)

    def training_step(self, batch, _):
        out = self.model(batch)
        loss = pinball_loss(out, batch["y"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["y"].shape[0])
        return loss

    def on_before_zero_grad(self, *a, **k):
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_start(self):
        if self.ema is not None:
            self.ema.store_and_apply(self.model)

    def on_validation_end(self):
        if self.ema is not None:
            self.ema.restore(self.model)

    def validation_step(self, batch, _):
        out = self.model(batch); y = batch["y"]
        loss = pinball_loss(out, y)
        self.log("val/loss", loss, prog_bar=True, batch_size=y.shape[0])
        med = out["q"][..., 3]
        for h in LEADS:
            self.log(f"val/mae_{h}h", (med[:, h - 1] - y[:, h - 1]).abs().mean(),
                     batch_size=y.shape[0])
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}