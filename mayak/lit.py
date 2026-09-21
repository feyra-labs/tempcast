"""LightningModule - один на все архитектуры.

Оптимизатор, расписание, EMA весов, функция потерь и логирование валидации берутся
из протокола (mayak/protocol.py) и одинаковы для любой архитектуры. Архитектура
отвечает только за прямой проход, свои регуляризаторы (не зависят от цели) и группы
весового затухания.

Критерий выбора чекпойнта ``val/loss`` - общая часть функции потерь (нормированный
pinball) без регуляризаторов: одно и то же число для всех моделей.

Чекпойнт самодостаточен: в гиперпараметрах лежат архитектура, её конфиг, протокол
и конфиг данных, а под ключом ``RUN_KEY`` - полностью разрешённый конфиг прогона,
сиды, хеш коммита и версии библиотек. ``load_model`` восстанавливает архитектуру
из чекпойнта, а не из значений по умолчанию.
"""
import copy

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import Callback

from mayak.baselines import DLinear, GRUSeq2Seq
from mayak.config import DataConfig, RunConfig, model_config_for
from mayak.leakage import SELECTION_KEY, selection_record
from mayak.loss import forecast_loss
from mayak.model import MAYAK
from mayak.protocol import ARCH_NAMES, DEFAULT_PROTOCOL, Protocol

LEADS = [1, 3, 6, 12, 24, 48, 72, 120, 168]
RUN_KEY = "mayak_run"

ARCHS = {"mayak": MAYAK, "gru": GRUSeq2Seq, "dlinear": DLinear}
assert tuple(ARCHS) == ARCH_NAMES, "реестр архитектур расходится с mayak.protocol.ARCH_NAMES"


def build_model(arch, model_config=None):
    """Модель архитектуры arch по её конфигу (None / словарь / датакласс)."""
    if arch not in ARCHS:
        raise ValueError(f"неизвестная архитектура {arch!r}; есть {tuple(ARCHS)}")
    return ARCHS[arch](model_config_for(arch, model_config))


def regularization(model, out):
    """Регуляризаторы архитектуры (0, если их нет)."""
    fn = getattr(model, "regularization", None)
    return fn(out) if fn is not None else out["q"].new_zeros(())


def optim_groups(model, weight_decay):
    """Группы параметров архитектуры; по умолчанию - одна группа с базовым затуханием."""
    fn = getattr(model, "optim_groups", None)
    if fn is not None:
        return fn(weight_decay)
    return [dict(name="all", params=[p for p in model.parameters() if p.requires_grad],
                 weight_decay=weight_decay)]


def param_group_summary(model, weight_decay):
    return [dict(name=g["name"], n_tensors=len(g["params"]),
                 n_params=int(sum(p.numel() for p in g["params"])),
                 weight_decay=float(g["weight_decay"]))
            for g in optim_groups(model, weight_decay)]


class SelectionProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись о том, на чём он выбирался:
    метрика монитора, роль станций и временное окно валидационного датасета.
     Без этой записи чек-лист не примет чекпойнт."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        cb = trainer.checkpoint_callback
        monitor = getattr(cb, "monitor", None)
        checkpoint[SELECTION_KEY] = selection_record(trainer.datamodule.val_ds, monitor)


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


def log_val_metrics(module, out, batch, loss):
    y, w = batch["y"], batch["y_mask"]
    module.log("val/loss", loss, prog_bar=True, batch_size=max(int(w.sum()), 1))
    med = out["q"][..., 3]
    for h in LEADS:
        wj = w[:, h - 1]
        n = int(wj.sum())
        if n == 0:
            continue
        mae = ((med[:, h - 1] - y[:, h - 1]).abs() * wj).sum() / n
        module.log(f"val/mae_{h}h", mae, batch_size=n)


class LitForecaster(L.LightningModule):
    """Обучение любой архитектуры из ARCHS по протоколу.

    protocol     — словарь Protocol.to_dict() (хранится в гиперпараметрах чекпойнта);
    stage        — имя этапа протокола (для журнала);
    total_steps  — длина косинусного расписания этого этапа;
    model_config — конфиг архитектуры;
    data_config  — конфиг данных, с которым шло обучение (для записи в чекпойнт).
    """

    def __init__(self, arch="mayak", protocol=None, stage=None, total_steps=None,
                 model_config=None, data_config=None):
        super().__init__()
        if isinstance(protocol, Protocol):
            protocol = protocol.to_dict()
        if protocol is None:
            protocol = DEFAULT_PROTOCOL.to_dict()
        p = Protocol.from_dict(protocol)
        if total_steps is None:
            total_steps = p.total_steps
        mcfg = model_config_for(arch, model_config)
        dcfg = DataConfig() if data_config is None else (
            data_config if isinstance(data_config, DataConfig) else DataConfig.from_dict(data_config))
        self.save_hyperparameters(dict(arch=arch, protocol=p.to_dict(), stage=stage,
                                       total_steps=int(total_steps),
                                       model_config=mcfg.to_dict(),
                                       data_config=dcfg.to_dict()))
        self.protocol = p
        self.run_config = RunConfig(model=mcfg, data=dcfg, train=p)
        self.model = build_model(arch, mcfg)
        self.ema = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint[RUN_KEY] = run_record(self.run_config)

    def losses(self, batch):
        out = self.model(batch)
        return out, forecast_loss(out, batch), regularization(self.model, out)

    def on_train_start(self):
        if self.ema is None:
            self.ema = EMA(self.model, self.protocol.ema_decay)

    def training_step(self, batch, _):
        _out, data, reg = self.losses(batch)
        loss = data + reg
        bs = batch["y"].shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/pinball", data, batch_size=bs)
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
        out, data, reg = self.losses(batch)
        log_val_metrics(self, out, batch, data)
        self.log("val/total", data + reg, batch_size=batch["y"].shape[0])
        return data

    def configure_optimizers(self):
        p = self.protocol
        groups = optim_groups(self.model, p.weight_decay)
        opt = torch.optim.AdamW([dict(params=g["params"], weight_decay=g["weight_decay"])
                                 for g in groups if g["params"]],
                                lr=p.lr, betas=p.betas)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def run_record(run_config):
    """Запись о прогоне для чекпойнта: разрешённый конфиг, сиды, коммит, версии."""
    from mayak.provenance import provenance
    return dict(config=run_config.to_dict(), seeds=run_config.train.resolved_seeds(),
                provenance=provenance())


# Обратная совместимость.
LitMayak = LitForecaster


def LitBaseline(model_name="gru", **kw):
    """Совместимость со старым интерфейсом: то же, что LitForecaster(arch=model_name)."""
    return LitForecaster(arch=model_name, **kw)


def load_model(path, map_location="cpu"):
    """Модель из чекпойнта любой архитектуры.

    Архитектура и её конфиг берутся из гиперпараметров чекпойнта; у чекпойнтов до
    блока 6 конфига нет - тогда значения по умолчанию, совпадающие с прежними константами.
    """
    return LitForecaster.load_from_checkpoint(path, map_location=map_location).model


def load_run_record(path):
    """Запись о прогоне из чекпойнта (RUN_KEY) или None у старых чекпойнтов."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get(RUN_KEY)
