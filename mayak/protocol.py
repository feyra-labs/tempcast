"""Единый протокол обучения.

Протокол описан в одном месте датаклассом Protocol и исполняется одной функцией
run_protocol. Для любой архитектуры одинаковы:

* этапы обучения и число шагов каждого (A: только L=0, B: полный куррикулум);
* размер батча, число окон в эпохе, число воркеров и сид, то есть поток окон;
* оптимизатор AdamW (lr, betas, базовое весовое затухание), косинусное расписание,
  обрезка градиента, точность вычислений;
* функция потерь (mayak/loss.py) и метрика выбора чекпойнта ``val/loss`` общий
  нормированный pinball на валидационных станциях в валидационном окне;
* ранняя остановка, выбор лучшего чекпойнта и экспоненциальное усреднение весов.

Различается только архитектура (``arch``). Что архитектура определяет сама:
регуляризаторы, не зависящие от цели, и группы весового затухания — они описаны
в докстринге класса модели и записываются в журнал прогона.

Отклонения. Если архитектура не сходится с общим протоколом, отклонение объявляется
в ARCH_DEVIATIONS с причиной и обязано быть описано в докстринге класса модели
(имя изменённого поля). run_protocol применяет его, пишет в журнал прогона
``<out_root>/<tag>/protocol.json`` и в гиперпараметры каждого чекпойнта.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from mayak.constants import L_MAX

log = logging.getLogger(__name__)

ARCH_NAMES = ("mayak", "gru", "dlinear")
CURRICULA = ("L0", "full")
LR_SCHEDULES = ("cosine",)
JOURNAL = "protocol.json"


class ProtocolError(RuntimeError):
    """Нарушение единого протокола обучения."""


def _jsonable(v):
    return json.loads(json.dumps(v, default=lambda o: asdict(o)))


@dataclass(frozen=True)
class Stage:
    name: str
    curriculum: str
    steps: int
    val_L: int

    def __post_init__(self):
        if self.curriculum not in CURRICULA:
            raise ValueError(f"этап {self.name}: неизвестный куррикулум {self.curriculum!r}")
        if int(self.steps) < 1:
            raise ValueError(f"этап {self.name}: число шагов < 1")
        if not 0 <= int(self.val_L) <= L_MAX:
            raise ValueError(f"этап {self.name}: val_L вне [0, {L_MAX}]")


@dataclass(frozen=True)
class Deviation:
    """Отклонение от общего протокола: поле, новое и исходное значение, причина."""
    field: str
    value: Any
    default: Any
    reason: str


@dataclass(frozen=True)
class Protocol:
    stages: tuple = (Stage("A", "L0", 10_000, 0), Stage("B", "full", 200_000, L_MAX))
    batch_size: int = 256
    windows_per_epoch: int = 200_000
    num_workers: int = 8
    seed: int = 0
    lr: float = 3e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.95)
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    precision: str = "bf16-mixed"
    ema_decay: float = 0.999
    monitor: str = "val/loss"
    val_every: int = 2000
    val_batches: int = 20
    patience: int = 5
    deviations: tuple = ()

    def __post_init__(self):
        st = tuple(s if isinstance(s, Stage) else Stage(**s) for s in self.stages)
        dv = tuple(d if isinstance(d, Deviation) else Deviation(**d) for d in self.deviations)
        object.__setattr__(self, "stages", st)
        object.__setattr__(self, "deviations", dv)
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        if not st:
            raise ValueError("протокол без этапов")
        if len({s.name for s in st}) != len(st):
            raise ValueError("имена этапов повторяются")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"неизвестное расписание {self.lr_schedule!r}")
        if not str(self.monitor).startswith("val/"):
            raise ValueError(f"метрика выбора {self.monitor!r} не валидационная")

    def to_dict(self):
        """JSON-совместимый словарь."""
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def total_steps(self):
        return sum(s.steps for s in self.stages)

    def deviate(self, reason, **changes):
        """Новый протокол с изменёнными полями; каждое изменение записывается с причиной."""
        if not reason or not str(reason).strip():
            raise ProtocolError("отклонение от протокола без причины недопустимо")
        allowed = {f.name for f in fields(self)} - {"deviations"}
        unknown = set(changes) - allowed
        if unknown:
            raise ProtocolError(f"неизвестные поля протокола: {sorted(unknown)}")
        devs = tuple(Deviation(k, _jsonable(v), _jsonable(getattr(self, k)), str(reason))
                     for k, v in changes.items())
        return replace(self, **changes, deviations=self.deviations + devs)

    def common(self):
        if not self.deviations:
            return self
        d = self.to_dict()
        for dev in reversed(self.deviations):
            d[dev.field] = dev.default
        d["deviations"] = []
        return Protocol.from_dict(d)


DEFAULT_PROTOCOL = Protocol()

ARCH_DEVIATIONS: dict = {}


def protocol_for(arch, base=DEFAULT_PROTOCOL):
    """Протокол прогона архитектуры: общий + её объявленные отклонения."""
    if arch not in ARCH_NAMES:
        raise ProtocolError(f"неизвестная архитектура {arch!r}; есть {ARCH_NAMES}")
    p = base
    for changes, reason in ARCH_DEVIATIONS.get(arch, ()):
        p = p.deviate(reason, **changes)
    return p


def check_deviations_documented(model_cls, protocol):
    """Каждое отклонение обязано быть названо в докстринге класса модели."""
    doc = model_cls.__doc__ or ""
    missing = [d.field for d in protocol.deviations if d.field not in doc]
    if missing:
        raise ProtocolError(f"{model_cls.__name__}: отклонения {missing} не описаны в "
                            f"докстринге класса — описание бейзлайна обязано их называть")


# (флаг, поле протокола, тип)
_CLI = [
    ("--batch", "batch_size", int), ("--windows", "windows_per_epoch", int),
    ("--workers", "num_workers", int), ("--seed", "seed", int),
    ("--lr", "lr", float), ("--weight-decay", "weight_decay", float),
    ("--grad-clip", "grad_clip", float), ("--precision", "precision", str),
    ("--ema-decay", "ema_decay", float), ("--val-every", "val_every", int),
    ("--val-batches", "val_batches", int), ("--patience", "patience", int),
]


def add_protocol_args(ap, base=DEFAULT_PROTOCOL):
    """Флаги протокола. Одни и те же для всех архитектур и всех точек входа обучения."""
    g = ap.add_argument_group("протокол обучения (одинаков для всех архитектур)")
    for s in base.stages:
        g.add_argument(f"--steps-{s.name.lower()}", type=int, default=s.steps,
                       help=f"шаги этапа {s.name} ({s.curriculum})")
    for flag, name, typ in _CLI:
        g.add_argument(flag, type=typ, default=getattr(base, name))
    return ap


def protocol_from_args(args, base=DEFAULT_PROTOCOL):
    stages = tuple(replace(s, steps=getattr(args, f"steps_{s.name.lower()}")) for s in base.stages)
    kw = {name: getattr(args, flag.lstrip("-").replace("-", "_")) for flag, name, _ in _CLI}
    return replace(base, stages=stages, **kw)


def _write_journal(path, journal):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(journal, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_protocol(arch, manifest, protocol=None, out_root="runs", accelerator="auto",
                 tag=None, callbacks=(), enable_progress_bar=True):
    """Обучить архитектуру ``arch`` по протоколу. Единственная функция запуска обучения.

    protocol — общий протокол (по умолчанию DEFAULT_PROTOCOL); объявленные отклонения
    архитектуры добавляются здесь же. Этап k стартует с лучшего чекпойнта этапа k−1.
    Возвращает журнал прогона (он же лежит в <out_root>/<tag>/protocol.json).
    """
    import pytorch_lightning as L
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from mayak.data.datamodule import MayakData
    from mayak.leakage import run_checklist
    from mayak.lit import ARCHS, LitForecaster, SelectionProvenance, param_group_summary

    protocol = protocol_for(arch, protocol or DEFAULT_PROTOCOL)
    check_deviations_documented(ARCHS[arch], protocol)
    for d in protocol.deviations:
        log.warning("%s: отклонение от протокола %s = %r (было %r): %s",
                    arch, d.field, d.value, d.default, d.reason)
    tag = tag or arch
    run_dir = os.path.join(out_root, tag)
    journal_path = os.path.join(run_dir, JOURNAL)
    journal = dict(arch=arch, model_class=f"{ARCHS[arch].__module__}.{ARCHS[arch].__qualname__}",
                   protocol=protocol.to_dict(), deviations=[asdict(d) for d in protocol.deviations],
                   manifest=os.path.abspath(manifest), stages=[], final_ckpt=None)

    prev_ckpt = None
    for i, stage in enumerate(protocol.stages):
        L.seed_everything(protocol.seed + i, workers=False, verbose=False)
        dm = MayakData(manifest=manifest, curriculum=stage.curriculum,
                       batch_size=protocol.batch_size, windows_per_epoch=protocol.windows_per_epoch,
                       num_workers=protocol.num_workers, seed=protocol.seed, val_L=stage.val_L)
        dm.setup("fit")
        run_checklist(dm.store, datasets=[dm.train_ds, dm.val_ds])

        lit = LitForecaster(arch=arch, protocol=protocol.to_dict(), stage=stage.name,
                            total_steps=stage.steps)
        if prev_ckpt is not None:
            sd = torch.load(prev_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            lit.load_state_dict(sd)
        if i == 0:
            journal["param_groups"] = param_group_summary(lit.model, protocol.weight_decay)
            journal["n_params"] = int(sum(p.numel() for p in lit.model.parameters()))

        stage_dir = os.path.join(run_dir, f"stage{stage.name}")
        ckpt = ModelCheckpoint(dirpath=stage_dir, monitor=protocol.monitor, mode="min",
                               save_top_k=1, filename="best")
        trainer = L.Trainer(
            max_steps=stage.steps, accelerator=accelerator, devices=1,
            precision=protocol.precision, gradient_clip_val=protocol.grad_clip,
            val_check_interval=min(protocol.val_every, stage.steps), check_val_every_n_epoch=None,
            limit_val_batches=protocol.val_batches,
            logger=CSVLogger(out_root, name=f"{tag}/stage{stage.name}"), log_every_n_steps=20,
            callbacks=[ckpt, SelectionProvenance(), LearningRateMonitor("step"),
                       EarlyStopping(monitor=protocol.monitor, patience=protocol.patience,
                                     mode="min"), *callbacks],
            enable_progress_bar=enable_progress_bar, enable_model_summary=False)
        trainer.fit(lit, datamodule=dm)
        if not ckpt.best_model_path:
            raise ProtocolError(f"{arch}: этап {stage.name} не сохранил чекпойнт — "
                                f"валидация ни разу не прошла")
        prev_ckpt = ckpt.best_model_path
        best = ckpt.best_model_score
        journal["stages"].append(dict(name=stage.name, curriculum=stage.curriculum,
                                      steps_done=int(trainer.global_step),
                                      best_ckpt=prev_ckpt,
                                      best_score=None if best is None else float(best)))
        journal["final_ckpt"] = prev_ckpt
        _write_journal(journal_path, journal)
    return journal


def read_journal(run_dir):
    with open(os.path.join(run_dir, JOURNAL)) as f:
        return json.load(f)


__all__ = ["ARCH_NAMES", "ARCH_DEVIATIONS", "DEFAULT_PROTOCOL", "Deviation", "Protocol",
           "ProtocolError", "Stage", "add_protocol_args", "check_deviations_documented",
           "protocol_for", "protocol_from_args", "read_journal", "run_protocol"]
