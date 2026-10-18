"""LightningModule для всех архитектур."""
import copy
import functools
import json
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

from mayak.baselines import DLinear, GRUSeq2Seq, LRUForecaster, PatchTST
from mayak.config import DataConfig, RunConfig, model_config_for
from mayak.data.holdout import HISTORY_BINS
from mayak.leakage import SELECTION_KEY, selection_record
from mayak.loss import forecast_loss, forecast_terms, masked_mean
from mayak.model import MAYAK
from mayak.protocol import (ARCH_NAMES, DEFAULT_PROTOCOL, LR_FIELDS, Protocol, ProtocolError,
                            protocol_diff)
from mayak.stages import STAGE_KEY, jsonable
from mayak.tuning import (EXTRA_ARCH, PHASE_EXTRA, TUNING_KEY, describe_phase,
                          equal_terms_problems, search_terms)

LEADS = [1, 3, 6, 12, 24, 48, 72, 120, 168]
RUN_KEY = "mayak_run"

DECAY_LAYERS = (nn.Linear, nn.Conv1d, nn.GRU)

ARCHS = {"mayak": MAYAK, "gru": GRUSeq2Seq, "dlinear": DLinear, "lru": LRUForecaster,
         "patchtst": PatchTST}
assert tuple(ARCHS) == ARCH_NAMES, "реестр архитектур расходится с mayak.protocol.ARCH_NAMES"


def build_model(arch, model_config=None):
    """Модель архитектуры по её конфигу."""
    if arch not in ARCHS:
        raise ValueError(f"неизвестная архитектура {arch!r}; есть {tuple(ARCHS)}")
    return ARCHS[arch](model_config_for(arch, model_config))


def regularization(model, out):
    """Регуляризаторы архитектуры, не зависящие от цели."""
    fn = getattr(model, "regularization", None)
    return fn(out) if fn is not None else out["q"].new_zeros(())


def decayed_by_rule(model):
    """Параметры, которые по общему правилу получают весовое затухание."""
    return {f"{mod_name}.{name}" if mod_name else name
            for mod_name, mod in model.named_modules() if isinstance(mod, DECAY_LAYERS)
            for name, _ in mod.named_parameters(recurse=False) if name.startswith("weight")}


def optim_groups(model, weight_decay):
    """Группы параметров для оптимизатора по общему правилу весового затухания."""
    rule = decayed_by_rule(model)
    fn = getattr(model, "decay_exceptions", None)
    special = list(fn(weight_decay)) if fn is not None else []
    groups = [dict(name=s["name"], params=[], weight_decay=float(s["weight_decay"]))
              for s in special]
    decay = dict(name="decay", params=[], weight_decay=float(weight_decay))
    no_decay = dict(name="no_decay", params=[], weight_decay=0.0)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        by_rule = name in rule
        for s, g in zip(special, groups):
            if s["match"](name, by_rule):
                g["params"].append(p)
                break
        else:
            (decay if by_rule else no_decay)["params"].append(p)
    return [*groups, decay, no_decay]


def parameter_counts(model):
    """Число параметров модели: всего и по верхним блокам."""
    by_module = {name: int(sum(p.numel() for p in child.parameters()))
                 for name, child in model.named_children()}
    own = int(sum(p.numel() for p in model.parameters(recurse=False)))
    if own:
        by_module["own"] = own
    return dict(total=int(sum(p.numel() for p in model.parameters())), by_module=by_module)


def param_group_summary(model, weight_decay):
    """Сводка групп весового затухания для журнала прогона."""
    names = {id(p): n for n, p in model.named_parameters()}
    return [dict(name=g["name"], n_tensors=len(g["params"]),
                 n_params=int(sum(p.numel() for p in g["params"])),
                 weight_decay=float(g["weight_decay"]),
                 params=[names[id(p)] for p in g["params"]])
            for g in optim_groups(model, weight_decay)]


def history_bin_key(lo, hi):
    """Имя записи журнала для бина длины истории."""
    return f"val/pinball_L{lo}" if lo == hi else f"val/pinball_L{lo}-{hi}"


HISTORY_LOG = tuple((lo, hi, history_bin_key(lo, hi)) for lo, hi, _name in HISTORY_BINS)


def log_history_bins(module, per_pair, weight, hist_len):
    """Нормированный pinball валидации по бинам длины истории."""
    for lo, hi, key in HISTORY_LOG:
        inside = ((hist_len >= lo) & (hist_len <= hi)).to(weight.dtype)
        sel = weight * inside[:, None]
        n = int(sel.sum())
        if n:
            module.log(key, masked_mean(per_pair, sel), batch_size=n)


class SelectionProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись о том, на чём он выбирался."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        cb = trainer.checkpoint_callback
        monitor = getattr(cb, "monitor", None)
        scores = {k: float(v) for k, v in trainer.callback_metrics.items()
                  if k.startswith("val/")}
        checkpoint[SELECTION_KEY] = selection_record(trainer.datamodule.val_ds, monitor,
                                                     scores=scores)


class StageProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись об этапе."""

    def __init__(self, record):
        self.record = dict(record)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[STAGE_KEY] = dict(self.record, step=int(trainer.global_step))


class TuningProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись о подборе скорости обучения прогона."""

    def __init__(self, record):
        self.record = dict(record)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[TUNING_KEY] = self.record


class EMA:
    """Экспоненциальное скользящее среднее весов модели."""

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
    """Обучение любой архитектуры проекта по протоколу."""

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
            data_config if isinstance(data_config, DataConfig)
            else DataConfig.from_dict(data_config))
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
        out = self.model(batch)
        per_pair, weight = forecast_terms(out, batch)
        data = masked_mean(per_pair, weight)
        reg = regularization(self.model, out)
        log_val_metrics(self, out, batch, data)
        if "hist_len" in batch:
            log_history_bins(self, per_pair, weight, batch["hist_len"])
        self.log("val/total", data + reg, batch_size=batch["y"].shape[0])
        return data

    def configure_optimizers(self):
        p = self.protocol
        groups = optim_groups(self.model, p.weight_decay)
        opt = torch.optim.AdamW([dict(params=g["params"], weight_decay=g["weight_decay"])
                                 for g in groups if g["params"]],
                                lr=p.lr, betas=p.betas)
        # Множитель - функция номера шага этапа (Protocol.lr_factor). Лямбда-планировщик
        # шагает после каждого шага оптимизатора, поэтому шаг s идёт со скоростью lr * f(s).
        factor = functools.partial(p.lr_factor, total_steps=int(self.hparams.total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def run_record(run_config):
    """Запись о прогоне для чекпойнта."""
    from mayak.provenance import provenance
    return dict(config=run_config.to_dict(), seeds=run_config.train.resolved_seeds(),
                provenance=provenance())


def load_model(path, map_location="cpu"):
    """Модель из чекпойнта любой архитектуры."""
    return LitForecaster.load_from_checkpoint(path, map_location=map_location).model


@dataclass(frozen=True)
class CheckpointTerms:
    """Условия обучения чекпойнта, которые сверяет стенд оценки."""
    arch: str
    protocol: Protocol
    tuning: Optional[dict] = None
    model_config: Optional[dict] = None
    data_config: Optional[dict] = None
    data_key: Optional[str] = None
    stage: Optional[dict] = None


def checkpoint_terms(path):
    """Условия обучения чекпойнта."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    if "protocol" not in hp:
        raise ProtocolError(f"{path}: в чекпойнте нет протокола обучения")
    try:
        protocol = Protocol.from_dict(hp["protocol"])
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"{path}: протокол чекпойнта несовместим с текущим ({e}); "
                            f"чекпойнт обучен по старому протоколу, обучите модель заново"
                            ) from e
    stage = ck.get(STAGE_KEY) or {}
    return CheckpointTerms(arch=hp["arch"], protocol=protocol, tuning=ck.get(TUNING_KEY),
                           model_config=hp.get("model_config"),
                           data_config=hp.get("data_config"), data_key=stage.get("data_key"),
                           stage=stage or None)


def completion_problems(terms):
    """Почему чекпойнт не годится как итоговый: этап не завершён или этап не последний."""
    from mayak.stages import COMPLETE_FIELD
    st = terms.stage or {}
    last = terms.protocol.stages[-1]
    done = st.get(COMPLETE_FIELD)
    if not done:
        return [f"этап {st.get('stage', '?')} не отмечен как завершённый: прогон прерван "
                f"или обучен до этой проверки, обучите этап до конца"]
    out = []
    if st.get("stage") is not None and st.get("stage") != last.name:
        out.append(f"чекпойнт этапа {st.get('stage')}, а итоговый — этапа {last.name}")
    if int(done.get("steps_done", 0)) < last.steps:
        out.append(f"этап прошёл {done.get('steps_done')} из {last.steps} шагов")
    return out


def check_complete(path):
    """Проверка, что чекпойнт итоговый: последний этап протокола прошёл все шаги."""
    problems = completion_problems(checkpoint_terms(path))
    if problems:
        raise ProtocolError(f"{path}: " + "; ".join(problems))


# Путь к манифесту и каталог кэша зависят от того, откуда запущена команда. Совпадение
# данных проверяется по ключу кэша: он строится по содержимому манифеста и источников.
DATA_PATH_FIELDS = ("manifest", "cache_root")
ABLATIONS_FIELD = "ablations"


def _normalized(value):
    return json.loads(json.dumps(jsonable(value), sort_keys=True))


def config_diff(where, ref, own):
    """Отличия конфига чекпойнта от эталонного, по одной строке на поле."""
    ref, own = _normalized(ref), _normalized(own)
    if isinstance(ref, dict) and isinstance(own, dict):
        out = []
        for k in sorted(set(ref) | set(own)):
            if k not in ref:
                out.append(f"{where}.{k}: нет у эталона, у чекпойнта {own[k]!r}")
            elif k not in own:
                out.append(f"{where}.{k}: у эталона {ref[k]!r}, нет у чекпойнта")
            else:
                out += config_diff(f"{where}.{k}", ref[k], own[k])
        return out
    if ref != own:
        return [f"{where}: у эталона {ref!r}, у чекпойнта {own!r}"]
    return []


def _data_fields(d):
    return {k: v for k, v in dict(d or {}).items() if k not in DATA_PATH_FIELDS}


def _model_fields(d):
    return {k: v for k, v in dict(d or {}).items() if k != ABLATIONS_FIELD}


def missing_terms(terms):
    """Каких записей для сверки данных и модели в чекпойнте нет."""
    out = []
    if terms.model_config is None:
        out.append("нет конфига модели (hyper_parameters.model_config)")
    if terms.data_config is None:
        out.append("нет конфига данных (hyper_parameters.data_config)")
    if terms.data_key is None:
        out.append(f"нет ключа кэша данных в записи этапа ({STAGE_KEY})")
    return out


def data_problems(ref, terms):
    """Отличия данных чекпойнта от эталона."""
    out = []
    if terms.data_key != ref.data_key:
        out.append(f"ключ кэша данных {terms.data_key} отличается от эталонного {ref.data_key}")
    try:
        expected = RunConfig(model=model_config_for(terms.arch, terms.model_config),
                             data=DataConfig.from_dict(ref.data_config)).resolved().data
    except (TypeError, ValueError) as e:
        return [*out, f"конфиг данных или модели не читается текущим кодом ({e})"]
    diff = config_diff("data", _data_fields(expected.to_dict()), _data_fields(terms.data_config))
    if diff:
        out.append("конфиг данных отличается: " + "; ".join(diff))
    return out


def model_problems(ref, terms):
    """Отличия абляции или повтора основной модели от эталона."""
    out = []
    lr, ref_lr = float(terms.protocol.lr), float(ref.protocol.lr)
    if lr != ref_lr:
        out.append(f"скорость обучения {lr:g} отличается от эталонной {ref_lr:g}")
    diff = config_diff("model", _model_fields(ref.model_config), _model_fields(terms.model_config))
    if diff:
        out.append("конфиг модели отличается не только абляциями: " + "; ".join(diff))
    return out


def check_comparable(reference, others, ignore=(), same_model=False):
    """Проверка, что все чекпойнты сравнения обучены в одинаковых условиях."""
    ignore = (*ignore, *LR_FIELDS)
    ref = checkpoint_terms(reference)
    archs = {reference: ref.arch}
    ref_missing = missing_terms(ref)
    bad = [f"{reference} ({ref.arch}): {p}"
           for p in [*equal_terms_problems(ref.tuning, ref.protocol), *ref_missing,
                     *completion_problems(ref)]]
    ref_terms = search_terms(ref.tuning)
    for path in others:
        t = checkpoint_terms(path)
        archs[path] = t.arch
        problems = []
        diff = protocol_diff(ref.protocol, t.protocol, ignore)
        if diff:
            problems.append(f"отличаются {diff}")
        own = equal_terms_problems(t.tuning, t.protocol)
        problems += own
        own_terms = search_terms(t.tuning)
        terms_diff = sorted(k for k in ref_terms if ref_terms[k] != own_terms[k])
        if not own and ref.tuning and terms_diff:
            problems.append(f"подбор скорости обучения шёл в других условиях, "
                            f"отличаются {terms_diff}")
        missing = missing_terms(t)
        problems += missing
        problems += completion_problems(t)
        if not missing and not ref_missing:
            problems += data_problems(ref, t)
            if same_model:
                problems += model_problems(ref, t)
        bad += [f"{path} ({t.arch}): {p}" for p in problems]
    if bad:
        raise ProtocolError(f"модели сравнения обучены в разных условиях (эталон — "
                            f"{reference}, {ref.arch}):\n  " + "\n  ".join(bad))
    return archs


def check_extra_tuning(path):
    """Проверка, что чекпойнт - прогон дополнительной настройки МАЯК."""
    t = checkpoint_terms(path)
    phase = (t.tuning or {}).get("phase")
    if t.arch != EXTRA_ARCH or phase != PHASE_EXTRA:
        raise ProtocolError(f"{path} ({t.arch}): нужен прогон {describe_phase(PHASE_EXTRA)} "
                            f"(scripts/run.py run.extra_tuning=true), а это прогон "
                            f"{describe_phase(phase)}")
    problems = completion_problems(t)
    if problems:
        raise ProtocolError(f"{path} ({t.arch}): " + "; ".join(problems))


def load_run_record(path):
    """Запись о прогоне из чекпойнта."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if RUN_KEY not in ck:
        raise ValueError(f"{path}: в чекпойнте нет записи о прогоне")
    return ck[RUN_KEY]
