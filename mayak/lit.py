"""LightningModule - один на все архитектуры.

Оптимизатор, расписание, EMA весов, функция потерь, правило весового затухания и
логирование валидации берутся из протокола обучения и одинаковы для любой архитектуры;
скорость обучения каждая архитектура получает подбором по общей сетке протокола.
Архитектура отвечает только за прямой проход, свои регуляризаторы (не зависят от цели)
и объявленные отличия от правила весового затухания.

Критерий выбора чекпойнта ``val/loss`` - общая часть функции потерь (нормированный
pinball) без регуляризаторов: одно и то же число для всех моделей. Рядом пишется тот же
нормированный pinball по бинам длины истории; на выбор он не влияет и нужен, чтобы видеть,
на каком участке от холодного старта до полной истории модель лучше или хуже.

Чекпойнт самодостаточен: в гиперпараметрах лежат архитектура, её конфиг, протокол
и конфиг данных, под ключом ``RUN_KEY`` - полностью разрешённый конфиг прогона,
сиды, хеш коммита и версии библиотек, а под ключом ``TUNING_KEY`` - запись о подборе
скорости обучения, если прогон шёл с ней. ``load_model`` восстанавливает архитектуру
из чекпойнта, а не из значений по умолчанию.
"""
import copy
import functools

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from mayak.baselines import DLinear, GRUSeq2Seq, LRUForecaster, PatchTST
from mayak.config import DataConfig, RunConfig, model_config_for
from mayak.data.holdout import HISTORY_BINS
from mayak.leakage import SELECTION_KEY, selection_record
from mayak.loss import forecast_loss, forecast_terms, masked_mean
from mayak.model import MAYAK
from mayak.protocol import (ARCH_NAMES, DEFAULT_PROTOCOL, LR_FIELDS, Protocol, ProtocolError,
                            protocol_diff)
from mayak.stages import STAGE_KEY
from mayak.tuning import (EXTRA_ARCH, PHASE_EXTRA, TUNING_KEY, describe_phase,
                          equal_terms_problems, search_terms)

LEADS = [1, 3, 6, 12, 24, 48, 72, 120, 168]
RUN_KEY = "mayak_run"

DECAY_LAYERS = (nn.Linear, nn.Conv1d, nn.GRU)

ARCHS = {"mayak": MAYAK, "gru": GRUSeq2Seq, "dlinear": DLinear, "lru": LRUForecaster,
         "patchtst": PatchTST}
assert tuple(ARCHS) == ARCH_NAMES, "реестр архитектур расходится с mayak.protocol.ARCH_NAMES"


def build_model(arch, model_config=None):
    """Модель архитектуры по её конфигу.

    Args:
        arch: имя архитектуры.
        model_config: None, словарь или датакласс конфига.

    Returns:
        Новая модель.

    Raises:
        ValueError: архитектура неизвестна.
    """
    if arch not in ARCHS:
        raise ValueError(f"неизвестная архитектура {arch!r}; есть {tuple(ARCHS)}")
    return ARCHS[arch](model_config_for(arch, model_config))


def regularization(model, out):
    """Регуляризаторы архитектуры, не зависящие от цели.

    Args:
        model: модель любой архитектуры.
        out: выход модели.

    Returns:
        Скаляр; ноль, если у архитектуры регуляризаторов нет.
    """
    fn = getattr(model, "regularization", None)
    return fn(out) if fn is not None else out["q"].new_zeros(())


def decayed_by_rule(model):
    """Параметры, которые по общему правилу получают весовое затухание.

    Args:
        model: модель любой архитектуры.

    Returns:
        Множество имён параметров: веса слоёв ``Linear``, ``Conv1d`` и ``GRU``.
    """
    return {f"{mod_name}.{name}" if mod_name else name
            for mod_name, mod in model.named_modules() if isinstance(mod, DECAY_LAYERS)
            for name, _ in mod.named_parameters(recurse=False) if name.startswith("weight")}


def optim_groups(model, weight_decay):
    """Группы параметров для оптимизатора по общему правилу весового затухания.

    Базовое затухание протокола получают только веса слоёв ``Linear``, ``Conv1d`` и
    ``GRU``. Все остальные параметры - смещения, нормы, скалярные, векторные и прочие
    обучаемые тензоры - идут без затухания. Архитектура может объявить отличия методом
    ``decay_exceptions(weight_decay)``: список групп с именем, затуханием и признаком
    ``match(имя параметра, затухание по правилу)``. Параметр попадает в первую
    подходящую группу отличий, иначе - в группу по правилу.

    Args:
        model: модель любой архитектуры.
        weight_decay: базовое весовое затухание протокола.

    Returns:
        Список групп с ключами name, params и weight_decay: группы отличий в порядке
        объявления, затем ``decay`` и ``no_decay``. Каждый обучаемый параметр - ровно в
        одной группе; группа может быть пустой.
    """
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
    """Число параметров модели: всего и по верхним блокам.

    Args:
        model: модель любой архитектуры.

    Returns:
        Словарь с ключами total (все параметры) и by_module (число параметров каждого
        дочернего блока верхнего уровня, в порядке объявления; блоки без параметров
        тоже перечислены). Параметры, объявленные прямо на модели, идут под ключом
        own, если они есть. Сумма by_module равна total.
    """
    by_module = {name: int(sum(p.numel() for p in child.parameters()))
                 for name, child in model.named_children()}
    own = int(sum(p.numel() for p in model.parameters(recurse=False)))
    if own:
        by_module["own"] = own
    return dict(total=int(sum(p.numel() for p in model.parameters())), by_module=by_module)


def param_group_summary(model, weight_decay):
    """Сводка групп весового затухания для журнала прогона.

    Args:
        model: модель любой архитектуры.
        weight_decay: базовое весовое затухание протокола.

    Returns:
        Список по группам: имя, затухание, число тензоров и параметров и имена
        параметров группы.
    """
    names = {id(p): n for n, p in model.named_parameters()}
    return [dict(name=g["name"], n_tensors=len(g["params"]),
                 n_params=int(sum(p.numel() for p in g["params"])),
                 weight_decay=float(g["weight_decay"]),
                 params=[names[id(p)] for p in g["params"]])
            for g in optim_groups(model, weight_decay)]


def history_bin_key(lo, hi):
    """Имя записи журнала для бина длины истории.

    Args:
        lo: наименьшая длина истории бина, ч.
        hi: наибольшая длина истории бина, ч.

    Returns:
        Имя вида val/pinball_L0 или val/pinball_L1-24.
    """
    return f"val/pinball_L{lo}" if lo == hi else f"val/pinball_L{lo}-{hi}"


HISTORY_LOG = tuple((lo, hi, history_bin_key(lo, hi)) for lo, hi, _name in HISTORY_BINS)


def log_history_bins(module, per_pair, weight, hist_len):
    """Нормированный pinball валидации по бинам длины истории.

    Вес записи равен числу валидных пар окна и лида бина в батче. Поэтому среднее за
    проход валидации совпадает со средним по всем парам бина во всём наборе, так же как
    у общего числа. Бин без пар в батче не пишется.

    Args:
        module: модуль, в журнал которого идут записи.
        per_pair: нормированный pinball пар, форма (B, H).
        weight: вес пар, форма (B, H).
        hist_len: длина истории окон, ч, форма (B,).
    """
    for lo, hi, key in HISTORY_LOG:
        inside = ((hist_len >= lo) & (hist_len <= hi)).to(weight.dtype)
        sel = weight * inside[:, None]
        n = int(sel.sum())
        if n:
            module.log(key, masked_mean(per_pair, sel), batch_size=n)


class SelectionProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись о том, на чём он выбирался.

    В записи метрика монитора, роль станций, временное окно и правило длины истории
    набора валидации, его отпечаток и все валидационные числа на момент сохранения.
    Без этой записи чек-лист не примет чекпойнт.
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        cb = trainer.checkpoint_callback
        monitor = getattr(cb, "monitor", None)
        scores = {k: float(v) for k, v in trainer.callback_metrics.items()
                  if k.startswith("val/")}
        checkpoint[SELECTION_KEY] = selection_record(trainer.datamodule.val_ds, monitor,
                                                     scores=scores)


class StageProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись об этапе.

    В записи имя и номер этапа в протоколе, ключ кэша данных, каталог и журнал прогона,
    чекпойнт, с которого этап стартовал, и шаг сохранения.
    По этой записи следующий этап проверяет, что стартует с совместимой точки.

    Args:
        record: запись об этапе без шага сохранения.
    """

    def __init__(self, record):
        self.record = dict(record)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[STAGE_KEY] = dict(self.record, step=int(trainer.global_step))


class TuningProvenance(Callback):
    """Кладёт в каждый сохраняемый чекпойнт запись о подборе скорости обучения прогона.

    Args:
        record: запись о подборе.
    """

    def __init__(self, record):
        self.record = dict(record)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[TUNING_KEY] = self.record


class CandidateCheckpoint(ModelCheckpoint):
    """Сохранение чекпойнта после каждой валидации этапа.

    Сохранение идёт в конце прохода валидации, пока в модели стоят усреднённые веса,
    поэтому кандидат несёт те же веса, что увидела валидация. Отдельный класс нужен,
    чтобы состояние этого сохранения хранилось в чекпойнте отдельно от состояния выбора
    лучшего.
    """


class EMA:
    """Экспоненциальное скользящее среднее весов модели.

    Args:
        model: модель, веса которой усредняются.
        decay: доля старого среднего на каждом шаге.
    """

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
    """Обучение любой архитектуры проекта по протоколу.

    Args:
        arch: имя архитектуры.
        protocol: протокол или его словарь; словарь хранится в гиперпараметрах
            чекпойнта.
        stage: имя этапа протокола для журнала.
        total_steps: число шагов этого этапа; по нему расписание WSD ставит спад.
        model_config: конфиг архитектуры.
        data_config: конфиг данных, с которым шло обучение; пишется в чекпойнт.
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
    """Запись о прогоне для чекпойнта.

    Args:
        run_config: полная конфигурация прогона.

    Returns:
        Словарь: разрешённый конфиг, сиды, коммит и версии библиотек.
    """
    from mayak.provenance import provenance
    return dict(config=run_config.to_dict(), seeds=run_config.train.resolved_seeds(),
                provenance=provenance())


def load_model(path, map_location="cpu"):
    """Модель из чекпойнта любой архитектуры.

    Архитектура и её конфиг берутся из гиперпараметров чекпойнта.

    Args:
        path: путь к чекпойнту.
        map_location: устройство для весов.

    Returns:
        Модель с весами чекпойнта.
    """
    return LitForecaster.load_from_checkpoint(path, map_location=map_location).model


def checkpoint_terms(path):
    """Архитектура, протокол обучения и запись о подборе чекпойнта.

    Args:
        path: путь к чекпойнту.

    Returns:
        Тройка: имя архитектуры, протокол и запись о подборе или None.

    Raises:
        ProtocolError: в чекпойнте нет протокола или он не читается текущим протоколом,
            например чекпойнт обучен до смены расписания.
    """
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
    return hp["arch"], protocol, ck.get(TUNING_KEY)


def check_comparable(reference, others, ignore=()):
    """Проверка, что все чекпойнты сравнения обучены в одинаковых условиях.

    Протоколы должны совпадать во всех полях, кроме скорости обучения и полей из
    ``ignore``. Скорость обучения у каждой модели своя, поэтому у каждого чекпойнта,
    включая эталон, должна быть запись о подборе этапа 1 сравнения с той же сеткой, тем
    же числом шагов этапов и той же метрикой выбора, что у эталона, а скорость обучения
    чекпойнта - выбранная этим подбором.

    Args:
        reference: путь к эталонному чекпойнту.
        others: пути к остальным чекпойнтам.
        ignore: поля протокола, которые могут различаться, например сиды у повторов
            основной модели.

    Returns:
        Словарь из пути чекпойнта в его архитектуру.

    Raises:
        ProtocolError: протоколы различаются, у чекпойнта нет протокола или записи о
            подборе, или подбор шёл в других условиях.
    """
    ignore = (*ignore, *LR_FIELDS)
    ref_arch, ref, ref_tune = checkpoint_terms(reference)
    archs = {reference: ref_arch}
    bad = [f"{reference} ({ref_arch}): {p}" for p in equal_terms_problems(ref_tune, ref)]
    ref_terms = search_terms(ref_tune)
    for path in others:
        arch, p, tune = checkpoint_terms(path)
        archs[path] = arch
        diff = protocol_diff(ref, p, ignore)
        if diff:
            bad.append(f"{path} ({arch}): отличаются {diff}")
        own = equal_terms_problems(tune, p)
        bad += [f"{path} ({arch}): {x}" for x in own]
        terms = search_terms(tune)
        terms_diff = sorted(k for k in ref_terms if ref_terms[k] != terms[k])
        if not own and ref_tune and terms_diff:
            bad.append(f"{path} ({arch}): подбор скорости обучения шёл в других условиях, "
                       f"отличаются {terms_diff}")
    if bad:
        raise ProtocolError(f"модели сравнения обучены в разных условиях (эталон — "
                            f"{reference}, {ref_arch}):\n  " + "\n  ".join(bad))
    return archs


def check_extra_tuning(path):
    """Проверка, что чекпойнт - прогон дополнительной настройки МАЯК.

    Args:
        path: путь к чекпойнту.

    Raises:
        ProtocolError: архитектура не МАЯК или прогон не помечен как этап 2 сравнения.
    """
    arch, _protocol, tune = checkpoint_terms(path)
    phase = (tune or {}).get("phase")
    if arch != EXTRA_ARCH or phase != PHASE_EXTRA:
        raise ProtocolError(f"{path} ({arch}): нужен прогон {describe_phase(PHASE_EXTRA)} "
                            f"(scripts/run.py run.extra_tuning=true), а это прогон "
                            f"{describe_phase(phase)}")


def load_run_record(path):
    """Запись о прогоне из чекпойнта.

    Args:
        path: путь к чекпойнту.

    Returns:
        Запись о прогоне: разрешённый конфиг, сиды, коммит и версии библиотек.

    Raises:
        ValueError: в чекпойнте нет записи о прогоне.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if RUN_KEY not in ck:
        raise ValueError(f"{path}: в чекпойнте нет записи о прогоне")
    return ck[RUN_KEY]
