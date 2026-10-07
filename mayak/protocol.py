"""Единый протокол обучения.

Протокол описан в одном месте датаклассом Protocol и исполняется одной функцией
run_protocol. Для любой архитектуры одинаковы:

* этапы обучения и число шагов каждого: этап A только без истории, этап B - полный
  куррикулум длины истории;
* размер батча, число окон в эпохе, число воркеров и сид, то есть поток окон;
* оптимизатор AdamW (betas, базовое весовое затухание), расписание скорости обучения
  WSD (прогрев, плато, линейный спад до нуля в конце этапа), обрезка градиента, точность
  вычислений: fp32 без TF32, та же, в которой модель оценивают, калибруют и
  экспортируют;
* сетка скоростей обучения ``lr_grid`` и число шагов последнего этапа при подборе
  ``lr_search_steps``. Скорость обучения ``lr`` каждая архитектура получает подбором по
  этой сетке (``mayak.tuning``), поэтому ``lr`` - единственное поле протокола, которое у
  сравниваемых моделей различается;
* функция потерь и метрика выбора чекпойнта ``val/loss`` - общий нормированный
  pinball на всём наборе валидации: одинаковое число окон с каждой
  валидационной станции в валидационном окне, длина истории каждого окна - из того же
  распределения, что при обучении этапа, генератором с фиксированным сидом;
* выбор лучшего чекпойнта по метрике выбора среди валидаций и экспоненциальное
  усреднение весов. Ранней остановки нет: каждый этап идёт ровно заданное число шагов.

Сиды раздельные (``Seeds``): инициализация весов, поток окон, аугментации, подвыборка
при оценке. Не заданный явно сид равен базовому ``seed``; разрешённые значения пишутся
в журнал прогона и в чекпойнт.

Этапы идут подряд в одном запуске, каждый следующий стартует с лучшего по метрике выбора
чекпойнта предыдущего. Подбор скорости обучения проходит сетку и в том же запуске
обучает последний этап полной длины с выбранной скоростью со старта с лучшего чекпойнта
предпоследнего этапа прогона сетки с этой скоростью; при минимуме на краю сетки запуск
завершается ошибкой (``mayak.tuning``).

Различаются только архитектура (``arch``) и выбранная подбором скорость обучения. Что
архитектура определяет сама: регуляризаторы, не зависящие от цели, и отличия от общего
правила весового затухания. Группы затухания с именами параметров и число параметров
записываются в журнал прогона. Отдельных настроек протокола для отдельных архитектур
нет: протокол один на все модели.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

log = logging.getLogger(__name__)

ARCH_NAMES = ("mayak", "gru", "dlinear", "lru", "patchtst")
CURRICULA = ("L0", "full")
LR_SCHEDULES = ("wsd",)
JOURNAL = "protocol.json"
CONFIG_FILE = "config.json"
SEED_FIELDS = ("seed", "seeds")
LR_FIELDS = ("lr",)


class ProtocolError(RuntimeError):
    """Нарушение единого протокола обучения."""


def _jsonable(v):
    return json.loads(json.dumps(v, default=lambda o: asdict(o)))


@dataclass(frozen=True)
class Stage:
    """Этап обучения.

    Куррикулум задаёт распределение длины истории и обучающих окон, и окон валидации,
    поэтому отдельной длины истории для валидации у этапа нет.

    Attributes:
        name: имя этапа.
        curriculum: имя куррикулума длины истории.
        steps: число шагов оптимизатора.
    """
    name: str
    curriculum: str
    steps: int

    def __post_init__(self):
        if self.curriculum not in CURRICULA:
            raise ValueError(f"этап {self.name}: неизвестный куррикулум {self.curriculum!r}")
        if int(self.steps) < 1:
            raise ValueError(f"этап {self.name}: число шагов < 1")


@dataclass(frozen=True)
class Seeds:
    """Раздельные сиды. None - взять базовый сид протокола.

    Attributes:
        init: инициализация весов; каждый следующий этап получает этот сид плюс номер
            этапа.
        data: поток окон: выбор станции, момента и длины истории.
        augment: генератор аугментаций; меняется независимо от потока окон.
        eval: случайность при оценке: бутстрап, выбор примеров для графиков.
    """
    init: Optional[int] = None
    data: Optional[int] = None
    augment: Optional[int] = None
    eval: Optional[int] = None

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                object.__setattr__(self, f.name, int(v))

    def resolve(self, base):
        """Все сиды с подставленным базовым вместо незаданных.

        Args:
            base: базовый сид протокола.

        Returns:
            Словарь: имя сида и целое значение.
        """
        return {f.name: int(base) if getattr(self, f.name) is None else getattr(self, f.name)
                for f in fields(self)}


SEED_NAMES = tuple(f.name for f in fields(Seeds))


@dataclass(frozen=True)
class Protocol:
    """Общий протокол обучения всех архитектур.

    Расписание скорости обучения ``wsd`` (warmup - stable - decay) задаётся полями
    ``warmup_steps`` и ``decay_frac`` и одной формулой для любого этапа, включая
    укороченный последний этап при подборе (``lr_factor``). Поэтому короткий прогон
    подбора совпадает с началом полного этапа до начала спада, а спад у всех моделей
    приходится на одну и ту же долю этапа.
    """
    stages: tuple = (Stage("A", "L0", 10_000), Stage("B", "full", 200_000))
    batch_size: int = 256
    windows_per_epoch: int = 200_000
    num_workers: int = 8
    seed: int = 0
    lr: float = 3e-3
    lr_grid: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    lr_search_steps: int = 20_000
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.95)
    lr_schedule: str = "wsd"
    warmup_steps: int = 1000
    decay_frac: float = 0.2
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    monitor: str = "val/loss"
    val_every: int = 2000
    seeds: Seeds = Seeds()

    def __post_init__(self):
        st = tuple(s if isinstance(s, Stage) else Stage(**s) for s in self.stages)
        object.__setattr__(self, "stages", st)
        if not isinstance(self.seeds, Seeds):
            object.__setattr__(self, "seeds", Seeds(**dict(self.seeds or {})))
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        object.__setattr__(self, "warmup_steps", int(self.warmup_steps))
        object.__setattr__(self, "decay_frac", float(self.decay_frac))
        grid = tuple(sorted(float(v) for v in self.lr_grid))
        object.__setattr__(self, "lr_grid", grid)
        if not grid or len(set(grid)) != len(grid):
            raise ValueError(f"сетка скоростей обучения {grid}: нужна непустая без повторов")
        if not all(math.isfinite(v) and v > 0 for v in grid):
            raise ValueError(f"сетка скоростей обучения {grid}: нужны конечные числа больше нуля")
        if int(self.lr_search_steps) < 1:
            raise ValueError("число шагов подбора скорости обучения < 1")
        if not st:
            raise ValueError("протокол без этапов")
        if len({s.name for s in st}) != len(st):
            raise ValueError("имена этапов повторяются")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"неизвестное расписание {self.lr_schedule!r}")
        if self.warmup_steps < 0:
            raise ValueError(f"число шагов прогрева {self.warmup_steps} < 0")
        if not 0.0 <= self.decay_frac <= 1.0:
            raise ValueError(f"доля спада {self.decay_frac}: нужна от 0 до 1")
        if not str(self.monitor).startswith("val/"):
            raise ValueError(f"метрика выбора {self.monitor!r} не валидационная")

    def to_dict(self):
        """JSON-совместимый словарь."""
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d):
        """Протокол из словаря.

        Args:
            d: словарь с полями протокола.

        Returns:
            Протокол.

        Raises:
            TypeError: в словаре есть поле, которого нет у протокола или этапа.
        """
        return cls(**dict(d))

    def resolved_seeds(self):
        return self.seeds.resolve(self.seed)

    @property
    def total_steps(self):
        return sum(s.steps for s in self.stages)

    def lr_factor(self, step, total_steps):
        """Множитель скорости обучения расписания WSD на шаге этапа.

        ``f(s) = min(1, (s + 1) / W) * min(1, (S - s) / D)``, где ``W = warmup_steps``,
        ``S = total_steps``, ``D = max(1, round(decay_frac * S))``: линейный прогрев до
        пика, плато и линейный спад до нуля к концу этапа. Множитель - произведение
        прогрева и спада, поэтому он определён и для этапа короче ``W + D`` шагов.
        Значение зависит только от номера шага и длины этапа, поэтому у этапов разной
        длины с одним протоколом оно совпадает до начала спада более короткого.

        Args:
            step: номер шага оптимизатора внутри этапа, с нуля.
            total_steps: число шагов этапа.

        Returns:
            Множитель от 0 до 1.
        """
        warmup = max(1, self.warmup_steps)
        decay = max(1, round(self.decay_frac * total_steps))
        return max(0.0, min(1.0, (step + 1) / warmup) * min(1.0, (total_steps - step) / decay))


DEFAULT_PROTOCOL = Protocol()

def strict_fp32():
    """Выключить TF32 в матричных умножениях и свёртках.

    Обучение идёт в fp32, и на GPU с TF32 умножения и свёртки молча считались бы с
    10-битной мантиссой. Свёртки cuDNN по умолчанию TF32 разрешают, поэтому запрет
    ставится явно для обоих путей.
    """
    import torch
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def protocol_for(arch, base=DEFAULT_PROTOCOL):
    """Протокол прогона архитектуры: один и тот же для всех.

    Args:
        arch: имя архитектуры.
        base: общий протокол.

    Returns:
        Тот же общий протокол.

    Raises:
        ProtocolError: неизвестная архитектура.
    """
    if arch not in ARCH_NAMES:
        raise ProtocolError(f"неизвестная архитектура {arch!r}; есть {ARCH_NAMES}")
    return base


def protocol_diff(a, b, ignore=()):
    """Поля, в которых два протокола различаются.

    Args:
        a: первый протокол.
        b: второй протокол.
        ignore: поля, которые не сравниваются.

    Returns:
        Отсортированный список имён полей.
    """
    da, db = ({k: v for k, v in p.to_dict().items() if k not in ignore} for p in (a, b))
    return sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))


def _write_journal(path, journal):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(journal, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_config(path, cfg_dict):
    """Записать полностью разрешённый конфиг прогона в JSON атомарно.

    Args:
        path: путь к файлу.
        cfg_dict: конфиг в виде JSON-совместимого словаря.
    """
    _write_journal(path, cfg_dict)


def run_protocol(arch, manifest=None, protocol=None, out_root="runs", accelerator="auto",
                 tag=None, callbacks=(), enable_progress_bar=True, model_config=None,
                 data_config=None, tuning=None):
    """Обучить архитектуру по протоколу. Единственная функция запуска обучения.

    Этапы идут подряд, каждый следующий стартует с лучшего по метрике выбора чекпойнта
    предыдущего. Каждый этап идёт ровно ``steps`` шагов со своим расписанием WSD, без
    ранней остановки; чекпойнт этапа - лучший по метрике выбора среди всех валидаций.
    Номер этапа в протоколе задаёт его сид инициализации.

    После этапа холодного старта для его лучшего чекпойнта считается отчёт о поле с
    разрывом обобщения по окнам обучающих станций в валидационном окне и строятся
    графики. Сводка отчёта пишется в запись этапа и в запись следующего этапа о том,
    откуда он стартовал.

    При подборе скорости обучения на каждом значении сетки протокола идёт прогон с
    полными этапами, кроме последнего, и укороченным последним в подкаталоге
    ``lr_search`` каталога прогона. Запись о подборе пишется в журнал каталога прогона до
    обучения в нём. При минимуме на краю сетки запуск на этом завершается ошибкой. Иначе
    в каталоге прогона обучается последний этап полной длины со скоростью
    ``selected_lr`` со старта с лучшего чекпойнта предпоследнего этапа прогона сетки с
    этой скоростью. Запись о подборе, своём или взятом из журнала основного прогона,
    пишется в журнал и в каждый чекпойнт.

    Полностью разрешённый конфиг пишется в config.json в каталоге прогона, рядом с каждым
    чекпойнтом и внутрь него.

    Args:
        arch: имя архитектуры.
        manifest: путь к манифесту; если задан, перекрывает путь из конфига данных.
        protocol: общий протокол; None значит протокол по умолчанию.
        out_root: корень каталогов прогонов.
        accelerator: ускоритель обучения.
        tag: имя каталога прогона; None значит имя архитектуры.
        callbacks: дополнительные колбэки обучения.
        enable_progress_bar: показывать ли индикатор хода обучения.
        model_config: конфиг архитектуры, датакласс или словарь; None значит значения
            по умолчанию.
        data_config: конфиг данных, датакласс или словарь; None значит значения по
            умолчанию.
        tuning: подбор скорости обучения, её источник и этап сравнения; None значит
            скорость обучения из протокола без записи о подборе.

    Returns:
        Журнал прогона; он же лежит в protocol.json в каталоге прогона.

    Raises:
        ProtocolError: этап не сохранил ни одного чекпойнта; подбор скорости обучения
            невозможен или его источник не подходит; минимум подбора на краю сетки.
    """
    import pytorch_lightning as L
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from mayak import stage_report as SR
    from mayak import stages as ST
    from mayak import tuning as TU
    from mayak.config import (DataConfig, RunConfig, check_pipeline_compat, model_config_for)
    from mayak.data.datamodule import MayakData, train_stations_set
    from mayak.data.store import get_store
    from mayak.leakage import SELECTION_KEY, run_checklist
    from mayak.lit import (ARCHS, LitForecaster, SelectionProvenance, StageProvenance,
                           TuningProvenance, param_group_summary, parameter_counts)

    tuning = TU.Tuning.coerce(tuning)
    protocol = protocol_for(arch, protocol or DEFAULT_PROTOCOL)
    strict_fp32()
    last_index = len(protocol.stages) - 1
    model_cfg = check_pipeline_compat(model_config_for(arch, model_config))

    tag = tag or arch
    run_dir = os.path.join(out_root, tag)
    journal_path = os.path.join(run_dir, JOURNAL)

    def train_grid(p, sub_tag):
        j = run_protocol(arch, manifest, p, out_root=out_root, accelerator=accelerator,
                         tag=sub_tag, callbacks=callbacks,
                         enable_progress_bar=enable_progress_bar, model_config=model_config,
                         data_config=data_config)
        return j, os.path.abspath(os.path.join(out_root, sub_tag, JOURNAL))

    TU.check_run_dir(journal_path, tuning)
    protocol, tune = TU.resolve(arch, protocol, tuning, model_cfg, tag, train_grid)
    for line in TU.format_tuning(tune):
        log.info("%s: %s", arch, line)
    if data_config is None:
        data_cfg = DataConfig()
    elif isinstance(data_config, DataConfig):
        data_cfg = data_config
    else:
        data_cfg = DataConfig.from_dict(data_config)
    if manifest is not None:
        data_cfg = replace(data_cfg, manifest=str(manifest))
    run_cfg = RunConfig(model=model_cfg, data=data_cfg, train=protocol).resolved()
    data_cfg = run_cfg.data
    manifest = data_cfg.manifest
    seeds = protocol.resolved_seeds()
    cfg_dict = run_cfg.to_dict()
    device = SR.report_device(accelerator)
    store = get_store(manifest, cache_root=data_cfg.cache_root)

    base = dict(arch=arch, model_class=f"{ARCHS[arch].__module__}.{ARCHS[arch].__qualname__}",
                protocol=protocol.to_dict(),
                manifest=os.path.abspath(manifest), seeds=seeds, config_file=CONFIG_FILE,
                augment=data_cfg.augment.summary(), data_key=store.key, tuning=tune)
    journal = dict(base, stages=[], final_ckpt=None)
    write_config(os.path.join(run_dir, CONFIG_FILE), cfg_dict)
    _write_journal(journal_path, journal)

    first, prev = 0, None
    if tuning.lr_search:
        search = tune["lr_search"]
        if search["edge"]:
            raise ProtocolError(f"{arch}: {TU.edge_error(search)}")
        first = last_index
        prev_name = protocol.stages[first - 1].name
        init_ckpt, grid_journal = TU.search_start(search, prev_name)
        grid_entry = ST.journal_stage(grid_journal, prev_name)
        prev = dict(SR.describe_checkpoint(init_ckpt), report_file=grid_entry.get("report"),
                    report=grid_entry.get("report_summary"))
        log.info("%s: этап %s стартует с %s (этап %s, шаг %s, отпечаток %s)", arch,
                 protocol.stages[first].name, prev["ckpt"], prev["stage"], prev["step"],
                 prev["digest"])
    log.info("%s: сиды %s", arch, seeds)
    log.info("%s: аугментации %s", arch, journal["augment"])

    for i in range(first, len(protocol.stages)):
        stage = protocol.stages[i]
        # Сид этапа зависит от его номера в протоколе.
        L.seed_everything(seeds["init"] + i, workers=False, verbose=False)
        dm = MayakData(manifest=manifest, curriculum=stage.curriculum,
                       batch_size=protocol.batch_size, windows_per_epoch=protocol.windows_per_epoch,
                       num_workers=protocol.num_workers, seed=seeds["data"],
                       aug_seed=seeds["augment"], data_config=data_cfg.to_dict())
        dm.setup("fit")
        run_checklist(dm.store, datasets=[dm.train_ds, dm.val_ds])
        log.info("%s: этап %s, валидация %d окон на %d станциях, длины истории %s, "
                 "отпечаток %s", arch, stage.name, len(dm.val_ds), len(dm.val_ds.floor),
                 dm.val_ds.history_spec(), dm.val_ds.fingerprint())

        lit = LitForecaster(arch=arch, protocol=protocol.to_dict(), stage=stage.name,
                            total_steps=stage.steps, model_config=model_cfg.to_dict(),
                            data_config=data_cfg.to_dict())
        if prev is not None:
            ST.load_init_weights(lit, prev["ckpt"])
        if i == first:
            journal["param_groups"] = param_group_summary(lit.model, protocol.weight_decay)
            counts = parameter_counts(lit.model)
            journal["n_params"] = counts["total"]
            journal["n_params_by_module"] = counts["by_module"]
            log.info("%s: параметров %d (%s)", arch, counts["total"],
                     ", ".join(f"{k} {v}" for k, v in counts["by_module"].items()))

        stage_dir = os.path.join(run_dir, ST.stage_dir_name(stage.name))
        ST.clear_stale(stage_dir)
        write_config(os.path.join(stage_dir, CONFIG_FILE), cfg_dict)
        record = ST.stage_record(stage, i, data_key=store.key, run_dir=run_dir,
                                 journal=journal_path, init_from=ST.lineage(prev))
        # Одно имя чекпойнта этапа при любом числе запусков: путь stage<X>/best.ckpt
        # всегда указывает на веса последнего запуска.
        ckpt = ModelCheckpoint(dirpath=stage_dir, monitor=protocol.monitor, mode="min",
                               save_top_k=1, filename="best", enable_version_counter=False)
        stage_callbacks = [ckpt, SelectionProvenance(), StageProvenance(record)]
        if tune is not None:
            stage_callbacks.append(TuningProvenance(tune))
        logger = CSVLogger(out_root, name=f"{tag}/{ST.stage_dir_name(stage.name)}")
        trainer = L.Trainer(
            max_steps=stage.steps, accelerator=accelerator, devices=1,
            precision="32-true", gradient_clip_val=protocol.grad_clip,
            val_check_interval=min(protocol.val_every, stage.steps), check_val_every_n_epoch=None,
            # Проход валидации идёт по всему набору: размер набора задаётся числом
            # окон на станцию, обрезка по батчам выбросила бы последние станции.
            limit_val_batches=1.0,
            logger=logger, log_every_n_steps=20,
            callbacks=[*stage_callbacks, LearningRateMonitor("step"), *callbacks],
            enable_progress_bar=enable_progress_bar, enable_model_summary=False)
        trainer.fit(lit, datamodule=dm)
        if not ckpt.best_model_path:
            raise ProtocolError(f"{arch}: этап {stage.name} не сохранил чекпойнт — "
                                f"валидация ни разу не прошла")
        best_path = ckpt.best_model_path
        best = SR.describe_checkpoint(best_path)
        score = ckpt.best_model_score
        chosen = torch.load(best_path, map_location="cpu", weights_only=False)
        selection = (chosen.get(SELECTION_KEY) or {}).get("scores", {})
        entry = dict(name=stage.name, curriculum=stage.curriculum,
                     steps_done=int(trainer.global_step), best_ckpt=best_path,
                     best_score=None if score is None else float(score),
                     val_windows=len(dm.val_ds), val_set=dm.val_ds.fingerprint(),
                     selection=selection, best_step=best["step"], best_digest=best["digest"],
                     metrics_csv=os.path.join(logger.log_dir, "metrics.csv"), init_from=prev)
        report_summary = report_path = None
        if stage.curriculum == ST.FIELD_CURRICULUM:
            gap_ds = train_stations_set(dm.store, manifest, data_cfg, stage.curriculum)
            run_checklist(dm.store, datasets=[gap_ds])
            report, report_path, plots = SR.write_stage_report(
                stage_dir, stage.name, arch, best, dm.val_ds, train_dataset=gap_ds,
                device=device, seed=seeds["eval"], metrics_csv=entry["metrics_csv"])
            report_summary = SR.summary(report["best"])
            entry.update(report=report_path, report_summary=report_summary, plots=plots)
            for line in SR.format_field_report(report):
                log.info("%s", line)
        journal["stages"].append(ST.jsonable(entry))
        journal["final_ckpt"] = best_path if i == last_index else None
        _write_journal(journal_path, journal)
        prev = dict(best, report_file=report_path, report=report_summary)
    return journal


def run_experiment(cfg, out_root="runs", tag=None, accelerator="auto", callbacks=(),
                   enable_progress_bar=True, tuning=None):
    """Прогон по полной конфигурации: точка входа слоя композиции конфигов.

    Args:
        cfg: полная конфигурация прогона или словарь той же формы.
        out_root: корень каталогов прогонов.
        tag: имя каталога прогона.
        accelerator: ускоритель обучения.
        callbacks: дополнительные колбэки обучения.
        enable_progress_bar: показывать ли индикатор хода обучения.
        tuning: подбор скорости обучения и этап сравнения; None значит без подбора.

    Returns:
        Журнал прогона.
    """
    from mayak.config import RunConfig
    if not isinstance(cfg, RunConfig):
        cfg = RunConfig.from_dict(cfg)
    return run_protocol(cfg.arch, cfg.data.manifest, cfg.train, out_root=out_root, tag=tag,
                        accelerator=accelerator, callbacks=callbacks,
                        enable_progress_bar=enable_progress_bar, model_config=cfg.model,
                        data_config=cfg.data, tuning=tuning)


__all__ = ["ARCH_NAMES", "CONFIG_FILE", "DEFAULT_PROTOCOL", "JOURNAL", "LR_FIELDS", "Protocol",
           "ProtocolError", "SEED_FIELDS", "SEED_NAMES", "Seeds", "Stage", "protocol_diff",
           "protocol_for", "run_experiment", "run_protocol", "strict_fp32"]
