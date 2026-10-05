"""Единый протокол обучения.

Протокол описан в одном месте датаклассом Protocol и исполняется одной функцией
run_protocol. Для любой архитектуры одинаковы:

* этапы обучения и число шагов каждого: этап A только без истории, этап B - полный
  куррикулум длины истории;
* размер батча, число окон в эпохе, число воркеров и сид, то есть поток окон;
* оптимизатор AdamW (lr, betas, базовое весовое затухание), косинусное расписание,
  обрезка градиента, точность вычислений: fp32 без TF32, та же, в которой модель
  оценивают, калибруют и экспортируют;
* функция потерь и метрика выбора чекпойнта ``val/loss`` - общий нормированный
  pinball на всём наборе валидации: одинаковое число окон с каждой
  валидационной станции в валидационном окне, длина истории каждого окна - из того же
  распределения, что при обучении этапа, генератором с фиксированным сидом;
* ранняя остановка, выбор лучшего чекпойнта и экспоненциальное усреднение весов.

Сиды раздельные (``Seeds``): инициализация весов, поток окон, аугментации, подвыборка
при оценке. Не заданный явно сид равен базовому ``seed``; разрешённые значения пишутся
в журнал прогона и в чекпойнт.

Этапы можно запускать по отдельности: этап A, потом, после просмотра отчёта о поле,
этап B с выбранного человеком чекпойнта этапа A. Какие этапы запускать, с какого
чекпойнта стартовать, порог ворот, пробный запуск и сохранение кандидатов в протокол не
входят: этап B, запущенный отдельной командой с лучшего чекпойнта A, получает тот же
протокол и тот же сид, что в прогоне одной командой, и даёт тот же чекпойнт.

Различается только архитектура (``arch``). Что архитектура определяет сама:
регуляризаторы, не зависящие от цели, и группы весового затухания. Группы и число
параметров записываются в журнал прогона. Отдельных настроек протокола для отдельных
архитектур нет: протокол один на все модели.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

log = logging.getLogger(__name__)

ARCH_NAMES = ("mayak", "gru", "dlinear", "lru", "patchtst")
CURRICULA = ("L0", "full")
LR_SCHEDULES = ("cosine",)
JOURNAL = "protocol.json"
CONFIG_FILE = "config.json"


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
    stages: tuple = (Stage("A", "L0", 10_000), Stage("B", "full", 200_000))
    batch_size: int = 256
    windows_per_epoch: int = 200_000
    num_workers: int = 8
    seed: int = 0
    lr: float = 3e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.95)
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    monitor: str = "val/loss"
    val_every: int = 2000
    patience: int = 5
    seeds: Seeds = Seeds()

    def __post_init__(self):
        st = tuple(s if isinstance(s, Stage) else Stage(**s) for s in self.stages)
        object.__setattr__(self, "stages", st)
        if not isinstance(self.seeds, Seeds):
            object.__setattr__(self, "seeds", Seeds(**dict(self.seeds or {})))
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


# (флаг, поле протокола, тип)
_CLI = [
    ("--batch", "batch_size", int), ("--windows", "windows_per_epoch", int),
    ("--workers", "num_workers", int), ("--seed", "seed", int),
    ("--lr", "lr", float), ("--weight-decay", "weight_decay", float),
    ("--grad-clip", "grad_clip", float), ("--ema-decay", "ema_decay", float),
    ("--val-every", "val_every", int),
    ("--patience", "patience", int),
]


def add_protocol_args(ap, base=DEFAULT_PROTOCOL):
    """Добавить флаги протокола в разбор командной строки.

    Флаги одни и те же для всех архитектур и всех точек входа обучения.

    Args:
        ap: разборщик аргументов командной строки.
        base: протокол, значения которого становятся значениями флагов по умолчанию.

    Returns:
        Тот же разборщик.
    """
    g = ap.add_argument_group("протокол обучения (одинаков для всех архитектур)")
    for s in base.stages:
        g.add_argument(f"--steps-{s.name.lower()}", type=int, default=s.steps,
                       help=f"шаги этапа {s.name} ({s.curriculum})")
    for flag, name, typ in _CLI:
        g.add_argument(flag, type=typ, default=getattr(base, name))
    for name in SEED_NAMES:
        g.add_argument(f"--seed-{name}", type=int, default=getattr(base.seeds, name),
                       help=f"отдельный сид «{name}» (по умолчанию = --seed)")
    return ap


def protocol_from_args(args, base=DEFAULT_PROTOCOL):
    stages = tuple(replace(s, steps=getattr(args, f"steps_{s.name.lower()}")) for s in base.stages)
    kw = {name: getattr(args, flag.lstrip("-").replace("-", "_")) for flag, name, _ in _CLI}
    seeds = Seeds(**{n: getattr(args, f"seed_{n}") for n in SEED_NAMES})
    return replace(base, stages=stages, seeds=seeds, **kw)


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
                 data_config=None, launch=None):
    """Обучить архитектуру по протоколу. Единственная функция запуска обучения.

    Этап, который в протоколе идёт не первым, стартует с лучшего чекпойнта предыдущего
    этапа, если этапы идут одной командой, или с явно указанного чекпойнта, если этап
    запущен отдельно. Номер этапа в протоколе задаёт его сид инициализации, поэтому этап,
    запущенный отдельно, повторяет тот же этап прогона одной командой.

    После этапа холодного старта считается отчёт о поле по всем сохранённым чекпойнтам
    этапа и строятся графики. Если задан порог ворот, следующий этап не начинается, когда
    у стартового чекпойнта отношение MSE выше порога.

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
        launch: какие этапы запустить, с какого чекпойнта стартует первый из них, порог
            ворот, пробный запуск и сохранение кандидатов; None значит все этапы одной
            командой.

    Returns:
        Журнал прогона; он же лежит в protocol.json в каталоге прогона.

    Raises:
        ProtocolError: неверный выбор этапов или этап не сохранил ни одного чекпойнта.
        InitCheckpointError: чекпойнт инициализации не подходит этому запуску.
        GateError: поле стартового чекпойнта не прошло порог, следующий этап не начат.
    """
    import pytorch_lightning as L
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from mayak import stage_report as SR
    from mayak import stages as ST
    from mayak.config import (DataConfig, RunConfig, check_pipeline_compat, model_config_for)
    from mayak.data.datamodule import MayakData, validation_set
    from mayak.data.store import get_store
    from mayak.leakage import SELECTION_KEY, run_checklist
    from mayak.lit import (ARCHS, CandidateCheckpoint, LitForecaster, SelectionProvenance,
                           StageProvenance, param_group_summary, parameter_counts)

    launch = ST.Launch.coerce(launch)
    protocol = protocol_for(arch, protocol or DEFAULT_PROTOCOL)
    strict_fp32()
    plan = ST.plan_stages(protocol, launch)
    first = plan[0][0]
    last_index = len(protocol.stages) - 1
    keep_candidates = ST.candidate_stages(protocol, launch)
    model_cfg = check_pipeline_compat(model_config_for(arch, model_config))
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

    tag = tag or arch
    run_dir = os.path.join(out_root, tag)
    journal_path = os.path.join(run_dir, JOURNAL)
    store = get_store(manifest, cache_root=data_cfg.cache_root)

    prev = None
    if first > 0:
        prev_stage = protocol.stages[first - 1]
        prev_val = validation_set(store, manifest, data_cfg, prev_stage.curriculum)
        prev = ST.inspect_init_checkpoint(launch.init_from, arch=arch, protocol=protocol,
                                          model_config=model_cfg, data_config=data_cfg,
                                          stage=prev_stage, store=store,
                                          val_digest=prev_val.fingerprint())
        if prev_stage.curriculum == ST.FIELD_CURRICULUM:
            entry, report_file = SR.find_report_entry(prev["run_dir"], prev_stage.name,
                                                      prev["digest"], prev_val.fingerprint())
            if entry is None and launch.require_gate is not None:
                # Отчёта для этого файла нет: чекпойнт перенесён или изменён. Для решения
                # ворот он считается заново на том же наборе валидации.
                items = SR.report_items(SR.describe_checkpoint(prev["ckpt"]), [])
                report, _ = SR.build_field_report(items, prev_val, arch=arch,
                                                  stage=prev_stage.name, device=device,
                                                  seed=seeds["eval"], n_examples=0)
                entry, report_file = report["candidates"][0], None
            prev.update(report=entry, report_file=report_file)
        log.info("%s: этап %s стартует с %s (этап %s, шаг %s, отпечаток %s)", arch,
                 protocol.stages[first].name, prev["ckpt"], prev["stage"], prev["step"],
                 prev["digest"])

    base = dict(arch=arch, model_class=f"{ARCHS[arch].__module__}.{ARCHS[arch].__qualname__}",
                protocol=protocol.to_dict(),
                manifest=os.path.abspath(manifest), seeds=seeds, config_file=CONFIG_FILE,
                augment=data_cfg.augment.summary(), data_key=store.key)
    journal = ST.start_journal(journal_path, base, protocol, first, prev, run_dir)
    log.info("%s: сиды %s", arch, seeds)
    log.info("%s: аугментации %s", arch, journal["augment"])

    for i, stage in plan:
        if prev is not None and launch.require_gate is not None:
            if prev.get("report") is None:
                log.warning("%s: у этапа %s нет отчёта о поле, ворота перед этапом %s не "
                            "проверяются", arch, prev["stage"], stage.name)
            else:
                prev["gate"] = ST.gate_verdict(prev["report"], launch.require_gate)
                if not prev["gate"]["passed"]:
                    if i != first:
                        _write_journal(journal_path, journal)
                    raise ST.GateError(ST.gate_message(stage.name, prev))
        if i == first:
            write_config(os.path.join(run_dir, CONFIG_FILE), cfg_dict)
        # Сид этапа зависит от его номера в протоколе, а не от номера в этом запуске.
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
        ST.warn_stale(stage_dir)
        write_config(os.path.join(stage_dir, CONFIG_FILE), cfg_dict)
        probe = launch.probe_steps if i == plan[-1][0] else None
        record = ST.stage_record(stage, i, data_key=store.key, run_dir=run_dir,
                                 journal=journal_path, init_from=ST.lineage(prev),
                                 probe_steps=probe)
        ckpt = ModelCheckpoint(dirpath=stage_dir, monitor=protocol.monitor, mode="min",
                               save_top_k=1, filename="best")
        stage_callbacks = [ckpt, SelectionProvenance(), StageProvenance(record)]
        cands = None
        if stage.name in keep_candidates:
            cands = CandidateCheckpoint(dirpath=os.path.join(stage_dir, ST.CANDIDATE_DIR),
                                        monitor=protocol.monitor, mode="min", save_top_k=-1,
                                        filename="step{step:06d}", auto_insert_metric_name=False,
                                        save_on_train_epoch_end=False)
            stage_callbacks.append(cands)
        logger = CSVLogger(out_root, name=f"{tag}/{ST.stage_dir_name(stage.name)}")
        trainer = L.Trainer(
            # Пробный запуск короче этапа, но расписание скорости обучения и частота
            # валидации у него полные, поэтому он совпадает с началом полного этапа.
            max_steps=probe or stage.steps, accelerator=accelerator, devices=1,
            precision="32-true", gradient_clip_val=protocol.grad_clip,
            val_check_interval=min(protocol.val_every, stage.steps), check_val_every_n_epoch=None,
            # Проход валидации идёт по всему набору: размер набора задаётся числом
            # окон на станцию, обрезка по батчам выбросила бы последние станции.
            limit_val_batches=1.0,
            logger=logger, log_every_n_steps=20,
            callbacks=[*stage_callbacks, LearningRateMonitor("step"),
                       EarlyStopping(monitor=protocol.monitor, patience=protocol.patience,
                                     mode="min"), *callbacks],
            enable_progress_bar=enable_progress_bar, enable_model_summary=False)
        trainer.fit(lit, datamodule=dm)
        if not ckpt.best_model_path:
            raise ProtocolError(f"{arch}: этап {stage.name} не сохранил чекпойнт — "
                                f"валидация ни разу не прошла")
        best_path = ckpt.best_model_path
        best = SR.describe_checkpoint(best_path)
        candidates = sorted((SR.describe_checkpoint(p) for p in (cands.best_k_models if cands
                                                                  else {})),
                            key=lambda c: c["step"])
        score = ckpt.best_model_score
        chosen = torch.load(best_path, map_location="cpu", weights_only=False)
        selection = (chosen.get(SELECTION_KEY) or {}).get("scores", {})
        entry = dict(name=stage.name, curriculum=stage.curriculum,
                     steps_done=int(trainer.global_step), best_ckpt=best_path,
                     best_score=None if score is None else float(score),
                     val_windows=len(dm.val_ds), val_set=dm.val_ds.fingerprint(),
                     selection=selection, best_step=best["step"], best_digest=best["digest"],
                     candidates=[{k: c[k] for k in ("step", "ckpt", "digest", "val_loss")}
                                 for c in candidates],
                     metrics_csv=os.path.join(logger.log_dir, "metrics.csv"),
                     init_from=ST.init_summary(prev, SR.summary((prev or {}).get("report"))),
                     probe_steps=probe)
        report_entry = report_path = None
        if stage.curriculum == ST.FIELD_CURRICULUM:
            report, report_path, plots = SR.write_stage_report(
                stage_dir, stage.name, arch, best, candidates, dm.val_ds, device=device,
                seed=seeds["eval"], threshold=launch.require_gate,
                metrics_csv=entry["metrics_csv"])
            report_entry = next(e for e in report["candidates"] if e.get("is_best"))
            entry.update(report=report_path, report_best=SR.summary(report_entry), plots=plots)
            for line in SR.format_field_report(report):
                log.info("%s", line)
        journal["stages"].append(ST.jsonable(entry))
        journal["final_ckpt"] = best_path if i == last_index and probe is None else None
        _write_journal(journal_path, journal)
        prev = dict(ckpt=best_path, digest=best["digest"], stage=stage.name, step=best["step"],
                    run_dir=os.path.abspath(run_dir), journal=os.path.abspath(journal_path),
                    is_best=True, warnings=[], report=report_entry, report_file=report_path)
    return journal


def run_experiment(cfg, out_root="runs", tag=None, accelerator="auto", callbacks=(),
                   enable_progress_bar=True, launch=None):
    """Прогон по полной конфигурации: точка входа слоя композиции конфигов.

    Args:
        cfg: полная конфигурация прогона или словарь той же формы.
        out_root: корень каталогов прогонов.
        tag: имя каталога прогона.
        accelerator: ускоритель обучения.
        callbacks: дополнительные колбэки обучения.
        enable_progress_bar: показывать ли индикатор хода обучения.
        launch: какие этапы запустить и с какого чекпойнта; None значит все этапы.

    Returns:
        Журнал прогона.
    """
    from mayak.config import RunConfig
    if not isinstance(cfg, RunConfig):
        cfg = RunConfig.from_dict(cfg)
    return run_protocol(cfg.arch, cfg.data.manifest, cfg.train, out_root=out_root, tag=tag,
                        accelerator=accelerator, callbacks=callbacks,
                        enable_progress_bar=enable_progress_bar, model_config=cfg.model,
                        data_config=cfg.data, launch=launch)


def read_journal(run_dir):
    with open(os.path.join(run_dir, JOURNAL)) as f:
        return json.load(f)


__all__ = ["ARCH_NAMES", "CONFIG_FILE", "DEFAULT_PROTOCOL", "Protocol", "ProtocolError",
           "SEED_NAMES", "Seeds", "Stage", "add_protocol_args", "protocol_for",
           "protocol_from_args", "read_journal", "run_experiment", "run_protocol",
           "strict_fp32"]
