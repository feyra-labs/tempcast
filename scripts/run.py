r"""Обучение: единственный вход. Композиция конфигов Hydra, переопределения, групповые запуски.

Значения прогона берутся из YAML в conf/ и переопределений командной строки, полностью
разрешённый конфиг пишется в каталог прогона и в каждый чекпойнт. Каталог прогона -
``<run.out_root>/<run.tag>``, у одиночного и группового запуска он один и тот же. Тег:
``<arch>[-<абляция>][-aug_<профиль>][-tuned][-s<сид>]``, части со значением по умолчанию
опускаются.

Поток этапа 1 сравнения одинаков для всех моделей: подбор скорости обучения (только
подбор, без полного прогона), выбор человеком скорости из сетки и кандидата этапа A из
прогона сетки, затем этап B отдельным запуском в том же каталоге. После каждого шага
печатается готовая команда следующего.

Example:
    python scripts/run.py train=debug run.accelerator=cpu      # отладка на CPU: runs/mayak
    python scripts/run.py run.lr_search=true                   # подбор lr: runs/mayak/lr_search
    python scripts/run.py -m model=gru,dlinear,lru,patchtst run.lr_search=true
    python scripts/run.py run.stages=[B] \
        run.init_from=runs/mayak/lr_search/lr0.001/stageA/candidates/step006000.ckpt
    python scripts/run.py run.stages=[B] train.lr=0.003 \
        run.init_from=runs/mayak/lr_search/lr0.003/stageA/best.ckpt   # другое значение сетки
    python scripts/run.py ablation=no_solar run.lr_from=runs/mayak run.stages=[A]
    python scripts/run.py ablation=no_solar run.lr_from=runs/mayak run.stages=[B] \
        run.init_from=runs/mayak-no_solar/stageA/candidates/step006000.ckpt
    python scripts/run.py train.seed=1 run.lr_from=runs/mayak run.stages=[A]  # runs/mayak-s1
    python scripts/run.py augment=none run.lr_from=runs/mayak run.stages=[A]  # mayak-aug_none
    python scripts/run.py run.extra_tuning=true train.lr=0.001 model.encoder_width=64
"""
import logging
import os
import shlex

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

RUN_SECTIONS = ("model", "data", "train")
LR_OVERRIDE = "train.lr"
# Значения, которые в теге прогона не пишутся.
BASE_ABLATION, BASE_AUGMENT, BASE_SEED = "none", "default", 0


def run_tag(arch, ablation, profile, extra, seed):
    """Тег прогона, он же имя его каталога.

    Схема ``<arch>[-<абляция>][-aug_<профиль>][-tuned][-s<сид>]``: абляция ``none``,
    профиль аугментаций ``default``, прогон без дополнительной настройки и сид 0 в теге
    не пишутся.

    Args:
        arch: имя архитектуры.
        ablation: вариант группы ablation.
        profile: имя профиля аугментаций.
        extra: прогон дополнительной настройки МАЯК (этап 2 сравнения).
        seed: базовый сид протокола.

    Returns:
        Строка вида ``mayak``, ``mayak-no_solar``, ``mayak-aug_none``, ``mayak-s1``,
        ``mayak-tuned``, ``gru``.
    """
    from mayak.tuning import EXTRA_SUFFIX
    tag = str(arch)
    if ablation is not None and str(ablation) != BASE_ABLATION:
        tag += f"-{ablation}"
    if str(profile) != BASE_AUGMENT:
        tag += f"-aug_{profile}"
    if str(extra).lower() == "true":
        tag += EXTRA_SUFFIX
    if int(seed) != BASE_SEED:
        tag += f"-s{int(seed)}"
    return tag


OmegaConf.register_new_resolver("run_tag", run_tag, replace=True)


def to_run_config(cfg):
    """Полная конфигурация прогона из конфига Hydra.

    Берутся только секции модели, данных и обучения, все подстановки разрешены.

    Args:
        cfg: конфиг Hydra.

    Returns:
        Полная конфигурация прогона.
    """
    from mayak.config import RunConfig
    d = {k: OmegaConf.to_container(cfg[k], resolve=True) for k in RUN_SECTIONS}
    if d["model"].get("arch") != "mayak" and not d["model"].get("ablations", True):
        d["model"].pop("ablations")
    return RunConfig.from_dict(d)


def to_launch(cfg):
    """Настройки раздельного запуска этапов из секции run.

    Интерполяции секции не разрешаются: имя прогона зависит от выбора Hydra, а
    настройкам запуска оно не нужно.

    Args:
        cfg: конфиг Hydra.

    Returns:
        Настройки запуска.
    """
    from mayak.stages import launch_from_config
    return launch_from_config(OmegaConf.to_container(cfg.run, resolve=False))


def override_key(override):
    """Ключ переопределения командной строки Hydra без префикса и значения.

    Args:
        override: строка вида ``train.lr=0.001`` или ``+model.x=1``.

    Returns:
        Ключ, например ``train.lr``.
    """
    return override.lstrip("+~").split("=", 1)[0]


def to_tuning(cfg, overrides=()):
    """Настройки подбора скорости обучения из секции run.

    Явной считается скорость обучения, заданная в командной строке (``train.lr=...``):
    значение из YAML человек не выбирал, оно одинаково у всех прогонов.

    Args:
        cfg: конфиг Hydra.
        overrides: переопределения командной строки этого запуска.

    Returns:
        Настройки подбора.
    """
    from mayak.tuning import tuning_from_config
    section = OmegaConf.to_container(cfg.run, resolve=False)
    section["lr_explicit"] = any(override_key(o) == LR_OVERRIDE for o in overrides)
    return tuning_from_config(section)


def next_command(overrides, stage, init_from, drop=()):
    """Команда запуска следующего этапа в том же каталоге прогона.

    Переопределения настроек запуска (``run.stages``, ``run.init_from`` и остальные поля
    ``Launch``) описывают только этот запуск и не переносятся. Остальные переносятся без
    изменений, поэтому у следующего этапа те же конфиг, тег и запись о подборе.

    Args:
        overrides: переопределения командной строки этого запуска.
        stage: имя следующего этапа.
        init_from: чекпойнт, с которого он стартует.
        drop: ещё ключи, которые не переносятся, например ``run.lr_search``.

    Returns:
        Строка команды для оболочки.
    """
    from mayak.stages import LAUNCH_FIELDS
    skip = {f"run.{k}" for k in LAUNCH_FIELDS} | set(drop)
    kept = [o for o in overrides if override_key(o) not in skip]
    return shlex.join(["python", "scripts/run.py", *kept, f"run.stages=[{stage}]",
                       f"run.init_from={init_from}"])


def print_search_step(search, stage_names, overrides):
    """Печатает итог подбора и готовую команду последнего этапа.

    Args:
        search: запись о подборе по сетке из журнала.
        stage_names: имена этапов протокола по порядку.
        overrides: переопределения командной строки этого запуска.
    """
    from mayak.stage_report import REPORT_FILE
    from mayak.stages import CANDIDATE_DIR
    from mayak.tuning import SEARCH_DIR, search_lr
    prev, nxt = stage_names[-2], stage_names[-1]
    lr = search_lr(search)
    best = next(r for r in search["results"] if r["lr"] == lr)["best_ckpts"][prev]
    stage_dir = os.path.dirname(best)
    print(f"\nПодбор скорости обучения готов: минимум {search['monitor']} на этапе {nxt} при "
          f"lr {lr:g}. Полного прогона нет, этап {prev} заново не обучается.")
    if search.get("edge"):
        print("ВНИМАНИЕ: минимум на краю сетки, оптимум может лежать за её пределами.")
    print(f"Отчёт этапа {prev}: {os.path.join(stage_dir, REPORT_FILE)}")
    print(f"Выберите чекпойнт для этапа {nxt}: лучший по val/loss (в команде ниже) или любой "
          f"кандидат из {os.path.join(stage_dir, CANDIDATE_DIR)}.")
    print(f"Другое значение сетки: добавьте {LR_OVERRIDE}=<значение> и возьмите чекпойнт из "
          f"{SEARCH_DIR}/lr<значение>/stage{prev}/.")
    print(f"Этап {nxt}:")
    print("    " + next_command(overrides, nxt, best, drop=("run.lr_search",)))


def print_next_step(journal, stage_names, overrides):
    """Печатает путь к отчёту и готовую команду следующего этапа.

    Подсказка нужна, только если запуск остановился до итогового чекпойнта: чекпойнт для
    следующего этапа выбирает человек по отчёту. После подбора скорости обучения записей
    об этапах в журнале нет, и печатается итог подбора.

    Args:
        journal: журнал прогона.
        stage_names: имена этапов протокола по порядку.
        overrides: переопределения командной строки этого запуска.
    """
    from mayak.stages import CANDIDATE_DIR
    if not journal["stages"]:
        search = (journal.get("tuning") or {}).get("lr_search")
        if search:
            print_search_step(search, stage_names, overrides)
        return
    last = journal["stages"][-1]
    idx = stage_names.index(last["name"])
    if journal["final_ckpt"] or idx + 1 >= len(stage_names):
        return
    nxt = stage_names[idx + 1]
    cand_dir = os.path.join(os.path.dirname(last["best_ckpt"]), CANDIDATE_DIR)
    print(f"\nЭтап {last['name']} готов.")
    if last.get("report"):
        print(f"Отчёт: {last['report']}")
    print(f"Выберите чекпойнт для этапа {nxt}: лучший по val/loss (в команде ниже) или любой "
          f"кандидат из {cand_dir}.")
    print(f"Этап {nxt}:")
    print("    " + next_command(overrides, nxt, last["best_ckpt"]))


def exit_code(err):
    """Код выхода процесса при ошибке протокола.

    Args:
        err: исключение протокола.

    Returns:
        Отдельный код для неподходящего чекпойнта инициализации, иначе 1.
    """
    from mayak.stages import INIT_EXIT_CODE, InitCheckpointError
    if isinstance(err, InitCheckpointError):
        return INIT_EXIT_CODE
    return 1


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    import sys

    from hydra.core.hydra_config import HydraConfig

    from mayak.protocol import ProtocolError, run_experiment
    from mayak.tuning import format_tuning

    hc = HydraConfig.get()
    overrides = list(hc.overrides.task)
    rc = to_run_config(cfg)
    launch = to_launch(cfg)
    tuning = to_tuning(cfg, overrides)
    out_dir = hc.runtime.output_dir
    try:
        journal = run_experiment(rc, out_root=os.path.dirname(out_dir),
                                 tag=os.path.basename(out_dir), accelerator=cfg.run.accelerator,
                                 launch=launch, tuning=tuning)
    except ProtocolError as e:
        log.error("%s", e)
        sys.exit(exit_code(e))
    for line in format_tuning(journal.get("tuning")):
        log.info("%s", line)
    for st in journal["stages"]:
        log.info("лучшая модель этапа %s: %s", st["name"], st["best_ckpt"])
        if st.get("report"):
            log.info("отчёт о поле этапа %s: %s", st["name"], st["report"])
    if journal["final_ckpt"]:
        log.info("итоговый чекпойнт: %s", journal["final_ckpt"])
    print_next_step(journal, [s.name for s in rc.train.stages], overrides)
    return journal["stages"][-1]["best_score"] if journal["stages"] else None


if __name__ == "__main__":
    main()
