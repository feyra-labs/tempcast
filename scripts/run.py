r"""Обучение: единственный вход. Композиция конфигов Hydra, переопределения, групповые запуски.

Значения прогона берутся из YAML в conf/ и переопределений командной строки, полностью
разрешённый конфиг пишется в каталог прогона и в каждый чекпойнт. Каталог прогона -
``<run.out_root>/<run.tag>``, у одиночного и группового запуска он один и тот же. Тег:
``<arch>[-<абляция>][-aug_<профиль>][-tuned][-s<сид>]``, части со значением по умолчанию
опускаются.

Одна команда на модель. ``run.lr_search=true`` проходит сетку скоростей обучения, пишет
запись о подборе в журнал и в том же запуске обучает этап B полной длины; при минимуме
на краю сетки запуск завершается с кодом 1. ``run.lr_from`` берёт скорость обучения из
журнала основного МАЯК, этапы A и B идут подряд. Ошибка протокола в групповом запуске
останавливает только свой прогон: остальные прогоны группы идут до конца, код 1 - после
всей группы.

Example:
    python scripts/run.py train=debug run.accelerator=cpu      # отладка на CPU: runs/mayak
    python scripts/run.py run.lr_search=true                   # подбор lr и этап B: runs/mayak
    python scripts/run.py -m model=gru,dlinear,lru,patchtst run.lr_search=true
    python scripts/run.py -m run.lr_from=runs/mayak ablation=no_solar,no_passport
    python scripts/run.py -m run.lr_from=runs/mayak train.seed=1,2   # runs/mayak-s1, -s2
    python scripts/run.py run.lr_from=runs/mayak augment=none        # runs/mayak-aug_none
    python scripts/run.py run.extra_tuning=true run.lr_search=true model.encoder_width=96
"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

RUN_SECTIONS = ("model", "data", "train")
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


def to_tuning(cfg):
    """Настройки подбора скорости обучения из секции run.

    Интерполяции секции не разрешаются: имя прогона зависит от выбора Hydra, а
    настройкам подбора оно не нужно.

    Args:
        cfg: конфиг Hydra.

    Returns:
        Настройки подбора.
    """
    from mayak.tuning import tuning_from_config
    return tuning_from_config(OmegaConf.to_container(cfg.run, resolve=False))


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig

    from mayak.protocol import ProtocolError, run_experiment
    from mayak.tuning import format_tuning

    hc = HydraConfig.get()
    rc = to_run_config(cfg)
    tuning = to_tuning(cfg)
    out_dir = hc.runtime.output_dir
    try:
        journal = run_experiment(rc, out_root=os.path.dirname(out_dir),
                                 tag=os.path.basename(out_dir), accelerator=cfg.run.accelerator,
                                 tuning=tuning)
    except ProtocolError as e:
        # Исключение, а не выход: Hydra ловит только Exception, и выход из процесса
        # оборвал бы групповой запуск на первом прогоне с ошибкой.
        log.error("%s", e)
        raise
    for line in format_tuning(journal.get("tuning")):
        log.info("%s", line)
    for st in journal["stages"]:
        log.info("лучшая модель этапа %s: %s", st["name"], st["best_ckpt"])
        if st.get("report"):
            log.info("отчёт о поле этапа %s: %s", st["name"], st["report"])
    if journal["final_ckpt"]:
        log.info("итоговый чекпойнт: %s", journal["final_ckpt"])
    return journal["stages"][-1]["best_score"] if journal["stages"] else None


if __name__ == "__main__":
    main()
