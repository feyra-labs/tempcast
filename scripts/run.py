"""Обучение через Hydra: композиция конфигов, переопределения, групповые запуски.

Примеры:
    python scripts/run.py                                   # МАЯК, протокол по умолчанию
    python scripts/run.py model=gru                         # бейзлайн - другая группа model
    python scripts/run.py -m model=gru,dlinear,lru,patchtst # все нейробейзлайны
    python scripts/run.py train=debug run.accelerator=cpu   # отладка на CPU
    python scripts/run.py -m ablation=none,no_anchor,no_compression   # абляции (блок 6.7)
    python scripts/run.py -m train.seed=0,1,2               # три сида основной модели
"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf

RUN_SECTIONS = ("model", "data", "train")


def to_run_config(cfg):
    """DictConfig Hydra → RunConfig (только секции model/data/train, всё разрешено)."""
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


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    import logging

    from hydra.core.hydra_config import HydraConfig

    from mayak.protocol import run_experiment

    log = logging.getLogger(__name__)
    rc = to_run_config(cfg)
    launch = to_launch(cfg)
    out_dir = HydraConfig.get().runtime.output_dir
    journal = run_experiment(rc, out_root=os.path.dirname(out_dir),
                             tag=os.path.basename(out_dir), accelerator=cfg.run.accelerator,
                             launch=launch)
    for st in journal["stages"]:
        log.info("лучшая модель этапа %s: %s", st["name"], st["best_ckpt"])
        if st.get("report"):
            log.info("отчёт о поле этапа %s: %s", st["name"], st["report"])
    log.info("итоговый чекпойнт: %s", journal["final_ckpt"])
    return journal["stages"][-1]["best_score"] if journal["stages"] else None


if __name__ == "__main__":
    main()
