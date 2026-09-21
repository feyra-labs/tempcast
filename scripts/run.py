"""Обучение через Hydra: композиция конфигов, переопределения, групповые запуски.

Примеры:
    python scripts/run.py                                   # МАЯК, протокол по умолчанию
    python scripts/run.py model=gru                         # бейзлайн - другая группа model
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


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    import logging

    from hydra.core.hydra_config import HydraConfig

    from mayak.protocol import run_experiment

    log = logging.getLogger(__name__)
    rc = to_run_config(cfg)
    out_dir = HydraConfig.get().runtime.output_dir
    journal = run_experiment(rc, out_root=os.path.dirname(out_dir),
                             tag=os.path.basename(out_dir), accelerator=cfg.run.accelerator)
    for st in journal["stages"]:
        log.info("лучшая модель этапа %s: %s", st["name"], st["best_ckpt"])
    log.info("итоговый чекпойнт: %s", journal["final_ckpt"])
    return journal["stages"][-1]["best_score"]


if __name__ == "__main__":
    main()
