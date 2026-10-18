r"""Обучение: конфиги Hydra, переопределения, групповые запуски."""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

RUN_SECTIONS = ("model", "data", "train")
# Значения, которые в теге прогона не пишутся.
BASE_ABLATION, BASE_AUGMENT, BASE_SEED = "none", "default", 0


def run_tag(arch, ablation, profile, extra, seed):
    """Тег прогона, он же имя его каталога."""
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
    """Полная конфигурация прогона из конфига Hydra."""
    from mayak.config import RunConfig
    d = {k: OmegaConf.to_container(cfg[k], resolve=True) for k in RUN_SECTIONS}
    if d["model"].get("arch") != "mayak" and not d["model"].get("ablations", True):
        d["model"].pop("ablations")
    return RunConfig.from_dict(d)


def to_tuning(cfg):
    """Настройки подбора скорости обучения из секции run."""
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
