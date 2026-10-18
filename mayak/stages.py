"""Этапы обучения: каталоги, записи об этапах и переход между ними."""
from __future__ import annotations

import json
import logging
import math
import os

from mayak.protocol import ProtocolError

log = logging.getLogger(__name__)

STAGE_KEY = "mayak_stage"
# Отметка в записи этапа: этап прошёл все шаги. Ставится после обучения в лучший чекпойнт.
COMPLETE_FIELD = "completed"
FIELD_CURRICULUM = "L0"


def stage_dir_name(name):
    """Имя каталога этапа внутри каталога прогона."""
    return f"stage{name}"


def jsonable(obj):
    """Значение, которое JSON записывает и читает обратно без изменений."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def journal_stage(journal_path, stage_name):
    """Последняя запись этапа из журнала прогона."""
    if not journal_path or not os.path.isfile(journal_path):
        return {}
    with open(journal_path, encoding="utf-8") as f:
        journal = json.load(f)
    return next((s for s in reversed(journal.get("stages", []))
                 if s.get("name") == stage_name), {})


def load_init_weights(module, path):
    """Загружает в модуль веса чекпойнта предыдущего этапа."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    try:
        module.load_state_dict(sd)
    except RuntimeError as e:
        raise ProtocolError(f"веса чекпойнта {path} не подходят к модели этапа: {e}") from e


def stage_record(stage, index, data_key, run_dir, journal, init_from=None):
    """Запись об этапе, которая кладётся в каждый его чекпойнт."""
    return jsonable(dict(stage=stage.name, index=int(index), curriculum=stage.curriculum,
                         data_key=data_key, run_dir=os.path.abspath(run_dir),
                         journal=os.path.abspath(journal), init_from=init_from))


def mark_complete(path, steps, steps_done):
    """Отметить в лучшем чекпойнте этапа, что этап прошёл все шаги."""
    import torch
    if int(steps_done) < int(steps):
        raise ProtocolError(f"этап прошёл {steps_done} из {steps} шагов; чекпойнт {path} "
                            f"не отмечен как завершённый")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rec = dict(ck.get(STAGE_KEY) or {})
    rec[COMPLETE_FIELD] = dict(steps=int(steps), steps_done=int(steps_done))
    ck[STAGE_KEY] = rec
    torch.save(ck, path)


def lineage(prev):
    """Короткая запись о том, откуда стартовал этап, для чекпойнта."""
    if prev is None:
        return None
    return jsonable({k: prev.get(k) for k in ("ckpt", "digest", "stage", "step")})


def clear_stale(stage_dir):
    """Удаляет чекпойнты прежнего запуска из каталога этапа."""
    if not os.path.isdir(stage_dir):
        return
    old = sorted(n for n in os.listdir(stage_dir) if n.endswith(".ckpt"))
    if old:
        log.warning("в %s удаляются чекпойнты прежнего запуска: %s", stage_dir, ", ".join(old))
        for n in old:
            os.remove(os.path.join(stage_dir, n))


__all__ = ["COMPLETE_FIELD", "FIELD_CURRICULUM", "STAGE_KEY", "clear_stale", "journal_stage",
           "jsonable", "lineage", "load_init_weights", "mark_complete", "stage_dir_name",
           "stage_record"]
