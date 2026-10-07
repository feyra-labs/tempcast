"""Этапы обучения: каталоги, записи об этапах и переход между ними.

Этапы протокола идут подряд в одном запуске. Этап, который в протоколе идёт не первым,
стартует с лучшего по метрике выбора чекпойнта предыдущего этапа: своего прогона или,
после подбора скорости обучения, прогона сетки с выбранной скоростью
(``mayak.tuning``). В каждый чекпойнт этапа кладётся запись об этапе: имя и номер в
протоколе, ключ кэша данных, каталог и журнал прогона и стартовый чекпойнт.
"""
from __future__ import annotations

import json
import logging
import math
import os

from mayak.protocol import ProtocolError

log = logging.getLogger(__name__)

STAGE_KEY = "mayak_stage"
FIELD_CURRICULUM = "L0"


def stage_dir_name(name):
    """Имя каталога этапа внутри каталога прогона.

    Args:
        name: имя этапа.

    Returns:
        Строка вида stageA.
    """
    return f"stage{name}"


def jsonable(obj):
    """Значение, которое JSON записывает и читает обратно без изменений.

    Кортежи становятся списками, числа numpy и тензоры обычными числами, ключи словарей
    строками, а бесконечности и нечисла значением None: иначе прочитанный журнал не
    совпал бы с записанным.

    Args:
        obj: значение.

    Returns:
        Значение из словарей, списков, строк, чисел, логических значений и None.
    """
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
    """Последняя запись этапа из журнала прогона.

    Args:
        journal_path: путь к журналу.
        stage_name: имя этапа.

    Returns:
        Запись этапа; пустой словарь, если журнала нет или этапа в нём нет.
    """
    if not journal_path or not os.path.isfile(journal_path):
        return {}
    with open(journal_path) as f:
        journal = json.load(f)
    return next((s for s in reversed(journal.get("stages", []))
                 if s.get("name") == stage_name), {})


def load_init_weights(module, path):
    """Загружает в модуль веса чекпойнта предыдущего этапа.

    Args:
        module: обучаемый модуль этапа.
        path: путь к чекпойнту.

    Raises:
        ProtocolError: веса не подходят к модели.
    """
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    try:
        module.load_state_dict(sd)
    except RuntimeError as e:
        raise ProtocolError(f"веса чекпойнта {path} не подходят к модели этапа: {e}") from e


def stage_record(stage, index, data_key, run_dir, journal, init_from=None):
    """Запись об этапе, которая кладётся в каждый его чекпойнт.

    Args:
        stage: этап.
        index: номер этапа в протоколе.
        data_key: ключ кэша данных.
        run_dir: каталог прогона.
        journal: путь к журналу прогона.
        init_from: откуда этап стартовал: путь, отпечаток, этап и шаг; None для первого.

    Returns:
        Словарь без шага сохранения: его добавляет сохранение.
    """
    return jsonable(dict(stage=stage.name, index=int(index), curriculum=stage.curriculum,
                         data_key=data_key, run_dir=os.path.abspath(run_dir),
                         journal=os.path.abspath(journal), init_from=init_from))


def lineage(prev):
    """Короткая запись о том, откуда стартовал этап, для чекпойнта.

    Args:
        prev: запись о стартовом чекпойнте или None.

    Returns:
        Путь, отпечаток, этап и шаг; None для этапа, который стартует с нуля.
    """
    if prev is None:
        return None
    return jsonable({k: prev.get(k) for k in ("ckpt", "digest", "stage", "step")})


def clear_stale(stage_dir):
    """Удаляет чекпойнты прежнего запуска из каталога этапа.

    Чекпойнт этапа всегда лежит под одним именем, и путь к нему не зависит от того,
    сколько раз этап запускали. Чекпойнт прежнего запуска удаляется до обучения: иначе
    при сбое до первого сохранения под этим именем остались бы старые веса.

    Args:
        stage_dir: каталог этапа.
    """
    if not os.path.isdir(stage_dir):
        return
    old = sorted(n for n in os.listdir(stage_dir) if n.endswith(".ckpt"))
    if old:
        log.warning("в %s удаляются чекпойнты прежнего запуска: %s", stage_dir, ", ".join(old))
        for n in old:
            os.remove(os.path.join(stage_dir, n))


__all__ = ["FIELD_CURRICULUM", "STAGE_KEY", "clear_stale", "journal_stage", "jsonable",
           "lineage", "load_init_weights", "stage_dir_name", "stage_record"]
