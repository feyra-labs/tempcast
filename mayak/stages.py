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


def _normalized(value):
    return json.loads(json.dumps(jsonable(value), sort_keys=True))


def config_diff(where, old, new):
    """Отличия двух конфигов, по одной строке на поле.

    Args:
        where: имя конфига в начале каждой строки.
        old: значение из чекпойнта.
        new: значение этого запуска.

    Returns:
        Список строк вида «модель.encoder_width: в чекпойнте 16, в запуске 64».
    """
    old, new = _normalized(old), _normalized(new)
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            if k not in old:
                out.append(f"{where}.{k}: нет в чекпойнте, в запуске {new[k]!r}")
            elif k not in new:
                out.append(f"{where}.{k}: в чекпойнте {old[k]!r}, нет в запуске")
            else:
                out += config_diff(f"{where}.{k}", old[k], new[k])
        return out
    if old != new:
        return [f"{where}: в чекпойнте {old!r}, в запуске {new!r}"]
    return []


# Путь к манифесту и каталог кэша зависят от того, откуда запущена команда. Совпадение
# данных проверяется по ключу кэша: он строится по содержимому манифеста и источников.
DATA_PATH_FIELDS = ("manifest", "cache_root")


def _data_fields(d):
    return {k: v for k, v in dict(d or {}).items() if k not in DATA_PATH_FIELDS}


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


def warn_stale(stage_dir):
    """Предупреждает, если в каталоге этапа остались чекпойнты прежнего запуска.

    Args:
        stage_dir: каталог этапа.
    """
    if not os.path.isdir(stage_dir):
        return
    old = [n for n in os.listdir(stage_dir) if n.endswith(".ckpt")]
    if old:
        log.warning("в %s уже есть чекпойнты прежнего запуска (%d); новые получат суффикс "
                    "версии, журнал и отчёт укажут только на новые", stage_dir, len(old))


__all__ = ["DATA_PATH_FIELDS", "FIELD_CURRICULUM", "STAGE_KEY", "config_diff", "journal_stage",
           "jsonable", "lineage", "load_init_weights", "stage_dir_name", "stage_record",
           "warn_stale"]
