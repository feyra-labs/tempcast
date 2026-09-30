"""Смена координат и высоты прибора между записью состояния и текущим запуском.

Состояние на диске помнит точку, для которой оно записано. При загрузке точка
состояния сравнивается с текущими аргументами, исходов три.

* Та же точка - всё как обычно.
* Уточнение: широта, долгота и высота отличаются не больше порогов рантайма. Это
  исправление метаданных. Окно наблюдений сохраняется, всё модельное состояние
  пересчитывается из него уже для новой точки, множитель калибровки сохраняется.
  В лог пишется предупреждение с величиной сдвига, чтобы опечатка в координатах была
  заметна.
* Перенос: хотя бы одна разница больше порога. История чужой точки новой точке не
  нужна, промахи калибровки на старом месте ничего не говорят о новом. Окно
  опустошается, множитель калибровки обнуляется, абсолютный час последнего шага
  сохраняется: время прибора от места не зависит.

Точки сравниваются так, как они лежат в состоянии, в одинарной точности. Разница
долгот берётся по кратчайшей дуге. Пороги включительные. Точка с нечисловыми
значениями считается перенесённой.
"""
from __future__ import annotations

import math
import os

import numpy as np

from mayak.config import RuntimeConfig

SITE_SAME = "same"
SITE_REFINED = "refined"
SITE_MOVED = "moved"
SITE_CHANGES = (SITE_SAME, SITE_REFINED, SITE_MOVED)
DEFAULT_RUNTIME_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "conf", "runtime", "default.yaml")


def load_runtime_config(path=None):
    """Параметры хоста устройства из YAML.

    Args:
        path: путь к YAML или None - файл по умолчанию из каталога конфигов, если он
            есть, иначе значения датакласса.

    Returns:
        Параметры хоста.
    """
    path = path or (DEFAULT_RUNTIME_CONFIG if os.path.exists(DEFAULT_RUNTIME_CONFIG) else None)
    if path is None:
        return RuntimeConfig()
    import yaml
    with open(path, encoding="utf-8") as fh:
        return RuntimeConfig.from_dict(yaml.safe_load(fh) or {})


def as_site(site):
    """Широта, долгота и высота в одинарной точности, как они лежат в состоянии.

    Args:
        site: три числа.

    Returns:
        Кортеж из трёх чисел Python, точно равных значениям float32.
    """
    return tuple(float(np.float32(v)) for v in site)


def lon_gap(a, b):
    """Разница долгот по кратчайшей дуге.

    Args:
        a: долгота, градусы.
        b: долгота, градусы.

    Returns:
        Число от 0 до 180 градусов; нечисловое, если нечисловая хотя бы одна долгота.
    """
    d = math.fmod(abs(a - b), 360.0)
    return min(d, 360.0 - d) if d == d else d


def site_gap(old, new):
    """Разницы широты, долготы и высоты двух точек в одинарной точности.

    Args:
        old: широта, долгота и высота из состояния.
        new: широта, долгота и высота текущего запуска.

    Returns:
        Три неотрицательных числа: градусы, градусы, метры.
    """
    o, n = as_site(old), as_site(new)
    return abs(n[0] - o[0]), lon_gap(n[1], o[1]), abs(n[2] - o[2])


def site_change(old, new, cfg):
    """Исход сравнения точки состояния с точкой текущего запуска.

    Args:
        old: широта, долгота и высота из состояния.
        new: широта, долгота и высота текущего запуска.
        cfg: параметры хоста с порогами.

    Returns:
        Пара: исход (та же точка, уточнение или перенос) и три разницы.
    """
    gap = site_gap(old, new)
    limits = (cfg.site_max_dlat_deg, cfg.site_max_dlon_deg, cfg.site_max_delev_m)
    if all(g == 0.0 for g in gap):
        return SITE_SAME, gap
    if all(g <= lim for g, lim in zip(gap, limits)):
        return SITE_REFINED, gap
    return SITE_MOVED, gap


def describe_gap(gap, cfg):
    """Сдвиг точки и пороги одной строкой для лога.

    Args:
        gap: разницы широты, долготы и высоты.
        cfg: параметры хоста с порогами.

    Returns:
        Строка.
    """
    return (f"широта на {gap[0]:.4g}° (порог {cfg.site_max_dlat_deg:g}°), долгота на "
            f"{gap[1]:.4g}° (порог {cfg.site_max_dlon_deg:g}°), высота на {gap[2]:.4g} м "
            f"(порог {cfg.site_max_delev_m:g} м)")


__all__ = ["DEFAULT_RUNTIME_CONFIG", "SITE_CHANGES", "SITE_MOVED", "SITE_REFINED", "SITE_SAME",
           "as_site", "describe_gap", "load_runtime_config", "lon_gap", "site_change",
           "site_gap"]
