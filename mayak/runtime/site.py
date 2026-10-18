"""Смена координат и высоты прибора между записью состояния и текущим запуском."""
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
    """Параметры хоста устройства из YAML."""
    path = path or (DEFAULT_RUNTIME_CONFIG if os.path.exists(DEFAULT_RUNTIME_CONFIG) else None)
    if path is None:
        return RuntimeConfig()
    import yaml
    with open(path, encoding="utf-8") as fh:
        return RuntimeConfig.from_dict(yaml.safe_load(fh) or {})


def as_site(site):
    """Широта, долгота и высота в одинарной точности, как они лежат в состоянии."""
    return tuple(float(np.float32(v)) for v in site)


def lon_gap(a, b):
    """Разница долгот по кратчайшей дуге."""
    d = math.fmod(abs(a - b), 360.0)
    return min(d, 360.0 - d) if d == d else d


def site_gap(old, new):
    """Разницы широты, долготы и высоты двух точек в одинарной точности."""
    o, n = as_site(old), as_site(new)
    return abs(n[0] - o[0]), lon_gap(n[1], o[1]), abs(n[2] - o[2])


def site_change(old, new, cfg):
    """Исход сравнения точки состояния с точкой текущего запуска."""
    gap = site_gap(old, new)
    limits = (cfg.site_max_dlat_deg, cfg.site_max_dlon_deg, cfg.site_max_delev_m)
    if all(g == 0.0 for g in gap):
        return SITE_SAME, gap
    if all(g <= lim for g, lim in zip(gap, limits)):
        return SITE_REFINED, gap
    return SITE_MOVED, gap


def describe_gap(gap, cfg):
    """Сдвиг точки и пороги одной строкой для лога."""
    return (f"широта на {gap[0]:.4g}° (порог {cfg.site_max_dlat_deg:g}°), долгота на "
            f"{gap[1]:.4g}° (порог {cfg.site_max_dlon_deg:g}°), высота на {gap[2]:.4g} м "
            f"(порог {cfg.site_max_delev_m:g} м)")


__all__ = ["DEFAULT_RUNTIME_CONFIG", "SITE_CHANGES", "SITE_MOVED", "SITE_REFINED", "SITE_SAME",
           "as_site", "describe_gap", "load_runtime_config", "lon_gap", "site_change",
           "site_gap"]
