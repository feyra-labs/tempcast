"""Явные словари разрезов оценки: зоны Кёппена и сезоны."""
from __future__ import annotations

import numpy as np

ZONES_VERSION = "1"

KOPPEN_ZONES = (
    "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk",
    "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
    "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd",
    "Dfa", "Dfb", "Dfc", "Dfd", "ET", "EF",
)
UNKNOWN_ZONE = "UNK"
KOPPEN_ID = {z: i for i, z in enumerate(KOPPEN_ZONES)}
KOPPEN_ID[UNKNOWN_ZONE] = len(KOPPEN_ZONES)
N_KOPPEN = len(KOPPEN_ID)

KG_TIF_CODE = {i + 1: z for i, z in enumerate(KOPPEN_ZONES)}

SEASONS = ("winter", "spring", "summer", "autumn")
SEASON_RU = {"winter": "зима", "spring": "весна", "summer": "лето", "autumn": "осень"}
SEASON_ID = {s: i for i, s in enumerate(SEASONS)}
N_SEASONS = len(SEASONS)


def normalize_zone(zone) -> str:
    """Строка из манифеста → зона из таблицы либо ``UNK``."""
    z = str(zone).strip()
    return z if z in KOPPEN_ID else UNKNOWN_ZONE


def koppen_id(zone) -> int:
    """Полная зона Кёппена → стабильный идентификатор [0, N_KOPPEN)."""
    return KOPPEN_ID[normalize_zone(zone)]


def koppen_group(zone) -> str:
    """Крупный разрез: A/B/C/D/E либо UNK. Для мелких зон с малым числом станций."""
    z = normalize_zone(zone)
    return UNKNOWN_ZONE if z == UNKNOWN_ZONE else z[0]


def season_of(month, lat) -> str:
    """(месяц UTC 1..12, широта) → местный сезон.

    Северное полушарие: DJF → winter, MAM → spring, JJA → summer, SON → autumn.
    Южное — сдвиг на два бина (полгода).
    """
    m = int(month)
    if not 1 <= m <= 12:
        raise ValueError(f"месяц вне [1, 12]: {month}")
    i = (m % 12) // 3
    if float(lat) < 0.0:
        i = (i + 2) % 4
    return SEASONS[i]


def season_id(month, lat) -> int:
    return SEASON_ID[season_of(month, lat)]


def seasons_of(months, lat) -> np.ndarray:
    """Векторная версия ``season_of`` для массива месяцев одной станции."""
    m = np.asarray(months, dtype=np.int64)
    if m.size and (m.min() < 1 or m.max() > 12):
        raise ValueError("месяц вне [1, 12]")
    i = (m % 12) // 3
    if float(lat) < 0.0:
        i = (i + 2) % 4
    return np.asarray(SEASONS, dtype=object)[i]


__all__ = ["KG_TIF_CODE", "KOPPEN_ID", "KOPPEN_ZONES", "N_KOPPEN", "N_SEASONS", "SEASONS",
           "SEASON_ID", "SEASON_RU", "UNKNOWN_ZONE", "ZONES_VERSION", "koppen_group",
           "koppen_id", "normalize_zone", "season_id", "season_of", "seasons_of"]
