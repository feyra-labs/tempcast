"""Явные словари разрезов оценки: зоны Кёппена и сезоны."""
from __future__ import annotations

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
N_SEASONS = len(SEASONS)


def normalize_zone(zone) -> str:
    """Зона из манифеста, приведённая к таблице зон.

    Args:
        zone: зона Кёппена как в манифесте, любого типа.

    Returns:
        Зона из таблицы или ``UNK``, если такой зоны в таблице нет.
    """
    z = str(zone).strip()
    return z if z in KOPPEN_ID else UNKNOWN_ZONE


def koppen_group(zone) -> str:
    """Крупная группа зоны: первая буква кода или ``UNK``.

    Нужна для разрезов, где в мелких зонах слишком мало станций.

    Args:
        zone: зона Кёппена как в манифесте.

    Returns:
        Одна из букв A, B, C, D, E либо ``UNK``.
    """
    z = normalize_zone(zone)
    return UNKNOWN_ZONE if z == UNKNOWN_ZONE else z[0]


def season_of(month, lat) -> str:
    """Местный сезон по месяцу и широте.

    В северном полушарии зима - декабрь, январь и февраль, дальше по три месяца на
    сезон. В южном полушарии сезоны сдвинуты на полгода.

    Args:
        month: месяц UTC от 1 до 12.
        lat: широта, градусы.

    Returns:
        Имя сезона: ``winter``, ``spring``, ``summer`` или ``autumn``.

    Raises:
        ValueError: месяц вне диапазона от 1 до 12.
    """
    m = int(month)
    if not 1 <= m <= 12:
        raise ValueError(f"месяц вне [1, 12]: {month}")
    i = (m % 12) // 3
    if float(lat) < 0.0:
        i = (i + 2) % 4
    return SEASONS[i]


__all__ = ["KG_TIF_CODE", "KOPPEN_ID", "KOPPEN_ZONES", "N_KOPPEN", "N_SEASONS", "SEASONS",
           "SEASON_RU", "UNKNOWN_ZONE", "ZONES_VERSION", "koppen_group", "normalize_zone",
           "season_of"]
