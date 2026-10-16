"""Единая календарная конвенция проекта."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

_EPOCH_H = np.datetime64("1970-01-01T00", "h")


def _as_datetime64_s(ts) -> np.ndarray:
    """Момент или массив моментов в любом привычном виде как datetime64[s] в UTC.

    Понимает datetime, метки pandas, datetime64, строки и массивы из них.
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(ts, "s")
    if hasattr(ts, "tz_convert") or hasattr(ts, "dt"):          # pandas
        import pandas as pd
        idx = pd.DatetimeIndex(np.atleast_1d(ts))
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        out = idx.values.astype("datetime64[s]")
        return out if np.ndim(ts) else out[0]
    return np.asarray(ts, dtype="datetime64[s]")


def doy_hour(ts):
    """День года и час UTC по календарной конвенции проекта.

    Args:
        ts: момент UTC или массив моментов.

    Returns:
        Пара float64: день года с нуля с дробной частью, равной доле суток, и час UTC.
    """
    t = _as_datetime64_s(ts)
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    sec = (t - year_start).astype(np.int64)
    return sec / 86400.0, (sec % 86400) / 3600.0


def month_of(ts):
    """Календарный месяц UTC.

    Args:
        ts: момент UTC или массив моментов.

    Returns:
        Месяц от 1 до 12, int64.
    """
    t = _as_datetime64_s(ts)
    return t.astype("datetime64[M]").astype(np.int64) % 12 + 1


def window_month(t0_utc_h: int, idx):
    """Календарные месяцы UTC для часов ряда станции.

    Args:
        t0_utc_h: абсолютный час первой строки ряда, часы от эпохи.
        idx: номера строк ряда.

    Returns:
        Месяцы от 1 до 12, int64, форма ``idx``.
    """
    return month_of(from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64)))


def to_utc_hour(ts) -> int:
    """Целое число часов от эпохи для момента UTC.

    Args:
        ts: момент UTC или массив моментов.

    Returns:
        Часы от эпохи, int64.

    Raises:
        ValueError: момент не лежит на целом часе.
    """
    t = _as_datetime64_s(ts)
    h = t.astype("datetime64[h]")
    if np.any(h.astype("datetime64[s]") != t):
        raise ValueError(f"момент {t} не лежит на целом часе UTC")
    return (h - _EPOCH_H).astype(np.int64)


def from_utc_hour(t_h) -> np.ndarray:
    """Моменты для часов от эпохи.

    Args:
        t_h: часы от эпохи, число или массив.

    Returns:
        Массив datetime64[h].
    """
    return _EPOCH_H + np.asarray(t_h, dtype=np.int64).astype("timedelta64[h]")


def hour_of_year(t_h):
    """Номер часа от начала года UTC.

    Високосный год даёт номера от 0 до 8783, обычный от 0 до 8759. Номер однозначно
    задаёт календарь часа, из которого считается климат-поле: день года и час суток.

    Args:
        t_h: абсолютные часы UTC, целые часы от эпохи; число или массив.

    Returns:
        Массив int64 той же формы.
    """
    t = from_utc_hour(t_h)
    start = t.astype("datetime64[Y]").astype("datetime64[h]")
    return (t - start).astype(np.int64)


def window_calendar(t0_utc_h: int, idx):
    ts = from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64))
    doy, hour = doy_hour(ts)
    return doy.astype(np.float32), hour.astype(np.float32)


def to_hourly_grid(times, columns: dict):
    """Переиндексация ряда на полную почасовую сетку UTC.

    Args:
        times: метки времени наблюдений, каждая на целом часе.
        columns: словарь имя колонки - массив той же длины, что ``times``.

    Returns:
        Пара: абсолютный час первой строки сетки и словарь колонок float32 длины
        сетки, где отсутствующие часы заполнены NaN.

    Raises:
        ValueError: ряд пуст или метки времени повторяются.
    """
    t_h = np.asarray(to_utc_hour(times), dtype=np.int64).ravel()
    if t_h.size == 0:
        raise ValueError("пустой ряд")
    order = np.argsort(t_h, kind="stable")
    t_h = t_h[order]
    dup = np.flatnonzero(np.diff(t_h) == 0)
    if dup.size:
        raise ValueError(f"дубликаты меток времени: {from_utc_hour(t_h[dup[:5]])}")
    t0 = int(t_h[0])
    n = int(t_h[-1] - t0 + 1)
    pos = t_h - t0
    out = {}
    for name, v in columns.items():
        v = np.asarray(v, dtype=np.float32)[order]
        full = np.full(n, np.nan, np.float32)
        full[pos] = v
        out[name] = full
    return t0, out
