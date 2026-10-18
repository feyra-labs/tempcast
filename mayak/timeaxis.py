"""Календарь: день года, час UTC, абсолютные часы."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

_EPOCH_H = np.datetime64("1970-01-01T00", "h")


def _as_datetime64_s(ts) -> np.ndarray:
    """Момент или массив моментов в любом привычном виде как datetime64[s] в UTC."""
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
    """День года и час UTC по календарной конвенции проекта."""
    t = _as_datetime64_s(ts)
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    sec = (t - year_start).astype(np.int64)
    return sec / 86400.0, (sec % 86400) / 3600.0


def month_of(ts):
    """Календарный месяц UTC."""
    t = _as_datetime64_s(ts)
    return t.astype("datetime64[M]").astype(np.int64) % 12 + 1


def window_month(t0_utc_h: int, idx):
    """Календарные месяцы UTC для часов ряда станции."""
    return month_of(from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64)))


def to_utc_hour(ts) -> int:
    """Целое число часов от эпохи для момента UTC."""
    t = _as_datetime64_s(ts)
    h = t.astype("datetime64[h]")
    if np.any(h.astype("datetime64[s]") != t):
        raise ValueError(f"момент {t} не лежит на целом часе UTC")
    return (h - _EPOCH_H).astype(np.int64)


def from_utc_hour(t_h) -> np.ndarray:
    """Моменты для часов от эпохи."""
    return _EPOCH_H + np.asarray(t_h, dtype=np.int64).astype("timedelta64[h]")


def hour_of_year(t_h):
    """Номер часа от начала года UTC."""
    t = from_utc_hour(t_h)
    start = t.astype("datetime64[Y]").astype("datetime64[h]")
    return (t - start).astype(np.int64)


def window_calendar(t0_utc_h: int, idx):
    ts = from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64))
    doy, hour = doy_hour(ts)
    return doy.astype(np.float32), hour.astype(np.float32)


def to_hourly_grid(times, columns: dict):
    """Переиндексация ряда на полную почасовую сетку UTC."""
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
