"""Единая календарная конвенция проекта.

* момент времени хранится как целое число часов от эпохи Unix в UTC
  (``t0_utc_h`` для начала ряда станции); часы ряда — ``t0_utc_h + k``;
* ``doy`` — день года **с нуля** (1 января = 0) **с дробной частью**,
  равной доле прошедших суток: 1 января 06:00 UTC → ``0.25``;
  в високосный год 31 декабря 23:00 → ``365 + 23/24``;
* ``hour`` — час UTC отдельным числом в ``[0, 24)``;
  дробная часть ``doy`` и ``hour`` согласованы: ``frac(doy) == hour / 24``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

CALENDAR_VERSION = "1"
LEGACY_YEAR = 2001
_EPOCH_H = np.datetime64("1970-01-01T00", "h")


def _as_datetime64_s(ts) -> np.ndarray:
    """datetime / pd.Timestamp / np.datetime64 / строка / массив → datetime64[s] (UTC)."""
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
    """Момент(ы) UTC → (doy, hour) по конвенции модуля, float64. Векторизовано."""
    t = _as_datetime64_s(ts)
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    sec = (t - year_start).astype(np.int64)
    return sec / 86400.0, (sec % 86400) / 3600.0


def month_of(ts):
    """Момент(ы) UTC → календарный месяц 1..12 (int64). Векторизовано."""
    t = _as_datetime64_s(ts)
    return t.astype("datetime64[M]").astype(np.int64) % 12 + 1


def window_month(t0_utc_h: int, idx):
    """Часы ряда станции (индексы от ``t0_utc_h``) → календарные месяцы UTC."""
    return month_of(from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64)))


def to_utc_hour(ts) -> int:
    """Момент UTC → целое число часов от эпохи. Падает, если момент не на целом часе."""
    t = _as_datetime64_s(ts)
    h = t.astype("datetime64[h]")
    if np.any(h.astype("datetime64[s]") != t):
        raise ValueError(f"момент {t} не лежит на целом часе UTC")
    return (h - _EPOCH_H).astype(np.int64)


def from_utc_hour(t_h) -> np.ndarray:
    """Часы от эпохи → datetime64[h]."""
    return _EPOCH_H + np.asarray(t_h, dtype=np.int64).astype("timedelta64[h]")


def window_calendar(t0_utc_h: int, idx):
    ts = from_utc_hour(int(t0_utc_h) + np.asarray(idx, dtype=np.int64))
    doy, hour = doy_hour(ts)
    return doy.astype(np.float32), hour.astype(np.float32)


def utc_to_doy_hour(dt: datetime):
    """Скалярная обёртка для рантайма: datetime (UTC) → (doy, hour)."""
    d, h = doy_hour(dt)
    return float(d), float(h)


def future_calendar(last_obs_time: datetime, horizon: int):
    ts = [last_obs_time + timedelta(hours=h) for h in range(1, horizon + 1)]
    d, h = doy_hour(np.array([_as_datetime64_s(x) for x in ts]))
    return d.astype(np.float32), h.astype(np.float32)


def legacy_t0(t0_doy: float, t0_hour: float) -> int:
    day = int(np.floor(float(t0_doy)))
    hour = int(round(float(t0_hour)))
    base = datetime(LEGACY_YEAR, 1, 1) + timedelta(days=day, hours=hour)
    return int(to_utc_hour(base))


def to_hourly_grid(times, columns: dict):
    """Переиндексация ряда на полную почасовую сетку UTC.

    times   — метки времени наблюдений;
    columns — {имя: массив той же длины}.
    Возвращает (t0_utc_h, {имя: float32 (N,) с NaN в отсутствующих часах}).
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
