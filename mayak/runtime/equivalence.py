"""Измерение эквивалентности пакетного и потокового путей и стоимости часа."""
from __future__ import annotations

import time

import numpy as np
import torch

YEAR_H = 365 * 24


def synthetic_series(n, seed=0, start_hoy=360 * 24, p_valid=0.9):
    """Почасовой ряд длины n: T, P, RH (n, 3), маска (n, 3), doy и hour (n,)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    T = 8 + 6 * np.sin(2 * np.pi * t / 24) + 3 * np.sin(2 * np.pi * t / 170) + rng.standard_normal(n)
    P = 1005 + 8 * np.sin(2 * np.pi * t / 130 + rng.uniform(0, 6)) + 0.4 * rng.standard_normal(n)
    RH = np.clip(70 - 2 * (T - 8) + 5 * rng.standard_normal(n), 5, 100)
    x = np.stack([T, P, RH], -1).astype(np.float32)
    m = (rng.random((n, 3)) < p_valid).astype(np.float32)
    gap = rng.integers(0, max(n - 30, 1))
    m[gap:gap + 20, 0] = 0.0                                  # блочный пропуск T
    hoy = (start_hoy + t) % YEAR_H
    return dict(x=x * m, m=m, doy=(hoy / 24.0).astype(np.float32),
                hour=(hoy % 24).astype(np.float32), hoy0=int(start_hoy))


def future_calendar_after(series, end, horizon):
    hoy = (series["hoy0"] + end + np.arange(horizon)) % YEAR_H
    return (hoy / 24.0).astype(np.float32), (hoy % 24).astype(np.float32)


def batch_forecast(model, series, end, lat, lon, elev):
    """Пакетный выпуск по окну max_history часов, заканчивающемуся перед часом end."""
    cfg = model.cfg
    L = cfg.max_history
    x = np.zeros((L, 3), np.float32)
    mk = np.zeros((L, 3), np.float32)
    k = min(L, end)
    x[L - k:], mk[L - k:] = series["x"][end - k:end], series["m"][end - k:end]
    hoy = (series["hoy0"] + end - L + np.arange(L)) % YEAR_H
    doy_f, hour_f = future_calendar_after(series, end, cfg.horizon)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)[None]
    b = dict(lat=t([lat])[0], lon=t([lon])[0], elev=t([elev])[0], x_hist=t(x), mask_hist=t(mk),
             doy_hist=t(hoy / 24.0), hour_hist=t(hoy % 24), doy_fut=t(doy_f), hour_fut=t(hour_f))
    with torch.no_grad():
        return model(b)["q"][0].numpy()


def feed(stream, series, k0, k1):
    """Часы k0 … k1 − 1 ряда в поток через публичный step (с QC точки)."""
    x, m = series["x"], series["m"]
    for k in range(k0, k1):
        v = [float(x[k, j]) if m[k, j] > 0 else None for j in range(3)]
        stream.step(*v, series["doy"][k], series["hour"][k])
    return stream


def stream_forecast(model, series, end, lat, lon, elev):
    """Потоковый выпуск: холодный старт в первом часе пакетного окна, дальше step."""
    from mayak.runtime.streaming import StreamingMayak
    s = StreamingMayak(model, lat, lon, elev)
    start = end - model.cfg.max_history
    for i in range(max(0, -start)):
        hoy = (series["hoy0"] + start + i) % YEAR_H
        s.step(None, None, None, np.float32(hoy / 24.0), np.float32(hoy % 24))
    feed(s, series, max(start, 0), end)
    q, _ = s.forecast(*future_calendar_after(series, end, model.cfg.horizon))
    return q, s


def divergence(model, n_windows=8, seed=0, lat=52.37, lon=4.9, elev=0.0):
    """Расхождение пакет ↔ поток на полном выпуске по всем лидам и после перезапуска. """
    from mayak.runtime.streaming import StreamingMayak
    L, Hh = model.cfg.max_history, model.cfg.horizon
    per_lead = np.zeros(Hh)
    restart = 0.0
    rng = np.random.default_rng(seed)
    for w in range(n_windows):
        extra = int(rng.integers(1, 48))
        series = synthetic_series(L + extra + 1, seed=seed * 1000 + w)
        end = L
        qb = batch_forecast(model, series, end, lat, lon, elev)
        qs, s = stream_forecast(model, series, end, lat, lon, elev)
        per_lead = np.maximum(per_lead, np.abs(qb - qs).max(-1))
        s2 = StreamingMayak(model, lat, lon, elev)
        s2.load_state(s.serialize())
        feed(s, series, end, end + extra)
        feed(s2, series, end, end + extra)
        cal = future_calendar_after(series, end + extra, Hh)
        restart = max(restart, float(np.abs(s.forecast(*cal)[0] - s2.forecast(*cal)[0]).max()))
    return dict(n_windows=n_windows, batch_stream_max_abs=float(per_lead.max()),
                batch_stream_max_abs_per_lead=per_lead.tolist(), restart_max_abs=restart)


def _flops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn()
    return int(fc.get_total_flops())


def step_cost(model, reps=200, seed=0):
    """Стоимость часа: инкрементальный шаг против пересчёта энкодера по буферу RF.

    ``legacy`` воспроизводит прежний рантайм: каналы и энкодер целиком по окну
    stream_buffer на каждый час. Возвращает FLOP (матричные операции и свёртки, без
    поэлементных) и время на час, мс.
    """
    enc, cfg = model.encoder, model.cfg
    g = torch.Generator().manual_seed(seed)
    win = torch.randn(1, cfg.n_channels, cfg.stream_buffer, generator=g)
    x_t = win[..., -1]
    st = enc.init_state(1)
    fl_step = _flops(lambda: enc.step(x_t, st.clone()))
    fl_full = _flops(lambda: enc(win))

    def timed(fn):
        with torch.no_grad():
            for _ in range(10):
                fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
        return (time.perf_counter() - t0) / reps * 1e3

    ms_step = timed(lambda: enc.step(x_t, st))
    ms_full = timed(lambda: enc(win))
    return dict(receptive_field=cfg.receptive_field, stream_buffer=cfg.stream_buffer,
                encoder_flops_step=fl_step, encoder_flops_full_window=fl_full,
                flops_ratio=fl_full / max(fl_step, 1), encoder_ms_step=ms_step,
                encoder_ms_full_window=ms_full, time_ratio=ms_full / max(ms_step, 1e-9),
                encoder_buffer_bytes=st.nbytes)


def stream_hour_ms(model, hours=200, seed=0, lat=52.37, lon=4.9, elev=0.0):
    """Полная стоимость StreamingMayak.step (QC, каналы, энкодер, моды, сутки), мс."""
    from mayak.runtime.streaming import StreamingMayak
    series = synthetic_series(hours + 50, seed=seed)
    s = feed(StreamingMayak(model, lat, lon, elev), series, 0, 50)
    t0 = time.perf_counter()
    feed(s, series, 50, 50 + hours)
    return (time.perf_counter() - t0) / hours * 1e3, s


__all__ = ["batch_forecast", "divergence", "feed", "future_calendar_after", "step_cost",
           "stream_forecast", "stream_hour_ms", "synthetic_series"]
