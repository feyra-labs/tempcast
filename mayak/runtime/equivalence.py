"""Измерение эквивалентности пакетного и потокового путей и стоимости часа и выпуска."""
from __future__ import annotations

import time

import numpy as np
import torch

from mayak.timeaxis import to_utc_hour, window_calendar

DEFAULT_START = "2021-12-27T00"


def default_start():
    """Абсолютный час начала синтетических рядов.

    Конец декабря, чтобы ряды переходили через Новый год.
    """
    return int(to_utc_hour(np.datetime64(DEFAULT_START, "s")))


def synthetic_series(n, seed=0, t0=None, p_valid=0.9):
    """Почасовой синтетический ряд.

    Args:
        n: длина ряда, часы.
        seed: сид.
        t0: абсолютный час UTC первого часа; по умолчанию конец декабря.
        p_valid: доля часов с наблюдением в каждом канале.

    Returns:
        Словарь: ``x`` значения (n, 3) с нулями на месте пропусков, ``m`` маска наличия
        (n, 3), ``t0`` абсолютный час первого часа, ``doy`` и ``hour`` календарь (n,).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    T = (8 + 6 * np.sin(2 * np.pi * t / 24) + 3 * np.sin(2 * np.pi * t / 170)
         + rng.standard_normal(n))
    P = 1005 + 8 * np.sin(2 * np.pi * t / 130 + rng.uniform(0, 6)) + 0.4 * rng.standard_normal(n)
    RH = np.clip(70 - 2 * (T - 8) + 5 * rng.standard_normal(n), 5, 100)
    x = np.stack([T, P, RH], -1).astype(np.float32)
    m = (rng.random((n, 3)) < p_valid).astype(np.float32)
    gap = rng.integers(0, max(n - 30, 1))
    m[gap:gap + 20, 0] = 0.0
    t0 = default_start() if t0 is None else int(t0)
    doy, hour = window_calendar(t0, t)
    return dict(x=x * m, m=m, t0=t0, doy=doy, hour=hour)


def device_values(series, start, end):
    """Значения часов с start до end так, как их записывает прибор.

    Args:
        series: ряд с полями x и m.
        start: первый час.
        end: час после последнего.

    Returns:
        Массив float32, форма (end - start, 3), нули на месте пропусков.
    """
    from mayak.data.recording import record_values
    m = series["m"][start:end] > 0
    return np.where(m, record_values(series["x"][start:end]), 0.0).astype(np.float32)


def device_mask(series, start, end, elev):
    """Маска годности часов с start до end после причинного контроля качества прибора.

    Контроль видит и часы ряда перед start, насколько ему нужно прошлое: так же, как
    прибор, который работает с начала ряда, и как окно обучения с полной историей.

    Args:
        series: ряд с полями x и m.
        start: первый час.
        end: час после последнего.
        elev: высота станции, м.

    Returns:
        Маска float32, форма (end - start, 3).
    """
    from mayak.data.qc import DEFAULT_QC, qc_window
    lo = max(0, start - DEFAULT_QC.lookback_hours)
    past = (device_values(series, lo, start), series["m"][lo:start]) if start > lo else None
    mask, _ = qc_window(device_values(series, start, end), series["m"][start:end], elev=elev,
                        past=past)
    return mask


def batch_forecast(model, series, end, lat, lon, elev):
    """Пакетный выпуск по окну истории, заканчивающемуся перед часом end.

    Args:
        model: модель.
        series: ряд.
        end: индекс первого часа горизонта.
        lat: широта.
        lon: долгота.
        elev: высота, м.

    Returns:
        Квантили формы (H, число квантилей).
    """
    cfg = model.cfg
    L = cfg.max_history
    x = np.zeros((L, 3), np.float32)
    mk = np.zeros((L, 3), np.float32)
    k = min(L, end)
    if k:
        mk[L - k:] = device_mask(series, end - k, end, elev)
        x[L - k:] = np.where(mk[L - k:] > 0, device_values(series, end - k, end), 0.0)
    doy_h, hour_h = window_calendar(series["t0"], end - L + np.arange(L))
    doy_f, hour_f = window_calendar(series["t0"], end + np.arange(cfg.horizon))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)[None]
    b = dict(lat=t([lat])[0], lon=t([lon])[0], elev=t([elev])[0], x_hist=t(x), mask_hist=t(mk),
             doy_hist=t(doy_h), hour_hist=t(hour_h), doy_fut=t(doy_f), hour_fut=t(hour_f))
    with torch.no_grad():
        return model(b)["q"][0].numpy()


def feed(stream, series, k0, k1):
    """Подать часы ряда в поток через публичный шаг, с контролем качества прибора.

    Args:
        stream: потоковый рантайм.
        series: синтетический ряд со значениями, масками и первым часом.
        k0: первая строка.
        k1: строка после последней.

    Returns:
        Тот же поток.
    """
    x, m = series["x"], series["m"]
    for k in range(k0, k1):
        v = [float(x[k, j]) if m[k, j] > 0 else None for j in range(3)]
        stream.step(*v, series["t0"] + k)
    return stream


def stream_forecast(model, series, end, lat, lon, elev):
    """Потоковый выпуск: поток с начала ряда, выпуск после строки перед ``end``.

    Args:
        model: модель МАЯК.
        series: синтетический ряд.
        end: число поданных строк.
        lat: широта, градусы.
        lon: долгота, градусы.
        elev: высота, м.

    Returns:
        Пара: квантили и поток.
    """
    from mayak.runtime.streaming import StreamingMayak
    s = feed(StreamingMayak(model, lat, lon, elev), series, 0, end)
    q, _ = s.forecast()
    return q, s


def divergence(model, n_issues=8, hours=None, seed=0, lat=52.37, lon=4.9, elev=0.0):
    """Расхождение потока и пакета при выпусках в случайные часы одного длинного ряда.

    Поток идёт по ряду непрерывно. Моменты выпуска выбираются случайно после не меньше
    чем полной истории работы потока. В середине ряда состояние потока сохраняется и
    загружается в новый рантайм; оба рантайма идут дальше и сравниваются на всех
    следующих выпусках.

    Args:
        model: модель.
        n_issues: число выпусков.
        hours: длина ряда, часы; по умолчанию пять длин истории.
        seed: сид ряда и моментов выпуска.
        lat: широта.
        lon: долгота.
        elev: высота, м.

    Returns:
        Словарь: наибольшее расхождение по всем лидам, по каждому лиду, по каждому
        выпуску вместе с его часом и наибольшее расхождение после перезапуска.
    """
    from mayak.runtime.streaming import StreamingMayak
    L, Hh = model.cfg.max_history, model.cfg.horizon
    hours = 5 * L if hours is None else int(hours)
    rng = np.random.default_rng(seed)
    series = synthetic_series(hours, seed=seed)
    ends = np.sort(rng.choice(np.arange(L + 1, hours + 1), size=n_issues, replace=False))
    live = StreamingMayak(model, lat, lon, elev)
    back = None
    per_lead = np.zeros(Hh)
    by_issue, restart = [], 0.0
    done = 0
    for i, end in enumerate(int(e) for e in ends):
        feed(live, series, done, end)
        if back is not None:
            feed(back, series, done, end)
        done = end
        qs = live.forecast()[0]
        qb = batch_forecast(model, series, end, lat, lon, elev)
        d = np.abs(qb - qs).max(-1)
        per_lead = np.maximum(per_lead, d)
        by_issue.append((end, float(d.max())))
        if back is not None:
            restart = max(restart, float(np.abs(back.forecast()[0] - qs).max()))
        if i == n_issues // 2:
            back = StreamingMayak(model, lat, lon, elev)
            back.load_state(live.serialize())
    return dict(n_issues=int(n_issues), hours=hours, batch_stream_max_abs=float(per_lead.max()),
                batch_stream_max_abs_per_lead=per_lead.tolist(),
                batch_stream_by_issue=by_issue, restart_max_abs=restart)


def _flops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn()
    return int(fc.get_total_flops())


def _timed(fn, reps):
    with torch.no_grad():
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
    return (time.perf_counter() - t0) / reps * 1e3


def step_cost(model, reps=200, seed=0):
    """Стоимость часа энкодера: потактовый шаг против пересчёта по буферу рецептивного поля.

    FLOP считаются по матричным операциям и свёрткам, без поэлементных.

    Args:
        model: модель.
        reps: повторов для замера времени.
        seed: сид входа.

    Returns:
        Словарь FLOP и времени на час, мс, и размер буфера энкодера.
    """
    enc, cfg = model.encoder, model.cfg
    g = torch.Generator().manual_seed(seed)
    win = torch.randn(1, cfg.n_channels, cfg.stream_buffer, generator=g)
    x_t = win[..., -1]
    st = enc.init_state(1)
    fl_step = _flops(lambda: enc.step(x_t, st.clone()))
    fl_full = _flops(lambda: enc(win))
    ms_step = _timed(lambda: enc.step(x_t, st), reps)
    ms_full = _timed(lambda: enc(win), reps)
    return dict(receptive_field=cfg.receptive_field, stream_buffer=cfg.stream_buffer,
                encoder_flops_step=fl_step, encoder_flops_full_window=fl_full,
                flops_ratio=fl_full / max(fl_step, 1), encoder_ms_step=ms_step,
                encoder_ms_full_window=ms_full, time_ratio=ms_full / max(ms_step, 1e-9),
                encoder_buffer_bytes=st.nbytes)


def runtime_cost(model, reps=100, seed=0):
    """Стоимость графов потока и пакетного прохода по всей истории.

    Args:
        model: модель.
        reps: повторов для замера времени шага; выпуск и проход по окну меряются реже.
        seed: сид правдоподобных входов.

    Returns:
        Словарь FLOP и времени на вызов, мс, для шага часа, выпуска, прохода по окну при
        загрузке и пакетного прохода модели по всей истории.
    """
    from mayak.runtime.graphs import GRAPH_MODULES, example_inputs
    cfg = model.cfg
    inp = example_inputs(model, seed)
    mods = {n: GRAPH_MODULES[n](model) for n in ("step", "issue", "window")}
    L = cfg.max_history
    g = torch.Generator().manual_seed(seed)
    batch = dict(lat=torch.tensor([52.0]), lon=torch.tensor([5.0]), elev=torch.tensor([0.0]),
                 x_hist=torch.randn(1, L, 3, generator=g) + 10.0,
                 mask_hist=torch.ones(1, L, 3),
                 doy_hist=(100 + torch.arange(L) / 24.0)[None],
                 hour_hist=(torch.arange(L) % 24.0)[None],
                 doy_fut=(130 + torch.arange(cfg.horizon) / 24.0)[None],
                 hour_fut=(torch.arange(cfg.horizon) % 24.0)[None])
    rep = dict(receptive_field=cfg.receptive_field, stream_edge=cfg.stream_edge,
               stream_tail=cfg.stream_tail, stream_window=cfg.stream_window)
    calls = dict(step=lambda: mods["step"](*inp["step"]),
                 issue=lambda: mods["issue"](*inp["issue"]),
                 window=lambda: mods["window"](*inp["window"]),
                 pack=lambda: model(batch))
    for name, fn in calls.items():
        rep[f"{name}_flops"] = _flops(fn)
        rep[f"{name}_ms"] = _timed(fn, reps if name == "step" else max(3, reps // 10))
    return rep


def stream_hour_ms(model, hours=200, seed=0, lat=52.37, lon=4.9, elev=0.0):
    """Полная стоимость шага потока: контроль качества, каналы, энкодер, моды, кольца.

    Args:
        model: модель МАЯК.
        hours: число шагов замера.
        seed: сид синтетического ряда.
        lat: широта, градусы.
        lon: долгота, градусы.
        elev: высота, м.

    Returns:
        Пара: среднее время шага, мс, и поток.
    """
    from mayak.runtime.streaming import StreamingMayak
    series = synthetic_series(hours + 50, seed=seed)
    s = feed(StreamingMayak(model, lat, lon, elev), series, 0, 50)
    t0 = time.perf_counter()
    feed(s, series, 50, 50 + hours)
    return (time.perf_counter() - t0) / hours * 1e3, s


__all__ = ["batch_forecast", "default_start", "device_mask", "device_values", "divergence",
           "feed", "runtime_cost", "step_cost", "stream_forecast", "stream_hour_ms",
           "synthetic_series"]
