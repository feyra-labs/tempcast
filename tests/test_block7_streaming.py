"""Тесты: причинный инкрементальный энкодер и поток, равный пакету.

Что проверяется:
* нормализация энкодера причинна и не зависит от длины окна;
* потактовый шаг энкодера совпадает с пакетным проходом на всей длине окна;
* стоимость шага не зависит от рецептивного поля, буферы - десятки КБ;
* признаки края окна зависят от начала окна, признаки хвоста - нет;
* скользящая сумма мод равна точной сумме, пересинхронизация снимает ошибку округления;
* полный выпуск потока равен пакетному при выпусках в случайные часы длинного ряда;
* перезапуск в любой час продолжает непрерывный прогон, включая решения контроля
  качества; простой заполняется пустыми часами, долгий простой - холодный старт;
* состояние - только сырое окно, размер закреплён числом байт.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from mayak.config import CHANNEL_MAX_LAG, Ablations, ModelConfig
from mayak.constants import H, L_MAX, QUANTILES
from mayak.model import MAYAK
from mayak.modules.encoder import ChannelGroupNorm, SynopticEncoder
from mayak.runtime import streaming as S
from mayak.runtime.equivalence import (batch_forecast, divergence, feed, step_cost,
                                       stream_forecast, synthetic_series)
from mayak.runtime.streaming import (STATE_HEADER, StreamingMayak, decode_window,
                                     encode_window, to_store)

LAT, LON, ELEV = 52.37, 4.9, 0.0
RTOL_STEP = 1e-5
EXACT_F64 = 1e-10
ATOL_FORECAST = 5e-4
DEFAULT_STATE_BYTES = 3224
TINY = dict(encoder_width=16, encoder_dilations=(1, 2, 4, 8), passport_dim=8, field_hidden=24,
            heads_hidden=16, passport_hidden=16)


def _model(cfg=None, seed=0, perturb=True):
    """Модель со «встряхнутыми» аффинными параметрами нормализации.

    При инициализации weight = 1, bias = 0 - часть ошибок индексации в шаге была бы
    не видна; случайные значения делают сравнение строже.
    """
    torch.manual_seed(seed)
    m = MAYAK(cfg).eval()
    if perturb:
        with torch.no_grad():
            for b in m.encoder.blocks:
                b.norm.weight.add_(0.3 * torch.randn_like(b.norm.weight))
                b.norm.bias.add_(0.3 * torch.randn_like(b.norm.bias))
    return m


@pytest.fixture(scope="module")
def model():
    return _model()


def _ring(state, encoder):
    """Буферы энкодера в хронологическом порядке (старший час первым)."""
    return [buf[:, :, torch.arange(state.t - b.pad, state.t) % b.pad]
            for buf, b in zip(state.bufs, encoder.blocks)]


def _scale(*tensors):
    return max(1.0, *(float(t.abs().max()) for t in tensors if t.numel()))


def _step_tol(*ref):
    """Допуск float32 в единицах масштаба опорных значений."""
    return RTOL_STEP * _scale(*ref)


def _ring_diff(a, b, encoder):
    """Расхождение кольцевых буферов, отнесённое к их масштабу (сравнивать с RTOL_STEP)."""
    pairs = list(zip(_ring(a, encoder), _ring(b, encoder)))
    err = max((x - y).abs().max().item() for x, y in pairs) if pairs else 0.0
    return err / _scale(*(x for x, _ in pairs)) if pairs else 0.0


def test_encoder_has_no_time_axis_groupnorm(model):
    assert not any(isinstance(mod, nn.GroupNorm) for mod in model.encoder.modules())
    assert all(isinstance(b.norm, ChannelGroupNorm) for b in model.encoder.blocks)


def test_channel_group_norm_is_per_timestep():
    torch.manual_seed(0)
    n = ChannelGroupNorm(4, 48)
    with torch.no_grad():
        n.weight.normal_()
        n.bias.normal_()
    x = torch.randn(3, 48, 50)
    y = n(x)
    for t in (0, 17, 49):
        torch.testing.assert_close(y[..., t], n(x[..., t]), atol=1e-6, rtol=0)
    x2 = x.clone()
    x2[..., 30:] += 100.0
    torch.testing.assert_close(n(x2)[..., :30], y[..., :30], atol=0, rtol=0)


def test_encoder_causal_and_window_length_independent(model):
    enc = model.encoder
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, model.cfg.n_channels, 400, generator=g)
    with torch.no_grad():
        full = enc(x)
        x_fut = x.clone()
        x_fut[..., 300:] = 50 * torch.randn(2, model.cfg.n_channels, 100, generator=g)
        torch.testing.assert_close(enc(x_fut)[:, :300], full[:, :300], atol=0, rtol=0)
        rf = enc.receptive_field
        tail = enc(x[..., 400 - rf - 50:])
        torch.testing.assert_close(tail[:, -50:], full[:, -50:], atol=_step_tol(full), rtol=0)


def test_old_time_axis_groupnorm_checkpoint_is_rejected(model):
    sd = model.state_dict()
    old = {}
    for k, v in sd.items():
        old[k.replace(".norm.", ".gn.")] = v
    fresh = MAYAK()
    with pytest.raises(RuntimeError, match="переобучить"):
        fresh.load_state_dict(old)


@pytest.mark.parametrize("kw", [
    {},
    dict(encoder_dilations=(1, 2, 4), encoder_width=16),
    dict(encoder_kernel=2, encoder_dilations=(1, 3, 9)),
    dict(encoder_kernel=4, encoder_dilations=(1, 2, 5), encoder_norm_groups=1),
    dict(ablations=Ablations(no_solar=True)),
])
def test_encoder_step_matches_batch_whole_window(kw):
    m = _model(ModelConfig(**kw))
    enc = m.encoder
    x = torch.randn(2, m.cfg.n_channels, L_MAX, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = enc(x)
        st = enc.init_state(2)
        out = torch.stack([enc.step(x[..., k], st) for k in range(L_MAX)], dim=1)
    err = (out - ref).abs().max().item()
    assert err < _step_tol(ref), (f"потактовый энкодер ≠ пакетный: max|Δ| = {err:.3g} "
                                  f"при max|ref| = {ref.abs().max():.3g}")
    assert st.t == L_MAX


@pytest.mark.parametrize("kw", [
    {},
    dict(encoder_dilations=(1, 2, 4), encoder_width=16),
    dict(encoder_kernel=2, encoder_dilations=(1, 3, 9)),
    dict(encoder_kernel=4, encoder_dilations=(1, 2, 5), encoder_norm_groups=1),
])
def test_encoder_step_is_exact_in_float64(kw):
    """Алгоритмическая эквивалентность без шума округления float32: не зависит от CPU."""
    m = _model(ModelConfig(**kw))
    enc = m.encoder.double()
    x = torch.randn(2, m.cfg.n_channels, L_MAX, generator=torch.Generator().manual_seed(2),
                    dtype=torch.float64)
    with torch.no_grad():
        ref = enc(x)
        st = enc.init_state(2)
        out = torch.stack([enc.step(x[..., k], st) for k in range(L_MAX)], dim=1)
        pre_out, pre = enc.prefill(x[..., :300])
        rest = torch.stack([enc.step(x[..., k], pre) for k in range(300, L_MAX)], dim=1)
    assert out.dtype == torch.float64
    assert (out - ref).abs().max().item() < EXACT_F64
    assert (pre_out - ref[:, :300]).abs().max().item() < EXACT_F64
    assert (rest - ref[:, 300:]).abs().max().item() < EXACT_F64


@pytest.mark.parametrize("split", [0, 1, 37, 63, 64, 300, L_MAX])
def test_prefill_equals_stepping(model, split):
    enc = model.encoder
    x = torch.randn(1, model.cfg.n_channels, L_MAX, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = enc(x)
        stepped = enc.init_state(1)
        for k in range(split):
            enc.step(x[..., k], stepped)
        pre_out, pre = enc.prefill(x[..., :split])
        assert pre.t == stepped.t == split
        assert _ring_diff(pre, stepped, enc) < RTOL_STEP
        if split:
            torch.testing.assert_close(pre_out, ref[:, :split], atol=_step_tol(ref), rtol=0)
        rest = [enc.step(x[..., k], pre) for k in range(split, L_MAX)]
    if rest:
        err = (torch.stack(rest, 1) - ref[:, split:]).abs().max().item()
        assert err < _step_tol(ref)


def test_step_cost_does_not_depend_on_receptive_field():
    short = _model(ModelConfig(encoder_dilations=(1,) * 12))
    long = _model(ModelConfig(encoder_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                                                 2048)))
    a, b = step_cost(short, reps=3), step_cost(long, reps=3)
    assert a["encoder_flops_step"] == b["encoder_flops_step"] > 0
    assert b["receptive_field"] > 100 * a["receptive_field"]
    assert b["encoder_buffer_bytes"] > 100 * a["encoder_buffer_bytes"]


def test_step_cost_vs_full_window_recompute(model, record_property):
    c = step_cost(model, reps=5)
    record_property("encoder_flops_ratio", c["flops_ratio"])
    cfg = model.cfg
    per_block = 2 * cfg.encoder_width ** 2
    assert c["encoder_flops_step"] == (len(cfg.encoder_dilations) * per_block
                                       + 2 * cfg.n_channels * cfg.encoder_width)
    assert c["flops_ratio"] == pytest.approx(cfg.stream_buffer, rel=0.1)
    assert c["encoder_buffer_bytes"] == 4 * cfg.encoder_width * 2 * sum(cfg.encoder_dilations)
    assert 10_000 < c["encoder_buffer_bytes"] < 100_000


def test_stream_layout_formulas():
    c = ModelConfig()
    assert c.receptive_field == 2 * sum(c.encoder_dilations) + 1 == 253
    assert SynopticEncoder(dilations=c.encoder_dilations).receptive_field == 253
    assert c.stream_edge == c.receptive_field - 1 + CHANNEL_MAX_LAG == 276
    assert c.stream_tail == L_MAX - c.stream_edge == 396
    assert c.stream_window == L_MAX == 672
    assert c.stream_window % 8 == 0
    from mayak.data.qc import DEFAULT_QC
    assert c.stream_window > DEFAULT_QC.lookback_hours
    long = ModelConfig(encoder_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256))
    assert long.stream_edge == L_MAX and long.stream_tail == 0
    assert long.stream_window >= long.receptive_field - 1 + CHANNEL_MAX_LAG


def test_channels_have_finite_memory_of_channel_max_lag(model):
    """Каналы часа зависят только от сырых часов не старше наибольшего лага."""
    s = synthetic_series(120, seed=4, p_valid=1.0)
    x = torch.from_numpy(s["x"])[None]
    mk = torch.from_numpy(s["m"])[None]
    from mayak.astro import astro_features
    astro = astro_features(torch.from_numpy(s["doy"])[None], torch.from_numpy(s["hour"])[None],
                           torch.tensor([[LAT]]), torch.tensor([[LON]]))
    loc = model.loc(torch.tensor([LAT]), torch.tensor([LON]), torch.tensor([ELEV]))
    mu0, sg0, df0 = model.field.evaluate(model.field.coefficients(loc), astro)

    def ch(xx):
        with torch.no_grad():
            return model.build_channels(xx, mk, astro, mu0, sg0, df0)[0][..., -1]

    base = ch(x)
    far = x.clone()
    far[:, :-(CHANNEL_MAX_LAG + 1)] += 7.0
    torch.testing.assert_close(ch(far), base, atol=0, rtol=0)
    near = x.clone()
    near[:, -(CHANNEL_MAX_LAG + 1), 1] += 7.0
    assert not torch.equal(ch(near), base)


@pytest.mark.parametrize("n", [5, 60, 300])
def test_forward_with_buffer_matches_prefill(model, n):
    enc = model.encoder
    x = torch.randn(1, model.cfg.n_channels, n, generator=torch.Generator().manual_seed(n))
    with torch.no_grad():
        feats, buf = enc.forward_with_buffer(x)
        ref, st = enc.prefill(x)
    torch.testing.assert_close(feats, ref, atol=0, rtol=0)
    torch.testing.assert_close(buf, enc.ring_to_shift(st), atol=0, rtol=0)


def _history(model, x, m, t0):
    from mayak.astro import astro_features
    from mayak.timeaxis import window_calendar
    doy, hour = window_calendar(t0, np.arange(len(x)))
    astro = astro_features(torch.from_numpy(doy)[None], torch.from_numpy(hour)[None],
                           torch.tensor([[LAT]]), torch.tensor([[LON]]))
    loc = model.loc(torch.tensor([LAT]), torch.tensor([LON]), torch.tensor([ELEV]))
    with torch.no_grad():
        return model.history_pass(torch.from_numpy(x)[None], torch.from_numpy(m)[None], astro,
                                  model.field.coefficients(loc))


def test_edge_features_depend_on_window_start_tail_features_do_not(model):
    """Почему край пересчитывается при выпуске: у пакета признаки ранних часов окна
    зависят от того, что было до окна, у поздних - нет."""
    cfg = model.cfg
    s = synthetic_series(L_MAX + 200, seed=21, p_valid=1.0)
    x, m = s["x"], s["m"]
    long = _history(model, x, m, s["t0"])["u"][0, 200:]
    win = _history(model, x[200:], m[200:], s["t0"] + 200)["u"][0]
    E = cfg.stream_edge
    tail_err = float((long[E:] - win[E:]).abs().max())
    edge_err = float((long[:E] - win[:E]).abs().max())
    assert tail_err < _step_tol(long), tail_err
    assert edge_err > 1e-2, edge_err
    assert float((long[E - 1] - win[E - 1]).abs().max()) > 0


def test_sliding_sum_equals_exact_sum_in_float64():
    m = _model(ModelConfig(**TINY)).double()
    ro = m.readout
    M, span = ro.n_modes, 40
    g = torch.Generator().manual_seed(5)
    u = torch.randn(1, 300, 2 * M, generator=g, dtype=torch.float64)
    v = (torch.rand(1, 300, generator=g) < 0.8).double()
    state = [torch.zeros(1, M, dtype=torch.float64) for _ in range(3)]
    for t in range(300):
        old = t - span
        u_old = u[:, old] if old >= 0 else torch.zeros(1, 2 * M, dtype=torch.float64)
        v_old = v[:, old:old + 1] if old >= 0 else torch.zeros(1, 1, dtype=torch.float64)
        state = ro.step_window(state, u[:, t], v[:, t:t + 1], u_old, v_old, span)
        if t in (10, 39, 40, 41, 299):
            lo = max(0, t - span + 1)
            ref = ro.accumulate(u[:, lo:t + 1], v[:, lo:t + 1])
            for a, b in zip(state, ref):
                assert float((a - b).abs().max()) < EXACT_F64 * 1e2


@pytest.mark.parametrize("flag", [None, "no_solar", "no_mode_groups", "no_passport",
                                  "no_compression"])
def test_full_forecast_batch_equals_stream_at_random_hours(flag, record_property):
    cfg = ModelConfig(ablations=Ablations(**({flag: True} if flag else {})))
    m = _model(cfg)
    rep = divergence(m, n_issues=4, hours=3 * L_MAX, seed=7)
    per_lead = np.asarray(rep["batch_stream_max_abs_per_lead"])
    record_property("batch_stream_max_abs", rep["batch_stream_max_abs"])
    record_property("restart_max_abs", rep["restart_max_abs"])
    assert per_lead.shape == (H,)
    assert all(end > L_MAX for end, _ in rep["batch_stream_by_issue"])
    assert rep["batch_stream_max_abs"] < ATOL_FORECAST, (
        f"пакет и поток расходятся: max|Δq| = {rep['batch_stream_max_abs']:.3g} °C, "
        f"худший лид {int(per_lead.argmax()) + 1} ч")
    assert rep["restart_max_abs"] < ATOL_FORECAST


@pytest.mark.parametrize("end", [1, 100, 500])
def test_full_forecast_short_history(model, end):
    """Короткая история: поток с холодного старта равен пакетному окну с пустыми часами."""
    s = synthetic_series(600, seed=8)
    qb = batch_forecast(model, s, end, LAT, LON, ELEV)
    qs, _ = stream_forecast(model, s, end, LAT, LON, ELEV)
    assert qs.shape == (H, len(QUANTILES))
    assert np.abs(qb - qs).max() < ATOL_FORECAST


def test_long_run_random_hours_error_does_not_grow(record_property):
    """Выпуски в случайные часы ряда длиной десять тысяч часов: расхождение с пакетом
    в допуске и не растёт со временем."""
    m = _model(ModelConfig(**TINY))
    rep = divergence(m, n_issues=12, hours=10_000, seed=3)
    errs = np.array([e for _, e in rep["batch_stream_by_issue"]])
    record_property("long_run_max_abs", rep["batch_stream_max_abs"])
    assert rep["batch_stream_max_abs"] < ATOL_FORECAST
    assert rep["restart_max_abs"] < ATOL_FORECAST
    assert errs[-4:].max() <= 3 * errs[:4].max() + 1e-5, errs


def _mode_error(st):
    exact = st.b.run("resync", st.u_ring[st._hours(st.tail) % st.tail][None],
                     st.v_ring[st._hours(st.tail) % st.tail][None])
    return max(float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))
               for a, b in zip(st.modes, exact))


def test_resync_bounds_rounding_error(monkeypatch):
    """Без пересинхронизации ошибка скользящей суммы остаётся малой и не растёт;
    пересинхронизация обнуляет её раз в сутки."""
    m = _model(ModelConfig(**TINY))
    s = synthetic_series(4000, seed=12)
    monkeypatch.setattr(S, "RESYNC_HOURS", 10 ** 9)
    free = StreamingMayak(m, LAT, LON, ELEV)
    errs = []
    for k in range(0, 4000, 500):
        feed(free, s, k, k + 500)
        errs.append(_mode_error(free))
    assert max(errs) < 1e-5, errs
    assert max(errs[-3:]) <= 3 * max(errs[:3]) + 1e-7, errs
    monkeypatch.setattr(S, "RESYNC_HOURS", 24)
    synced = StreamingMayak(m, LAT, LON, ELEV)
    end = 1000 - (s["t0"] + 1000) % 24
    feed(synced, s, 0, end)
    assert (synced.last_hour + 1) % 24 == 0
    assert _mode_error(synced) == 0.0


@pytest.mark.parametrize("cut", [30, 24 * 12 + 13, L_MAX + 5, 2 * L_MAX + 17])
def test_restart_at_any_hour_equals_continuous_run(model, cut):
    s = synthetic_series(cut + 60, seed=11)
    live = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, cut)
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(live.serialize())
    assert back.filled == live.filled == min(cut, model.cfg.stream_window)
    assert back.last_hour == live.last_hour
    np.testing.assert_allclose(back.enc_buf, live.enc_buf, atol=1e-5 * max(1, np.abs(
        live.enc_buf).max()))
    order = live._hours(L_MAX) % L_MAX
    rb, rl = back.rows[order], live.rows[order]
    np.testing.assert_allclose(rb[24:], rl[24:], atol=1e-5)
    np.testing.assert_allclose(rb[:24, [0, 2]], rl[:24, [0, 2]], atol=1e-5)
    done = cut
    for end in (cut, cut + 7, cut + 60):
        feed(live, s, done, end)
        feed(back, s, done, end)
        done = end
        err = np.abs(live.forecast()[0] - back.forecast()[0]).max()
        assert err < ATOL_FORECAST, f"час {end}: max|Δq| = {err:.3g}"
    assert back.serialize() == live.serialize()


def test_restart_keeps_quality_control_decisions(model):
    """Кольцо контроля качества после перезапуска видит отбракованные значения так же,
    как непрерывный прогон: решения о следующих часах не меняются."""
    s = synthetic_series(400, seed=14, p_valid=1.0)
    s["x"][250:330, 0] = 7.0
    s["x"][250:330, 2] = 64.0
    s["x"][299, 2] = -5.0
    s["x"][298, 1] = 7000.0
    cut = 300
    live = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, cut)
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(live.serialize())
    feed(live, s, cut, 400)
    feed(back, s, cut, 400)
    pos = live._hours(100) % live.window
    assert np.array_equal(live.valid[pos], back.valid[pos])
    assert live.valid[pos, 0].min() == 0, "в ряду должно быть залипание температуры"
    assert back.serialize() == live.serialize()


def test_idle_gap_is_filled_with_empty_hours(model):
    s = synthetic_series(200, seed=15)
    a = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 120)
    b = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 120)
    for k in range(120, 127):
        b.step(None, None, None, s["t0"] + k)
    feed(a, s, 127, 200)
    feed(b, s, 127, 200)
    assert a.idle_hours == 7 and b.idle_hours == 0
    assert a.serialize() == b.serialize()
    np.testing.assert_allclose(a.forecast()[0], b.forecast()[0], atol=1e-5)


def test_idle_longer_than_window_is_cold_start(model):
    p = S.ACIParams()
    s = synthetic_series(100, seed=16)
    a = feed(StreamingMayak(model, LAT, LON, ELEV, aci=p), s, 0, 100)
    a.theta = 0.3
    later = s["t0"] + 100 + model.cfg.stream_window + 3
    a.step(8.0, 1003.0, 71.0, later)
    fresh = StreamingMayak(model, LAT, LON, ELEV, aci=p)
    fresh.step(8.0, 1003.0, 71.0, later)
    assert a.filled == 1 and a.theta == pytest.approx(0.3)
    np.testing.assert_allclose(a.raw_forecast(), fresh.raw_forecast(), atol=1e-5)


def test_hours_must_increase_and_forecast_needs_a_step(model):
    st = StreamingMayak(model, LAT, LON, ELEV)
    with pytest.raises(ValueError, match="момент выпуска"):
        st.forecast()
    st.step(8.0, 1003.0, 71.0, 1000)
    with pytest.raises(ValueError, match="не позже"):
        st.step(8.0, 1003.0, 71.0, 1000)


def test_warm_start_equals_stepping(model):
    L = 24 * 10 + 13
    s = synthetic_series(L, seed=10)
    stepped = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, L)
    warm = StreamingMayak(model, LAT, LON, ELEV)
    warm.warm_start(s["x"], s["m"], s["t0"] + L - 1)
    assert warm.serialize() == stepped.serialize()
    np.testing.assert_allclose(warm.forecast()[0], stepped.forecast()[0], atol=ATOL_FORECAST)


def test_state_size_is_pinned(model):
    st = StreamingMayak(model, LAT, LON, ELEV)
    assert st.state_nbytes == len(st.serialize()) == DEFAULT_STATE_BYTES < 4096
    assert STATE_HEADER.itemsize == 32
    feed(st, synthetic_series(100, seed=14), 0, 100)
    assert len(st.serialize()) == DEFAULT_STATE_BYTES


def test_state_header_and_roundtrip(model):
    s = synthetic_series(400, seed=15)
    st = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 400)
    raw = st.serialize()
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    assert int(hdr["last_hour"]) == s["t0"] + 399 and int(hdr["filled"]) == 400
    assert (float(hdr["lat"]), float(hdr["lon"])) == (np.float32(LAT), np.float32(LON))
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(raw)
    assert back.serialize() == raw
    assert back.loaded_site == (np.float32(LAT), np.float32(LON), np.float32(ELEV))
    fresh = StreamingMayak(model, LAT, LON, ELEV)
    fresh.load_state(StreamingMayak(model, LAT, LON, ELEV).serialize())
    assert fresh.last_hour is None


def test_window_storage_is_lossless_on_record_grid():
    rng = np.random.default_rng(0)
    x = np.stack([rng.integers(-90, 61, 800), rng.integers(3000, 11001, 800) / 10.0,
                  rng.integers(0, 101, 800)], -1).astype(np.float32)
    present = (rng.random((800, 3)) < 0.9).astype(np.uint8)
    valid = present * (rng.random((800, 3)) < 0.9)
    buf = encode_window(np.where(present > 0, x, 0.0), present, valid)
    assert len(buf) == 800 * 4 + 2 * 300
    got, p2, v2 = decode_window(buf, 800)
    assert np.array_equal(p2, present) and np.array_equal(v2, valid)
    assert np.array_equal(got[present > 0], x[present > 0])
    wild = to_store(np.array([[-200.0, 7000.0, 300.0], [75.0, -3.0, -5.0]]))
    lo = np.array([-90.0, 300.0, 0.0])
    hi = np.array([60.0, 1100.0, 100.0])
    assert ((wild < lo) | (wild > hi)).all(), "значение вне диапазона остаётся вне него"


def test_invalid_states_are_rejected(model):
    st = feed(StreamingMayak(model, LAT, LON, ELEV), synthetic_series(60, seed=16), 0, 60)
    raw = st.serialize()
    fresh = StreamingMayak(model, LAT, LON, ELEV)
    with pytest.raises(ValueError, match="заголовка"):
        fresh.load_state(bytes(len(raw)))
    bad = bytearray(raw)
    bad[3] = 3
    with pytest.raises(ValueError, match="версия"):
        fresh.load_state(bytes(bad))
    with pytest.raises(ValueError, match="конфигу"):
        fresh.load_state(raw[:-1])
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0].copy()
    hdr["filled"] = 5000
    with pytest.raises(ValueError, match="заголовок"):
        fresh.load_state(hdr.tobytes() + raw[STATE_HEADER.itemsize:])
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0].copy()
    hdr["aci_theta"] = np.nan
    with pytest.raises(ValueError, match="калибровки"):
        fresh.load_state(hdr.tobytes() + raw[STATE_HEADER.itemsize:])
    W = model.cfg.stream_window
    body = bytearray(raw[STATE_HEADER.itemsize:])
    off = 4 * W
    body[off:off + S.mask_bytes(W)] = bytes(S.mask_bytes(W))
    with pytest.raises(ValueError, match="годный час без значения"):
        fresh.load_state(raw[:STATE_HEADER.itemsize] + bytes(body))
    assert fresh.last_hour is None
