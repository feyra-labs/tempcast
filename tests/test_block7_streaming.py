"""ТЕсты: строго причинный инкрементальный энкодер и честный O(1) на час.

Что проверяется:
* нормализация энкодера причинна и не зависит от длины окн;
* потактовый шаг энкодера совпадает с пакетным проходом на всей длине окна;
* стоимость шага не зависит от рецептивного поля, буферы - десятки КБ;
* рецептивное поле и окно рантайма - из формулы, окно достаточно и минимально;
* полный выпуск прогноза: пакет = поток на всех лидах, расхождение - числом;
* перезапуск из сохранённого состояния продолжает непрерывный прогон, включая
  незавершённые сутки; формат состояния закреплён числом байт.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from mayak.config import CHANNEL_MAX_LAG, Ablations, ModelConfig
from mayak.constants import H, L_MAX, QUANTILES
from mayak.model import MAYAK
from mayak.modules.encoder import ChannelGroupNorm, SynopticEncoder
from mayak.runtime.equivalence import (batch_forecast, divergence, feed, future_calendar_after,
                                       step_cost, stream_forecast, synthetic_series)
from mayak.runtime.streaming import (RAW_STEP, STATE_HEADER, StreamingMayak, decode_raw,
                                     encode_raw)

LAT, LON, ELEV = 52.37, 4.9, 0.0
ATOL_STEP = 1e-5
ATOL_FORECAST = 1e-4
ATOL_RESTART = 2e-3
DEFAULT_STATE_BYTES = 3348


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


def _ring_diff(a, b, encoder):
    return max((x - y).abs().max().item() for x, y in zip(_ring(a, encoder), _ring(b, encoder)))


def _grid(series):
    """Ряд, значения которого лежат на сетке квантования окна на диске."""
    s = dict(series)
    s["x"] = decode_raw(encode_raw(series["x"], series["m"]), series["m"])
    return s


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
        torch.testing.assert_close(tail[:, -50:], full[:, -50:], atol=ATOL_STEP, rtol=0)


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
    assert err < ATOL_STEP, f"потактовый энкодер ≠ пакетный: max|Δ| = {err:.3g}"
    assert st.t == L_MAX


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
        assert _ring_diff(pre, stepped, enc) < ATOL_STEP
        if split:
            torch.testing.assert_close(pre_out, ref[:, :split], atol=ATOL_STEP, rtol=0)
        rest = [enc.step(x[..., k], pre) for k in range(split, L_MAX)]
    if rest:
        err = (torch.stack(rest, 1) - ref[:, split:]).abs().max().item()
        assert err < ATOL_STEP


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


def test_receptive_field_and_window_formulas():
    c = ModelConfig()
    assert c.receptive_field == 2 * sum(c.encoder_dilations) + 1 == 253
    assert SynopticEncoder(dilations=c.encoder_dilations).receptive_field == 253
    assert c.stream_window % 16 == 0
    assert c.stream_window >= c.receptive_field - 1 + CHANNEL_MAX_LAG
    assert c.stream_window == 288


def test_channels_have_finite_memory_of_channel_max_lag(model):
    """Каналы в момент t зависят только от сырых часов t − CHANNEL_MAX_LAG … t."""
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


def test_restore_window_is_sufficient_and_minimal():
    """По (RF − 1) + CHANNEL_MAX_LAG часам буферы восстанавливаются точно, по часу меньше - нет."""
    cfg = ModelConfig(encoder_dilations=(1, 2))
    m = _model(cfg)
    s = _grid(synthetic_series(200, seed=5, p_valid=1.0))
    live = feed(StreamingMayak(m, LAT, LON, ELEV), s, 0, 150)
    need = cfg.receptive_field - 1 + CHANNEL_MAX_LAG
    assert cfg.stream_window >= need
    diffs = {}
    for w in (cfg.stream_window, need, need - 1):
        r = StreamingMayak(m, LAT, LON, ELEV)
        r.load_state(live.serialize())
        r.filled = w
        r._rebuild_from_window()
        diffs[w] = _ring_diff(live.enc, r.enc, m.encoder)
    assert diffs[cfg.stream_window] < ATOL_STEP and diffs[need] < ATOL_STEP, diffs
    assert diffs[need - 1] > 1e-3, diffs


@pytest.mark.parametrize("flag", [None, "no_solar", "no_mode_groups", "no_passport"])
def test_full_forecast_batch_equals_stream_all_leads(flag, record_property):
    cfg = ModelConfig(ablations=Ablations(**({flag: True} if flag else {})))
    m = _model(cfg)
    rep = divergence(m, n_windows=2, seed=7)
    per_lead = np.asarray(rep["batch_stream_max_abs_per_lead"])
    record_property("batch_stream_max_abs", rep["batch_stream_max_abs"])
    record_property("restart_max_abs", rep["restart_max_abs"])
    assert per_lead.shape == (H,)
    assert rep["batch_stream_max_abs"] < ATOL_FORECAST, (
        f"пакет ≠ поток: max|Δq| = {rep['batch_stream_max_abs']:.3g} °C, худший лид "
        f"{int(per_lead.argmax()) + 1} ч")
    assert rep["restart_max_abs"] < ATOL_RESTART


@pytest.mark.parametrize("end", [0, 1, 100, 500])
def test_full_forecast_short_history(model, end):
    """Короткая история: поток с нуля = пакетное окно с нулями и нулевой маской."""
    s = synthetic_series(600, seed=8)
    qb = batch_forecast(model, s, end, LAT, LON, ELEV)
    qs, _ = stream_forecast(model, s, end, LAT, LON, ELEV)
    assert qs.shape == (H, len(QUANTILES))
    assert np.abs(qb - qs).max() < ATOL_FORECAST


def test_stream_equivalence_over_long_sequence(model):
    from mayak.astro import astro_features
    n = int(2.5 * L_MAX)
    s = synthetic_series(n, seed=9)
    st = StreamingMayak(model, LAT, LON, ELEV)
    done = 0
    for end in (L_MAX, 2 * L_MAX, n):
        feed(st, s, done, end)
        done = end
        x = torch.from_numpy(s["x"][:end])[None]
        mk = torch.from_numpy(s["m"][:end])[None]
        astro = astro_features(torch.from_numpy(s["doy"][:end])[None],
                               torch.from_numpy(s["hour"][:end])[None],
                               torch.tensor([[LAT]]), torch.tensor([[LON]]))
        with torch.no_grad():
            mu0, sg0, df0 = model.field.evaluate(st.base_coefs, astro)
            ch, _, vt = model.build_channels(x, mk, astro, mu0, sg0, df0)
            feats, ref_state = model.encoder.prefill(ch)
            a_re, a_im, e = model.readout(feats, vt)
            s_re, s_im = model.readout.normalize(st.n_re, st.n_im, st.e)
        assert _ring_diff(ref_state, st.enc, model.encoder) < ATOL_STEP
        assert (s_re - a_re).abs().max() < 1e-4 and (s_im - a_im).abs().max() < 1e-4
        assert ((st.e - e).abs() / e.abs().clamp(min=1.0)).max() < 1e-4


def test_warm_start_equals_stepping_with_partial_day(model):
    L = 24 * 10 + 13
    s = synthetic_series(L, seed=10)
    stepped = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, L)
    warm = StreamingMayak(model, LAT, LON, ELEV)
    warm.warm_start(s["x"], s["m"], s["doy"], s["hour"])
    assert warm._hours_in_day == stepped._hours_in_day == 13
    np.testing.assert_allclose(warm._day, stepped._day, atol=1e-5)
    assert _ring_diff(warm.enc, stepped.enc, model.encoder) < ATOL_STEP
    cal = future_calendar_after(s, L, H)
    np.testing.assert_allclose(warm.forecast(*cal)[0], stepped.forecast(*cal)[0], atol=ATOL_FORECAST)
    assert warm.serialize()[:STATE_HEADER.itemsize] == stepped.serialize()[:STATE_HEADER.itemsize]


@pytest.mark.parametrize("cut", [30, 24 * 12, 24 * 12 + 13, L_MAX + 5])
def test_restart_continues_continuous_run(model, cut):
    """После восстановления из окна выход совпадает с непрерывным прогоном."""
    s = _grid(synthetic_series(cut + 60, seed=11))
    live = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, cut)
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(live.serialize())
    assert back.filled == min(cut, model.cfg.stream_window)
    assert back._hours_in_day == live._hours_in_day == cut % 24
    np.testing.assert_allclose(back._day, live._day, atol=1e-5)
    assert _ring_diff(live.enc, back.enc, model.encoder) < ATOL_STEP
    done = cut
    for end in (cut, cut + 7, cut + 60):
        feed(live, s, done, end)
        feed(back, s, done, end)
        done = end
        cal = future_calendar_after(s, end, H)
        err = np.abs(live.forecast(*cal)[0] - back.forecast(*cal)[0]).max()
        assert err < ATOL_FORECAST, f"час {end}: max|Δq| = {err:.3g}"
    np.testing.assert_allclose(back.day_summ.numpy(), live.day_summ.numpy(), atol=2e-3)
    np.testing.assert_allclose(back.z.numpy(), live.z.numpy(), atol=1e-3)


def test_restart_with_quantized_window_is_within_tolerance(model, record_property):
    """На произвольных (не сеточных) данных расхождение даёт только квантование окна."""
    s = synthetic_series(L_MAX + 50, seed=12)
    live = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, L_MAX)
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(live.serialize())
    feed(live, s, L_MAX, L_MAX + 50)
    feed(back, s, L_MAX, L_MAX + 50)
    cal = future_calendar_after(s, L_MAX + 50, H)
    err = float(np.abs(live.forecast(*cal)[0] - back.forecast(*cal)[0]).max())
    record_property("restart_quantized_max_abs", err)
    assert err < ATOL_RESTART


def test_partial_day_survives_restart_and_completes_full(model):
    """7.5: незавершённые сутки не теряются - сводка после перезапуска по полному дню."""
    s = _grid(synthetic_series(24 * 6, seed=13, p_valid=1.0))
    cut = 24 * 5 + 9
    live = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, cut)
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(live.serialize())
    feed(live, s, cut, 24 * 6)
    feed(back, s, cut, 24 * 6)
    assert back._hours_in_day == live._hours_in_day == 0
    frac_valid = back.day_summ[0, -1, 4].item()
    assert frac_valid == pytest.approx(1.0)
    np.testing.assert_allclose(back.day_summ[0, -1].numpy(), live.day_summ[0, -1].numpy(),
                               atol=1e-4)


def test_state_size_is_pinned(model):
    st = StreamingMayak(model, LAT, LON, ELEV)
    assert st.state_nbytes == len(st.serialize()) == DEFAULT_STATE_BYTES < 4096
    feed(st, synthetic_series(100, seed=14), 0, 100)
    assert len(st.serialize()) == DEFAULT_STATE_BYTES


def test_serialize_roundtrip_is_stable(model):
    s = synthetic_series(400, seed=15)
    st = feed(StreamingMayak(model, LAT, LON, ELEV), s, 0, 400)
    raw = st.serialize()
    back = StreamingMayak(model, LAT, LON, ELEV)
    back.load_state(raw)
    assert back.serialize() == raw


def test_raw_window_quantization_resolution():
    rng = np.random.default_rng(0)
    x = np.stack([rng.uniform(-90, 60, 1000), rng.uniform(300, 1100, 1000),
                  rng.uniform(0, 100, 1000)], -1).astype(np.float32)
    m = np.ones_like(x)
    m[::7, 1] = 0
    d = decode_raw(encode_raw(x, m), m)
    err = np.abs(d - x * m).max(0)
    assert (err <= RAW_STEP / 2 + 1e-4).all()
    assert RAW_STEP[1] < 0.02
    assert (d[m == 0] == 0).all()
    assert np.array_equal(encode_raw(d, m), encode_raw(x, m))


def test_invalid_states_are_rejected(model):
    st = feed(StreamingMayak(model, LAT, LON, ELEV), synthetic_series(60, seed=16), 0, 60)
    raw = bytearray(st.serialize())
    fresh = StreamingMayak(model, LAT, LON, ELEV)
    with pytest.raises(ValueError, match="блока 7"):
        fresh.load_state(bytes(3048))
    bad = bytearray(raw)
    bad[3] = 1
    with pytest.raises(ValueError, match="версия"):
        fresh.load_state(bytes(bad))
    hdr = np.frombuffer(bytes(raw), STATE_HEADER, count=1)[0].copy()
    hdr["hoy_last"] = (int(hdr["hoy_last"]) + 5) % 8760
    with pytest.raises(ValueError, match="календарь"):
        fresh.load_state(hdr.tobytes() + bytes(raw[STATE_HEADER.itemsize:]))
    hdr = np.frombuffer(bytes(raw), STATE_HEADER, count=1)[0].copy()
    hdr["hours_in_day"] = 30
    with pytest.raises(ValueError, match="курсор"):
        fresh.load_state(hdr.tobytes() + bytes(raw[STATE_HEADER.itemsize:]))


def test_window_calendar_across_new_year_and_leap_year():
    for year_h in (8760, 8784):
        n = 100
        first = year_h - 30
        doy, hour = StreamingMayak._window_calendar(n, first, (first + n - 1) % year_h)
        hoy = np.rint(doy * 24).astype(int)
        assert hoy[0] == first and hoy[29] == year_h - 1 and hoy[30] == 0 and hoy[-1] == n - 31
        assert np.array_equal(hour, hoy % 24)


def test_calendar_gap_is_counted(model):
    st = StreamingMayak(model, LAT, LON, ELEV)
    st.step(10.0, 1000.0, 70.0, np.float32(100.0), np.float32(0.0))
    st.step(10.0, 1000.0, 70.0, np.float32(100 + 1 / 24), np.float32(1.0))
    assert st.calendar_breaks == 0
    st.step(10.0, 1000.0, 70.0, np.float32(100 + 3 / 24), np.float32(3.0))
    assert st.calendar_breaks == 1
