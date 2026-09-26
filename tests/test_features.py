"""Тесты: общий построитель входных признаков."""
import math

import pytest
import torch

from mayak.astro import astro_features
from mayak.baselines import NEURAL, GRUSeq2Seq, LRUForecaster
from mayak.config import ENCODER_CHANNELS, SOLAR_CHANNELS, Ablations, ModelConfig
from mayak.constants import H, L_MAX
from mayak.features import (FUTURE_CHANNELS, HISTORY_CHANNELS, N_HISTORY, N_SITE,
                            dewpoint_deficit, future_features, history_features,
                            pressure_tendency, site_features)
from mayak.model import MAYAK

SHARED = tuple(n for n in ENCODER_CHANNELS if n in HISTORY_CHANNELS)


def _batch(B=3, L=L_MAX, seed=0, p_valid=0.8):
    """Окно с историей длины L у правого края, пропусками и мусором под маской."""
    g = torch.Generator().manual_seed(seed)
    h = torch.arange(L_MAX, dtype=torch.float32)
    x = torch.stack([12 + 6 * torch.sin(2 * math.pi * h / 24) + torch.randn(B, L_MAX, generator=g),
                     1005 + 3 * torch.randn(B, L_MAX, generator=g),
                     (65 + 10 * torch.randn(B, L_MAX, generator=g)).clamp(5, 100)], dim=-1)
    m = (torch.rand(B, L_MAX, 3, generator=g) < p_valid).float()
    m[:, :L_MAX - L] = 0
    x = torch.where(m > 0, x, torch.full_like(x, -999.0))
    return {
        "lat": torch.tensor([10.0, -40.0, 62.0])[:B],
        "lon": torch.tensor([30.0, -170.0, 129.0])[:B],
        "elev": torch.tensor([0.0, 300.0, 100.0])[:B],
        "x_hist": x, "mask_hist": m,
        "doy_hist": (h / 24.0 + 100.0).expand(B, L_MAX).clone(),
        "hour_hist": (h % 24).expand(B, L_MAX).clone(),
        "doy_fut": (torch.arange(H) / 24.0 + 128.0).expand(B, H).clone(),
        "hour_fut": (torch.arange(H) % 24).float().expand(B, H).clone(),
    }


def _mayak_channels(model, b):
    """Каналы энкодера основной модели ровно так, как их строит прямой проход."""
    loc = model.loc(b["lat"], b["lon"], b["elev"])
    astro = astro_features(b["doy_hist"], b["hour_hist"], b["lat"][:, None], b["lon"][:, None])
    mu0, sg0, df0 = model.field.evaluate(model.field.coefficients(loc), astro)
    ch, _, _ = model.build_channels(b["x_hist"], b["mask_hist"], astro, mu0, sg0, df0)
    return ch


def _shake(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval()


def test_shared_channel_set():
    """Основная модель берёт из общего построителя всё, кроме аномалий относительно поля."""
    assert set(ENCODER_CHANNELS) - set(SHARED) == {"aT", "adef"}
    assert set(SOLAR_CHANNELS) <= set(FUTURE_CHANNELS)
    assert len(HISTORY_CHANNELS) == N_HISTORY == len(set(HISTORY_CHANNELS))


@pytest.mark.parametrize("L", [0, 1, 24, 100, L_MAX])
@pytest.mark.parametrize("ablations", [Ablations(), Ablations(no_solar=True)])
def test_shared_channels_identical_across_models(L, ablations):
    """Для одного батча общие каналы у основной модели, LRU и GRU совпадают поэлементно."""
    torch.manual_seed(0)
    mayak = _shake(MAYAK(ModelConfig(ablations=ablations)))
    lru, gru = LRUForecaster().eval(), GRUSeq2Seq().eval()
    b = _batch(L=L)
    with torch.no_grad():
        ch = _mayak_channels(mayak, b)
        ref = history_features(b)
        x_lru, x_gru = lru.inputs(b), gru.inputs(b)
    assert ref.shape == (3, L_MAX, N_HISTORY)
    assert x_lru.shape == x_gru.shape == (3, L_MAX, N_HISTORY + N_SITE)
    assert torch.equal(x_lru, x_gru)
    assert torch.equal(x_lru[..., :N_HISTORY], ref)
    names = [n for n in mayak.cfg.channel_names if n in HISTORY_CHANNELS]
    assert names
    for n in names:
        assert torch.equal(mayak.channel(ch, n), ref[..., HISTORY_CHANNELS.index(n)]), n
    for i, n in enumerate(("x", "y", "z", "elev")):
        col = x_lru[..., N_HISTORY + i]
        assert torch.equal(col, col[:, :1].expand_as(col)), n


def test_future_covariates_identical_to_main_model_heads():
    torch.manual_seed(0)
    mayak = MAYAK()
    b = _batch()
    astro_f = astro_features(b["doy_fut"], b["hour_fut"], b["lat"][:, None], b["lon"][:, None])
    fut = future_features(b)
    assert fut.shape == (3, H, len(FUTURE_CHANNELS))
    idx = [FUTURE_CHANNELS.index(n) for n in SOLAR_CHANNELS]
    assert torch.equal(mayak.solar_future(astro_f), fut[..., idx])


def test_junk_under_mask_does_not_reach_features():
    """Значения под нулевой маской, в том числе бесконечные, не меняют признаки."""
    b = _batch(L=200)
    junk = dict(b, x_hist=b["x_hist"].clone())
    bad = b["mask_hist"] == 0
    junk["x_hist"][bad] = float("inf")
    junk["x_hist"][..., 0][b["mask_hist"][..., 0] == 0] = -243.04
    ref, got = history_features(b), history_features(junk)
    assert torch.isfinite(got).all()
    assert torch.equal(ref, got)


def test_dewpoint_deficit_is_zero_safe_and_unchanged_on_valid_hours():
    x = torch.tensor([[[20.0, 1000.0, 50.0], [-243.04, 1000.0, 0.0], [5.0, 1000.0, 100.0]]])
    m = torch.tensor([[[1.0, 1, 1], [0, 1, 0], [1, 1, 1]]])
    d = dewpoint_deficit(x, m)
    assert torch.isfinite(d).all() and (d >= 0).all()
    assert d[0, 0] > 5 and abs(float(d[0, 2])) < 1e-4


def test_pressure_tendency_needs_both_hours():
    P = torch.arange(30.0)[None] + 1000.0
    v = torch.ones(1, 30)
    v[0, 10] = 0
    d3 = pressure_tendency(P, v, 3)
    assert (d3[0, :3] == 0).all()
    assert d3[0, 10] == 0 and d3[0, 13] == 0
    assert torch.allclose(d3[0, 20], torch.tensor(1.0))


def test_site_features_on_unit_sphere_and_continuous_across_dateline():
    lat = torch.tensor([0.0, 45.0, -89.9, 10.0, 10.0])
    lon = torch.tensor([0.0, 90.0, 12.0, 179.999, -179.999])
    s = site_features(lat, lon, torch.tensor([0.0, 1000.0, 2800.0, 5.0, 5.0]))
    assert s.shape == (5, N_SITE)
    assert torch.allclose(s[:, :3].norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(s[3], s[4], atol=1e-4)
    assert s[1, 3] == 1.0


def test_features_do_not_depend_on_other_windows_in_batch():
    """Нормировка фиксированная: признаки окна не зависят от соседей по батчу."""
    b = _batch()
    one = {k: v[:1] for k, v in b.items()}
    assert torch.equal(history_features(b)[:1], history_features(one))


@pytest.mark.parametrize("arch", ["gru", "lru"])
def test_recurrent_baselines_use_calendar_and_coordinates(arch):
    """При пустой истории прогноз всё равно зависит от сезона и от точки."""
    torch.manual_seed(0)
    m = _shake(NEURAL[arch]())
    b = _batch(L=0)
    later = dict(b, doy_fut=b["doy_fut"] + 150.0, doy_hist=b["doy_hist"] + 150.0)
    moved = dict(b, lat=b["lat"] + 30.0)
    with torch.no_grad():
        q0, q1, q2 = m(b)["q"], m(later)["q"], m(moved)["q"]
    assert not torch.allclose(q0, q1) and not torch.allclose(q0, q2)
