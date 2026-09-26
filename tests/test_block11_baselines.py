"""Тесты: бейзлайны."""
import dataclasses
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mayak import baselines as BL
from mayak.baselines.lru import (LRULayer, LRUForecaster, lru_recurrent, lru_scan_associative,
                                 lru_scan_chunked)
from mayak.baselines.patchtst import make_patches, masked_instance_stats
from mayak.config import (ConfigError, DLinearConfig, GRUConfig, LRUConfig, PatchTSTConfig,
                          check_pipeline_compat, model_config_for)
from mayak.constants import H, L_MAX, NQ
from mayak.protocol import DEFAULT_PROTOCOL, Protocol, ProtocolError

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "conf"
NEW_ARCHS = ("lru", "patchtst", "gru")


def _batch(B=3, L=L_MAX, seed=0, p_valid=0.85):
    """Окно с историей длины L (выровнена по правому краю) и пропусками внутри неё."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(B, L_MAX, 3)
    x = torch.zeros(B, L_MAX, 3)
    if L > 0:
        h = torch.arange(L, dtype=torch.float32)
        x[:, -L:, 0] = 12 + 6 * torch.sin(2 * math.pi * h / 24) + torch.randn(B, L, generator=g)
        x[:, -L:, 1] = 1005 + 3 * torch.randn(B, L, generator=g)
        x[:, -L:, 2] = (65 + 10 * torch.randn(B, L, generator=g)).clamp(5, 100)
        m[:, -L:] = (torch.rand(B, L, 3, generator=g) < p_valid).float()
        x = x * m
    return {
        "lat": torch.rand(B, generator=g) * 120 - 60,
        "lon": torch.rand(B, generator=g) * 360 - 180,
        "elev": torch.rand(B, generator=g) * 500,
        "x_hist": x, "mask_hist": m,
        "doy_hist": (torch.arange(L_MAX) / 24.0 + 100.0).expand(B, L_MAX).clone(),
        "hour_hist": (torch.arange(L_MAX) % 24).float().expand(B, L_MAX).clone(),
        "doy_fut": (torch.arange(H) / 24.0 + 128.0).expand(B, H).clone(),
        "hour_fut": (torch.arange(H) % 24).float().expand(B, H).clone(),
        "y": 12 + torch.randn(B, H, generator=g), "y_mask": torch.ones(B, H),
        "norm_scale": torch.full((B, H), 3.0),
    }


def _model(arch, cfg=None, seed=0):
    torch.manual_seed(seed)
    return BL.NEURAL[arch](cfg).eval()


def _trained_like(m, seed=1, scale=0.05):
    """Сбить инициализацию (нулевые смещения квантилей и т.п.), не трогая BatchNorm-буферы."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


def _rand_lambda(N, seed=0, r=(0.3, 0.999), phase=3.0):
    g = torch.Generator().manual_seed(seed)
    mod = r[0] + (r[1] - r[0]) * torch.rand(N, generator=g, dtype=torch.float64)
    th = phase * torch.rand(N, generator=g, dtype=torch.float64)
    return -torch.log(mod), th


@pytest.mark.parametrize("L", [1, 7, 32, 33, 100, 672])
@pytest.mark.parametrize("chunk", [1, 8, 32])
def test_lru_scans_match_naive_recurrence(L, chunk):
    """Блочный и полный ассоциативные сканы = наивный цикл h_t = λ h_{t−1} + u_t."""
    nu, theta = _rand_lambda(16)
    g = torch.Generator().manual_seed(L)
    u_re = torch.randn(2, L, 16, generator=g, dtype=torch.float64)
    u_im = torch.randn(2, L, 16, generator=g, dtype=torch.float64)
    a_re, a_im = torch.exp(-nu) * torch.cos(theta), torch.exp(-nu) * torch.sin(theta)
    ref = lru_recurrent(a_re, a_im, u_re, u_im)
    for got in (lru_scan_associative(a_re, a_im, u_re, u_im),
                lru_scan_chunked(nu, theta, u_re, u_im, chunk)):
        for r, x in zip(ref, got):
            assert x.shape == r.shape
            assert torch.allclose(x, r, atol=1e-11, rtol=0), (x - r).abs().max()


def test_lru_scan_matches_recurrence_in_float32_on_full_window():
    """То же во float32 на 672 ч: расхождение — ошибка округления, а не метода."""
    torch.manual_seed(0)
    lay = LRULayer(16, 32, 0.7, 0.999, 0.5)
    x = torch.randn(3, L_MAX, 16)
    ref = lay.states(x, "recurrent")
    scale = max(float(ref[0].detach().abs().max()), 1.0)
    for scan in ("chunked", "associative"):
        got = lay.states(x, scan)
        for r, v in zip(ref, got):
            assert (v - r).abs().max() / scale < 1e-5


def test_lru_full_forecast_identical_for_all_scans():
    """Эквивалентность на уровне полного выпуска: все лиды, все квантили."""
    m = _trained_like(_model("lru", LRUConfig(d_model=16, d_state=16, layers=2, head_hidden=32)))
    b = _batch(B=2)
    with torch.no_grad():
        ref = m(b, scan="recurrent")
        for scan in ("chunked", "associative"):
            out = m(b, scan=scan)
            for k in ("q", "mu", "sigma"):
                assert torch.allclose(out[k], ref[k], atol=1e-4, rtol=1e-5), (scan, k)
    assert LRUForecaster(LRUConfig(scan="recurrent", d_model=8, d_state=8, layers=1)).cfg.scan \
        == "recurrent"


def test_lru_scan_does_not_run_python_loop_over_hours(monkeypatch):
    """Развёртка по умолчанию не зовёт наивный цикл."""
    from mayak.baselines import lru as mod

    def boom(*a, **k):
        raise AssertionError("питоновский цикл по часам в режиме по умолчанию")

    monkeypatch.setattr(mod, "lru_recurrent", boom)
    m = _model("lru", LRUConfig(d_model=8, d_state=8, layers=1, head_hidden=16))
    m(_batch(B=1))


def test_lru_eigenvalues_inside_unit_disk_for_any_parameters():
    """Экспоненциальная параметризация: |λ| = exp(−exp(ν_log)) < 1 при любых ν_log, θ_log
    (в float64 различимо до ν_log ≈ −36; в float32 |λ| округляется к 1 при ν_log < −17)."""
    lay = LRULayer(4, 64, 0.5, 0.99, 1.0).double()
    with torch.no_grad():
        lay.nu_log.copy_(torch.linspace(-30, 5, 64, dtype=torch.float64))
        lay.theta_log.copy_(torch.linspace(-10, 5, 64))
    mod, _ = lay.eigenvalues()
    assert (mod < 1).all() and (mod >= 0).all()


def test_lru_init_follows_config_ranges():
    cfg = LRUConfig()
    assert math.isclose(cfg.r_min, math.exp(-1 / 3.0))
    assert math.isclose(cfg.r_max, math.exp(-1 / 240.0))
    assert math.isclose(cfg.max_phase, 2 * math.pi / 12.0)
    torch.manual_seed(0)
    lay = LRULayer(8, 4096, cfg.r_min, cfg.r_max, cfg.max_phase)
    mod, phase = lay.eigenvalues()
    assert mod.min() >= cfg.r_min - 1e-6 and mod.max() <= cfg.r_max + 1e-6
    assert phase.min() >= 0 and phase.max() <= cfg.max_phase + 1e-6
    # |λ|² ~ U[r_min², r_max²]
    m2 = (mod.detach() ** 2).numpy()
    assert abs(m2.mean() - (cfg.r_min ** 2 + cfg.r_max ** 2) / 2) < 0.01
    assert torch.allclose(torch.exp(lay.gamma_log), torch.sqrt(1 - mod ** 2), atol=1e-6)


@pytest.mark.parametrize("modulus", [0.5, 0.9, 0.995])
def test_lru_input_normalization_keeps_state_variance(modulus):
    """γ = √(1 − |λ|²): при белом шуме на входе E|h|² ≈ E|Bu|² вне зависимости от |λ|."""
    torch.manual_seed(0)
    lay = LRULayer(1, 1, 0.5, 0.6, 1.0).double()
    with torch.no_grad():
        lay.nu_log.fill_(math.log(-math.log(modulus)))
        lay.theta_log.fill_(math.log(0.3))
        lay.gamma_log.fill_(math.log(math.sqrt(1 - modulus ** 2)))
        lay.B_re.fill_(1.0)
        lay.B_im.fill_(0.0)
    x = torch.randn(16, 6000, 1, dtype=torch.float64)
    h_re, h_im = lay.states(x)
    burn = 2000
    var = (h_re[:, burn:] ** 2 + h_im[:, burn:] ** 2).mean().item()
    assert abs(var - 1.0) < 0.1, var


def test_lru_optim_groups_exclude_recurrent_params_from_weight_decay():
    m = _model("lru")
    groups = m.optim_groups(0.01)
    names = {n for n, _ in m.named_parameters()}
    by_id = {id(p): n for n, p in m.named_parameters()}
    rec = [by_id[id(p)] for p in groups[0]["params"]]
    oth = [by_id[id(p)] for p in groups[1]["params"]]
    assert groups[0]["name"] == "recurrent" and groups[0]["weight_decay"] == 0.0
    assert groups[1]["weight_decay"] == 0.01
    assert set(rec) | set(oth) == names and not set(rec) & set(oth)
    assert {n.rsplit(".", 1)[-1] for n in rec} == {"nu_log", "theta_log", "gamma_log",
                                                   "B_re", "B_im"}


def test_lru_scan_runs_in_float32_under_bf16_autocast():
    m = _model("lru", LRUConfig(d_model=16, d_state=16, layers=2, head_hidden=32))
    b = _batch(B=2)
    with torch.no_grad():
        ref = m(b)["q"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(b)["q"].float()
    assert torch.isfinite(out).all()
    assert (out - ref).abs().max() < 0.5


def test_patchtst_patching():
    """Суточные патчи без перекрытия по умолчанию; разбиение с дополнением повтором."""
    cfg = PatchTSTConfig()
    assert (cfg.patch_len, cfg.stride, cfg.padding_patch) == (24, 24, "none")
    assert cfg.n_patches == cfg.input_len // 24 == 21
    assert PatchTSTConfig(input_len=L_MAX, patch_len=16, stride=8,
                          padding_patch="end").n_patches == 84
    z = torch.arange(20.0).view(1, 20, 1)
    p = make_patches(z, 8, 4, "end")
    assert p.shape == (1, (20 - 8) // 4 + 2, 8)
    assert torch.equal(p[0, 0], torch.arange(8.0))
    assert torch.equal(p[0, -1], torch.tensor([16, 17, 18, 19, 19, 19, 19, 19.0]))
    two = make_patches(torch.stack([z[..., 0], -z[..., 0]], -1), 8, 4, "none")
    assert torch.equal(two[0, 1], torch.cat([torch.arange(4.0, 12), -torch.arange(4.0, 12)]))


def test_patchtst_last_patch_ends_at_issue_time():
    """Последний патч заканчивается последним часом истории: вход не теряет свежие часы."""
    cfg = PatchTSTConfig()
    z = torch.arange(float(cfg.input_len)).view(1, -1, 1)
    p = make_patches(z, cfg.patch_len, cfg.stride, cfg.padding_patch)
    assert p[0, -1, -1] == cfg.input_len - 1 and p[0, 0, 0] == 0
    with pytest.raises(ConfigError, match="не делится на патчи"):
        PatchTSTConfig(input_len=500)


def test_masked_revin_statistics():
    g = torch.Generator().manual_seed(0)
    x = 5 + 3 * torch.randn(4, 200, generator=g)
    m = (torch.rand(4, 200, generator=g) < 0.6).float()
    m[3] = 0
    m[3, 7] = 1
    mean, std = masked_instance_stats(x * m + 1e6 * (1 - m), m)
    for i in range(3):
        v = x[i][m[i] > 0]
        assert torch.allclose(mean[i, 0], v.mean(), atol=1e-4)
        assert torch.allclose(std[i, 0], torch.sqrt(v.var(unbiased=False) + 1e-5), atol=1e-4)
    assert mean[3, 0] == 0 and std[3, 0] == 1


def test_patchtst_is_shift_and_scale_equivariant():
    """RevIN: q(a·T + b) = a·q(T) + b при a > 0 (с точностью до eps в σ)."""
    m = _trained_like(_model("patchtst"))
    b = _batch(B=2, p_valid=1.0)
    a, s = 2.5, -7.0
    b2 = dict(b, x_hist=b["x_hist"].clone())
    b2["x_hist"][..., 0] = (b["x_hist"][..., 0] * a + s) * b["mask_hist"][..., 0]
    with torch.no_grad():
        q1, q2 = m(b)["q"], m(b2)["q"]
    assert torch.allclose(q2, a * q1 + s, atol=1e-3, rtol=1e-4)


def test_patchtst_channel_independence():
    """Прогноз температуры не зависит от истории давления и влажности."""
    m = _trained_like(_model("patchtst"))
    b = _batch(B=2)
    other = dict(b, x_hist=b["x_hist"].clone(), mask_hist=b["mask_hist"].clone())
    other["x_hist"][..., 1:] = torch.randn_like(other["x_hist"][..., 1:]) * 100
    other["mask_hist"][..., 1:] = 0
    with torch.no_grad():
        assert torch.equal(m(b)["q"], m(other)["q"])


def test_patchtst_matches_revin_denormalization():
    """mu, sigma и квантили связаны обратной RevIN: q = q̂·σ + μ, монотонно."""
    m = _trained_like(_model("patchtst"))
    b = _batch(B=3)
    with torch.no_grad():
        out = m(b)
        _, _, mean, std = m.normalize(b)
    med = out["q"][..., NQ // 2]
    assert torch.allclose(med, out["mu"], atol=1e-4)
    assert (out["q"].diff(dim=-1) >= 0).all()
    n = m.cfg.input_len
    x, v = b["x_hist"][:, -n:, 0], b["mask_hist"][:, -n:, 0]
    assert torch.allclose(mean[:, 0], (x * v).sum(-1) / v.sum(-1), atol=1e-4)
    assert (std > 0).all()


@pytest.mark.parametrize("arch", list(BL.NEURAL))
def test_masked_history_values_are_ignored(arch):
    """Вход с маской валидности: мусор под нулевой маской не меняет выход."""
    m = _trained_like(_model(arch))
    b = _batch(B=2)
    junk = dict(b, x_hist=torch.where(b["mask_hist"] > 0, b["x_hist"],
                                      torch.full_like(b["x_hist"], 1e4)))
    with torch.no_grad():
        o0, o1 = m(b), m(junk)
    for k in ("q", "mu", "sigma"):
        assert torch.allclose(o0[k], o1[k], atol=1e-5), k


@pytest.mark.parametrize("arch", NEW_ARCHS)
@pytest.mark.parametrize("case", ["empty", "one_hour", "full", "extreme_hi", "extreme_lo"])
def test_new_baselines_monotone_and_finite_on_extreme_inputs(arch, case):
    m = _trained_like(_model(arch))
    b = _batch(B=2, L={"empty": 0, "one_hour": 1}.get(case, L_MAX), p_valid=1.0)
    if case.startswith("extreme"):
        hi = case == "extreme_hi"
        full = torch.tensor([60.0, 1100.0, 100.0] if hi else [-90.0, 300.0, 0.0])
        b["x_hist"] = full.expand_as(b["x_hist"]).clone()
        b["mask_hist"] = torch.ones_like(b["mask_hist"])
    with torch.no_grad():
        out = m(b)
    assert out["q"].shape == (2, H, NQ) and out["mu"].shape == out["sigma"].shape == (2, H)
    assert torch.isfinite(out["q"]).all() and (out["sigma"] > 0).all()
    assert (out["q"].diff(dim=-1) >= 0).all()


@pytest.mark.parametrize("arch", NEW_ARCHS)
@pytest.mark.parametrize("L", [0, 24, L_MAX])
def test_new_baselines_train_with_finite_gradients(arch, L):
    from mayak.loss import pinball_loss
    torch.manual_seed(0)
    m = BL.NEURAL[arch]().train()
    b = _batch(B=4, L=L)
    loss = pinball_loss(m(b), b)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_baseline_parameter_counts_are_pinned():
    """Архитектуры по умолчанию закреплены числом параметров."""
    counts = {a: sum(p.numel() for p in _model(a).parameters()) for a in BL.NEURAL}
    assert counts == {"gru": 120754, "dlinear": 114408, "lru": 103986, "patchtst": 138944}


def test_baseline_sizes_within_band_of_main_model():
    """Каждая нейросеть по умолчанию в полосе размеров относительно основной модели."""
    from mayak.model import MAYAK
    ref = sum(p.numel() for p in MAYAK().parameters())
    lo, hi = BL.SIZE_BAND
    for arch, cls in BL.NEURAL.items():
        n = sum(p.numel() for p in cls().parameters())
        assert lo <= n / ref <= hi, f"{arch}: {n} параметров, доля {n / ref:.2f}"


class _RefMovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


def _ref_dlinear(x, lin_seasonal, lin_trend, kernel):
    moving_mean = _RefMovingAvg(kernel, 1)(x)
    seasonal_init, trend_init = x - moving_mean, moving_mean
    seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
    out = lin_seasonal(seasonal_init) + lin_trend(trend_init)
    return out.permute(0, 2, 1)


def test_dlinear_matches_reference_implementation():
    m = _model("dlinear")
    b = _batch(B=3, p_valid=1.0)
    T = b["x_hist"][:, -m.input_len:, 0]
    with torch.no_grad():
        ours = m(b)["mu"]
        ref = _ref_dlinear(T[..., None], m.lin_resid, m.lin_trend, m.cfg.kernel)[..., 0]
    assert torch.allclose(ours, ref, atol=1e-4)
    assert m.cfg.kernel == 25 and DLinearConfig().kernel == 25


def test_dlinear_reads_only_last_hours_of_history():
    """Вход DLinear - последние 336 ч; более ранние часы на прогноз не влияют."""
    m = _model("dlinear")
    assert m.input_len == DLinearConfig().input_len == 336
    b = _batch(B=2, p_valid=1.0)
    early = dict(b, x_hist=b["x_hist"].clone())
    early["x_hist"][:, :L_MAX - m.input_len, 0] += 50.0
    with torch.no_grad():
        assert torch.equal(m(b)["q"], m(early)["q"])


def test_pipeline_compat_allows_shorter_input_only():
    check_pipeline_compat(DLinearConfig(input_len=336))
    check_pipeline_compat(PatchTSTConfig(input_len=L_MAX, patch_len=24, stride=24))
    with pytest.raises(ConfigError, match="длина входа"):
        check_pipeline_compat(DLinearConfig(input_len=L_MAX + 24))
    with pytest.raises(ConfigError, match="длина истории"):
        check_pipeline_compat(LRUConfig(max_history=336))


def test_dlinear_has_no_input_normalization():
    m = _model("dlinear")
    g = torch.Generator().manual_seed(0)
    n = m.input_len
    x1, x2 = torch.randn(2, n, generator=g), 10 * torch.randn(2, n, generator=g)
    with torch.no_grad():
        lhs = m.point(x1 + x2) - m.point(x2)
        rhs = m.point(x1) - m.point(torch.zeros_like(x1))
    assert torch.allclose(lhs, rhs, atol=1e-3)
    assert not any(isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d)) for mod in m.modules())


def test_seasonal_naive_matches_textbook_formula_on_complete_history():
    rng = np.random.default_rng(0)
    B = 2
    x = rng.normal(10, 5, (B, L_MAX, 3)).astype(np.float32)
    m = np.ones_like(x)
    mu, q = BL.seasonal_naive_forecast(x, m, np.zeros((B, H), np.float32), np.ones(B, np.float32))
    T_last = L_MAX - 1
    for h in range(1, H + 1):
        k = (h - 1) // 24
        idx = T_last + h - 24 * (k + 1)
        assert np.array_equal(mu[:, h - 1], x[:, idx, 0])
    assert (np.diff(q, axis=-1) >= 0).all()


def test_seasonal_naive_falls_back_to_earlier_day_then_climatology():
    B = 1
    x = np.zeros((B, L_MAX, 3), np.float32)
    m = np.zeros_like(x)
    x[0, L_MAX - 48 + 5, 0], m[0, L_MAX - 48 + 5, 0] = 7.0, 1
    x[0, L_MAX - 24 + 6, 0], m[0, L_MAX - 24 + 6, 0] = 9.0, 1
    clim = np.full((B, H), -3.0, np.float32)
    mu, _ = BL.seasonal_naive_forecast(x, m, clim, np.ones(B, np.float32))
    assert mu[0, 5] == 7.0 and mu[0, 29] == 7.0
    assert mu[0, 6] == 9.0
    assert mu[0, 0] == -3.0 and mu[0, 7] == -3.0


def test_damped_persistence_formula():
    B = 3
    clim = np.tile(np.linspace(0, 5, H, dtype=np.float32), (B, 1))
    sig = np.array([1.0, 2.0, 3.0], np.float32)
    a = np.array([2.0, -1.0, 0.5], np.float32)
    r0, r1 = np.zeros(H, np.float32), np.ones(H, np.float32)
    mu, q = BL.damped_persistence_forecast(a, clim, sig, r0)
    mc, qc = BL.climatology_forecast(clim, sig)
    assert np.allclose(mu, mc) and np.allclose(q, qc)
    mu, q = BL.damped_persistence_forecast(a, clim, sig, r1)
    assert np.allclose(mu, clim + a[:, None])
    width = q[..., -1] - q[..., 0]
    z = BL.ZQ[-1] - BL.ZQ[0]
    assert np.allclose(width, z * sig[:, None] * math.sqrt(BL.DAMPED_VAR_FLOOR), rtol=1e-5)
    r = np.linspace(0.9, 0.1, H).astype(np.float32)
    mu, q = BL.damped_persistence_forecast(a, clim, sig, r)
    assert np.allclose(mu, clim + r * a[:, None])
    assert np.allclose(q[..., 3], mu)
    assert np.allclose(q[..., -1] - q[..., 0], z * sig[:, None] * np.sqrt(1 - r ** 2), rtol=1e-5)


def test_damped_coefficients_recover_known_decay():
    """МНК через ноль восстанавливает r_h, если a(t+h) = r_h·ā точно; обрезка в [0, 1]."""
    rng = np.random.default_rng(0)
    r_true = np.exp(-np.arange(1, H + 1) / 30.0)
    Sxx, Sxy = np.zeros(H), np.zeros(H)
    for a in rng.normal(0, 2, 500):
        Sxx += a * a
        Sxy += a * (r_true * a)
    assert np.allclose(BL.damped_coefficients(Sxx, Sxy), r_true, atol=1e-6)
    assert (BL.damped_coefficients(Sxx, -Sxy) == 0).all()
    assert (BL.damped_coefficients(Sxx, 3 * Sxy)[:5] == 1).all()
    assert (BL.damped_coefficients(np.zeros(H), Sxy) == 0).all()


@pytest.mark.parametrize("arch, cls", [("gru", GRUConfig), ("dlinear", DLinearConfig),
                                       ("lru", LRUConfig), ("patchtst", PatchTSTConfig)])
def test_yaml_defaults_match_dataclasses(arch, cls):
    import yaml
    d = yaml.safe_load((CONF / "model" / f"{arch}.yaml").read_text(encoding="utf-8"))
    assert set(d) == {f.name for f in dataclasses.fields(cls)}
    assert cls.from_dict(d) == cls()


@pytest.mark.parametrize("arch", NEW_ARCHS)
def test_hydra_composes_new_baselines(arch):
    from hydra import compose, initialize_config_dir
    spec = importlib.util.spec_from_file_location("run", REPO / "scripts" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose("config", overrides=[f"model={arch}"])
    rc = run.to_run_config(cfg)
    assert rc.arch == arch and rc.model == model_config_for(arch)


@pytest.mark.parametrize("cfg", [LRUConfig(d_model=8, scan="associative", tau_bounds=(2, 50)),
                                 PatchTSTConfig(patch_len=24, stride=12, norm="layer",
                                                revin=False, d_model=32, n_heads=4),
                                 GRUConfig(hidden=16, layers=1, head_hidden=16),
                                 DLinearConfig(input_len=168, kernel=5)])
def test_new_configs_roundtrip_and_build(cfg):
    import json
    back = model_config_for(cfg.arch, json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg
    check_pipeline_compat(cfg)
    m = BL.NEURAL[cfg.arch](cfg).eval()
    with torch.no_grad():
        assert m(_batch(B=1))["q"].shape == (1, H, NQ)


def test_new_configs_reject_invalid_values():
    with pytest.raises(ConfigError, match="scan"):
        LRUConfig(scan="loop")
    with pytest.raises(ConfigError, match="tau_bounds"):
        LRUConfig(tau_bounds=(10, 5))
    with pytest.raises(ConfigError, match="min_period"):
        LRUConfig(min_period=2)
    with pytest.raises(ConfigError, match="n_heads"):
        PatchTSTConfig(d_model=100, n_heads=16)
    with pytest.raises(ConfigError, match="padding_patch"):
        PatchTSTConfig(padding_patch="start")
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        LRUConfig.from_dict({"d_modl": 8})


def _fake_ckpt(path, arch="mayak", protocol=DEFAULT_PROTOCOL):
    hp = {"arch": arch}
    if protocol is not None:
        hp["protocol"] = protocol.to_dict()
    torch.save({"hyper_parameters": hp, "state_dict": {}}, path)
    return str(path)


def test_check_comparable_accepts_same_protocol(tmp_path):
    from mayak.lit import check_comparable
    ref = _fake_ckpt(tmp_path / "m.ckpt")
    lru = _fake_ckpt(tmp_path / "l.ckpt", "lru")
    pt = _fake_ckpt(tmp_path / "p.ckpt", "patchtst")
    assert check_comparable(ref, [lru, pt]) == {ref: "mayak", lru: "lru", pt: "patchtst"}


def test_check_comparable_rejects_mismatch_and_pre_protocol_checkpoints(tmp_path):
    from mayak.lit import SEED_FIELDS, check_comparable
    ref = _fake_ckpt(tmp_path / "m.ckpt")
    other = _fake_ckpt(tmp_path / "g.ckpt", "gru",
                       Protocol(batch_size=DEFAULT_PROTOCOL.batch_size * 2))
    with pytest.raises(ProtocolError, match="batch_size"):
        check_comparable(ref, [other])
    old = _fake_ckpt(tmp_path / "old.ckpt", "dlinear", None)
    with pytest.raises(ProtocolError, match="блок 4"):
        check_comparable(ref, [old])
    seed1 = _fake_ckpt(tmp_path / "s1.ckpt", "mayak", Protocol(seed=1))
    with pytest.raises(ProtocolError, match="seed"):
        check_comparable(ref, [seed1])
    check_comparable(ref, [seed1], ignore=SEED_FIELDS)
