import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from mayak.astro import astro_features
from mayak.constants import L_MAX, QUANTILES, H
from mayak.data.climatology import Climatology
from mayak.data.dataset import footprint, history_len, slice_history, slice_target
from mayak.data.qc import PHYS
from mayak.model import MAYAK
from mayak.timeaxis import doy_hour, from_utc_hour, to_utc_hour, window_calendar

NQ = len(QUANTILES)
I_MEDIAN = list(QUANTILES).index(0.5)


def _uniform_batch(T, P, RH, *, valid_hours, lat, lon, elev, B=1, seed=0):
    """Окно, в котором все валидные часы несут одно и то же значение каналов.

    valid_hours - сколько последних часов буфера L_MAX валидны (0 - пустая история).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, L_MAX, 3)
    m = torch.zeros(B, L_MAX, 3)
    if valid_hours > 0:
        x[:, L_MAX - valid_hours:, 0] = T
        x[:, L_MAX - valid_hours:, 1] = P
        x[:, L_MAX - valid_hours:, 2] = RH
        m[:, L_MAX - valid_hours:] = 1.0

    hours = torch.arange(L_MAX, dtype=torch.float32) % 24.0
    doy = (torch.arange(L_MAX, dtype=torch.float32) / 24.0) % 365.0
    fut_h = torch.arange(1, H + 1, dtype=torch.float32)
    return {
        "lat": torch.full((B,), float(lat)),
        "lon": torch.full((B,), float(lon)),
        "elev": torch.full((B,), float(elev)),
        "x_hist": x, "mask_hist": m,
        "doy_hist": doy[None].expand(B, L_MAX).contiguous(),
        "hour_hist": hours[None].expand(B, L_MAX).contiguous(),
        "doy_fut": ((fut_h / 24.0) % 365.0)[None].expand(B, H).contiguous(),
        "hour_fut": (fut_h % 24.0)[None].expand(B, H).contiguous(),
        "y": torch.randn(B, H, generator=g),
        "y_mask": torch.ones(B, H),
        "norm_scale": torch.full((B, H), 3.0),
    }


EXTREME_VALUES = {
    "lower_bounds": (PHYS["T"][0], PHYS["P"][0], PHYS["RH"][0]),
    "upper_bounds": (PHYS["T"][1], PHYS["P"][1], PHYS["RH"][1]),
    "cold_dry": (PHYS["T"][0], PHYS["P"][1], PHYS["RH"][0]),
    "hot_wet": (PHYS["T"][1], PHYS["P"][0], PHYS["RH"][1]),
}
EXTREME_HISTORY = {"empty": 0, "one_hour": 1, "one_day": 24, "full": L_MAX}
EXTREME_SITES = {"north_pole": (90.0, 180.0, 0.0), "south_pole": (-90.0, -180.0, 0.0),
                 "equator_high": (0.0, 0.0, 5000.0)}


@pytest.mark.parametrize("values", sorted(EXTREME_VALUES))
@pytest.mark.parametrize("history", sorted(EXTREME_HISTORY))
def test_quantiles_ordered_on_extreme_inputs(values, history):
    """Границы физических диапазонов, пустая история и история из одного часа.

    Проверяется контракт выхода, а не качество прогноза: квантили конечны,
    неубывающие, медианный квантиль совпадает с mu, масштаб интервала положителен.
    """
    torch.manual_seed(0)
    model = MAYAK().eval()
    T, P, RH = EXTREME_VALUES[values]
    batch = _uniform_batch(T, P, RH, valid_hours=EXTREME_HISTORY[history],
                           lat=55.0, lon=37.0, elev=150.0)
    with torch.no_grad():
        out = model(batch)

    q = out["q"]
    assert q.shape == (1, H, NQ)
    assert torch.isfinite(q).all(), f"{values}/{history}: неконечные квантили"
    assert torch.isfinite(out["mu"]).all()
    assert (q.diff(dim=-1) >= 0).all(), (
        f"{values}/{history}: квантили не упорядочены, "
        f"min Δ = {q.diff(dim=-1).min().item():.3g}")
    assert torch.allclose(q[..., I_MEDIAN], out["mu"], atol=0, rtol=0), (
        "медианный квантиль обязан совпадать с mu поэлементно")
    assert (out["ratio"] > 0).all() and (out["sigma_c"] > 0).all()


@pytest.mark.parametrize("site", sorted(EXTREME_SITES))
def test_quantiles_ordered_at_extreme_sites(site):
    """Полюса, антимеридиан и большая высота не ломают ни поле, ни квантили."""
    torch.manual_seed(0)
    model = MAYAK().eval()
    lat, lon, elev = EXTREME_SITES[site]
    batch = _uniform_batch(-40.0, 1050.0, 100.0, valid_hours=L_MAX,
                           lat=lat, lon=lon, elev=elev)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out["q"]).all(), site
    assert (out["q"].diff(dim=-1) >= 0).all(), site


def test_empty_history_gives_zero_evidence_and_zero_anomaly():
    """Пустая история: масса свидетельств и амплитуды мод строго нулевые."""
    torch.manual_seed(0)
    model = MAYAK().eval()
    with torch.no_grad():
        out = model(_uniform_batch(0.0, 0.0, 0.0, valid_hours=0,
                                   lat=10.0, lon=20.0, elev=0.0))
    assert torch.equal(out["e"], torch.zeros_like(out["e"]))
    assert torch.equal(out["a_re"], torch.zeros_like(out["a_re"]))
    assert torch.equal(out["a_im"], torch.zeros_like(out["a_im"]))
    assert torch.equal(out["o"], torch.zeros_like(out["o"]))


W_YEAR, W_DAY = 2 * np.pi / 365.24, 2 * np.pi / 24.0
TRUE_HARMONICS = dict(mean=11.7, year_amp=9.3, year_phase=20.0, day_amp=4.1,
                      day_phase=15.0, year2_amp=1.6, mixed_amp=1.9)


def _known_series(n_hours=3 * 365 * 24, t_start=datetime(2015, 1, 1, tzinfo=timezone.utc)):
    """Ряд с гармониками, целиком лежащими в базисе Climatology._design."""
    t0 = to_utc_hour(t_start.replace(tzinfo=None))
    doy, hour = doy_hour(from_utc_hour(t0 + np.arange(n_hours, dtype=np.int64)))
    p = TRUE_HARMONICS
    T = (p["mean"]
         + p["year_amp"] * np.cos(W_YEAR * (doy - p["year_phase"]))
         + p["year2_amp"] * np.cos(2 * W_YEAR * doy)
         + p["day_amp"] * np.cos(W_DAY * (hour - p["day_phase"]))
         + p["mixed_amp"] * np.cos(W_YEAR * doy) * np.cos(W_DAY * hour))
    return doy, hour, T


def test_climatology_recovers_known_harmonics_exactly_without_noise():
    doy, hour, T = _known_series()
    clim = Climatology().fit(doy, hour, T, np.ones_like(T))
    err = np.abs(clim.predict(doy, hour) - T).max()
    assert err < 1e-6, f"чистый ряд восстановлен с ошибкой {err:.3g} °C"
    assert clim.sigma < 1e-5


@pytest.mark.parametrize("noise_sd", [0.3, 1.0])
def test_climatology_recovers_known_harmonics_under_noise(noise_sd):
    """Шум не смещает восстановленное среднее: ошибка падает как 1/sqrt(N)."""
    doy, hour, T = _known_series()
    rng = np.random.default_rng(3)
    obs = T + noise_sd * rng.standard_normal(T.shape)
    clim = Climatology().fit(doy, hour, obs, np.ones_like(obs))

    pred = clim.predict(doy, hour)
    tol = 12.0 * noise_sd / math.sqrt(len(T))
    assert np.abs(pred - T).max() < tol, (
        f"σ = {noise_sd}: max|Δ| = {np.abs(pred - T).max():.4g} > {tol:.4g}")
    assert clim.sigma == pytest.approx(noise_sd, rel=0.05)


def _expected_beta():
    """Аналитические коэффициенты базиса ``_design(doy, hour, 3, 3)`` для ряда выше.

    Порядок колонок: 1, {cos,sin}(k·ω_год·doy) k=1..3, {cos,sin}(k·ω_сут·hour) k=1..3,
    cos(ω_год·doy)·{cos,sin}(ω_сут·hour). Разложение по формуле сложения углов даёт
    ненулевыми ровно семь коэффициентов.
    """
    p = TRUE_HARMONICS
    b = np.zeros(15)
    b[0] = p["mean"]
    b[1] = p["year_amp"] * np.cos(W_YEAR * p["year_phase"])
    b[2] = p["year_amp"] * np.sin(W_YEAR * p["year_phase"])
    b[3] = p["year2_amp"]
    b[7] = p["day_amp"] * np.cos(W_DAY * p["day_phase"])
    b[8] = p["day_amp"] * np.sin(W_DAY * p["day_phase"])
    b[13] = p["mixed_amp"]
    return b


def test_climatology_recovers_harmonic_coefficients_not_only_values():
    """Восстановлены именно амплитуды и фазы, а не просто значения в узлах сетки.

    Ряд лежит в базисе подгонки точно, поэтому коэффициенты определены однозначно:
    сравниваем их с аналитическим разложением, а не с предсказанием на тех же точках.
    """
    doy, hour, T = _known_series()
    beta = Climatology().fit(doy, hour, T, np.ones_like(T)).beta
    expected = _expected_beta()
    assert beta.shape == expected.shape
    np.testing.assert_allclose(beta, expected, atol=1e-9)

    rng = np.random.default_rng(3)
    noisy = Climatology().fit(doy, hour, T + 0.5 * rng.standard_normal(T.shape),
                              np.ones_like(T)).beta
    assert np.abs(noisy - expected).max() < 0.05, (
        f"шум смещает коэффициенты: max|Δβ| = {np.abs(noisy - expected).max():.4g}")


def _cos_zenith(doy, hour_utc, lat, lon):
    """Косинус зенитного угла из astro_features, numpy in - numpy out."""
    t = torch.as_tensor(np.atleast_1d(hour_utc), dtype=torch.float64)
    d = torch.as_tensor(np.broadcast_to(np.atleast_1d(doy), t.shape).copy(),
                        dtype=torch.float64)
    la = torch.as_tensor(np.broadcast_to(np.atleast_1d(lat), t.shape).copy(),
                         dtype=torch.float64)
    lo = torch.as_tensor(np.broadcast_to(np.atleast_1d(lon), t.shape).copy(),
                         dtype=torch.float64)
    return astro_features(d, t, la, lo)[2].numpy()


def _solar_noon_utc(doy, lon):
    """Час UTC солнечного полудня по той же формуле уравнения времени, что в astro."""
    B = 2 * math.pi * (doy - 81.0) / 364.0
    eot_min = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    return (12.0 - lon / 15.0 - eot_min / 60.0) % 24.0


SOLAR_SITES = [(55.75, 37.62), (0.0, 0.0), (-33.9, 151.2), (64.1, -21.9), (35.7, 139.7)]


@pytest.mark.parametrize("lat, lon", SOLAR_SITES)
@pytest.mark.parametrize("doy", [0.0, 80.0, 172.0, 265.0, 355.0])
def test_solar_noon_maximises_cos_zenith(lat, lon, doy):
    """В солнечный полдень косинус зенитного угла максимален по часу суток."""
    grid = np.arange(0.0, 24.0, 1.0 / 60.0)
    cz_grid = _cos_zenith(doy, grid, lat, lon)
    noon = _solar_noon_utc(doy, lon)
    cz_noon = float(_cos_zenith(doy, np.array([noon]), lat, lon)[0])

    assert cz_noon >= cz_grid.max() - 1e-9, (
        f"полдень не максимум: cz(полдень) = {cz_noon:.6f}, "
        f"max по сетке = {cz_grid.max():.6f}")
    best = float(grid[int(np.argmax(cz_grid))])
    assert min(abs(best - noon), 24.0 - abs(best - noon)) <= 1.0 / 60.0


def _day_length_hours(doy, lat, lon, step=1.0 / 600.0):
    grid = np.arange(0.0, 24.0, step)
    return float((_cos_zenith(doy, grid, lat, lon) > 0).sum()) * step


EQUINOX_DOY = 81.31


def test_equator_equinox_day_length_is_twelve_hours():
    for lon in (-180.0, -75.0, 0.0, 100.0, 179.0):
        got = _day_length_hours(EQUINOX_DOY, 0.0, lon)
        assert got == pytest.approx(12.0, abs=0.02), f"долгота {lon}: {got:.4f} ч"


def test_declination_is_zero_at_equinox_and_extreme_at_solstices():
    decl = lambda d: -0.40928 * math.cos(2 * math.pi * (d + 10.0) / 365.24)
    assert abs(decl(EQUINOX_DOY)) < 1e-3
    assert decl(171.5) == pytest.approx(0.40928, abs=1e-3)
    assert decl(355.0) == pytest.approx(-0.40928, abs=1e-3)


def test_day_length_follows_season_and_hemisphere():
    """Полярное лето длиннее полярной зимы, южное полушарие зеркально северному."""
    june, december = 171.5, 355.0
    assert _day_length_hours(june, 60.0, 0.0) > 18.0
    assert _day_length_hours(december, 60.0, 0.0) < 6.5
    assert _day_length_hours(june, -60.0, 0.0) == pytest.approx(
        _day_length_hours(december, 60.0, 0.0), abs=0.2)


SERIES_T0 = int(to_utc_hour(datetime(2019, 11, 15, 3)))
SERIES_N = 20_000


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(11)
    x = np.stack([10 + 5 * rng.standard_normal(SERIES_N),
                  1000 + rng.standard_normal(SERIES_N),
                  (60 + 5 * rng.standard_normal(SERIES_N)).clip(1, 100)],
                 axis=-1).astype(np.float32)
    mask = np.ones((SERIES_N, 3), np.float32)
    mask[5000:5100] = 0.0
    x = x * mask
    return x, mask


@pytest.mark.parametrize("L", [0, 1, 24, L_MAX // 2, L_MAX])
def test_history_is_right_aligned_in_the_buffer(series, L):
    x, mask = series
    t = 9_000
    xh, mh = slice_history(x, mask, t, L)

    assert xh.shape == (L_MAX, 3) and mh.shape == (L_MAX, 3)
    assert np.array_equal(xh[:L_MAX - L], np.zeros((L_MAX - L, 3), np.float32))
    assert np.array_equal(mh[:L_MAX - L], np.zeros((L_MAX - L, 3), np.float32))
    if L:
        np.testing.assert_array_equal(xh[L_MAX - L:], x[t - L:t])
        np.testing.assert_array_equal(mh[L_MAX - L:], mask[t - L:t])
    assert np.all(xh[mh == 0] == 0.0), "инвариант «маска ноль → значение ноль»"


def test_last_history_hour_is_the_hour_before_the_horizon(series):
    x, mask = series
    t = 9_000
    xh, _ = slice_history(x, mask, t, L_MAX)
    np.testing.assert_array_equal(xh[-1], x[t - 1])
    y, ym = slice_target(x, mask, t)
    assert y.shape == (H,) and ym.shape == (H,)
    np.testing.assert_array_equal(y, x[t:t + H, 0] * mask[t:t + H, 0])
    np.testing.assert_array_equal(ym, mask[t:t + H, 0])


@pytest.mark.parametrize("L, t, lo, expect", [
    (None, 5_000, 0, L_MAX),
    (10 * L_MAX, 5_000, 0, L_MAX),
    (L_MAX, 100, 0, 100),
    (L_MAX, 5_000, 4_800, 200),
    (0, 5_000, 0, 0),
])
def test_history_len_is_clamped_by_buffer_and_split(L, t, lo, expect):
    assert history_len(L, t, lo) == expect


def test_footprint_matches_the_hours_actually_read(series):
    x, mask = series
    lo, t, L = 4_800, 5_400, L_MAX
    L_eff = history_len(L, t, lo)
    lo_f, hi_f = footprint(t, L, lo)
    assert (lo_f, hi_f) == (t - L_eff, t + H) == (lo, t + H)

    xh, _ = slice_history(x, mask, t, L_eff)
    y, _ = slice_target(x, mask, t)
    np.testing.assert_array_equal(xh[L_MAX - L_eff:, 0], (x[:, 0] * mask[:, 0])[lo_f:t])
    np.testing.assert_array_equal(y, (x[:, 0] * mask[:, 0])[t:hi_f])


@pytest.mark.parametrize("t", [700, 9_000, 19_000])
def test_window_calendar_matches_absolute_hours(t):
    """doy/hour окна соответствуют ровно тем абсолютным часам, из которых собрано окно."""
    k = np.arange(L_MAX)
    doy_h, hour_h = window_calendar(SERIES_T0, t - L_MAX + k)
    fut = np.arange(t, t + H)
    doy_f, hour_f = window_calendar(SERIES_T0, fut)

    for idx, (doy, hour) in ((t - L_MAX + k, (doy_h, hour_h)), (fut, (doy_f, hour_f))):
        ref = [datetime(1970, 1, 1) + timedelta(hours=int(SERIES_T0 + i)) for i in idx]
        ref_doy = np.array([(r - datetime(r.year, 1, 1)).total_seconds() / 86400.0
                            for r in ref])
        ref_hour = np.array([r.hour for r in ref], float)
        np.testing.assert_allclose(doy, ref_doy, atol=1e-3)
        np.testing.assert_allclose(hour, ref_hour, atol=1e-3)
        np.testing.assert_allclose(np.mod(doy, 1.0) * 24.0, hour, atol=1e-2,
                                   err_msg="дробная часть doy обязана совпадать с hour / 24")

    np.testing.assert_allclose(np.diff(hour_h) % 24.0, 1.0, atol=1e-3)
    np.testing.assert_allclose(np.diff(hour_f) % 24.0, 1.0, atol=1e-3)
    assert hour_f[0] == pytest.approx((hour_h[-1] + 1) % 24.0, abs=1e-3)


def test_horizon_starts_one_hour_after_the_last_history_hour():
    t = 9_000
    _, hour_h = window_calendar(SERIES_T0, np.arange(t - L_MAX, t))
    doy_f, hour_f = window_calendar(SERIES_T0, np.arange(t, t + H))
    assert len(doy_f) == H
    assert hour_f[0] == pytest.approx((hour_h[-1] + 1) % 24.0, abs=1e-4)
    assert (doy_f >= 0).all() and (doy_f < 366.0).all()
