"""Тесты: маска валидности цели и инварианты данных."""
import csv

import numpy as np
import pytest
import torch

from mayak.constants import H, L_MAX, QUANTILES
from mayak.data.masking import (TargetMaskConfig, enforce_invariant, target_window_ok)
from mayak.loss import mayak_loss, pinball, pinball_loss
from mayak.metrics import (fit_conformal_shift, metric_table, skill, wmean)

NQ = len(QUANTILES)

def _rand_q(g, B, Hh=H):
    """Монотонные квантили вокруг случайной медианы."""
    mu = 10 + 5 * torch.randn(B, Hh, generator=g)
    gaps = torch.rand(B, Hh, NQ - 1, generator=g)
    offs = torch.cat([torch.zeros(B, Hh, 1), gaps.cumsum(-1)], -1)
    return mu[..., None] + offs - offs[..., 3:4]


def _half_mask(g, B, Hh=H):
    m = (torch.rand(B, Hh, generator=g) < 0.5).float()
    m[:, 0] = 1.0
    return m


def _with_garbage(y, m, value=float("nan")):
    return torch.where(m > 0, y, torch.full_like(y, value))


def _toy_batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.stack([15 + 3 * torch.randn(B, L_MAX, generator=g),
                     1000 + 5 * torch.randn(B, L_MAX, generator=g),
                     (60 + 10 * torch.randn(B, L_MAX, generator=g)).clamp(5, 100)], -1)
    mask = (torch.rand(B, L_MAX, 3, generator=g) < 0.8).float()
    x, mask = enforce_invariant(x, mask)
    y = 12 + 4 * torch.randn(B, H, generator=g)
    y_mask = _half_mask(g, B)
    y, y_mask = enforce_invariant(y, y_mask)
    return {
        "lat": torch.rand(B, generator=g) * 120 - 60,
        "lon": torch.rand(B, generator=g) * 360 - 180,
        "elev": torch.rand(B, generator=g) * 500,
        "x_hist": x, "mask_hist": mask,
        "doy_hist": torch.rand(B, L_MAX, generator=g) * 365,
        "hour_hist": torch.rand(B, L_MAX, generator=g) * 24,
        "doy_fut": torch.rand(B, H, generator=g) * 365,
        "hour_fut": torch.rand(B, H, generator=g) * 24,
        "y": y, "y_mask": y_mask,
    }


def test_enforce_invariant_numpy_and_torch():
    x = np.array([[1.0, np.nan], [np.inf, 4.0]], np.float32)
    m = np.array([[1, 0], [0, 2]], np.float32)
    xn, mn = enforce_invariant(x, m)
    assert np.array_equal(xn, [[1.0, 0.0], [0.0, 4.0]])
    assert np.array_equal(mn, [[1.0, 0.0], [0.0, 1.0]])

    xt, mt = enforce_invariant(torch.from_numpy(x), torch.from_numpy(m))
    assert torch.equal(xt, torch.from_numpy(xn))
    assert torch.equal(mt, torch.from_numpy(mn))


def test_run_qc_zeroes_only_its_channel():
    from mayak.data.qc import run_qc
    n = 200
    h = np.arange(n)
    T = (10 + 5 * np.sin(2 * np.pi * h / 24)).astype(np.float32)
    P = np.full(n, 1000.0, np.float32) + np.sin(h / 7).astype(np.float32)
    RH = np.full(n, 60.0, np.float32) + np.cos(h / 5).astype(np.float32)
    T[50] = 999.0
    valid = np.ones(n, np.uint8)
    x, mask = run_qc(T, P, RH, valid)
    assert mask[50, 0] == 0 and x[50, 0] == 0.0
    assert mask[50, 1] == 1 and mask[50, 2] == 1
    assert np.all(x[mask == 0] == 0.0)


def test_pinball_ignores_values_under_mask():
    g = torch.Generator().manual_seed(0)
    B = 8
    q = _rand_q(g, B).requires_grad_(True)
    y = 10 + 5 * torch.randn(B, H, generator=g)
    m = _half_mask(g, B)
    scale = torch.full((B, H), 3.0)

    l0 = pinball(q, y * m, m, scale)
    (g0,) = torch.autograd.grad(l0, q)
    for garbage in (float("nan"), 1e6, -273.0):
        l1 = pinball(q, _with_garbage(y, m, garbage), m, scale)
        (g1,) = torch.autograd.grad(l1, q)
        assert torch.equal(l0, l1)
        assert torch.equal(g0, g1)
        assert torch.isfinite(g1).all()


def test_pinball_equals_loss_on_valid_pairs_only():
    """Половина часов под маской ≡ этих часов нет вовсе."""
    g = torch.Generator().manual_seed(1)
    B = 6
    q = _rand_q(g, B)
    y = 10 + 5 * torch.randn(B, H, generator=g)
    m = _half_mask(g, B)
    scale = 1.0 + torch.rand(B, H, generator=g)

    masked = pinball(q, y, m, scale)
    keep = m > 0
    only_valid = pinball(q[keep][None], y[keep][None],
                         torch.ones(1, int(keep.sum())), scale[keep][None])
    assert torch.allclose(masked, only_valid, rtol=1e-6, atol=1e-7)

    taus = torch.tensor(QUANTILES)
    err = (y[..., None] - q) / scale[..., None]
    per_pair = torch.maximum(taus * err, (taus - 1) * err).mean(-1)
    manual = per_pair[keep].sum() / keep.sum()
    assert torch.allclose(masked, manual, rtol=1e-6)


def test_pinball_all_masked_is_zero_and_finite():
    g = torch.Generator().manual_seed(2)
    q = _rand_q(g, 2).requires_grad_(True)
    y = torch.full((2, H), float("nan"))
    loss = pinball(q, y, torch.zeros(2, H), torch.ones(2, H))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(q.grad).all() and (q.grad == 0).all()


def test_training_with_masked_hours_equals_training_without_them():
    """Две копии модели, одинаковая инициализация, одинаковые батчи; различается только
    содержимое невалидных часов цели. После нескольких шагов оптимизатора лосс на
    каждом шаге и все веса совпадают бит в бит.
    """
    from mayak.model import MAYAK
    torch.manual_seed(0)
    base = MAYAK()
    models = [MAYAK(), MAYAK()]
    for mdl in models:
        mdl.load_state_dict(base.state_dict())
    opts = [torch.optim.Adam(mdl.parameters(), lr=1e-3) for mdl in models]

    losses = [[], []]
    for step in range(3):
        batch = _toy_batch(B=2, seed=step)
        garbage = dict(batch, y=_with_garbage(batch["y"], batch["y_mask"], 1e4))
        for i, (mdl, opt, b) in enumerate(zip(models, opts, (batch, garbage))):
            torch.manual_seed(100 + step)
            opt.zero_grad()
            loss = mayak_loss(mdl(b), b["y"], b["y_mask"])
            loss.backward()
            opt.step()
            losses[i].append(loss.item())

    assert losses[0] == losses[1]
    for p0, p1 in zip(*(mdl.parameters() for mdl in models)):
        assert torch.equal(p0, p1)


@pytest.mark.parametrize("name", ["gru", "dlinear"])
def test_baselines_loss_ignores_masked_hours(name):
    from mayak.baselines import DLinear, GRUSeq2Seq
    torch.manual_seed(0)
    model = {"gru": GRUSeq2Seq, "dlinear": DLinear}[name]()
    batch = _toy_batch(B=2)
    grads = []
    for y in (batch["y"], _with_garbage(batch["y"], batch["y_mask"], float("nan"))):
        model.zero_grad()
        loss = pinball_loss(model(batch), y, batch["y_mask"])
        loss.backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    assert all(torch.equal(a, b) for a, b in zip(*grads))


def _np_case(seed=0, N=400):
    rng = np.random.default_rng(seed)
    y = 10 + 5 * rng.standard_normal((N, H))
    mu = y + rng.standard_normal((N, H))
    mu_clim = y + 3 * rng.standard_normal((N, H))
    q = mu[..., None] + np.linspace(-2, 2, NQ)
    w = (rng.random((N, H)) < 0.5).astype(np.float32)
    w[:, 0] = 1
    return y, mu, q, mu_clim, w


def test_metric_table_ignores_masked_pairs():
    y, mu, q, mu_clim, w = _np_case()
    ref = metric_table(y * w, mu, q, mu_clim, w, leads=(1, 24, 168))
    garbage = np.where(w > 0, y, 1e6)
    got = metric_table(garbage, mu, q, mu_clim, w, leads=(1, 24, 168))
    assert ref == got


def test_metrics_match_manual_on_valid_pairs():
    y, mu, q, mu_clim, w = _np_case(seed=1)
    tbl = metric_table(y, mu, q, mu_clim, w, leads=(24,))[24]
    k = w[:, 23] > 0
    e = mu[k, 23] - y[k, 23]
    assert tbl["MAE"] == pytest.approx(np.abs(e).mean())
    assert tbl["RMSE"] == pytest.approx(np.sqrt((e ** 2).mean()))
    mse_c = ((mu_clim[k, 23] - y[k, 23]) ** 2).mean()
    assert tbl["Skill"] == pytest.approx(1 - (e ** 2).mean() / mse_c)
    lo, hi = q[k, 23, 0], q[k, 23, 6]
    assert tbl["PICP90"] == pytest.approx(((y[k, 23] >= lo) & (y[k, 23] <= hi)).mean())
    assert tbl["n_valid"] == int(k.sum())


def test_skill_denominator_uses_same_pairs():
    """Невалидные пары с огромной ошибкой климатологии не должны раздувать скилл."""
    y = np.zeros((4, 1))
    mu = np.array([[1.0], [1.0], [0.0], [0.0]])
    mu_clim = np.array([[2.0], [2.0], [100.0], [100.0]])
    w = np.array([[1.0], [1.0], [0.0], [0.0]])
    assert skill(y, mu, mu_clim, w) == pytest.approx(1 - 1 / 4)


def test_wmean_empty_is_nan():
    assert np.isnan(wmean(np.ones(5), np.zeros(5)))


def _slow_ok(mask_T, t, horizon, cfg):
    seg = mask_T[t:t + horizon] > 0
    return seg.mean() >= cfg.min_valid_frac and seg[:cfg.first_day_hours].any()


def test_target_window_rule_edge_cases():
    cfg = TargetMaskConfig(min_valid_frac=0.5, first_day_hours=24)
    m = np.zeros(H)
    m[:84] = 1
    assert target_window_ok(m, [0], H, cfg)[0]
    m[83] = 0
    assert not target_window_ok(m, [0], H, cfg)[0]
    m = np.ones(H)
    m[:24] = 0
    assert not target_window_ok(m, [0], H, cfg)[0]
    assert not target_window_ok(np.zeros(H), [0], H, cfg)[0]


def test_target_window_vectorized_matches_slow():
    rng = np.random.default_rng(0)
    cfg = TargetMaskConfig(min_valid_frac=0.6, first_day_hours=12)
    m = (rng.random(3000) < 0.62).astype(np.float32)
    m[1000:1300] = 0
    starts = np.arange(0, 3000 - H)
    fast = target_window_ok(m, starts, H, cfg)
    slow = np.array([_slow_ok(m, t, H, cfg) for t in starts])
    assert np.array_equal(fast, slow)
    assert fast.any() and not fast.all()


N_HOURS = 12_000
GAP = (700, 1000)
GAP_EVAL = (11_000, 11_300)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    (root / "stations").mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i, (split, lat) in enumerate([("train", 50.0), ("train", 10.0),
                                       ("unseen_val", 45.0), ("unseen_test", 30.0)]):
        h = np.arange(N_HOURS)
        T = 10 + 6 * np.sin(2 * np.pi * h / 24) + 0.3 * rng.standard_normal(N_HOURS)
        P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(N_HOURS)
        RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(N_HOURS)
        valid = np.ones(N_HOURS, np.uint8)
        if i in (0, 3):
            valid[GAP[0]:GAP[1]] = 0
            valid[GAP_EVAL[0]:GAP_EVAL[1]] = 0
        np.savez(root / "stations" / f"s{i}.npz", T=T.astype(np.float32),
                 P=P.astype(np.float32), RH=RH.astype(np.float32), valid=valid,
                 t0_doy=0.0, t0_hour=0.0)
        rows.append(dict(id=f"s{i}", lat=lat, lon=0.0, elev=100.0, koppen="Cfb", split=split))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def train_ds(manifest):
    from mayak.data.dataset import WindowDataset
    return WindowDataset(manifest, windows_per_epoch=64, seed=0)


def _check_item(item):
    x, m = item["x_hist"].numpy(), item["mask_hist"].numpy()
    y, ym = item["y"].numpy(), item["y_mask"].numpy()
    assert np.all(x[m == 0] == 0), "инвариант нарушен в истории"
    assert np.all(y[ym == 0] == 0), "инвариант нарушен в цели"
    assert set(np.unique(m)) <= {0.0, 1.0} and set(np.unique(ym)) <= {0.0, 1.0}
    assert ym.mean() >= 0.5 and ym[:24].any(), "окно не прошло бы правило 1.5"


def test_window_dataset_items_obey_rules(train_ds):
    assert train_ds.filter_stats.rejected_frac > 0
    for i in range(len(train_ds)):
        _check_item(train_ds[i])


def test_window_dataset_never_samples_rejected_starts(train_ds):
    s0 = train_ds.st[0]
    starts = s0["starts"]
    assert GAP[0] + 10 not in set(starts.tolist())
    assert target_window_ok(s0["mask"][:, 0], starts, H).all()


def test_target_mask_comes_only_from_data(train_ds):
    """Аугментации не трогают маску цели, даже когда задевают историю."""
    s0 = train_ds.st[0]
    ok = s0["starts"]
    partial = [t for t in ok.tolist() if t + H > GAP[0] and t < GAP[0]][:20]
    assert partial
    for t in partial:
        for L in (0, 48, 400):
            item = train_ds.build(s0, t, min(L, t))
            expected = s0["mask"][t:t + H, 0]
            assert np.array_equal(item["y_mask"].numpy(), expected)
            _check_item(item)


def test_holdout_and_eval_sets_share_filter(manifest, tmp_path, monkeypatch):
    from mayak import baselines as BL
    from mayak.data.dataset import HoldoutDataset
    from mayak.evaluate import EvalSet

    hd = HoldoutDataset(manifest, station_split="unseen_val", time_key="calib", every_hours=24)
    assert len(hd) > 0
    for i in range(len(hd)):
        _check_item(hd[i])

    monkeypatch.chdir(tmp_path)
    clims = BL.fit_climatologies(manifest, force=True)
    ds_full = EvalSet(clims, station_splits=("unseen_test",), manifest=manifest,
                      time_key="test", every_hours=24)
    ds_l0 = EvalSet(clims, station_splits=("unseen_test",), manifest=manifest,
                    time_key="test", every_hours=24, L=0)
    assert ds_full.items == ds_l0.items
    assert ds_full.filter_stats.rejected_frac > 0
    for i in range(len(ds_full)):
        _check_item(ds_full[i])


def test_model_output_ignores_values_under_input_mask():
    """Каналы энкодера, разности давления с лагом, суточные сводки и масса свидетельств
    читают значения только через маску: мусор под маской не меняет прогноз."""
    from mayak.model import MAYAK
    torch.manual_seed(0)
    model = MAYAK().eval()
    batch = _toy_batch(B=2, seed=3)
    g = torch.Generator().manual_seed(9)
    noise = torch.stack([30 + 5 * torch.rand(2, L_MAX, generator=g),
                         950 + 50 * torch.rand(2, L_MAX, generator=g),
                         20 + 60 * torch.rand(2, L_MAX, generator=g)], -1)
    garbage = dict(batch, x_hist=torch.where(batch["mask_hist"] > 0, batch["x_hist"], noise))
    with torch.no_grad():
        a, b = model(batch), model(garbage)
    for k in ("q", "mu", "e", "z"):
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_daily_summaries_mask_convention():
    from mayak.model import MAYAK
    aT = torch.randn(1, 48)
    vt = torch.ones(1, 48)
    vt[0, 24:] = 0
    dp = torch.randn(1, 48)
    vp = torch.ones(1, 48)
    vp[0, :12] = 0
    dp = dp * vp
    s, has = MAYAK.daily_summaries(aT * vt, dp, vt, vp)
    assert has.tolist() == [[1.0, 0.0]]
    assert s[0, 0, 4] == pytest.approx(1.0) and s[0, 1, 4] == 0.0
    assert s[0, 0, 3] == pytest.approx(dp[0, 12:24].mean().item(), abs=1e-6)
    assert torch.all(s[0, 1, :3] == 0)


def test_streaming_daily_summaries_match_batch():
    from mayak.astro import astro_features
    from mayak.model import MAYAK
    from mayak.runtime.streaming import StreamingMayak
    torch.manual_seed(0)
    model = MAYAK().eval()
    lat, lon, elev = 50.0, 5.0, 10.0
    n = 72
    rng = np.random.default_rng(0)
    T = 10 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.standard_normal(n)
    P = 1000 + rng.standard_normal(n)
    RH = np.full(n, 70.0)
    vT = np.ones(n, bool)
    vT[5:9] = False
    vT[24:48] = False
    vP = np.ones(n, bool)
    vP[50:60] = False
    doy = (100 + np.arange(n) / 24).astype(np.float32)
    hour = (np.arange(n) % 24).astype(np.float32)

    stream = StreamingMayak(model, lat, lon, elev)
    for k in range(n):
        stream.step(T[k] if vT[k] else None, P[k] if vP[k] else None, RH[k], doy[k], hour[k])

    x = torch.tensor(np.stack([np.where(vT, T, 0), np.where(vP, P, 0), RH], -1),
                     dtype=torch.float32)[None]
    mk = torch.tensor(np.stack([vT, vP, np.ones(n, bool)], -1), dtype=torch.float32)[None]
    with torch.no_grad():
        astro = astro_features(torch.from_numpy(doy)[None], torch.from_numpy(hour)[None],
                               torch.tensor([[lat]]), torch.tensor([[lon]]))
        loc = model.loc(torch.tensor([lat]), torch.tensor([lon]), torch.tensor([elev]))
        mu0, sg0, df0 = model.field.evaluate(model.field.coefficients(loc), astro)
        ch, aT, vt = model.build_channels(x, mk, astro, mu0, sg0, df0)
        summ, has = model.daily_summaries(aT, ch[:, 3], vt, model.lag_valid(mk[..., 1], 24))

    assert torch.equal(stream.day_mask[:, -3:], has)
    assert has[0, 1] == 0
    assert torch.allclose(stream.day_summ[:, -3:], summ, atol=1e-5)


def test_seasonal_naive_never_uses_invalid_values():
    from mayak.baselines import seasonal_naive_forecast
    x = np.zeros((2, L_MAX, 3), np.float32)
    m = np.zeros((2, L_MAX, 3), np.float32)
    days = np.arange(L_MAX) // 24
    x[0, :, 0] = 20 + days
    m[0, :, 0] = 1
    m[0, -24:, 0] = 0
    m[0, -48 + 5, 0] = 0
    x, m = enforce_invariant(x, m)
    mu_clim = np.full((2, H), -7.0, np.float32)
    mu, q = seasonal_naive_forecast(x, m, mu_clim, np.ones(2, np.float32))
    assert mu[0, 0] == 20 + 26
    assert mu[0, 5] == 20 + 25
    assert np.all(mu[1] == -7.0)
    assert not np.any(mu == 0.0)
    assert np.all(np.diff(q, axis=-1) >= 0)


def test_recent_anomaly_rule():
    from mayak.baselines import recent_anomaly
    from mayak.data.climatology import Climatology
    clim = Climatology()
    clim.beta = np.zeros(15)
    xT = np.full(100, 3.0)
    mT = np.zeros(100)
    mT[90:95] = 1
    assert recent_anomaly(xT, mT, clim, 100, 0.0, 0.0) == (0.0, False)
    mT[85] = 1
    a, ok = recent_anomaly(np.where(mT > 0, xT, 1e6), mT, clim, 100, 0.0, 0.0)
    assert ok and a == pytest.approx(3.0)


def _clims_case(garbage):
    from mayak.data.climatology import Climatology
    from mayak.data.dataset import _calendar
    rng = np.random.default_rng(0)
    n = N_HOURS
    h = np.arange(n)
    T = (10 + 6 * np.sin(2 * np.pi * h / 24) + 2 * rng.standard_normal(n)).astype(np.float32)
    m = (rng.random(n) < 0.7).astype(np.float32)
    x = np.zeros((n, 3), np.float32)
    x[:, 0] = np.where(m > 0, T, garbage)
    mask = np.stack([m, m, m], -1)
    doy, hour = _calendar(0.0, 0.0, h)
    clim = Climatology().fit(doy[:1794], hour[:1794], x[:1794, 0], m[:1794])
    return {"s": dict(clim=clim, N=n, t0d=0.0, t0h=0.0, x=x, mask=mask)}


def test_climatology_and_damped_persistence_ignore_masked_values():
    from mayak.baselines import fit_damped_persistence
    a, b = _clims_case(0.0), _clims_case(1e4)
    assert np.allclose(a["s"]["clim"].beta, b["s"]["clim"].beta)
    ra = fit_damped_persistence(a, n_windows=200, seed=0)
    rb = fit_damped_persistence(b, n_windows=200, seed=0)
    assert np.array_equal(ra, rb)


def test_conformal_shift_ignores_masked_pairs():
    y, mu, q, mu_clim, w = _np_case(seed=2)
    ref = fit_conformal_shift(np.where(w > 0, y, 0.0), q, w)
    got = fit_conformal_shift(np.where(w > 0, y, 1e6), q, w)
    assert np.array_equal(ref, got)
