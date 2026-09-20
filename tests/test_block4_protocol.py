"""Тесты: климатологический масштаб, единая нормировка функции потерь,
единый протокол обучения."""
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.constants import H, L_MAX, QUANTILES
from mayak.data import store as S
from mayak.data.climatology import ABS_TO_SD, SCALE_FLOOR_FRAC, Climatology
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL, time_bounds
from mayak.loss import NORM_SCALE_CLAMP, forecast_loss, pinball
from mayak.protocol import (ARCH_NAMES, DEFAULT_PROTOCOL, Protocol, ProtocolError, Stage,
                            check_deviations_documented, protocol_for, read_journal,
                            run_protocol)
from mayak.timeaxis import window_calendar

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 12_000
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("t2", ROLE_TRAIN),
            ("v0", ROLE_VAL), ("v1", ROLE_VAL), ("x0", ROLE_TEST)]


def _write_station(root, sid, seed, n=N_HOURS):
    """Шум с суточным ходом масштаба: ночью спокойнее, днём разброс больше."""
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    sd = 1.0 + 0.5 * np.cos(2 * np.pi * h / 24)
    T = 10 + 6 * np.sin(2 * np.pi * h / 24) + sd * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data4")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb", split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def store(manifest):
    S._STORES.clear()
    return S.get_store(manifest)


def _hetero_series(n_years=8, seed=0, t0=0):
    """Ряд с известным масштабом остатка sd(doy, hour), представимым базисом масштаба."""
    rng = np.random.default_rng(seed)
    k = np.arange(int(n_years * 8766))
    doy, hour = (v.astype(np.float64) for v in window_calendar(t0, k))
    wy, wd = 2 * np.pi / 365.24, 2 * np.pi / 24
    mean = 8 + 10 * np.cos(wy * doy) + 5 * np.sin(wd * hour)
    sd = 2.0 + 0.8 * np.cos(wd * hour) + 0.5 * np.cos(wy * doy)
    T = mean + sd * rng.standard_normal(k.size)
    return doy, hour, T, sd


def test_scale_recovers_known_heteroscedastic_noise():
    doy, hour, T, sd = _hetero_series()
    clim = Climatology().fit(doy, hour, T, np.ones_like(T))
    est = clim.scale(doy, hour)
    rel = np.abs(est / sd - 1)
    assert rel.mean() < 0.02 and rel.max() < 0.08, (rel.mean(), rel.max())
    day, night = np.abs(hour - 0) < 0.5, np.abs(hour - 12) < 0.5
    assert est[day].mean() / est[night].mean() == pytest.approx(2.8 / 1.2, rel=0.05)


def test_constant_noise_scale_matches_scalar_sigma():
    rng = np.random.default_rng(1)
    doy, hour, _, _ = _hetero_series(n_years=5)
    T = 5 + 3 * np.sin(2 * np.pi * hour / 24) + 1.7 * rng.standard_normal(doy.size)
    clim = Climatology().fit(doy, hour, T, np.ones_like(T))
    est = clim.scale(doy, hour)
    assert est.std() / est.mean() < 0.03
    assert est.mean() == pytest.approx(clim.sigma, rel=0.03)
    assert est.mean() == pytest.approx(1.7, rel=0.03)


def test_scale_fit_ignores_values_under_mask():
    doy, hour, T, _ = _hetero_series(n_years=4, seed=2)
    m = (np.random.default_rng(3).random(T.size) < 0.6).astype(np.float32)
    ref = Climatology().fit(doy, hour, np.where(m > 0, T, 0.0), m)
    got = Climatology().fit(doy, hour, np.where(m > 0, T, 1e6), m)
    assert np.array_equal(ref.scale_beta, got.scale_beta) and np.array_equal(ref.beta, got.beta)


def test_scale_is_positive_with_floor():
    doy, hour, T, _ = _hetero_series(n_years=3, seed=4)
    clim = Climatology().fit(doy, hour, T, np.ones_like(T))
    clim.scale_beta = -np.abs(clim.scale_beta)
    s = clim.scale(doy[:100], hour[:100])
    assert np.all(s == pytest.approx(SCALE_FLOOR_FRAC * clim.sigma)) and np.all(s > 0)


def test_scale_uses_abs_residual_convention():
    """Для нормального остатка масштаб — это σ, а не E|r| (множитель √(π/2))."""
    assert ABS_TO_SD == pytest.approx(np.sqrt(np.pi / 2))


def test_climatology_without_scale_fails_loudly():
    c = Climatology.from_params(np.zeros(15), 1.0, None)
    with pytest.raises(RuntimeError, match="масштаб"):
        c.scale(np.zeros(3), np.zeros(3))


def test_cache_stores_scale_fitted_on_train_window(store):
    cache = Path(store.path)
    assert (cache / "clim_scale_beta.npy").exists()
    for sid, s in store.stations.items():
        lo, hi = time_bounds(s["N"])["train"]
        d, h = window_calendar(s["t0"], np.arange(lo, hi))
        ref = S.new_climatology().fit(d.astype(np.float64), h.astype(np.float64),
                                      s["x"][lo:hi, 0], s["mask"][lo:hi, 0])
        assert np.allclose(s["clim"].scale_beta, ref.scale_beta, rtol=1e-10, atol=1e-10), sid
        dd, hh = window_calendar(s["t0"], np.arange(24))
        sc = s["clim"].scale(dd, hh)
        assert sc[0] > 1.8 * sc[12], sid


def test_cache_key_depends_on_climatology_rules(manifest, monkeypatch):
    key = S.cache_key(S.key_payload(manifest))
    monkeypatch.setattr(S, "CLIM_VERSION", "999")
    assert S.cache_key(S.key_payload(manifest)) != key
    monkeypatch.undo()
    monkeypatch.setitem(S.CLIM_PARAMS, "scale_n_day", 3)
    assert S.cache_key(S.key_payload(manifest)) != key


def test_deep_check_catches_scale_not_from_train_window(store):
    import copy
    from mayak.leakage import LeakageError, check_climatology
    bad = copy.copy(store)
    bad.stations = {k: dict(v) for k, v in store.stations.items()}
    s = bad.stations["t1"]
    lo, hi = time_bounds(s["N"])["test"]
    d, h = window_calendar(s["t0"], np.arange(lo, hi))
    other = S.new_climatology().fit(d.astype(float), h.astype(float), s["x"][lo:hi, 0],
                                    s["mask"][lo:hi, 0])
    s["clim"] = Climatology.from_params(s["clim"].beta, s["clim"].sigma, other.scale_beta,
                                        **{k: S.CLIM_PARAMS[k] for k in S.CLIM_BASIS})
    check_climatology(bad)
    with pytest.raises(LeakageError, match="масштаба"):
        check_climatology(bad, deep=True)


def _expected_scale(clim, t0, t):
    d, h = window_calendar(t0, np.arange(t, t + H))
    return clim.scale(d, h).astype(np.float32)


def test_window_dataset_norm_scale_is_station_climatology(manifest, store):
    from mayak.data.dataset import WindowDataset
    ds = WindowDataset(manifest, windows_per_epoch=8, seed=0, store=store)
    for s in ds.st:
        t = int(s["starts"][len(s["starts"]) // 2])
        item = ds.build(s, t, L_MAX)
        ns = item["norm_scale"].numpy()
        assert ns.shape == (H,) and ns.dtype == np.float32 and np.all(ns > 0)
        assert np.allclose(ns, _expected_scale(store.stations[s["id"]]["clim"], s["t0"], t))
    for i in range(4):
        assert ds[i]["norm_scale"].shape == (H,)


def test_norm_scale_unaffected_by_augmentations(manifest, store):
    from mayak.data.dataset import WindowDataset, make_streams
    ds = WindowDataset(manifest, windows_per_epoch=8, seed=0, store=store)
    s = ds.st[0]
    t = int(s["starts"][10])
    items = []
    for aug_seed in range(4):
        _, ds.rng_aug = make_streams(1000 + aug_seed)
        items.append(ds.build(s, t, L_MAX))
    assert any(not torch.equal(items[0]["x_hist"], it["x_hist"]) for it in items[1:])
    assert all(torch.equal(items[0]["norm_scale"], it["norm_scale"]) for it in items[1:])


def test_holdout_and_eval_sets_emit_same_norm_scale(manifest, store):
    from mayak.data.dataset import HoldoutDataset
    from mayak.evaluate import EvalSet
    hd = HoldoutDataset(manifest, station_split=ROLE_VAL, time_key="val", store=store)
    for i in range(0, len(hd), max(1, len(hd) // 5)):
        m = hd.meta[i]
        assert np.allclose(hd[i]["norm_scale"].numpy(),
                           _expected_scale(store.stations[m["id"]]["clim"], m["t0"], m["t"]))
    ev = EvalSet(store.clims(), station_splits=(ROLE_VAL,), manifest=manifest, time_key="val")
    by_key = {(m["id"], m["t"]): hd[i]["norm_scale"] for i, m in enumerate(hd.meta)}
    n = 0
    for i, (sid, t) in enumerate(ev.items):
        if (sid, t) in by_key:
            assert torch.equal(ev[i]["norm_scale"], by_key[(sid, t)])
            n += 1
    assert n > 0


def _toy_batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(B, L_MAX, 3, generator=g) < 0.8).float()
    x = torch.stack([15 + 3 * torch.randn(B, L_MAX, generator=g),
                     1000 + 5 * torch.randn(B, L_MAX, generator=g),
                     (60 + 10 * torch.randn(B, L_MAX, generator=g)).clamp(5, 100)], -1) * mask
    y_mask = (torch.rand(B, H, generator=g) < 0.7).float()
    y_mask[:, 0] = 1
    return {
        "lat": torch.rand(B, generator=g) * 120 - 60,
        "lon": torch.rand(B, generator=g) * 360 - 180,
        "elev": torch.rand(B, generator=g) * 500,
        "x_hist": x, "mask_hist": mask,
        "doy_hist": torch.rand(B, L_MAX, generator=g) * 365,
        "hour_hist": torch.rand(B, L_MAX, generator=g) * 24,
        "doy_fut": torch.rand(B, H, generator=g) * 365,
        "hour_fut": torch.rand(B, H, generator=g) * 24,
        "y": (12 + 4 * torch.randn(B, H, generator=g)) * y_mask, "y_mask": y_mask,
        "norm_scale": 0.3 + 5 * torch.rand(B, H, generator=g),
    }


def _shared_q(B=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    mu = 10 + 5 * torch.randn(B, H, generator=g)
    offs = torch.cat([torch.zeros(B, H, 1), torch.rand(B, H, 6, generator=g).cumsum(-1)], -1)
    return mu[..., None] + offs - offs[..., 3:4]


def _lit(arch):
    from mayak.lit import LitForecaster
    torch.manual_seed(0)
    return LitForecaster(arch=arch).eval()


def test_loss_identical_for_all_models_on_identical_inputs():
    """Одинаковые q, y, маска и масштаб → одно и то же значение и один и тот же градиент
    по q у всех архитектур; значение совпадает с ручным расчётом."""
    batch = _toy_batch()
    q0 = _shared_q()
    vals, grads = [], []
    for arch in ARCH_NAMES:
        lit = _lit(arch)
        with torch.no_grad():
            out = lit.model(batch)
        q = q0.clone().requires_grad_(True)
        out = dict(out, q=q)
        loss = forecast_loss(out, batch)
        (gq,) = torch.autograd.grad(loss, q)
        vals.append(loss.detach())
        grads.append(gq)
    assert all(torch.equal(vals[0], v) for v in vals[1:])
    assert all(torch.equal(grads[0], gr) for gr in grads[1:])

    taus = torch.tensor(QUANTILES)
    sc = batch["norm_scale"].clamp(*NORM_SCALE_CLAMP)
    err = (batch["y"][..., None] - q0) / sc[..., None]
    per = torch.maximum(taus * err, (taus - 1) * err).mean(-1)
    w = batch["y_mask"]
    assert torch.allclose(vals[0], (per * w).sum() / w.sum(), rtol=1e-6)


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_loss_ignores_model_own_scale(arch):
    """Ни одна модель не нормирует функцию потерь на свой выход."""
    batch = _toy_batch(seed=1)
    lit = _lit(arch)
    with torch.no_grad():
        out = lit.model(batch)
    ref = forecast_loss(out, batch)
    own = [k for k in ("sigma_c", "sigma") if k in out]
    assert own, "у модели должен быть собственный разброс — иначе проверка пуста"
    for k in own:
        for f in (1e-3, 1e3):
            assert torch.equal(forecast_loss(dict(out, **{k: out[k] * f}), batch), ref)
    changed = dict(batch, norm_scale=batch["norm_scale"] * 2 + 1)
    assert not torch.equal(forecast_loss(out, changed), ref)


def test_loss_requires_norm_scale_in_batch():
    batch = _toy_batch()
    del batch["norm_scale"]
    with pytest.raises(KeyError, match="norm_scale"):
        forecast_loss({"q": _shared_q()}, batch)


def test_norm_scale_clamped_by_shared_constants():
    batch = _toy_batch()
    q = _shared_q()
    lo, hi = NORM_SCALE_CLAMP
    a = forecast_loss({"q": q}, dict(batch, norm_scale=torch.full((3, H), lo / 10)))
    b = forecast_loss({"q": q}, dict(batch, norm_scale=torch.full((3, H), lo)))
    c = forecast_loss({"q": q}, dict(batch, norm_scale=torch.full((3, H), hi * 10)))
    d = forecast_loss({"q": q}, dict(batch, norm_scale=torch.full((3, H), hi)))
    assert torch.equal(a, b) and torch.equal(c, d)
    assert torch.equal(b, pinball(q, batch["y"], batch["y_mask"], torch.full((3, H), lo)))


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_regularizers_do_not_depend_on_target(arch):
    batch = _toy_batch(seed=2)
    lit = _lit(arch)
    other = dict(batch, y=batch["y"] + 7.0, y_mask=torch.ones_like(batch["y_mask"]),
                 norm_scale=batch["norm_scale"] + 1)
    with torch.no_grad():
        _, d0, r0 = lit.losses(batch)
        _, d1, r1 = lit.losses(other)
    assert torch.equal(r0, r1)
    assert not torch.equal(d0, d1)
    assert torch.isfinite(r0) and r0.ndim == 0


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_training_objective_is_common_loss_plus_regularizer(arch):
    from mayak.lit import regularization
    batch = _toy_batch(seed=3)
    lit = _lit(arch)
    with torch.no_grad():
        out, data, reg = lit.losses(batch)
    assert torch.equal(data, forecast_loss(out, batch))
    assert torch.equal(reg, regularization(lit.model, out))
    if arch != "mayak":
        assert reg.item() == 0.0


def test_protocol_roundtrip_through_json():
    p = DEFAULT_PROTOCOL.deviate("причина", lr=1e-3)
    d = json.loads(json.dumps(p.to_dict()))
    assert Protocol.from_dict(d) == p
    assert Protocol.from_dict(d).common() == DEFAULT_PROTOCOL


def test_deviation_requires_reason_and_known_field():
    with pytest.raises(ProtocolError, match="причин"):
        DEFAULT_PROTOCOL.deviate("", lr=1e-3)
    with pytest.raises(ProtocolError, match="неизвестные"):
        DEFAULT_PROTOCOL.deviate("причина", arch="gru")
    p = DEFAULT_PROTOCOL.deviate("не сходится", lr=1e-3, patience=9)
    assert p.lr == 1e-3 and p.patience == 9
    assert [(d.field, d.value, d.default) for d in p.deviations] == \
        [("lr", 1e-3, DEFAULT_PROTOCOL.lr), ("patience", 9, DEFAULT_PROTOCOL.patience)]


def test_protocol_rejects_invalid_values():
    with pytest.raises(ValueError):
        Protocol(monitor="train/loss")
    with pytest.raises(ValueError):
        Protocol(lr_schedule="step")
    with pytest.raises(ValueError):
        Protocol(stages=(Stage("A", "L0", 1, 0), Stage("A", "full", 1, L_MAX)))


def test_every_architecture_gets_the_same_protocol():
    ps = [protocol_for(a) for a in ARCH_NAMES]
    assert all(p == ps[0] for p in ps[1:])
    assert all(p.common() == p for p in ps), "объявлены отклонения — обновите тест и описания"
    with pytest.raises(ProtocolError):
        protocol_for("transformer")


def test_registry_matches_protocol_names():
    from mayak.lit import ARCHS
    assert tuple(ARCHS) == ARCH_NAMES


def test_undocumented_deviation_is_rejected(monkeypatch):
    from mayak import protocol as P
    from mayak.baselines import GRUSeq2Seq
    monkeypatch.setitem(P.ARCH_DEVIATIONS, "gru", (({"patience": 9}, "долго выходит на плато"),))
    p = protocol_for("gru")
    assert p != protocol_for("dlinear") and p.common() == protocol_for("dlinear")
    with pytest.raises(ProtocolError, match="patience"):
        check_deviations_documented(GRUSeq2Seq, p)
    monkeypatch.setattr(GRUSeq2Seq, "__doc__", "Отклонение: patience = 9 (плато).")
    check_deviations_documented(GRUSeq2Seq, p)


def test_optimizer_and_schedule_identical_across_architectures():
    seen = []
    for arch in ARCH_NAMES:
        lit = _lit(arch)
        cfg = lit.configure_optimizers()
        opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        n_all = sum(p.numel() for p in lit.model.parameters() if p.requires_grad)
        assert n_opt == n_all, f"{arch}: не все параметры попали в оптимизатор"
        seen.append((type(opt), {g["lr"] for g in opt.param_groups},
                     {tuple(g["betas"]) for g in opt.param_groups}, type(sched), sched.T_max,
                     cfg["lr_scheduler"]["interval"], lit.protocol.ema_decay))
    assert all(s == seen[0] for s in seen[1:]), seen


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launch_config_differs_only_by_arch():
    from mayak.protocol import protocol_from_args
    tr = _load_script("train")
    common = ["--steps-a", "7", "--steps-b", "11", "--batch", "3", "--seed", "5"]
    parsed = [tr.make_parser().parse_args(["--arch", a, *common]) for a in ARCH_NAMES]
    protos = [protocol_for(a, protocol_from_args(ns)) for a, ns in zip(ARCH_NAMES, parsed)]
    assert all(p == protos[0] for p in protos[1:])
    assert [s.steps for s in protos[0].stages] == [7, 11] and protos[0].seed == 5
    diff = [set(k for k in vars(parsed[0]) if vars(parsed[0])[k] != vars(ns)[k]) for ns in parsed]
    assert all(d <= {"arch"} for d in diff), diff


def test_protocol_module_never_mentions_test_window():
    src = (REPO / "mayak" / "protocol.py").read_text(encoding="utf-8")
    assert not re.findall(r"\btest\b|unseen_test|ROLE_TEST", src)


TINY = Protocol(stages=(Stage("A", "L0", 2, 0), Stage("B", "full", 2, L_MAX)),
                batch_size=2, windows_per_epoch=8, num_workers=0, seed=3, precision="32",
                val_every=1, val_batches=1)


class _BatchRecorder:
    """Колбэк: хеш каждого обучающего батча по порядку."""

    def __new__(cls):
        from pytorch_lightning.callbacks import Callback

        class Rec(Callback):
            def __init__(self):
                self.hashes = []

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                h = hashlib.sha256()
                for k in ("x_hist", "mask_hist", "y", "y_mask", "norm_scale", "lat", "doy_fut"):
                    h.update(batch[k].detach().cpu().numpy().tobytes())
                self.hashes.append(h.hexdigest())

        return Rec()


@pytest.fixture(scope="module")
def runs(manifest, store, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs4")
    res = {}
    for arch in ARCH_NAMES:
        rec = _BatchRecorder()
        journal = run_protocol(arch, manifest, TINY, out_root=str(out), accelerator="cpu",
                               callbacks=[rec], enable_progress_bar=False)
        res[arch] = dict(journal=journal, hashes=rec.hashes)
    return out, res


def test_all_architectures_see_identical_window_stream(runs):
    _, res = runs
    ref = res[ARCH_NAMES[0]]["hashes"]
    assert len(ref) == sum(s.steps for s in TINY.stages)
    for arch in ARCH_NAMES[1:]:
        assert res[arch]["hashes"] == ref, f"{arch}: другой поток окон"


def test_journals_differ_only_by_architecture(runs):
    out, res = runs
    for arch in ARCH_NAMES:
        j = read_journal(out / arch)
        assert j == res[arch]["journal"]
        assert j["arch"] == arch and j["deviations"] == []
        assert Protocol.from_dict(j["protocol"]) == TINY
        assert [s["name"] for s in j["stages"]] == ["A", "B"]
        assert j["final_ckpt"] == j["stages"][-1]["best_ckpt"] and Path(j["final_ckpt"]).exists()
        assert j["param_groups"] and all(g["n_params"] >= 0 for g in j["param_groups"])
    keys = {k for j in (r["journal"] for r in res.values()) for k in j}
    varying = {k for k in keys
               if len({json.dumps(r["journal"].get(k), sort_keys=True) for r in res.values()}) > 1}
    assert varying <= {"arch", "model_class", "stages", "final_ckpt", "param_groups", "n_params"}


def test_checkpoints_carry_protocol_and_pass_checklist(runs, store):
    from mayak.leakage import run_checklist
    from mayak.lit import LitForecaster
    _, res = runs
    ckpts = []
    for arch in ARCH_NAMES:
        for st in res[arch]["journal"]["stages"]:
            ck = torch.load(st["best_ckpt"], map_location="cpu", weights_only=False)
            hp = ck["hyper_parameters"]
            assert hp["arch"] == arch and hp["stage"] == st["name"]
            assert Protocol.from_dict(hp["protocol"]) == TINY
            ckpts.append(st["best_ckpt"])
        lit = LitForecaster.load_from_checkpoint(res[arch]["journal"]["final_ckpt"],
                                                 map_location="cpu")
        assert type(lit.model).__name__ == type(_lit(arch).model).__name__
    run_checklist(store, checkpoints=ckpts)


def test_stage_b_starts_from_stage_a_weights(runs):
    _, res = runs
    j = res["dlinear"]["journal"]
    a = torch.load(j["stages"][0]["best_ckpt"], map_location="cpu", weights_only=False)
    from mayak.lit import LitForecaster
    lit = LitForecaster(arch="dlinear", protocol=TINY, stage="B", total_steps=2)
    lit.load_state_dict(a["state_dict"])
    for k, v in a["state_dict"].items():
        assert torch.equal(lit.state_dict()[k], v)


def test_declared_deviation_is_journaled(manifest, store, tmp_path, monkeypatch):
    from mayak import protocol as P
    from mayak.baselines import DLinear
    monkeypatch.setitem(P.ARCH_DEVIATIONS, "dlinear",
                        (({"patience": 9}, "долго выходит на плато"),))
    monkeypatch.setattr(DLinear, "__doc__", "Отклонение от протокола: patience = 9.")
    j = run_protocol("dlinear", manifest, TINY, out_root=str(tmp_path), accelerator="cpu",
                     enable_progress_bar=False)
    assert [(d["field"], d["value"], d["reason"]) for d in j["deviations"]] == \
        [("patience", 9, "долго выходит на плато")]
    p = Protocol.from_dict(j["protocol"])
    assert p.patience == 9 and p.common() == TINY
    hp = torch.load(j["final_ckpt"], map_location="cpu", weights_only=False)["hyper_parameters"]
    assert Protocol.from_dict(hp["protocol"]).deviations == p.deviations
    assert read_journal(tmp_path / "dlinear")["deviations"] == j["deviations"]
