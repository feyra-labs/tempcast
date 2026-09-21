"""Тесты: конфигурация, чекпойнт, флаги абляций, раздельные сиды."""
import csv
import dataclasses
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.config import (ABLATION_NAMES, Ablations, AugmentConfig, ConfigError, DataConfig,
                          DLinearConfig, GRUConfig, ModeGroup, ModelConfig, RunConfig,
                          check_pipeline_compat, model_config_for)
from mayak.constants import H, L_MAX, QUANTILES
from mayak.data import store as S
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from mayak.model import MAYAK
from mayak.protocol import DEFAULT_PROTOCOL, Protocol, Seeds, Stage

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "conf"
MODEL_ABLATIONS = [a for a in ABLATION_NAMES if a != "no_offset_aug"]

def _batch(B=2, L=L_MAX, horizon=H, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, L_MAX, 3)
    m = torch.zeros(B, L_MAX, 3)
    if L > 0:
        x[:, -L:, 0] = 12 + 4 * torch.randn(B, L, generator=g)
        x[:, -L:, 1] = 1005 + 3 * torch.randn(B, L, generator=g)
        x[:, -L:, 2] = (65 + 10 * torch.randn(B, L, generator=g)).clamp(5, 100)
        m[:, -L:] = (torch.rand(B, L, 3, generator=g) < 0.9).float()
        x = x * m
    return {
        "lat": torch.tensor([52.0, -33.0][:B]), "lon": torch.tensor([4.9, 151.0][:B]),
        "elev": torch.tensor([0.0, 50.0][:B]),
        "x_hist": x, "mask_hist": m,
        "doy_hist": (torch.arange(L_MAX) / 24.0 + 100.0).expand(B, L_MAX).clone(),
        "hour_hist": (torch.arange(L_MAX) % 24).float().expand(B, L_MAX).clone(),
        "doy_fut": (torch.arange(horizon) / 24.0 + 128.0).expand(B, horizon).clone(),
        "hour_fut": (torch.arange(horizon) % 24).float().expand(B, horizon).clone(),
        "y": 12 + torch.randn(B, horizon, generator=g), "y_mask": torch.ones(B, horizon),
        "norm_scale": torch.full((B, horizon), 3.0),
    }


def _model(cfg=None, seed=0):
    torch.manual_seed(seed)
    return MAYAK(cfg).eval()


def _write_station(root, sid, seed, n=12_000):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data6")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate([("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL),
                                     ("v1", ROLE_VAL), ("x0", ROLE_TEST)]):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb", split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    S.get_store(str(path))
    return str(path)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compose(overrides=(), hydra_cfg=False):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose("config", overrides=list(overrides), return_hydra_config=hydra_cfg)


def test_default_config_reproduces_pre_block6_architecture():
    """Значения по умолчанию = прежние константы: те же веса и формы (чекпойнты читаются)."""
    m = _model()
    assert sum(p.numel() for p in m.parameters()) == 52194
    sd = m.state_dict()
    assert len(sd) == 107
    assert sd["readout.raw_tau"].shape == (24,) and sd["propagator.w_re"].shape == (24,)
    assert sd["encoder.stem.weight"].shape == (48, 13, 1)
    assert sd["heads.fc1.weight"].shape == (48, 16)
    assert sd["passport.obs.weight"].shape == (32, 32)
    from mayak.baselines import DLinear, GRUSeq2Seq
    assert sum(p.numel() for p in GRUSeq2Seq().parameters()) == 190944
    assert sum(p.numel() for p in DLinear().parameters()) == 227304


def test_derived_dimensions_are_computed():
    c = ModelConfig()
    assert (c.n_modes, c.group_sizes, c.n_groups) == (24, (8, 6, 4, 6), 4)
    assert c.n_channels == 13 and c.heads_in_dim == 16
    assert c.receptive_field == 2 * sum(c.encoder_dilations) + 1 == 253
    assert c.stream_buffer == 256 >= c.receptive_field
    assert c.history_days == 28
    ns = ModelConfig(ablations=Ablations(no_solar=True))
    assert ns.n_channels == 10 and ns.heads_in_dim == 13
    ng = ModelConfig(ablations=Ablations(no_mode_groups=True))
    assert ng.n_groups == 1 and ng.group_sizes == (24,) and ng.heads_in_dim == 13
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    assert not fields & {"n_modes", "n_channels", "heads_in_dim", "receptive_field", "n_groups"}


def test_modules_do_not_use_architecture_globals():
    import mayak.constants as C
    for name in ("M", "GROUPS", "DZ", "C_ENC", "N_CH"):
        assert not hasattr(C, name), f"mayak.constants.{name} должна была переехать в конфиг"
    for p in (REPO / "mayak").rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"#\s*128\b", src), f"{p}: закомментированная альтернатива"
    enc = (REPO / "mayak" / "modules" / "encoder.py").read_text(encoding="utf-8")
    assert "127" not in enc and "253" in enc


@pytest.mark.parametrize("groups", [
    (ModeGroup("R", (3, 24, 240)), ModeGroup("D", (24, 96), (24,))),
    (ModeGroup("A", (6, 12)), ModeGroup("B", (24,), (24,)), ModeGroup("C", (48,), (12,)),
     ModeGroup("E", (72, 120), (60, 90)), ModeGroup("F", (168,), (120,))),
    (ModeGroup("only", tuple(np.geomspace(3, 240, 10)), (0.0,)),),
])
def test_changing_mode_groups_keeps_shapes(groups):
    cfg = ModelConfig(mode_groups=groups)
    m = _model(cfg).train()
    out = m(_batch())
    M, G = sum(g.size for g in groups), len(groups)
    assert out["a_re"].shape == (2, M) and out["Eg"].shape == (2, H, G)
    assert out["q"].shape == (2, H, len(QUANTILES))
    assert (out["q"].diff(dim=-1) >= 0).all()
    from mayak.loss import mayak_loss
    loss = mayak_loss(out, _batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_model_is_parametric_in_horizon_and_quantiles():
    cfg = ModelConfig(horizon=24, quantiles=(0.1, 0.5, 0.9))
    out = _model(cfg)(_batch(horizon=24))
    assert out["q"].shape == (2, 24, 3) and (out["q"].diff(dim=-1) >= 0).all()
    with pytest.raises(ConfigError, match="контракт"):
        check_pipeline_compat(cfg)
    check_pipeline_compat(ModelConfig())


@pytest.mark.parametrize("bad, match", [
    (dict(quantiles=(0.1, 0.9)), "медиан"),
    (dict(quantiles=(0.5, 0.25, 0.75)), "возрастать"),
    (dict(max_history=100), "кратное 24"),
    (dict(encoder_width=50), "не делится"),
    (dict(mode_groups=(ModeGroup("R", (1000.0,)),)), "вне"),
    (dict(mode_groups=(ModeGroup("R", (3.0,)), ModeGroup("R", (6.0,)))), "повторяются"),
])
def test_invalid_model_config_fails_loudly(bad, match):
    with pytest.raises(ConfigError, match=match):
        ModelConfig(**bad)


def test_unknown_keys_are_rejected_everywhere():
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        ModelConfig.from_dict({"encoder_widht": 64})
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        ModelConfig.from_dict({"ablations": {"no_anchr": True}})
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        DataConfig.from_dict({"augment": {"ofset_max": 0.0}})
    with pytest.raises(ConfigError):
        model_config_for("gru", {"arch": "dlinear"})
    with pytest.raises(ConfigError, match="контрактом сплитов"):
        DataConfig(time_bounds={**DataConfig().time_bounds, "val_days": 30})


def test_configs_roundtrip_through_json():
    rc = RunConfig(model=ModelConfig(encoder_width=16, passport_dim=8,
                                     ablations=Ablations(no_passport=True)),
                   data=DataConfig(val_every_hours=48),
                   train=DEFAULT_PROTOCOL.__class__(seeds=Seeds(augment=11)))
    d = json.loads(json.dumps(rc.to_dict()))
    assert RunConfig.from_dict(d) == rc
    for cfg in (GRUConfig(hidden=8), DLinearConfig(kernel=5)):
        assert model_config_for(cfg.arch, json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_dataclasses_do_not_import_hydra():
    src = (REPO / "mayak" / "config.py").read_text(encoding="utf-8")
    assert "hydra" not in re.sub(r'""".*?"""|#.*', "", src, flags=re.S).lower()


@pytest.mark.parametrize("group, name, cls", [
    ("model", "mayak", ModelConfig), ("model", "gru", GRUConfig),
    ("model", "dlinear", DLinearConfig), ("data", "default", DataConfig),
])
def test_yaml_defaults_match_dataclasses(group, name, cls):
    """YAML в conf/ полон (все поля) и совпадает с датаклассом - дрейфа нет."""
    import yaml
    d = yaml.safe_load((CONF / group / f"{name}.yaml").read_text(encoding="utf-8"))
    assert set(d) == {f.name for f in dataclasses.fields(cls)}
    assert cls.from_dict(d) == cls()


def test_yaml_train_matches_default_protocol():
    import yaml
    d = yaml.safe_load((CONF / "train" / "default.yaml").read_text(encoding="utf-8"))
    assert set(d) == {f.name for f in dataclasses.fields(Protocol)}
    assert Protocol.from_dict(d) == DEFAULT_PROTOCOL


def test_every_ablation_has_a_config_group_option():
    assert {p.stem for p in (CONF / "ablation").glob("*.yaml")} == {"none", *ABLATION_NAMES}


def test_hydra_composition_to_run_config():
    run = _load_script("run")
    assert _compose(hydra_cfg=True).hydra.job.chdir is False
    rc = run.to_run_config(_compose())
    assert rc == RunConfig()
    rc = run.to_run_config(_compose(["model=mayak_wide", "ablation=no_anchor",
                                     "model.encoder_width=64", "train.seed=3"]))
    assert rc.model.field_hidden == 128 and rc.model.encoder_width == 64
    assert rc.model.ablations.active() == ("no_anchor",) and rc.train.seed == 3
    rc = run.to_run_config(_compose(["model=gru"]))
    assert rc.arch == "gru" and rc.model == GRUConfig()
    with pytest.raises(ConfigError, match="ablations"):
        run.to_run_config(_compose(["model=gru", "ablation=no_anchor"]))


def test_hydra_run_tag_distinguishes_sweep_jobs():
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    tags = set()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        for ov in (["ablation=none"], ["ablation=no_anchor"], ["train.seed=1"], ["model=gru"]):
            cfg = compose("config", overrides=ov, return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            tags.add(cfg.run.tag)
    assert tags == {"mayak-none-s0", "mayak-no_anchor-s0", "mayak-none-s1", "gru-none-s0"}


def _trained_like(m, seed=1):
    """Сбить инициализацию: у свежей модели веса выхода голов нулевые, и часть входов
    (паспорт, энергии групп) на выход ещё не влияет."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return m


ABLATION_DEFINED = ("readout.omega0", "readout.raw_tau")


@pytest.mark.parametrize("flag", MODEL_ABLATIONS)
def test_each_ablation_changes_output_and_keeps_shapes(flag):
    base = _trained_like(_model())
    abl = _model(ModelConfig(ablations=Ablations(**{flag: True})))
    shared = {k: v for k, v in base.state_dict().items()
              if k in abl.state_dict() and abl.state_dict()[k].shape == v.shape
              and k not in ABLATION_DEFINED}
    abl.load_state_dict(shared, strict=False)
    b = _batch()
    with torch.no_grad():
        o0, o1 = base(b), abl(b)
    for k in ("q", "mu", "sigma_c", "o", "r", "a_re", "e"):
        assert o1[k].shape == o0[k].shape, (flag, k)
    assert o1["Eg"].shape[:2] == o0["Eg"].shape[:2]
    assert (o1["q"].diff(dim=-1) >= 0).all()
    assert not torch.allclose(o0["q"], o1["q"], atol=1e-5), f"{flag} не меняет выход"


@pytest.mark.parametrize("flag", MODEL_ABLATIONS)
def test_each_ablation_trains(flag):
    from mayak.loss import mayak_loss
    m = _model(ModelConfig(ablations=Ablations(**{flag: True}))).train()
    loss = mayak_loss(m(_batch()), _batch())
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ablation_semantics():
    b = _batch()
    na = _model(ModelConfig(ablations=Ablations(no_anchor=True)))
    with torch.no_grad():
        out = na(b)
    assert torch.allclose(out["sigma_c"], out["sigma_c"][:1, :1].expand_as(out["sigma_c"]))
    assert not hasattr(na.field, "fc1")
    npz = _model(ModelConfig(ablations=Ablations(no_passport=True)))
    with torch.no_grad():
        out = npz(b)
    assert (out["z"] == 0).all() and float(out["kl"]) == 0.0
    assert not list(npz.passport.parameters())
    nc = _model(ModelConfig(ablations=Ablations(no_compression=True)))
    with torch.no_grad():
        out = nc(_batch(L=0))
    assert torch.isfinite(out["q"]).all() and out["a_re"].abs().max() == 0
    ns = _model(ModelConfig(ablations=Ablations(no_solar=True)))
    assert ns.encoder.stem.in_channels == 10
    assert not {"sin_d", "cos_d", "czp"} & set(ns.cfg.channel_names)


def test_ablation_is_bound_to_parameter_groups():
    """Константный якорь не попадает под затухание поля (иначе уровень тянется к 0 °C)."""
    m = _model(ModelConfig(ablations=Ablations(no_anchor=True)))
    groups = {g["name"]: {id(p) for p in g["params"]} for g in m.optim_groups(1e-2)}
    assert {id(p) for p in m.field.parameters()} <= groups["no_decay"]


def test_no_offset_aug_moves_to_data_without_shifting_other_augmentations(manifest):
    from mayak.data.dataset import WindowDataset
    legacy = DataConfig(augment=AugmentConfig.from_profile("base"))
    rc = RunConfig(model=ModelConfig(ablations=Ablations(no_offset_aug=True)),
                   data=legacy).resolved()
    assert rc.data.augment.offset_max == 0.0 and rc.resolved() == rc
    assert RunConfig().resolved().data.augment.offset_max == AugmentConfig().offset_max
    base = WindowDataset(manifest, windows_per_epoch=16, seed=0, augment=legacy.augment)
    off = WindowDataset(manifest, windows_per_epoch=16, seed=0, augment=rc.data.augment)
    n_diff_y = 0
    for i in range(16):
        a, b = base[i], off[i]
        for k in ("mask_hist", "y_mask", "doy_fut", "lat", "lon", "norm_scale"):
            assert torch.equal(a[k], b[k]), k
        assert torch.equal(a["x_hist"][:, 1:], b["x_hist"][:, 1:])
        d = (a["y"] - b["y"])[a["y_mask"] > 0]
        if d.numel() and d.abs().max() > 0:
            assert torch.allclose(d, d[:1].expand_as(d), atol=1e-4)
            n_diff_y += 1
    assert n_diff_y > 0


def test_knockout_no_compression_is_effective():
    """Раньше knockout ставил несуществующий флаг и ничего не выключал."""
    from mayak.knockout import knockout, variants_for
    m = _model()
    b = _batch()
    with torch.no_grad():
        ref = m(b)["q"]
        with knockout(m, "no_compression"):
            ko = m(b)["q"]
        again = m(b)["q"]
    assert not torch.allclose(ref, ko) and torch.equal(ref, again)
    assert {"no_R", "no_D", "no_S", "no_W"} <= set(variants_for(m))
    one = _model(ModelConfig(ablations=Ablations(no_mode_groups=True)))
    assert not [v for v in variants_for(one) if v.startswith("no_") and v[3:] in "RDSW"]


@pytest.mark.parametrize("flag", [None, *MODEL_ABLATIONS])
def test_runtime_follows_model_config(flag):
    """Рантайм берёт размеры и абляции из модели: прогрев по окну = пакетный прогноз."""
    from mayak.runtime.streaming import StreamingMayak
    cfg = ModelConfig(ablations=Ablations(**({flag: True} if flag else {})))
    m = _model(cfg)
    b = {k: (v[:1] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in _batch().items()}
    with torch.no_grad():
        ref = m(b)
    st = StreamingMayak(m, float(b["lat"]), float(b["lon"]), float(b["elev"]))
    st.warm_start(b["x_hist"][0].numpy(), b["mask_hist"][0].numpy(),
                  b["doy_hist"][0].numpy(), b["hour_hist"][0].numpy())
    q, mu = st.forecast(b["doy_fut"][0].numpy(), b["hour_fut"][0].numpy())
    assert q.shape == (H, len(QUANTILES))
    np.testing.assert_allclose(q, ref["q"][0].numpy(), atol=1e-4)
    raw = st.serialize()
    st2 = StreamingMayak(m, float(b["lat"]), float(b["lon"]), float(b["elev"]))
    st2.load_state(raw)
    assert st2.serialize() == raw


def test_runtime_state_size_follows_config_and_mismatch_fails():
    from mayak.runtime.streaming import StreamingMayak
    small = _model(ModelConfig(mode_groups=(ModeGroup("R", (3, 24)),), passport_dim=4))
    big = _model()
    s_small = StreamingMayak(small, 50.0, 5.0, 0.0).serialize()
    s_big = StreamingMayak(big, 50.0, 5.0, 0.0).serialize()
    assert len(s_big) - len(s_small) == 3 * 4 * (24 - 2) + 4 * (16 - 4)
    assert len(s_big) == StreamingMayak(big, 50.0, 5.0, 0.0).state_nbytes
    with pytest.raises(ValueError, match="конфигу модели"):
        StreamingMayak(big, 50.0, 5.0, 0.0).load_state(s_small)


def test_seeds_resolve_to_base_and_override_separately():
    p = Protocol(seed=5, seeds=Seeds(augment=9))
    assert p.resolved_seeds() == dict(init=5, data=5, augment=9, eval=5)
    assert Protocol.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    tr = _load_script("train")
    from mayak.protocol import protocol_from_args
    ns = tr.make_parser().parse_args(["--seed", "4", "--seed-init", "7", "--seed-eval", "1"])
    assert protocol_from_args(ns).resolved_seeds() == dict(init=7, data=4, augment=4, eval=1)


def test_augment_seed_changes_augmentation_but_not_window_stream(manifest):
    from mayak.data.dataset import WindowDataset
    a = WindowDataset(manifest, windows_per_epoch=12, seed=0)
    b = WindowDataset(manifest, windows_per_epoch=12, seed=0, aug_seed=0)
    c = WindowDataset(manifest, windows_per_epoch=12, seed=0, aug_seed=123)
    ia, ib, ic = ([ds[i] for i in range(12)] for ds in (a, b, c))
    assert all(torch.equal(u["x_hist"], w["x_hist"]) for u, w in zip(ia, ib)), \
        "aug_seed = seed обязан воспроизводить прежний поток"
    for u, w in zip(ia, ic):
        assert torch.equal(u["doy_fut"], w["doy_fut"]) and torch.equal(u["y_mask"], w["y_mask"])
    assert not all(torch.equal(u["x_hist"], w["x_hist"]) for u, w in zip(ia, ic))


TINY = Protocol(stages=(Stage("A", "L0", 2, 0), Stage("B", "full", 2, L_MAX)),
                batch_size=2, windows_per_epoch=8, num_workers=0, seed=3, precision="32",
                val_every=1, val_batches=1, seeds=Seeds(augment=21, eval=4))

CUSTOM = dict(encoder_width=16, encoder_dilations=(1, 2, 4), passport_dim=8, field_hidden=24,
              heads_hidden=16, loc_freqs=8,
              mode_groups=(ModeGroup("R", (3, 48)), ModeGroup("D", (24,), (24,)),
                           ModeGroup("W", (96, 168), (84, 120))),
              ablations=Ablations(no_solar=True))


@pytest.fixture(scope="module")
def custom_run(manifest, tmp_path_factory):
    """Прогон через слой Hydra с конфигом модели, отличным от значений по умолчанию."""
    from mayak.protocol import run_experiment
    run = _load_script("run")
    cfg = _compose(["ablation=no_solar", "model.encoder_width=16",
                    "model.encoder_dilations=[1,2,4]", "model.passport_dim=8",
                    "model.field_hidden=24", "model.heads_hidden=16", "model.loc_freqs=8",
                    "model.mode_groups=[{name:R,tau0:[3,48]},{name:D,tau0:[24],period:[24]},"
                    "{name:W,tau0:[96,168],period:[84,120]}]",
                    f"data.manifest={manifest}"])
    rc = dataclasses.replace(run.to_run_config(cfg), train=TINY)
    assert rc.model == ModelConfig(**CUSTOM)
    out = tmp_path_factory.mktemp("runs6")
    journal = run_experiment(rc, out_root=str(out), tag="custom", accelerator="cpu",
                             enable_progress_bar=False)
    return out / "custom", journal, rc


def test_checkpoint_restores_non_default_architecture(custom_run):
    from mayak.lit import build_model, load_model
    _, journal, rc = custom_run
    m = load_model(journal["final_ckpt"])
    assert m.cfg == rc.model and m.cfg != ModelConfig()
    assert m.encoder.stem.in_channels == 10 and m.readout.n_modes == 5
    out = m(_batch())
    assert out["q"].shape == (2, H, len(QUANTILES)) and out["Eg"].shape == (2, H, 3)
    sd = torch.load(journal["final_ckpt"], map_location="cpu", weights_only=False)["state_dict"]
    with pytest.raises(RuntimeError):
        build_model("mayak").load_state_dict({k[len("model."):]: v for k, v in sd.items()})


def test_checkpoint_carries_resolved_config_seeds_and_provenance(custom_run):
    from mayak.lit import RUN_KEY, load_run_record
    run_dir, journal, rc = custom_run
    for st in journal["stages"]:
        ck = torch.load(st["best_ckpt"], map_location="cpu", weights_only=False)
        rec = ck[RUN_KEY]
        assert RunConfig.from_dict(rec["config"]) == rc.resolved()
        assert rec["seeds"] == dict(init=3, data=3, augment=21, eval=4)
        assert set(rec["provenance"]) == {"git", "versions", "created_utc"}
        assert {"commit", "dirty"} == set(rec["provenance"]["git"])
        v = rec["provenance"]["versions"]
        assert v["torch"].split("+")[0] == torch.__version__.split("+")[0]
        assert v["python"] and v["numpy"] and "pytorch-lightning" in v
        hp = ck["hyper_parameters"]
        assert hp["model_config"] == rc.model.to_dict() and hp["data_config"] == rc.data.to_dict()
        side = json.loads((Path(st["best_ckpt"]).parent / "config.json").read_text())
        assert side == rec["config"]
    assert json.loads((run_dir / "config.json").read_text()) == rc.resolved().to_dict()
    assert journal["seeds"] == dict(init=3, data=3, augment=21, eval=4)
    assert journal["config_file"] == "config.json"
    assert load_run_record(journal["final_ckpt"])["config"]["model"]["ablations"]["no_solar"]


def test_resolved_config_in_checkpoint_carries_ablation_data_effect(manifest, tmp_path):
    from mayak.lit import RUN_KEY
    from mayak.protocol import run_protocol
    proto = Protocol(stages=(Stage("A", "L0", 1, 0),), batch_size=2, windows_per_epoch=4,
                     num_workers=0, precision="32", val_every=1, val_batches=1)
    j = run_protocol("mayak", manifest, proto, out_root=str(tmp_path), accelerator="cpu",
                     model_config=ModelConfig(ablations=Ablations(no_offset_aug=True)),
                     enable_progress_bar=False)
    rec = torch.load(j["final_ckpt"], map_location="cpu", weights_only=False)[RUN_KEY]
    assert rec["config"]["data"]["augment"]["offset_max"] == 0.0


def test_old_checkpoint_without_config_loads_with_defaults(tmp_path):
    from mayak.lit import LitForecaster, load_model
    lit = LitForecaster(arch="mayak")
    hp = {k: v for k, v in dict(lit.hparams).items() if k not in ("model_config", "data_config")}
    path = tmp_path / "old.ckpt"
    torch.save({"state_dict": lit.state_dict(), "hyper_parameters": hp,
                "pytorch-lightning_version": "2.1.0"}, path)
    m = load_model(str(path))
    assert m.cfg == ModelConfig()
    for k, v in lit.model.state_dict().items():
        assert torch.equal(m.state_dict()[k], v)


def test_train_script_ablate_flag_builds_ablated_config(manifest, tmp_path, monkeypatch):
    tr = _load_script("train")
    ap = tr.make_parser()
    ns = ap.parse_args(["--arch", "mayak", "--ablate", "no_anchor", "no_compression"])
    assert ns.ablate == ["no_anchor", "no_compression"]
    with pytest.raises(SystemExit):
        ap.parse_args(["--ablate", "no_such_flag"])
