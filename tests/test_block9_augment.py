"""Тесты: аугментации, имитирующие реальный прибор."""
import csv
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mayak.config import (AUGMENT_PROB_FIELDS, AUGMENT_PROFILES, AugmentConfig, ConfigError,
                          DataConfig, RunConfig)
from mayak.constants import H, L_MAX
from mayak.data import augment as A
from mayak.data import store as S
from mayak.data.augment import (AUG_ORDER, EXPECTED_QC, apply_one, augment_window,
                                clean_history, make_window, qc_effect, reference_windows)
from mayak.data.masking import enforce_invariant
from mayak.data import qc as Q
from mayak.data.recording import is_recorded, record_values
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "conf"
VALUE_AUGS = ("scale", "drift", "offset", "noise", "rh_dewpoint", "spike", "stuck", "units")
INSTRUMENT_ON_TARGET = ("scale", "drift", "offset")
MASK_AUGS = ("dropout", "gap", "outage", "drop_pressure", "drop_humidity")
ALL_ON = {f: 1.0 for f in AUGMENT_PROB_FIELDS.values()}


def _window(seed=0, L=L_MAX, elev=200.0, holes=0.0):
    """Окно из чистой истории; holes - доля исходных пропусков (как после QC источника)."""
    x, hour = clean_history(seed=seed, elev=elev)
    w = make_window(x, hour, L=L, elev=elev,
                    y=12 + np.random.default_rng(seed).standard_normal(H))
    if holes:
        drop = np.random.default_rng(100 + seed).random((L_MAX, 3)) < holes
        w.m[drop] = 0.0
        w.x[drop] = 0.0
        w.y_mask[np.random.default_rng(200 + seed).random(H) < holes] = 0.0
        w.y[w.y_mask == 0] = 0.0
    return w


def _copy(w):
    return dataclasses.replace(w, x=w.x.copy(), m=w.m.copy(), y=w.y.copy(),
                               y_mask=w.y_mask.copy(), hour=w.hour.copy(), applied={})


def test_default_profile_is_aggressive_everywhere():
    assert AugmentConfig() == AugmentConfig.from_profile("aggressive")
    assert DataConfig().augment.profile == "aggressive"
    assert RunConfig().data.augment == AugmentConfig()


def test_profile_yaml_matches_python():
    files = {p.stem for p in (CONF / "augment").glob("*.yaml")}
    assert files == set(AUGMENT_PROFILES)
    for name in AUGMENT_PROFILES:
        d = yaml.safe_load((CONF / "augment" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert d["profile"] == name
        assert AugmentConfig.from_dict(d) == AugmentConfig.from_profile(name), name
        assert AugmentConfig.from_profile(name).deviations() == {}


def test_soft_is_not_stronger_than_aggressive():
    soft, agg = AugmentConfig.from_profile("soft"), AugmentConfig.from_profile("aggressive")
    for f in AUGMENT_PROB_FIELDS.values():
        assert getattr(soft, f) <= getattr(agg, f), f
    for f in ("scale_max", "drift_max", "noise_sd"):
        assert all(a <= b for a, b in zip(getattr(soft, f), getattr(agg, f))), f
    assert soft.offset_max <= agg.offset_max and soft.gap_max_len <= agg.gap_max_len


def test_every_augmentation_has_probability_and_is_off_in_none():
    assert set(AUG_ORDER) == set(AUGMENT_PROB_FIELDS) == set(EXPECTED_QC)
    none = AugmentConfig.from_profile("none")
    assert all(getattr(none, f) == 0.0 for f in AUGMENT_PROB_FIELDS.values())
    for name in AUG_ORDER:
        only = AugmentConfig.only(name)
        on = [n for n, f in AUGMENT_PROB_FIELDS.items() if getattr(only, f) > 0]
        assert on == [name]


def _load_run():
    spec = importlib.util.spec_from_file_location("run", REPO / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compose(overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def test_hydra_augment_group_selects_profile():
    run = _load_run()
    assert run.to_run_config(_compose()).data.augment == AugmentConfig()
    for name in AUGMENT_PROFILES:
        rc = run.to_run_config(_compose([f"augment={name}"]))
        assert rc.data.augment == AugmentConfig.from_profile(name), name
    rc = run.to_run_config(_compose(["augment=soft", "data.augment.gap_prob=0.9"]))
    assert rc.data.augment.profile == "soft" and rc.data.augment.gap_prob == 0.9
    assert rc.data.augment.deviations() == {"gap_prob": (0.3, 0.9)}


def test_config_validation_and_partial_dicts():
    with pytest.raises(ConfigError, match="профиль"):
        AugmentConfig(profile="brutal")
    with pytest.raises(ConfigError, match="вне"):
        AugmentConfig(spike_prob=1.5)
    with pytest.raises(ConfigError, match="spike_min"):
        AugmentConfig(spike_min=(40.0, 15.0, 40.0), spike_max=(30.0, 40.0, 80.0))
    with pytest.raises(ConfigError, match="lo ≤ hi"):
        AugmentConfig(stuck_hours=(48, 12))
    for removed in ("sparse_prob", "sparse_every", "units_p_frac", "quant_prob"):
        with pytest.raises(ConfigError, match="неизвестные ключи"):
            AugmentConfig.from_dict({removed: 0.5})
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        AugmentConfig.from_dict({"spik_prob": 0.1})
    a = AugmentConfig.from_dict({"profile": "base", "gap_prob": 0.0})
    assert a.gap_prob == 0.0 and a.spike_prob == 0.0 and a.gap_max_len == 24
    assert AugmentConfig(offset_max=0.0).offset_min == 0.0


def test_summary_goes_to_run_journal_form():
    s = AugmentConfig.from_profile("soft", gap_prob=0.9).summary()
    assert s["profile"] == "soft" and s["deviations"] == {"gap_prob": [0.3, 0.9]}
    assert json.loads(json.dumps(s)) == s
    assert AugmentConfig.from_profile("none").summary()["enabled"] == []


@pytest.mark.parametrize("L", [0, 1, 30, 200, L_MAX])
@pytest.mark.parametrize("holes", [0.0, 0.3])
def test_invariant_holds_after_all_augmentations(L, holes):
    cfg = AugmentConfig.from_profile("aggressive", **ALL_ON)
    rng = np.random.default_rng(L)
    for seed in range(6):
        w = augment_window(_window(seed, L=L, holes=holes), cfg, rng)
        x, m = enforce_invariant(w.x, w.m)
        assert np.all(x[m == 0] == 0) and np.all(np.isfinite(x))
        assert set(np.unique(m)) <= {0.0, 1.0}
        assert np.all(m[:L_MAX - L] == 0), "история не вылезает левее своего начала"


@pytest.mark.parametrize("name", VALUE_AUGS)
def test_value_distortions_touch_only_valid_points(name):
    cfg = AugmentConfig.only(name)
    for seed in range(8):
        w0 = _window(seed, holes=0.3)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.array_equal(w.m, w0.m), "искажение значений не трогает маску"
        assert np.all(w.x[w0.m == 0] == 0), "значения в дырах не появляются"


@pytest.mark.parametrize("name", MASK_AUGS)
def test_availability_only_removes_validity(name):
    cfg = AugmentConfig.only(name)
    for seed in range(8):
        w0 = _window(seed, holes=0.1)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.all(w.m <= w0.m) and np.array_equal(w.x, w0.x)
        assert name in w.applied and w.m.sum() < w0.m.sum()


@pytest.mark.parametrize("name", [n for n in AUG_ORDER if n not in INSTRUMENT_ON_TARGET])
def test_target_untouched_except_instrument_properties(name):
    cfg = AugmentConfig.only(name)
    for seed in range(6):
        w0 = _window(seed, holes=0.2)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.array_equal(w.y, w0.y) and np.array_equal(w.y_mask, w0.y_mask)


def test_offset_shifts_history_and_target_consistently():
    cfg = AugmentConfig.only("offset")
    for seed in range(10):
        w0 = _window(seed, holes=0.2)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        b = np.float32(w.applied["offset"]["b"])
        assert cfg.offset_min <= abs(b) <= cfg.offset_max
        tol = A.DITHER_FRAC * 0.5 / np.array([1.0, 10.0, 1.0]) + 1e-4
        assert np.all(np.abs(w.y - w0.y - b * w0.y_mask) <= tol[0])
        assert np.array_equal(w.y[w0.y_mask == 0], w0.y[w0.y_mask == 0])
        assert np.array_equal(w.y_mask, w0.y_mask), "маска цели - только из данных (1.6)"
        mT = w0.m[:, 0] > 0
        assert np.all(np.abs(w.x[mT, 0] - w0.x[mT, 0] - b) <= tol[0])
        for ch in (1, 2):
            assert np.all(np.abs(w.x[:, ch] - w0.x[:, ch]) <= tol[ch]), "непрерывность"
        A.record_window(w)
        w1 = A.record_window(_copy(w0))
        assert np.array_equal(w.x[:, 1:], w1.x[:, 1:]), "после записи P и RH прежние"


def test_offset_needs_history_and_obeys_ablation():
    w = augment_window(_window(0, L=0), AugmentConfig.only("offset"), np.random.default_rng(0))
    assert "offset" not in w.applied, "без истории смещение не выучить - цель не трогаем"
    rc = RunConfig(model={"arch": "mayak", "ablations": {"no_offset_aug": True}}).resolved()
    cfg = dataclasses.replace(rc.data.augment, **ALL_ON)
    for seed in range(5):
        w = augment_window(_window(seed), cfg, np.random.default_rng(seed))
        assert "offset" not in w.applied


def test_window_consumes_exactly_one_number():
    for prof in AUGMENT_PROFILES:
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        augment_window(_window(0), AugmentConfig.from_profile(prof), r1)
        r2.integers(2 ** 63)
        assert r1.bit_generator.state == r2.bit_generator.state, prof


def test_toggling_one_augmentation_does_not_shift_others():
    agg = AugmentConfig.from_profile("aggressive", **ALL_ON)
    variants = [dataclasses.replace(agg, offset_max=0.0),
                dataclasses.replace(agg, spike_prob=0.0),
                dataclasses.replace(agg, gap_max_len=5, noise_sd=(1.0, 1.0, 1.0))]
    for seed in range(10):
        ref = augment_window(_window(seed), agg, np.random.default_rng(seed)).applied
        for v, changed in zip(variants, ("offset", "spike", ("gap", "noise"))):
            got = augment_window(_window(seed), v, np.random.default_rng(seed)).applied
            for name in set(ref) | set(got):
                if name in np.atleast_1d(changed):
                    continue
                assert ref.get(name) == got.get(name), (changed, name)


def test_firing_frequencies_follow_probabilities():
    cfg = AugmentConfig.from_profile("aggressive")
    n, fired = 3000, dict.fromkeys(AUG_ORDER, 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        w = augment_window(_window(i % 4), cfg, rng)
        for name in w.applied:
            fired[name] += 1
    for name in AUG_ORDER:
        p = getattr(cfg, AUGMENT_PROB_FIELDS[name])
        tol = 4 * np.sqrt(p * (1 - p) / n) + 1e-9
        assert abs(fired[name] / n - p) <= tol, (name, fired[name] / n, p)


def test_drift_starts_calibrated_and_reaches_current_offset():
    for walk in (False, True):
        for b in (2.0, -1.5):
            d = A.drift_profile(300, b, walk, seed=1)
            assert d.shape == (300,) and d[0] == 0.0, "в первом часе истории прибор точен"
            assert d[-1] == np.float32(b), "к последнему часу смещение равно текущему"
    lin = A.drift_profile(301, 3.0, False, 0)
    np.testing.assert_allclose(np.diff(lin), 0.01, atol=1e-6)
    mids = [A.drift_profile(501, 2.0, True, s)[250] - 1.0 for s in range(400)]
    assert 0.7 < np.sqrt(np.mean(np.square(mids))) < 1.3, "блуждание в середине около b / 2"
    assert A.drift_profile(1, 0.7, True, 0).tolist() == [np.float32(0.7)]
    assert A.drift_profile(0, 0.7, True, 0).shape == (0,)


@pytest.mark.parametrize("L", [1, 30, L_MAX])
def test_scale_and_drift_change_target_by_expected_amount(L):
    w0 = _window(0, L=L, holes=0.2)
    ok = w0.y_mask > 0
    w = apply_one(_copy(w0), "scale", dict(k=[1.03, 1.0, 1.1]))
    np.testing.assert_array_equal(w.y[ok], w0.y[ok] * np.float32(1.03))
    assert np.array_equal(w.y[~ok], w0.y[~ok]) and np.array_equal(w.y_mask, w0.y_mask)
    for walk in (False, True):
        w = apply_one(_copy(w0), "drift", dict(b=[-1.25, 0.5, 4.0], walk=walk, seed=2))
        np.testing.assert_array_equal(w.y[ok], w0.y[ok] + np.float32(-1.25))
        assert np.array_equal(w.y[~ok], w0.y[~ok]) and np.array_equal(w.y_mask, w0.y_mask)
        last = L_MAX - 1
        for ch, b in enumerate((-1.25, 0.5, 4.0)):
            if w0.m[last, ch] > 0:
                assert w.x[last, ch] - w0.x[last, ch] == pytest.approx(b, abs=1e-4)
    for name, p in (("scale", dict(k=[1.03, 1.0, 1.1])), ("offset", dict(b=2.0)),
                    ("drift", dict(b=[1.0, 0.0, 0.0], walk=False, seed=0))):
        w = apply_one(_copy(w0), name, dict(p, target=False))
        assert np.array_equal(w.y, w0.y), f"{name}: вариант без цели трогает цель"


@pytest.mark.parametrize("name", INSTRUMENT_ON_TARGET)
def test_instrument_properties_never_change_target_mask(name):
    cfg = AugmentConfig.only(name)
    for seed in range(20):
        w0 = _window(seed, L=(0, 5, 200, L_MAX)[seed % 4], holes=0.3)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.array_equal(w.y_mask, w0.y_mask)
        assert np.all(w.y[w0.y_mask == 0] == 0)


def test_rh_from_integer_dewpoint_jumps_by_a_few_percent():
    w0 = _window(0, holes=0.2)
    w = apply_one(_copy(w0), "rh_dewpoint", {})
    both = (w0.m[:, 0] > 0) & (w0.m[:, 2] > 0)
    d = w.x[both, 2] - w0.x[both, 2]
    assert np.abs(d).max() > 1.0 and np.abs(d).max() < 12.0
    assert np.all((w.x[:, 2] >= 0) & (w.x[:, 2] <= 100))
    assert np.array_equal(w.x[~both, 2], w0.x[~both, 2]), "без пары T и RH ничего не меняется"
    assert np.array_equal(w.x[:, :2], w0.x[:, :2]) and np.array_equal(w.m, w0.m)
    again = apply_one(_copy(w), "rh_dewpoint", {})
    assert np.abs(again.x[both, 2] - w.x[both, 2]).max() < 2.0, "повтор почти ничего не меняет"


def test_units_and_outage_semantics():
    """Подмена единиц - только давление на уровне моря: температуру в градусах Цельсия
    обеспечивает владелец прибора."""
    w0 = _window(0, elev=1500.0)
    w = apply_one(_copy(w0), "units", dict(i=100, n=10))
    ratio = w.x[100:110, 1] / w0.x[100:110, 1]
    assert np.allclose(ratio, A.sea_level_ratio(1500.0)) and ratio[0] > 1.15
    assert np.array_equal(w.x[:, [0, 2]], w0.x[:, [0, 2]])
    for seed in range(50):
        w = augment_window(_window(seed), AugmentConfig.only("units"),
                           np.random.default_rng(seed))
        assert np.array_equal(w.x[:, 0], _window(seed).x[:, 0])
    cfg = AugmentConfig.only("outage")
    for seed in range(10):
        w = augment_window(_window(seed), cfg, np.random.default_rng(seed))
        p = w.applied["outage"]
        assert np.all(w.m[p["i"]:p["i"] + p["n"], p["ch"]] == 0)
        assert w.m[-1, p["ch"]] == 1, "канал возвращается до конца истории"
        assert cfg.outage_hours[0] <= p["n"] <= cfg.outage_hours[1]


def test_metadata_jitter_bounds():
    cfg = AugmentConfig.only("coords")
    d_elev = []
    for seed in range(200):
        w0 = _window(seed % 3)
        w0.lon = 179.9
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert abs(w.lat - w0.lat) <= cfg.coord_jitter_deg + 1e-9
        assert -180.0 <= w.lon < 180.0
        assert w.elev - w0.elev == pytest.approx(w.qc_elev - w0.qc_elev)
        d_elev.append(w.elev - w0.elev)
    assert 0.7 * cfg.elev_jitter_m < np.std(d_elev) < 1.3 * cfg.elev_jitter_m


def test_clean_reference_window_is_clean():
    for seed in range(3):
        for elev in (200.0, 1500.0):
            codes = A.window_codes(_window(seed, elev=elev))
            assert not np.any(codes), "эталон должен быть чистым, иначе проверка ничего не значит"


@pytest.mark.parametrize("case", [c[0] for c in A.REFERENCE_CASES])
def test_reference_window_produces_expected_codes(case):
    (_, name, before, after, ch, rows), = [r for r in reference_windows() if r[0] == case]
    eff = qc_effect(name, before, after, ch, rows)
    if EXPECTED_QC[name]:
        assert eff["hit"], f"{case}: нет кода {sorted(c.name for c in EXPECTED_QC[name])}"
        if rows is not None:
            assert eff["hit_frac"] == 1.0, f"{case}: код не на всём затронутом участке"
    else:
        assert eff["side_frac"] < 0.001, f"{case}: QC не должен видеть это искажение: {eff}"
    assert eff["side_frac"] < 0.002, eff


MIN_HIT = {"spike": 0.8, "stuck": 0.7, "units": 0.6, "dropout": 1.0, "gap": 1.0,
           "outage": 1.0, "drop_pressure": 1.0, "drop_humidity": 1.0}


def _detectable(name, params, w):
    """Распознаёт ли причинный QC искажение по прошлому.

    Залипание одного канала видно только после срока залипания этого канала;
    более короткое устройство не отличит от нормы. Давление на уровне моря отличимо от
    станционного только у достаточно высокой станции.
    """
    from mayak.data.qc import DEFAULT_QC as C
    if name == "units":
        sep = Q.P_SEA_LEVEL - Q.station_pressure_expected(w.qc_elev)
        return sep >= C.slp_min_sep
    if name != "stuck":
        return True
    n = min(params["n"], L_MAX - params["i"])
    need = C.stuck_T_alone_hours if params["ch"] == 0 else C.stuck_hours[params["ch"]]
    return n > need


@pytest.mark.parametrize("name", AUG_ORDER)
def test_sampled_augmentation_produces_expected_codes(name):
    cfg = AugmentConfig.only(name, "aggressive")
    hits, sides, n = [], [], 40
    for seed in range(n):
        elev = (200.0, 1500.0)[seed % 2]
        before = _window(seed, elev=elev)
        after = augment_window(_copy(before), cfg, np.random.default_rng(seed))
        assert name in after.applied
        eff = qc_effect(name, before, after)
        if _detectable(name, after.applied[name], before):
            hits.append(eff["hit"])
        sides.append(eff["side_frac"])
    assert len(hits) >= n // 4, f"{name}: мало распознаваемых случаев для проверки"
    if EXPECTED_QC[name]:
        assert np.mean(hits) >= MIN_HIT[name], f"{name}: слишком слабая, {np.mean(hits):.2f}"
    else:
        assert all(h is None for h in hits)
    assert np.mean(sides) < 0.002, f"{name}: побочные коды {np.mean(sides):.4f}"


def test_reference_script_writes_artifacts(tmp_path):
    spec = importlib.util.spec_from_file_location("aug_reference",
                                                  REPO / "scripts" / "aug_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path)]) == 0
    table = json.loads((tmp_path / "aug_reference.json").read_text(encoding="utf-8"))["cases"]
    assert [r["case"] for r in table] == [c[0] for c in A.REFERENCE_CASES]
    z = np.load(tmp_path / "aug_reference.npz")
    for case in ("spike_T", "gap_3d", "rh_dewpoint"):
        for tag in ("before", "after"):
            assert z[f"{case}/{tag}/x"].shape == (L_MAX, 3)
    again = tmp_path / "again"
    mod.main(["--out", str(again)])
    z2 = np.load(again / "aug_reference.npz")
    assert all(np.array_equal(z[k], z2[k]) for k in z.files), "эталон детерминирован"


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("data9")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role, elev) in enumerate([("t0", ROLE_TRAIN, 200.0), ("t1", ROLE_TRAIN, 1500.0),
                                           ("v0", ROLE_VAL, 300.0), ("x0", ROLE_TEST, 100.0)]):
        x, _ = clean_history(n=12_000, lat=45.0, lon=10.0, elev=elev, seed=i)
        np.savez(root / "stations" / f"{sid}.npz", T=x[:, 0], P=x[:, 1], RH=x[:, 2],
                 valid=np.ones((len(x), 3), np.uint8), t0_utc_h=np.int64(0))
        rows.append(dict(id=sid, lat=45.0, lon=10.0, elev=elev, koppen="Cfb", split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    S.get_store(str(path))
    return str(path)


def test_dataset_none_profile_passes_data_through(manifest):
    from mayak.data.dataset import WindowDataset, slice_history, slice_target
    ds = WindowDataset(manifest, windows_per_epoch=4, seed=0, augment={"profile": "none"},
                       window_qc=False)
    s = ds.st[0]
    t = int(s["starts"][5])
    info = {}
    it = ds.build(s, t, L_MAX, info)
    x, m = slice_history(s["x"], s["mask"], t, L_MAX)
    y, ym = slice_target(s["x"], s["mask"], t)
    assert info == {}
    assert np.array_equal(it["x_hist"].numpy(), x) and np.array_equal(it["mask_hist"].numpy(), m)
    assert np.array_equal(it["y"].numpy(), y) and np.array_equal(it["y_mask"].numpy(), ym)
    assert it["lat"].item() == pytest.approx(s["lat"])


def test_dataset_aggressive_keeps_contract(manifest):
    from mayak.data.dataset import WindowDataset, slice_target
    ds = WindowDataset(manifest, windows_per_epoch=4, seed=0)
    assert ds.augment == AugmentConfig()
    seen = set()
    for i, s in enumerate(ds.st * 12):
        t = int(s["starts"][(7 * i) % len(s["starts"])])
        info = {}
        it = ds.build(s, t, (0, 24, 200, L_MAX)[i % 4], info)
        seen |= set(info)
        x, m = it["x_hist"].numpy(), it["mask_hist"].numpy()
        assert np.all(x[m == 0] == 0) and np.all(np.isfinite(x))
        y0, ym0 = slice_target(s["x"], s["mask"], t)
        assert np.array_equal(it["y_mask"].numpy(), ym0)
        y = y0.astype(np.float64)
        if "scale" in info:
            y = np.where(ym0 > 0, y * info["scale"]["k"][0], y)
        if "drift" in info:
            y = y + info["drift"]["b"][0] * ym0
        if "offset" in info:
            y = y + info["offset"]["b"] * ym0
        got = it["y"].numpy()
        assert np.all(np.abs(got - y)[ym0 > 0] <= 1.0 + 1e-4), \
            "цель - запись прибора после его свойств: не дальше полушага шума и полушага записи"
        assert np.array_equal(got[ym0 == 0], y0[ym0 == 0])
        assert is_recorded(it["x_hist"].numpy(), it["mask_hist"].numpy())
        assert np.array_equal(it["y"].numpy(), np.round(it["y"].numpy()))
        for k, shape in (("x_hist", (L_MAX, 3)), ("mask_hist", (L_MAX, 3)), ("y", (H,))):
            assert tuple(it[k].shape) == shape
    assert len(seen) >= 10, f"за 24 окна сработали только {sorted(seen)}"


def test_dataset_window_qc_masks_augmented_artifacts(manifest):
    from mayak.data.dataset import WindowDataset
    for name in ("spike", "stuck", "units"):
        on = WindowDataset(manifest, windows_per_epoch=4, seed=0,
                           augment=AugmentConfig.only(name), window_qc=True)
        off = WindowDataset(manifest, windows_per_epoch=4, seed=0,
                            augment=AugmentConfig.only(name), window_qc=False)
        masked = 0
        for i in range(12):
            s = on.st[i % len(on.st)]
            t = int(s["starts"][11 * i % len(s["starts"])])
            a, b = on.build(s, t, L_MAX), off.build(s, t, L_MAX)
            assert np.all(a["mask_hist"].numpy() <= b["mask_hist"].numpy())
            masked += int(b["mask_hist"].sum() - a["mask_hist"].sum())
        assert masked > 0, f"{name}: QC окна не снял валидность ни с одного часа"


@pytest.mark.heavy
def test_journal_records_augment_profile(manifest, tmp_path):
    from mayak.protocol import Protocol, Stage, run_protocol
    proto = Protocol(stages=(Stage("A", "L0", 1, 0),), batch_size=2, windows_per_epoch=4,
                     num_workers=0, precision="32", val_every=1, val_batches=1)
    data = DataConfig(manifest=manifest, augment=AugmentConfig.from_profile("soft", gap_prob=0.9))
    j = run_protocol("mayak", manifest, proto, out_root=str(tmp_path), accelerator="cpu",
                     data_config=data, enable_progress_bar=False)
    assert j["augment"]["profile"] == "soft"
    assert j["augment"]["deviations"] == {"gap_prob": [0.3, 0.9]}


def _recorded_window(seed=0, holes=0.1):
    w = _window(seed, holes=holes)
    w.y[:] = np.where(w.y_mask > 0, np.round(w.y), 0.0)
    assert is_recorded(w.x, w.m)
    return w


def test_dither_then_record_returns_the_same_record():
    for seed in range(20):
        w0 = _recorded_window(seed)
        w = A.dither_window(_copy(w0), np.random.default_rng(seed))
        assert not np.array_equal(w.x, w0.x) and not np.array_equal(w.y, w0.y)
        half = A.DITHER_FRAC * 0.5 / np.array([1.0, 10.0, 1.0])
        assert np.all(np.abs(w.x - w0.x) <= half + 1e-4)
        A.record_window(w)
        assert np.array_equal(w.x, w0.x) and np.array_equal(w.y, w0.y)
        assert np.array_equal(w.m, w0.m) and np.array_equal(w.y_mask, w0.y_mask)


def test_small_offset_shifts_the_record_like_a_real_sensor():
    """Смещение на 0.3 градуса у настоящего датчика переворачивает запись примерно в 30 %
    часов; без непрерывности перед искажением запись не менялась бы вовсе."""
    diffs, flat = [], []
    for seed in range(10):
        w0 = _recorded_window(seed)
        ok = w0.m[:, 0] > 0
        w = A.dither_window(_copy(w0), np.random.default_rng(seed))
        A.record_window(apply_one(w, "offset", dict(b=0.3)))
        diffs.append(w.x[ok, 0] - w0.x[ok, 0])
        v = A.record_window(apply_one(_copy(w0), "offset", dict(b=0.3)))
        flat.append(v.x[ok, 0] - w0.x[ok, 0])
    d = np.concatenate(diffs)
    assert set(np.unique(d)) <= {0.0, 1.0}
    assert abs(d.mean() - 0.3) < 0.04, d.mean()
    assert not np.any(np.concatenate(flat)), "без непрерывности смещение 0.3 исчезает"


def test_weak_noise_survives_recording():
    cfg = AugmentConfig.only("noise")
    changed = []
    for seed in range(10):
        w0 = _recorded_window(seed)
        w = A.record_window(augment_window(_copy(w0), cfg, np.random.default_rng(seed)))
        ok = w0.m[:, 0] > 0
        changed.append(np.mean(w.x[ok, 0] != w0.x[ok, 0]))
        assert np.array_equal(w.y, w0.y), "шум не трогает цель"
    assert 0.10 < np.mean(changed) < 0.22, "шум 0.2 градуса меняет запись в доле часов E|шум|"


@pytest.mark.parametrize("name", MASK_AUGS + ("spike", "stuck", "units", "coords"))
def test_no_dither_without_sensor_distortion(name):
    for seed in range(5):
        w0 = _recorded_window(seed)
        w = augment_window(_copy(w0), AugmentConfig.only(name), np.random.default_rng(seed))
        assert np.array_equal(w.y, w0.y)
        same = (w.m > 0) & (w0.m > 0)
        if name in MASK_AUGS or name == "coords":
            assert np.array_equal(w.x[same], w0.x[same]), name
        assert is_recorded(record_values(w.x), w.m)
