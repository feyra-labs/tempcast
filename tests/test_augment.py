"""Тесты: аугментации, имитирующие реальный прибор."""
import csv
import dataclasses

import numpy as np
import pytest

from mayak.config import AUGMENT_PROB_FIELDS, AUGMENT_PROFILES, AugmentConfig, RunConfig
from mayak.constants import H, L_MAX
from mayak.data import augment as A
from mayak.data import store as S
from mayak.data.augment import (AUG_ORDER, EXPECTED_QC, apply_one, augment_window,
                                clean_history, make_window, qc_effect, reference_windows)
from mayak.data.masking import enforce_invariant
from mayak.data.recording import record_values
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL

VALUE_AUGS = ("offset", "noise", "rh_dewpoint", "stuck")
ON_TARGET = ("offset",)
MASK_AUGS = ("dropout", "gap", "drop_channel")
ALL_ON = {f: 1.0 for f in AUGMENT_PROB_FIELDS.values()}


def _only(name, profile="default"):
    """Профиль, в котором срабатывает только одна аугментация, всегда."""
    probs = {f: 0.0 for f in AUGMENT_PROB_FIELDS.values()}
    probs[AUGMENT_PROB_FIELDS[name]] = 1.0
    return AugmentConfig.from_profile(profile, **probs)


def _on_record_grid(x, mask=None):
    """Каждое имеющееся значение лежит на сетке записи прибора."""
    x = np.asarray(x, np.float32)
    ok = np.ones(x.shape, bool) if mask is None else np.asarray(mask) > 0
    ok &= np.isfinite(x)
    return bool(np.array_equal(record_values(x)[ok], x[ok]))


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
                               y_mask=w.y_mask.copy(), hour=w.hour.copy(),
                               hour_fut=w.hour_fut.copy(), applied={})


@pytest.mark.parametrize("L", [0, 1, 30, 200, L_MAX])
@pytest.mark.parametrize("holes", [0.0, 0.3])
def test_invariant_holds_after_all_augmentations(L, holes):
    cfg = AugmentConfig.from_profile("default", **ALL_ON)
    rng = np.random.default_rng(L)
    for seed in range(6):
        w = augment_window(_window(seed, L=L, holes=holes), cfg, rng)
        x, m = enforce_invariant(w.x, w.m)
        assert np.all(x[m == 0] == 0) and np.all(np.isfinite(x))
        assert set(np.unique(m)) <= {0.0, 1.0}
        assert np.all(m[:L_MAX - L] == 0), "история не вылезает левее своего начала"


@pytest.mark.parametrize("name", VALUE_AUGS)
def test_value_distortions_touch_only_valid_points(name):
    cfg = _only(name)
    for seed in range(8):
        w0 = _window(seed, holes=0.3)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.array_equal(w.m, w0.m), "искажение значений не трогает маску"
        assert np.all(w.x[w0.m == 0] == 0), "значения в дырах не появляются"


@pytest.mark.parametrize("name", MASK_AUGS)
def test_availability_only_removes_validity(name):
    cfg = _only(name)
    for seed in range(8):
        w0 = _window(seed, holes=0.1)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.all(w.m <= w0.m) and np.array_equal(w.x, w0.x)
        assert name in w.applied and w.m.sum() < w0.m.sum()


@pytest.mark.parametrize("name", [n for n in AUG_ORDER if n not in ON_TARGET])
def test_target_untouched_except_station_offset(name):
    cfg = _only(name)
    for seed in range(6):
        w0 = _window(seed, holes=0.2)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        assert np.array_equal(w.y, w0.y) and np.array_equal(w.y_mask, w0.y_mask)


def test_offset_shifts_history_and_target_consistently():
    """Смещение станции одинаково на истории и цели в один и тот же солнечный час.

    Долгота у истории и цели одна, поэтому один солнечный час - один час UTC.
    """
    cfg = dataclasses.replace(_only("offset"), offset_diurnal_frac=1.0)
    tol = A.DITHER_FRAC * 0.5 / np.array([1.0, 10.0, 1.0]) + 1e-4
    for seed in range(10):
        w0 = _window(seed, holes=0.2)
        w = augment_window(_copy(w0), cfg, np.random.default_rng(seed))
        p = w.applied["offset"]
        assert cfg.offset_min <= abs(p["b"]) <= cfg.offset_max
        assert 0.0 < abs(p["amp"]) <= cfg.offset_diurnal_max
        assert cfg.offset_peak_hours[0] <= p["peak"] <= cfg.offset_peak_hours[1]
        exact = apply_one(_copy(w0), "offset", p)
        mT, mY = w0.m[:, 0] > 0, w0.y_mask > 0
        dx = exact.x[:, 0] - w0.x[:, 0]
        dy = exact.y - w0.y
        for h in range(24):
            at_x, at_y = dx[mT & (w0.hour == h)], dy[mY & (w0.hour_fut == h)]
            assert at_x.size and at_y.size
            np.testing.assert_allclose(at_x, at_y[0], atol=1e-5)
            np.testing.assert_allclose(at_y, at_y[0], atol=1e-5)
        per_hour = [dy[mY & (w0.hour_fut == h)][0] for h in range(24)]
        assert np.mean(per_hour) == pytest.approx(p["b"], abs=1e-4), "постоянная часть"
        assert np.ptp(per_hour) > abs(p["amp"]), "суточная составляющая"
        assert np.all(np.abs(w.y - w0.y - dy * w0.y_mask) <= tol[0])
        assert np.array_equal(w.y[~mY], w0.y[~mY])
        assert np.array_equal(w.y_mask, w0.y_mask), "маска цели - только из данных (1.6)"
        assert np.all(np.abs(w.x[mT, 0] - w0.x[mT, 0] - dx[mT]) <= tol[0])
        for ch in (1, 2):
            assert np.all(np.abs(w.x[:, ch] - w0.x[:, ch]) <= tol[ch]), "непрерывность"
        A.record_window(w)
        w1 = A.record_window(_copy(w0))
        assert np.array_equal(w.x[:, 1:], w1.x[:, 1:]), "после записи P и RH прежние"
        v = _copy(w0).add_station_offset(p["b"], p["amp"], p["peak"], target=False)
        assert np.array_equal(v.y, w0.y), "вариант без цели трогает цель"


def test_offset_needs_history_and_obeys_ablation():
    w = augment_window(_window(0, L=0), _only("offset"), np.random.default_rng(0))
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
    base = AugmentConfig.from_profile("default", **ALL_ON)
    variants = [dataclasses.replace(base, offset_max=0.0),
                dataclasses.replace(base, stuck_prob=0.0),
                dataclasses.replace(base, gap_max_len=5, noise_sd=(1.0, 1.0, 1.0))]
    for seed in range(10):
        ref = augment_window(_window(seed), base, np.random.default_rng(seed)).applied
        for v, changed in zip(variants, ("offset", "stuck", ("gap", "noise"))):
            got = augment_window(_window(seed), v, np.random.default_rng(seed)).applied
            for name in set(ref) | set(got):
                if name in np.atleast_1d(changed):
                    continue
                assert ref.get(name) == got.get(name), (changed, name)


def test_firing_frequencies_follow_probabilities():
    cfg = AugmentConfig.from_profile("default")
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


@pytest.mark.parametrize("name", ON_TARGET)
def test_station_offset_never_changes_target_mask(name):
    cfg = _only(name)
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


MIN_HIT = {"stuck": 0.7, "dropout": 1.0, "gap": 1.0, "drop_channel": 1.0}


def _detectable(name, params, w):
    """Распознаёт ли причинный QC искажение по прошлому.

    Залипание одного канала видно только после срока залипания этого канала;
    более короткое устройство не отличит от нормы.
    """
    from mayak.data.qc import DEFAULT_QC as C
    if name != "stuck":
        return True
    n = min(params["n"], L_MAX - params["i"])
    need = C.stuck_T_alone_hours if params["ch"] == 0 else C.stuck_hours[params["ch"]]
    return n > need


@pytest.mark.parametrize("name", AUG_ORDER)
def test_sampled_augmentation_produces_expected_codes(name):
    cfg = _only(name)
    # Залипание короче срока QC не распознаётся, поэтому для него окон больше.
    hits, sides, n = [], [], 80 if name == "stuck" else 40
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


def test_dataset_default_keeps_contract(manifest):
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
        if "offset" in info:
            o = info["offset"]
            y = y + A.station_offset(it["hour_fut"].numpy(), s["lon"], o["b"], o["amp"],
                                     o["peak"]) * ym0
        got = it["y"].numpy()
        assert np.all(np.abs(got - y)[ym0 > 0] <= 1.0 + 1e-4), \
            "цель - запись прибора после смещения: не дальше полушага шума и полушага записи"
        assert np.array_equal(got[ym0 == 0], y0[ym0 == 0])
        assert _on_record_grid(it["x_hist"].numpy(), it["mask_hist"].numpy())
        assert np.array_equal(it["y"].numpy(), np.round(it["y"].numpy()))
        for k, shape in (("x_hist", (L_MAX, 3)), ("mask_hist", (L_MAX, 3)), ("y", (H,))):
            assert tuple(it[k].shape) == shape
    assert len(seen) >= 5, f"за 24 окна сработали только {sorted(seen)}"


def test_dataset_window_qc_masks_augmented_artifacts(manifest):
    from mayak.data.dataset import WindowDataset
    on = WindowDataset(manifest, windows_per_epoch=4, seed=0,
                       augment=_only("stuck"), window_qc=True)
    off = WindowDataset(manifest, windows_per_epoch=4, seed=0,
                        augment=_only("stuck"), window_qc=False)
    masked = 0
    for i in range(12):
        s = on.st[i % len(on.st)]
        t = int(s["starts"][11 * i % len(s["starts"])])
        a, b = on.build(s, t, L_MAX), off.build(s, t, L_MAX)
        assert np.all(a["mask_hist"].numpy() <= b["mask_hist"].numpy())
        masked += int(b["mask_hist"].sum() - a["mask_hist"].sum())
    assert masked > 0, "stuck: QC окна не снял валидность ни с одного часа"


def _recorded_window(seed=0, holes=0.1):
    w = _window(seed, holes=holes)
    w.y[:] = np.where(w.y_mask > 0, np.round(w.y), 0.0)
    assert _on_record_grid(w.x, w.m)
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
    """Смещение на 0.3 градуса у настоящего датчика переворачивает запись примерно в 30 % часов.

    Без непрерывности перед искажением запись не менялась бы вовсе.
    """
    diffs, flat = [], []
    for seed in range(10):
        w0 = _recorded_window(seed)
        ok = w0.m[:, 0] > 0
        w = A.dither_window(_copy(w0), np.random.default_rng(seed))
        A.record_window(apply_one(w, "offset", dict(b=0.3, amp=0.0, peak=0.0)))
        diffs.append(w.x[ok, 0] - w0.x[ok, 0])
        v = A.record_window(apply_one(_copy(w0), "offset", dict(b=0.3, amp=0.0, peak=0.0)))
        flat.append(v.x[ok, 0] - w0.x[ok, 0])
    d = np.concatenate(diffs)
    assert set(np.unique(d)) <= {0.0, 1.0}
    assert abs(d.mean() - 0.3) < 0.04, d.mean()
    assert not np.any(np.concatenate(flat)), "без непрерывности смещение 0.3 исчезает"


def test_weak_noise_survives_recording():
    cfg = _only("noise")
    changed = []
    for seed in range(10):
        w0 = _recorded_window(seed)
        w = A.record_window(augment_window(_copy(w0), cfg, np.random.default_rng(seed)))
        ok = w0.m[:, 0] > 0
        changed.append(np.mean(w.x[ok, 0] != w0.x[ok, 0]))
        assert np.array_equal(w.y, w0.y), "шум не трогает цель"
    assert 0.10 < np.mean(changed) < 0.22, "шум 0.2 градуса меняет запись в доле часов E|шум|"


@pytest.mark.parametrize("name", MASK_AUGS + ("stuck",))
def test_no_dither_without_sensor_distortion(name):
    for seed in range(5):
        w0 = _recorded_window(seed)
        w = augment_window(_copy(w0), _only(name), np.random.default_rng(seed))
        assert np.array_equal(w.y, w0.y)
        same = (w.m > 0) & (w0.m > 0)
        if name in MASK_AUGS:
            assert np.array_equal(w.x[same], w0.x[same]), name
        assert _on_record_grid(record_values(w.x), w.m)
