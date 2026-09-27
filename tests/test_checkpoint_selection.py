"""Тесты: набор валидации для выбора чекпойнта и журнал выбора."""
import collections
import csv

import numpy as np
import pytest
import torch

from mayak.config import ConfigError, DataConfig
from mayak.constants import H, L_MAX, QUANTILES
from mayak.data import store as S
from mayak.data.dataset import (HISTORY_MIX, WindowDataset, history_probability,
                                sample_history_len)
from mayak.data.holdout import (HISTORY_BINS, EvalSet, stratified_items,
                                window_history_lengths)
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL, time_layout
from mayak.protocol import CURRICULA

N_HOURS = 12_000
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("v0", ROLE_VAL), ("v1", ROLE_VAL),
            ("v2", ROLE_VAL), ("x0", ROLE_TEST)]
SHORT = "v2"


def _write_station(root, sid, seed):
    """Чистый ряд; у короткой станции цель валидна только в первом блоке валидации."""
    rng = np.random.default_rng(seed)
    h = np.arange(N_HOURS)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + 0.3 * rng.standard_normal(N_HOURS)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(N_HOURS)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(N_HOURS)
    valid = np.ones((N_HOURS, 3), np.uint8)
    if sid == SHORT:
        for lo, hi in time_layout(N_HOURS).blocks["val"][1:]:
            valid[lo:hi] = 0
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32),
             P=P.astype(np.float32), RH=RH.astype(np.float32), valid=valid,
             t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("selection")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb",
                         split=role))
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


def _val_set(store, manifest, **kw):
    kw = {**dict(every_hours=24, max_windows=None, curriculum="full"), **kw}
    return EvalSet(store.clims(), station_splits=(ROLE_VAL,), manifest=manifest,
                   time_key="val", **kw)


def _datamodule(manifest, curriculum, **data):
    from mayak.data.datamodule import MayakData
    dm = MayakData(manifest, curriculum=curriculum, windows_per_epoch=4, num_workers=0,
                   data_config=DataConfig(manifest=manifest, **data))
    dm.setup()
    return dm


def _old_sample(rng, curriculum):
    """Прежняя выборка длины истории обучающего окна, записанная как эталон."""
    if curriculum == "L0":
        return 0
    u = rng.random()
    if u < 0.05:
        return 0
    if u < 0.20:
        return int(rng.integers(1, 49))
    if u < 0.45:
        return int(rng.integers(48, 241))
    return int(rng.integers(240, L_MAX + 1))


def test_history_mix_covers_protocol_curricula():
    assert set(HISTORY_MIX) == set(CURRICULA)
    for curriculum, parts in HISTORY_MIX.items():
        assert parts[-1][0] == 1.0, curriculum
        assert all(0 <= lo <= hi <= L_MAX for _p, lo, hi in parts)
        assert history_probability(curriculum, 0, L_MAX) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sample_history_len(np.random.default_rng(0), "half")


@pytest.mark.parametrize("curriculum", sorted(HISTORY_MIX))
def test_training_stream_is_unchanged(curriculum):
    """Общая функция выборки длины расходует генератор так же, как прежняя."""
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    old = [_old_sample(a, curriculum) for _ in range(5000)]
    new = [sample_history_len(b, curriculum) for _ in range(5000)]
    assert old == new
    assert a.random() == b.random()


def test_window_dataset_draws_from_the_shared_mix(manifest, store):
    ds = WindowDataset(manifest, windows_per_epoch=4, seed=5, store=store)
    ref = np.random.default_rng(0)
    ds.rng_sample = np.random.default_rng(0)
    assert [ds._sample_L() for _ in range(300)] == \
        [sample_history_len(ref, "full") for _ in range(300)]


def test_validation_set_is_deterministic_by_seed(store, manifest):
    a = _val_set(store, manifest, history_seed=0, windows_per_station=10)
    b = _val_set(store, manifest, history_seed=0, windows_per_station=10)
    c = _val_set(store, manifest, history_seed=1, windows_per_station=10)
    assert a.items == b.items == c.items, "сид длин не меняет выбор окон"
    assert a.requested == b.requested and a.fingerprint() == b.fingerprint()
    assert a.requested != c.requested and a.fingerprint() != c.fingerprint()
    for i in range(len(a)):
        assert torch.equal(a[i]["x_hist"], b[i]["x_hist"])


def test_validation_set_does_not_depend_on_architecture_or_protocol_seeds(manifest):
    """Набор строится из конфига данных и куррикулума этапа; сиды протокола в него не входят."""
    a = _datamodule(manifest, "full", val_windows_per_station=8)
    from mayak.data.datamodule import MayakData
    b = MayakData(manifest, curriculum="full", windows_per_epoch=4, num_workers=0, seed=11,
                  aug_seed=12, data_config=DataConfig(manifest=manifest,
                                                      val_windows_per_station=8))
    b.setup()
    assert a.val_ds.fingerprint() == b.val_ds.fingerprint()
    assert a.val_ds.requested == b.val_ds.requested


def test_window_length_depends_only_on_station_and_time():
    items = [(f"s{i}", t) for i in range(4) for t in range(0, 4000, 24)]
    full = window_history_lengths(items, "full", 3)
    part = window_history_lengths(items[::5], "full", 3)
    assert (part == full[::5]).all(), "длина окна сдвинулась от состава набора"
    with pytest.raises(ValueError):
        window_history_lengths(items, "full", -1)


def test_stage_a_validation_is_cold_start_only(manifest):
    dm = _datamodule(manifest, "L0", val_windows_per_station=6)
    assert len(dm.val_ds) > 0 and set(dm.val_ds.requested) == {0}
    assert all(int(dm.val_ds[i]["hist_len"]) == 0 for i in range(len(dm.val_ds)))


def test_each_station_gives_the_same_number_of_windows(store, manifest):
    cand = collections.Counter(sid for sid, _t in _val_set(store, manifest).items)
    assert cand[SHORT] < cand["v0"] == cand["v1"]
    k = cand[SHORT] + 3
    ds = _val_set(store, manifest, windows_per_station=k)
    got = collections.Counter(sid for sid, _t in ds.items)
    assert got == {"v0": k, "v1": k, SHORT: cand[SHORT]}, "короткая станция отдаёт все свои"
    firsts = {lo for lo, _hi in time_layout(N_HOURS).blocks["val"]}
    for sid in ("v0", "v1"):
        ts = [t for s, t in ds.items if s == sid]
        blocks = {max(lo for lo in firsts if lo <= t) for t in ts}
        assert blocks == firsts, "окна станции разнесены по всем блокам валидации"


def test_stratified_items_keeps_small_stations_whole():
    per_station = {"a": list(range(50)), "b": [3, 7], "c": []}
    items = stratified_items(per_station, windows_per_station=5)
    count = collections.Counter(sid for sid, _t in items)
    assert count == {"a": 5, "b": 2}


def test_datamodule_validation_set_is_stratified_over_all_stations(manifest):
    dm = _datamodule(manifest, "full", val_windows_per_station=4)
    count = collections.Counter(sid for sid, _t in dm.val_ds.items)
    assert set(count) == {"v0", "v1", SHORT} and set(count.values()) == {4}


def test_history_distribution_matches_curriculum():
    items = [(f"s{i}", t) for i in range(25) for t in range(0, 40_000, 25)]
    lengths = window_history_lengths(items, "full", 0)
    n = len(lengths)
    edges = ((0, 0), (1, 47), (48, 48), (49, 239), (240, 240), (241, L_MAX))
    for lo, hi in edges:
        p = history_probability("full", lo, hi)
        share = float(((lengths >= lo) & (lengths <= hi)).mean())
        assert abs(share - p) <= 4 * np.sqrt(p * (1 - p) / n), (lo, hi, share, p)


def test_validation_items_carry_their_history_length(store, manifest):
    ds = _val_set(store, manifest, windows_per_station=12)
    assert len(set(ds.requested)) > 1
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["hist_len"]) == ds.requested[i] == ds.history_length(i)
        assert int(item["mask_hist"][:, 0].sum()) == ds.requested[i]
    meta = ds.window_meta()
    assert meta["history"].tolist() == ds.requested


def test_one_length_and_curriculum_are_exclusive(store, manifest):
    with pytest.raises(ValueError):
        _val_set(store, manifest, L=24)


def test_data_config_validation_fields():
    assert DataConfig().val_windows_per_station >= 1 and DataConfig().val_seed == 0
    with pytest.raises(ConfigError):
        DataConfig(val_windows_per_station=0)
    with pytest.raises(ConfigError):
        DataConfig(val_seed=-1)
    with pytest.raises(ConfigError, match="val_windows_per_station"):
        DataConfig.from_dict(dict(val_max_windows=8000))


def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = 10 + 3 * torch.randn(B, H, generator=g)
    y_mask = (torch.rand(B, H, generator=g) < 0.8).float()
    q = y[..., None] + torch.randn(B, H, len(QUANTILES), generator=g).sort(-1).values
    return dict(y=y * y_mask, y_mask=y_mask, norm_scale=1 + torch.rand(B, H, generator=g)), \
        dict(q=q)


def test_forecast_loss_is_mean_of_its_terms():
    from mayak.loss import forecast_loss, forecast_terms, masked_mean
    batch, out = _batch()
    per_pair, weight = forecast_terms(out, batch)
    assert per_pair.shape == weight.shape == batch["y"].shape
    assert torch.equal(masked_mean(per_pair, weight), forecast_loss(out, batch))


class _Journal:
    """Собирает записи журнала так же, как их сводит проход валидации: взвешенное среднее."""

    def __init__(self):
        self.acc = collections.defaultdict(lambda: [0.0, 0])

    def log(self, key, value, batch_size):
        self.acc[key][0] += float(value) * batch_size
        self.acc[key][1] += batch_size

    def means(self):
        return {k: s / n for k, (s, n) in self.acc.items()}


def test_history_bins_average_to_the_whole_set_mean():
    from mayak.lit import HISTORY_LOG, history_bin_key, log_history_bins
    from mayak.loss import forecast_terms, masked_mean
    assert [k for *_b, k in HISTORY_LOG] == [history_bin_key(lo, hi) for lo, hi, _n in HISTORY_BINS]
    assert history_bin_key(0, 0) == "val/pinball_L0"
    assert history_bin_key(1, 24) == "val/pinball_L1-24"
    journal, pairs, weights, lengths = _Journal(), [], [], []
    for seed in range(4):
        batch, out = _batch(seed=seed)
        hist = torch.tensor([0, 5, 30, 400, 672, 20])[: batch["y"].shape[0]]
        per_pair, weight = forecast_terms(out, batch)
        log_history_bins(journal, per_pair, weight, hist)
        pairs.append(per_pair)
        weights.append(weight)
        lengths.append(hist)
    per_pair, weight, hist = torch.cat(pairs), torch.cat(weights), torch.cat(lengths)
    got = journal.means()
    for lo, hi, key in HISTORY_LOG:
        inside = ((hist >= lo) & (hist <= hi)).float()[:, None]
        want = float(masked_mean(per_pair, weight * inside))
        assert got[key] == pytest.approx(want, rel=1e-6), key


def test_history_bins_skip_empty_bins():
    from mayak.lit import log_history_bins
    from mayak.loss import forecast_terms
    journal = _Journal()
    batch, out = _batch(B=3)
    per_pair, weight = forecast_terms(out, batch)
    log_history_bins(journal, per_pair, weight, torch.zeros(3, dtype=torch.int64))
    assert set(journal.means()) == {"val/pinball_L0"}
