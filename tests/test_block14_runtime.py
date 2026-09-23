"""Тесты: рантайм на компилируемом языке.

Что проверяется на стороне Python (сторона Rust - runtime-rs/tests/golden.rs):
* шаг энкодера без состояния (step_shift) совпадает с кольцевым шагом и пакетным проходом;
* четыре графа (PyTorch и ONNX) в хосте GraphRuntime совпадают с эталонным
  потоковым рантаймом - на модели по умолчанию и на каждой абляции;
* экспорт не меняет режим модели (регрессия: после экспорта модель оставалась в train);
* манифест согласован с конфигом, QC, форматом состояния;
* эталонные векторы не устарели относительно текущего кода: сценарии, состояния,
  календарь и калибровка воспроизводятся заново;
* закоммиченные графы эталона совпадают с моделью эталона;
* календарная конвенция рантайма совпадает с конвенцией сборщика, восстановление
  состояния восстанавливает незавершённые сутки.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from mayak.config import Ablations, ModelConfig
from mayak.runtime import golden as G
from mayak.runtime.equivalence import divergence, future_calendar_after, synthetic_series
from mayak.runtime.graphs import (GRAPH_IO, GRAPH_NAMES, GraphRuntime, OnnxBackend, TorchBackend,
                                  export_graphs, state_nbytes)
from mayak.runtime.streaming import StreamingMayak

LAT, LON, ELEV = 52.37, 4.9, 0.0
ATOL_TORCH_GRAPHS = 1e-4
ATOL_ONNX = 2e-4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, G.DEFAULT_DIR)


@pytest.fixture(scope="module")
def model():
    return G.golden_model()


@pytest.fixture(scope="module")
def golden():
    return G.load(GOLDEN)


def _feed_compare(ref, others, series, every=40):
    """Один и тот же поток в эталон и в хосты на графах → max|Δq| каждого."""
    worst = [0.0] * len(others)
    H = ref.m.cfg.horizon
    for k in range(len(series["doy"])):
        obs = [float(series["x"][k, j]) if series["m"][k, j] > 0 else None for j in range(3)]
        for r in (ref, *others):
            r.step(*obs, series["doy"][k], series["hour"][k])
        if k % every == every - 1:
            cal = future_calendar_after(series, k + 1, H)
            q0 = ref.forecast(*cal)[0]
            worst = [max(w, float(np.abs(q0 - o.forecast(*cal)).max()))
                     for w, o in zip(worst, others)]
    return worst


def test_step_shift_matches_ring_step_and_batch(model):
    enc = model.encoder
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, model.cfg.n_channels, 320, generator=g)
    with torch.no_grad():
        full = enc(x)
        ring = enc.init_state(2)
        buf = enc.ring_to_shift(ring)
        assert buf.shape == (2, model.cfg.encoder_width, sum(enc.buffer_pads))
        err = 0.0
        for t in range(x.shape[-1]):
            h, buf = enc.step_shift(x[..., t], buf)
            h_ring = enc.step(x[..., t], ring)
            err = max(err, float((h - full[:, t]).abs().max()), float((h - h_ring).abs().max()))
        scale = max(1.0, float(full.abs().max()))
        assert err <= 1e-5 * scale
        assert float((enc.ring_to_shift(ring) - buf).abs().max()) <= 1e-5 * scale


def test_torch_graphs_match_streaming(model):
    s = synthetic_series(260, seed=5)
    ref = StreamingMayak(model, LAT, LON, ELEV)
    (err,) = _feed_compare(ref, [GraphRuntime(TorchBackend(model), model.cfg, LAT, LON, ELEV)], s)
    assert err <= ATOL_TORCH_GRAPHS, f"разбиение на графы расходится с эталоном: {err:.2e}"


def test_export_keeps_model_in_eval(model, tmp_path):
    export_graphs(model, tmp_path / "m")
    assert not model.training
    assert divergence(model, 2, seed=1)["batch_stream_max_abs"] <= 1e-4


ABLATIONS = [None, "no_anchor", "no_passport", "no_solar", "no_mode_groups", "no_compression"]


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_onnx_graphs_match_streaming(ablation, tmp_path):
    cfg = ModelConfig(ablations=Ablations(**{ablation: True})) if ablation else None
    m = G.golden_model(cfg)
    man = export_graphs(m, str(tmp_path / "m"), int8=ablation is None)
    for name in GRAPH_NAMES:
        assert set(man["graphs"][name]["inputs"]) <= set(GRAPH_IO[name][0])
        assert man["export_check_max_abs"][name] <= 1e-4
    s = synthetic_series(150, seed=7)
    ref = StreamingMayak(m, LAT, LON, ELEV)
    rts = [GraphRuntime(OnnxBackend(str(tmp_path / "m")), m.cfg, LAT, LON, ELEV)]
    if ablation is None:
        rts.append(GraphRuntime(OnnxBackend(str(tmp_path / "m"), "int8"), m.cfg, LAT, LON, ELEV))
    errs = _feed_compare(ref, rts, s, every=30)
    assert errs[0] <= ATOL_ONNX, f"{ablation}: ONNX fp32 расходится с эталоном: {errs[0]:.2e}"
    if len(errs) > 1:
        assert np.isfinite(errs[1])


def test_manifest_contract(model, tmp_path):
    from mayak.baselines.statistical import ZQ
    from mayak.data.qc import PHYS
    man = export_graphs(model, str(tmp_path / "m"), conformal=G.GOLDEN_SHIFT, aci=G.GOLDEN_ACI)
    cfg, d = model.cfg, man["dims"]
    assert (d["horizon"], d["n_quantiles"], d["n_modes"]) == (cfg.horizon, cfg.n_quantiles,
                                                              cfg.n_modes)
    assert d["stream_window"] == cfg.stream_window
    assert d["enc_buf_len"] == sum(model.encoder.buffer_pads)
    assert man["state"]["nbytes"] == state_nbytes(cfg) == 3352
    assert man["state"]["nbytes"] == StreamingMayak(model, LAT, LON, ELEV).state_nbytes
    assert {c: tuple(v) for c, v in man["phys"].items()} == {c: PHYS[c] for c in ("T", "P", "RH")}
    np.testing.assert_array_equal(np.float32(man["zq"]), ZQ)
    assert ModelConfig.from_dict(man["model_config"]) == cfg
    table = np.fromfile(tmp_path / "m" / "conformal.f32", "<f4").reshape(cfg.horizon, -1)
    from mayak.metrics import conformal_table
    np.testing.assert_array_equal(table, conformal_table(G.GOLDEN_SHIFT, cfg.horizon))


def _stream_factory(model):
    def make(sc):
        s = StreamingMayak(model, sc["lat"], sc["lon"], sc["elev"],
                           conformal=G.GOLDEN_SHIFT if sc["conformal"] else None,
                           aci=G.GOLDEN_ACI if sc["aci"] else None)
        if sc["init_state"]:
            with open(os.path.join(GOLDEN, sc["init_state"]), "rb") as fh:
                s.load_state(fh.read())
        make.last = s
        return s
    return make


def test_golden_is_fresh(model, golden):
    """Эталон воспроизводится текущим кодом. Упало - поведение эталона изменилось:
    пересоздать scripts/make_runtime_golden.py и прогнать cargo test."""
    doc, blob = golden
    assert doc["format"] == G.GOLDEN_FORMAT and doc["seed"] == G.GOLDEN_SEED
    make = _stream_factory(model)
    for sc in doc["scenarios"]:
        one = dict(doc, scenarios=[sc])
        err = G.replay(one, blob, make, model.cfg.horizon)[sc["name"]]
        assert err <= G.FRESH_ATOL, f"{sc['name']}: эталон устарел, max|Δq| = {err:.2e}"
        s, fin = make.last, sc["final"]
        assert (s.aci_updates, s.aci_misses, s.calendar_breaks, s.filled, s._hours_in_day) == (
            fin["aci_updates"], fin["aci_misses"], fin["calendar_breaks"], fin["filled"],
            fin["hours_in_day"])
        assert abs(s.theta - fin["theta"]) <= 1e-6
    assert doc["aci_margin_min"] >= G.MIN_ACI_MARGIN


def test_golden_state_restores_incomplete_day(model, golden):
    doc, _ = golden
    sc = next(s for s in doc["scenarios"] if s["name"] == "restart")
    with open(os.path.join(GOLDEN, sc["init_state"]), "rb") as fh:
        raw = fh.read()
    s = StreamingMayak(model, sc["lat"], sc["lon"], sc["elev"], aci=G.GOLDEN_ACI)
    s.load_state(raw)
    assert 0 < s._hours_in_day < 24, "эталон рестарта должен начинаться с незавершённых суток"
    assert np.abs(s._day[:, :s._hours_in_day]).sum() > 0
    assert s.serialize() == raw
    assert len(raw) == 3352


def test_golden_onnx_matches_golden_model(model, golden):
    """Закоммиченные графы - это графы модели эталона (сценарий без калибровки)."""
    doc, blob = golden
    sc = next(s for s in doc["scenarios"] if s["name"] == "extremes")
    be = OnnxBackend(os.path.join(GOLDEN, "model"))
    err = G.replay(dict(doc, scenarios=[sc]), blob,
                   lambda s: GraphRuntime(be, model.cfg, s["lat"], s["lon"], s["elev"]),
                   model.cfg.horizon)["extremes"]
    assert err <= G.Q_ATOL, f"графы эталона расходятся с моделью эталона: {err:.2e}"
    with open(os.path.join(GOLDEN, "model", "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    assert ModelConfig.from_dict(man["model_config"]) == model.cfg


def test_golden_calendar_and_calibration_are_fresh(model, golden):
    doc, blob = golden
    assert G.calendar_cases() == doc["calendar"]
    fresh = G._Blob()
    cal = G.calibration_cases(model, fresh)
    arr = fresh.array()
    for a, b in zip(cal["cases"], doc["calibration"]["cases"]):
        np.testing.assert_array_equal(G.take(arr, a["expect"]), G.take(blob, b["expect"]))
    assert cal["aci"] == doc["calibration"]["aci"]
    assert cal["score"]["expect"] == doc["calibration"]["score"]["expect"]


def test_runtime_calendar_matches_collector_across_leap_new_year(model):
    """Календарь рантайма (hour_of_year по doy) и сборщика (timeaxis) - одна конвенция:
    поток, собранный по timeaxis через 29 февраля и Новый год, не даёт разрывов."""
    from mayak.timeaxis import to_utc_hour, window_calendar
    t0 = int(to_utc_hour(np.datetime64("2024-12-30T00", "s")))
    doy, hour = window_calendar(t0, np.arange(96))
    s = StreamingMayak(model, LAT, LON, ELEV)
    for d, h in zip(doy, hour):
        s.step(None, None, None, d, h)
    assert s.calendar_breaks == 0
    t1 = int(to_utc_hour(np.datetime64("2024-02-28T12", "s")))
    d2, h2 = window_calendar(t1, np.arange(48))
    assert np.all(np.diff(np.rint(d2 * 24.0)) == 1)


@pytest.mark.skipif(not os.environ.get("MAYAK_RUST_TESTS") or shutil.which("cargo") is None,
                    reason="MAYAK_RUST_TESTS=1 и cargo - прогон тестов runtime-rs из pytest")
def test_rust_runtime_matches_golden():
    out = subprocess.run(["cargo", "test", "--release"], cwd=os.path.join(ROOT, "runtime-rs"),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
