"""Тесты: рантайм на компилируемом языке.

Что проверяется на стороне Python (сторона Rust - runtime-rs/tests/golden.rs):
* шаг энкодера без состояния (step_shift) совпадает с кольцевым шагом и пакетным проходом;
* графы (PyTorch и ONNX) в хосте на графах совпадают с потоковым рантаймом на
  PyTorch - на модели по умолчанию и на каждой абляции, дольше полного окна;
* экспорт не меняет режим модели (регрессия: после экспорта модель оставалась в train);
* манифест согласован с конфигом, QC, форматом состояния;
* эталонные векторы не устарели относительно текущего кода: сценарии, состояния,
  календарь и калибровка воспроизводятся заново;
* закоммиченные графы эталона совпадают с моделью эталона;
* состояние эталона - сырое окно, которое читается и пишется без изменений.
"""
import json
import os

import numpy as np
import pytest
import torch

from mayak.config import Ablations, ModelConfig
from mayak.runtime import golden as G
from mayak.runtime.equivalence import divergence, synthetic_series
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
    """Один и тот же поток в эталон и в хосты на графах; наибольшее расхождение каждого."""
    worst = [0.0] * len(others)
    for k in range(len(series["x"])):
        obs = [float(series["x"][k, j]) if series["m"][k, j] > 0 else None for j in range(3)]
        for r in (ref, *others):
            r.step(*obs, series["t0"] + k)
        if k % every == every - 1:
            q0 = ref.forecast()[0]
            worst = [max(w, float(np.abs(q0 - o.forecast()[0]).max()))
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
    assert divergence(model, 2, hours=2 * model.cfg.max_history, seed=1)[
        "batch_stream_max_abs"] <= 5e-4


ABLATIONS = [None, "no_anchor", "no_passport", "no_solar", "no_mode_groups", "no_compression"]


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_onnx_graphs_match_streaming(ablation, tmp_path):
    cfg = ModelConfig(ablations=Ablations(**{ablation: True})) if ablation else None
    m = G.golden_model(cfg)
    man = export_graphs(m, str(tmp_path / "m"), int8=ablation is None)
    for name in GRAPH_NAMES:
        assert set(man["graphs"][name]["inputs"]) <= set(GRAPH_IO[name][0])
        assert man["export_check_max_rel"][name] <= 1e-4
    s = synthetic_series(m.cfg.stream_window + 60, seed=7)
    ref = StreamingMayak(m, LAT, LON, ELEV)
    rts = [GraphRuntime(OnnxBackend(str(tmp_path / "m")), m.cfg, LAT, LON, ELEV)]
    if ablation is None:
        rts.append(GraphRuntime(OnnxBackend(str(tmp_path / "m"), "int8"), m.cfg, LAT, LON, ELEV))
    errs = _feed_compare(ref, rts, s, every=150)
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
    assert (d["stream_edge"], d["stream_tail"], d["history"]) == (cfg.stream_edge,
                                                                  cfg.stream_tail,
                                                                  cfg.max_history)
    assert d["enc_buf_len"] == sum(model.encoder.buffer_pads)
    assert man["state"]["nbytes"] == state_nbytes(cfg) == 3224
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
        assert (s.aci_updates, s.aci_misses, s.filled, s.last_hour, s.idle_hours) == (
            fin["aci_updates"], fin["aci_misses"], fin["filled"], fin["last_hour"],
            fin["idle_hours"])
        assert abs(s.theta - fin["theta"]) <= 1e-6
    assert doc["aci_margin_min"] >= G.MIN_ACI_MARGIN


def test_golden_state_is_raw_window(model, golden):
    doc, _ = golden
    sc = next(s for s in doc["scenarios"] if s["name"] == "restart")
    with open(os.path.join(GOLDEN, sc["init_state"]), "rb") as fh:
        raw = fh.read()
    s = StreamingMayak(model, sc["lat"], sc["lon"], sc["elev"], aci=G.GOLDEN_ACI)
    s.load_state(raw)
    assert s.filled == model.cfg.stream_window, "эталон рестарта начинается с полного окна"
    assert s.theta != 0.0
    assert s.serialize() == raw
    assert len(raw) == 3224


def test_golden_onnx_matches_golden_model(model, golden):
    """Закоммиченные графы - это графы модели эталона (сценарий без калибровки)."""
    doc, blob = golden
    be = OnnxBackend(os.path.join(GOLDEN, "model"))

    def make(sc):
        rt = GraphRuntime(be, model.cfg, sc["lat"], sc["lon"], sc["elev"],
                          conformal=G.GOLDEN_SHIFT if sc["conformal"] else None,
                          aci=G.GOLDEN_ACI if sc["aci"] else None)
        if sc["init_state"]:
            with open(os.path.join(GOLDEN, sc["init_state"]), "rb") as fh:
                rt.load_state(fh.read())
        return rt

    for name in ("extremes", "restart"):
        sc = next(s for s in doc["scenarios"] if s["name"] == name)
        err = G.replay(dict(doc, scenarios=[sc]), blob, make, model.cfg.horizon)[name]
        assert err <= G.Q_ATOL, f"{name}: графы эталона расходятся с моделью: {err:.2e}"
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
