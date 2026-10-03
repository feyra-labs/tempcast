"""Тесты: рантайм на компилируемом языке.

Что проверяется на стороне Python; сторону Rust проверяют тесты крейта рантайма:
* шаг энкодера без состояния (step_shift) совпадает с кольцевым шагом и пакетным проходом;
* графы (PyTorch и ONNX) в хосте на графах совпадают с потоковым рантаймом на
  PyTorch - на модели по умолчанию и на каждой абляции, дольше полного окна;
* экспорт не меняет режим модели (регрессия: после экспорта модель оставалась в train);
* манифест согласован с конфигом, QC, форматом состояния и порогами смены точки;
* эталонные сценарии хоста не устарели относительно текущего кода: хост на Python
  проходит их на модели PyTorch и на закоммиченных графах fp32 и int8 с допусками хоста
  на Rust;
* эталон покрывает коды контроля качества, запись половин со сверкой состояния, редкую
  отчётность, int8, откат, прогноз без наблюдений, уточнение координат и перенос прибора;
* состояние эталона - сырое окно, которое читается и пишется без изменений.
"""
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


ABLATIONS = [None, "no_passport", "no_solar", "no_mode_groups", "no_compression"]


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
    assert d["hours_of_year"] == 8784
    assert man["graphs"]["init"]["outputs"][-2:] == ["clim_mu", "clim_sig"]
    assert man["state"]["nbytes"] == state_nbytes(cfg) == 3236
    assert man["state"]["nbytes"] == StreamingMayak(model, LAT, LON, ELEV).state_nbytes
    assert {c: tuple(v) for c, v in man["phys"].items()} == {c: PHYS[c] for c in ("T", "P", "RH")}
    np.testing.assert_array_equal(np.float32(man["zq"]), ZQ)
    assert ModelConfig.from_dict(man["model_config"]) == cfg
    from mayak.config import RuntimeConfig
    from mayak.runtime.graphs import GRAPH_FORMAT
    assert man["format"] == GRAPH_FORMAT == 5
    assert RuntimeConfig.from_dict(man["runtime"]) == RuntimeConfig()
    from mayak.constants import HISTORY_BINS
    from mayak.metrics import LEAD_BINS, conformal_table
    assert man["calibration"]["lead_bins"] == [list(b) for b in LEAD_BINS]
    assert man["calibration"]["history_bins"] == [[b[0], b[1]] for b in HISTORY_BINS]
    table = np.fromfile(tmp_path / "m" / "conformal.f32", "<f4").reshape(
        len(HISTORY_BINS), cfg.horizon, -1)
    for k, (lo, hi, _name) in enumerate(HISTORY_BINS):
        for L in (lo, hi):
            np.testing.assert_array_equal(table[k], conformal_table(G.GOLDEN_SHIFT, L,
                                                                    cfg.horizon))


def _scenario(doc, name):
    return next(s for s in doc["scenarios"] if s["name"] == name)


def _replay(model, golden, name, state_dir, onnx):
    from mayak.metrics import I_MED
    doc, blob = golden
    sc = _scenario(doc, name)
    make = lambda s, site: G.scenario_runtime(model, GOLDEN, s, site, onnx=onnx)
    return sc, G.replay_scenario(sc, blob, make, str(state_dir), GOLDEN, model.cfg.n_quantiles,
                                 I_MED)


@pytest.mark.parametrize("name", G.SCENARIOS)
def test_golden_is_fresh(model, golden, name, tmp_path):
    """Эталон воспроизводится текущим кодом хоста.

    Упало - поведение эталона изменилось: пересоздать эталон скриптом и прогнать cargo test.
    """
    sc, err = _replay(model, golden, name, tmp_path, onnx=False)
    tol = G.Q_ATOL_INT8 if sc["precision"] == "int8" else G.FRESH_ATOL
    assert err <= tol, f"{name}: эталон устарел, max|Δq| = {err:.2e}"


@pytest.mark.parametrize("name", G.SCENARIOS)
def test_python_host_on_exported_graphs(model, golden, name, tmp_path):
    """Хост на Python поверх закоммиченных графов проходит эталонные сценарии.

    Допуски те же, что у хоста на Rust.
    """
    sc, err = _replay(model, golden, name, tmp_path, onnx=True)
    tol = G.Q_ATOL_INT8 if sc["precision"] == "int8" else G.Q_ATOL
    assert err <= tol, f"{name}: графы эталона расходятся с эталоном: {err:.2e}"


def _lines(sc, kind):
    return [e for e in sc["events"] if e["op"] == "cmd" and e["line"].split()[0] == kind]


def test_golden_covers_device_cases(golden):
    doc, _ = golden
    assert doc["format"] == G.GOLDEN_FORMAT and doc["seed"] == G.GOLDEN_SEED
    assert tuple(s["name"] for s in doc["scenarios"]) == G.SCENARIOS
    assert doc["aci_margin_min"] >= G.MIN_ACI_MARGIN

    seen = 0
    for e in _lines(_scenario(doc, "qc"), "obs"):
        for c in e["expect"]["codes"]:
            seen |= c
    assert seen & (4 | 32 | 64) == 4 | 32 | 64, "выброс, залипание, давление на уровне моря"

    ev = _scenario(doc, "restart")["events"]
    end = max(i for i, e in enumerate(ev) if e["op"] == "state")
    halves = [float(v) for e in _lines(dict(events=ev[:end]), "obs")
              for v in e["line"].split()[2:] if v not in ("-", "nan")]
    assert any(v < 0 and v % 1 == 0.5 for v in halves), "запись половин сверяется по состоянию"
    sparse = [int(e["line"].split()[1]) // 3600 for e in _lines(_scenario(doc, "sparse"), "obs")]
    assert {2, 3} <= set(np.diff(sparse).tolist())
    assert _scenario(doc, "int8")["precision"] == "int8"

    fb = _scenario(doc, "fallback")
    assert fb["model"] == "model_nan"
    assert all(e["expect"]["fallback"] for e in _lines(fb, "forecast"))
    assert len(_lines(fb, "forecast")[0]["line"].split()) == 2, "откат до первого наблюдения"
    no_obs = _lines(_scenario(doc, "no_obs"), "forecast")
    assert not no_obs[0]["expect"]["fallback"] and len(no_obs[0]["line"].split()) == 2
    shift = _scenario(doc, "site_shift")
    assert any(e["op"] == "restart" and e["site"] for e in shift["events"])
    reloc = _lines(_scenario(doc, "relocation"), "status")
    changes = [e["expect"]["site_change"] for e in reloc]
    assert {"refined", "moved", "same"} <= set(changes), "уточнение, перенос и та же точка"
    first_moved = reloc[changes.index("moved")]["expect"]
    assert first_moved["filled"] == 0 and first_moved["theta"] == [0.0] * 4
    assert any(e["expect"]["codes"][1] & 64 for e in _lines(_scenario(doc, "relocation"), "obs")), \
        "после уточнения высоты давление на уровне моря решается по новой высоте"


def test_golden_state_is_raw_window(model, golden):
    doc, _ = golden
    sc = _scenario(doc, "restart")
    with open(os.path.join(GOLDEN, sc["init_files"][0]["file"]), "rb") as fh:
        raw = fh.read()
    s = StreamingMayak(model, sc["lat"], sc["lon"], sc["elev"], aci=G.GOLDEN_ACI)
    s.load_state(raw)
    assert s.filled == model.cfg.stream_window, "эталон рестарта начинается с полного окна"
    assert any(t != 0.0 for t in s.theta)
    assert s.serialize() == raw
    assert len(raw) == 3236


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
