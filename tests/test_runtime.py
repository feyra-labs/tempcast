"""Тесты: графы устройства и хост на графах.

* шаг энкодера без состояния (step_shift) совпадает с кольцевым шагом и пакетным проходом;
* графы (PyTorch и ONNX) в хосте на графах совпадают с потоковым рантаймом на
  PyTorch - на модели по умолчанию и на каждой абляции, дольше полного окна;
* экспорт не меняет режим модели (регрессия: после экспорта модель оставалась в train);
* манифест согласован с конфигом, QC, форматом состояния и порогами смены точки.
"""
import numpy as np
import pytest
import torch

from mayak.config import Ablations, ModelConfig
from mayak.metrics import ACIParams
from mayak.runtime.equivalence import divergence, synthetic_series
from mayak.runtime.graphs import (GRAPH_IO, GRAPH_NAMES, GraphRuntime, OnnxBackend, TorchBackend,
                                  export_graphs, state_nbytes)
from mayak.runtime.streaming import StreamingMayak

LAT, LON, ELEV = 52.37, 4.9, 0.0
ATOL_TORCH_GRAPHS = 1e-4
ATOL_ONNX = 2e-4
SEED, PERTURB = 1414, 0.05
ACI = ACIParams(target=0.10, gamma=0.05, max_factor=4.0)
SHIFT = ((np.array([-0.3, -0.2, -0.08, 0.0, 0.08, 0.2, 0.3], np.float32)[None, :]
          * np.array([1.0, 1.5, 2.0, 2.5], np.float32)[:, None]
          + np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1], np.float32))[:, None, :]
         * np.array([1.6, 1.3, 1.1, 1.0], np.float32)[None, :, None])


def _model(cfg=None):
    """Модель с сидом и шумом на всех параметрах: нулевые инициализации не прячут ошибки."""
    from mayak.model import MAYAK
    torch.manual_seed(SEED)
    m = MAYAK(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(PERTURB * torch.randn_like(p))
    return m


@pytest.fixture(scope="module")
def model():
    return _model()


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
    m = _model(cfg)
    man = export_graphs(m, str(tmp_path / "m"))
    for name in GRAPH_NAMES:
        assert set(man["graphs"][name]["inputs"]) <= set(GRAPH_IO[name][0])
        assert man["export_check_max_rel"][name] <= 1e-4
    s = synthetic_series(m.cfg.stream_window + 60, seed=7)
    ref = StreamingMayak(m, LAT, LON, ELEV)
    rt = GraphRuntime(OnnxBackend(str(tmp_path / "m")), m.cfg, LAT, LON, ELEV)
    (err,) = _feed_compare(ref, [rt], s, every=150)
    assert err <= ATOL_ONNX, f"{ablation}: ONNX расходится с эталоном: {err:.2e}"


def test_manifest_contract(model, tmp_path):
    from mayak.baselines.statistical import ZQ
    from mayak.data.qc import PHYS
    man = export_graphs(model, str(tmp_path / "m"), conformal=SHIFT, aci=ACI)
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
    assert man["format"] == GRAPH_FORMAT == 6
    assert set(man["calibration"]) == {"conformal", "aci", "lead_bins", "history_bins"}
    assert all(set(g) == {"file", "inputs", "outputs", "shapes_in", "shapes_out"}
               for g in man["graphs"].values())
    assert RuntimeConfig.from_dict(man["runtime"]) == RuntimeConfig()
    from mayak.constants import HISTORY_BINS
    from mayak.metrics import LEAD_BINS, conformal_table
    assert man["calibration"]["lead_bins"] == [list(b) for b in LEAD_BINS]
    assert man["calibration"]["history_bins"] == [[b[0], b[1]] for b in HISTORY_BINS]
    table = np.fromfile(tmp_path / "m" / "conformal.f32", "<f4").reshape(
        len(HISTORY_BINS), cfg.horizon, -1)
    for k, (lo, hi, _name) in enumerate(HISTORY_BINS):
        for L in (lo, hi):
            np.testing.assert_array_equal(table[k], conformal_table(SHIFT, L, cfg.horizon))
