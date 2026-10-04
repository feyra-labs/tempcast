"""Тесты: устройство и экспорт.

* ряд, поданный устройству час за часом через ONNX-бэкенд, даёт те же входы и квантили,
  что проход модели на окне оценки с тем же моментом выпуска - на модели по умолчанию и
  на каждой абляции;
* круговая сериализация состояния: перезапуск продолжает непрерывный прогон, битое
  состояние отвергается;
* смена точки: уточнение сохраняет окно и калибровку, перенос - холодный старт;
* простой заполняется пустыми часами, простой не короче окна - холодный старт;
* нечисловой выход графа прогноза - откат к климатологии точки;
* манифест согласован с конфигом, QC, форматом состояния и порогами смены точки.
"""
import os

import numpy as np
import pytest
import torch

from mayak.config import ABLATION_NAMES, Ablations, ModelConfig
from mayak.metrics import ACIParams
from mayak.runtime.device import (FORECAST_INPUTS, STATE_HEADER, Device, mask_bytes,
                                  state_nbytes)
from mayak.runtime.graphs import (GRAPH_IO, GRAPH_NAMES, TorchBackend, eval_inputs, eval_set,
                                  export_graphs, feed, runtime_from_export, synthetic_series)
from mayak.runtime.host import Host

LAT, LON, ELEV = 52.37, 4.9, 0.0
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


def _device(model, lat=LAT, lon=LON, elev=ELEV, **kw):
    return Device(TorchBackend(model), model.cfg, lat, lon, elev, **kw)


@pytest.mark.parametrize("ablation", [None, *ABLATION_NAMES])
def test_device_equals_eval_window(ablation, tmp_path):
    cfg = ModelConfig(ablations=Ablations(**{ablation: True})) if ablation else None
    m = _model(cfg)
    out = str(tmp_path / "m")
    man = export_graphs(m, out)
    assert sorted(os.listdir(out)) == ["climatology.onnx", "forecast.onnx", "manifest.json"]
    for name in GRAPH_NAMES:
        assert set(man["graphs"][name]["inputs"]) <= set(GRAPH_IO[name][0])
        assert man["export_check_max_rel"][name] <= 1e-4
    L, Hh = m.cfg.max_history, m.cfg.horizon
    ends = [0, 1, 37, L - 1, L, m.cfg.device_window + 29]
    s = synthetic_series(ends[-1] + Hh, seed=7)
    ds = eval_set(s, ends, LAT, LON, ELEV)
    dev = runtime_from_export(out, LAT, LON, ELEV)
    done = 0
    for i, end in enumerate(ends):
        feed(dev, s, done, end)
        done = end
        now = s["t0"] + end - 1
        ref = eval_inputs(ds, i)
        got = dev.model_inputs(now)
        for k in FORECAST_INPUTS:
            np.testing.assert_array_equal(got[k], ref[k], err_msg=f"{ablation}, час {end}: {k}")
        with torch.no_grad():
            q_ref = m({k: torch.from_numpy(v) for k, v in ref.items()})["q"][0].numpy()
        err = float(np.abs(dev.raw_forecast(now) - q_ref).max())
        assert err <= ATOL_ONNX, f"{ablation}, час {end}: устройство и оценка, {err:.2e}"


def test_state_roundtrip(model):
    s = synthetic_series(460, seed=14, p_valid=1.0)
    s["x"][250:330, 0] = 7.0
    s["x"][250:330, 2] = 64.0
    s["x"][299, 2] = -5.0
    s["x"][298, 1] = 7000.0
    cut = 300
    live = feed(_device(model, aci=ACI), s, 0, cut)
    live.reset_calibration(0.2)
    raw = live.serialize()
    assert len(raw) == live.state_nbytes == 3236
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    assert int(hdr["last_hour"]) == s["t0"] + cut - 1 and int(hdr["filled"]) == cut
    assert (float(hdr["lat"]), float(hdr["lon"])) == (np.float32(LAT), np.float32(LON))
    back = _device(model, aci=ACI)
    back.load_state(raw)
    assert back.serialize() == raw
    assert back.loaded_site == (np.float32(LAT), np.float32(LON), np.float32(ELEV))
    assert back.site_change == "same" and back.theta == pytest.approx((0.2,) * 4)
    feed(live, s, cut, 460)
    feed(back, s, cut, 460)
    assert back.serialize() == live.serialize()
    pos = live._hours(160) % live.window
    assert live.valid[pos, 0].min() == 0, "в ряду должно быть залипание температуры"
    np.testing.assert_array_equal(back.raw_forecast(), live.raw_forecast())

    empty = _device(model)
    empty.load_state(_device(model).serialize())
    assert empty.last_hour is None
    with pytest.raises(ValueError, match="заголовка"):
        empty.load_state(bytes(len(raw)))
    bad = bytearray(raw)
    bad[3] = 3
    with pytest.raises(ValueError, match="версия"):
        empty.load_state(bytes(bad))
    with pytest.raises(ValueError, match="конфигу"):
        empty.load_state(raw[:-1])
    h = hdr.copy()
    h["filled"] = 5000
    with pytest.raises(ValueError, match="заголовок"):
        empty.load_state(h.tobytes() + raw[STATE_HEADER.itemsize:])
    h = hdr.copy()
    h["aci_theta"] = np.nan
    with pytest.raises(ValueError, match="калибровки"):
        empty.load_state(h.tobytes() + raw[STATE_HEADER.itemsize:])
    W = model.cfg.device_window
    body = bytearray(raw[STATE_HEADER.itemsize:])
    body[4 * W:4 * W + mask_bytes(W)] = bytes(mask_bytes(W))
    with pytest.raises(ValueError, match="годный час без значения"):
        empty.load_state(raw[:STATE_HEADER.itemsize] + bytes(body))
    assert empty.last_hour is None


def test_site_change(model):
    s = synthetic_series(200, seed=3)
    live = feed(_device(model, aci=ACI), s, 0, 200)
    live.reset_calibration(0.3)
    raw = live.serialize()

    refined = _device(model, LAT + 0.1, LON - 0.1, ELEV + 20.0, aci=ACI)
    refined.load_state(raw)
    assert refined.site_change == "refined"
    assert refined.filled == 200 and refined.last_hour == live.last_hour
    assert refined.theta == pytest.approx((0.3,) * 4)
    inp = refined.model_inputs()
    np.testing.assert_array_equal(inp["x_hist"], live.model_inputs()["x_hist"])
    assert float(inp["lat"][0]) == np.float32(LAT + 0.1)
    assert refined.qc.elev == ELEV + 20.0

    moved = _device(model, LAT + 5.0, LON, ELEV, aci=ACI)
    moved.load_state(raw)
    assert moved.site_change == "moved"
    assert moved.filled == 0 and moved.last_hour == live.last_hour
    assert moved.theta == (0.0,) * 4
    assert not moved.model_inputs()["mask_hist"].any()
    moved.step(8.0, 1003.0, 71.0, live.last_hour + 1)
    assert moved.filled == 1


def test_idle_and_cold_start(model):
    s = synthetic_series(200, seed=15)
    a = feed(_device(model), s, 0, 120)
    b = feed(_device(model), s, 0, 120)
    for k in range(120, 127):
        b.step(None, None, None, s["t0"] + k)
    feed(a, s, 127, 200)
    feed(b, s, 127, 200)
    assert a.idle_hours == 7 and b.idle_hours == 0
    assert a.serialize() == b.serialize()

    p = ACIParams()
    c = feed(_device(model, aci=p), s, 0, 100)
    c.reset_calibration(0.3)
    later = s["t0"] + 100 + model.cfg.device_window + 3
    c.step(8.0, 1003.0, 71.0, later)
    fresh = _device(model, aci=p)
    fresh.step(8.0, 1003.0, 71.0, later)
    assert c.filled == 1 and c.theta == pytest.approx((0.3,) * 4)
    for k, v in fresh.model_inputs().items():
        np.testing.assert_array_equal(c.model_inputs()[k], v)


class _NanForecast(TorchBackend):
    """Исполнитель, у которого граф прогноза выдаёт NaN."""

    def run(self, name, feed):
        out = super().run(name, feed)
        return [np.full_like(o, np.nan) for o in out] if name == "forecast" else out


def test_fallback_on_non_finite_output(model):
    dev = Device(_NanForecast(model), model.cfg, LAT, LON, ELEV)
    host = Host(dev)
    t0 = 1_000_000
    reply = host.handle(f"forecast {t0 * 3600 + 1800}")
    assert reply["fallback"] and reply["after_unix_hour"] == t0
    q, mu = dev.climatology_forecast(t0)
    np.testing.assert_array_equal(np.float32(reply["q"]), q)
    assert host.handle(f"obs {(t0 + 1) * 3600} 8 1003 71")["ok"]
    reply = host.handle("forecast")
    assert reply["fallback"] and reply["after_unix_hour"] == t0 + 1
    assert dev.fallbacks == 2 and host.handle("status")["fallbacks"] == 2


def test_manifest_contract(model, tmp_path):
    from mayak.baselines.statistical import ZQ
    from mayak.config import RuntimeConfig
    from mayak.constants import HISTORY_BINS
    from mayak.data.qc import PHYS
    from mayak.metrics import LEAD_BINS, conformal_table
    from mayak.runtime.graphs import GRAPH_FORMAT
    out = tmp_path / "m"
    man = export_graphs(model, str(out), conformal=SHIFT, aci=ACI)
    assert sorted(os.listdir(out)) == ["climatology.onnx", "conformal.f32", "forecast.onnx",
                                       "manifest.json"]
    cfg, d = model.cfg, man["dims"]
    assert d == dict(horizon=cfg.horizon, n_quantiles=cfg.n_quantiles, history=cfg.max_history,
                     window=cfg.device_window, hours_of_year=8784)
    assert man["graphs"]["forecast"]["outputs"] == ["q"]
    assert man["graphs"]["climatology"]["outputs"] == ["clim_mu", "clim_sig"]
    assert man["state"]["nbytes"] == state_nbytes(cfg) == 3236
    assert man["state"]["nbytes"] == _device(model).state_nbytes
    assert {c: tuple(v) for c, v in man["phys"].items()} == {c: PHYS[c] for c in ("T", "P", "RH")}
    np.testing.assert_array_equal(np.float32(man["zq"]), ZQ)
    assert ModelConfig.from_dict(man["model_config"]) == cfg
    assert man["format"] == GRAPH_FORMAT == 7
    assert set(man["calibration"]) == {"conformal", "aci", "lead_bins", "history_bins"}
    assert all(set(g) == {"file", "inputs", "outputs", "shapes_in", "shapes_out"}
               for g in man["graphs"].values())
    assert RuntimeConfig.from_dict(man["runtime"]) == RuntimeConfig()
    assert man["calibration"]["lead_bins"] == [list(b) for b in LEAD_BINS]
    assert man["calibration"]["history_bins"] == [[b[0], b[1]] for b in HISTORY_BINS]
    table = np.fromfile(out / "conformal.f32", "<f4").reshape(len(HISTORY_BINS), cfg.horizon, -1)
    for k, (lo, hi, _name) in enumerate(HISTORY_BINS):
        for L in (lo, hi):
            np.testing.assert_array_equal(table[k], conformal_table(SHIFT, L, cfg.horizon))
    dev = runtime_from_export(str(out), LAT, LON, ELEV, aci=True)
    np.testing.assert_allclose(dev.conformal, SHIFT, rtol=0, atol=1e-7)
    assert dev.aci == ACI
    assert not model.training
