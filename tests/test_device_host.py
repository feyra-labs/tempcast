"""Тесты хоста устройства на Python.

Что проверяется:
* откат - климатология своей точки: в Якутии в январе медиана отката в пределах
  градуса от климат-поля этой точки, а не прежние десять градусов для любой точки;
* таблица климатологии графа старта совпадает с климат-полем модели на любом часе, в
  том числе в последние сутки високосного года;
* прогноз сразу после перезапуска без новых наблюдений и прогноз до первого
  наблюдения по часам устройства; сам прогноз состояние не меняет;
* свежее состояние выбирается по содержимому, неразборный файл идёт в конец, запись
  чередует файлы и не оставляет временных;
* сбой графа старта - ошибка конфигурации: исключение в рантайме и код возврата 1 в
  командной строке;
* командная строка говорит построчным протоколом и восстанавливает состояние.
"""
import io
import json
import os
import sys

import numpy as np
import pytest
import torch

from mayak.astro import astro_features
from mayak.runtime import golden as G
from mayak.runtime.equivalence import synthetic_series
from mayak.runtime.host import STATE_FILES, Host, StateStore
from mayak.runtime.streaming import StreamingMayak
from mayak.timeaxis import hour_of_year, to_utc_hour, window_calendar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_MODEL = os.path.join(ROOT, G.DEFAULT_DIR, "model")
AMS = (52.37, 4.9, -2.0)
YAKUTSK = (62.03, 129.73, 100.0)
YAKUTSK_JANUARY = -38.0


def _hour(stamp):
    return int(to_utc_hour(np.datetime64(stamp, "s")))


@pytest.fixture(scope="module")
def model():
    return G.golden_model()


@pytest.fixture(scope="module")
def cold_model():
    """Модель эталона, поле которой всюду около минус 38 градусов.

    Так проверка отката отличает климатологию точки от прежних десяти градусов.
    """
    m = G.golden_model()
    with torch.no_grad():
        m.field.head.bias[0] += YAKUTSK_JANUARY
    return m


class _NanIssue:
    """Исполнитель графов, у которого выпуск возвращает нечисловые квантили."""

    def __init__(self, backend):
        self.inner = backend

    def run(self, name, *args):
        out = self.inner.run(name, *args)
        if name == "issue":
            out = [np.full_like(out[0], np.nan)]
        return out


def _field(m, site, hours):
    """Среднее климат-поля с паспортом холодного старта на заданных часах."""
    lat, lon, elev = (torch.tensor([[v]], dtype=torch.float32) for v in site)
    doy, hr = window_calendar(0, np.asarray(hours, np.int64))
    with torch.no_grad():
        loc = m.loc(lat[:, 0], lon[:, 0], elev[:, 0])
        z0 = m.passport_from_rows(loc, loc.new_zeros(1, m.cfg.max_history, 4))
        astro = astro_features(torch.from_numpy(doy)[None], torch.from_numpy(hr)[None], lat, lon)
        mu, _sig, _ = m.field.evaluate(m.field.coefficients(loc, z0), astro)
    return mu[0].numpy()


def _feed(rt, series, k0, k1):
    for k in range(k0, k1):
        obs = [float(series["x"][k, j]) if series["m"][k, j] > 0 else None for j in range(3)]
        rt.step(*obs, series["t0"] + k)
    return rt


def test_fallback_is_point_climatology_in_yakutia(cold_model):
    rt = StreamingMayak(cold_model, *YAKUTSK)
    rt.b = _NanIssue(rt.b)
    now = _hour("2025-01-15T06")
    q, mu, fallback = rt.safe_forecast(now)
    assert fallback and rt.fallbacks == 1
    field = _field(cold_model, YAKUTSK, now + 1 + np.arange(rt.horizon))
    assert np.abs(mu - field).max() <= 1.0, "откат - климат-поле точки"
    assert np.abs(mu - 10.0).min() > 20.0, "откат не прежние 10 градусов"
    assert np.all(np.diff(q, axis=-1) > 0) and np.array_equal(q[:, 3], mu)


def test_fallback_after_steps_and_on_model_error(model):
    s = synthetic_series(60, seed=3)
    rt = _feed(StreamingMayak(model, *AMS), s, 0, 60)
    good_q, _mu, fb = rt.safe_forecast()
    assert not fb
    rt.b = _NanIssue(rt.b)
    q, mu, fb = rt.safe_forecast()
    assert fb
    np.testing.assert_array_equal(mu, rt.climatology_forecast(rt.last_hour)[1])
    assert not np.array_equal(q, good_q)
    with pytest.raises(FloatingPointError):
        rt.forecast()


def test_climatology_table_matches_field(model):
    rt = StreamingMayak(model, *AMS)
    hours = np.array([_hour("2024-12-31T23"), _hour("2025-12-31T23"), _hour("2024-02-29T12"),
                      _hour("2026-07-04T05"), _hour("2100-03-01T00")])
    assert hour_of_year(hours[:2]).tolist() == [8783, 8759]
    np.testing.assert_allclose(rt.clim_mu[hour_of_year(hours)], _field(model, AMS, hours),
                               atol=1e-5)
    assert rt.clim_mu.shape == rt.clim_sig.shape == (8784,)
    assert (rt.clim_sig > 0).all()


def test_init_failure_is_configuration_error(model):
    from mayak.runtime.graphs import GraphRuntime, TorchBackend

    class BadInit:
        def __init__(self):
            self.inner = TorchBackend(model)

        def run(self, name, *args):
            out = self.inner.run(name, *args)
            if name == "init":
                out[-1] = np.full_like(out[-1], np.nan)
            return out

    with pytest.raises(RuntimeError, match="конфигурации"):
        GraphRuntime(BadInit(), model.cfg, *AMS)


def test_forecast_before_first_obs_uses_device_clock(model):
    now = _hour("2024-11-03T09")
    rt = StreamingMayak(model, *AMS)
    host = Host(rt, clock=lambda: now * 3600 + 1234)
    reply = host.handle("forecast")
    assert reply["after_unix_hour"] == now and not reply["fallback"]
    assert np.isfinite(np.asarray(reply["q"])).all()
    assert rt.last_hour is None and rt.filled == 0, "прогноз не меняет состояние"
    assert host.handle(f"obs {now * 3600} 7 1012.3 80")["ok"], "час выпуска ещё можно принять"
    later = host.handle(f"forecast {(now + 50) * 3600}")
    assert later["after_unix_hour"] == now, "после шага момент выпуска - последний шаг"


def test_forecast_right_after_restart_without_obs(model, tmp_path):
    s = synthetic_series(200, seed=21)
    a = Host(StreamingMayak(model, *AMS), StateStore(tmp_path))
    for k in range(200):
        obs = [f"{s['x'][k, j]:.3f}" if s["m"][k, j] > 0 else "-" for j in range(3)]
        assert a.handle(f"obs {(s['t0'] + k) * 3600} " + " ".join(obs))["ok"]
    live = a.handle("forecast")
    b = Host(StreamingMayak(model, *AMS), StateStore(tmp_path), clock=lambda: 0)
    assert b.restore() is not None
    back = b.handle("forecast")
    assert back["after_unix_hour"] == live["after_unix_hour"] == s["t0"] + 199
    assert not back["fallback"]
    assert np.abs(np.asarray(back["q"]) - np.asarray(live["q"])).max() <= 5e-4


def test_idle_between_processes_is_filled(model, tmp_path):
    s = synthetic_series(120, seed=22, p_valid=1.0)
    a = _feed(StreamingMayak(model, *AMS), s, 0, 100)
    ref = _feed(StreamingMayak(model, *AMS), s, 0, 100)
    StateStore(tmp_path).save(a.serialize())
    host = Host(StreamingMayak(model, *AMS), StateStore(tmp_path))
    host.restore()
    txt = [f"{float(v):.4f}" for v in s["x"][110]]
    assert host.handle(f"obs {(s['t0'] + 110) * 3600} " + " ".join(txt))["ok"]
    for k in range(100, 110):
        ref.step(None, None, None, s["t0"] + k)
    ref.step(*(float(v) for v in txt), s["t0"] + 110)
    assert host.rt.idle_hours == 10
    assert host.rt.serialize() == ref.serialize()


def _state(model, n, idle=None):
    rt = _feed(StreamingMayak(model, *AMS), synthetic_series(n, seed=23), 0, n)
    if idle is not None:
        rt.step(5.0, 1001.0, 60.0, rt.last_hour + idle)
    return rt.serialize(), rt.last_hour


def test_store_orders_by_content_not_mtime(model, tmp_path):
    W = model.cfg.stream_window
    old, _ = _state(model, W + 10)
    new, new_last = _state(model, W + 10, idle=W + 20)
    store = StateStore(tmp_path)
    a, b = (os.path.join(tmp_path, f) for f in STATE_FILES)
    for path, raw, t in ((a, old, 2_000_000_000), (b, new, 1_000_000_000)):
        with open(path, "wb") as fh:
            fh.write(raw)
        os.utime(path, (t, t))
    assert [os.path.basename(f) for f, _ in store.ordered(W)] == ["state_b.bin", "state_a.bin"]
    with open(a, "wb") as fh:
        fh.write(b"MYK\x04 not a state")
    os.utime(a, (3_000_000_000, 3_000_000_000))
    assert [os.path.basename(f) for f, _ in store.ordered(W)] == ["state_b.bin", "state_a.bin"]
    host = Host(StreamingMayak(model, *AMS), store)
    assert os.path.basename(host.restore()) == "state_b.bin"
    assert host.rt.last_hour == new_last
    written = host.handle(f"obs {(new_last + 1) * 3600} 6 1001 60")
    assert written["ok"]
    with open(a, "rb") as fh:
        assert fh.read() == host.rt.serialize(), "запись идёт в невосстановленный файл"
    with open(b, "rb") as fh:
        assert fh.read() == new
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_empty_state_is_older_than_any_step(model, tmp_path):
    W = model.cfg.stream_window
    empty = StreamingMayak(model, *AMS).serialize()
    stepped, _ = _state(model, 5)
    store = StateStore(tmp_path)
    for name, raw in (("state_a.bin", stepped), ("state_b.bin", empty)):
        with open(os.path.join(tmp_path, name), "wb") as fh:
            fh.write(raw)
    assert os.path.basename(store.ordered(W)[0][0]) == "state_a.bin"


def test_protocol_errors_do_not_stop_the_host(model):
    host = Host(StreamingMayak(model, *AMS), clock=lambda: 0)
    assert "error" in host.handle("obs 1800 1 2 3")
    assert "error" in host.handle("obs 3600 1 x 3")
    assert "error" in host.handle("predict")
    assert host.handle("   ") is None
    assert host.handle("obs 7200 1 1000 50")["codes"] == [0, 0, 0]
    assert "error" in host.handle("obs 7200 1 1000 50")
    st = host.handle("status")
    assert st["last_unix_hour"] == 2 and st["fallbacks"] == 0 and st["state_bytes"] == 3224


def test_cli_speaks_protocol_and_restores(tmp_path, monkeypatch, capsys):
    from mayak.runtime import run_inference
    t0 = _hour("2024-05-01T00") * 3600
    lines = [f"obs {t0 + 3600 * k} {10 + k % 3} 1012.{k % 10} {70 + k % 5}" for k in range(30)]
    args = ["--model", GOLDEN_MODEL, "--lat", "52.37", "--lon", "4.9", "--elev", "-2",
            "--state-dir", str(tmp_path), "--aci"]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines + ["forecast", "status"])))
    assert run_inference.main(args) == 0
    out = [json.loads(v) for v in capsys.readouterr().out.splitlines()]
    assert all(r.get("ok") for r in out[:30])
    assert out[30]["after_unix_hour"] == t0 // 3600 + 29 and not out[30]["fallback"]
    assert out[31]["conformal"] and out[31]["filled"] == 30
    monkeypatch.setattr(sys, "stdin", io.StringIO("forecast\n"))
    assert run_inference.main(args) == 0
    again = json.loads(capsys.readouterr().out.splitlines()[0])
    assert again["after_unix_hour"] == out[30]["after_unix_hour"]
    np.testing.assert_allclose(again["q"], out[30]["q"], atol=5e-4)


def test_cli_exits_with_code_on_bad_model(tmp_path, capsys):
    from mayak.runtime import run_inference
    code = run_inference.main(["--model", str(tmp_path / "нет"), "--lat", "1", "--lon", "2",
                               "--state-dir", str(tmp_path)])
    assert code == 1
    assert "mayak-rt:" in capsys.readouterr().err
