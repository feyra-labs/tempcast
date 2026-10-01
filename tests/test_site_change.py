"""Тесты смены координат и высоты прибора между перезапусками.

Что проверяется:
* правило сравнения точек: та же точка, уточнение в пределах порогов включительно,
  перенос; долгота по кратчайшей дуге через 180 градусов; нечисловая точка - перенос;
* уточнение на 0.3 градуса: выпуск после загрузки совпадает с рантаймом, который с
  самого начала работал с новыми координатами на тех же наблюдениях; множитель
  калибровки сохраняется, в логе и в сводке видно уточнение;
* перенос: окно пустое, множитель калибровки нулевой, момент последнего шага сохранён,
  выпуск и дальнейший поток совпадают с новым рантаймом на новой точке, а первое же
  наблюдение делает состояние на диске свежее файла старой точки;
* проверка давления на уровне моря после загрузки берёт высоту из текущих аргументов;
* пороги лежат в конфиге рантайма и в манифесте графов, рантайм на графах берёт их из
  манифеста, манифест прежнего формата не принимается.
"""
import dataclasses
import json
import logging
import math
import os

import numpy as np
import pytest
import yaml

from mayak.config import ConfigError, RuntimeConfig
from mayak.runtime import golden as G
from mayak.runtime.equivalence import synthetic_series
from mayak.runtime.graphs import export_graphs, runtime_from_export
from mayak.runtime.host import Host, StateStore
from mayak.runtime.site import (DEFAULT_RUNTIME_CONFIG, SITE_MOVED, SITE_REFINED, SITE_SAME,
                                load_runtime_config, lon_gap, site_change)
from mayak.runtime.streaming import StreamingMayak

AMS = (52.37, 4.9, -2.0)
AMS_FIXED = (52.67, 5.2, -2.0)
OSLO = (59.91, 10.75, 20.0)
ATOL_FORECAST = 5e-4
CFG = RuntimeConfig()


@pytest.fixture(scope="module")
def model():
    return G.golden_model()


def _feed(rt, s, k0, k1):
    for k in range(k0, k1):
        obs = [float(s["x"][k, j]) if s["m"][k, j] > 0 else None for j in range(3)]
        rt.step(*obs, s["t0"] + k)
    return rt


@pytest.mark.parametrize("old, new, kind", [
    (AMS, AMS, SITE_SAME),
    (AMS, AMS_FIXED, SITE_REFINED),
    ((52.0, 4.9, 10.0), (52.5, 4.9, 10.0), SITE_REFINED),
    ((52.0, 4.9, 10.0), (52.500004, 4.9, 10.0), SITE_MOVED),
    ((52.0, 4.9, 10.0), (52.0, 5.41, 10.0), SITE_MOVED),
    ((52.0, 4.9, 10.0), (52.0, 4.9, 110.0), SITE_REFINED),
    ((52.0, 4.9, 10.0), (52.0, 4.9, 110.5), SITE_MOVED),
    ((10.0, 179.9, 0.0), (10.0, -179.9, 0.0), SITE_REFINED),
    ((10.0, 180.0, 0.0), (10.0, -180.0, 0.0), SITE_SAME),
    ((math.nan, 4.9, 10.0), (52.0, 4.9, 10.0), SITE_MOVED),
    (AMS, OSLO, SITE_MOVED),
])
def test_site_change_rule(old, new, kind):
    assert site_change(old, new, CFG)[0] == kind


def test_longitude_gap_is_the_short_arc():
    assert lon_gap(179.9, -179.9) == pytest.approx(0.2)
    assert lon_gap(10.0, 370.0) == 0.0
    assert lon_gap(-170.0, 170.0) == pytest.approx(20.0)
    assert math.isnan(lon_gap(math.nan, 1.0))


def test_refined_site_equals_runtime_started_at_new_site(model, caplog):
    """Уточнение на 0.3 градуса.

    Всё модельное состояние пересчитывается для новой точки, выпуск совпадает с рантаймом, который с
    начала работал с новыми координатами.
    """
    s = synthetic_series(900, seed=81)
    cut = 760
    old = _feed(StreamingMayak(model, *AMS), s, 0, cut)
    old.reset_calibration(0.37)
    fresh = _feed(StreamingMayak(model, *AMS_FIXED), s, 0, cut)
    back = StreamingMayak(model, *AMS_FIXED)
    with caplog.at_level(logging.WARNING):
        back.load_state(old.serialize())
    assert back.site_change == SITE_REFINED and back.loaded_site == old.site
    assert "координаты уточнены" in caplog.text
    assert back.filled == fresh.filled and back.last_hour == fresh.last_hour
    assert back.theta == pytest.approx(0.37), "множитель калибровки прибора сохраняется"
    before = np.abs(old.raw_forecast() - fresh.raw_forecast()).max()
    assert before > 20 * ATOL_FORECAST, "сдвиг точки должен быть заметен в выпуске"
    done = cut
    for end in (cut, cut + 5, cut + 140):
        _feed(back, s, done, end)
        _feed(fresh, s, done, end)
        done = end
        err = np.abs(back.raw_forecast() - fresh.raw_forecast()).max()
        assert err < ATOL_FORECAST, f"час {end}: max|Δq| = {err:.3g}"
    assert np.array_equal(back.valid, fresh.valid)
    st = Host(back).status()
    assert st["site_change"] == SITE_REFINED and st["loaded_site"] == list(old.site)
    assert st["site"] == list(back.site)


def test_move_beyond_threshold_is_cold_start(model, caplog):
    s = synthetic_series(400, seed=83)
    old = _feed(StreamingMayak(model, *AMS, aci=G.GOLDEN_ACI), s, 0, 300)
    old.reset_calibration(0.4)
    back = StreamingMayak(model, *OSLO, aci=G.GOLDEN_ACI)
    with caplog.at_level(logging.WARNING):
        back.load_state(old.serialize())
    assert "прибор перенесён" in caplog.text
    assert back.site_change == SITE_MOVED
    assert back.filled == 0 and back.theta == 0.0
    assert not back.present.any() and not back.valid.any() and not back.raw.any()
    assert back.last_hour == old.last_hour, "время прибора от места не зависит"
    new = StreamingMayak(model, *OSLO, aci=G.GOLDEN_ACI)
    err = np.abs(back.raw_forecast() - new.raw_forecast(old.last_hour)).max()
    assert err < ATOL_FORECAST, f"выпуск сразу после переноса: max|Δq| = {err:.3g}"
    _feed(back, s, 300, 400)
    _feed(new, s, 300, 400)
    err = np.abs(back.raw_forecast() - new.raw_forecast()).max()
    assert err < ATOL_FORECAST, f"поток после переноса: max|Δq| = {err:.3g}"
    assert np.array_equal(back.valid, new.valid)
    st = Host(back).status()
    assert st["site_change"] == SITE_MOVED and st["filled"] == new.filled
    assert st["loaded_site"] == list(old.site) and st["site"] == list(back.site)


@pytest.mark.parametrize("axis, delta", [(0, 0.6), (1, -0.6), (2, 101.0)])
def test_each_axis_over_threshold_resets_window_and_theta(model, axis, delta):
    s = synthetic_series(100, seed=85)
    old = _feed(StreamingMayak(model, *AMS), s, 0, 100)
    old.reset_calibration(0.25)
    site = list(AMS)
    site[axis] += delta
    back = StreamingMayak(model, *site)
    back.load_state(old.serialize())
    assert back.site_change == SITE_MOVED
    assert back.filled == 0 and back.theta == 0.0 and not back.valid.any()


def test_state_without_steps_on_moved_site(model):
    old = StreamingMayak(model, *AMS)
    old.reset_calibration(0.3)
    back = StreamingMayak(model, *OSLO)
    back.load_state(old.serialize())
    assert back.site_change == SITE_MOVED and back.last_hour is None and back.theta == 0.0


def test_first_obs_after_move_outranks_old_site_file(model, tmp_path):
    """После переноса первое же наблюдение даёт состояние свежее файла старой точки.

    Поэтому следующий перезапуск не вернётся к чужой истории.
    """
    s = synthetic_series(60, seed=87)
    host = Host(StreamingMayak(model, *AMS), StateStore(tmp_path))
    for k in range(40):
        host.obs(str((s["t0"] + k) * 3600), "5", "1000", "70")
    host = Host(StreamingMayak(model, *OSLO), StateStore(tmp_path))
    host.restore()
    assert host.rt.site_change == SITE_MOVED
    host.obs(str((s["t0"] + 40) * 3600), "6", "1001", "71")
    again = Host(StreamingMayak(model, *OSLO), StateStore(tmp_path))
    again.restore()
    assert again.rt.site_change == SITE_SAME and again.rt.filled == 1


def test_new_hours_use_current_elevation(model):
    """После уточнения высоты проверка давления на уровне моря решает о новых часах по новой высоте.

    На 700 м давление 970 гПа станционное, на 790 м - уже нет.
    """
    s = synthetic_series(120, seed=89, p_valid=1.0)
    s["x"][:, 1] = 970.0 + 1.2 * np.sin(np.arange(120) / 30.0)
    low = _feed(StreamingMayak(model, 46.95, 7.45, 700.0), s, 0, 100)
    assert low.valid[low._hours(80) % low.window, 1].all(), "на 700 м давление годное"
    high = StreamingMayak(model, 46.95, 7.45, 790.0)
    high.load_state(low.serialize())
    assert high.site_change == SITE_REFINED
    same = StreamingMayak(model, 46.95, 7.45, 700.0)
    same.load_state(low.serialize())
    obs = [float(v) for v in s["x"][100]]
    assert same.step(*obs, s["t0"] + 100)[1] == 0
    assert high.step(*obs, s["t0"] + 100)[1] & 64, "давление на уровне моря по новой высоте"


def test_runtime_yaml_matches_dataclass():
    with open(DEFAULT_RUNTIME_CONFIG, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    assert set(d) == {f.name for f in dataclasses.fields(RuntimeConfig)}
    assert RuntimeConfig.from_dict(d) == RuntimeConfig() == load_runtime_config()
    for bad in (dict(site_max_dlat_deg=-1.0), dict(site_max_delev_m=math.inf),
                dict(site_max_dlon=1.0)):
        with pytest.raises(ConfigError):
            RuntimeConfig.from_dict(bad)


def test_thresholds_travel_in_manifest(model, tmp_path):
    """Экспорт пишет пороги в манифест, рантайм на графах берёт их оттуда.

    С порогом широты 0.1 градуса сдвиг на 0.3 - уже перенос.
    """
    tight = RuntimeConfig(site_max_dlat_deg=0.1)
    out = str(tmp_path / "m")
    man = export_graphs(model, out, runtime=tight)
    assert man["runtime"] == tight.to_dict()
    s = synthetic_series(50, seed=91)
    old = _feed(runtime_from_export(out, *AMS), s, 0, 50)
    back = runtime_from_export(out, *AMS_FIXED)
    assert back.runtime_cfg == tight
    back.load_state(old.serialize())
    assert back.site_change == SITE_MOVED and back.filled == 0
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["format"] = 3
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    with pytest.raises(ValueError, match="формат манифеста"):
        runtime_from_export(out, *AMS)


def test_golden_site_cases_are_fresh():
    """Сравнения точек в эталоне совпадают с правилом; их же проходит хост на Rust."""
    doc, _ = G.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 G.DEFAULT_DIR))
    assert doc["site"] == G.site_cases()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           G.DEFAULT_DIR, "model", "manifest.json"), encoding="utf-8") as fh:
        assert json.load(fh)["runtime"] == RuntimeConfig().to_dict()
