"""Эталонные векторы компилируемого рантайма.

Эталон - потоковый рантайм на PyTorch, калибровка и календарь проекта. Векторы
генерируются на детерминированной модели: сид и шум на всех параметрах, чтобы нулевые
инициализации голов, модуляции поля и подстройки мод не прятали ошибки. Векторы
записываются в каталог эталона и коммитятся, рантайм на Rust сверяется с ними.

Состав каталога:
* model/        - графы ONNX, манифест, конформная таблица;
* golden.json   - сценарии из событий шага и выпуска, календарь, калибровка;
* golden.f32    - ожидаемые квантили и входы калибровки, float32 little-endian;
* state_*.bin   - состояния, записанные эталоном.

Сценарии:
* cold_aci  - холодный старт, конформная таблица и адаптивная калибровка, переход через
              Новый год, дробные наблюдения, значения у границ физических диапазонов до и
              после записи, половины между целыми, NaN, пропуски, простой в несколько
              часов;
* restart   - старт из состояния эталона, записанного после полного окна: хвост и край
              истории заполнены, множитель калибровки не нулевой;
* extremes  - полярная точка, долгие пустые часы, значения ровно на границах диапазонов,
              простой в час и простой длиннее окна, без калибровки.
"""
import json
import os

import numpy as np
import torch

from mayak.metrics import ACIParams, aci_run, aci_score, calibrate_forecast
from mayak.runtime.equivalence import synthetic_series
from mayak.runtime.streaming import StreamingMayak
from mayak.timeaxis import to_utc_hour, window_calendar

GOLDEN_FORMAT = 2
GOLDEN_SEED = 1414
GOLDEN_PERTURB = 0.05
GOLDEN_ACI = ACIParams(target=0.10, gamma=0.05, max_factor=4.0)
GOLDEN_SHIFT = (np.array([-0.3, -0.2, -0.08, 0.0, 0.08, 0.2, 0.3], np.float32)[None, :]
                * np.array([1.0, 1.5, 2.0, 2.5], np.float32)[:, None]
                + np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1], np.float32))
MIN_ACI_MARGIN = 1e-3
Q_ATOL = 5e-4
FRESH_ATOL = 2e-4
DEFAULT_DIR = os.path.join("tests", "data", "runtime_golden")


def golden_model(cfg=None):
    """Детерминированная модель эталона: сид + шум на всех параметрах."""
    from mayak.model import MAYAK
    torch.manual_seed(GOLDEN_SEED)
    m = MAYAK(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(GOLDEN_PERTURB * torch.randn_like(p))
    return m


def _f(x):
    """float32 → float Python (repr JSON однозначно возвращает то же float32)."""
    return float(np.float32(x))


def _enc_obs(v):
    """Наблюдение → JSON: число, null (нет данных) или "nan" (NaN от датчика)."""
    if v is None:
        return None
    return "nan" if v != v else float(v)


def dec_obs(v):
    return float("nan") if v == "nan" else v


def _enc_score(v):
    return "nan" if np.isnan(v) else ("inf" if np.isinf(v) else float(v))


def _obs(series, k):
    return [_f(series["x"][k, j]) if series["m"][k, j] > 0 else None for j in range(3)]


def _hour(stamp):
    return int(to_utc_hour(np.datetime64(stamp, "s")))


class _Recorder:
    """Прогон эталона по событиям с записью ожидаемых значений."""

    def __init__(self, stream, blob):
        self.s, self.blob = stream, blob
        self.events = []
        self.margins = []

    def step(self, obs, hour):
        s = self.s
        p = s._pending
        if s.aci is not None and p is not None and obs[0] is not None:
            k = hour - p["first"]
            if 0 <= k < len(p["q"]) and k > p["last"]:
                sc = float(aci_score(float(np.float32(obs[0])), p["q"][k], s.aci.interval))
                if np.isfinite(sc):
                    self.margins.append(abs(sc - np.exp(s.theta)))
        s.step(*obs, hour)
        self.events.append(dict(op="step", obs=[_enc_obs(v) for v in obs], hour=int(hour)))

    def forecast(self, record=True):
        """Выпуск; без записи квантилей выпуск нужен только как обратная связь
        калибровки, множитель пишется всегда."""
        q, _mu = self.s.forecast()
        self.events.append(dict(op="forecast", q=self.blob.put(q) if record else None,
                                theta=_f(self.s.theta)))

    def final(self):
        s = self.s
        return dict(theta=_f(s.theta), aci_updates=s.aci_updates, aci_misses=s.aci_misses,
                    filled=s.filled, last_hour=s.last_hour, idle_hours=s.idle_hours)


class _Blob:
    def __init__(self):
        self.parts, self.n = [], 0

    def put(self, a):
        a = np.ascontiguousarray(a, "<f4").ravel()
        off = self.n
        self.parts.append(a)
        self.n += a.size
        return dict(offset=off, len=int(a.size))

    def array(self):
        return np.concatenate(self.parts) if self.parts else np.zeros(0, "<f4")


def scenario_cold_aci(model, blob):
    lat, lon, elev = 52.37, 4.9, -2.0
    st = StreamingMayak(model, lat, lon, elev, conformal=GOLDEN_SHIFT, aci=GOLDEN_ACI)
    rec = _Recorder(st, blob)
    s = synthetic_series(330, seed=11, t0=_hour("2021-12-27T20"))
    bad = {40: (75.0, None, None), 41: (None, None, -5.0), 42: (None, 250.0, None),
           43: (float("nan"), None, None), 44: (60.6, 1100.06, 100.6),
           45: (60.4, 1100.04, 100.4), 46: (-0.5, 1013.25, 12.5), 47: (-200.0, 7000.0, 300.0)}
    skip = set(range(200, 205))
    for k in range(330):
        if k in skip:
            continue
        obs = _obs(s, k)
        if k in bad:
            obs = [b if b is not None else o for b, o in zip(bad[k], obs)]
        rec.step(obs, s["t0"] + k)
        if k % 6 == 5:
            rec.forecast(record=k % 24 == 23)
    return dict(name="cold_aci", lat=lat, lon=lon, elev=elev, conformal=True, aci=True,
                init_state=None, events=rec.events, final=rec.final()), rec.margins


def scenario_restart(model, blob, out_dir):
    lat, lon, elev = -33.87, 151.21, 58.0
    pre_n = model.cfg.stream_window + 88
    s = synthetic_series(pre_n + 100, seed=29, t0=_hour("2023-03-24T05"))
    a = StreamingMayak(model, lat, lon, elev, conformal=GOLDEN_SHIFT, aci=GOLDEN_ACI)
    pre = _Recorder(a, _Blob())
    for k in range(pre_n):
        pre.step(_obs(s, k), s["t0"] + k)
        if k % 8 == 7:
            pre.forecast()
    raw = a.serialize()
    with open(os.path.join(out_dir, "state_restart.bin"), "wb") as fh:
        fh.write(raw)
    b = StreamingMayak(model, lat, lon, elev, conformal=GOLDEN_SHIFT, aci=GOLDEN_ACI)
    b.load_state(raw)
    rec = _Recorder(b, blob)
    rec.forecast()
    for k in range(pre_n, len(s["x"])):
        rec.step(_obs(s, k), s["t0"] + k)
        if k % 12 == 11:
            rec.forecast()
    end = b.serialize()
    with open(os.path.join(out_dir, "state_restart_end.bin"), "wb") as fh:
        fh.write(end)
    fin = rec.final()
    fin["state"] = "state_restart_end.bin"
    return dict(name="restart", lat=lat, lon=lon, elev=elev, conformal=True, aci=True,
                init_state="state_restart.bin", events=rec.events,
                final=fin), pre.margins + rec.margins


def scenario_extremes(model, blob):
    lat, lon, elev = 78.22, 15.65, 2000.0
    st = StreamingMayak(model, lat, lon, elev)
    rec = _Recorder(st, blob)
    hour = _hour("2022-06-28T04")
    for _ in range(30):
        rec.step([None, None, None], hour)
        hour += 1
    rec.forecast()
    edges = [(-90.0, 300.0, 0.0), (60.0, 1100.0, 100.0), (-90.0, 1100.0, 100.0),
             (60.0, 300.0, 0.0)]
    for k in range(60):
        rec.step(list(edges[k % 4]), hour)
        hour += 1 if k != 30 else 2
    rec.forecast()
    rec.step([-5.0, 800.0, 55.0], hour)
    rec.forecast()
    hour += model.cfg.stream_window + 5
    rec.step([-4.0, 801.0, 56.0], hour)
    rec.forecast()
    return dict(name="extremes", lat=lat, lon=lon, elev=elev, conformal=False, aci=False,
                init_state=None, events=rec.events, final=rec.final())


def calendar_cases():
    """Часы от эпохи вокруг границ годов, включая 2000 (високосный) и 2100 (нет)."""
    from mayak.timeaxis import to_utc_hour
    pts = ["1970-01-01T00", "1999-12-31T23", "2000-02-28T12", "2000-12-31T23",
           "2023-12-31T20", "2024-02-28T22", "2024-12-31T23", "2025-06-15T11",
           "2100-02-28T23", "2100-12-31T23"]
    base = [int(to_utc_hour(np.datetime64(p, "s"))) for p in pts]
    hours = sorted({h + d for h in base for d in (-1, 0, 1, 2)})
    doy, hr = window_calendar(0, np.array(hours, np.int64))
    return dict(unix_hours=hours, doy=[_f(v) for v in doy], hour=[_f(v) for v in hr])


def calibration_cases(model, blob):
    """Калибровка как отдельный узел: calibrate_forecast и aci_run эталона."""
    rng = np.random.default_rng(GOLDEN_SEED)
    H = model.cfg.horizon
    cases = []
    for theta, conf in ((0.0, True), (0.37, True), (-0.52, True), (0.37, False)):
        q = np.sort(rng.normal(0.0, 3.0, (H, len(model.cfg.quantiles))), axis=-1)
        q[5, 2] = q[5, 4] + 0.5                          # немонотонный вход
        q = q.astype(np.float32)
        out, _mu = calibrate_forecast(q, GOLDEN_SHIFT if conf else None, _f(theta))
        cases.append(dict(q=blob.put(q), theta=_f(theta), conformal=conf,
                          expect=blob.put(out)))
    scores = rng.exponential(0.8, 400)
    scores[::37] = np.nan
    scores[5] = np.inf
    run = aci_run(scores, GOLDEN_ACI, theta0=0.1)
    aci = dict(scores=[_enc_score(v) for v in scores],
               theta0=0.1, theta_before=[_f(v) for v in run["theta"]],
               miss=[None if np.isnan(v) else bool(v) for v in run["miss"]],
               theta_end=_f(run["theta_end"]))
    ys = rng.normal(0.0, 4.0, 64)
    qs = np.sort(rng.normal(0.0, 3.0, (64, len(model.cfg.quantiles))), axis=-1).astype(np.float32)
    qs[3] = qs[3, 3]                                     # вырожденный интервал
    ys[7] = qs[7, 3]
    sc = aci_score(ys.astype(np.float32).astype(np.float64), qs, GOLDEN_ACI.interval)
    score = dict(y=[_f(v) for v in ys], q=blob.put(qs),
                 expect=[_enc_score(v) for v in sc])
    return dict(cases=cases, aci=aci, score=score)


def generate(out_dir=DEFAULT_DIR):
    """Полная генерация эталона → golden.json. Модель и графы - в out_dir/model."""
    from mayak.runtime.graphs import export_graphs
    os.makedirs(out_dir, exist_ok=True)
    model = golden_model()
    manifest = export_graphs(model, os.path.join(out_dir, "model"), conformal=GOLDEN_SHIFT,
                             aci=GOLDEN_ACI)
    manifest["provenance"].pop("created_utc", None)
    with open(os.path.join(out_dir, "model", "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    blob = _Blob()
    cold, m1 = scenario_cold_aci(model, blob)
    restart, m2 = scenario_restart(model, blob, out_dir)
    extremes = scenario_extremes(model, blob)
    margins = m1 + m2
    if margins and min(margins) < MIN_ACI_MARGIN:
        raise RuntimeError(f"обратная связь ACI эталона в {min(margins):.1e} от порога e^θ: "
                           f"решение о промахе неустойчиво к float32; смените сид")
    doc = dict(format=GOLDEN_FORMAT, seed=GOLDEN_SEED,
               tolerance=dict(q_abs=Q_ATOL, fresh_abs=FRESH_ATOL),
               aci_margin_min=float(min(margins)) if margins else None,
               scenarios=[cold, restart, extremes], calendar=calendar_cases(),
               calibration=calibration_cases(model, blob))
    blob.array().astype("<f4").tofile(os.path.join(out_dir, "golden.f32"))
    with open(os.path.join(out_dir, "golden.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    return doc


def load(out_dir=DEFAULT_DIR):
    with open(os.path.join(out_dir, "golden.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    blob = np.fromfile(os.path.join(out_dir, "golden.f32"), "<f4")
    return doc, blob


def take(blob, ref, shape=None):
    a = blob[ref["offset"]:ref["offset"] + ref["len"]]
    return a.reshape(shape) if shape is not None else a


def replay(doc, blob, runtime_factory, horizon):
    """Прогон сценариев эталона через любую реализацию.

    Args:
        doc: документ эталона.
        blob: массив ожидаемых значений.
        runtime_factory: по сценарию возвращает рантайм с методами ``step(T, P, RH,
            hour)`` и ``forecast()``; для сценария с начальным состоянием фабрика сама
            загружает состояние.
        horizon: горизонт модели.

    Returns:
        Словарь: имя сценария и наибольшее расхождение квантилей с эталоном.
    """
    out = {}
    for sc in doc["scenarios"]:
        rt = runtime_factory(sc)
        err = 0.0
        for ev in sc["events"]:
            if ev["op"] == "step":
                rt.step(*(dec_obs(v) for v in ev["obs"]), ev["hour"])
                continue
            q = rt.forecast()
            if ev["q"] is None:
                continue
            q = q[0] if isinstance(q, tuple) else q
            ref = take(blob, ev["q"], (horizon, -1))
            err = max(err, float(np.abs(np.asarray(q) - ref).max()))
        out[sc["name"]] = err
    return out


__all__ = ["DEFAULT_DIR", "FRESH_ATOL", "GOLDEN_ACI", "GOLDEN_SHIFT", "Q_ATOL", "generate",
           "dec_obs", "golden_model", "load", "replay", "take"]
