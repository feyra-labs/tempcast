"""Эталонные сценарии хоста устройства.

Эталон записывает хост на Python: строки протокола, ответы на них, перезапуски и
состояние на диске. Хост на Rust и хост на Python поверх графов экспорта проходят те же
сценарии и должны дать те же ответы: коды контроля качества, моменты выпуска, признаки
отката, множитель калибровки, сводку и квантили в пределах допуска. Так две реализации
сверяются через тот интерфейс, которым пользуется устройство.

Модель эталона детерминированная: сид и шум на всех параметрах, чтобы нулевые
инициализации голов, модуляции поля и подстройки мод не прятали ошибки.

Состав каталога:
* model/      - графы ONNX fp32 и int8, манифест, конформная таблица;
* model_nan/  - та же модель, но граф выпуска возвращает нечисловые квантили; остальные
                файлы берутся из соседнего каталога;
* golden.json - сценарии, календарь, калибровка, сравнения точек при смене координат;
* golden.f32  - ожидаемые квантили и входы калибровки, float32 little-endian;
* state_*.bin - состояния: начальные файлы сценариев и ожидаемые состояния.

События сценария:
* ``restart`` - новый процесс хоста с тем же каталогом состояния, при желании с новыми
  координатами; ожидается имя восстановленного файла или его отсутствие. Файл
  состояния другой точки тоже восстановлен: при переносе прибора из него берётся
  момент последнего шага;
* ``cmd``     - строка протокола и ожидаемый ответ;
* ``state``   - сверка состояния рантайма с файлом эталона до байта, кроме множителей
  калибровки, которые сверяются с допуском.

Конформная таблица эталона разбита по бинам лидов и длины истории, адаптивная
калибровка - по бинам лидов. Отдельные сценарии проверяют ежечасный выпуск, при котором
обратную связь получают все бины лидов, и старт из состояния прежней версии с одним
множителем калибровки.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from mayak.data.qc import station_pressure_expected
from mayak.data.recording import record_channel
from mayak.metrics import ACIParams, aci_run, aci_score, calibrate_forecast
from mayak.runtime.equivalence import synthetic_series
from mayak.config import RuntimeConfig
from mayak.runtime.host import Host, StateStore
from mayak.runtime.site import SITE_MOVED, SITE_REFINED, SITE_SAME, site_change, site_gap
from mayak.runtime.streaming import STATE_HEADER, STATE_HEADER_V4, StreamingMayak
from mayak.timeaxis import hour_of_year, to_utc_hour, window_calendar

GOLDEN_FORMAT = 5
GOLDEN_SEED = 1414
GOLDEN_PERTURB = 0.05
GOLDEN_ACI = ACIParams(target=0.10, gamma=0.05, max_factor=4.0)
GOLDEN_SHIFT = ((np.array([-0.3, -0.2, -0.08, 0.0, 0.08, 0.2, 0.3], np.float32)[None, :]
                 * np.array([1.0, 1.5, 2.0, 2.5], np.float32)[:, None]
                 + np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1], np.float32))[:, None, :]
                * np.array([1.6, 1.3, 1.1, 1.0], np.float32)[None, :, None])
V4_THETA = float(np.float32(0.123))
MIN_ACI_MARGIN = 1e-3
MAX_VARIANTS = 20
Q_ATOL = 5e-4
Q_ATOL_INT8 = 3e-2
FRESH_ATOL = 2e-4
THETA_ATOL = 1e-6
MTIME_BASE = 1_700_000_000
DEFAULT_DIR = os.path.join("tests", "data", "runtime_golden")
STATUS_KEYS = ("filled", "history_hours", "theta", "conformal", "aci_lead_bins", "aci_updates",
               "aci_misses", "idle_hours", "fallbacks", "state_bytes", "last_unix_hour",
               "memory_bytes", "site", "loaded_site", "site_change")
SCENARIOS = ("cold_aci", "restart", "restart_v4", "extremes", "long", "qc", "rounding", "sparse",
             "int8", "fallback", "no_obs", "site_shift", "relocation", "store_order",
             "hourly_aci")


def golden_model(cfg=None):
    """Детерминированная модель эталона: сид и шум на всех параметрах.

    Args:
        cfg: конфиг модели; None - уменьшенная модель эталона.

    Returns:
        Модель в режиме вывода.
    """
    from mayak.model import MAYAK
    torch.manual_seed(GOLDEN_SEED)
    m = MAYAK(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(GOLDEN_PERTURB * torch.randn_like(p))
    return m


def _f(x):
    """Число float32 как float Python: JSON возвращает то же float32."""
    return float(np.float32(x))


def _num(v):
    """Значение наблюдения в строке протокола."""
    if v is None:
        return "-"
    if v != v:
        return "nan"
    return repr(float(v))


def _hour(stamp):
    return int(to_utc_hour(np.datetime64(stamp, "s")))


def _obs(series, k):
    return [_f(series["x"][k, j]) if series["m"][k, j] > 0 else None for j in range(3)]


def _enc_score(v):
    return "nan" if np.isnan(v) else ("inf" if np.isinf(v) else float(v))


def _no_clock():
    raise RuntimeError("в эталоне часы устройства не используются: момент задаётся командой")


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

    def mark(self):
        return len(self.parts), self.n

    def rollback(self, mark):
        del self.parts[mark[0]:]
        self.n = mark[1]


def scenario_runtime(model, golden_dir, sc, site, onnx=False):
    """Рантайм сценария для хоста на Python.

    Args:
        model: модель эталона.
        golden_dir: каталог эталона.
        sc: сценарий.
        site: широта, долгота и высота.
        onnx: всегда брать графы экспорта; иначе сценарий fp32 на основной модели идёт
            через модель PyTorch.

    Returns:
        Потоковый рантайм.
    """
    from mayak.runtime.graphs import runtime_from_export
    if onnx or sc["model"] != "model" or sc["precision"] != "fp32":
        return runtime_from_export(os.path.join(golden_dir, sc["model"]), *site,
                                   precision=sc["precision"], conformal=sc["conformal"],
                                   aci=sc["aci"])
    return StreamingMayak(model, *site, conformal=GOLDEN_SHIFT if sc["conformal"] else None,
                          aci=GOLDEN_ACI if sc["aci"] else None)


def place_init_files(sc, golden_dir, state_dir):
    """Разложить начальные файлы состояния сценария с заданным порядком времени изменения.

    Args:
        sc: описание сценария.
        golden_dir: каталог эталонов.
        state_dir: каталог состояния хоста.
    """
    for f in sc["init_files"]:
        dst = os.path.join(state_dir, f["as"])
        shutil.copyfile(os.path.join(golden_dir, f["file"]), dst)
        t = MTIME_BASE + f["mtime"]
        os.utime(dst, (t, t))


class _Session:
    """Прогон сценария хостом на Python с записью ожидаемых ответов."""

    def __init__(self, sc, make, golden_dir, blob):
        self.sc, self.make, self.golden_dir, self.blob = sc, make, golden_dir, blob
        self.dir = tempfile.mkdtemp(prefix=f"golden_{sc['name']}_")
        place_init_files(sc, golden_dir, self.dir)
        self.site = (sc["lat"], sc["lon"], sc["elev"])
        self.host = None
        self.margins = []

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def restart(self, site=None):
        if site is not None:
            self.site = tuple(float(v) for v in site)
        self.host = Host(self.make(self.sc, self.site), StateStore(self.dir), clock=_no_clock)
        got = self.host.restore()
        got = None if got is None else os.path.basename(got)
        self.sc["events"].append(dict(op="restart", site=None if site is None else list(site),
                                      expect=dict(restored=got)))
        return got

    def _margin(self, value, hour):
        rt = self.host.rt
        if rt.aci is None or value is None or value != value:
            return
        y = float(record_channel(value, 0))
        for b, sc in rt.cal.scores(y, hour):
            if np.isfinite(sc):
                self.margins.append(abs(sc - np.exp(rt.theta[b])))

    def cmd(self, line, record=True):
        parts = line.split()
        if parts[0] == "obs" and len(parts) == 5 and int(parts[1]) % 3600 == 0:
            v = parts[2]
            self._margin(None if v == "-" else float(v), int(parts[1]) // 3600)
        reply = self.host.handle(line)
        if "error" in reply:
            exp = dict(error=True)
        elif parts[0] == "obs":
            exp = dict(ok=True, codes=reply["codes"])
        elif parts[0] == "forecast":
            q = np.asarray(reply["q"], np.float32)
            exp = dict(after_unix_hour=reply["after_unix_hour"], fallback=reply["fallback"],
                       theta=[_f(v) for v in reply["theta"]],
                       q=self.blob.put(q) if record else None)
        else:
            exp = {k: reply[k] for k in STATUS_KEYS}
        self.sc["events"].append(dict(op="cmd", line=line, expect=exp))
        return reply

    def obs(self, values, hour):
        return self.cmd(f"obs {int(hour) * 3600} " + " ".join(_num(v) for v in values))

    def forecast(self, record=True, at=None):
        line = "forecast" if at is None else f"forecast {int(at)}"
        return self.cmd(line, record)

    def status(self):
        return self.cmd("status")

    def state(self, name):
        with open(os.path.join(self.golden_dir, name), "wb") as fh:
            fh.write(self.host.rt.serialize())
        self.sc["events"].append(dict(op="state", file=name))


def _scenario(name, lat, lon, elev, model="model", precision="fp32", conformal=False,
              aci=False, init_files=()):
    return dict(name=name, model=model, precision=precision, lat=lat, lon=lon, elev=elev,
                conformal=conformal, aci=aci, init_files=list(init_files), events=[])


def _run(sc, make, golden_dir, blob, body):
    ses = _Session(sc, make, golden_dir, blob)
    try:
        body(ses)
    finally:
        ses.close()
    return sc, ses.margins


def scenario_cold_aci(make, out, blob, variant=0):
    """Холодный старт с калибровкой.

    Переход через Новый год, края диапазонов, половины между целыми, NaN, пропуски,
    простой в несколько часов, ошибочные команды.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("cold_aci", 52.37, 4.9, -2.0, conformal=True, aci=True)

    def body(ses):
        ses.restart()
        s = synthetic_series(330, seed=11 + 100 * variant, t0=_hour("2021-12-27T20"))
        bad = {40: (75.0, None, None), 41: (None, None, -5.0), 42: (None, 250.0, None),
               43: (float("nan"), None, None), 44: (60.6, 1100.06, 100.6),
               45: (60.4, 1100.04, 100.4), 46: (-0.5, 1013.25, 12.5),
               47: (-200.0, 7000.0, 300.0)}
        skip = set(range(200, 205))
        for k in range(330):
            if k in skip:
                continue
            obs = _obs(s, k)
            if k in bad:
                obs = [b if b is not None else o for b, o in zip(bad[k], obs)]
            ses.obs(obs, s["t0"] + k)
            if k == 3:
                ses.cmd(f"obs {(s['t0'] + k) * 3600 + 1800} 1 1000 50")
                ses.cmd(f"obs {(s['t0'] + k) * 3600} 1 1000 50")
                ses.cmd(f"obs {(s['t0'] + k + 1) * 3600} 1 x 50")
                ses.cmd("predict")
            if k % 6 == 5:
                ses.forecast(record=k % 24 == 23)
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_restart(make, out, blob, variant=0):
    """Старт из состояния после полного окна: прогноз сразу, без новых наблюдений.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("restart", -33.87, 151.21, 58.0, conformal=True, aci=True,
                   init_files=[dict(file="state_restart.bin", **{"as": "state_a.bin"},
                                    mtime=0)])
    pre_sc = dict(sc, name="restart_pre", init_files=[], events=[])
    s = None
    margins = []

    def pre_body(ses):
        nonlocal s
        ses.restart()
        n = ses.host.rt.window + 88
        s = synthetic_series(n + 100, seed=29 + 100 * variant, t0=_hour("2023-03-24T05"))
        for k in range(n):
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 8 == 7:
                ses.forecast(record=False)
        with open(os.path.join(out, "state_restart.bin"), "wb") as fh:
            fh.write(ses.host.rt.serialize())
        ses.sc["_n"] = n

    _, m_pre = _run(pre_sc, make, out, _Blob(), pre_body)
    margins += m_pre
    n = pre_sc["_n"]

    def body(ses):
        got = ses.restart()
        if got != "state_a.bin":
            raise RuntimeError("эталон рестарта не восстановил начальное состояние")
        ses.forecast()
        for k in range(n, len(s["x"])):
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 12 == 11:
                ses.forecast()
        ses.state("state_restart_end.bin")
        ses.status()
    sc, m = _run(sc, make, out, blob, body)
    return sc, margins + m


def scenario_extremes(make, out, blob, variant=0):
    """Крайние случаи без калибровки.

    Полярная точка, долгие пустые часы, значения ровно на границах диапазонов, простой в
    час и простой длиннее окна.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("extremes", 78.22, 15.65, 2000.0)

    def body(ses):
        ses.restart()
        W = ses.host.rt.window
        hour = _hour("2022-06-28T04")
        for _ in range(30):
            ses.obs([None, None, None], hour)
            hour += 1
        ses.forecast()
        edges = [(-90.0, 300.0, 0.0), (60.0, 1100.0, 100.0), (-90.0, 1100.0, 100.0),
                 (60.0, 300.0, 0.0)]
        for k in range(60):
            ses.obs(list(edges[k % 4]), hour)
            hour += 1 if k != 30 else 2
        ses.forecast()
        ses.obs([-5.0, 800.0, 55.0], hour)
        ses.forecast()
        hour += W + 5
        ses.obs([-4.0, 801.0, 56.0], hour)
        ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_long(make, out, blob, variant=0):
    """Длинный прогон.

    Выпуски в случайные часы, простой короче и длиннее окна, перезапуски в произвольные
    часы, в том числе посреди простоя.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("long", 55.75, 37.62, 150.0, conformal=True, aci=False)
    rng = np.random.default_rng(GOLDEN_SEED + 1 + variant)

    def body(ses):
        ses.restart()
        W = ses.host.rt.window
        n = 3000
        s = synthetic_series(n, seed=37 + 100 * variant, t0=_hour("2022-10-01T00"))
        skip = set(range(700, 730)) | set(range(1400, 1400 + W + 50))
        restarts = set(int(k) for k in rng.choice(np.arange(100, n - 100), 3, replace=False))
        restarts.add(1500)
        for k in range(n):
            if k in restarts:
                ses.restart()
                ses.forecast()
            if k in skip:
                continue
            ses.obs(_obs(s, k), s["t0"] + k)
            if rng.random() < 0.12:
                ses.forecast(record=bool(rng.random() < 0.12))
            if k % 500 == 499:
                ses.status()
        ses.forecast()
        ses.state("state_long_end.bin")
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_qc(make, out, blob, variant=0):
    """Коды причинного контроля качества.

    Выбросы, возврат к прежнему уровню, скачок, залипание, насыщение влажности, давление
    на уровне моря вместо станционного.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    elev = 800.0
    sc = _scenario("qc", 46.95, 7.45, elev)

    def body(ses):
        ses.restart()
        n = 720
        rng = np.random.default_rng(5)
        k = np.arange(n)
        T = 9 + 5 * np.cos(2 * np.pi * ((k % 24) - 15) / 24) + 0.4 * rng.standard_normal(n)
        P = station_pressure_expected(elev) + 3 * np.sin(k / 40) + 0.2 * rng.standard_normal(n)
        RH = np.clip(65 - 2 * (T - 9) + 3 * rng.standard_normal(n), 5, 99)
        T[150] += 25
        T[220:223] += 18
        T[300:] += 14
        P[350] -= 30
        T[400:440], RH[400:440] = T[400], RH[400]
        P[450:500] = P[450]
        RH[520:610] = 100.0
        P[620:] += 1013.25 - station_pressure_expected(elev)
        t0 = _hour("2023-05-02T00")
        seen = 0
        for i in range(n):
            reply = ses.obs([float(T[i]), float(P[i]), float(RH[i])], t0 + i)
            for c in reply["codes"]:
                seen |= int(c)
            if i % 100 == 99:
                ses.forecast()
        ses.status()
        need = 4 | 32 | 64
        if seen & need != need:
            raise RuntimeError(f"сценарий кодов не породил выброс, залипание и давление на "
                               f"уровне моря: встреченные коды {seen:#x}")
    return _run(sc, make, out, blob, body)


def scenario_rounding(make, out, blob, variant=0):
    """Запись при поступлении.

    Значения ровно посередине между целыми, в том числе отрицательные, давление
    посередине между десятыми, значения у границ диапазонов.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("rounding", 60.17, 24.94, 20.0)

    def body(ses):
        ses.restart()
        T = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 12.5, -12.5, 0.49999, -0.50001, 3.5, -3.5,
             60.5, 61.5, -90.5, -89.5]
        RH = [50.5, 51.5, 0.5, 99.5, 100.5, -0.5, 49.5, 48.5, 60.49999, 60.50001, 1.5, 2.5,
              101.5, -1.5, 70.5, 71.5]
        P = [1013.25, 1013.35, 1013.45, 999.95, 1000.05, 1013.15, 1012.85, 1011.75, 1011.65,
             1010.55, 1009.45, 1008.35, 1100.05, 1100.04, 299.95, 299.96]
        hour = _hour("2023-01-10T00")
        for rep in range(4):
            for i in range(len(T)):
                ses.obs([T[i] + rep, P[i], RH[i]], hour)
                hour += 1
        ses.forecast()
        ses.state("state_rounding_end.bin")
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_sparse(make, out, blob, variant=0):
    """Отчёты каждый второй, затем каждый третий час: пропуски заполняются пустыми.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("sparse", 40.42, -3.70, 650.0, conformal=True)

    def body(ses):
        ses.restart()
        s = synthetic_series(900, seed=31, t0=_hour("2022-07-01T00"), p_valid=1.0)
        for k in range(900):
            step = 2 if k < 450 else 3
            if k % step:
                continue
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 48 == 0:
                ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_int8(make, out, blob, variant=0):
    """int8-графы со своим допуском.

    Конформная таблица подогнана на fp32 и поэтому не применяется.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("int8", -33.87, 151.21, 58.0, precision="int8", conformal=True)

    def body(ses):
        ses.restart()
        s = synthetic_series(500, seed=41, t0=_hour("2024-02-20T00"))
        for k in range(500):
            if k == 260:
                ses.restart()
                ses.forecast()
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 50 == 49:
                ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_fallback(make, out, blob, variant=0):
    """Откат: граф выпуска возвращает нечисловые квантили.

    Точка в Якутии, январь.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("fallback", 62.03, 129.73, 100.0, model="model_nan", conformal=True)

    def body(ses):
        ses.restart()
        t0 = _hour("2025-01-15T06")
        ses.forecast(at=t0 * 3600 + 1234)
        s = synthetic_series(30, seed=43, t0=t0 + 1)
        for k in range(30):
            ses.obs([-38.0 + 0.3 * k, 1030.0, 70.0], s["t0"] + k)
        ses.forecast()
        ses.status()
        ses.restart()
        ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_no_obs(make, out, blob, variant=0):
    """Прогноз до первого наблюдения по часам устройства: пустое окно, не откат.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("no_obs", -33.92, 18.42, 10.0, conformal=True, aci=True)

    def body(ses):
        ses.restart()
        t = _hour("2024-11-03T09") * 3600 + 777
        ses.forecast(at=t)
        ses.status()
        ses.restart()
        ses.forecast(at=t + 5 * 3600)
        s = synthetic_series(12, seed=47, t0=t // 3600 + 5)
        for k in range(12):
            ses.obs(_obs(s, k), s["t0"] + k)
        ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_site_shift(make, out, blob, variant=0):
    """Перезапуск с уточнёнными координатами: 0.3 градуса и 42 м.

    Окно сохраняется, всё модельное состояние пересчитывается для новой точки.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("site_shift", 52.37, 4.9, -2.0, conformal=True, aci=True)

    def body(ses):
        ses.restart()
        s = synthetic_series(360, seed=51 + 100 * variant, t0=_hour("2023-09-01T00"))
        for k in range(300):
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 24 == 23:
                ses.forecast(record=k % 96 == 95)
        ses.restart(site=(52.67, 5.2, 40.0))
        ses.forecast()
        for k in range(300, 360):
            ses.obs(_obs(s, k), s["t0"] + k)
        ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_relocation(make, out, blob, variant=0):
    """Уточнение высоты, перенос прибора и перезапуск на той же точке.

    Точка в горах на 700 м, станционное давление около 970 гПа. Затем высота уточняется
    до 790 м: окно и множитель калибровки сохраняются, а новые часы того же давления
    проверка давления на уровне моря уже отбраковывает по новой высоте. Затем прибор
    перенесён в Осло: окно пустое, множитель нулевой, момент последнего шага сохранён,
    выпуск сразу после перезапуска строится по пустому окну. Последний перезапуск - на
    той же точке.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    alps, alps_fixed, oslo = (46.95, 7.45, 700.0), (46.95, 7.45, 790.0), (59.91, 10.75, 20.0)
    sc = _scenario("relocation", *alps, conformal=True, aci=True)
    n1, n2, n3 = 300, 30, 60

    def body(ses):
        s = synthetic_series(n1 + n2 + n3, seed=71 + 100 * variant, t0=_hour("2023-11-20T00"))
        k = np.arange(n1 + n2 + n3)
        s["x"][:, 1] = np.where(k < n1 + n2, 970.0 + 1.2 * np.sin(k / 30.0), s["x"][:, 1])
        s["m"][:, 1] = np.where(k < n1 + n2, 1.0, s["m"][:, 1])
        ses.restart()
        for i in range(n1):
            ses.obs(_obs(s, i), s["t0"] + i)
            if i % 24 == 23:
                ses.forecast(record=i % 96 == 95)
        ses.restart(site=alps_fixed)
        ses.status()
        ses.forecast()
        slp = 0
        for i in range(n1, n1 + n2):
            slp |= ses.obs(_obs(s, i), s["t0"] + i)["codes"][1]
        ses.forecast()
        ses.status()
        ses.restart(site=oslo)
        rt = ses.host.rt
        moved = (rt.site_change, rt.filled, max(abs(t) for t in rt.theta), rt.last_hour)
        ses.status()
        ses.forecast()
        for i in range(n1 + n2, n1 + n2 + n3):
            ses.obs(_obs(s, i), s["t0"] + i)
            if i % 12 == 11:
                ses.forecast()
        ses.state("state_relocation_end.bin")
        ses.status()
        ses.restart(site=oslo)
        same = ses.host.rt.site_change
        ses.status()
        ses.forecast()
        if not slp & 64:
            raise RuntimeError("сценарий смены точки: после уточнения высоты давление на "
                               "уровне моря не отбраковано")
        if moved != (SITE_MOVED, 0, 0.0, s["t0"] + n1 + n2 - 1) or same != SITE_SAME:
            raise RuntimeError(f"сценарий смены точки: перенос дал {moved}, повторный "
                               f"перезапуск {same}")
    return _run(sc, make, out, blob, body)


def scenario_store_order(make, out, blob, variant=0):
    """Выбор свежего состояния по содержимому.

    Старое состояние с полным окном изменено позже, новое состояние после холодного
    старта изменено раньше; выбирается новое.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("store_order", 59.91, 10.75, 20.0,
                   init_files=[dict(file="state_order_old.bin", mtime=100,
                                    **{"as": "state_a.bin"}),
                               dict(file="state_order_new.bin", mtime=0,
                                    **{"as": "state_b.bin"})])
    pre_sc = dict(sc, name="store_order_pre", init_files=[], events=[])
    last = {}

    def pre_body(ses):
        ses.restart()
        rt = ses.host.rt
        W = rt.window
        s = synthetic_series(W + 30, seed=61, t0=_hour("2024-04-01T00"))
        for k in range(W + 30):
            rt.step(*_obs(s, k), s["t0"] + k)
        with open(os.path.join(out, "state_order_old.bin"), "wb") as fh:
            fh.write(rt.serialize())
        h = rt.last_hour + W + 20
        for k in range(6):
            rt.step(5.0 + k, 1001.0, 60.0, h + k)
        with open(os.path.join(out, "state_order_new.bin"), "wb") as fh:
            fh.write(rt.serialize())
        last["hour"] = rt.last_hour

    _run(pre_sc, make, out, _Blob(), pre_body)

    def body(ses):
        if ses.restart() != "state_b.bin":
            raise RuntimeError("эталон порядка файлов выбрал не то состояние")
        ses.forecast()
        for k in range(1, 4):
            ses.obs([6.0 + k, 1002.0, 61.0], last["hour"] + k)
        ses.restart()
        ses.forecast()
        ses.status()
    return _run(sc, make, out, blob, body)


def state_v4(raw, theta):
    """Состояние версии 4 из состояния текущей версии: то же окно, один множитель калибровки.

    Args:
        raw: байты состояния текущей версии.
        theta: логарифм множителя для заголовка версии 4.

    Returns:
        Байты состояния версии 4.
    """
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    old = np.zeros((), STATE_HEADER_V4)
    for k in ("magic", "filled", "reserved", "last_hour", "lat", "lon", "elev"):
        old[k] = hdr[k]
    old["version"], old["aci_theta"] = 4, theta
    return old.tobytes() + raw[STATE_HEADER.itemsize:]


def scenario_restart_v4(make, out, blob, variant=0):
    """Старт из состояния версии 4: один множитель калибровки переходит во все бины лидов.

    Окно - то же, что у эталона рестарта. После старта прибор выпускает и подстраивает
    множители бинов по отдельности; состояние на диске - уже текущей версии.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    with open(os.path.join(out, "state_restart.bin"), "rb") as fh:
        raw = fh.read()
    last = int(np.frombuffer(raw, STATE_HEADER, count=1)[0]["last_hour"])
    with open(os.path.join(out, "state_v4.bin"), "wb") as fh:
        fh.write(state_v4(raw, V4_THETA))
    sc = _scenario("restart_v4", -33.87, 151.21, 58.0, conformal=True, aci=True,
                   init_files=[dict(file="state_v4.bin", **{"as": "state_a.bin"}, mtime=0)])

    def body(ses):
        if ses.restart() != "state_a.bin":
            raise RuntimeError("эталон состояния версии 4 не восстановил начальное состояние")
        st = ses.status()
        if st["theta"] != [V4_THETA] * len(st["theta"]):
            raise RuntimeError(f"множитель состояния версии 4 не перенесён во все бины: "
                               f"{st['theta']}")
        ses.forecast()
        s = synthetic_series(72, seed=91 + 100 * variant, t0=last + 1)
        for k in range(72):
            ses.obs(_obs(s, k), s["t0"] + k)
            if k % 6 == 5:
                ses.forecast(record=k % 24 == 23)
        ses.state("state_v4_end.bin")
        ses.status()
    return _run(sc, make, out, blob, body)


def scenario_hourly_aci(make, out, blob, variant=0):
    """Ежечасный выпуск с калибровкой: обратную связь получают все бины лидов.

    Args:
        make: фабрика рантайма по описанию сценария и точке.
        out: каталог эталонов.
        blob: накопитель эталонных векторов прогнозов.
        variant: номер варианта; меняет сид синтетического ряда.

    Returns:
        Пара: описание сценария с командами и ожидаемыми ответами и наибольшие
        отклонения, которые нужны для допусков.
    """
    sc = _scenario("hourly_aci", 48.85, 2.35, 35.0, conformal=True, aci=True)

    def body(ses):
        ses.restart()
        n = 240
        s = synthetic_series(n, seed=81 + 100 * variant, t0=_hour("2024-05-10T00"))
        for k in range(n):
            ses.obs(_obs(s, k), s["t0"] + k)
            ses.forecast(record=k % 24 == 23)
        st = ses.status()
        if min(st["aci_updates"]) == 0:
            raise RuntimeError(f"ежечасный выпуск: не все бины лидов получили обратную связь: "
                               f"{st['aci_updates']}")
    return _run(sc, make, out, blob, body)


def write_nan_model(out_dir):
    """Каталог модели, граф выпуска которой всегда возвращает нечисловые квантили.

    Остальные графы и конформная таблица берутся из соседнего каталога модели.

    Args:
        out_dir: каталог, куда пишется модель.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    src = os.path.join(out_dir, "model")
    dst = os.path.join(out_dir, "model_nan")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    d = man["dims"]
    q = np.full((1, d["horizon"], d["n_quantiles"]), np.nan, np.float32)
    node = helper.make_node("Constant", [], ["q"], value=numpy_helper.from_array(q, "q_nan"))
    graph = helper.make_graph(
        [node], "issue_nan",
        [helper.make_tensor_value_info("loc", TensorProto.FLOAT, [1, d["loc_dim"]])],
        [helper.make_tensor_value_info("q", TensorProto.FLOAT, list(q.shape))])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", man["opset"])])
    proto.ir_version = 8
    onnx.save(proto, os.path.join(dst, "issue_nan.onnx"))
    for name, g in man["graphs"].items():
        if name == "issue":
            g.update(fp32="issue_nan.onnx", inputs=["loc"])
            if "int8" in g:
                g["int8"] = "issue_nan.onnx"
            continue
        g["fp32"] = "../model/" + g["fp32"]
        if "int8" in g:
            g["int8"] = "../model/" + g["int8"]
    if man["calibration"].get("conformal"):
        man["calibration"]["conformal"] = "../model/" + man["calibration"]["conformal"]
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)


SITE_CASES = (
    ((52.37, 4.9, -2.0), (52.37, 4.9, -2.0)),
    ((52.37, 4.9, -2.0), (52.67, 5.2, 40.0)),
    ((52.0, 4.9, 10.0), (52.5, 4.9, 10.0)),
    ((52.0, 4.9, 10.0), (52.500004, 4.9, 10.0)),
    ((52.0, 4.9, 10.0), (51.4, 4.9, 10.0)),
    ((10.0, 179.9, 0.0), (10.0, -179.9, 0.0)),
    ((10.0, 179.8, 0.0), (10.0, -179.5, 0.0)),
    ((10.0, 180.0, 0.0), (10.0, -180.0, 0.0)),
    ((10.0, 0.25, 0.0), (10.0, -0.25, 0.0)),
    ((10.0, 20.0, 500.0), (10.0, 20.0, 600.0)),
    ((10.0, 20.0, 500.0), (10.0, 20.0, 600.5)),
    ((-33.87, 151.21, 58.0), (59.91, 10.75, 20.0)),
)


def site_cases(cfg=None):
    """Сравнения точки состояния с текущей: границы порогов, долгота через 180 градусов.

    Args:
        cfg: параметры хоста с порогами или None - значения датакласса.

    Returns:
        Словарь: пороги и список случаев с разницами и ожидаемым исходом.
    """
    cfg = RuntimeConfig() if cfg is None else cfg
    cases = []
    for old, new in SITE_CASES:
        kind, _ = site_change(old, new, cfg)
        cases.append(dict(old=[_f(v) for v in old], new=[_f(v) for v in new],
                          gap=[float(g) for g in site_gap(old, new)], expect=kind))
    kinds = {c["expect"] for c in cases}
    if kinds != {SITE_SAME, SITE_REFINED, SITE_MOVED}:
        raise RuntimeError(f"сравнения точек покрывают не все исходы: {sorted(kinds)}")
    return dict(runtime=cfg.to_dict(), cases=cases)


def calendar_cases():
    """Часы от эпохи вокруг границ годов, включая 2000 (високосный) и 2100 (нет)."""
    pts = ["1970-01-01T00", "1999-12-31T23", "2000-02-28T12", "2000-12-31T23",
           "2023-12-31T20", "2024-02-28T22", "2024-12-31T23", "2025-06-15T11",
           "2100-02-28T23", "2100-12-31T23"]
    base = [int(to_utc_hour(np.datetime64(p, "s"))) for p in pts]
    hours = sorted({h + d for h in base for d in (-1, 0, 1, 2)})
    doy, hr = window_calendar(0, np.array(hours, np.int64))
    how = hour_of_year(np.array(hours, np.int64))
    return dict(unix_hours=hours, doy=[_f(v) for v in doy], hour=[_f(v) for v in hr],
                hour_of_year=[int(v) for v in how])


def calibration_cases(model, blob):
    """Эталоны калибровки как отдельного узла: поправка квантилей и онлайн-подстройка множителя.

    Args:
        model: модель эталона.
        blob: накопитель эталонных векторов.

    Returns:
        Словарь: случаи поправки, прогон подстройки и нормированные выходы факта.
    """
    rng = np.random.default_rng(GOLDEN_SEED)
    H = model.cfg.horizon
    cases = []
    for theta, conf, hist in (((0.0, 0.0, 0.0, 0.0), True, 0),
                              ((0.37, 0.37, 0.37, 0.37), True, 5),
                              ((-0.52, 0.0, 0.21, -0.1), True, 100),
                              ((0.0, 0.0, 0.0, 0.0), True, 672),
                              ((0.37, -0.2, 0.0, 0.11), False, 30)):
        q = np.sort(rng.normal(0.0, 3.0, (H, len(model.cfg.quantiles))), axis=-1)
        q[5, 2] = q[5, 4] + 0.5
        q[100, 1] = q[100, 5] + 0.3
        q = q.astype(np.float32)
        th = [_f(v) for v in theta]
        out, _mu = calibrate_forecast(q, GOLDEN_SHIFT if conf else None, th, hist)
        cases.append(dict(q=blob.put(q), theta=th, conformal=conf, history=hist,
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
    qs[3] = qs[3, 3]
    ys[7] = qs[7, 3]
    sc = aci_score(ys.astype(np.float32).astype(np.float64), qs, GOLDEN_ACI.interval)
    score = dict(y=[_f(v) for v in ys], q=blob.put(qs),
                 expect=[_enc_score(v) for v in sc])
    return dict(cases=cases, aci=aci, score=score)


BUILDERS = (scenario_cold_aci, scenario_restart, scenario_restart_v4, scenario_extremes,
            scenario_long, scenario_qc, scenario_rounding, scenario_sparse, scenario_int8,
            scenario_fallback, scenario_no_obs, scenario_site_shift, scenario_relocation,
            scenario_store_order, scenario_hourly_aci)


def generate(out_dir=DEFAULT_DIR):
    """Полная генерация эталона: графы, сценарии, состояния, календарь, калибровка.

    Args:
        out_dir: каталог эталона.

    Returns:
        Документ эталона.

    Raises:
        RuntimeError: решение о промахе калибровки в каком-то сценарии слишком близко к
            порогу и неустойчиво к float32, или сценарий не проверяет то, ради чего он
            написан.
    """
    from mayak.runtime.graphs import export_graphs, quantize_graph
    os.makedirs(out_dir, exist_ok=True)
    model = golden_model()
    mdir = os.path.join(out_dir, "model")
    manifest = export_graphs(model, mdir, conformal=GOLDEN_SHIFT, aci=GOLDEN_ACI)
    # int8-копии добавляются к экспорту fp32: таблица эталона подогнана на fp32, и на
    # int8-графах хост обязан её не применять.
    for name, g in manifest["graphs"].items():
        g["int8"] = g["fp32"] if name == "init" else quantize_graph(os.path.join(mdir, g["fp32"]))
    manifest["provenance"].pop("created_utc", None)
    with open(os.path.join(out_dir, "model", "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    write_nan_model(out_dir)
    make = lambda sc, site: scenario_runtime(model, out_dir, sc, site)
    blob, scenarios, margins = _Blob(), [], []
    for build in BUILDERS:
        # Ряд сценария с калибровкой подбирается так, чтобы ни одно решение о промахе не
        # лежало у самого порога: иначе разница float32 двух реализаций его переворачивает.
        for variant in range(MAX_VARIANTS):
            mark = blob.mark()
            sc, m = build(make, out_dir, blob, variant)
            if not m or min(m) >= MIN_ACI_MARGIN:
                break
            blob.rollback(mark)
        else:
            raise RuntimeError(f"{sc['name']}: обратная связь ACI в {min(m):.1e} от порога "
                               f"при всех {MAX_VARIANTS} вариантах ряда")
        scenarios.append(sc)
        margins += m
    doc = dict(format=GOLDEN_FORMAT, seed=GOLDEN_SEED, mtime_base=MTIME_BASE,
               tolerance=dict(q_abs=Q_ATOL, q_abs_int8=Q_ATOL_INT8, fresh_abs=FRESH_ATOL,
                              theta_abs=THETA_ATOL),
               aci_margin_min=float(min(margins)) if margins else None,
               scenarios=scenarios, calendar=calendar_cases(), site=site_cases(),
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


class GoldenMismatch(AssertionError):
    """Ответ хоста расходится с эталоном не в пределах допуска, а по существу."""


def _check(ok, sc, i, what):
    if not ok:
        raise GoldenMismatch(f"{sc['name']}: событие {i}: {what}")


def compare_state(got, ref, theta_atol=THETA_ATOL):
    """Совпадение двух состояний: заголовок и окно до байта, множители с допуском.

    Returns:
        Пустая строка при совпадении или описание расхождения.
    """
    if len(got) != len(ref):
        return f"размер {len(got)} Б против {len(ref)} Б"
    if got[:16] != ref[:16]:
        return "заголовок до множителей калибровки"
    ha = np.frombuffer(got, STATE_HEADER, count=1)[0]
    hb = np.frombuffer(ref, STATE_HEADER, count=1)[0]
    if np.abs(ha["aci_theta"].astype(np.float64) - hb["aci_theta"]).max() > theta_atol:
        return f"множители калибровки {ha['aci_theta']} против {hb['aci_theta']}"
    off = STATE_HEADER.fields["lat"][1]
    if got[off:] != ref[off:]:
        return "координаты или сырое окно"
    return ""


def _close(a, b, atol):
    """Множители по бинам совпадают с допуском."""
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


def replay_scenario(sc, blob, make, state_dir, golden_dir, n_quantiles, i_med,
                    theta_atol=THETA_ATOL):
    """Прогон сценария эталона хостом на Python.

    Args:
        sc: сценарий эталона.
        blob: массив ожидаемых значений.
        make: по сценарию и координатам возвращает рантайм.
        state_dir: пустой каталог состояния.
        golden_dir: каталог эталона.
        n_quantiles: число квантилей.
        i_med: номер медианы среди квантилей.
        theta_atol: допуск множителя калибровки.

    Returns:
        Наибольшее расхождение квантилей с эталоном.

    Raises:
        GoldenMismatch: расхождение не в квантилях: коды, моменты, откат, сводка,
            выбранный файл состояния или состояние на диске.
    """
    place_init_files(sc, golden_dir, state_dir)
    site = (sc["lat"], sc["lon"], sc["elev"])
    host, worst = None, 0.0
    for i, ev in enumerate(sc["events"]):
        if ev["op"] == "restart":
            if ev["site"] is not None:
                site = tuple(ev["site"])
            host = Host(make(sc, site), StateStore(state_dir), clock=_no_clock)
            got = host.restore()
            got = None if got is None else os.path.basename(got)
            _check(got == ev["expect"]["restored"], sc, i,
                   f"восстановлен {got}, эталон {ev['expect']['restored']}")
            continue
        if ev["op"] == "state":
            with open(os.path.join(golden_dir, ev["file"]), "rb") as fh:
                why = compare_state(host.rt.serialize(), fh.read(), theta_atol)
            _check(not why, sc, i, f"состояние: {why}")
            continue
        line, exp = ev["line"], ev["expect"]
        reply = host.handle(line)
        if exp.get("error"):
            _check("error" in reply, sc, i, f"{line}: ожидалась ошибка, ответ {reply}")
            continue
        _check("error" not in reply, sc, i, f"{line}: {reply.get('error')}")
        kind = line.split()[0]
        if kind == "obs":
            _check(reply["codes"] == exp["codes"], sc, i,
                   f"коды {reply['codes']} против {exp['codes']}")
        elif kind == "forecast":
            _check(reply["after_unix_hour"] == exp["after_unix_hour"], sc, i,
                   f"момент выпуска {reply['after_unix_hour']} против {exp['after_unix_hour']}")
            _check(reply["fallback"] == exp["fallback"], sc, i, "признак отката")
            _check(_close(reply["theta"], exp["theta"], theta_atol), sc, i,
                   f"множители {reply['theta']} против {exp['theta']}")
            q = np.asarray(reply["q"], np.float32)
            _check(np.array_equal(np.asarray(reply["mu"], np.float32), q[:, i_med]), sc, i,
                   "медиана не равна среднему квантилю")
            _check(bool(np.all(np.diff(q, axis=-1) >= 0)), sc, i, "квантили не монотонны")
            if exp["q"] is not None:
                ref = take(blob, exp["q"], (-1, n_quantiles))
                worst = max(worst, float(np.abs(q - ref).max()))
        else:
            for k, v in exp.items():
                if k == "theta":
                    _check(_close(reply[k], v, theta_atol), sc, i, f"сводка {k}")
                else:
                    _check(reply[k] == v, sc, i, f"сводка {k}: {reply[k]} против {v}")
    return worst


__all__ = ["DEFAULT_DIR", "FRESH_ATOL", "GOLDEN_ACI", "GOLDEN_FORMAT", "GOLDEN_SHIFT",
           "GoldenMismatch", "MIN_ACI_MARGIN", "Q_ATOL", "Q_ATOL_INT8", "SCENARIOS",
           "STATUS_KEYS", "THETA_ATOL", "V4_THETA", "calendar_cases", "calibration_cases",
           "compare_state", "generate", "golden_model", "load", "place_init_files",
           "replay_scenario", "scenario_runtime", "site_cases", "state_v4", "take"]
