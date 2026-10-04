"""МАЯК как два графа без состояния для устройства, экспорт и исполнители графов.

* ``forecast``    - проход модели (``MAYAK.forward``) по окну истории: координаты,
  значения и маски истории после контроля качества, календарь истории и горизонта. На
  выходе квантили до калибровки.
* ``climatology`` - таблица климатологии точки для отката: среднее и масштаб
  климат-поля с паспортом холодного старта на каждый час високосного года. Считается
  один раз при старте устройства.

Своей арифметики модели в обёртках нет. Окно, контроль качества, калибровку и
сериализацию держит устройство.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from mayak.astro import astro_features
from mayak.runtime.device import (CLIMATOLOGY_INPUTS, FORECAST_INPUTS, HOURS_OF_YEAR,
                                  RAW_CHANNELS, STATE_HEADER, STATE_VERSION, Device,
                                  state_nbytes)
from mayak.timeaxis import to_utc_hour, window_calendar

GRAPH_FORMAT = 7
GRAPH_NAMES = ("forecast", "climatology")
GRAPH_IO = {
    "forecast": (FORECAST_INPUTS, ("q",)),
    "climatology": (CLIMATOLOGY_INPUTS, ("clim_mu", "clim_sig")),
}
SYNTHETIC_START = "2021-12-27T00"
EVAL_STATION = "device"


class _Graph(nn.Module):
    """Обёртка в режиме eval.

    Экспорт восстанавливает режим экспортируемого модуля рекурсивно: обёртка в режиме
    train после экспорта перевела бы в train и общую модель, и паспорт начал бы
    сэмплировать.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.eval()


def year_calendar():
    """Календарь всех часов високосного года в конвенции проекта.

    Номер часа в таблице климатологии совпадает с номером часа от начала года, поэтому
    таблица подходит и обычному году: его часы просто не доходят до последних суток.

    Returns:
        Пара массивов float32 формы (8784,): день года с долей суток и час UTC.
    """
    sec = np.arange(HOURS_OF_YEAR, dtype=np.int64) * 3600
    return (sec / 86400.0).astype(np.float32), ((sec % 86400) / 3600.0).astype(np.float32)


class ForecastGraph(_Graph):
    """Выпуск: проход модели по окну истории, на выходе квантили."""

    def forward(self, lat, lon, elev, x_hist, mask_hist, doy_hist, hour_hist, doy_fut,
                hour_fut):
        batch = dict(lat=lat, lon=lon, elev=elev, x_hist=x_hist, mask_hist=mask_hist,
                     doy_hist=doy_hist, hour_hist=hour_hist, doy_fut=doy_fut,
                     hour_fut=hour_fut)
        return self.m(batch)["q"]


class ClimatologyGraph(_Graph):
    """Таблица климатологии точки для отката.

    Таблица - то, что выдаёт модель при пустой истории без мод и поправки: климат-поле
    с паспортом по пустым суточным сводкам.
    """

    def __init__(self, model):
        super().__init__(model)
        doy, hour = year_calendar()
        self.register_buffer("year_doy", torch.from_numpy(doy)[None], persistent=False)
        self.register_buffer("year_hour", torch.from_numpy(hour)[None], persistent=False)

    def forward(self, lat, lon, elev):
        m = self.m
        loc = m.loc(lat, lon, elev)
        empty = lat.new_zeros(lat.shape[0], m.cfg.max_history)
        summ, has = m.daily_summaries(empty, empty, empty, empty)
        z, _ = m.passport(loc, summ, has, sample=False)
        # Календарь года привязан к входу нулевым слагаемым: иначе экспорт развернул бы
        # все его производные в константы и граф вырос бы в несколько раз.
        zero = lat[:, None] * 0.0
        astro = astro_features(self.year_doy + zero, self.year_hour + zero, lat[:, None],
                               lon[:, None])
        clim_mu, clim_sig, _ = m.field.evaluate(m.field.coefficients(loc, z), astro)
        return clim_mu, clim_sig


GRAPH_MODULES = dict(forecast=ForecastGraph, climatology=ClimatologyGraph)


def dims(cfg):
    """Размеры входов и выходов графов и окна устройства для конфига модели.

    Args:
        cfg: конфиг модели.

    Returns:
        Словарь размеров.
    """
    return dict(horizon=cfg.horizon, n_quantiles=cfg.n_quantiles, history=cfg.max_history,
                window=cfg.device_window, hours_of_year=HOURS_OF_YEAR)


def default_start():
    """Абсолютный час начала синтетических рядов.

    Конец декабря, чтобы ряды переходили через Новый год.
    """
    return int(to_utc_hour(np.datetime64(SYNTHETIC_START, "s")))


def synthetic_series(n, seed=0, t0=None, p_valid=0.9):
    """Почасовой синтетический ряд.

    Args:
        n: длина ряда, часы.
        seed: сид.
        t0: абсолютный час UTC первого часа; по умолчанию конец декабря.
        p_valid: доля часов с наблюдением в каждом канале.

    Returns:
        Словарь: ``x`` значения (n, 3) с нулями на месте пропусков, ``m`` маска наличия
        (n, 3), ``t0`` абсолютный час первого часа.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    T = (8 + 6 * np.sin(2 * np.pi * t / 24) + 3 * np.sin(2 * np.pi * t / 170)
         + rng.standard_normal(n))
    P = 1005 + 8 * np.sin(2 * np.pi * t / 130 + rng.uniform(0, 6)) + 0.4 * rng.standard_normal(n)
    RH = np.clip(70 - 2 * (T - 8) + 5 * rng.standard_normal(n), 5, 100)
    x = np.stack([T, P, RH], -1).astype(np.float32)
    m = (rng.random((n, 3)) < p_valid).astype(np.float32)
    gap = rng.integers(0, max(n - 30, 1))
    m[gap:gap + 20, 0] = 0.0
    t0 = default_start() if t0 is None else int(t0)
    return dict(x=x * m, m=m, t0=t0)


def feed(device, series, k0, k1):
    """Подать часы ряда устройству через шаг часа.

    Args:
        device: устройство.
        series: ряд со значениями, масками и первым часом.
        k0: первая строка.
        k1: строка после последней.

    Returns:
        То же устройство.
    """
    x, m = series["x"], series["m"]
    for k in range(k0, k1):
        v = [float(x[k, j]) if m[k, j] > 0 else None for j in range(3)]
        device.step(*v, series["t0"] + k)
    return device


def eval_set(series, ends, lat, lon, elev):
    """Набор окон оценки по одной станции из синтетического ряда.

    Станция - ряд, записанный так, как его пишет прибор, с климатологией по нему же.
    Окно номер i начинает горизонт со строки ``ends[i]``; история - полный буфер, самый
    ранний доступный час - начало ряда. Ряд должен вмещать горизонт после каждого окна.

    Args:
        series: синтетический ряд.
        ends: строки начала горизонта.
        lat: широта, градусы.
        lon: долгота, градусы.
        elev: высота, м.

    Returns:
        Набор окон оценки.
    """
    from mayak.data.climatology import Climatology
    from mayak.data.holdout import EvalSet
    from mayak.data.recording import record_values
    m = np.asarray(series["m"], np.float32)
    raw = np.where(m > 0, record_values(np.where(m > 0, series["x"], 0.0)), 0.0)
    raw = raw.astype(np.float32)
    doy, hour = window_calendar(series["t0"], np.arange(len(raw)))
    clim = Climatology().fit(doy, hour, raw[:, 0], m[:, 0], min_valid=1)
    station = dict(raw=raw, present=m, x=raw, mask=m, clim=clim, t0=int(series["t0"]),
                   lat=float(lat), lon=float(lon), elev=float(elev), N=len(raw))
    ds = EvalSet.__new__(EvalSet)
    ds.clims, ds.floor = {EVAL_STATION: station}, {EVAL_STATION: 0}
    ds.items = [(EVAL_STATION, int(e)) for e in ends]
    ds.requested = [None] * len(ds.items)
    return ds


def eval_inputs(ds, i):
    """Входы модели окна оценки с осью батча длины 1.

    Args:
        ds: набор окон оценки.
        i: номер окна.

    Returns:
        Словарь массивов float32 по входам графа прогноза.
    """
    b = ds[i]
    return {k: b[k][None].numpy().astype(np.float32) for k in FORECAST_INPUTS}


def example_inputs(model, seed=0):
    """Правдоподобные входы графов для трассировки и сверки экспорта.

    Входы прогноза - окно оценки синтетического ряда с историей короче полного буфера,
    чтобы в окне были и пустые, и заполненные часы.

    Args:
        model: модель МАЯК.
        seed: сид ряда.

    Returns:
        Словарь: имя графа и словарь его входов.
    """
    cfg = model.cfg
    lat, lon, elev = 52.37, 4.9, 12.0
    end = cfg.max_history - cfg.max_history // 4
    s = synthetic_series(end + cfg.horizon, seed=seed)
    inp = eval_inputs(eval_set(s, [end], lat, lon, elev), 0)
    return dict(forecast=inp, climatology={k: inp[k] for k in CLIMATOLOGY_INPUTS})


def calibration_bins():
    """Бины лидов и длины истории калибровки в том виде, в каком они пишутся в манифест.

    Returns:
        Пара списков пар границ включительно: бины лидов и бины длины истории.
    """
    from mayak.constants import HISTORY_BINS
    from mayak.metrics import LEAD_BINS
    return ([[int(a), int(b)] for a, b in LEAD_BINS],
            [[int(b[0]), int(b[1])] for b in HISTORY_BINS])


def conformal_for_export(conformal, checkpoint=None):
    """Таблица поправок для экспорта.

    Args:
        conformal: путь к таблице с записью о подгонке рядом или сама таблица.
        checkpoint: путь к экспортируемому чекпойнту; если задан, таблица должна быть
            подогнана по нему же.

    Returns:
        Таблица float32 по бинам лидов и длины истории.

    Raises:
        ValueError: таблица подогнана по другому чекпойнту или сдвигает медиану.
    """
    from mayak.leakage import file_digest, load_conformal
    from mayak.metrics import check_conformal_shape
    if not isinstance(conformal, str):
        return check_conformal_shape(conformal)
    shift, rec = load_conformal(conformal)
    want = rec.get("checkpoint_digest")
    if checkpoint is not None and want is not None and file_digest(checkpoint) != want:
        raise ValueError(f"конформная таблица {conformal} подогнана по другому чекпойнту "
                         f"({rec.get('checkpoint')}), а экспортируется {checkpoint}")
    return shift


def export_graphs(model, out_dir, *, conformal=None, aci=None, opset=17, check_atol=1e-4,
                  seed=0, checkpoint=None, runtime=None):
    """Экспорт графов, манифеста и развёрнутой по лидам конформной таблицы.

    Таблица в экспорте - подряд по бинам длины истории, в каждом бине строка на каждый
    лид горизонта, в строке по значению на квантиль. Бины лидов адаптивной калибровки и
    бины длины истории таблицы записаны в раздел калибровки манифеста.

    Каждый граф при экспорте сверяется с PyTorch на правдоподобных входах. Входы, от
    которых граф не зависит, экспорт выбрасывает; манифест перечисляет фактические входы
    графа, и исполнитель подаёт только их.

    Args:
        model: модель.
        out_dir: каталог экспорта.
        conformal: таблица поправок по бинам лидов и длины истории или путь к ней с
            записью о подгонке рядом.
        aci: параметры адаптивной калибровки устройства или None.
        opset: версия набора операций ONNX.
        check_atol: допустимое расхождение графа с PyTorch в единицах масштаба выхода:
            наибольшая разность делится на наибольший модуль выхода, но не меньше единицы.
        seed: сид правдоподобных входов для сверки.
        checkpoint: путь к экспортируемому чекпойнту для сверки с записью о подгонке
            таблицы.
        runtime: параметры хоста устройства с порогами смены точки или None - конфиг
            рантайма по умолчанию.

    Returns:
        Манифест экспорта. В разделе рантайма записаны пороги смены точки.

    Raises:
        ValueError: таблица подогнана по другому чекпойнту или сдвигает медиану.
        RuntimeError: граф расходится с PyTorch больше допуска.
    """
    import onnx
    import onnxruntime as ort
    from mayak.baselines.statistical import ZQ
    from mayak.data.qc import DEFAULT_QC, PHYS
    from mayak.metrics import I_MED, conformal_table
    from mayak.provenance import provenance
    from mayak.runtime.site import load_runtime_config

    runtime = load_runtime_config() if runtime is None else runtime
    was_training = model.training
    model = model.eval()
    cfg = model.cfg
    table = None if conformal is None else conformal_for_export(conformal, checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    inputs = example_inputs(model, seed)
    graphs, checks = {}, {}
    for name in GRAPH_NAMES:
        mod = GRAPH_MODULES[name](model)
        names_in, names_out = GRAPH_IO[name]
        args = tuple(torch.from_numpy(inputs[name][n]) for n in names_in)
        path = os.path.join(out_dir, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod, args, path, input_names=list(names_in),
                              output_names=list(names_out), opset_version=opset,
                              dynamo=False, do_constant_folding=True)
            ref = mod(*args)
        ref = ref if isinstance(ref, tuple) else (ref,)
        proto = onnx.load(path)
        used = [i.name for i in proto.graph.input]
        meta = proto.metadata_props.add()
        meta.key, meta.value = "mayak_graph", name
        onnx.save(proto, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {n: inputs[name][n] for n in names_in if n in used})
        err = max(float(np.abs(o - r.numpy()).max() / max(1.0, float(np.abs(r.numpy()).max())))
                  for o, r in zip(out, ref))
        if err > check_atol:
            raise RuntimeError(f"граф {name}: расхождение ONNX и PyTorch {err:.2e} в единицах "
                               f"масштаба выхода, допуск {check_atol}")
        checks[name] = err
        graphs[name] = dict(file=f"{name}.onnx", inputs=used, outputs=list(names_out),
                            shapes_in={n: list(inputs[name][n].shape) for n in names_in},
                            shapes_out={n: list(r.shape) for n, r in zip(names_out, ref)})
    model.train(was_training)

    lead_bins, history_bins = calibration_bins()
    cal = dict(conformal=None, aci=None, lead_bins=lead_bins, history_bins=history_bins)
    if table is not None:
        conformal_table(table, [lo for lo, _hi in history_bins], cfg.horizon).astype(
            "<f4").tofile(os.path.join(out_dir, "conformal.f32"))
        cal["conformal"] = "conformal.f32"
    if aci is not None:
        cal["aci"] = dict(target=aci.target, gamma=aci.gamma, max_factor=aci.max_factor,
                          interval=list(aci.interval))
    manifest = dict(
        format=GRAPH_FORMAT, model_config=cfg.to_dict(), dims=dims(cfg),
        quantiles=list(cfg.quantiles), i_med=I_MED, zq=[float(v) for v in ZQ],
        raw_channels=list(RAW_CHANNELS), phys={c: list(PHYS[c]) for c in RAW_CHANNELS},
        qc=dict(DEFAULT_QC.to_dict(), lookback_hours=DEFAULT_QC.lookback_hours),
        state=dict(version=STATE_VERSION, header_bytes=STATE_HEADER.itemsize,
                   nbytes=state_nbytes(cfg)),
        graphs=graphs, calibration=cal, runtime=runtime.to_dict(),
        export_check_max_rel=checks, opset=opset,
        provenance=provenance())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest


class TorchBackend:
    """Исполнитель графов на PyTorch: те же обёртки, что идут в экспорт.

    Args:
        model: модель МАЯК.
    """

    def __init__(self, model):
        self.mods = {n: GRAPH_MODULES[n](model) for n in GRAPH_NAMES}

    @torch.no_grad()
    def run(self, name, feed):
        args = (torch.from_numpy(np.ascontiguousarray(feed[k], np.float32))
                for k in GRAPH_IO[name][0])
        out = self.mods[name](*args)
        out = out if isinstance(out, tuple) else (out,)
        return [o.detach().numpy() for o in out]


class OnnxBackend:
    """Исполнитель графов экспорта на ONNX Runtime.

    Args:
        model_dir: каталог экспорта с графами и манифестом.
        threads: число потоков внутри операции.
    """

    def __init__(self, model_dir, threads=1):
        import onnxruntime as ort
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess, self.names = {}, {}
        for n in GRAPH_NAMES:
            g = self.manifest["graphs"][n]
            self.sess[n] = ort.InferenceSession(os.path.join(model_dir, g["file"]), so,
                                                providers=["CPUExecutionProvider"])
            self.names[n] = set(g["inputs"])

    def run(self, name, feed):
        return self.sess[name].run(None, {k: np.ascontiguousarray(a, np.float32)
                                          for k, a in feed.items() if k in self.names[name]})


def manifest_conformal(manifest, model_dir):
    """Конформная таблица экспорта по бинам лидов и длины истории.

    В экспорте таблица лежит развёрнутой по лидам для каждого бина длины истории; здесь
    она сворачивается обратно в бины, чтобы поправку применяла та же функция, что и
    везде в проекте.

    Args:
        manifest: манифест экспорта.
        model_dir: каталог экспорта.

    Returns:
        Таблица float32 формы (число бинов лидов, число бинов длины истории, число
        квантилей) или None, если в экспорте таблицы нет.

    Raises:
        ValueError: бины манифеста не совпадают с бинами кода; таблица не того размера,
            сдвигает медиану или меняется внутри бина лидов.
    """
    from mayak.metrics import I_MED, LEAD_BINS, conformal_table
    cal = manifest["calibration"]
    check_manifest_bins(manifest)
    if not cal.get("conformal"):
        return None
    d = manifest["dims"]
    los = [lo for lo, _hi in cal["history_bins"]]
    n = len(los) * d["horizon"] * d["n_quantiles"]
    table = np.fromfile(os.path.join(model_dir, cal["conformal"]), "<f4")
    if table.size != n:
        raise ValueError(f"{cal['conformal']}: {4 * table.size} Б, ожидалось {4 * n} "
                         f"(бины длины истории × лиды × квантили)")
    table = table.reshape(len(los), d["horizon"], d["n_quantiles"])
    if np.any(table[..., I_MED] != 0.0):
        raise ValueError(f"{cal['conformal']}: поправка медианы не нулевая - таблица сдвигает "
                         f"точечный прогноз; подгоните таблицу заново")
    shift = np.ascontiguousarray(np.moveaxis(table[:, [lo - 1 for lo, _ in LEAD_BINS]], 0, 1))
    if not np.array_equal(conformal_table(shift, los, d["horizon"]), table):
        raise ValueError(f"{cal['conformal']}: поправка меняется внутри бина лидов - таблица "
                         f"записана не этим экспортом")
    return shift


def check_manifest_bins(manifest):
    """Проверяет, что бины калибровки в манифесте совпадают с бинами кода.

    Args:
        manifest: манифест экспорта.

    Raises:
        ValueError: бины лидов или длины истории другие.
    """
    lead_bins, history_bins = calibration_bins()
    cal = manifest["calibration"]
    if cal.get("lead_bins") != lead_bins or cal.get("history_bins") != history_bins:
        raise ValueError(f"манифест: бины калибровки лидов {cal.get('lead_bins')} и длины "
                         f"истории {cal.get('history_bins')}, в коде {lead_bins} и "
                         f"{history_bins}; экспортируйте графы заново")


def runtime_from_export(model_dir, lat, lon, elev, threads=1, conformal=True, aci=False):
    """Устройство на графах экспорта.

    Пороги смены точки берутся из манифеста.

    Args:
        model_dir: каталог экспорта с графами и манифестом.
        lat: широта точки.
        lon: долгота точки.
        elev: высота точки, м.
        threads: число потоков исполнителя графов.
        conformal: применять конформную таблицу экспорта.
        aci: подстраивать множитель калибровки с параметрами из манифеста.

    Returns:
        Устройство на графах ONNX.

    Raises:
        ValueError: манифест другого формата или калибровка включена, а её параметров в
            манифесте нет.
        RuntimeError: граф климатологии не дал годной таблицы.
    """
    from mayak.config import ModelConfig, RuntimeConfig
    from mayak.metrics import ACIParams
    backend = OnnxBackend(model_dir, threads)
    man = backend.manifest
    if man.get("format") != GRAPH_FORMAT:
        raise ValueError(f"формат манифеста {man.get('format')}, рантайм читает {GRAPH_FORMAT}; "
                         f"экспортируйте графы заново: python scripts/export_runtime.py")
    check_manifest_bins(man)
    table = manifest_conformal(man, model_dir) if conformal else None
    params = None
    if aci:
        a = man["calibration"].get("aci")
        if a is None:
            raise ValueError("ACI включена, но параметров ACI в манифесте нет")
        params = ACIParams(target=a["target"], gamma=a["gamma"], max_factor=a["max_factor"])
        if list(params.interval) != list(a["interval"]):
            raise ValueError(f"манифест: интервал ACI {a['interval']} не соответствует цели "
                             f"{a['target']}")
    return Device(backend, ModelConfig.from_dict(man["model_config"]), lat, lon, elev,
                  conformal=table, aci=params,
                  runtime_cfg=RuntimeConfig.from_dict(man["runtime"]))


__all__ = ["GRAPH_FORMAT", "GRAPH_IO", "GRAPH_NAMES", "ClimatologyGraph", "ForecastGraph",
           "OnnxBackend", "TorchBackend", "calibration_bins", "check_manifest_bins",
           "conformal_for_export", "default_start", "dims", "eval_inputs", "eval_set",
           "example_inputs", "export_graphs", "feed", "manifest_conformal",
           "runtime_from_export", "synthetic_series", "year_calendar"]
