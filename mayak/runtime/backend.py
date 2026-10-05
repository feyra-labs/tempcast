"""Графы экспорта на устройстве: имена и входы графов, исполнитель ONNX Runtime, манифест.

Экспорт пишет в каталог два графа без состояния и манифест:

* ``forecast``    - проход модели по окну истории, на выходе квантили до калибровки;
* ``climatology`` - таблица климатологии точки для отката на каждый час високосного
  года, считается один раз при старте устройства.

Здесь графы только исполняются: модуль зависит от numpy и onnxruntime и не загружает
torch. Построение графов и их исполнение в PyTorch - в ``mayak.export``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from mayak.runtime.device import CLIMATOLOGY_INPUTS, FORECAST_INPUTS, Device

GRAPH_FORMAT = 7
GRAPH_NAMES = ("forecast", "climatology")
GRAPH_IO = {
    "forecast": (FORECAST_INPUTS, ("q",)),
    "climatology": (CLIMATOLOGY_INPUTS, ("clim_mu", "clim_sig")),
}


def calibration_bins():
    """Бины лидов и длины истории калибровки в том виде, в каком они пишутся в манифест.

    Returns:
        Пара списков пар границ включительно: бины лидов и бины длины истории.
    """
    from mayak.constants import HISTORY_BINS
    from mayak.metrics import LEAD_BINS
    return ([[int(a), int(b)] for a, b in LEAD_BINS],
            [[int(b[0]), int(b[1])] for b in HISTORY_BINS])


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


__all__ = ["GRAPH_FORMAT", "GRAPH_IO", "GRAPH_NAMES", "OnnxBackend", "calibration_bins",
           "check_manifest_bins", "manifest_conformal", "runtime_from_export"]
