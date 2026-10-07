"""Сплит-конформная таблица МАЯК.

Таблица подгоняется на валидационных станциях в блоках калибровки, разнесённых по всему
году. Длина истории каждого окна берётся из того распределения, на котором обучен
чекпойнт, генератором с фиксированным сидом, тем же, что у набора валидации.

Таблица разбита по бинам лидов и бинам длины истории: у каждого бина свои поправки,
подогнанные по окнам, у которых столько часов с валидной температурой во входе модели
после причинного QC. По этому же числу строку выбирают устройство и оценка, поэтому
после простоя или при редких наблюдениях интервал берётся из строки короткой истории.
Так номинальное покрытие держится в каждом режиме прибора, в том числе при полной
истории, в которой устройство проводит почти всё время. Бин, в котором окон меньше порога из конфига
калибровки, получает маргинальную строку по всем окнам; это пишется в запись о подгонке.

Поправка медианы равна нулю: таблица меняет только ширину интервалов, точечный прогноз
остаётся прогнозом модели.

Прогноз для подгонки считает модель из чекпойнта. Отпечаток чекпойнта пишется в запись
о подгонке рядом с таблицей. Рядом же ложится отчёт: покрытие до и после таблицы по
сезонам и длине истории.
"""
import argparse
import os

import numpy as np

from mayak.data.masking import DEFAULT_TARGET_MASK
from mayak.data.splits import ROLE_VAL
from mayak.data.store import get_store
from mayak.data.window import valid_history_hours
from mayak.evaluate import EvalSet, gather
from mayak.leakage import (CONFORMAL_TIME_KEY, SELECTION_KEY, LeakageError, conformal_record,
                           run_checklist, save_conformal)
from mayak.metrics import LEAD_BINS, apply_conformal, coverage, fit_conformal_shift

USAGE = """примеры:
  python scripts/calibrate.py --ckpt runs/mayak/stageB/best.ckpt --out runs/conformal.npy
  python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \\
      --conformal runs/conformal.npy --aci --out runtime/model
"""


def calibration_set(clims, manifest="data/manifest.csv", curriculum="full", cfg=None,
                    target_mask=DEFAULT_TARGET_MASK):
    """Окна, на которых подгоняется таблица.

    Args:
        clims: словарь станций набора.
        manifest: путь к манифесту с ролями станций.
        curriculum: распределение длины истории, на котором обучена модель.
        cfg: настройки калибровки; из них берутся шаг кандидатов, число окон на станцию
            и сид выбора длин истории.
        target_mask: правило годности цели окна.

    Returns:
        Набор окон валидационных станций в блоках калибровки.
    """
    from mayak.config import CalibrationConfig
    cfg = cfg or CalibrationConfig()
    return EvalSet(clims, station_splits=(ROLE_VAL,), manifest=manifest,
                   time_key=CONFORMAL_TIME_KEY, every_hours=cfg.fit_every_hours,
                   max_windows=None, windows_per_station=cfg.fit_windows_per_station,
                   target_mask=target_mask, curriculum=curriculum,
                   history_seed=cfg.fit_seed)


def checkpoint_setup(path):
    """Распределение длины истории и правило годности цели, с которыми обучен чекпойнт.

    Args:
        path: путь к чекпойнту.

    Returns:
        Пара: имя куррикулума последнего этапа и правило годности цели.

    Raises:
        LeakageError: в чекпойнте нет записи о выборе с куррикулумом.
    """
    import torch

    from mayak.config import DataConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    curriculum = ((ck.get(SELECTION_KEY) or {}).get("history") or {}).get("curriculum")
    if curriculum is None:
        raise LeakageError(f"{path}: в записи о выборе нет куррикулума; неизвестно, на какой "
                           f"смеси длин истории обучена модель")
    data = DataConfig.from_dict((ck.get("hyper_parameters") or {}).get("data_config"))
    return curriculum, data.target_mask


def fit(model, ds, checkpoint=None, min_windows=None):
    """Подгонка таблицы на калибровочном наборе.

    Args:
        model: модель из чекпойнта.
        ds: калибровочный набор окон.
        checkpoint: путь к чекпойнту для записи о подгонке.
        min_windows: наименьшее число окон бина длины истории для своей строки
            таблицы; None - из конфига калибровки по умолчанию.

    Returns:
        Тройка: таблица поправок, запись о подгонке и прогон набора (факт, квантили,
        веса, маски истории, число валидных часов температуры окон и остальное про окна).
    """
    from mayak.config import CalibrationConfig
    if min_windows is None:
        min_windows = CalibrationConfig().fit_min_windows
    D = gather(model, ds)
    D["history_valid"] = valid_history_hours(D["mask_hist"])
    shift, history_fit = fit_conformal_shift(D["y"], D["q"], D["y_mask"], D["history_valid"],
                                             LEAD_BINS, min_windows=min_windows)
    rec = conformal_record(ds, checkpoint=checkpoint, history_fit=history_fit)
    rec["min_windows"] = int(min_windows)
    return shift, rec, D


def report(D, meta, shift, cfg=None):
    """Покрытие калибровочного набора до и после таблицы по сезонам и длине истории.

    Разрезы по длине истории - по запрошенной длине из метаданных набора, строка таблицы -
    по числу валидных часов температуры окна из прогона.

    Args:
        D: прогон набора.
        meta: метаданные окон набора.
        shift: таблица поправок.
        cfg: настройки анализа калибровки.

    Returns:
        Отчёт о покрытии.
    """
    from mayak.calibration import evaluation_of, fit_report
    meta = dict(meta, history_valid=D["history_valid"])
    aux = dict(y=D["y"], y_mask=D["y_mask"], mu_clim=D["mu_clim"], meta=meta)
    pred = dict(mu=D["mu"], q=D["q"])
    return fit_report(evaluation_of(pred, aux), evaluation_of(pred, aux, shift), meta, cfg)


def print_table(shift, rec):
    """Печатает таблицу по бинам длины истории и отметку о маргинальных строках."""
    for k, row in enumerate(rec["history_fit"]):
        tag = "маргинальная строка" if row["marginal"] else "своя строка"
        print(f"\n{row['bin']}: окон {row['windows']}, {tag} (порог {rec['min_windows']})")
        print(np.round(shift[:, k], 3))


def main(argv=None):
    import logging

    from mayak.calibration import load_config, print_fit_report
    from mayak.lit import load_model
    from mayak.results import save_json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="сплит-конформная таблица МАЯК: валидационные станции, блоки калибровки "
                    "по всему году, длины истории по куррикулуму чекпойнта, поправки по "
                    "бинам лидов и бинам длины истории, медиана без поправки",
        epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/conformal.npy")
    ap.add_argument("--config", default=None,
                    help="YAML калибровки (по умолчанию conf/calibration/default.yaml)")
    ap.add_argument("--bootstrap", type=int, default=None,
                    help="повторов бутстрапа для отчёта (по умолчанию из конфига; 0 - без)")
    args = ap.parse_args(argv)

    from dataclasses import replace
    cfg = load_config(args.config)
    if args.bootstrap is not None:
        cfg = replace(cfg, bootstrap=args.bootstrap)
    store = get_store(args.manifest)
    curriculum, target_mask = checkpoint_setup(args.ckpt)
    ds = calibration_set(store.clims(), args.manifest, curriculum, cfg, target_mask)
    run_checklist(store, datasets=[ds], checkpoints=[args.ckpt])
    print(f"Калибровочный набор: окон {len(ds)}, станций {len({s for s, _t in ds.items})}, "
          f"куррикулум {curriculum!r}, сид {cfg.fit_seed}")

    model = load_model(args.ckpt)
    shift, rec, D = fit(model, ds, checkpoint=args.ckpt, min_windows=cfg.fit_min_windows)
    save_conformal(args.out, shift, rec)
    print("Таблица поправок по бинам длины истории (бины лидов × квантили), °C; столбец "
          "медианы - нули:")
    print_table(shift, rec)
    before = coverage(D["y"], D["q"], D["y_mask"])
    after = coverage(D["y"], apply_conformal(D["q"], shift, D["history_valid"]), D["y_mask"])
    print(f"PICP-90 на калибровочном наборе (в выборке): до {before:.1%}, после {after:.1%}")

    rep = report(D, ds.window_meta(), shift, cfg)
    print_fit_report(rep)
    path = os.path.splitext(args.out)[0] + ".report.json"
    save_json(dict(record=rec, report=rep), path)
    print("Запись о подгонке:", os.path.splitext(args.out)[0] + ".meta.json")
    print("Отчёт:", path)


if __name__ == "__main__":
    main()
