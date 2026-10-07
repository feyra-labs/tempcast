"""Отчёт о поле после этапа холодного старта.

Отчёт считается для лучшего по метрике выбора чекпойнта этапа на наборе валидации этапа:
валидационные станции, валидационное окно, нулевая история. Это тот же набор, по
которому выбирался чекпойнт. Считаются:

* отношение MSE медианы модели к MSE эмпирической климатологии станции по всем парам
  «окно и лид», его интервал блочного бутстрапа по станциям и то же отношение в среднем
  по станциям;
* покрытие 90- и 80-процентного интервалов, средняя ширина 90-процентного интервала,
  средняя абсолютная ошибка и CRPS;
* скилл, покрытие и ошибка по лидам;
* станции с худшим отношением;
* разрыв обобщения поля: отношение MSE медианы на валидационных станциях к MSE медианы на
  обучающих станциях в том же валидационном окне при той же нулевой истории, с интервалом
  бутстрапа по станциям.

Вместе с числами отчёт записывает ограничители запоминания координат из конфига модели
(``loc_freq_max`` и ``field_weight_decay``), если они у архитектуры есть. Отчёт
информационный: автоматических порогов в нём нет, на ход обучения он не влияет.

Числа считаются по сырым выходам моделей и нужны только медиана и квантили, поэтому
отчёт одинаков для всех архитектур. Графики строятся по тем же числам: кривые обучения
с отметкой лучшего чекпойнта, метрики по лидам и примеры прогнозов.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

from mayak.stages import jsonable

log = logging.getLogger(__name__)

REPORT_FILE = "report.json"
REPORT_DIR = "report"
REPORT_LEADS = (1, 3, 6, 12, 24, 48, 72, 120, 168)
SUMMARY_LEADS = (24, 72, 168)
N_BOOT = 200
CI_LEVEL = 0.90
N_WORST = 5
N_EXAMPLES = 6
COVERAGE_BAND = (0.86, 0.94)
# Отношение MSE медианы к MSE климатологии станции, при котором модель не лучше неё.
CLIM_LEVEL = 1.0
PLOT_FILES = ("curves.png", "leads.png", "examples.png")
LIMIT_FIELDS = ("loc_freq_max", "field_weight_decay")


def report_device(accelerator):
    """Устройство для прогона моделей в отчёте.

    Args:
        accelerator: ускоритель обучения.

    Returns:
        Строка cuda, если обучение шло на видеокарте и она доступна, иначе cpu.
    """
    import torch
    if str(accelerator) in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def describe_checkpoint(path):
    """Короткая запись о чекпойнте: путь, отпечаток, шаг и метрика выбора.

    Args:
        path: путь к чекпойнту.

    Returns:
        Словарь с полями ckpt, digest, step, val_loss и stage.
    """
    import torch

    from mayak.leakage import SELECTION_KEY, file_digest
    ck = torch.load(path, map_location="cpu", weights_only=False)
    scores = (ck.get(SELECTION_KEY) or {}).get("scores") or {}
    return jsonable(dict(ckpt=os.path.abspath(path), digest=file_digest(path),
                         step=int(ck.get("global_step", -1)), val_loss=scores.get("val/loss"),
                         stage=(ck.get("hyper_parameters") or {}).get("stage")))


def collect_outputs(model, dataset, device="cpu", batch_size=256):
    """Медиана и квантили модели на наборе окон.

    Args:
        model: модель.
        dataset: набор окон.
        device: устройство.
        batch_size: размер батча.

    Returns:
        Пара: медиана формы (N, H) и квантили формы (N, H, 7); данные окон: цель, маска
        цели, климатология на горизонте, станция и час начала горизонта каждого окна.
    """
    import torch
    from torch.utils.data import DataLoader
    model.eval().to(device)
    mu, q, y, w, muc = [], [], [], [], []
    with torch.no_grad():
        for b in DataLoader(dataset, batch_size=batch_size):
            y.append(b["y"].numpy())
            w.append(b["y_mask"].numpy())
            muc.append(b["mu_clim_fut"].numpy())
            o = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
            mu.append(o["mu"].float().cpu().numpy())
            q.append(o["q"].float().cpu().numpy())
    cat = lambda parts: np.concatenate(parts, 0)
    pred = dict(mu=cat(mu), q=cat(q))
    aux = dict(y=cat(y), y_mask=cat(w), mu_clim=cat(muc),
               station=np.array([sid for sid, _t in dataset.items], object),
               t=np.array([t for _sid, t in dataset.items], np.int64))
    return pred, aux


def field_metrics(pred, aux, seed=0, n_boot=N_BOOT, leads=REPORT_LEADS, n_worst=N_WORST):
    """Числа отчёта о поле для одной модели.

    Args:
        pred: медиана и квантили модели.
        aux: цель, маска цели, климатология и станции тех же окон.
        seed: сид бутстрапа.
        n_boot: число выборок бутстрапа.
        leads: лиды таблицы по лидам, ч.
        n_worst: сколько станций с худшим отношением показать.

    Returns:
        Словарь чисел; нечисла записаны как None.
    """
    from mayak.metrics import EPS, Evaluation, metric_table, quantile_ci, wmean
    y = np.asarray(aux["y"], np.float64)
    w = np.asarray(aux["y_mask"], np.float64)
    muc = np.asarray(aux["mu_clim"], np.float64)
    mu = np.asarray(pred["mu"], np.float64)
    q = np.asarray(pred["q"], np.float64)
    ev = Evaluation(y=y, mu=mu, q=q, mu_clim=muc, w=w, station=aux["station"])
    pooled = ev.pooled()
    mse_model = float(wmean((mu - y) ** 2, w))
    mse_clim = float(wmean((muc - y) ** 2, w))
    ratio = mse_model / mse_clim if mse_clim > EPS else float("nan")
    ci = [float("nan"), float("nan")]
    boot = ev.bootstrap_samples(n_boot=n_boot, seed=seed)
    if boot is not None:
        ci = list(quantile_ci({"r": 1.0 - boot[0]["Skill"]}, level=CI_LEVEL)["r"])
    st, per = ev.per_station()
    st_ratio = 1.0 - np.asarray(per["Skill"], np.float64)
    finite = np.isfinite(st_ratio)
    macro = float(st_ratio[finite].mean()) if finite.any() else float("nan")
    order = np.argsort(-np.where(finite, st_ratio, -np.inf))[:n_worst]
    worst = [dict(station=str(st[i]), mse_ratio=float(st_ratio[i])) for i in order if finite[i]]
    table = metric_table(y, mu, q, muc, w, leads=leads)
    by_lead = [dict(lead=int(h), Skill=table[h]["Skill"], MAE=table[h]["MAE"],
                    CRPS=table[h]["CRPS"], PICP90=table[h]["PICP90"],
                    n_valid=table[h]["n_valid"]) for h in leads]
    return jsonable(dict(mse_model=mse_model, mse_clim=mse_clim, mse_ratio=ratio,
                         mse_ratio_ci=ci, mse_ratio_macro=macro, picp90=pooled["PICP90"],
                         picp80=pooled["PICP80"], width90=pooled["Width90"],
                         crps=pooled["CRPS"], mae=pooled["MAE"], rmse=pooled["RMSE"],
                         by_lead=by_lead, worst_stations=worst, **ev.counts()))


def _evaluation(pred, aux):
    from mayak.metrics import Evaluation
    return Evaluation(y=np.asarray(aux["y"], np.float64), mu=np.asarray(pred["mu"], np.float64),
                      q=np.asarray(pred["q"], np.float64),
                      mu_clim=np.asarray(aux["mu_clim"], np.float64),
                      w=np.asarray(aux["y_mask"], np.float64), station=aux["station"])


def memorization_gap(val_pred, val_aux, train_pred, train_aux, seed=0, n_boot=N_BOOT):
    """Разрыв обобщения поля для одной модели.

    Отношение пуловой MSE медианы на окнах валидационных станций к пуловой MSE медианы на
    окнах обучающих станций. Интервал - блочный бутстрап по станциям: станции каждой роли
    перевыбираются независимо, в каждой выборке берётся отношение пуловых MSE.

    Args:
        val_pred: медиана и квантили модели на окнах валидационных станций.
        val_aux: данные тех же окон.
        train_pred: медиана и квантили модели на окнах обучающих станций или None, если
            таких окон нет.
        train_aux: данные окон обучающих станций или None.
        seed: сид бутстрапа.
        n_boot: число выборок бутстрапа.

    Returns:
        Словарь: отношение gap_ratio, его интервал gap_ratio_ci, MSE медианы на обучающих
        станциях gap_mse_train и число обучающих станций с валидными парами
        gap_stations_train; нечисла записаны как None.
    """
    from mayak.metrics import EPS, quantile_ci
    nan = float("nan")
    ev_val = _evaluation(val_pred, val_aux)
    ev_train = None if train_pred is None else _evaluation(train_pred, train_aux)
    mse_val = float(ev_val.pooled()["RMSE"]) ** 2
    mse_train = nan if ev_train is None else float(ev_train.pooled()["RMSE"]) ** 2
    ratio = mse_val / mse_train if mse_train > EPS else nan
    ci = [nan, nan]
    if ev_train is not None:
        boot_val = ev_val.bootstrap_samples(n_boot=n_boot, seed=seed)
        # Обучающие станции перевыбираются своим потоком, независимо от валидационных.
        boot_train = ev_train.bootstrap_samples(n_boot=n_boot, seed=seed + 1)
        if boot_val is not None and boot_train is not None:
            m_val, m_train = boot_val[0]["RMSE"] ** 2, boot_train[0]["RMSE"] ** 2
            ok = m_train > EPS
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(ok, m_val / np.where(ok, m_train, 1.0), np.nan)
            ci = list(quantile_ci({"r": r}, level=CI_LEVEL)["r"])
    n_train = 0 if ev_train is None else ev_train.counts()["n_stations"]
    return jsonable(dict(gap_ratio=ratio, gap_ratio_ci=ci, gap_mse_train=mse_train,
                         gap_stations_train=n_train))


def memorization_limits(model):
    """Ограничители запоминания координат из конфига модели.

    Args:
        model: модель.

    Returns:
        Словарь значений полей ``LIMIT_FIELDS`` или None, если в конфиге модели их нет.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None or not all(hasattr(cfg, k) for k in LIMIT_FIELDS):
        return None
    return {k: float(getattr(cfg, k)) for k in LIMIT_FIELDS}


def _set_record(dataset):
    return dict(time_key=dataset.time_key, station_role=list(dataset.station_splits),
                windows=len(dataset), stations=len({sid for sid, _t in dataset.items}),
                fingerprint=dataset.fingerprint(), history=dataset.history_spec())


def check_gap_set(dataset, train_dataset):
    """Проверяет, что набор обучающих станций сопоставим с набором валидации.

    Args:
        dataset: набор валидации этапа.
        train_dataset: набор окон обучающих станций.

    Raises:
        ValueError: временное окно или правило длины истории различаются.
    """
    a, b = _set_record(dataset), _set_record(train_dataset)
    diff = [k for k in ("time_key", "history") if a[k] != b[k]]
    if diff:
        raise ValueError(f"разрыв обобщения: набор обучающих станций отличается от набора "
                         f"валидации полями {diff}: {[a[k] for k in diff]} против "
                         f"{[b[k] for k in diff]}")


def skill_at(entry, lead):
    """Скилл записи отчёта на лиде.

    Args:
        entry: запись отчёта о чекпойнте.
        lead: лид, ч.

    Returns:
        Скилл или None, если лида нет в таблице.
    """
    for row in (entry or {}).get("by_lead", []):
        if row.get("lead") == lead:
            return row.get("Skill")
    return None


def pick_examples(pred, aux, n=N_EXAMPLES, seed=0):
    """Окна для примеров прогнозов.

    Args:
        pred: медиана и квантили модели.
        aux: данные окон.
        n: число окон.
        seed: сид выбора окон.

    Returns:
        Словарь массивов выбранных окон или None, если окон с валидной целью нет.
    """
    w = aux["y_mask"]
    ok = np.flatnonzero((w > 0).any(1))
    if n <= 0 or ok.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ok, size=min(int(n), ok.size), replace=False))
    return dict(station=[str(s) for s in aux["station"][idx]],
                t=[int(v) for v in aux["t"][idx]],
                y=np.where(w[idx] > 0, aux["y"][idx], np.nan),
                mu_clim=aux["mu_clim"][idx], mu=pred["mu"][idx], q=pred["q"][idx])


def build_field_report(best, dataset, arch, stage, train_dataset, device="cpu", seed=0,
                       n_boot=N_BOOT, n_examples=N_EXAMPLES):
    """Отчёт о поле для лучшего чекпойнта этапа.

    Разрыв обобщения считается по набору валидации и набору окон обучающих станций в том
    же временном окне.

    Args:
        best: запись о лучшем чекпойнте этапа (``describe_checkpoint``).
        dataset: набор валидации этапа.
        arch: архитектура.
        stage: имя этапа.
        train_dataset: набор окон обучающих станций по правилу набора валидации.
        device: устройство.
        seed: сид бутстрапа и выбора примеров.
        n_boot: число выборок бутстрапа.
        n_examples: число окон с примерами прогнозов.

    Returns:
        Пара: отчёт, пригодный для записи в JSON, и массивы примеров для графиков.

    Raises:
        ValueError: набор обучающих станций не сопоставим с набором валидации.
    """
    from mayak.lit import load_model
    check_gap_set(dataset, train_dataset)
    model = load_model(best["ckpt"])
    pred, aux = collect_outputs(model, dataset, device=device)
    train_pred, train_aux = ((None, None) if len(train_dataset) == 0
                             else collect_outputs(model, train_dataset, device=device))
    entry = dict(best, **field_metrics(pred, aux, seed=seed, n_boot=n_boot))
    entry.update(memorization_gap(pred, aux, train_pred, train_aux, seed=seed, n_boot=n_boot))
    report = dict(arch=arch, stage=stage,
                  created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  val_set=_set_record(dataset), train_set=_set_record(train_dataset),
                  limits=memorization_limits(model), leads=list(REPORT_LEADS), best=entry)
    return jsonable(report), pick_examples(pred, aux, n_examples, seed)


def write_report(path, report):
    """Записать отчёт в JSON атомарно.

    Args:
        path: путь к файлу.
        report: отчёт.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def summary(entry):
    """Короткая сводка записи отчёта для журнала.

    Args:
        entry: запись отчёта о чекпойнте или None.

    Returns:
        Словарь основных чисел или None.
    """
    if entry is None:
        return None
    out = {k: entry.get(k) for k in ("step", "mse_ratio", "mse_ratio_ci", "mse_ratio_macro",
                                     "picp90", "width90", "gap_ratio", "gap_ratio_ci")}
    for h in SUMMARY_LEADS:
        out[f"skill_{h}h"] = skill_at(entry, h)
    return jsonable(out)


def _num(v, digits=3):
    return "—" if v is None or not np.isfinite(float(v)) else f"{float(v):.{digits}f}"


def _pct(v):
    return "—" if v is None or not np.isfinite(float(v)) else f"{100 * float(v):.1f}%"


def format_field_report(report):
    """Таблица отчёта о поле для консоли.

    Args:
        report: отчёт.

    Returns:
        Список строк.
    """
    e = report.get("best") or {}
    vs = report.get("val_set") or {}
    lines = [f"Отчёт этапа {report.get('stage')} ({report.get('arch')}): лучший по val/loss "
             f"чекпойнт, окон {vs.get('windows')} на {vs.get('stations')} валидационных "
             f"станциях, L=0"]
    head = (f"{'шаг':>8} {'val/loss':>9} {'MSE/клим':>9} {'интервал 90%':>17} "
            f"{'по станц.':>9} {'PICP90':>7} {'шир.90':>7}"
            + "".join(f" {'Skill ' + str(h) + 'ч':>11}" for h in SUMMARY_LEADS)
            + f" {'вал/обуч':>9} {'интервал 90%':>17}")
    lines.append(head)
    ci = e.get("mse_ratio_ci") or [None, None]
    gci = e.get("gap_ratio_ci") or [None, None]
    lines.append(f"{e.get('step', -1):>8} {_num(e.get('val_loss'), 4):>9} "
                 f"{_num(e.get('mse_ratio')):>9} "
                 f"{'[' + _num(ci[0]) + ', ' + _num(ci[1]) + ']':>17} "
                 f"{_num(e.get('mse_ratio_macro')):>9} {_pct(e.get('picp90')):>7} "
                 f"{_num(e.get('width90'), 2):>7}"
                 + "".join(f" {_num(skill_at(e, h)):>11}" for h in SUMMARY_LEADS)
                 + f" {_num(e.get('gap_ratio')):>9}"
                 + f" {'[' + _num(gci[0]) + ', ' + _num(gci[1]) + ']':>17}")
    if e.get("worst_stations"):
        worst = ", ".join(f"{w['station']} {_num(w['mse_ratio'])}" for w in e["worst_stations"])
        lines.append(f"Худшие станции: {worst}")
    ts = report.get("train_set") or {}
    lines.append(f"Разрыв обобщения вал/обуч: MSE медианы на валидационных станциях к MSE на "
                 f"{ts.get('stations')} обучающих станциях ({ts.get('windows')} окон) в том же "
                 f"окне {vs.get('time_key')}, L=0.")
    limits = report.get("limits")
    lines.append("Ограничители запоминания координат: "
                 + (", ".join(f"{k} {v:g}" for k, v in limits.items()) if limits
                    else "в конфиге модели нет"))
    lines.append(f"Отсчёт MSE/клим {CLIM_LEVEL:.1f} — уровень климатологии станции; цель "
                 f"покрытия 90%-интервала {COVERAGE_BAND[0]:.0%}–{COVERAGE_BAND[1]:.0%}. "
                 f"Отчёт информационный, на ход обучения не влияет.")
    return lines


def _arr(values):
    return np.array([np.nan if v is None else float(v) for v in values], np.float64)


def _plot_curves(plt, report, metrics_csv, out_dir, title):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    best = report["best"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for col, label, kw in (("train/pinball", "обучение", dict(lw=1.0, alpha=0.6)),
                           ("val/loss", "валидация", dict(marker=".", ms=7, lw=1.5))):
        if col in df.columns:
            sub = df[["step", col]].dropna()
            ax.plot(sub["step"], sub[col], label=label, **kw)
    ax.axvline(best["step"], color="tab:red", lw=1.2, ls="--",
               label=f"лучший по val/loss, шаг {best['step']}")
    ax.set_xlabel("шаг")
    ax.set_ylabel("нормированный pinball")
    ax.set_title(f"{title}: кривые обучения", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = os.path.join(out_dir, PLOT_FILES[0])
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_leads(plt, report, out_dir, title):
    rows = report["best"].get("by_lead", [])
    leads = [row["lead"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(leads, _arr(row.get("Skill") for row in rows), "o-", ms=3)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_title("Скилл по лидам", fontsize=10)
    axes[1].plot(leads, _arr(row.get("PICP90") for row in rows), "o-", ms=3)
    axes[1].axhspan(*COVERAGE_BAND, color="tab:green", alpha=0.12, label="цель для 90%")
    axes[1].set_title("Покрытие 90%-интервала по лидам", fontsize=10)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("лид, ч")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}: по лидам, лучший по val/loss", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, PLOT_FILES[1])
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_examples(plt, report, examples, out_dir, title):
    n = len(examples["station"])
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows), squeeze=False)
    axes = axes.ravel()
    leads = np.arange(1, examples["y"].shape[1] + 1)
    for k in range(n):
        ax = axes[k]
        q = examples["q"][k]
        ax.fill_between(leads, q[:, 0], q[:, -1], color="tab:blue", alpha=0.15,
                        label="90%-интервал")
        ax.plot(leads, examples["y"][k], color="black", lw=1.5, label="факт")
        ax.plot(leads, examples["mu_clim"][k], color="tab:red", lw=1.0, ls="--",
                label="климатология станции")
        ax.plot(leads, examples["mu"][k], color="tab:blue", lw=2.0,
                label=f"медиана, шаг {report['best']['step']}")
        ax.set_title(f"{examples['station'][k]}, час {examples['t'][k]}", fontsize=9)
        ax.set_xlabel("лид, ч")
        ax.set_ylabel("T, °C")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}: примеры прогнозов при L=0 на валидационных станциях", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, PLOT_FILES[2])
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_field_report(report, examples=None, metrics_csv=None, out_dir="."):
    """Графики отчёта о поле.

    Args:
        report: отчёт.
        examples: массивы примеров прогнозов или None.
        metrics_csv: журнал метрик обучения этапа или None.
        out_dir: каталог для картинок.

    Returns:
        Список путей к картинкам.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    title = f"{report.get('arch')}, этап {report.get('stage')}"
    paths = []
    if metrics_csv and os.path.isfile(metrics_csv):
        paths.append(_plot_curves(plt, report, metrics_csv, out_dir, title))
    paths.append(_plot_leads(plt, report, out_dir, title))
    if examples is not None:
        paths.append(_plot_examples(plt, report, examples, out_dir, title))
    return paths


def write_stage_report(stage_dir, stage, arch, best, dataset, train_dataset, device="cpu",
                       seed=0, metrics_csv=None, n_examples=N_EXAMPLES):
    """Считает отчёт о поле для лучшего чекпойнта этапа, пишет его и строит графики.

    Сбой графиков не отменяет отчёт: он пишется в журнал и в поле plots_error.

    Args:
        stage_dir: каталог этапа.
        stage: имя этапа.
        arch: архитектура.
        best: запись о лучшем чекпойнте (``describe_checkpoint``).
        dataset: набор валидации этапа.
        train_dataset: набор окон обучающих станций по правилу набора валидации.
        device: устройство.
        seed: сид бутстрапа и примеров.
        metrics_csv: журнал метрик обучения этапа или None.
        n_examples: число окон с примерами прогнозов.

    Returns:
        Тройка: отчёт, путь к нему и пути к картинкам.
    """
    report, examples = build_field_report(best, dataset, arch=arch, stage=stage,
                                          train_dataset=train_dataset, device=device,
                                          seed=seed, n_examples=n_examples)
    path = os.path.join(stage_dir, REPORT_FILE)
    plots = []
    try:
        plots = plot_field_report(report, examples, metrics_csv,
                                  os.path.join(stage_dir, REPORT_DIR))
    except Exception as e:
        # Сбой картинок не отменяет посчитанный отчёт.
        log.exception("графики отчёта этапа %s не построены", stage)
        report["plots_error"] = f"{type(e).__name__}: {e}"
    report["plots"] = plots
    write_report(path, report)
    return report, path, plots


__all__ = ["CLIM_LEVEL", "LIMIT_FIELDS", "PLOT_FILES", "REPORT_DIR", "REPORT_FILE",
           "SUMMARY_LEADS", "build_field_report", "check_gap_set", "collect_outputs",
           "describe_checkpoint", "field_metrics", "format_field_report", "memorization_gap",
           "memorization_limits", "pick_examples", "plot_field_report", "report_device",
           "skill_at", "summary", "write_report", "write_stage_report"]
