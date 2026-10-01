"""Отчёт о поле после этапа холодного старта и сравнение запусков следующего этапа.

Отчёт считается на наборе валидации этапа: валидационные станции, валидационное окно,
нулевая история. Это тот же набор, по которому выбирался чекпойнт. В отчёт идёт каждый
сохранённый чекпойнт этапа, а не только лучший по метрике выбора: чекпойнт с меньшим
числом шагов иногда оказывается лучшей стартовой точкой для следующего этапа, и выбирать
между ними должен человек. По каждому чекпойнту считаются:

* отношение MSE медианы модели к MSE эмпирической климатологии станции по всем парам
  «окно и лид», его интервал блочного бутстрапа по станциям и то же отношение в среднем
  по станциям;
* покрытие 90- и 80-процентного интервалов, средняя ширина 90-процентного интервала,
  средняя абсолютная ошибка и CRPS;
* скилл, покрытие и ошибка по лидам;
* станции с худшим отношением.

Числа считаются по сырым выходам моделей и нужны только медиана и квантили, поэтому
отчёт одинаков для всех архитектур. Графики строятся по тем же числам: кривые обучения
с отметками сохранённых чекпойнтов, метрики по шагам, метрики по лидам и примеры
прогнозов нескольких чекпойнтов на одних и тех же окнах.

Сравнение запусков следующего этапа читает их журналы и кривые валидации и показывает,
с какого чекпойнта стартовал каждый запуск и чего он достиг, в том числе на общем для
всех шаге.
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
FIELD_GATE_REFERENCE = 1.05
PLOT_FILES = ("curves.png", "candidates.png", "leads.png", "examples.png")


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


def report_items(best, candidates):
    """Список чекпойнтов для отчёта: все кандидаты и лучший по метрике выбора.

    Кандидат, сохранённый на том же шаге, что и лучший, несёт те же веса и помечается
    как лучший; отдельной записи у лучшего тогда нет.

    Args:
        best: запись о лучшем чекпойнте.
        candidates: записи о кандидатах.

    Returns:
        Записи, упорядоченные по шагу.
    """
    items, found = [], False
    for c in sorted(candidates, key=lambda c: c["step"]):
        e = dict(name=f"step{c['step']:06d}", ckpt=c["ckpt"], digest=c["digest"],
                 step=c["step"], val_loss=c.get("val_loss"), is_best=False)
        if c["step"] == best["step"]:
            e.update(is_best=True, best_ckpt=best["ckpt"], best_digest=best["digest"])
            found = True
        items.append(e)
    if not found:
        items.append(dict(name=f"best{best['step']:06d}", ckpt=best["ckpt"],
                          digest=best["digest"], step=best["step"],
                          val_loss=best.get("val_loss"), is_best=True, best_ckpt=best["ckpt"],
                          best_digest=best["digest"]))
    return sorted(items, key=lambda e: e["step"])


def collect_outputs(models, dataset, device="cpu", batch_size=256):
    """Медиана и квантили всех моделей на наборе окон за один проход.

    Args:
        models: словарь из имени в модель.
        dataset: набор окон.
        device: устройство.
        batch_size: размер батча.

    Returns:
        Пара: словарь из имени модели в медиану формы (N, H) и квантили формы (N, H, 7);
        данные окон: цель, маска цели, климатология на горизонте, станция и час начала
        горизонта каждого окна.
    """
    import torch
    from torch.utils.data import DataLoader
    for m in models.values():
        m.eval().to(device)
    outs = {n: dict(mu=[], q=[]) for n in models}
    y, w, muc = [], [], []
    with torch.no_grad():
        for b in DataLoader(dataset, batch_size=batch_size):
            y.append(b["y"].numpy())
            w.append(b["y_mask"].numpy())
            muc.append(b["mu_clim_fut"].numpy())
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            for n, m in models.items():
                o = m(bb)
                outs[n]["mu"].append(o["mu"].float().cpu().numpy())
                outs[n]["q"].append(o["q"].float().cpu().numpy())
    cat = lambda parts: np.concatenate(parts, 0)
    preds = {n: {k: cat(v) for k, v in d.items()} for n, d in outs.items()}
    aux = dict(y=cat(y), y_mask=cat(w), mu_clim=cat(muc),
               station=np.array([sid for sid, _t in dataset.items], object),
               t=np.array([t for _sid, t in dataset.items], np.int64))
    return preds, aux


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


def pick_examples(preds, aux, n=N_EXAMPLES, seed=0):
    """Окна для примеров прогнозов: одни и те же для всех чекпойнтов.

    Args:
        preds: словарь из имени модели в медиану и квантили.
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
                mu_clim=aux["mu_clim"][idx],
                preds={k: dict(mu=v["mu"][idx], q=v["q"][idx]) for k, v in preds.items()})


def build_field_report(items, dataset, arch, stage, device="cpu", seed=0, threshold=None,
                       n_boot=N_BOOT, n_examples=N_EXAMPLES):
    """Отчёт о поле для нескольких чекпойнтов этапа на одном наборе окон.

    Args:
        items: записи о чекпойнтах: имя, путь, отпечаток, шаг, метрика выбора и пометка
            лучшего.
        dataset: набор валидации этапа.
        arch: архитектура.
        stage: имя этапа.
        device: устройство.
        seed: сид бутстрапа и выбора примеров.
        threshold: порог ворот или None.
        n_boot: число выборок бутстрапа.
        n_examples: число окон с примерами прогнозов.

    Returns:
        Пара: отчёт, пригодный для записи в JSON, и массивы примеров для графиков.
    """
    from mayak.lit import load_model
    from mayak.stages import gate_verdict
    models = {e["name"]: load_model(e["ckpt"]) for e in items}
    preds, aux = collect_outputs(models, dataset, device=device)
    entries = []
    for e in items:
        entry = dict(e, **field_metrics(preds[e["name"]], aux, seed=seed, n_boot=n_boot))
        entry["gate"] = None if threshold is None else gate_verdict(entry, threshold)
        entries.append(jsonable(entry))
    best = next((e["name"] for e in entries if e.get("is_best")), None)
    report = dict(arch=arch, stage=stage,
                  created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  val_set=dict(time_key=dataset.time_key,
                               station_role=list(dataset.station_splits),
                               windows=len(dataset),
                               stations=len({sid for sid, _t in dataset.items}),
                               fingerprint=dataset.fingerprint(),
                               history=dataset.history_spec()),
                  reference_gate=FIELD_GATE_REFERENCE, gate=threshold,
                  leads=list(REPORT_LEADS), candidates=entries, best=best)
    return jsonable(report), pick_examples(preds, aux, n_examples, seed)


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


def read_report(path):
    """Отчёт из JSON.

    Args:
        path: путь к файлу.

    Returns:
        Отчёт.
    """
    with open(path) as f:
        return json.load(f)


def find_report_entry(run_dir, stage_name, digest, val_digest):
    """Запись отчёта о поле для чекпойнта с заданным отпечатком.

    Args:
        run_dir: каталог прогона, где сохранён чекпойнт.
        stage_name: имя этапа.
        digest: отпечаток файла чекпойнта.
        val_digest: отпечаток набора валидации, на котором нужен отчёт.

    Returns:
        Пара: запись и путь к отчёту; запись None, если отчёта нет, он посчитан на
        другом наборе или чекпойнта в нём нет.
    """
    from mayak.stages import stage_dir_name
    if not run_dir:
        return None, None
    path = os.path.join(run_dir, stage_dir_name(stage_name), REPORT_FILE)
    if not os.path.isfile(path):
        return None, None
    report = read_report(path)
    if (report.get("val_set") or {}).get("fingerprint") != val_digest:
        return None, path
    for e in report.get("candidates", []):
        if digest in (e.get("digest"), e.get("best_digest")):
            return e, path
    return None, path


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
                                     "picp90", "width90")}
    for h in SUMMARY_LEADS:
        out[f"skill_{h}h"] = skill_at(entry, h)
    out["gate"] = entry.get("gate")
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
    ents = report.get("candidates") or []
    vs = report.get("val_set") or {}
    gate = report.get("gate")
    lines = [f"Отчёт этапа {report.get('stage')} ({report.get('arch')}): чекпойнтов "
             f"{len(ents)}, окон {vs.get('windows')} на {vs.get('stations')} валидационных "
             f"станциях, L=0"]
    head = (f"{'шаг':>8} {'val/loss':>9} {'MSE/клим':>9} {'интервал 90%':>17} "
            f"{'по станц.':>9} {'PICP90':>7} {'шир.90':>7}"
            + "".join(f" {'Skill ' + str(h) + 'ч':>11}" for h in SUMMARY_LEADS)
            + ("  ворота" if gate is not None else ""))
    lines.append(head)
    for e in ents:
        ci = e.get("mse_ratio_ci") or [None, None]
        row = (f"{e['step']:>8} {_num(e.get('val_loss'), 4):>9} {_num(e.get('mse_ratio')):>9} "
               f"{'[' + _num(ci[0]) + ', ' + _num(ci[1]) + ']':>17} "
               f"{_num(e.get('mse_ratio_macro')):>9} {_pct(e.get('picp90')):>7} "
               f"{_num(e.get('width90'), 2):>7}"
               + "".join(f" {_num(skill_at(e, h)):>11}" for h in SUMMARY_LEADS))
        if gate is not None:
            row += "  " + ("пройдены" if (e.get("gate") or {}).get("passed") else "закрыты")
        if e.get("is_best"):
            row += "  * лучший по val/loss"
        lines.append(row)
    best = next((e for e in ents if e.get("is_best")), None)
    if best and best.get("worst_stations"):
        worst = ", ".join(f"{w['station']} {_num(w['mse_ratio'])}"
                          for w in best["worst_stations"])
        lines.append(f"Худшие станции у лучшего по val/loss: {worst}")
    lines.append(f"Ориентир спецификации для поля: отношение не выше "
                 f"{report.get('reference_gate')}; цель покрытия 90%-интервала "
                 f"{COVERAGE_BAND[0]:.0%}–{COVERAGE_BAND[1]:.0%}.")
    return lines


def _arr(values):
    return np.array([np.nan if v is None else float(v) for v in values], np.float64)


def _mark_best(ax, best):
    if best is not None:
        ax.axvline(best["step"], color="tab:red", lw=1.0, ls="--", alpha=0.7)


def _plot_curves(plt, report, metrics_csv, out_dir, title):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    ents = report["candidates"]
    best = next((e for e in ents if e.get("is_best")), None)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for col, label, kw in (("train/pinball", "обучение", dict(lw=1.0, alpha=0.6)),
                           ("val/loss", "валидация", dict(marker=".", ms=7, lw=1.5))):
        if col in df.columns:
            sub = df[["step", col]].dropna()
            ax.plot(sub["step"], sub[col], label=label, **kw)
    for e in ents:
        ax.axvline(e["step"], color="grey", lw=0.6, ls=":")
    if best is not None:
        ax.axvline(best["step"], color="tab:red", lw=1.2, ls="--",
                   label=f"лучший по val/loss, шаг {best['step']}")
    ax.set_xlabel("шаг")
    ax.set_ylabel("нормированный pinball")
    ax.set_title(f"{title}: кривые обучения; пунктир — сохранённые чекпойнты", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = os.path.join(out_dir, PLOT_FILES[0])
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_candidates(plt, report, out_dir, title):
    ents = report["candidates"]
    best = next((e for e in ents if e.get("is_best")), None)
    x = np.array([e["step"] for e in ents], np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    ax = axes[0, 0]
    r = _arr(e.get("mse_ratio") for e in ents)
    lo = _arr((e.get("mse_ratio_ci") or [None, None])[0] for e in ents)
    hi = _arr((e.get("mse_ratio_ci") or [None, None])[1] for e in ents)
    yerr = np.nan_to_num(np.clip(np.vstack([r - lo, hi - r]), 0.0, None))
    ax.errorbar(x, r, yerr=yerr, fmt="o-", capsize=3, label="по всем парам, интервал 90%")
    ax.plot(x, _arr(e.get("mse_ratio_macro") for e in ents), "s--", ms=4,
            label="в среднем по станциям")
    ax.axhline(1.0, color="black", lw=0.8, label="климатология")
    ax.axhline(report.get("reference_gate", FIELD_GATE_REFERENCE), color="tab:orange", ls="--",
               lw=1.0, label="ориентир спецификации")
    if report.get("gate") is not None:
        ax.axhline(report["gate"], color="tab:red", lw=1.2, label="порог ворот")
    ax.set_title("MSE медианы / MSE климатологии при L=0", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(x, _arr(e.get("picp90") for e in ents), "o-", label="90%")
    ax.plot(x, _arr(e.get("picp80") for e in ents), "o-", label="80%")
    ax.axhspan(*COVERAGE_BAND, color="tab:green", alpha=0.12, label="цель для 90%")
    ax.axhline(0.8, color="grey", lw=0.8, ls=":")
    ax.set_title("Покрытие интервалов при L=0, сырые выходы", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for h in SUMMARY_LEADS:
        ax.plot(x, _arr(skill_at(e, h) for e in ents), "o-", label=f"лид {h} ч")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Скилл относительно климатологии", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    ax.plot(x, _arr(e.get("val_loss") for e in ents), "o-")
    ax.set_title("Метрика выбора val/loss", fontsize=10)
    for ax in axes.ravel():
        _mark_best(ax, best)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("шаг чекпойнта")
    fig.suptitle(f"{title}: чекпойнты этапа; красный пунктир — лучший по val/loss", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, PLOT_FILES[1])
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _colors(plt, n):
    cmap = plt.get_cmap("viridis")
    return [cmap(v) for v in np.linspace(0.05, 0.9, max(n, 1))]


def _plot_leads(plt, report, out_dir, title):
    ents = report["candidates"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for e, c in zip(ents, _colors(plt, len(ents))):
        leads = [row["lead"] for row in e.get("by_lead", [])]
        kw = dict(color=c, lw=2.4 if e.get("is_best") else 1.2, marker="o", ms=3,
                  label=f"шаг {e['step']}" + (" *" if e.get("is_best") else ""))
        axes[0].plot(leads, _arr(row.get("Skill") for row in e.get("by_lead", [])), **kw)
        axes[1].plot(leads, _arr(row.get("PICP90") for row in e.get("by_lead", [])), **kw)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_title("Скилл по лидам", fontsize=10)
    axes[1].axhspan(*COVERAGE_BAND, color="tab:green", alpha=0.12)
    axes[1].set_title("Покрытие 90%-интервала по лидам", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("лид, ч")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{title}: по лидам, * — лучший по val/loss", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, PLOT_FILES[2])
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_examples(plt, report, examples, out_dir, title):
    ents = report["candidates"]
    n = len(examples["station"])
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows), squeeze=False)
    axes = axes.ravel()
    leads = np.arange(1, examples["y"].shape[1] + 1)
    best = next((e for e in ents if e.get("is_best")), ents[-1])
    colors = dict(zip([e["name"] for e in ents], _colors(plt, len(ents))))
    for k in range(n):
        ax = axes[k]
        qb = examples["preds"][best["name"]]["q"][k]
        ax.fill_between(leads, qb[:, 0], qb[:, -1], color="tab:blue", alpha=0.15,
                        label="90%-интервал лучшего")
        ax.plot(leads, examples["y"][k], color="black", lw=1.5, label="факт")
        ax.plot(leads, examples["mu_clim"][k], color="tab:red", lw=1.0, ls="--",
                label="климатология станции")
        for e in ents:
            mu = examples["preds"][e["name"]]["mu"][k]
            ax.plot(leads, mu, color=colors[e["name"]], lw=2.0 if e is best else 0.9,
                    label=f"медиана, шаг {e['step']}")
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
    p = os.path.join(out_dir, PLOT_FILES[3])
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
    paths.append(_plot_candidates(plt, report, out_dir, title))
    paths.append(_plot_leads(plt, report, out_dir, title))
    if examples is not None:
        paths.append(_plot_examples(plt, report, examples, out_dir, title))
    return paths


def write_stage_report(stage_dir, stage, arch, best, candidates, dataset, device="cpu", seed=0,
                       threshold=None, metrics_csv=None, n_examples=N_EXAMPLES):
    """Считает отчёт о поле для чекпойнтов этапа, пишет его и строит графики.

    Сбой графиков не отменяет отчёт: он пишется в журнал и в поле plots_error.

    Args:
        stage_dir: каталог этапа.
        stage: имя этапа.
        arch: архитектура.
        best: запись о лучшем чекпойнте.
        candidates: записи о кандидатах.
        dataset: набор валидации этапа.
        device: устройство.
        seed: сид бутстрапа и примеров.
        threshold: порог ворот или None.
        metrics_csv: журнал метрик обучения этапа или None.
        n_examples: число окон с примерами прогнозов.

    Returns:
        Тройка: отчёт, путь к нему и пути к картинкам.
    """
    items = report_items(best, candidates)
    report, examples = build_field_report(items, dataset, arch=arch, stage=stage,
                                          device=device, seed=seed, threshold=threshold,
                                          n_examples=n_examples)
    path = os.path.join(stage_dir, REPORT_FILE)
    plots = []
    try:
        plots = plot_field_report(report, examples, metrics_csv,
                                  os.path.join(stage_dir, REPORT_DIR))
    except Exception as e:
        # Картинки нужны человеку, но их сбой не должен отменять посчитанный отчёт.
        log.exception("графики отчёта этапа %s не построены", stage)
        report["plots_error"] = f"{type(e).__name__}: {e}"
    report["plots"] = plots
    write_report(path, report)
    return report, path, plots


def _val_curve(metrics_csv):
    if not metrics_csv or not os.path.isfile(metrics_csv):
        return [], []
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    if "val/loss" not in df.columns:
        return [], []
    sub = df[["step", "val/loss"]].dropna()
    return [int(s) for s in sub["step"]], [float(v) for v in sub["val/loss"]]


def stage_runs(run_dirs, stage="B"):
    """Сводка запусков одного этапа, начатых с разных чекпойнтов предыдущего.

    Args:
        run_dirs: каталоги прогонов.
        stage: имя этапа.

    Returns:
        Пара: строки сводки по запускам и предупреждения о несопоставимости.
    """
    from mayak.protocol import read_journal
    rows, warnings = [], []
    for d in run_dirs:
        j = read_journal(d)
        entry = next((s for s in reversed(j.get("stages", [])) if s.get("name") == stage), None)
        if entry is None:
            warnings.append(f"{d}: в журнале нет этапа {stage}")
            continue
        init = entry.get("init_from") or {}
        rep = init.get("report") or {}
        steps, values = _val_curve(entry.get("metrics_csv"))
        rows.append(dict(run=d, arch=j.get("arch"), protocol=j.get("protocol"),
                         val_set=entry.get("val_set"), init_ckpt=init.get("ckpt"),
                         init_step=init.get("step"), init_is_best=init.get("is_best"),
                         init_ratio=rep.get("mse_ratio"), init_picp90=rep.get("picp90"),
                         best_score=entry.get("best_score"), best_step=entry.get("best_step"),
                         steps_done=entry.get("steps_done"), probe_steps=entry.get("probe_steps"),
                         selection=entry.get("selection") or {}, steps=steps, values=values))
    for key, what in (("arch", "архитектура"), ("protocol", "протокол"),
                      ("val_set", "набор валидации")):
        if len({json.dumps(r[key], sort_keys=True) for r in rows}) > 1:
            warnings.append(f"{what} различается между запусками: сравнение некорректно")
    common = set.intersection(*(set(r["steps"]) for r in rows)) if rows else set()
    at = max(common) if common else None
    for r in rows:
        r["common_step"] = at
        r["loss_at_common"] = (None if at is None
                               else r["values"][len(r["steps"]) - 1 - r["steps"][::-1].index(at)])
    return rows, warnings


def format_stage_runs(rows, warnings=()):
    """Таблица сравнения запусков этапа для консоли.

    Args:
        rows: строки сводки.
        warnings: предупреждения о несопоставимости.

    Returns:
        Список строк.
    """
    at = rows[0]["common_step"] if rows else None
    lines = [f"{'запуск':<32} {'старт: шаг':>10} {'MSE/клим':>9} {'PICP90':>7} "
             f"{'лучший val/loss':>15} {'на шаге':>8} {'шагов':>13} "
             f"{('val/loss на ' + str(at)) if at is not None else '':>16}"]
    for r in rows:
        mark = " *" if r["init_is_best"] else ""
        probe = " (проба)" if r["probe_steps"] else ""
        lines.append(f"{os.path.basename(os.path.normpath(r['run'])):<32} "
                     f"{str(r['init_step']) + mark:>10} {_num(r['init_ratio']):>9} "
                     f"{_pct(r['init_picp90']):>7} {_num(r['best_score'], 4):>15} "
                     f"{str(r['best_step']):>8} {str(r['steps_done']) + probe:>13} "
                     f"{_num(r['loss_at_common'], 4):>16}")
    lines.append("* старт с лучшего по val/loss чекпойнта предыдущего этапа")
    lines += [f"ВНИМАНИЕ: {w}" for w in warnings]
    return lines


def plot_stage_runs(rows, out_path):
    """Кривые валидации запусков этапа и pinball по длинам истории на лучшем шаге.

    Args:
        rows: строки сводки.
        out_path: путь к картинке.

    Returns:
        Путь к картинке.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = _colors(plt, len(rows))
    bins = sorted({k for r in rows for k in r["selection"] if k.startswith("val/pinball_L")},
                  key=lambda k: int(k[len("val/pinball_L"):].split("-")[0]))
    width = 0.8 / max(len(rows), 1)
    for i, (r, c) in enumerate(zip(rows, colors)):
        label = f"старт с шага {r['init_step']}" + (" *" if r["init_is_best"] else "")
        axes[0].plot(r["steps"], r["values"], marker=".", color=c, label=label)
        vals = _arr(r["selection"].get(k) for k in bins)
        axes[1].bar(np.arange(len(bins)) + i * width, vals, width=width, color=c, label=label)
    if rows and rows[0]["common_step"] is not None:
        axes[0].axvline(rows[0]["common_step"], color="grey", ls=":", lw=1.0,
                        label="общий шаг")
    axes[0].set_xlabel("шаг")
    axes[0].set_ylabel("val/loss")
    axes[0].set_title("Валидация следующего этапа", fontsize=10)
    axes[1].set_xticks(np.arange(len(bins)) + 0.4 - width / 2)
    axes[1].set_xticklabels([k[len("val/pinball_"):] for k in bins], fontsize=8)
    axes[1].set_title("Нормированный pinball по длине истории на лучшем шаге", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


__all__ = ["FIELD_GATE_REFERENCE", "PLOT_FILES", "REPORT_DIR", "REPORT_FILE", "SUMMARY_LEADS",
           "build_field_report", "collect_outputs", "describe_checkpoint", "field_metrics",
           "find_report_entry", "format_field_report", "format_stage_runs", "pick_examples",
           "plot_field_report", "plot_stage_runs", "read_report", "report_device",
           "report_items", "skill_at", "stage_runs", "summary", "write_report",
           "write_stage_report"]
