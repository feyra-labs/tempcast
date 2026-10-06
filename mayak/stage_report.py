"""Отчёт о поле после этапа холодного старта: по нему человек выбирает старт этапа B.

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
* станции с худшим отношением;
* разрыв обобщения поля: отношение MSE медианы на валидационных станциях к MSE медианы на
  обучающих станциях в том же валидационном окне при той же нулевой истории, с интервалом
  бутстрапа по станциям.

Вместе с числами отчёт записывает ограничители запоминания координат из конфига модели
(``loc_freq_max`` и ``field_weight_decay``), если они у архитектуры есть. Автоматических
порогов в отчёте нет: числа и графики только для человека, который выбирает кандидата.

Числа считаются по сырым выходам моделей и нужны только медиана и квантили, поэтому
отчёт одинаков для всех архитектур. Графики строятся по тем же числам: кривые обучения
с отметками сохранённых чекпойнтов, метрики по шагам, метрики по лидам и примеры
прогнозов нескольких чекпойнтов на одних и тех же окнах.
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
PLOT_FILES = ("curves.png", "candidates.png", "leads.png", "examples.png")
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


def build_field_report(items, dataset, arch, stage, train_dataset, device="cpu", seed=0,
                       n_boot=N_BOOT, n_examples=N_EXAMPLES):
    """Отчёт о поле для нескольких чекпойнтов этапа на одном наборе окон.

    Разрыв обобщения каждого чекпойнта считается по набору валидации и набору окон
    обучающих станций в том же временном окне.

    Args:
        items: записи о чекпойнтах: имя, путь, отпечаток, шаг, метрика выбора и пометка
            лучшего.
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
    models = {e["name"]: load_model(e["ckpt"]) for e in items}
    preds, aux = collect_outputs(models, dataset, device=device)
    train_preds, train_aux = ((None, None) if len(train_dataset) == 0
                              else collect_outputs(models, train_dataset, device=device))
    entries = []
    for e in items:
        entry = dict(e, **field_metrics(preds[e["name"]], aux, seed=seed, n_boot=n_boot))
        entry.update(memorization_gap(preds[e["name"]], aux,
                                      None if train_preds is None else train_preds[e["name"]],
                                      train_aux, seed=seed, n_boot=n_boot))
        entries.append(jsonable(entry))
    best = next((e["name"] for e in entries if e.get("is_best")), None)
    limits = memorization_limits(models[best if best is not None else items[-1]["name"]])
    report = dict(arch=arch, stage=stage,
                  created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  val_set=_set_record(dataset), train_set=_set_record(train_dataset),
                  limits=limits, leads=list(REPORT_LEADS), candidates=entries, best=best)
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
    ents = report.get("candidates") or []
    vs = report.get("val_set") or {}
    lines = [f"Отчёт этапа {report.get('stage')} ({report.get('arch')}): чекпойнтов "
             f"{len(ents)}, окон {vs.get('windows')} на {vs.get('stations')} валидационных "
             f"станциях, L=0"]
    head = (f"{'шаг':>8} {'val/loss':>9} {'MSE/клим':>9} {'интервал 90%':>17} "
            f"{'по станц.':>9} {'PICP90':>7} {'шир.90':>7}"
            + "".join(f" {'Skill ' + str(h) + 'ч':>11}" for h in SUMMARY_LEADS)
            + f" {'вал/обуч':>9} {'интервал 90%':>17}")
    lines.append(head)
    for e in ents:
        ci = e.get("mse_ratio_ci") or [None, None]
        gci = e.get("gap_ratio_ci") or [None, None]
        row = (f"{e['step']:>8} {_num(e.get('val_loss'), 4):>9} {_num(e.get('mse_ratio')):>9} "
               f"{'[' + _num(ci[0]) + ', ' + _num(ci[1]) + ']':>17} "
               f"{_num(e.get('mse_ratio_macro')):>9} {_pct(e.get('picp90')):>7} "
               f"{_num(e.get('width90'), 2):>7}"
               + "".join(f" {_num(skill_at(e, h)):>11}" for h in SUMMARY_LEADS)
               + f" {_num(e.get('gap_ratio')):>9}"
               + f" {'[' + _num(gci[0]) + ', ' + _num(gci[1]) + ']':>17}")
        if e.get("is_best"):
            row += "  * лучший по val/loss"
        lines.append(row)
    best = next((e for e in ents if e.get("is_best")), None)
    if best and best.get("worst_stations"):
        worst = ", ".join(f"{w['station']} {_num(w['mse_ratio'])}"
                          for w in best["worst_stations"])
        lines.append(f"Худшие станции у лучшего по val/loss: {worst}")
    ts = report.get("train_set") or {}
    lines.append(f"Разрыв обобщения вал/обуч: MSE медианы на валидационных станциях к MSE на "
                 f"{ts.get('stations')} обучающих станциях ({ts.get('windows')} окон) в том же "
                 f"окне {vs.get('time_key')}, L=0; автоматического порога нет.")
    limits = report.get("limits")
    lines.append("Ограничители запоминания координат: "
                 + (", ".join(f"{k} {v:g}" for k, v in limits.items()) if limits
                    else "в конфиге модели нет"))
    lines.append(f"Отсчёт MSE/клим {CLIM_LEVEL:.1f} — уровень климатологии станции; цель "
                 f"покрытия 90%-интервала {COVERAGE_BAND[0]:.0%}–{COVERAGE_BAND[1]:.0%}. "
                 f"Кандидата для старта этапа B выбирает человек.")
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
    ax.axhline(CLIM_LEVEL, color="black", lw=1.0, ls="--",
               label="уровень климатологии станции")
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


def write_stage_report(stage_dir, stage, arch, best, candidates, dataset, train_dataset,
                       device="cpu", seed=0, metrics_csv=None, n_examples=N_EXAMPLES):
    """Считает отчёт о поле для чекпойнтов этапа, пишет его и строит графики.

    Сбой графиков не отменяет отчёт: он пишется в журнал и в поле plots_error.

    Args:
        stage_dir: каталог этапа.
        stage: имя этапа.
        arch: архитектура.
        best: запись о лучшем чекпойнте.
        candidates: записи о кандидатах.
        dataset: набор валидации этапа.
        train_dataset: набор окон обучающих станций по правилу набора валидации.
        device: устройство.
        seed: сид бутстрапа и примеров.
        metrics_csv: журнал метрик обучения этапа или None.
        n_examples: число окон с примерами прогнозов.

    Returns:
        Тройка: отчёт, путь к нему и пути к картинкам.
    """
    items = report_items(best, candidates)
    report, examples = build_field_report(items, dataset, arch=arch, stage=stage,
                                          train_dataset=train_dataset, device=device,
                                          seed=seed, n_examples=n_examples)
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


__all__ = ["CLIM_LEVEL", "LIMIT_FIELDS", "PLOT_FILES", "REPORT_DIR", "REPORT_FILE",
           "SUMMARY_LEADS", "build_field_report", "check_gap_set", "collect_outputs",
           "describe_checkpoint", "field_metrics", "find_report_entry", "format_field_report",
           "memorization_gap", "memorization_limits", "pick_examples", "plot_field_report",
           "read_report", "report_device", "report_items", "skill_at", "summary",
           "write_report", "write_stage_report"]
