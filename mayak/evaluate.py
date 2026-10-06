"""Единый стенд оценки МАЯК.

Стенд собирает окна оценки, прогоняет по ним все модели и печатает таблицы.

Основной внутренний набор - станции unseen_test в их тестовом окне. По нему считаются
метрики по лидам, разрез по длине истории, сезоны и зоны, надёжность, раздел после
калибровки с офлайн-прогоном адаптивной калибровки и сопоставление с внешним тестом.
Обучающие станции в их тестовом окне - отдельный набор со своим каталогом результатов;
по нему считаются только метрики по лидам. Разрез по ролям станций собирается из двух
наборов: строка train - из набора обучающих станций, строка unseen_test - из основного.

Сравнение моделей идёт только по их сырым выходам. Основные таблицы, разрезы,
надёжность, острота против покрытия, графики и сохранённые предсказания не зависят от
того, задана ли конформная таблица. Таблица МАЯК даёт отдельный раздел «МАЯК после
калибровки»: сырой и откалиброванный МАЯК рядом, покрытие по тем же разрезам и
офлайн-прогон адаптивной калибровки устройства.

Каждое окно оценивается на сетке длин истории, от холодного старта до полной истории.
Основные таблицы считаются при полной истории. Разрез по длине истории и кривые скилла
по длине истории строятся для всех моделей с интервалами бутстрапа по станциям. Эталоны,
которым нужна история, видят ту же историю окна, что модели. Климатологии история не
нужна, поэтому она считается один раз и одинакова на всей сетке.

Кроме того, стенд считает таблицы в двух видах агрегирования (пуловом и макро) с
интервалами блочного бутстрапа по станциям, разрезы по зонам Кёппена, сезонам и доле
валидных часов истории, внешний тест на наблюдениях реальной сети с его собственными
разрезами и сопоставлением с внутренним тестом, проверки поля при холодном старте,
суточные амплитуды, строки переобученных абляций и разброс основной модели по сидам на
внутреннем и внешнем наборах.

Модели сравниваются на равных: у каждого чекпойнта есть запись о подборе скорости
обучения по одной и той же сетке с одним и тем же числом шагов. Прогон дополнительной
настройки МАЯК проверку не проходит и попадает в таблицы только по явному аргументу,
отдельной помеченной строкой рядом со строкой МАЯК.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from mayak import baselines as BL
from mayak.constants import H, QUANTILES
from mayak.data.holdout import (HISTORY_GRID, NOMINAL_HISTORY, EvalSet, check_history_grid,
                                history_label, history_strata)
from mayak.data.splits import ROLE_EXTERNAL, ROLE_TEST, ROLE_TRAIN
from mayak.data.window import valid_history_hours
from mayak.loss import NORM_SCALE_CLAMP
from mayak.metrics import (FINE_LEADS, LEAD_BINS, NQ, Evaluation, breakdown, by_lead, coverage,
                           metric_table, seed_spread)
from mayak.results import (evaluation_tables, lead_tables, run_record, set_record,
                           transfer_tables, write_tables)
from mayak.zones import normalize_zone

Q = np.array(QUANTILES, np.float32)

HIST_VALID_BINS = ((-0.01, 0.5, "<50%"), (0.5, 0.8, "50-80%"),
                   (0.8, 0.95, "80-95%"), (0.95, 1.01, "95-100%"))
MIN_WINDOWS, MIN_STATIONS = 20, 2
BOOTSTRAP = dict(n_boot=1000, seed=0, level=0.90)
TABLE_LEADS = (1, 3, 6, 12, 24, 48, 72, 120, 168)
HISTORY_LEADS = (24, 72, 168)
BREAKDOWN_LEAD = 24

MAIN_MODEL = "МАЯК"
TUNED_MODEL = "МАЯК (доп. настройка)†"
TUNED_NOTE = (
    "† МАЯК (доп. настройка) - этап 2 сравнения: гиперпараметры МАЯК дополнительно\n"
    "  настроены по валидации сверх общего бюджета подбора. Сравнение с бейзлайнами не\n"
    "  на равных; на равных - строка «МАЯК».")
TRAIN_DISTANCE_DIM = "расстояние до обучающей точки"
ROLE_DIM = "роль станции"
TRAIN_STATIONS_DIR = "train_stations"
CLIMATOLOGY, DAMPED, SEASONAL = "Климатология", "Damped persistence", "Seasonal-naive 24ч"
HISTORY_FREE = (CLIMATOLOGY,)

NEURAL_BASELINES = {"gru": "GRU seq2seq", "dlinear": "DLinear", "lru": "LRU",
                    "patchtst": "PatchTST"}

BENCHMARK_NOTE = (
    "Эталон скилла — эмпирическая климатология самой станции, подогнанная по её\n"
    "многолетнему обучающему окну. Её гармонический базис грубее базиса климат-поля\n"
    "(15 членов против 35), поэтому при короткой истории и на дальних лидах часть\n"
    "скилла моделей с календарём отражает разницу базисов.\n"
    "Пуловая метрика — по всем парам «окно × лид» сразу, её доминируют станции\n"
    "с большой изменчивостью; макро — среднее по станциям, «типичная станция».\n"
    "Все модели сравниваются по сырым выходам, без калибровки.")


def bin_label(value, bins):
    for lo, hi, name in bins:
        if lo <= value <= hi:
            return name
    return "прочее"


_AUX_KEYS = (("y", "y"), ("y_mask", "y_mask"), ("mu_clim", "mu_clim_fut"),
             ("a_recent", "a_recent"), ("a_recent_ok", "a_recent_ok"),
             ("norm_scale", "norm_scale"), ("x_hist", "x_hist"),
             ("mask_hist", "mask_hist"))
_OPTIONAL_AUX_KEYS = (("mu_ref", "mu_ref_fut"),)


@torch.no_grad()
def gather_all(named, dataset, device="cpu", batch_size=128):
    """Прогон всех моделей по набору окон за один проход.

    Окно собирается один раз и подаётся всем моделям. Это важно, потому что сборка окна
    с причинным QC дороже прогона небольшой модели.

    Args:
        named: словарь из имени модели в модель; может быть пустым.
        dataset: набор окон.
        device: устройство, на котором считают модели.
        batch_size: размер батча.

    Returns:
        Пара: словарь из имени модели в её медиану формы (N, H) и квантили формы
        (N, H, 7), и словарь с целью, маской цели, климатологией на горизонте, недавней
        аномалией, нормировочным масштабом на часах горизонта, историей окон и числом
        часов с валидной температурой во входе модели (history_valid, по нему
        выбирается строка конформной таблицы). Если набор окон даёт отдельный эталон
        скилла, он лежит в том же словаре под ключом mu_ref.
    """
    for m in named.values():
        m.eval().to(device)
    outs = {name: dict(mu=[], q=[]) for name in named}
    acc = {k: [] for k, _src in _AUX_KEYS}
    for b in DataLoader(dataset, batch_size=batch_size):
        for k, src in _AUX_KEYS:
            acc[k].append(b[src].numpy())
        for k, src in _OPTIONAL_AUX_KEYS:
            if src in b:
                acc.setdefault(k, []).append(b[src].numpy())
        if not named:
            continue
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        for name, m in named.items():
            o = m(bb)
            outs[name]["mu"].append(o["mu"].cpu().numpy())
            outs[name]["q"].append(o["q"].cpu().numpy())
    cat = lambda parts: np.concatenate(parts, 0)
    aux = {k: cat(v) for k, v in acc.items()}
    aux["history_valid"] = valid_history_hours(aux["mask_hist"])
    preds = {name: {k: cat(v) for k, v in d.items()} for name, d in outs.items()}
    return preds, aux


def gather(model, dataset, device="cpu", batch_size=128):
    """Прогон одной модели по набору окон.

    Args:
        model: модель.
        dataset: набор окон.
        device: устройство.
        batch_size: размер батча.

    Returns:
        Словарь: медиана, квантили и всё, что возвращает общий прогон про окна.
    """
    preds, aux = gather_all({"model": model}, dataset, device=device, batch_size=batch_size)
    return dict(aux, **preds["model"])


@torch.no_grad()
def _gather_full(model, ds, device="cpu", batch_size=128):
    """Как gather, но дополнительно тащит o, o_p, r, e и mu_c (нужны для L=0-проверки)."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size)
    keys = ["mu", "q", "o", "o_p", "r", "e", "mu_c"]
    acc = {k: [] for k in keys}
    acc["y"] = []
    acc["y_mask"] = []
    acc["mu_clim"] = []
    for b in dl:
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(bb)
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
        acc["y"].append(b["y"].numpy())
        acc["y_mask"].append(b["y_mask"].numpy())
        acc["mu_clim"].append(b["mu_clim_fut"].numpy())
    return {k: np.concatenate(v, 0) for k, v in acc.items()}


def collect_predictions(named, ds, device="cpu"):
    """Предсказания всех моделей и данные окон для таблиц.

    Args:
        named: словарь из имени модели в модель.
        ds: набор окон.
        device: устройство.

    Returns:
        Пара: словарь из имени модели в её медиану и квантили, и данные окон вместе с
        метаданными для разрезов. В метаданных, кроме меток набора, число часов с
        валидной температурой во входе модели (history_valid): по нему выбирается
        строка конформной таблицы.
    """
    preds, aux = gather_all(named, ds, device=device)
    # Новый словарь: набор может кэшировать свои метаданные между вызовами.
    aux["meta"] = dict(ds.window_meta(), history_valid=aux["history_valid"])
    return preds, aux


def baseline_scale(aux):
    """Масштаб интервалов статистических эталонов на часах горизонта.

    Это масштаб, на который функция потерь делит ошибку: масштаб остатка климатологии
    станции на часах горизонта, обрезанный теми же пределами. У всех моделей и эталонов
    поэтому один масштаб разброса.

    Args:
        aux: данные окон с нормировочным масштабом формы (N, H).

    Returns:
        Масштаб, °C, форма (N, H), float32.
    """
    return np.clip(np.asarray(aux["norm_scale"], np.float32), *NORM_SCALE_CLAMP)


def history_free_baselines(aux):
    """Эталоны, которым история окна не нужна.

    Args:
        aux: данные окон.

    Returns:
        Словарь из имени эталона в его медиану и квантили.
    """
    mu, q = BL.climatology_forecast(aux["mu_clim"], baseline_scale(aux))
    return {CLIMATOLOGY: dict(mu=mu, q=q)}


def history_baselines(aux, r_damped=None):
    """Эталоны, которые читают историю окна.

    Оба эталона видят ровно ту историю, что модели: усечённую до длины истории окна и
    прошедшую причинный QC. Недавняя аномалия затухающей персистентности считается при
    сборке окна по этой же истории.

    Args:
        aux: данные окон.
        r_damped: коэффициенты затухания по лидам; None значит без затухающей
            персистентности.

    Returns:
        Словарь из имени эталона в его медиану и квантили.
    """
    mucl, sig = aux["mu_clim"], baseline_scale(aux)
    out = {}
    if r_damped is not None:
        mu, q = BL.damped_persistence_forecast(aux["a_recent"], mucl, sig, r_damped,
                                               valid=aux["a_recent_ok"])
        out[DAMPED] = dict(mu=mu, q=q)
    mu, q = BL.seasonal_naive_forecast(aux["x_hist"], aux["mask_hist"], mucl, sig, period=24)
    out[SEASONAL] = dict(mu=mu, q=q)
    return out


def add_statistical_baselines(preds, aux, r_damped=None, history_free=None):
    """Добавляет к предсказаниям статистические эталоны.

    Args:
        preds: словарь предсказаний моделей; дополняется на месте.
        aux: данные окон.
        r_damped: коэффициенты затухания по лидам; None значит без затухающей
            персистентности.
        history_free: готовые предсказания эталонов, которым история не нужна. Их
            можно взять с любой длины истории тех же окон; None значит посчитать.

    Returns:
        Тот же словарь предсказаний.
    """
    preds.update(history_free if history_free is not None else history_free_baselines(aux))
    preds.update(history_baselines(aux, r_damped))
    return preds


def evaluation_for(pred, aux):
    """Оценка одной модели по её сырым выходам.

    Args:
        pred: медиана и квантили модели.
        aux: цель, маска цели, климатология и метаданные тех же окон.

    Returns:
        Оценка модели.
    """
    return Evaluation(y=aux["y"], mu=pred["mu"], q=pred["q"], mu_clim=aux["mu_clim"],
                      w=aux["y_mask"], station=aux["meta"]["station"])


def evaluations(preds, aux):
    return {name: evaluation_for(p, aux) for name, p in preds.items()}


def build_tables(preds, aux, leads=FINE_LEADS):
    """Пуловые таблицы метрик по лидам - вход для графиков кривых метрик.

    Args:
        preds: словарь: имя модели и её медиана и квантили.
        aux: факт, маска факта и прогноз климатологии.
        leads: лиды таблиц.

    Returns:
        Словарь: имя модели и таблица по лидам.
    """
    y, w, muc = aux["y"], aux["y_mask"], aux["mu_clim"]
    return {name: metric_table(y, p["mu"], p["q"], muc, w, leads=leads)
            for name, p in preds.items()}


def _fmt(value, metric):
    """Число в единицах метрики: скилл и покрытие - в процентах, остальное - в °C."""
    if not np.isfinite(value):
        return "—"
    if metric == "Skill":
        return f"{value:+.1%}"
    if metric in ("PICP80", "PICP90"):
        return f"{value:.1%}"
    return f"{value:.2f}"


def _cell(point, metric, ci_block=None):
    """Значение метрики и, если есть, её доверительный интервал в тех же единицах."""
    out = _fmt(point[metric], metric)
    if ci_block is not None:
        lo, hi = ci_block[metric]
        if np.isfinite(lo) and np.isfinite(hi):
            out += f" [{_fmt(lo, metric)}; {_fmt(hi, metric)}]"
    return out


def print_rows(rows, label="разрез", metrics=("Skill", "MAE", "RMSE", "CRPS", "PICP90"),
               ci=False):
    width = 24 if ci else 12
    head = f"{label:>20} {'окон':>7} {'ст.':>5} {'агрег.':>7}"
    for m in metrics:
        head += f" {m:>{width}}"
    print(head)
    for name, s in rows.items():
        cis = s.get("ci") if ci else None
        for agg, title in (("pooled", "пул"), ("macro", "макро")):
            line = f"{str(name):>20} {s['n_windows']:>7} {s['n_stations']:>5} {title:>7}"
            for m in metrics:
                line += f" {_cell(s[agg], m, cis and cis[agg]):>{width}}"
            print(line)


def all_breakdowns(ev, meta, leads=None, min_windows=MIN_WINDOWS, min_stations=MIN_STATIONS,
                   ci=False, history=None, **kw):
    """Разрезы одной модели по метаданным окон одного набора.

    Разреза по ролям станций здесь нет: набор состоит из станций одной роли, разрез по
    ролям собирает role_breakdown из отдельных наборов.

    Args:
        ev: оценка модели.
        meta: метаданные тех же окон.
        leads: лиды, на которых считаются метрики.
        min_windows: страта с меньшим числом окон не показывается.
        min_stations: страта с меньшим числом станций не показывается.
        ci: считать интервалы бутстрапа по станциям.
        history: готовые строки разреза по длине истории, посчитанные по сетке длин на
            тех же окнах. None значит строить разрез по метаданным набора.
        **kw: параметры бутстрапа.

    Returns:
        Словарь из имени разреза в словарь из метки страты в её сводку.
    """
    hist, hist_order = history_strata(meta)
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    kwargs = dict(leads=leads, min_windows=min_windows, min_stations=min_stations, ci=ci, **kw)
    return {
        "зона Кёппена": breakdown(ev, meta["zone"], **kwargs),
        "сезон": breakdown(ev, meta["season"], **kwargs),
        "длина истории": (history if history is not None
                          else breakdown(ev, hist, order=hist_order, **kwargs)),
        "валидность истории": breakdown(ev, hvalid, **kwargs),
    }


def role_breakdown(parts, lead=BREAKDOWN_LEAD, min_windows=MIN_WINDOWS,
                   min_stations=MIN_STATIONS, ci=False, **kw):
    """Разрез основной модели по ролям станций, собранный из отдельных наборов.

    Args:
        parts: пары из оценки основной модели на наборе и метаданных окон этого набора,
            в порядке строк разреза.
        lead: лид, ч.
        min_windows: страта с меньшим числом окон не показывается.
        min_stations: страта с меньшим числом станций не показывается.
        ci: считать интервалы бутстрапа по станциям.
        **kw: параметры бутстрапа.

    Returns:
        Словарь из роли станций в её сводку на лиде.
    """
    rows = {}
    for ev, meta in parts:
        rows.update(breakdown(ev, meta["role"], leads=[lead], min_windows=min_windows,
                              min_stations=min_stations, ci=ci, **kw))
    return rows


def external_breakdowns(ev, meta, leads=None, min_windows=MIN_WINDOWS,
                        min_stations=MIN_STATIONS, ci=False, **kw):
    """Разрезы, которые есть только у внешнего теста.

    Разреза по шагу отчётности нет: во внешний тест входят только почасовые станции.

    Args:
        ev: оценка одной модели.
        meta: метаданные тех же окон.
        leads: лиды, на которых считаются метрики.
        min_windows: страта с меньшим числом окон не показывается.
        min_stations: страта с меньшим числом станций не показывается.
        ci: считать интервалы бутстрапа по станциям.
        **kw: параметры бутстрапа.

    Returns:
        Словарь из имени разреза в словарь из метки страты в её сводку.
    """
    from mayak.external import TRAIN_DISTANCE_ORDER
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    kwargs = dict(leads=leads, min_windows=min_windows, min_stations=min_stations, ci=ci, **kw)
    return {
        "валидность истории": breakdown(ev, hvalid, **kwargs),
        TRAIN_DISTANCE_DIM: breakdown(ev, meta["train_distance"], order=TRAIN_DISTANCE_ORDER,
                                      **kwargs),
        "Δ высоты станция−ЦМР": breakdown(ev, meta["elev_gap"], **kwargs),
        "канал давления": breakdown(ev, meta["has_pressure"], **kwargs),
    }


def print_breakdowns(rows_by_dim):
    """Печатает разрезы; интервалы печатаются, если они посчитаны.

    Args:
        rows_by_dim: словарь из имени разреза в словарь из метки страты в её сводку.
    """
    for name, rows in rows_by_dim.items():
        print(f"\n--- разрез: {name} ---")
        if not rows:
            print("  (все страты меньше порога по числу окон или станций)")
            continue
        print_rows(rows, label=name, ci=any("ci" in r for r in rows.values()))


def print_reliability(ev, lead_bins=LEAD_BINS):
    """Печать гистограммы PIT по всему горизонту и по бинам лидов, надёжности и остроты.

    Args:
        ev: оценка одной модели.
        lead_bins: бины лидов.
    """
    pit = ev.pit_histogram()
    edges = ["<q05"] + [f"q{int(100 * Q[i]):02d}-q{int(100 * Q[i + 1]):02d}"
                        for i in range(NQ - 1)] + [">q95"]
    print("\n--- PIT: доля факта в бинах между квантилями (ожидание в скобках) ---")
    print("  " + " ".join(f"{e:>11}" for e in edges))
    print("  " + " ".join(f"{o:>5.1%}({e:>4.0%})"
                          for o, e in zip(pit["observed"], pit["expected"])))
    for name, p in ev.pit_by_lead_bin(lead_bins).items():
        print(f"  лиды {name:>7}: " + " ".join(f"{o:>6.1%}" for o in p["observed"]))

    rel = ev.reliability()
    print("\n--- диаграмма надёжности: P(факт ≤ квантиль) ---")
    print("  " + " ".join(f"{t:>8.0%}" for t in rel["nominal"]))
    print("  " + " ".join(f"{e:>8.1%}" for e in rel["empirical"]))

    print("\n--- острота против покрытия ---")
    print(f"{'номинал':>9} {'факт':>9} {'ширина, °C':>12}")
    for r in ev.sharpness_coverage():
        print(f"{r['nominal']:>9.0%} {r['coverage']:>9.1%} {r['width']:>12.2f}")


def print_seed_spread(evs_by_seed, lead=24):
    """Печатает разброс метрик основной модели по сидам на одном лиде.

    Args:
        evs_by_seed: оценки одной и той же модели, обученной с разными сидами.
        lead: лид, ч.

    Returns:
        Словарь из имени метрики в среднее, минимум, максимум и стандартное отклонение.
    """
    summaries = [ev.restrict(leads=[lead]).summary() for ev in evs_by_seed]
    spread = seed_spread(summaries)
    print(f"\n--- разброс по {len(evs_by_seed)} сидам, лид {lead} ч ---")
    print(f"{'метрика':>10} {'среднее':>10} {'мин':>10} {'макс':>10} {'ст.откл.':>10}")
    for m, s in spread.items():
        if not np.isfinite(s["mean"]):
            continue
        print(f"{m:>10} {s['mean']:>10.3f} {s['min']:>10.3f} {s['max']:>10.3f} {s['std']:>10.3f}")
    return dict(lead=int(lead), n_seeds=len(evs_by_seed), metrics=spread)


def seed_spread_table(models, res, ds, lead=24):
    """Разброс метрик основной модели по сидам на окнах одного набора при полной истории.

    Первая модель - основная: её предсказания уже есть в результате оценки набора.
    Остальные - та же модель, обученная с другими сидами; они прогоняются по тем же окнам и
    оцениваются по тем же целям и эталону. По этому разбросу проверяется значимость
    разницы абляций с полной моделью на том же наборе.

    Args:
        models: модели основной архитектуры с разными сидами, основная первой.
        res: результат оценки набора (``evaluate_set``) с основной моделью.
        ds: тот же набор окон.
        lead: лид, ч.

    Returns:
        Таблица разброса, как у ``print_seed_spread``.
    """
    preds, aux = res["bench"].preds, res["bench"].aux
    nominal = ds.with_history(NOMINAL_HISTORY)
    evs = [evaluation_for(preds[MAIN_MODEL], aux)]
    evs += [evaluation_for(dict(zip(("mu", "q"), _mu_q(m, nominal))), aux) for m in models[1:]]
    return print_seed_spread(evs, lead=lead)


def zone_breakdown(preds, aux, koppen=None, leads=(24, 72), model="МАЯК",
                   min_windows=MIN_WINDOWS, min_stations=MIN_STATIONS):
    """Разрез одной модели по полным зонам Кёппена.

    Args:
        preds: словарь: имя модели и её медиана и квантили.
        aux: факт, маска факта, прогноз климатологии и метаданные окон.
        koppen: зона каждого окна; None - из метаданных.
        leads: лиды разреза.
        model: имя модели.
        min_windows: страта с меньшим числом окон не показывается.
        min_stations: страта с меньшим числом станций не показывается.

    Returns:
        Словарь: зона, затем лид - сводка.
    """
    ev = evaluation_for(preds[model], aux)
    keys = aux["meta"]["zone"] if koppen is None else np.asarray(koppen)
    out = {}
    for h in leads:
        rows = breakdown(ev, keys, leads=[h], min_windows=min_windows, min_stations=min_stations)
        for z, s in rows.items():
            out.setdefault(z, {"n": s["n_windows"], "n_stations": s["n_stations"]})[h] = s
    return out


def print_zone_breakdown(rows, leads=(24, 72)):
    hdr = f"{'зона':>6} {'окон':>6} {'ст.':>4}"
    for h in leads:
        hdr += f" {'Sk@' + str(h) + 'пул':>10} {'Sk@' + str(h) + 'макро':>12} {'MAE@' + str(h):>9}"
    print(hdr)
    for z, d in rows.items():
        line = f"{z:>6} {d['n']:>6} {d['n_stations']:>4}"
        for h in leads:
            s = d.get(h)
            if s is None:
                line += f" {'—':>10} {'—':>12} {'—':>9}"
                continue
            line += (f" {s['pooled']['Skill']:>+10.1%} {s['macro']['Skill']:>+12.1%} "
                     f"{s['pooled']['MAE']:>9.2f}")
        print(line)


def plot_metric_curves(tables, out_dir="runs/plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["Skill", "MAE", "RMSE", "CRPS", "PICP90", "Winkler90"]
    paths = []
    for met in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name, tbl in tables.items():
            leads = sorted(tbl)
            ax.plot(leads, [tbl[h][met] for h in leads], marker="o", ms=3, label=name)
        if met == "PICP90":
            ax.axhspan(0.86, 0.94, alpha=0.12, color="green", label="цель 86–94%")
            ax.axhline(0.90, ls="--", lw=1, color="gray")
            ax.set_ylim(0, 1)
        if met == "Skill":
            ax.axhline(0.0, ls="--", lw=1, color="gray")
        ax.set_xlabel("лид, ч")
        ax.set_ylabel(met)
        ax.set_title(f"{met} по горизонту прогноза")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"metric_{met}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_reliability(ev, out_dir="runs/plots", name="МАЯК"):
    """Диаграмма надёжности и кривая «острота против покрытия» одной картинкой.

    Args:
        ev: оценка одной модели.
        out_dir: каталог картинок.
        name: имя модели в заголовке и имени файла.

    Returns:
        Путь к картинке.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    rel, sharp = ev.reliability(), ev.sharpness_coverage()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="идеал")
    a1.plot(rel["nominal"], rel["empirical"], marker="o", label=name)
    a1.set_xlabel("номинальный уровень")
    a1.set_ylabel("фактическая доля")
    a1.set_title("Диаграмма надёжности")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    cov = [r["coverage"] for r in sharp]
    wid = [r["width"] for r in sharp]
    a2.plot(cov, wid, marker="o")
    for r in sharp:
        a2.annotate(f"{r['nominal']:.0%}", (r["coverage"], r["width"]), fontsize=8,
                    textcoords="offset points", xytext=(4, 4))
    a2.set_xlabel("фактическое покрытие")
    a2.set_ylabel("средняя ширина интервала, °C")
    a2.set_title("Острота против покрытия")
    a2.grid(alpha=0.3)
    p = os.path.join(out_dir, "reliability.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_pit(ev, out_dir="runs/plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    pit = ev.pit_histogram()
    x = np.arange(len(pit["observed"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, pit["observed"], width=0.7, label="факт")
    ax.step(x, pit["expected"], where="mid", color="black", lw=1.2, label="ожидание")
    ax.set_xticks(x)
    ax.set_xticklabels(["<q05"] + [f"q{int(100 * Q[i]):02d}+" for i in range(NQ - 1)] + [">q95"],
                       fontsize=7)
    ax.set_ylabel("доля часов")
    ax.set_title("Гистограмма PIT по бинам квантилей")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    p = os.path.join(out_dir, "pit_histogram.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


@torch.no_grad()
def plot_forecast_examples(model, clims, manifest="data/manifest.csv", n=10,
                           out_dir="runs/plots", time_key="test",
                           station_splits=(ROLE_TEST,), seed=0, L=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    kw = {} if L is None else {"L": L}
    ds = EvalSet(clims, station_splits=station_splits, manifest=manifest,
                 time_key=time_key, **kw)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    leads = np.arange(1, H + 1)
    cols = 2
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(idx):
        item = ds[int(i)]
        batch = {key: (v[None] if torch.is_tensor(v) else v) for key, v in item.items()}
        out = model(batch)
        mu = out["mu"][0].cpu().numpy()
        q = out["q"][0].cpu().numpy()
        y = np.where(item["y_mask"].numpy() > 0, item["y"].numpy(), np.nan)  # дыры видны
        muc = item["mu_clim_fut"].numpy()
        sid, _t = ds.items[int(i)]
        zone = normalize_zone(ds.clims[sid]["koppen"])
        role = ds.roles[sid]
        ax = axes[k]
        ax.fill_between(leads, q[:, 0], q[:, 6], alpha=0.2, color="tab:blue", label="90%-интервал")
        ax.plot(leads, y, color="black", lw=1.6, label="факт")
        ax.plot(leads, mu, color="tab:blue", lw=1.5, label="МАЯК (медиана)")
        ax.plot(leads, muc, color="tab:red", lw=1.0, ls="--", label="климатология")
        ax.set_title(f"{sid} · зона {zone} · {role}", fontsize=9)
        ax.set_xlabel("лид, ч")
        ax.set_ylabel("T, °C")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.tight_layout()

    suff = "" if L is None else f"_L{L}"
    fig.suptitle(f"Прогноз МАЯК vs факт (примеры{', L=' + str(L) if L is not None else ''})",
                 y=1.0, fontsize=12)
    p = os.path.join(out_dir, f"forecast_examples{suff}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Сохранено:", p)
    return p


def diurnal_amplitude(series, w=None):
    n = series.shape[-1]
    t = np.arange(n)
    X = np.stack([np.ones(n), np.cos(2 * np.pi * t / 24.0), np.sin(2 * np.pi * t / 24.0)], -1)
    w = np.ones_like(series) if w is None else (np.asarray(w) > 0).astype(np.float64)
    ys = np.where(w > 0, series, 0.0)
    A = np.einsum("nh,hi,hj->nij", w, X, X) + 1e-9 * np.eye(3)
    b = np.einsum("nh,hi,nh->ni", w, X, ys)
    coef = np.linalg.solve(A, b[..., None])[..., 0]
    amp = np.hypot(coef[:, 1], coef[:, 2])
    return np.where(w.sum(-1) >= 6, amp, np.nan)


def plot_amplitude_scatter(model, clims, manifest="data/manifest.csv", out_dir="runs/plots",
                           time_key="test", max_points=4000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    D = gather(model, EvalSet(clims, manifest=manifest, time_key=time_key))
    ap = diurnal_amplitude(D["mu"], D["y_mask"])
    ar = diurnal_amplitude(D["y"], D["y_mask"])
    keep = np.isfinite(ap) & np.isfinite(ar)
    ap, ar = ap[keep], ar[keep]
    idx = np.random.default_rng(seed).choice(len(ap), min(max_points, len(ap)), replace=False)
    ap, ar = ap[idx], ar[idx]
    slope = float(np.polyfit(ar, ap, 1)[0])
    bias = float((ap - ar).mean())
    lim = float(max(ar.max(), ap.max())) * 1.05
    fig, axx = plt.subplots(figsize=(6, 6))
    axx.scatter(ar, ap, s=6, alpha=0.3)
    axx.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 (идеал)")
    axx.set_xlim(0, lim)
    axx.set_ylim(0, lim)
    axx.set_xlabel("факт: суточная амплитуда, °C")
    axx.set_ylabel("прогноз: суточная амплитуда, °C")
    axx.set_title(f"Суточная амплитуда\nнаклон={slope:.2f}, смещение={bias:+.2f}°C")
    axx.legend()
    axx.grid(alpha=0.3)
    p = os.path.join(out_dir, "amplitude_scatter.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Сохранено: {p}  (наклон {slope:.2f} — <1 значит модель ЗАНИЖАЕТ суточный ход)")
    return p


@dataclass
class Bench:
    """Прогон всех моделей по одним окнам на всей сетке длин истории.

    Attributes:
        grid: длины истории, ч, по возрастанию.
        preds: предсказания всех моделей при полной истории.
        aux: данные окон при полной истории.
        history: сводки по длинам истории: словарь из имени модели в словарь из длины в
            словарь из лида в сводку.
        main: имя основной модели.
        main_history: предсказания основной модели на каждой длине истории: словарь из
            длины в медиану, квантили и метаданные окон при этой длине.
    """

    grid: tuple
    preds: dict
    aux: dict
    history: dict
    main: str
    main_history: dict


def _same_targets(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ("y", "y_mask", "mu_clim", "norm_scale"))


def run_bench(named, base, grid=HISTORY_GRID, r_damped=None, leads=HISTORY_LEADS, ci=True,
              bootstrap=BOOTSTRAP, device="cpu", main=MAIN_MODEL):
    """Прогоняет модели и эталоны по окнам набора на каждой длине истории из сетки.

    Окна, цели и эталон одинаковы на всей сетке, меняется только длина истории. Эталоны
    без истории считаются один раз, их сводки на всех длинах совпадают. Длины
    обрабатываются по одной: в памяти остаются только сводки, предсказания при полной
    истории и предсказания основной модели.

    Args:
        named: словарь из имени модели в модель.
        base: набор окон.
        grid: длины истории, ч; должна включать полную историю.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        leads: лиды сводок по длине истории.
        ci: считать интервалы бутстрапа по станциям.
        bootstrap: параметры бутстрапа.
        device: устройство.
        main: имя основной модели.

    Returns:
        Результат прогона.

    Raises:
        RuntimeError: цели или эталон окон разошлись между длинами истории.
    """
    grid = check_history_grid(grid)
    kw = bootstrap if ci else {}
    history, main_history = {}, {}
    free, first, nominal = None, None, None
    for L in grid:
        ds = base.with_history(L)
        preds, aux = collect_predictions(named, ds, device=device)
        if first is None:
            first = aux
        elif not _same_targets(first, aux):
            raise RuntimeError(f"цели окон при L = {L} ч отличаются от целей при L = {grid[0]} "
                               f"ч: сетка длин истории должна менять только историю")
        preds = add_statistical_baselines(preds, aux, r_damped=r_damped, history_free=free)
        for name, p in preds.items():
            rows = history.setdefault(name, {})
            if free is not None and name in free:
                rows[L] = rows[grid[0]]
                continue
            ev = evaluation_for(p, aux)
            rows[L] = {int(h): ev.restrict(leads=[h]).summary(ci=ci, **kw) for h in leads}
        if free is None:
            free = {n: preds[n] for n in HISTORY_FREE}
        if main in preds:
            main_history[L] = dict(preds[main], meta=aux["meta"])
        if L == NOMINAL_HISTORY:
            nominal = (preds, aux)
    return Bench(grid=grid, preds=nominal[0], aux=nominal[1], history=history, main=main,
                 main_history=main_history)


def history_rows(bench, model, lead):
    """Разрез по длине истории одной модели на одном лиде.

    Args:
        bench: результат прогона по сетке.
        model: имя модели.
        lead: лид, ч.

    Returns:
        Словарь из метки длины истории в сводку, по возрастанию длины. В нём все длины
        сетки.
    """
    return {history_label(L): bench.history[model][L][int(lead)] for L in bench.grid}


def history_evaluations(bench, shift=None):
    """Оценки основной модели на каждой длине истории.

    Оценки создаются по одной, по мере чтения, чтобы в памяти не лежали все сразу.

    Args:
        bench: результат прогона по сетке.
        shift: конформная таблица; None значит сырые выходы. Строка выбирается по
            числу валидных часов температуры каждого окна при этой длине сетки.

    Yields:
        Пары из метки длины истории и оценки.
    """
    aux = bench.aux
    for L in bench.grid:
        p = bench.main_history[L]
        ev = Evaluation(y=aux["y"], mu=p["mu"], q=p["q"], mu_clim=aux["mu_clim"],
                        w=aux["y_mask"], station=p["meta"]["station"])
        yield history_label(L), ev.with_conformal(shift, p["meta"]["history_valid"])


def history_predictions(bench):
    """Предсказания основной модели по всей сетке, сложенные в один набор окон.

    Каждое окно повторяется столько раз, сколько длин в сетке; метаданные повтора - те,
    что у окна при этой длине истории.

    Args:
        bench: результат прогона по сетке.

    Returns:
        Пара из предсказаний основной модели и данных окон в том виде, в каком их
        сохраняют для анализа калибровки.
    """
    parts = [bench.main_history[L] for L in bench.grid]
    aux = bench.aux
    n = len(parts)
    pred = {k: np.concatenate([p[k] for p in parts], 0) for k in ("mu", "q")}
    out = {k: np.concatenate([aux[k]] * n, 0) for k in ("y", "y_mask", "mu_clim")}
    out["meta"] = {k: np.concatenate([p["meta"][k] for p in parts], 0)
                   for k in parts[0]["meta"]}
    return {bench.main: pred}, out


def calibration_config(ci, bootstrap):
    """Настройки анализа калибровки с параметрами бутстрапа стенда оценки.

    Args:
        ci: считать ли интервалы бутстрапа.
        bootstrap: число повторов и сид бутстрапа.

    Returns:
        Конфиг калибровки; без интервалов - с нулевым числом повторов.
    """
    from dataclasses import replace

    from mayak.calibration import load_config
    cfg = load_config()
    if not ci:
        return replace(cfg, bootstrap=0)
    return replace(cfg, bootstrap=int(bootstrap["n_boot"]), seed=int(bootstrap["seed"]),
                   ci_level=float(bootstrap["level"]))


def evaluate_set(named, base, grid=HISTORY_GRID, r_damped=None, shift=None, ci=True,
                 bootstrap=BOOTSTRAP, device="cpu", external=False, main=MAIN_MODEL):
    """Все числа стенда на одном наборе окон.

    Внутренний тест подаётся сюда набором станций unseen_test; обучающие станции
    оцениваются отдельно, только по лидам.

    Сравнительная часть считается по сырым выходам всех моделей и от конформной таблицы
    не зависит. Таблица влияет только на раздел основной модели после калибровки; строка
    таблицы у каждого окна - по числу часов с валидной температурой во входе модели, как
    на устройстве. Разрезы по длине истории остаются по запрошенной длине. В этом разделе же
    офлайн-прогон адаптивной калибровки: основная модель выпускает прогноз каждый час
    на непрерывном периоде части станций набора, как прибор.

    Args:
        named: словарь из имени модели в модель; основная модель обязательна.
        base: набор окон.
        grid: длины истории, ч.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        shift: конформная таблица основной модели; None значит без раздела после
            калибровки.
        ci: считать интервалы бутстрапа по станциям.
        bootstrap: параметры бутстрапа.
        device: устройство.
        external: набор внешнего теста.
        main: имя основной модели.

    Returns:
        Словарь. Сравнительная часть: прогон по сетке, таблицы по лидам и сводки по всему
        горизонту для всех моделей, разрезы по длине истории для всех моделей, разрезы
        основной модели, её надёжность, кривые остроты всех моделей и покрытие основной
        модели по разрезам. Отдельно: раздел после калибровки или None и ежечасные
        выпуски основной модели для сохранения или None.
    """
    from mayak.calibration import (aci_hourly_replay, calibration_effect, conditional_gate,
                                   coverage_report, sharpness_curves)
    bench = run_bench(named, base, grid=grid, r_damped=r_damped, ci=ci, bootstrap=bootstrap,
                      device=device, main=main)
    kw = bootstrap if ci else {}
    preds, aux = bench.preds, bench.aux
    meta = aux["meta"]
    evs = evaluations(preds, aux)
    cfg = calibration_config(ci, bootstrap)
    history = {n: {h: history_rows(bench, n, h) for h in HISTORY_LEADS} for n in preds}
    hist_main = {k: s for k, s in history[main][BREAKDOWN_LEAD].items()
                 if s["n_windows"] >= MIN_WINDOWS and s["n_stations"] >= MIN_STATIONS}
    breakdowns = all_breakdowns(evs[main], meta, leads=[BREAKDOWN_LEAD], history=hist_main)
    if external:
        breakdowns.update(external_breakdowns(evs[main], meta, leads=[BREAKDOWN_LEAD]))
    report = coverage_report(evs[main], meta, cfg, external=external,
                             history=history_evaluations(bench))
    res = dict(
        bench=bench, external=external, main=main,
        leads={n: by_lead(ev, leads=TABLE_LEADS, ci=ci, **kw) for n, ev in evs.items()},
        overall={n: ev.summary(ci=ci, **kw) for n, ev in evs.items()},
        history=history, breakdowns=breakdowns, reliability=evs[main],
        sharpness=sharpness_curves(evs, cfg),
        coverage=dict(report=report, gate=conditional_gate(report, cfg)),
        calibrated=None, hourly=None, config=cfg)
    if shift is not None:
        ev_cal = evs[main].with_conformal(shift, meta["history_valid"])
        rep = coverage_report(ev_cal, meta, cfg, external=external,
                              history=history_evaluations(bench, shift))
        hourly = collect_predictions({main: named[main]},
                                     base.hourly(cfg.aci_stations, cfg.aci_hours), device=device)
        res["hourly"] = hourly
        res["calibrated"] = dict(effect=calibration_effect(evs[main], ev_cal), report=rep,
                                 gate=conditional_gate(rep, cfg),
                                 aci=aci_hourly_replay(hourly[0][main], hourly[1], cfg.aci(),
                                                       shift))
    return res


def evaluate_leads(named, base, r_damped=None, ci=True, bootstrap=BOOTSTRAP, device="cpu",
                   main=MAIN_MODEL):
    """Метрики по лидам всех моделей и эталонов на одном наборе окон при полной истории.

    Args:
        named: словарь из имени модели в модель; основная модель обязательна.
        base: набор окон.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        ci: считать интервалы бутстрапа по станциям.
        bootstrap: параметры бутстрапа.
        device: устройство.
        main: имя основной модели.

    Returns:
        Словарь: имя основной модели, сводки по лидам и по всему горизонту для всех
        моделей, оценка основной модели и метаданные окон.
    """
    preds, aux = collect_predictions(named, base.with_history(NOMINAL_HISTORY), device=device)
    preds = add_statistical_baselines(preds, aux, r_damped=r_damped)
    evs = evaluations(preds, aux)
    kw = bootstrap if ci else {}
    return dict(main=main,
                leads={n: by_lead(ev, leads=TABLE_LEADS, ci=ci, **kw) for n, ev in evs.items()},
                overall={n: ev.summary(ci=ci, **kw) for n, ev in evs.items()},
                main_eval=evs[main], meta=aux["meta"])


def print_history_table(history, lead, metric="Skill"):
    """Печатает метрику всех моделей по длинам истории на одном лиде.

    Args:
        history: словарь из имени модели в словарь из лида в разрез по длине истории.
        lead: лид, ч.
        metric: имя метрики.
    """
    labels = list(next(iter(history.values()))[lead])
    print(f"{'модель':>24} {'агрег.':>7} " + " ".join(f"{k:>22}" for k in labels))
    for name, by_lead_rows in history.items():
        rows = by_lead_rows[lead]
        for agg, title in (("pooled", "пул"), ("macro", "макро")):
            cells = [_cell(rows[k][agg], metric, rows[k].get("ci") and rows[k]["ci"][agg])
                     for k in labels]
            print(f"{name:>24} {title:>7} " + " ".join(f"{c:>22}" for c in cells))


def print_leads(res, tag=""):
    """Печатает метрики всех моделей по лидам и по всему горизонту.

    Args:
        res: результат оценки набора со сводками по лидам и по всему горизонту.
        tag: метка набора в заголовках.
    """
    ci = any("ci" in s for s in res["overall"].values())
    for name, rows in res["leads"].items():
        print(f"\n=== {tag}{name} (полная история) ===")
        print_rows({str(h): s for h, s in rows.items()}, label="лид, ч", ci=ci)
    print(f"\n=== {tag}Общие метрики по всему горизонту (полная история) ===")
    print_rows(res["overall"], label="модель", ci=ci)


def print_evaluation(res, bootstrap=BOOTSTRAP):
    """Печатает все таблицы стенда на одном наборе окон.

    Args:
        res: результат оценки набора.
        bootstrap: параметры бутстрапа, с которыми он посчитан.
    """
    from mayak.calibration import (print_aci_replay, print_calibration_effect,
                                   print_coverage_report, print_sharpness)
    tag = "[внешний] " if res["external"] else ""
    main = res["main"]
    print(BENCHMARK_NOTE)
    if TUNED_MODEL in res["leads"]:
        print(TUNED_NOTE)
    print_leads(res, tag)
    for h in HISTORY_LEADS:
        print(f"\n=== {tag}Длина истории: скилл всех моделей, лид {h} ч ===")
        print_history_table(res["history"], h)
    print(f"\n=== {tag}Разрезы ({main}, лид {BREAKDOWN_LEAD} ч) ===")
    print_breakdowns(res["breakdowns"])
    print(f"\n=== {tag}Надёжность ({main}, сырые выходы) ===")
    print_reliability(res["reliability"])
    print_sharpness(res["sharpness"], res["config"].nominal)
    cov = res["coverage"]
    print_coverage_report(cov["report"], cov["gate"],
                          title=f"\n=== {tag}Покрытие по разрезам ({main}, сырые выходы, "
                                f"весь горизонт) ===")
    cal = res["calibrated"]
    if cal is None:
        return
    print_calibration_effect(cal["effect"],
                             title=f"\n########## {tag}{main} после калибровки ##########")
    print_coverage_report(cal["report"], cal["gate"],
                          title=f"\n=== {tag}Покрытие по разрезам ({main} после калибровки) ===")
    print_aci_replay(cal["aci"], res["config"].tolerance)


def plot_history_curves(history, out_dir="runs/plots", set_name="internal",
                        leads=HISTORY_LEADS):
    """Скилл всех моделей по длине истории на нескольких лидах.

    Длины истории стоят на оси через равные промежутки. Полоса вокруг линии - интервал
    бутстрапа по станциям для пуловой метрики, если он посчитан.

    Args:
        history: словарь из имени модели в словарь из лида в разрез по длине истории.
        out_dir: каталог графика.
        set_name: имя набора в имени файла.
        leads: лиды, по панели на каждый.

    Returns:
        Путь к файлу.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(leads), figsize=(5.2 * len(leads), 4.4), squeeze=False)
    for ax, h in zip(axes[0], leads):
        labels = None
        for name, rows_by_lead in history.items():
            rows = rows_by_lead[h]
            labels = list(rows)
            x = np.arange(len(labels))
            y = [rows[k]["pooled"]["Skill"] for k in labels]
            line, = ax.plot(x, y, marker="o", ms=3, label=name)
            if all("ci" in rows[k] for k in labels):
                lo = [rows[k]["ci"]["pooled"]["Skill"][0] for k in labels]
                hi = [rows[k]["ci"]["pooled"]["Skill"][1] for k in labels]
                ax.fill_between(x, lo, hi, alpha=0.12, color=line.get_color())
        ax.axhline(0.0, ls="--", lw=1, color="gray")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("длина истории")
        ax.set_title(f"лид {h} ч")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Skill")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"[{set_name}] скилл по длине истории, сырые выходы", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, f"skill_by_history_{set_name}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def coldstart_L0_check(model, ds):
    """Проверка холодного старта: при нулевой истории медиана равна полю модели.

    Args:
        model: модель МАЯК.
        ds: окна с нулевой историей.

    Returns:
        Словарь чисел проверки.
    """
    D = _gather_full(model, ds)
    o_abs = float(np.abs(D["o"]).mean() + np.abs(D["o_p"]).mean())
    e_mean = float(np.abs(D["e"]).mean())
    dev = np.abs(D["mu"] - D["mu_c"])
    p_before = coverage(D["y"], D["q"], D["y_mask"])
    diff_clim = float(np.abs(D["mu"] - D["mu_clim"]).mean())

    print(f"  средний модуль вклада мод = {o_abs:.3f} (ожидается около 0)")
    print(f"  средняя масса свидетельств = {e_mean:.3f} (ожидается около 0)")
    print(f"  |медиана − поле модели| = {dev.mean():.3f} °C  (макс {dev.max():.2f})")
    print(f"  |медиана − эмпирич. климатология| = {diff_clim:.3f} °C  "
          f"(ожидается мало: поле близко к климатологии)")
    print(f"  PICP-90 при L=0, сырые выходы = {p_before:.1%}  (цель 86–94%)")
    return dict(o_abs=o_abs, e_mean=e_mean, dev_deg=float(dev.mean()),
                picp90=p_before, diff_clim=diff_clim)


@torch.no_grad()
def pure_field_check(model, ds):
    """Чистое поле МАЯК при нулевой истории: без паспорта и без поправки голов.

    Args:
        model: модель МАЯК.
        ds: окна с нулевой историей.

    Returns:
        Словарь: отношение MSE чистого поля к MSE климатологии станции и средний модуль
        разности поля и климатологии, °C.
    """
    from mayak.astro import astro_features
    model.eval()
    e2 = ec = bias = 0.0
    n = 0
    for b in DataLoader(ds, batch_size=128):
        lat, lon, elev = b["lat"], b["lon"], b["elev"]
        loc = model.loc(lat, lon, elev)
        astro_f = astro_features(b["doy_fut"], b["hour_fut"], lat[:, None], lon[:, None])
        coefs = model.field.coefficients(loc)
        mu_c, _, _ = model.field.evaluate(coefs, astro_f)
        y, muc, w = b["y"], b["mu_clim_fut"], b["y_mask"]
        e2 += float((((mu_c - y) ** 2) * w).sum())
        ec += float((((muc - y) ** 2) * w).sum())
        bias += float((mu_c - muc).abs().sum())
        n += y.numel()
    ratio, bias = e2 / max(ec, 1e-9), bias / max(n, 1)
    print(f"ЧИСТОЕ поле на {'/'.join(ds.station_splits)}: ratio={ratio:.3f}, "
          f"|поле−клим|={bias:.3f}°C  (окон: {len(ds)})")
    return dict(ratio=ratio, bias=bias)


@torch.no_grad()
def l0_decompose(model, ds):
    """Разложение выхода МАЯК при нулевой истории.

    Args:
        model: модель МАЯК.
        ds: окна с нулевой историей.

    Returns:
        Словарь: разброс паспорта по окнам, средние модули поправки голов, вклада мод
        погоды и группы P и средняя масса свидетельств.
    """
    model.eval()
    Z = []
    sr = oo = op = ee = 0.0
    n = 0
    for b in DataLoader(ds, batch_size=128):
        out = model(b)
        Z.append(out["z"])
        sr += float((out["sigma_0"] * out["r"]).abs().sum())
        oo += float(out["o"].abs().sum())
        op += float(out["o_p"].abs().sum())
        ee += float(out["e"].abs().sum())
        n += out["mu"].numel()
    Z = torch.cat(Z, 0)
    res = dict(z_std=float(Z.std(0).mean()), correction=sr / n, o=oo / n, o_p=op / n,
               e=ee / Z.numel() * Z.shape[1])
    print(f"[{'/'.join(ds.station_splits)}] std(z) по станциям = {res['z_std']:.3f}  "
          f"(≈0 → прайор глобальный; >0 → прайор зависит от loc = меморизатор)")
    print(f"        |σ₀·r| = {res['correction']:.3f}°C  "
          f"(≈0 → r заглушён; >0 → r ещё активен и фитит)")
    print(f"        |o| = {res['o']:.3f}   |o_P| = {res['o_p']:.3f}   "
          f"e = {res['e']:.3f}  (ждём ≈0 при L=0)")
    return res


def evaluate_external(named, external_manifest, store, grid=HISTORY_GRID, r_damped=None,
                      shift=None, ci=True, bootstrap=BOOTSTRAP, checkpoints=(), conformal=None,
                      internal=None, transfer_level="group"):
    """Внешний тест: станции реальной сети, только их тестовое окно.

    Эталон скилла для всех моделей один: климатология каждой внешней станции по её
    собственному обучающему окну, построенная при сборке кэша внешнего набора.

    Args:
        named: словарь из имени модели в модель; те же модели, что во внутренней оценке.
        external_manifest: манифест внешнего набора.
        store: основной набор, нужен чек-листу изоляции внешнего теста.
        grid: длины истории, ч.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        shift: конформная таблица основной модели; влияет только на раздел после
            калибровки.
        ci: считать интервалы бутстрапа по станциям.
        bootstrap: параметры бутстрапа.
        checkpoints: чекпойнты всех моделей, для чек-листа.
        conformal: путь к конформной таблице, для чек-листа.
        internal: предсказания и данные окон основного внутреннего набора (станции
            unseen_test) при полной истории, для сопоставления внутреннего и внешнего
            теста; None значит без сопоставления.
        transfer_level: уровень зон в сопоставлении.

    Returns:
        Тройка: результат оценки внешнего набора, набор окон и таблицы сопоставления.
    """
    from mayak.data.store import get_store
    from mayak.external import nearest_train_km, print_transfer, transfer_table
    from mayak.leakage import check_external, run_checklist
    ext_store = get_store(external_manifest)
    ds = EvalSet(ext_store.clims(), station_splits=(ROLE_EXTERNAL,), manifest=external_manifest,
                 time_key="test", train_km=nearest_train_km(store, ext_store))
    run_checklist(ext_store, datasets=[ds])
    summary = check_external(store, ext_store, checkpoints=checkpoints, conformal=conformal)
    kw = bootstrap if ci else {}

    print(f"\n########## ВНЕШНИЙ ТЕСТ: {len(ext_store.stations)} станций, "
          f"окон {len(ds)} ##########")
    print_external_isolation(summary)
    res = evaluate_set(named, ds, grid=grid, r_damped=r_damped, shift=shift, ci=ci,
                       bootstrap=bootstrap, external=True)
    print_evaluation(res, bootstrap)

    transfer = {}
    if internal is not None:
        p_int, a_int = internal
        preds, aux = res["bench"].preds, res["bench"].aux
        tag = "/".join(sorted(set(np.asarray(a_int["meta"]["role"]).astype(str).tolist())))
        for name in (MAIN_MODEL, CLIMATOLOGY):
            if name not in p_int or name not in preds:
                continue
            ev_i = evaluation_for(p_int[name], a_int)
            ev_e = evaluation_for(preds[name], aux)
            tbl = transfer_table(ev_i, a_int["meta"]["zone"], ev_e, aux["meta"]["zone"],
                                 level=transfer_level, n_boot=kw.get("n_boot", 0),
                                 seed=kw.get("seed", 0), ci_level=kw.get("level", 0.90))
            transfer[(name, tag)] = tbl
            print_transfer(tbl, title=f"\n=== Перенос: {name}, внешний против внутреннего "
                                      f"({tag}); Δ = внешний − внутренний ===")
    return res, ds, transfer


def print_external_isolation(summary):
    """Печатает календарный запас внешнего теста и число его станций по расстоянию.

    Число станций печатается по всем бинам, даже если в таблице разреза бин скрыт
    порогом по числу станций.

    Args:
        summary: сводка проверки изоляции внешнего теста.
    """
    from mayak.external import NO_DATA, TRAIN_DISTANCE_ORDER
    cal = summary.get("calendar")
    if cal:
        print(f"  обучение до {cal['train_last']}, внешний тест с {cal['external_first']}, "
              f"запас {cal['margin_hours']} ч")
    counts = summary.get("train_distance") or {}
    names = TRAIN_DISTANCE_ORDER + ((NO_DATA,) if counts.get(NO_DATA) else ())
    cells = [f"{name}: {counts.get(name, 0)}" for name in names]
    print("  внешних станций по расстоянию до обучающей точки: " + ", ".join(cells))


def save_bench(res, out_dir, set_name, shift=None, info=None):
    """Сохраняет сырые предсказания набора для анализа калибровки без повторного прогона.

    Пишутся два файла: предсказания всех моделей при полной истории и предсказания
    основной модели на всей сетке длин истории. Если в результате есть ежечасные
    выпуски основной модели, они ложатся третьим файлом.

    Args:
        res: результат оценки набора.
        out_dir: каталог.
        set_name: имя набора в именах файлов.
        shift: конформная таблица; сохраняется рядом, сами предсказания сырые.
        info: сведения о прогоне для заголовка файлов.

    Returns:
        Пути к записанным файлам.
    """
    from mayak.calibration import save_predictions
    bench = res["bench"]
    info = dict(info or {}, history_grid=list(bench.grid))
    nominal = save_predictions(os.path.join(out_dir, f"{set_name}.npz"), bench.preds,
                               bench.aux, shift=shift, info=dict(info, history=NOMINAL_HISTORY))
    h_preds, h_aux = history_predictions(bench)
    grid_path = save_predictions(os.path.join(out_dir, f"{set_name}_history.npz"), h_preds,
                                 h_aux, shift=shift, info=info)
    if res.get("hourly") is None:
        return nominal, grid_path
    r_preds, r_aux = res["hourly"]
    hourly = save_predictions(os.path.join(out_dir, f"{set_name}_hourly.npz"), r_preds, r_aux,
                              shift=shift, info=dict(info, rhythm="hourly"))
    return nominal, grid_path, hourly


def save_history_table(res, out_dir, set_name):
    """Сводки всех моделей по длине истории в JSON рядом с графиком.

    Args:
        res: результат оценки набора.
        out_dir: каталог.
        set_name: имя набора в имени файла.

    Returns:
        Путь к файлу.
    """
    from mayak.calibration import save_json
    blob = dict(grid=list(res["bench"].grid), leads=list(HISTORY_LEADS),
                models={n: {str(h): rows for h, rows in by_lead_rows.items()}
                        for n, by_lead_rows in res["history"].items()})
    return save_json(blob, os.path.join(out_dir, f"skill_by_history_{set_name}.json"))


def parse_grid(text):
    """Сетка длин истории из строки вида «0,6,24,672».

    Args:
        text: длины через запятую.

    Returns:
        Кортеж длин по возрастанию.

    Raises:
        ValueError: в строке не числа или сетка негодна.
    """
    return check_history_grid(int(v) for v in str(text).split(",") if v.strip())


def print_parameter_counts(named, reference="МАЯК"):
    """Печатает таблицу числа параметров нейросетевых моделей и их долю от эталона.

    Args:
        named: словарь из имени строки таблицы в модель.
        reference: имя строки, относительно которой считается доля. Если такой строки
            нет, доля не печатается.

    Returns:
        Словарь из имени строки в полное число параметров модели.
    """
    from mayak.lit import parameter_counts
    counts = {name: parameter_counts(m)["total"] for name, m in named.items()}
    ref = counts.get(reference)
    print("\n=== Число параметров ===")
    for name, n in counts.items():
        share = f"{n / ref:>8.2f}" if ref else ""
        print(f"  {name:<40}{n:>12,}{share}".replace(",", " "))
    return counts


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse
    from mayak.config import run_label
    from mayak.lit import check_comparable, check_extra_tuning, load_model, load_run_record
    from mayak.protocol import SEED_FIELDS, ProtocolError
    ap = argparse.ArgumentParser(description="единый стенд оценки МАЯК")
    ap.add_argument("--ckpt", required=True, nargs="+",
                    help="чекпойнты МАЯК; несколько = прогоны с разными сидами (повторы "
                         "берут скорость обучения первого: scripts/run.py run.lr_from=...), "
                         "разброс по сидам - таблица seeds внутреннего и внешнего набора")
    ap.add_argument("--manifest", default="data/manifest.csv")
    for arch, name in NEURAL_BASELINES.items():
        ap.add_argument(f"--{arch}-ckpt", default=None,
                        help=f"чекпойнт бейзлайна «{name}» "
                             f"(scripts/run.py model={arch})")
    ap.add_argument("--allow-protocol-mismatch", action="store_true",
                    help="не падать, если модели обучены в разных условиях: по разным "
                         "протоколам или без одинакового подбора скорости обучения (только "
                         "для диагностики: такие таблицы несопоставимы)")
    ap.add_argument("--ablation-ckpt", nargs="*", default=[],
                    help="чекпойнты переобученных абляций МАЯК (тот же сид и протокол, "
                         "скорость обучения основного МАЯК: run.lr_from); имя строки таблицы "
                         "берётся из конфига в чекпойнте")
    ap.add_argument("--tuned-ckpt", default=None,
                    help=f"чекпойнт дополнительной настройки МАЯК (этап 2 сравнения, "
                         f"scripts/run.py run.extra_tuning=true): отдельная строка "
                         f"«{TUNED_MODEL}», в сравнение на равных не входит")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="сид оценки (бутстрап, примеры); по умолчанию - seeds.eval "
                         "из первого чекпойнта, иначе 0")
    ap.add_argument("--conformal", default=None,
                    help="конформная таблица МАЯК (runs/conformal.npy); влияет только на "
                         "раздел «МАЯК после калибровки», сравнительные таблицы от неё не "
                         "зависят")
    ap.add_argument("--history-grid", default=",".join(str(v) for v in HISTORY_GRID),
                    help="длины истории, ч, на которых оценивается каждое окно; полная "
                         "история обязательна, при ней считаются основные таблицы")
    ap.add_argument("--out-dir", default="runs/plots")
    ap.add_argument("--n-examples", type=int, default=10, help="число примеров прогноз vs факт")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP["n_boot"],
                    help="итераций блочного бутстрапа по станциям; 0 — без интервалов")
    ap.add_argument("--ci-level", type=float, default=BOOTSTRAP["level"])
    ap.add_argument("--external-manifest", default=None,
                    help="манифест внешнего теста (реальная сеть, роль external_test), "
                         "например data/ghcnh/manifest.csv; собирается scripts/make_ghcnh.py")
    ap.add_argument("--transfer-zones", choices=("group", "full"), default="group",
                    help="уровень зон для сопоставления внутреннего и внешнего теста")
    ap.add_argument("--save-preds", default=None, metavar="DIR",
                    help="сохранить сырые предсказания основного внутреннего набора "
                         "(станции unseen_test): DIR/internal.npz (все модели при полной "
                         "истории) и DIR/internal_history.npz (МАЯК на всей сетке длин "
                         "истории), с --conformal ещё DIR/internal_hourly.npz (МАЯК с "
                         "ежечасным выпуском), для внешнего теста - DIR/external*.npz; их "
                         "читает python -m mayak.calibration")
    ap.add_argument("--results-dir", default=None, metavar="DIR",
                    help="записать каждую таблицу в свой JSON с записью о прогоне: "
                         "DIR/internal/*.json (станции unseen_test), "
                         f"DIR/{TRAIN_STATIONS_DIR}/metrics.json (обучающие станции в "
                         "тестовом окне), DIR/external/*.json, DIR/params.json")
    args = ap.parse_args()
    grid = parse_grid(args.history_grid)

    from mayak.data.store import get_store
    from mayak.leakage import run_checklist
    baseline_ckpts = {a: getattr(args, f"{a}_ckpt") for a in NEURAL_BASELINES
                      if getattr(args, f"{a}_ckpt")}
    all_ckpts = [*args.ckpt, *baseline_ckpts.values(), *args.ablation_ckpt]
    if args.tuned_ckpt:
        check_extra_tuning(args.tuned_ckpt)
        all_ckpts.append(args.tuned_ckpt)
    try:
        check_comparable(args.ckpt[0], [*baseline_ckpts.values(), *args.ablation_ckpt])
        check_comparable(args.ckpt[0], args.ckpt[1:], ignore=SEED_FIELDS)
    except ProtocolError as e:
        if not args.allow_protocol_mismatch:
            raise
        print(f"ВНИМАНИЕ: {e}\nТаблицы ниже несопоставимы (--allow-protocol-mismatch).")

    store = get_store(args.manifest)
    clims = store.clims()
    base = EvalSet(clims, station_splits=(ROLE_TEST,), manifest=args.manifest, time_key="test")
    train_set = EvalSet(clims, station_splits=(ROLE_TRAIN,), manifest=args.manifest,
                        time_key="test")
    run_checklist(store, datasets=[base, train_set], conformal=args.conformal,
                  checkpoints=all_ckpts)
    rec = load_run_record(args.ckpt[0])
    eval_seed = args.eval_seed
    if eval_seed is None:
        eval_seed = int(rec["seeds"]["eval"])
    print(f"Сид оценки: {eval_seed}")

    r = BL.fit_damped_persistence({k: s for k, s in clims.items() if s["role"] == ROLE_TRAIN},
                                  n_windows=20000)

    seeds = [load_model(c) for c in args.ckpt]
    mayak = seeds[0]
    named_extra = {}
    for c in args.ablation_ckpt:
        m = load_model(c)
        named_extra[f"МАЯК [{run_label(m.cfg)}]"] = m
    for arch, c in baseline_ckpts.items():
        named_extra[NEURAL_BASELINES[arch]] = load_model(c)

    tuned = {TUNED_MODEL: load_model(args.tuned_ckpt)} if args.tuned_ckpt else {}
    named_all = {MAIN_MODEL: mayak, **tuned, **named_extra}
    n_params = print_parameter_counts(named_all)
    shift = None
    if args.conformal:
        from mayak.leakage import load_conformal
        shift, _rec = load_conformal(args.conformal)
    boot = dict(n_boot=args.bootstrap, seed=eval_seed, level=args.ci_level)
    ci = args.bootstrap > 0
    set_roles = {"internal": list(base.station_splits),
                 TRAIN_STATIONS_DIR: list(train_set.station_splits)}
    if args.external_manifest:
        set_roles["external"] = [ROLE_EXTERNAL]
    record = run_record(ckpt=args.ckpt, baselines=baseline_ckpts, ablations=args.ablation_ckpt,
                        tuned=args.tuned_ckpt,
                        unequal={TUNED_MODEL: TUNED_NOTE} if args.tuned_ckpt else {},
                        conformal=args.conformal, manifest=args.manifest,
                        external_manifest=args.external_manifest, eval_seed=eval_seed,
                        bootstrap=boot, history_grid=list(grid), sets=set_roles)
    written = []

    print(f"\n=== Таблицы метрик: станции {'/'.join(base.station_splits)}, окон {len(base)}, "
          f"сетка длин истории {list(grid)} ===")
    res = evaluate_set(named_all, base, grid=grid, r_damped=r, shift=shift, ci=ci,
                       bootstrap=boot)
    res_train = evaluate_leads(named_all, train_set, r_damped=r, ci=ci, bootstrap=boot)
    roles = role_breakdown([(res_train["main_eval"], res_train["meta"]),
                            (res["reliability"], res["bench"].aux["meta"])])
    res["breakdowns"] = {ROLE_DIM: roles, **res["breakdowns"]}
    print_evaluation(res, boot)
    print(f"\n########## Обучающие станции в тестовом окне: станции "
          f"{'/'.join(train_set.station_splits)}, окон {len(train_set)}, только метрики по "
          f"лидам ##########")
    print_leads(res_train, "[обучающие] ")
    preds, aux = res["bench"].preds, res["bench"].aux
    info = dict(ckpt=args.ckpt, conformal=args.conformal, manifest=args.manifest,
                eval_seed=eval_seed, n_params=n_params)
    if args.save_preds:
        for p in save_bench(res, args.save_preds, "internal", shift=shift, info=info):
            print("Предсказания:", p)

    tables = evaluation_tables(res)
    if len(seeds) > 1:
        tables["seeds"] = seed_spread_table(seeds, res, base)

    print("\n=== Графики (сырые выходы) ===")
    for p in plot_metric_curves(build_tables(preds, aux), args.out_dir):
        print("  ", p)
    ev_mayak = evaluation_for(preds[MAIN_MODEL], aux)
    print("  ", plot_reliability(ev_mayak, args.out_dir))
    print("  ", plot_pit(ev_mayak, args.out_dir))
    print("  ", plot_history_curves(res["history"], args.out_dir, "internal"))
    print("  ", save_history_table(res, args.out_dir, "internal"))

    print("\n=== Разрез по зонам Кёппена (МАЯК) ===")
    tables["zones"] = zone_breakdown(preds, aux)
    print_zone_breakdown(tables["zones"])

    print("\n=== Холодный старт L=0 ===")
    tables["coldstart"] = coldstart_L0_check(mayak, base.with_history(0))
    if args.results_dir:
        written += write_tables(tables, os.path.join(args.results_dir, "internal"),
                                set_record(record, base))
        written += write_tables(lead_tables(res_train),
                                os.path.join(args.results_dir, TRAIN_STATIONS_DIR),
                                set_record(record, train_set))
        written += write_tables(dict(params=n_params), args.results_dir, record)

    print("\n=== Графики прогноз vs факт (примеры МАЯК, сырые выходы) ===")
    plot_forecast_examples(mayak, clims, manifest=args.manifest,
                           n=args.n_examples, out_dir=args.out_dir, seed=eval_seed)
    plot_forecast_examples(mayak, clims, manifest=args.manifest, n=args.n_examples, L=0,
                           out_dir=args.out_dir, seed=eval_seed)
    print("\n=== Суточные амплитуды ===")
    plot_amplitude_scatter(mayak, clims, manifest=args.manifest, out_dir=args.out_dir)

    if args.external_manifest:
        res_e, ds_e, tr_e = evaluate_external(
            named_all, args.external_manifest, store, grid=grid, r_damped=r, shift=shift,
            ci=ci, bootstrap=boot, checkpoints=all_ckpts, conformal=args.conformal,
            internal=(preds, aux), transfer_level=args.transfer_zones)
        print("  ", plot_history_curves(res_e["history"], args.out_dir, "external"))
        print("  ", save_history_table(res_e, args.out_dir, "external"))
        if args.save_preds:
            for p in save_bench(res_e, args.save_preds, "external", shift=shift,
                                info=dict(info, manifest=args.external_manifest)):
                print("Предсказания:", p)
        ext = dict(evaluation_tables(res_e), transfer=transfer_tables(tr_e))
        if len(seeds) > 1:
            print("\n=== Разброс по сидам, внешний тест ===")
            ext["seeds"] = seed_spread_table(seeds, res_e, ds_e)
        if args.results_dir:
            written += write_tables(ext, os.path.join(args.results_dir, "external"),
                                    set_record(record, ds_e))

    if written:
        print("\n=== Таблицы результатов (JSON) ===")
        for p in written:
            print("  ", p)


def _mu_q(model, ds):
    D = gather(model, ds)
    return D["mu"], D["q"]


if __name__ == "__main__":
    main()
