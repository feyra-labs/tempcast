r"""Анализ калибровки интервалов обученной модели.

Всё считается по готовым предсказаниям и оценкам, модель здесь не запускается. Функции
модуля вызывают стенд оценки (``mayak.evaluate``) и подгонка конформной таблицы
(``scripts/calibrate.py``).

Сравнение моделей идёт только по сырым выходам: кривые «острота против покрытия» всех
моделей и покрытие по разрезам основной модели считаются без калибровки. Конформная
таблица, если она есть, даёт отдельный раздел: та же модель после калибровки рядом с
сырой, покрытие по тем же разрезам и офлайн-прогон адаптивной калибровки устройства.

Покрытие центральных интервалов разбирается по лидам, бинам лидов, полным зонам Кёппена,
длине истории и доле валидных часов истории. Набор окон состоит из станций одной роли,
поэтому разреза по ролям нет. Для внешнего теста разрезы свои. У каждой страты:
покрытие, доли промахов ниже и выше интервала, ширина, интервал блочного бутстрапа по
станциям и два вердикта. «Мимо номинала» значит, что покрытие дальше допуска от номинала
и интервал бутстрапа номинал не содержит.
«Отличается от набора» значит то же самое относительно покрытия всего набора: общее для
всех страт отклонение исправляет маргинальная поправка, отличие страты от набора нет.

Разрез по длине истории строится по сетке длин, если переданы оценки модели на каждой
длине сетки. Тогда каждая длина из сетки становится своей стратой. Строка конформной
таблицы у каждого окна выбирается по числу часов с валидной температурой во входе модели
(history_valid), как на устройстве; разрезы по длине истории остаются по запрошенной
длине.

Офлайн-прогон адаптивной калибровки идёт с ритмом устройства: непрерывный период с
выпуском каждый час на части станций набора. Ежечасные предсказания основной модели для
него собирает стенд оценки.
"""
from __future__ import annotations

import math
import os

import numpy as np

from mayak.config import COVERAGE_DIMS_EXTERNAL, COVERAGE_DIMS_INTERNAL, CalibrationConfig
from mayak.data.holdout import history_strata
from mayak.metrics import (FINE_LEADS, LEAD_BINS, AdaptiveCalibration, Evaluation,
                           aci_effective_level, aci_score, apply_conformal, lead_bin_of,
                           ordered_labels, sharpness_scales, width_at_coverage)

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "conf", "calibration", "default.yaml")
LEAD_DIM, LEAD_BIN_DIM, HISTORY_DIM = "лид", "бин лидов", "длина истории"
OFF_UNDER, OFF_OVER = "занижено", "завышено"
KIND_NARROW, KIND_WIDE = "узкий интервал", "широкий интервал"
KIND_BELOW, KIND_ABOVE = "факт ниже интервала", "факт выше интервала"
ONE_SIDED = 2.0 / 3.0


def load_config(path=None):
    """Конфиг калибровки из YAML.

    Args:
        path: путь к YAML; None - файл калибровки по умолчанию из каталога конфигов,
            если он есть, иначе значения по умолчанию.

    Returns:
        Конфиг калибровки.
    """
    path = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if path is None:
        return CalibrationConfig()
    import yaml
    with open(path, encoding="utf-8") as f:
        return CalibrationConfig.from_dict(yaml.safe_load(f) or {})


def table_hours(meta):
    """Число валидных часов температуры окон, по которому выбирается строка таблицы.

    Args:
        meta: метаданные окон.

    Returns:
        Массив int64 формы (N,).

    Raises:
        KeyError: в метаданных нет числа валидных часов.
    """
    if "history_valid" not in meta:
        raise KeyError("в метаданных окон нет 'history_valid' - числа валидных часов "
                       "температуры во входе модели")
    return np.asarray(meta["history_valid"], np.int64)


def evaluation_of(pred, aux, shift=None, theta=0.0):
    """Оценка модели по готовым предсказаниям, при желании после калибровки.

    Args:
        pred: медиана и квантили модели.
        aux: факт, веса, эталон и метаданные окон; с таблицей в метаданных нужно число
            валидных часов температуры окон.
        shift: конформная таблица; строка выбирается по числу валидных часов
            температуры окна.
        theta: логарифм адаптивного множителя: число или по бинам лидов.

    Returns:
        Оценка.
    """
    hist = table_hours(aux["meta"]) if shift is not None else None
    return Evaluation(y=aux["y"], mu=pred["mu"], q=pred["q"], mu_clim=aux["mu_clim"],
                      w=aux["y_mask"],
                      station=aux["meta"]["station"]).with_calibration(shift, theta, hist)


def coverage_strata(meta, external=False):
    """Метки окон по каждому разрезу покрытия.

    Args:
        meta: метаданные окон.
        external: разрезы внешнего теста вместо внутреннего.

    Returns:
        Словарь из имени разреза в пару: метки окон формы (N,) и порядок меток, в
        котором разрез показывается; None значит алфавитный порядок.
    """
    from mayak.evaluate import HIST_VALID_BINS, TRAIN_DISTANCE_DIM, bin_label
    from mayak.external import TRAIN_DISTANCE_ORDER
    hist, hist_order = history_strata(meta)
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    keys = {"зона Кёппена": meta.get("zone"), HISTORY_DIM: hist, "валидность истории": hvalid,
            TRAIN_DISTANCE_DIM: meta.get("train_distance"),
            "Δ высоты станция−ЦМР": meta.get("elev_gap"),
            "канал давления": meta.get("has_pressure")}
    orders = {HISTORY_DIM: hist_order, TRAIN_DISTANCE_DIM: TRAIN_DISTANCE_ORDER}
    dims = COVERAGE_DIMS_EXTERNAL if external else COVERAGE_DIMS_INTERNAL
    return {d: (np.asarray(keys[d], object), orders.get(d)) for d in dims
            if keys.get(d) is not None}


def _coverage(ev, nominal):
    return next(r["coverage"] for r in ev.sharpness_coverage()
                if abs(r["nominal"] - nominal) < 1e-6)


def coverage_row(ev, nominal=0.9, n_boot=0, seed=0, level=0.90):
    """Покрытие одной страты.

    Args:
        ev: оценка страты.
        nominal: номинал разбираемого интервала.
        n_boot: число повторов бутстрапа; 0 - без интервалов.
        seed: сид бутстрапа.
        level: уровень интервалов бутстрапа.

    Returns:
        Словарь: число окон, станций и пар, покрытие, доли промахов ниже и выше, ширина,
        макро-покрытие, покрытие всех центральных интервалов и интервалы бутстрапа.
    """
    prof = {r["nominal"]: r for r in ev.sharpness_coverage()}
    main = prof[round(nominal, 6)]
    metric = f"PICP{int(round(100 * nominal))}"
    ci = macro = (float("nan"), float("nan"))
    if n_boot:
        b = ev.bootstrap_ci(n_boot=n_boot, seed=seed, level=level)
        ci, macro = b["pooled"][metric], b["macro"][metric]
    return dict(**ev.counts(), coverage=main["coverage"], below=main["below"],
                above=main["above"], width=main["width"],
                macro=float(ev.macro()[metric]),
                cov={f"{int(round(100 * k))}": v["coverage"] for k, v in prof.items()},
                ci=tuple(float(x) for x in ci), ci_macro=tuple(float(x) for x in macro))


def _outside(value, ref, ci, tol):
    """Отличается ли значение от отсчёта и существенно, и значимо.

    Существенно - дальше допуска, значимо - интервал бутстрапа отсчёт не содержит.
    """
    if not abs(value - ref) > tol:
        return False
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return True
    return not lo <= ref <= hi


def verdict(row, nominal, overall, tol):
    """Вердикты страты.

    Args:
        row: строка покрытия страты.
        nominal: номинал интервала.
        overall: покрытие всего набора.
        tol: допуск отклонения покрытия.

    Returns:
        Словарь: мимо номинала и в какую сторону или None; отличается ли от набора;
        характер промаха - широкий, узкий или сдвиг вниз либо вверх.
    """
    cov = row["coverage"]
    off = _outside(cov, nominal, row["ci"], tol)
    het = _outside(cov, overall, row["ci"], tol)
    miss = row["below"] + row["above"]
    if cov > nominal:
        kind = KIND_WIDE
    elif miss > 0 and row["below"] >= ONE_SIDED * miss:
        kind = KIND_BELOW
    elif miss > 0 and row["above"] >= ONE_SIDED * miss:
        kind = KIND_ABOVE
    else:
        kind = KIND_NARROW
    return dict(off_nominal=(OFF_UNDER if cov < nominal else OFF_OVER) if off else None,
                heterogeneous=bool(het), kind=kind)


def _strata_rows(ev, keys, cfg, overall, leads=None, boot=None, order=None):
    """Строки покрытия по меткам окон с вердиктами; страты меньше порога отброшены."""
    keys = np.asarray(keys).astype(str)
    rows = {}
    for k in ordered_labels(keys, order):
        sub = ev.restrict(windows=keys == k, leads=leads)
        c = sub.counts()
        if c["n_windows"] < cfg.min_windows or c["n_stations"] < cfg.min_stations:
            continue
        r = coverage_row(sub, cfg.nominal, **(boot or {}))
        r.update(verdict(r, cfg.nominal, overall, cfg.tolerance))
        rows[k] = r
    return rows


def _lead_bin_coverage(ev, nominal, lead_bins):
    return {f"{a}-{b}": _coverage(ev.restrict(leads=np.arange(a, b + 1)), nominal)
            for a, b in lead_bins}


def history_coverage(history, cfg, boot=None, lead_bins=LEAD_BINS):
    """Покрытие по длине истории, когда одни и те же окна оценены при разных длинах.

    Каждая длина истории - своя страта. Набором для вердикта «отличается от набора»
    служат все длины сетки вместе. Оценки приходят по одной и сразу отпускаются, поэтому
    в памяти одновременно лежит только одна длина.

    Args:
        history: пары из метки длины и оценки модели при этой длине, по возрастанию
            длины.
        cfg: настройки анализа калибровки.
        boot: параметры бутстрапа по станциям.
        lead_bins: бины лидов для матрицы «страта против бина лидов».

    Returns:
        Тройка: строки страт с вердиктами, матрица покрытия по бинам лидов и покрытие
        всей сетки.
    """
    rows, matrix = {}, {}
    hits = pairs = 0.0
    for label, ev in history:
        c = ev.counts()
        r = coverage_row(ev, cfg.nominal, **(boot or {}))
        hits += r["coverage"] * c["n_pairs"] if c["n_pairs"] else 0.0
        pairs += c["n_pairs"]
        if c["n_windows"] < cfg.min_windows or c["n_stations"] < cfg.min_stations:
            continue
        rows[label] = r
        matrix[label] = _lead_bin_coverage(ev, cfg.nominal, lead_bins)
    overall = hits / pairs if pairs else float("nan")
    for r in rows.values():
        r.update(verdict(r, cfg.nominal, overall, cfg.tolerance))
    return rows, matrix, overall


def coverage_report(ev, meta, cfg=None, external=False, leads=FINE_LEADS,
                    lead_bins=LEAD_BINS, history=None):
    """Покрытие одной модели по всем разрезам.

    Args:
        ev: оценка модели на наборе окон, сырая или уже откалиброванная.
        meta: метаданные окон того же набора.
        cfg: настройки анализа калибровки.
        external: разрезы внешнего теста.
        leads: лиды для отдельных строк.
        lead_bins: бины лидов.
        history: пары из метки длины истории и оценки той же модели на тех же окнах при
            этой длине. Если заданы, разрез по длине истории строится по ним, а не по
            метаданным набора.

    Returns:
        Словарь: номинал, допуск, признак внешнего теста, строка всего набора, строки по
        разрезам и матрицы «страта против бина лидов». Для разреза по длине истории,
        построенного по сетке, отдельно записано покрытие всей сетки.
    """
    cfg = cfg or CalibrationConfig()
    boot = dict(n_boot=cfg.bootstrap, seed=cfg.seed, level=cfg.ci_level)
    total = coverage_row(ev, cfg.nominal, **boot)
    overall = total["coverage"]
    total.update(verdict(total, cfg.nominal, overall, cfg.tolerance))
    dims = {}
    lead_rows = {}
    for h in leads:
        sub = ev.restrict(leads=[h])
        r = coverage_row(sub, cfg.nominal, **boot)
        r.update(verdict(r, cfg.nominal, overall, cfg.tolerance))
        lead_rows[str(int(h))] = r
    dims[LEAD_DIM] = lead_rows
    bin_rows = {}
    for a, b in lead_bins:
        r = coverage_row(ev.restrict(leads=np.arange(a, b + 1)), cfg.nominal, **boot)
        r.update(verdict(r, cfg.nominal, overall, cfg.tolerance))
        bin_rows[f"{a}-{b}"] = r
    dims[LEAD_BIN_DIM] = bin_rows
    matrix = {}
    grid_overall = None
    for name, (keys, order) in coverage_strata(meta, external=external).items():
        if name == HISTORY_DIM and history is not None:
            dims[name], matrix[name], grid_overall = history_coverage(history, cfg, boot,
                                                                      lead_bins)
            continue
        dims[name] = _strata_rows(ev, keys, cfg, overall, boot=boot, order=order)
        keys_s = np.asarray(keys).astype(str)
        matrix[name] = {k: _lead_bin_coverage(ev.restrict(windows=keys_s == k), cfg.nominal,
                                              lead_bins)
                        for k in dims[name]}
    return dict(nominal=cfg.nominal, tolerance=cfg.tolerance, external=bool(external),
                overall=total, dims=dims, matrix=matrix, history_overall=grid_overall)


def conditional_gate(report, cfg=None):
    cfg = cfg or CalibrationConfig()
    out = {}
    for name, rows in report["dims"].items():
        if name in (LEAD_DIM, LEAD_BIN_DIM):
            continue
        het = sorted(k for k, r in rows.items() if r["heterogeneous"])
        out[name] = dict(candidate=name in cfg.conditional_dims, strata=het,
                         recommended=bool(het) and name in cfg.conditional_dims,
                         n_strata=len(rows))
    return out


def _pct(v, signed=False):
    if not np.isfinite(v):
        return "—"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _ci(ci):
    lo, hi = ci
    return f"[{_pct(lo)}; {_pct(hi)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def _flag(r):
    parts = []
    if r["off_nominal"]:
        parts.append(f"мимо номинала ({r['off_nominal']})")
    if r["heterogeneous"]:
        parts.append("отличается от набора")
    if r["off_nominal"] or r["heterogeneous"]:
        parts.append(r["kind"])
    return "; ".join(parts)


def print_coverage_report(report, gate=None, title=None):
    nom = report["nominal"]
    key = f"{int(round(100 * nom))}"
    o = report["overall"]
    if title:
        print(title)
    print(f"Номинал {nom:.0%}, допуск ±{report['tolerance']:.0%}. Весь набор: "
          f"покрытие {_pct(o['coverage'])} {_ci(o['ci'])}, макро {_pct(o['macro'])}, "
          f"ниже {_pct(o['below'])} / выше {_pct(o['above'])}, ширина {o['width']:.2f} °C; "
          f"окон {o['n_windows']}, станций {o['n_stations']}")
    head = (f"{'страта':>22} {'окон':>6} {'ст.':>4} {'PICP' + key:>8} {'ДИ':>17} {'макро':>7} "
            f"{'50%':>6} {'80%':>6} {'ниже':>6} {'выше':>6} {'шир.':>6}  вердикт")
    for name, rows in report["dims"].items():
        print(f"\n--- покрытие: {name} ---")
        if not rows:
            print("  (все страты меньше порога по числу окон или станций)")
            continue
        print(head)
        for k, r in rows.items():
            print(f"{k:>22} {r['n_windows']:>6} {r['n_stations']:>4} {_pct(r['coverage']):>8} "
                  f"{_ci(r['ci']):>17} {_pct(r['macro']):>7} {_pct(r['cov']['50']):>6} "
                  f"{_pct(r['cov']['80']):>6} {_pct(r['below']):>6} {_pct(r['above']):>6} "
                  f"{r['width']:>6.2f}  {_flag(r)}")
    for name, m in report["matrix"].items():
        if not m:
            continue
        bins = list(next(iter(m.values())))
        print(f"\n--- PICP{key}: {name} × бин лидов ---")
        print(f"{'страта':>22} " + " ".join(f"{b:>8}" for b in bins))
        for k, row in m.items():
            print(f"{k:>22} " + " ".join(f"{_pct(row[b]):>8}" for b in bins))
    if gate is not None:
        print_gate(gate)


def print_gate(gate):
    print("\n--- условная конформная поправка (13.4) ---")
    rec = [n for n, g in gate.items() if g["recommended"]]
    for n, g in gate.items():
        tag = "кандидат" if g["candidate"] else "для сведения"
        s = ", ".join(g["strata"]) if g["strata"] else "нет"
        print(f"  {n:>22} ({tag}): отличаются от набора - {s}")
    if rec:
        print(f"  ИТОГ: условная поправка оправдана по разрезу(ам): {', '.join(rec)}")
    else:
        print("  ИТОГ: систематического расхождения по разрезам-кандидатам нет - "
              "условная поправка не нужна, достаточно маргинальной")


def sharpness_curves(evs, cfg=None, lead_bins=LEAD_BINS):
    """Кривые «острота против покрытия» всех моделей.

    Args:
        evs: словарь: имя модели и её оценка.
        cfg: конфиг калибровки; None - по умолчанию.
        lead_bins: бины лидов для отдельных панелей.

    Returns:
        Словарь: модель, затем панель - кривая, покрытие и ширина выхода модели как
        есть и ширина при фактическом покрытии, равном номиналу.
    """
    cfg = cfg or CalibrationConfig()
    scales = sharpness_scales(*cfg.sharpness_range, cfg.sharpness_points)
    out = {}
    for name, ev in evs.items():
        curves = ev.sharpness_curve(scales, nominal=cfg.nominal, lead_bins=lead_bins)
        for c in curves.values():
            k = int(np.flatnonzero(np.isclose(c["scale"], 1.0))[0])
            c["model_coverage"] = float(c["coverage"][k])
            c["model_width"] = float(c["width"][k])
            c["width_at_nominal"] = width_at_coverage(c["coverage"], c["width"], cfg.nominal)
            c["scale_at_nominal"] = width_at_coverage(c["coverage"], c["scale"], cfg.nominal)
        out[name] = curves
    return out


def print_sharpness(curves, nominal=0.9, panel="весь горизонт"):
    print(f"\n--- острота против покрытия ({panel}): ширина {nominal:.0%}-интервала ---")
    print(f"{'модель':>24} {'покрытие как есть':>18} {'ширина как есть':>16} "
          f"{'ширина при факт. ' + f'{nominal:.0%}':>22} {'нужный множитель':>17}")
    for name, c in curves.items():
        p = c[panel]
        print(f"{name:>24} {_pct(p['model_coverage']):>18} {p['model_width']:>16.2f} "
              f"{p['width_at_nominal']:>22.2f} {p['scale_at_nominal']:>17.2f}")


def aci_hourly_replay(pred, aux, params, shift=None, lead_bins=LEAD_BINS):
    """Офлайн-прогон адаптивной калибровки с ритмом устройства.

    Окна станции - выпуски каждый час подряд. Прогон повторяет прибор: на каждом часе
    сначала валидный факт часа сверяется с кольцом калибровки, затем выпуск, который
    начинается со следующего часа, записывается в кольцо. Кольцо и подстройка - тот же
    объект, что в рантайме устройства. На каждой станции множители стартуют с нуля.

    Покрытие считается по всем валидным парам выпуска и лида: так пользователь видит
    прогноз прибора на любом лиде. Интервал пары растянут множителем бина её лида,
    действовавшим в момент выпуска.

    Args:
        pred: медиана и квантили модели до калибровки на ежечасных окнах.
        aux: факт, веса и метаданные тех же окон; нужны станция, момент начала горизонта
            и, с таблицей, число валидных часов температуры.
        params: параметры адаптивной калибровки.
        shift: конформная таблица; None - прибор без таблицы.
        lead_bins: бины лидов.

    Returns:
        Словарь: ритм прогона, сводки без подстройки и с подстройкой по всему потоку и
        бинам лидов, покрытие по станциям, число обратных связей и итоговые множители по
        бинам лидов, число упоров в границы.

    Raises:
        KeyError: в метаданных нет момента начала горизонта.
        ValueError: нет ни одной валидной пары выпуска и лида.
    """
    meta = aux["meta"]
    if "t" not in meta:
        raise KeyError("в метаданных окон нет времени начала горизонта 't'")
    q = np.asarray(pred["q"], np.float32)
    if shift is not None:
        q = apply_conformal(q, shift, table_hours(meta), lead_bins)
    y = np.asarray(aux["y"], np.float64)
    w = np.asarray(aux["y_mask"]) > 0
    n, horizon = y.shape
    nb = len(lead_bins)
    st = np.asarray(meta["station"]).astype(str)
    t = np.asarray(meta["t"], np.int64)
    theta_at = np.zeros((n, nb), np.float64)
    updates, clipped, theta_end, hours = np.zeros(nb, np.int64), 0, [], 0
    order = np.lexsort((t, st))
    bounds = np.flatnonzero(np.r_[True, st[order][1:] != st[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        ts = t[idx]
        h0, span = int(ts[0]), int(ts[-1] - ts[0]) + horizon
        yl = np.full(span, np.nan)
        for k in idx:
            off = int(t[k]) - h0
            yl[off:off + horizon] = np.where(w[k], y[k], yl[off:off + horizon])
        start = {int(ts[j]): int(idx[j]) for j in range(len(idx))}
        cal = AdaptiveCalibration(params, horizon, lead_bins)
        for h in range(h0 - 1, h0 + span):
            if h >= h0 and np.isfinite(yl[h - h0]):
                cal.feedback(float(yl[h - h0]), h)
            k = start.get(h + 1)
            if k is not None:
                theta_at[k] = cal.theta
                cal.record(h, q[k])
        updates += np.asarray(cal.updates, np.int64)
        clipped += cal.clipped
        theta_end.append(list(cal.theta))
        hours += len(idx)
    win, lead = np.nonzero(w)
    if not len(win):
        raise ValueError("ACI: нет ни одной валидной пары выпуска и лида для прогона")
    i, j = params.interval
    qp = q[win, lead]
    score = aci_score(y[win, lead], qp, (i, j))
    lb = lead_bin_of(horizon, lead_bins)[lead]
    factor = np.exp(theta_at[win, lb])
    miss0 = (score > 1.0).astype(np.float64)
    miss1 = (score > factor).astype(np.float64)
    width0 = (qp[:, j] - qp[:, i]).astype(np.float64)
    width1 = factor * width0
    nominal = 1.0 - params.target

    def summ(sel):
        k = int(sel.sum())
        if not k:
            return None
        return dict(n=k, base=float(1 - miss0[sel].mean()), aci=float(1 - miss1[sel].mean()),
                    width_base=float(width0[sel].mean()), width_aci=float(width1[sel].mean()))

    by_bin = {f"{a}-{b}": summ(lb == k) for k, (a, b) in enumerate(lead_bins)}
    sw = st[win]
    per_st = {s_: (1 - miss0[sw == s_].mean(), 1 - miss1[sw == s_].mean())
              for s_ in sorted(set(sw.tolist()))}
    dev0 = np.array([abs(v[0] - nominal) for v in per_st.values()])
    dev1 = np.array([abs(v[1] - nominal) for v in per_st.values()])
    te = np.asarray(theta_end, np.float64).reshape(-1, nb)
    return dict(
        rhythm="ежечасный выпуск", nominal=nominal, gamma=params.gamma,
        max_factor=params.max_factor, stations_n=int(len(theta_end)), issues=int(hours),
        overall=summ(np.ones(len(win), bool)), by_lead_bin=by_bin,
        stations=dict(n=len(per_st), mad_base=float(dev0.mean()) if len(dev0) else float("nan"),
                      mad_aci=float(dev1.mean()) if len(dev1) else float("nan"),
                      per_station={k: dict(base=float(a), aci=float(b))
                                   for k, (a, b) in per_st.items()}),
        updates={f"{a}-{b}": int(updates[k]) for k, (a, b) in enumerate(lead_bins)},
        theta_end={f"{a}-{b}": dict(median=float(np.median(te[:, k])), min=float(te[:, k].min()),
                                    max=float(te[:, k].max()))
                   for k, (a, b) in enumerate(lead_bins)},
        clipped=int(clipped))


def print_aci_replay(r, tol=0.04):
    if r is None:
        print("\n--- адаптивная калибровка на устройстве: ежечасного прогона нет ---")
        return
    print(f"\n--- адаптивная калибровка на устройстве (офлайн-прогон, {r['rhythm']}): "
          f"станций {r['stations_n']}, выпусков {r['issues']}, γ = {r['gamma']}, "
          f"множитель ≤ {r['max_factor']:g} ---")
    print(f"{'':>18} {'пар':>8} {'PICP без ACI':>13} {'PICP с ACI':>11} "
          f"{'ширина без':>11} {'ширина с':>9}")

    def line(name, s):
        if s is None:
            return
        print(f"{name:>18} {s['n']:>8} {_pct(s['base']):>13} {_pct(s['aci']):>11} "
              f"{s['width_base']:>11.2f} {s['width_aci']:>9.2f}")

    line("весь поток", r["overall"])
    for k, s in r["by_lead_bin"].items():
        line(f"лиды {k}", s)
    s = r["stations"]
    within = lambda key: np.mean([abs(v[key] - r["nominal"]) <= tol
                                  for v in s["per_station"].values()]) if s["n"] else float("nan")
    print(f"  станций {s['n']}: среднее |PICP − {r['nominal']:.0%}| без ACI {_pct(s['mad_base'])}, "
          f"с ACI {_pct(s['mad_aci'])}; в допуске ±{tol:.0%}: без {_pct(within('base'))}, "
          f"с {_pct(within('aci'))}")
    for k, te in r["theta_end"].items():
        print(f"  лиды {k}: обратных связей {r['updates'][k]}, θ в конце: медиана "
              f"{te['median']:+.3f} (эквивалент номинала "
              f"{aci_effective_level(te['median'], 1 - r['nominal']):.1%}), "
              f"диапазон [{te['min']:+.3f}; {te['max']:+.3f}]")
    print(f"  обновлений на границе: {r['clipped']}")


def calibration_effect(ev_raw, ev_cal, lead_bins=LEAD_BINS):
    """Метрики одной модели до и после калибровки рядом.

    Args:
        ev_raw: оценка по сырым выходам.
        ev_cal: оценка тех же окон после калибровки.
        lead_bins: бины лидов для отдельных строк.

    Returns:
        Словарь из имени строки («весь горизонт» или бин лидов) в словарь с парами
        значений до и после для покрытия 80 и 90 %, ширины 90 %-интервала, CRPS и MAE.
    """
    panels = {"весь горизонт": None}
    panels.update({f"{a}-{b}": np.arange(a, b + 1) for a, b in lead_bins})
    out = {}
    for name, leads in panels.items():
        before = ev_raw.restrict(leads=leads).pooled()
        after = ev_cal.restrict(leads=leads).pooled()
        out[name] = {m: (float(before[m]), float(after[m]))
                     for m in ("PICP80", "PICP90", "Width90", "CRPS", "MAE")}
    return out


def print_calibration_effect(effect, title=None):
    if title:
        print(title)
    print(f"{'лиды':>14} {'PICP90 до':>10} {'после':>7} {'шир.90 до':>10} {'после':>7} "
          f"{'CRPS до':>8} {'после':>7} {'MAE до':>7} {'после':>7}")
    for name, r in effect.items():
        print(f"{name:>14} {_pct(r['PICP90'][0]):>10} {_pct(r['PICP90'][1]):>7} "
              f"{r['Width90'][0]:>10.2f} {r['Width90'][1]:>7.2f} {r['CRPS'][0]:>8.3f} "
              f"{r['CRPS'][1]:>7.3f} {r['MAE'][0]:>7.3f} {r['MAE'][1]:>7.3f}")


FIT_DIMS = ("сезон", HISTORY_DIM)
SEASON_ORDER = ("зима", "весна", "лето", "осень")


def fit_report(ev_raw, ev_cal, meta, cfg=None):
    """Покрытие на калибровочном наборе до и после таблицы по сезонам и длине истории.

    Числа считаются на тех же окнах, по которым подогнана таблица, поэтому покрытие после
    таблицы в целом и в бинах длины истории со своей строкой близко к номиналу по
    построению. Смысл отчёта - в стратах: таблица одна на все сезоны, а бины длины
    истории с маргинальной строкой держат номинал не по построению, и отчёт показывает,
    держит ли таблица номинал в каждой из них.

    Args:
        ev_raw: оценка модели по сырым выходам.
        ev_cal: оценка тех же окон после таблицы.
        meta: метаданные окон калибровочного набора; нужны сезон и длина истории.
        cfg: настройки анализа калибровки.

    Returns:
        Словарь: номинал, допуск, строки всего набора до и после и для каждого разреза
        строки страт до и после с интервалами бутстрапа и вердиктами.
    """
    cfg = cfg or CalibrationConfig()
    boot = dict(n_boot=cfg.bootstrap, seed=cfg.seed, level=cfg.ci_level)
    hist, hist_order = history_strata(meta)
    keys = {"сезон": (np.asarray(meta["season"], object), SEASON_ORDER),
            HISTORY_DIM: (hist, hist_order)}
    out = dict(nominal=cfg.nominal, tolerance=cfg.tolerance, overall={}, dims={})
    for tag, ev in (("raw", ev_raw), ("calibrated", ev_cal)):
        total = coverage_row(ev, cfg.nominal, **boot)
        total.update(verdict(total, cfg.nominal, total["coverage"], cfg.tolerance))
        out["overall"][tag] = total
        for name in FIT_DIMS:
            k, order = keys[name]
            rows = _strata_rows(ev, k, cfg, total["coverage"], boot=boot, order=order)
            out["dims"].setdefault(name, {})[tag] = rows
    return out


def print_fit_report(rep):
    """Печатает отчёт о покрытии калибровочного набора до и после таблицы."""
    key = f"{int(round(100 * rep['nominal']))}"
    o = rep["overall"]
    print(f"\nPICP{key} на калибровочном наборе (в выборке): до {_pct(o['raw']['coverage'])} "
          f"{_ci(o['raw']['ci'])}, после {_pct(o['calibrated']['coverage'])} "
          f"{_ci(o['calibrated']['ci'])}; окон {o['raw']['n_windows']}, станций "
          f"{o['raw']['n_stations']}")
    for name, both in rep["dims"].items():
        print(f"\n--- PICP{key} по разрезу: {name} ---")
        print(f"{'страта':>16} {'окон':>6} {'до':>7} {'ДИ до':>17} {'после':>7} "
              f"{'ДИ после':>17} {'шир. до':>8} {'после':>7}  вердикт после")
        for k, r in both["calibrated"].items():
            b = both["raw"].get(k)
            if b is None:
                continue
            print(f"{k:>16} {r['n_windows']:>6} {_pct(b['coverage']):>7} {_ci(b['ci']):>17} "
                  f"{_pct(r['coverage']):>7} {_ci(r['ci']):>17} {b['width']:>8.2f} "
                  f"{r['width']:>7.2f}  {_flag(r)}")
