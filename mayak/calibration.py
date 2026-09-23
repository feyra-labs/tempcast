"""Калибровка интервалов обученной модели.

Всё считается по уже собранным предсказаниям - модель здесь не запускается. Числа
считает ``mayak/metrics.py``; здесь - сбор разрезов, вердикты, печать и графики.

* ``coverage_report`` - покрытие центральных интервалов по разрезам: лиды, бины
  лидов, роли станций, полные зоны Кёппена, длина истории, доля валидных часов истории;
  для внешнего теста - его собственные разрезы. На каждую страту: покрытие 50/80/90 %,
  доли промахов ниже и выше интервала, ширина, интервал блочного бутстрапа по станциям
  и два вердикта:
    - «мимо номинала» - покрытие дальше допуска от номинала и интервал бутстрапа
      номинал не содержит;
    - «отличается от набора» - то же относительно покрытия всего набора. Общее для
      всех страт отклонение исправляет маргинальная поправка (сплит-конформная таблица
      или ACI); отличие страты от набора - нет;
  плюс характер промаха: «узкий интервал» (промахи с обеих сторон), «факт ниже» /
  «факт выше» (≥ 2/3 промахов с одной стороны - сдвиг, а не ширина), «широкий интервал».
  Матрица «страта × бин лидов» показывает, есть ли в расхождении система по горизонту.
* ``conditional_gate`` - решение, нужна ли условная конформная поправка: да,
  только если в разрезах из ``CalibrationConfig.conditional_dims`` есть страты
  «отличается от набора». Сама поправка заранее не реализуется.
* ``sharpness_curves`` - кривые «острота против покрытия» для всех моделей:
  90 %-интервал растягивается вокруг медианы в s раз, s по логарифмической сетке;
  модели сравниваются по ширине, нужной для фактического покрытия 90 %.
* ``aci_replay`` - офлайн-прогон адаптивной калибровки устройства по окнам
  оценки в порядке времени на каждой станции: прибор выпускает прогноз в начале окна и
  до следующего выпуска сверяет каждый валидный час со своим последним прогнозом (те же
  ``aci_score`` / ``ACIParams.step``, что в рантайме). Покрытие считается
  проспективно: каждый час - интервалом, выпущенным до того, как факт стал известен.

Предсказания сохраняет ``python -m mayak.evaluate ... --save-preds runs/preds``.

Запуск::

    python -m mayak.calibration --preds runs/preds/internal.npz \\
        --external-preds runs/preds/external.npz --out-dir runs/calibration
"""
from __future__ import annotations

import json
import logging
import math
import os

import numpy as np

from mayak.config import COVERAGE_DIMS_EXTERNAL, COVERAGE_DIMS_INTERNAL, CalibrationConfig
from mayak.metrics import (FINE_LEADS, LEAD_BINS, Evaluation, aci_effective_level, aci_run,
                           aci_score, lead_bin_of, sharpness_scales, width_at_coverage)

log = logging.getLogger(__name__)

MAIN_MODEL = "МАЯК"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "conf", "calibration", "default.yaml")
LEAD_DIM, LEAD_BIN_DIM = "лид", "бин лидов"
OFF_UNDER, OFF_OVER = "занижено", "завышено"
KIND_NARROW, KIND_WIDE = "узкий интервал", "широкий интервал"
KIND_BELOW, KIND_ABOVE = "факт ниже интервала", "факт выше интервала"
ONE_SIDED = 2.0 / 3.0
META_KEYS = ("station", "role", "zone", "season", "history", "hist_valid", "has_pressure",
             "report_class", "elev_gap", "t")
FORMAT_VERSION = 1


def load_config(path=None):
    """YAML → ``CalibrationConfig``; None - ``conf/calibration/default.yaml``, если есть."""
    path = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if path is None:
        return CalibrationConfig()
    import yaml
    with open(path, encoding="utf-8") as f:
        return CalibrationConfig.from_dict(yaml.safe_load(f) or {})


def save_predictions(path, preds, aux, shift=None, info=None):
    """Предсказания всех моделей и всё, что нужно для оценки без модели, → .npz.

    preds - {имя: dict(mu, q)} до калибровки; aux - y, y_mask, mu_clim, meta (выход
    ``mayak.evaluate.collect_predictions``); shift - конформная таблица, с которой
    велась оценка (сохраняется, чтобы анализ применил ту же самую).
    """
    names = list(preds)
    arrays = dict(y=np.asarray(aux["y"], np.float32), y_mask=np.asarray(aux["y_mask"], np.float32),
                  mu_clim=np.asarray(aux["mu_clim"], np.float32))
    for i, n in enumerate(names):
        arrays[f"pred{i}_mu"] = np.asarray(preds[n]["mu"], np.float32)
        arrays[f"pred{i}_q"] = np.asarray(preds[n]["q"], np.float32)
    meta = aux["meta"]
    meta_keys = [k for k in META_KEYS if k in meta]
    for k in meta_keys:
        v = np.asarray(meta[k])
        arrays[f"meta_{k}"] = v.astype(str) if v.dtype == object else v
    if shift is not None:
        arrays["shift"] = np.asarray(shift, np.float32)
    header = dict(format=FORMAT_VERSION, names=names, meta=meta_keys, info=info or {})
    arrays["header"] = np.array(json.dumps(header, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_predictions(path):
    """.npz ``save_predictions`` → (preds, aux, shift, info)."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        if header.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path}: формат {header.get('format')}, читается {FORMAT_VERSION}")
        preds = {n: dict(mu=z[f"pred{i}_mu"], q=z[f"pred{i}_q"])
                 for i, n in enumerate(header["names"])}
        meta = {}
        for k in header["meta"]:
            v = z[f"meta_{k}"]
            meta[k] = v.astype(object) if v.dtype.kind == "U" else v
        aux = dict(y=z["y"], y_mask=z["y_mask"], mu_clim=z["mu_clim"], meta=meta)
        shift = z["shift"] if "shift" in z.files else None
    return preds, aux, shift, header.get("info", {})


def evaluation_of(pred, aux, shift=None, theta=0.0):
    return Evaluation(y=aux["y"], mu=pred["mu"], q=pred["q"], mu_clim=aux["mu_clim"],
                      w=aux["y_mask"],
                      station=aux["meta"]["station"]).with_calibration(shift, theta)


def coverage_strata(meta, external=False):
    """{имя разреза: метки окон} - те же биннинги, что в таблицах ``mayak.evaluate``."""
    from mayak.evaluate import HIST_VALID_BINS, HISTORY_BINS, bin_label
    hist = np.array([bin_label(int(v), HISTORY_BINS) for v in meta["history"]], object)
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    keys = {"роль станции": meta.get("role"), "зона Кёппена": meta.get("zone"),
            "длина истории": hist, "валидность истории": hvalid,
            "частота отчётности": meta.get("report_class"),
            "Δ высоты станция−ЦМР": meta.get("elev_gap"),
            "канал давления": meta.get("has_pressure")}
    dims = COVERAGE_DIMS_EXTERNAL if external else COVERAGE_DIMS_INTERNAL
    return {d: np.asarray(keys[d], object) for d in dims if keys.get(d) is not None}


def _coverage(ev, nominal):
    return next(r["coverage"] for r in ev.sharpness_coverage()
                if abs(r["nominal"] - nominal) < 1e-6)


def coverage_row(ev, nominal=0.9, n_boot=0, seed=0, level=0.90):
    """Покрытие одной страты: все центральные интервалы, хвосты, ширина, интервал."""
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
    """Существенно (дальше допуска) и значимо (интервал не содержит ref) отличается."""
    if not abs(value - ref) > tol:
        return False
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return True
    return not lo <= ref <= hi


def verdict(row, nominal, overall, tol):
    """Вердикты страты: мимо номинала / отличается от набора / характер промаха."""
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


def _strata_rows(ev, keys, cfg, overall, leads=None, boot=None):
    """Строки по меткам keys (страты меньше порога отброшены), с вердиктами."""
    keys = np.asarray(keys).astype(str)
    rows = {}
    for k in sorted(set(keys.tolist())):
        sub = ev.restrict(windows=keys == k, leads=leads)
        c = sub.counts()
        if c["n_windows"] < cfg.min_windows or c["n_stations"] < cfg.min_stations:
            continue
        r = coverage_row(sub, cfg.nominal, **(boot or {}))
        r.update(verdict(r, cfg.nominal, overall, cfg.tolerance))
        rows[k] = r
    return rows


def coverage_report(ev, meta, cfg=None, external=False, leads=FINE_LEADS,
                    lead_bins=LEAD_BINS):
    """Покрытие по всем разрезам для одной модели (оценка уже откалибрована)."""
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
    for name, keys in coverage_strata(meta, external=external).items():
        dims[name] = _strata_rows(ev, keys, cfg, overall, boot=boot)
        keys_s = np.asarray(keys).astype(str)
        matrix[name] = {}
        for k in dims[name]:
            sel = keys_s == k
            matrix[name][k] = {f"{a}-{b}": _coverage(ev.restrict(windows=sel,
                                                                  leads=np.arange(a, b + 1)),
                                                      cfg.nominal)
                               for a, b in lead_bins}
    return dict(nominal=cfg.nominal, tolerance=cfg.tolerance, external=bool(external),
                overall=total, dims=dims, matrix=matrix)


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
    """{модель: {панель: кривая + ширина при фактическом покрытии номинала}}."""
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


def plot_sharpness(curves, out_dir, set_name="internal", nominal=0.9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    panels = list(next(iter(curves.values())))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for name, c in curves.items():
            p = c[panel]
            line, = ax.plot(p["coverage"], p["width"], lw=1.4, label=name)
            ax.plot([p["model_coverage"]], [p["model_width"]], "o", color=line.get_color(), ms=5)
        ax.axvline(nominal, ls="--", lw=1, color="gray")
        ax.set_title(f"лиды: {panel}", fontsize=10)
        ax.set_xlabel(f"фактическое покрытие {nominal:.0%}-интервала")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("средняя ширина интервала, °C")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"[{set_name}] острота против покрытия (точка - выход модели как есть)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, f"sharpness_coverage_{set_name}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_coverage_strata(report, out_dir, set_name="internal", model=MAIN_MODEL):
    """Точечный график PICP страт с интервалами бутстрапа по всем разрезам."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    labels, vals, los, his, colors = [], [], [], [], []
    for name, rows in report["dims"].items():
        if name == LEAD_DIM:
            continue
        for k, r in rows.items():
            labels.append(f"{name}: {k}")
            vals.append(r["coverage"])
            lo, hi = r["ci"]
            los.append(r["coverage"] - lo if np.isfinite(lo) else 0.0)
            his.append(hi - r["coverage"] if np.isfinite(hi) else 0.0)
            colors.append("tab:red" if r["heterogeneous"] else
                          ("tab:orange" if r["off_nominal"] else "tab:blue"))
    nom, tol = report["nominal"], report["tolerance"]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * max(len(labels), 4) + 1.5))
    y = np.arange(len(labels))[::-1]
    ax.axvspan(nom - tol, nom + tol, color="green", alpha=0.12, label="номинал ± допуск")
    ax.axvline(nom, color="gray", lw=1, ls="--")
    ax.axvline(report["overall"]["coverage"], color="black", lw=1, ls=":", label="весь набор")
    ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor="gray", lw=1)
    ax.scatter(vals, y, c=colors, s=18, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"фактическое покрытие {nom:.0%}-интервала")
    ax.set_title(f"[{set_name}] {model}: покрытие по разрезам (красный - отличается от набора)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p = os.path.join(out_dir, f"coverage_strata_{set_name}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def served_pairs(station, t, horizon):
    """Пары (окно, индекс лида), которые прибор проверил бы по своим прогнозам.

    Окна одной станции упорядочиваются по времени; прогноз окна i действует до выпуска
    следующего: лиды 1 … min(H, t_{i+1} − t_i); у последнего окна - весь горизонт.
    Возвращает индексы окон и лидов, упорядоченные по (станция, время).
    """
    station = np.asarray(station).astype(str)
    t = np.asarray(t, np.int64)
    order = np.lexsort((t, station))
    st, ts = station[order], t[order]
    nxt = np.empty(len(order), np.int64)
    nxt[:-1] = ts[1:] - ts[:-1]
    nxt[-1:] = horizon
    last = np.ones(len(order), bool)
    last[:-1] = st[1:] != st[:-1]
    k = np.where(last, horizon, np.clip(nxt, 0, horizon))
    win = np.repeat(order, k)
    starts = np.repeat(np.cumsum(k) - k, k)
    lead = np.arange(int(k.sum())) - starts
    return win, lead


def aci_replay(ev, meta, params, lead_bins=LEAD_BINS):
    """Офлайн-прогон ACI по окнам оценки (оценка уже со сплит-конформной поправкой).

    На каждой станции θ стартует с нуля. Возвращает сводки «без ACI» (θ = 0 - то, что
    прибор выдал бы без подстройки) и «с ACI» по всему потоку, бинам лидов и ролям,
    распределение покрытия по станциям и траекторию θ.
    """
    if "t" not in meta:
        raise KeyError("в метаданных окон нет времени начала 't' - пересохраните "
                       "предсказания текущей версией mayak.evaluate --save-preds")
    win, lead = served_pairs(meta["station"], meta["t"], ev.horizon)
    ok = ev.w[win, lead] > 0
    win, lead = win[ok], lead[ok]
    if not len(win):
        raise ValueError("ACI: нет ни одной валидной пары «окно × лид» для прогона")
    i, j = params.interval
    q = np.asarray(ev.q, np.float64)[win, lead]
    score = aci_score(np.asarray(ev.y, np.float64)[win, lead], q, (i, j))
    width0 = q[:, j] - q[:, i]
    st = np.asarray(meta["station"]).astype(str)[win]
    theta = np.zeros(len(win))
    miss_aci = np.zeros(len(win))
    theta_end, clipped = {}, 0
    bounds = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        r = aci_run(score[a:b], params)
        theta[a:b], miss_aci[a:b] = r["theta"], r["miss"]
        theta_end[st[a]] = r["theta_end"]
        clipped += r["clipped"]
    miss0 = (score > 1.0).astype(np.float64)
    width1 = np.exp(theta) * width0
    nominal = 1.0 - params.target

    def summ(sel):
        n = int(sel.sum())
        if not n:
            return None
        return dict(n=n, base=float(1 - miss0[sel].mean()), aci=float(1 - miss_aci[sel].mean()),
                    width_base=float(width0[sel].mean()), width_aci=float(width1[sel].mean()))

    lb = lead_bin_of(ev.horizon, lead_bins)[lead]
    by_bin = {f"{a}-{b}": summ(lb == k) for k, (a, b) in enumerate(lead_bins)}
    role = np.asarray(meta.get("role", np.full(len(ev.y), "—"))).astype(str)[win]
    by_role = {r: summ(role == r) for r in sorted(set(role.tolist()))}
    per_st = {}
    for s in sorted(set(st.tolist())):
        sel = st == s
        per_st[s] = (1 - miss0[sel].mean(), 1 - miss_aci[sel].mean())
    dev0 = np.array([abs(v[0] - nominal) for v in per_st.values()])
    dev1 = np.array([abs(v[1] - nominal) for v in per_st.values()])
    te = np.array(list(theta_end.values()))
    return dict(
        nominal=nominal, gamma=params.gamma, max_factor=params.max_factor,
        overall=summ(np.ones(len(win), bool)), by_lead_bin=by_bin, by_role=by_role,
        stations=dict(n=len(per_st), mad_base=float(dev0.mean()) if len(dev0) else float("nan"),
                      mad_aci=float(dev1.mean()) if len(dev1) else float("nan"),
                      per_station={k: dict(base=float(a), aci=float(b))
                                   for k, (a, b) in per_st.items()}),
        theta_end=dict(median=float(np.median(te)) if te.size else float("nan"),
                       min=float(te.min()) if te.size else float("nan"),
                       max=float(te.max()) if te.size else float("nan")),
        clipped=int(clipped))


def print_aci_replay(r, tol=0.04):
    print(f"\n--- адаптивная калибровка на устройстве (офлайн-прогон), γ = {r['gamma']}, "
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
    for k, s in r["by_role"].items():
        line(str(k), s)
    s = r["stations"]
    within = lambda key: np.mean([abs(v[key] - r["nominal"]) <= tol
                                  for v in s["per_station"].values()]) if s["n"] else float("nan")
    print(f"  станций {s['n']}: среднее |PICP − {r['nominal']:.0%}| без ACI {_pct(s['mad_base'])}, "
          f"с ACI {_pct(s['mad_aci'])}; в допуске ±{tol:.0%}: без {_pct(within('base'))}, "
          f"с {_pct(within('aci'))}")
    te = r["theta_end"]
    print(f"  θ в конце: медиана {te['median']:+.3f} (эквивалент номинала "
          f"{aci_effective_level(te['median'], 1 - r['nominal']):.1%}), "
          f"диапазон [{te['min']:+.3f}; {te['max']:+.3f}]; обновлений на границе: {r['clipped']}")


def analyze(preds, aux, shift=None, cfg=None, external=False, model=MAIN_MODEL,
            out_dir=None, set_name="internal"):
    cfg = cfg or CalibrationConfig()
    if model not in preds:
        raise KeyError(f"модели {model!r} нет в предсказаниях; есть {list(preds)}")
    evs = {n: evaluation_of(p, aux, shift) for n, p in preds.items()}
    report = coverage_report(evs[model], aux["meta"], cfg, external=external)
    gate = conditional_gate(report, cfg)
    curves = sharpness_curves(evs, cfg)
    replay = aci_replay(evs[model], aux["meta"], cfg.aci())
    tag = "ВНЕШНИЙ ТЕСТ" if external else "внутренний тест"
    print_coverage_report(report, gate, title=f"\n########## Калибровка: {model}, {tag} ##########")
    for panel in next(iter(curves.values())):
        print_sharpness(curves, cfg.nominal, panel)
    print_aci_replay(replay, cfg.tolerance)
    result = dict(model=model, set=set_name, config=cfg.to_dict(), report=report, gate=gate,
                  sharpness=curves, aci=replay)
    if out_dir:
        paths = [plot_sharpness(curves, out_dir, set_name, cfg.nominal),
                 plot_coverage_strata(report, out_dir, set_name, model),
                 save_json(result, os.path.join(out_dir, f"calibration_{set_name}.json"))]
        result["paths"] = paths
    return result


def jsonable(o):
    """Результаты анализа → строгий JSON: массивы - списки, NaN/inf - null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    return o


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
    return path


def main(argv=None):
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="калибровка интервалов по сохранённым предсказаниям: разрезы покрытия, "
                    "критерий условной поправки, острота против покрытия, офлайн-прогон "
                    "адаптивной калибровки устройства (блок 13)")
    ap.add_argument("--preds", required=True,
                    help="предсказания внутреннего теста (mayak.evaluate --save-preds)")
    ap.add_argument("--external-preds", default=None, help="предсказания внешнего теста")
    ap.add_argument("--config", default=None,
                    help="YAML (по умолчанию conf/calibration/default.yaml)")
    ap.add_argument("--model", default=MAIN_MODEL, help="модель для разрезов и ACI")
    ap.add_argument("--no-conformal", action="store_true",
                    help="не применять сохранённую конформную таблицу (сырые квантили)")
    ap.add_argument("--bootstrap", type=int, default=None,
                    help="итераций бутстрапа по станциям (по умолчанию из конфига; 0 - без)")
    ap.add_argument("--out-dir", default="runs/calibration")
    args = ap.parse_args(argv)

    from dataclasses import replace
    cfg = load_config(args.config)
    if args.bootstrap is not None:
        cfg = replace(cfg, bootstrap=args.bootstrap)
    for path, external, name in ((args.preds, False, "internal"),
                                 (args.external_preds, True, "external")):
        if not path:
            continue
        preds, aux, shift, info = load_predictions(path)
        if args.no_conformal:
            shift = None
        print(f"\n{path}: моделей {len(preds)}, окон {len(aux['y'])}, конформная таблица "
              f"{'применена' if shift is not None else 'не применяется'}; {info}")
        res = analyze(preds, aux, shift, cfg, external=external, model=args.model,
                      out_dir=args.out_dir, set_name=name)
        for p in res["paths"]:
            print("  ", p)


if __name__ == "__main__":
    main()
