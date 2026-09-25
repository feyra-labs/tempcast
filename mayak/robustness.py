"""Робастность обученной модели без переобучения.

Главное заявление проекта - плавная деградация при отказе входа и отсутствие
провала ниже климатологии. Здесь это измеряется:

* ``RobustnessSet`` - окна ``EvalSet`` (тот же отбор, та же стратифицированная
  подвыборка), к которым на лету применяется сценарий из ``mayak.data.scenarios``;
  все уровни и все модели оцениваются на одном и том же множестве окон и лидов;
* ``robustness_sweep`` - сетка «сценарий × уровень × модель × лид» → плоские строки
  с пуловыми и макро-метриками (MAE, RMSE, CRPS, покрытие, Winkler, скилл) и
  доверительными интервалами блочного бутстрапа по станциям (блок 5). Для сценариев
  «свойство прибора» дополнительно - величина искажения цели и «превышение»:
  на сколько рост MAE больше самого искажения (> 0 - прогноз разрушается сильнее,
  чем сдвинут прибор);
* ``check_skill_guard`` - утверждение «скилл ``guard_models`` не ниже −допуска ни в
  одном сценарии с проверкой, ни на одном уровне и лиде». Нарушение - исключение
  ``RobustnessError`` (и код выхода 1 в командной строке), а не строка в логе;
* графики «параметр деградации → метрика» и сводный график скилла по всем сценариям.

Запуск::

    python -m mayak.robustness --ckpt runs/mayak/stageB/best.ckpt \\
        --conformal runs/conformal.npy --external-manifest data/ghcnh/manifest.csv
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from mayak import baselines as BL
from mayak.config import ConfigError, RobustnessConfig, SCENARIO_INSTRUMENT, ScenarioSpec
from mayak.constants import L_MAX
from mayak.data.augment import AugWindow
from mayak.data.dataset import history_len
from mayak.data.masking import enforce_invariant
from mayak.data.qc import qc_window
from mayak.data.scenarios import (SCENARIOS, apply_scenario, level_label, point_qc_mask,
                                  scenario_rng, variants_of)
from mayak.evaluate import (NEURAL_BASELINES, EvalSet, add_statistical_baselines,
                            collect_predictions, evaluation_for)
from mayak.metrics import METRICS, wmean

log = logging.getLogger(__name__)

MAIN_MODEL = "МАЯК"
CURVE_METRICS = ("MAE", "CRPS", "PICP90", "Skill")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "conf", "robustness", "default.yaml")
KIND_RU = {SCENARIO_INSTRUMENT: "свойство прибора", "input": "отказ входа"}


class RobustnessError(RuntimeError):
    """Скилл опустился ниже −допуска в сценарии с проверкой (регрессия)."""


def _np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def load_config(path=None):
    """YAML → ``RobustnessConfig``; None - ``conf/robustness/default.yaml``, если он есть."""
    path = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if path is None:
        return RobustnessConfig()
    import yaml
    with open(path, encoding="utf-8") as f:
        return RobustnessConfig.from_dict(yaml.safe_load(f) or {})


class RobustnessSet(Dataset):
    """Окна базового ``EvalSet`` с применённым сценарием name уровня level."""

    def __init__(self, base: EvalSet, name, level, params=None, qc="point", seed=0):
        if name not in SCENARIOS:
            raise ConfigError(f"неизвестный сценарий {name!r}; есть {tuple(SCENARIOS)}")
        if qc not in ("none", "point", "window"):
            raise ConfigError(f"qc = {qc!r}; допустимо none | point | window")
        self.base, self.name, self.level = base, name, float(level)
        self.params = dict(params or {})
        self.qc, self.seed = qc, int(seed)
        self.items = base.items

    def __len__(self):
        return len(self.base)

    def footprints(self):
        return self.base.footprints()

    def window_meta(self):
        meta = getattr(self.base, "_robustness_meta", None)
        if meta is None:
            meta = self.base.window_meta()
            self.base._robustness_meta = meta
        return meta

    def window(self, i, item=None):
        """(окно после сценария, исходный батч-элемент, запись станции)."""
        item = self.base[i] if item is None else item
        sid, t = self.base.items[i]
        s = self.base.clims[sid]
        L = history_len(self.base.L, t, self.base.floor[sid])
        dem = s.get("dem_elev")
        w = AugWindow(x=_np(item["x_hist"]).astype(np.float32, copy=True),
                      m=_np(item["mask_hist"]).astype(np.float32, copy=True),
                      y=_np(item["y"]).astype(np.float32, copy=True),
                      y_mask=_np(item["y_mask"]).astype(np.float32, copy=True),
                      L=L, hour=_np(item["hour_hist"]),
                      lat=float(s["lat"]), lon=float(s["lon"]), elev=float(s["elev"]),
                      qc_elev=float(dem) if dem is not None else float(s["elev"]))
        apply_scenario(w, self.name, self.level, scenario_rng(self.seed, self.name, i),
                       self.params)
        return w, item, s

    def _qc(self, x, m, w):
        if self.qc == "point":
            m = point_qc_mask(x, m)
        elif self.qc == "window" and w.L > 0:
            m, _codes = qc_window(x, m, elev=w.qc_elev)
        return enforce_invariant(x, m)

    def __getitem__(self, i):
        w, item, s = self.window(i)
        _sid, t = self.base.items[i]
        x, m = enforce_invariant(w.x, w.m)
        x, m = self._qc(x, m, w)
        y, _ = enforce_invariant(w.y, w.y_mask)
        a_recent, _ok = BL.recent_anomaly(x[:, 0], m[:, 0], s["clim"], L_MAX,
                                          int(s["t0"]) + int(t) - L_MAX)
        out = dict(item)
        out.update(
            lat=torch.tensor(w.lat, dtype=torch.float32),
            lon=torch.tensor(w.lon, dtype=torch.float32),
            elev=torch.tensor(w.elev, dtype=torch.float32),
            x_hist=torch.from_numpy(np.ascontiguousarray(x, np.float32)),
            mask_hist=torch.from_numpy(np.ascontiguousarray(m, np.float32)),
            y=torch.from_numpy(np.ascontiguousarray(y, np.float32)),
            a_recent=torch.tensor(a_recent, dtype=torch.float32),
        )
        return out


def base_eval_set(clims, manifest, cfg: RobustnessConfig, roles=None, time_key=None):
    """Окна, на которых оцениваются все сценарии, уровни и модели."""
    return EvalSet(clims, station_splits=tuple(roles or cfg.roles), manifest=manifest,
                   time_key=time_key or cfg.time_key, every_hours=cfg.every_hours,
                   max_windows=None, windows_per_station=cfg.windows_per_station)


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _row(set_name, spec: ScenarioSpec, level, model, lead, summ):
    rule = spec.rule
    r = dict(set=set_name, scenario=spec.name, kind=rule.kind, target=bool(rule.target),
             guard=bool(spec.guard), variant_of=rule.variant_of or "", level=float(level),
             level_label=level_label(spec.name, level), model=model, lead=int(lead),
             n_windows=int(summ["n_windows"]), n_stations=int(summ["n_stations"]))
    ci = summ.get("ci")
    for m in METRICS:
        r[m] = _finite(summ["pooled"][m])
        r[f"{m}_macro"] = _finite(summ["macro"][m])
        if ci is not None:
            r[f"{m}_lo"], r[f"{m}_hi"] = (_finite(v) for v in ci["pooled"][m])
            r[f"{m}_macro_lo"], r[f"{m}_macro_hi"] = (_finite(v) for v in ci["macro"][m])
    return r


def add_excess(rows):
    """ΔMAE относительно нулевого уровня и превышение ΔMAE над искажением цели."""
    ref = {(r["set"], r["scenario"], r["model"], r["lead"]): r["MAE"]
           for r in rows if r["level"] == 0.0}
    for r in rows:
        mae0 = ref.get((r["set"], r["scenario"], r["model"], r["lead"]), float("nan"))
        r["dMAE"] = r["MAE"] - mae0
        r["excess"] = r["dMAE"] - r["distortion"]
    return rows


def robustness_sweep(named, base, cfg: RobustnessConfig, shift=None, r_damped=None,
                     statistical=True, device="cpu", n_boot=None, boot_seed=0,
                     set_name="internal"):
    """Сетка «сценарий × уровень × модель × лид» → плоский список строк метрик.

    named       - {имя: модель} (как в ``mayak.evaluate``);
    base        - ``EvalSet``: одно множество окон для всех сценариев, уровней и моделей;
    shift       - конформная таблица (применяется ко всем моделям, как в оценке);
    r_damped    - коэффициенты затухающей персистентности; None - без неё;
    statistical - добавить статистические эталоны (климатология, сезонно-наивный, ...);
    n_boot      - итераций бутстрапа по станциям; None - ``cfg.bootstrap``.
    """
    n_boot = cfg.bootstrap if n_boot is None else int(n_boot)
    rows = []
    for spec in cfg.scenarios:
        y_ref = None
        for level in spec.levels:
            ds = RobustnessSet(base, spec.name, level, spec.params, qc=cfg.qc, seed=cfg.seed)
            preds, aux = collect_predictions(named, ds, device=device)
            if statistical:
                preds = add_statistical_baselines(preds, aux, r_damped=r_damped)
            if y_ref is None:
                y_ref = aux["y"]                       # первый уровень - 0, данные чистые
            dist = np.abs(aux["y"].astype(np.float64) - y_ref)
            for model, p in preds.items():
                ev = evaluation_for(p, aux, shift)
                for lead in cfg.leads:
                    summ = ev.restrict(leads=[lead]).summary(
                        ci=n_boot > 0, n_boot=n_boot, seed=boot_seed, level=cfg.ci_level)
                    row = _row(set_name, spec, level, model, lead, summ)
                    row["distortion"] = float(wmean(dist[:, lead - 1],
                                                    aux["y_mask"][:, lead - 1]))
                    rows.append(row)
            log.info("[%s] %s = %s: %d окон", set_name, spec.name,
                     level_label(spec.name, level), len(ds))
    return add_excess(rows)


def skill_violations(rows, tolerance, models=(MAIN_MODEL,)):
    """Строки, где скилл модели из ``models`` в сценарии с проверкой ниже −tolerance."""
    return [r for r in rows if r["guard"] and r["model"] in models
            and np.isfinite(r["Skill"]) and r["Skill"] < -tolerance]


def check_skill_guard(rows, tolerance, models=(MAIN_MODEL,)):
    """Утверждение блока 12.3: скилл ни в одном сценарии не ниже −tolerance."""
    checked = [r for r in rows if r["guard"] and r["model"] in models]
    if not checked:
        raise RobustnessError(f"проверка скилла пуста: нет строк моделей {tuple(models)} "
                              f"в сценариях с проверкой")
    bad = skill_violations(rows, tolerance, models)
    if bad:
        lines = [f"  [{r['set']}] {r['model']}: {r['scenario']} = {r['level_label']}, "
                 f"лид {r['lead']} ч: Skill {r['Skill']:+.1%}" for r in bad[:20]]
        more = f"\n  ... и ещё {len(bad) - 20}" if len(bad) > 20 else ""
        raise RobustnessError(f"скилл ниже −{tolerance:.0%} в {len(bad)} случаях "
                              f"(регрессия блока 12):\n" + "\n".join(lines) + more)
    return len(checked)


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def save_results(rows, out_dir, cfg: RobustnessConfig, meta=None, violations=()):
    """robustness.json (конфиг, строки, нарушения, метаданные) и robustness.csv."""
    os.makedirs(out_dir, exist_ok=True)
    rows_c = [{k: _clean(v) for k, v in r.items()} for r in rows]
    blob = dict(config=cfg.to_dict(), meta=meta or {}, rows=rows_c,
                violations=[{k: _clean(v) for k, v in r.items()} for r in violations])
    pj = os.path.join(out_dir, "robustness.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    pc = os.path.join(out_dir, "robustness.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(pc, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows_c)
    return pj, pc


def _select(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def print_report(rows, cfg: RobustnessConfig, set_name="internal", model=MAIN_MODEL, lead=24):
    """Таблица по каждому сценарию: скилл по лидам и метрики на лиде ``lead``."""
    lead = lead if lead in cfg.leads else cfg.leads[len(cfg.leads) // 2]
    for spec in cfg.scenarios:
        rule = spec.rule
        tgt = "история и цель" if rule.target else "только история"
        print(f"\n=== [{set_name}] {spec.name}: {rule.title} ({KIND_RU[rule.kind]}; "
              f"искажается {tgt}; проверка скилла: {'да' if spec.guard else 'нет'}) ===")
        print(f"    уровень: {rule.unit}")
        head = f"{'уровень':>14} " + "".join(f"{'Sk@' + str(h):>8}" for h in cfg.leads)
        head += (f" {'MAE@' + str(lead):>8} {'CRPS':>7} {'PICP90':>7} {'Sk макро':>9}"
                 f" {'ΔMAE':>7} {'искаж.':>7} {'превыш.':>8}")
        print(head)
        for level in spec.levels:
            sel = _select(rows, set=set_name, scenario=spec.name, model=model, level=level)
            if not sel:
                continue
            by = {r["lead"]: r for r in sel}
            r = by[lead]
            line = f"{level_label(spec.name, level)[:14]:>14} "
            for h in cfg.leads:
                sk = by[h]["Skill"]
                flag = "!" if spec.guard and np.isfinite(sk) and sk < -cfg.skill_tolerance else " "
                line += f"{sk:>+7.1%}{flag}" if np.isfinite(sk) else f"{'—':>8}"
            line += (f" {r['MAE']:>8.2f} {r['CRPS']:>7.2f} {r['PICP90']:>7.1%}"
                     f" {r['Skill_macro']:>+9.1%} {r['dMAE']:>+7.2f} {r['distortion']:>7.2f}"
                     f" {r['excess']:>+8.2f}")
            print(line)


def _band(ax, xs, sel, metric, **kw):
    ys = [r[metric] for r in sel]
    line, = ax.plot(xs, ys, marker="o", ms=3, **kw)
    if f"{metric}_lo" in sel[0]:
        lo = [r[f"{metric}_lo"] for r in sel]
        hi = [r[f"{metric}_hi"] for r in sel]
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    return line


def plot_scenario(rows, cfg, spec, out_dir, set_name="internal", model=MAIN_MODEL):
    """Кривые «параметр → MAE, CRPS, покрытие, скилл» по лидам с интервалами по станциям.

    Для смещения и дрейфа поверх - вариант «незамеченное смещение прибора» пунктиром.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rule = spec.rule
    variants = [cfg.scenario(v) for v in variants_of(spec.name)
                if any(sc.name == v for sc in cfg.scenarios)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric in zip(axes.ravel(), CURVE_METRICS):
        for lead in cfg.leads:
            sel = sorted(_select(rows, set=set_name, scenario=spec.name, model=model,
                                 lead=lead), key=lambda r: r["level"])
            if not sel:
                continue
            xs = np.arange(len(sel)) if spec.name == "drop_channel" else [r["level"] for r in sel]
            line = _band(ax, xs, sel, metric, label=f"лид {lead} ч")
            for vs in variants:
                vsel = sorted(_select(rows, set=set_name, scenario=vs.name, model=model,
                                      lead=lead), key=lambda r: r["level"])
                if vsel:
                    ax.plot([r["level"] for r in vsel], [r[metric] for r in vsel], ls="--",
                            color=line.get_color(), lw=1,
                            label="незамеченное смещение прибора" if lead == cfg.leads[0]
                            else None)
        if metric == "Skill":
            ax.axhline(0.0, color="gray", lw=1)
            if spec.guard:
                ax.axhline(-cfg.skill_tolerance, color="red", lw=1, ls=":",
                           label=f"−допуск {cfg.skill_tolerance:.0%}")
        if metric == "PICP90":
            ax.axhline(0.90, color="gray", lw=1, ls="--")
        if spec.name == "drop_channel":
            ax.set_xticks(np.arange(len(spec.levels)))
            ax.set_xticklabels([level_label(spec.name, v) for v in spec.levels], fontsize=7)
        ax.set_xlabel(rule.unit)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    fig.suptitle(f"[{set_name}] {model}: {rule.title} ({KIND_RU[rule.kind]}, искажается "
                 f"{'история и цель' if rule.target else 'только история'})", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, f"robustness_{set_name}_{spec.name}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_summary(rows, cfg, out_dir, set_name="internal", model=MAIN_MODEL, lead=24):
    """Скилл на лиде ``lead`` против нормированного уровня деградации - все сценарии."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lead = lead if lead in cfg.leads else cfg.leads[len(cfg.leads) // 2]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for spec in cfg.scenarios:
        sel = sorted(_select(rows, set=set_name, scenario=spec.name, model=model, lead=lead),
                     key=lambda r: r["level"])
        if not sel:
            continue
        top = max(spec.levels) or 1.0
        ax.plot([r["level"] / top for r in sel], [r["Skill"] for r in sel], marker="o", ms=3,
                ls="-" if spec.guard else "--", label=spec.name)
    ax.axhline(0.0, color="gray", lw=1)
    ax.axhline(-cfg.skill_tolerance, color="red", lw=1, ls=":",
               label=f"−допуск {cfg.skill_tolerance:.0%}")
    ax.set_xlabel("уровень деградации / максимальный уровень сценария")
    ax.set_ylabel(f"Skill @ {lead} ч")
    ax.set_title(f"[{set_name}] {model}: скилл при деградации входа (пунктир - без проверки)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(out_dir, f"robustness_{set_name}_summary_skill{lead}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_all(rows, cfg, out_dir, set_name="internal", model=MAIN_MODEL):
    os.makedirs(out_dir, exist_ok=True)
    paths = [plot_scenario(rows, cfg, spec, out_dir, set_name, model)
             for spec in cfg.scenarios if spec.rule.variant_of is None]
    paths.append(plot_summary(rows, cfg, out_dir, set_name, model))
    return paths


def main(argv=None):
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="робастность обученной модели: сценарии деградации входа без "
                    "переобучения, кривые метрик и проверка «скилл не ниже −допуска» (блок 12)")
    ap.add_argument("--ckpt", required=True, help="чекпойнт МАЯК")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--external-manifest", default=None,
                    help="манифест внешнего теста (роль external_test) - те же сценарии")
    for arch, name in NEURAL_BASELINES.items():
        ap.add_argument(f"--{arch}-ckpt", default=None, help=f"чекпойнт бейзлайна «{name}»")
    ap.add_argument("--allow-protocol-mismatch", action="store_true")
    ap.add_argument("--config", default=None,
                    help="YAML сценариев (по умолчанию conf/robustness/default.yaml)")
    ap.add_argument("--scenarios", default=None,
                    help="подмножество сценариев через запятую, например offset,dropout")
    ap.add_argument("--conformal", default=None, help="runs/conformal.npy (если есть)")
    ap.add_argument("--bootstrap", type=int, default=None,
                    help="итераций бутстрапа по станциям (по умолчанию из конфига; 0 - без)")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="сид бутстрапа; по умолчанию seeds.eval из чекпойнта, иначе 0")
    ap.add_argument("--no-statistical", action="store_true",
                    help="без статистических эталонов в таблицах")
    ap.add_argument("--report-only", action="store_true",
                    help="не завершаться с кодом 1 при нарушении проверки скилла")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="runs/robustness")
    args = ap.parse_args(argv)

    from mayak.data.splits import ROLE_EXTERNAL, ROLE_TRAIN
    from mayak.data.store import get_store
    from mayak.leakage import check_external, run_checklist
    from mayak.lit import check_comparable, load_model, load_run_record
    from mayak.protocol import ProtocolError

    cfg = load_config(args.config)
    if args.scenarios:
        cfg = cfg.select([s.strip() for s in args.scenarios.split(",") if s.strip()])
    baseline_ckpts = {a: getattr(args, f"{a}_ckpt") for a in NEURAL_BASELINES
                      if getattr(args, f"{a}_ckpt")}
    all_ckpts = [args.ckpt, *baseline_ckpts.values()]
    try:
        check_comparable(args.ckpt, list(baseline_ckpts.values()))
    except ProtocolError as e:
        if not args.allow_protocol_mismatch:
            raise
        print(f"ВНИМАНИЕ: {e}")

    store = get_store(args.manifest)
    clims = store.clims()
    base = base_eval_set(clims, args.manifest, cfg)
    run_checklist(store, datasets=[base], conformal=args.conformal, checkpoints=all_ckpts)
    rec = load_run_record(args.ckpt)
    boot_seed = args.eval_seed if args.eval_seed is not None else (
        int(rec["seeds"]["eval"]) if rec else 0)

    named = {MAIN_MODEL: load_model(args.ckpt)}
    for arch, c in baseline_ckpts.items():
        named[NEURAL_BASELINES[arch]] = load_model(c)
    r_damped = None
    if not args.no_statistical:
        r_damped = BL.fit_damped_persistence(
            {k: s for k, s in clims.items() if s["role"] == ROLE_TRAIN}, n_windows=20000)
    shift = np.load(args.conformal) if args.conformal else None
    kw = dict(shift=shift, r_damped=r_damped, statistical=not args.no_statistical,
              device=args.device, n_boot=args.bootstrap, boot_seed=boot_seed)

    sets = {"internal": base}
    rows = robustness_sweep(named, base, cfg, set_name="internal", **kw)
    if args.external_manifest:
        ext_store = get_store(args.external_manifest)
        ext = base_eval_set(ext_store.clims(), args.external_manifest, cfg,
                            roles=(ROLE_EXTERNAL,), time_key="test")
        run_checklist(ext_store, datasets=[ext])
        check_external(store, ext_store, checkpoints=all_ckpts, conformal=args.conformal)
        sets["external"] = ext
        rows += robustness_sweep(named, ext, cfg, set_name="external", **kw)

    for name in sets:
        print_report(rows, cfg, set_name=name)
        for p in plot_all(rows, cfg, args.out_dir, set_name=name):
            print("  ", p)
    bad = skill_violations(rows, cfg.skill_tolerance, cfg.guard_models)
    meta = dict(ckpt=args.ckpt, baselines=baseline_ckpts, conformal=args.conformal,
                qc=cfg.qc, boot_seed=boot_seed,
                sets={n: dict(n_windows=len(d), n_stations=len({s for s, _t in d.items}))
                      for n, d in sets.items()})
    for p in save_results(rows, args.out_dir, cfg, meta=meta, violations=bad):
        print("  ", p)
    try:
        n = check_skill_guard(rows, cfg.skill_tolerance, cfg.guard_models)
        print(f"\nПроверка скилла пройдена: {n} строк, допуск −{cfg.skill_tolerance:.0%}.")
    except RobustnessError as e:
        print(f"\nНАРУШЕНИЕ: {e}")
        if not args.report_only:
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
