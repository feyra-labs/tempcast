r"""Робастность обученной модели без переобучения.

Проверяется главное заявление проекта: при отказе входа прогноз деградирует плавно и не
проваливается ниже того, что модель умеет без истории. Сценарий применяется на лету к
окнам базового набора оценки; все уровни и все модели оцениваются на одних и тех же
окнах и лидах, по сырым выходам, без калибровки.

Сценарии делятся на два класса.

* Отказ входа искажает только историю. Порог проверки на лиде: не хуже собственного
  холодного старта модели на тех же окнах, если холодный старт хуже климатологии
  станции, и не хуже климатологии иначе, в обоих случаях с допуском. Скилл холодного
  старта относительно климатологии печатается отдельной строкой без проверки: это
  измерение качества климат-поля, а не устойчивости.
* Свойство прибора (смещение, масштаб, дрейф) искажает и историю, и цель. Эталон скилла
  здесь - климатология, записанная тем же прибором, то есть искажённая тем же
  преобразованием, что цель. Порог - минус допуск. Варианты, где искажён только вход,
  идут отдельными кривыми без проверки.

Нарушение порога - исключение RobustnessError и код выхода 1 в командной строке, а не
строка в журнале. Для каждой строки считаются пуловые и макро-метрики с интервалами
блочного бутстрапа по станциям, а для свойств прибора ещё величина искажения цели и
превышение роста MAE над ним.

Запуск::

    python -m mayak.robustness --ckpt runs/mayak/stageB/best.ckpt \
        --external-manifest data/ghcnh/manifest.csv
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
from mayak.config import (ROBUSTNESS_QC, SCENARIO_INPUT, SCENARIO_INSTRUMENT, ConfigError,
                          RobustnessConfig, ScenarioSpec)
from mayak.constants import L_MAX
from mayak.data.augment import AugWindow, dither_window, record_window
from mayak.data.masking import enforce_invariant
from mayak.data.qc import qc_window
from mayak.data.scenarios import (DITHER_SCENARIOS, SCENARIOS, apply_scenario, dither_rng,
                                  instrument_reference, level_label, scenario_rng,
                                  variants_of)
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
    """Скилл в сценарии с проверкой опустился ниже порога, или проверять нечего."""


def _np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def load_config(path=None):
    """Конфиг робастности из YAML.

    Args:
        path: путь к YAML; None - файл робастности по умолчанию из каталога конфигов,
            если он есть, иначе значения по умолчанию.

    Returns:
        Конфиг робастности.
    """
    path = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if path is None:
        return RobustnessConfig()
    import yaml
    with open(path, encoding="utf-8") as f:
        return RobustnessConfig.from_dict(yaml.safe_load(f) or {})


class RobustnessSet(Dataset):
    """Окна базового набора оценки с применённым сценарием.

    Args:
        base: базовый набор окон оценки.
        name: имя сценария.
        level: уровень сценария.
        params: параметры сценария.
        qc: режим QC после сценария.
        seed: сид сценариев.
    """

    def __init__(self, base: EvalSet, name, level, params=None, qc="device", seed=0):
        if name not in SCENARIOS:
            raise ConfigError(f"неизвестный сценарий {name!r}; есть {tuple(SCENARIOS)}")
        if qc not in ROBUSTNESS_QC:
            raise ConfigError(f"qc = {qc!r}; допустимо {' | '.join(ROBUSTNESS_QC)}")
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
        """Окно после сценария и записи прибором вместе с эталоном скилла.

        Перед сценарием, искажающим значения, записанным значениям возвращается
        непрерывность; шум один для всех уровней, поэтому кривые по уровням парные.
        Эталон скилла - климатология на горизонте, искажённая тем же преобразованием, что
        цель; в сценариях, не трогающих цель, это сама климатология.

        Args:
            i: номер окна.
            item: готовый элемент базового набора, если он уже посчитан.

        Returns:
            Четвёрка: окно после сценария, элемент базового набора, запись станции и
            эталон скилла формы (H,).
        """
        item = self.base[i] if item is None else item
        sid, t = self.base.items[i]
        s = self.base.clims[sid]
        raw = self.base.raw_window(i)
        L = raw["L"]
        w = AugWindow(x=np.array(raw["x"], np.float32), m=np.array(raw["m"], np.float32),
                      y=_np(item["y"]).astype(np.float32, copy=True),
                      y_mask=_np(item["y_mask"]).astype(np.float32, copy=True),
                      L=L, hour=_np(item["hour_hist"]), hour_fut=_np(item["hour_fut"]),
                      lat=float(s["lat"]), lon=float(s["lon"]), elev=float(s["elev"]),
                      qc_elev=float(raw["qc_elev"]))
        if self.name in DITHER_SCENARIOS:
            dither_window(w, dither_rng(self.seed, i),
                          target=SCENARIOS[self.name].rule.kind == SCENARIO_INSTRUMENT)
        ref = instrument_reference(self.name, self.level, w, _np(item["mu_clim_fut"]),
                                   scenario_rng(self.seed, self.name, i), self.params)
        apply_scenario(w, self.name, self.level, scenario_rng(self.seed, self.name, i),
                       self.params)
        record_window(w)
        return w, item, s, ref

    def _qc(self, i, x, m, w):
        if self.qc == "device" and w.L > 0:
            past = self.base.raw_window(i)["past"]
            m, _codes = qc_window(x, m, elev=w.qc_elev, past=past)
        return enforce_invariant(x, m)

    def __getitem__(self, i):
        w, item, s, ref = self.window(i)
        _sid, t = self.base.items[i]
        x, m = enforce_invariant(w.x, w.m)
        x, m = self._qc(i, x, m, w)
        y, _ = enforce_invariant(w.y, w.y_mask)
        a_recent, a_ok = BL.recent_anomaly(x[:, 0], m[:, 0], s["clim"], L_MAX,
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
            a_recent_ok=torch.tensor(bool(a_ok)),
            hist_len=torch.tensor(w.L, dtype=torch.int64),
            mu_ref_fut=torch.from_numpy(np.ascontiguousarray(ref, np.float32)),
        )
        return out


def base_eval_set(clims, manifest, cfg: RobustnessConfig, roles=None, time_key=None):
    """Окна, на которых оцениваются все сценарии, уровни и модели.

    Args:
        clims: климатологии станций.
        manifest: путь к манифесту.
        cfg: конфиг робастности.
        roles: роли станций; None - из конфига.
        time_key: временное окно; None - из конфига.

    Returns:
        Набор окон оценки.
    """
    return EvalSet(clims, station_splits=tuple(roles or cfg.roles), manifest=manifest,
                   time_key=time_key or cfg.time_key, every_hours=cfg.every_hours,
                   max_windows=None, windows_per_station=cfg.windows_per_station)


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _add_metrics(r, summ):
    """Дописывает в строку пуловые и макро-метрики и их интервалы, если они есть."""
    ci = summ.get("ci")
    for m in METRICS:
        r[m] = _finite(summ["pooled"][m])
        r[f"{m}_macro"] = _finite(summ["macro"][m])
        if ci is not None:
            r[f"{m}_lo"], r[f"{m}_hi"] = (_finite(v) for v in ci["pooled"][m])
            r[f"{m}_macro_lo"], r[f"{m}_macro_hi"] = (_finite(v) for v in ci["macro"][m])
    return r


def _row(set_name, spec: ScenarioSpec, level, model, lead, summ):
    rule = spec.rule
    r = dict(set=set_name, scenario=spec.name, kind=rule.kind, target=bool(rule.target),
             guard=bool(spec.guard), variant_of=rule.variant_of or "", level=float(level),
             level_label=level_label(spec.name, level), model=model, lead=int(lead),
             n_windows=int(summ["n_windows"]), n_stations=int(summ["n_stations"]))
    return _add_metrics(r, summ)


def reference_aux(aux):
    """Данные окон, в которых эталоном скилла служит климатология того же прибора.

    Статистические эталоны строятся до этой подмены, по исходной климатологии: они
    ничего не знают об искажении прибора и прогнозируют так же, как без него.

    Args:
        aux: данные окон; эталон прибора лежит под ключом mu_ref, если набор его даёт.

    Returns:
        Те же данные, где климатология для скилла заменена эталоном прибора. Если
        эталона нет, возвращаются исходные данные.
    """
    if "mu_ref" not in aux:
        return aux
    return dict(aux, mu_clim=aux["mu_ref"])


def skill_floor(row, tolerance):
    """Порог проверки скилла для одной строки.

    В сценарии отказа входа модель не должна опускаться ниже своего холодного старта на
    тех же окнах и лиде, если он хуже климатологии, и ниже климатологии, если холодный
    старт не хуже её. В сценарии свойства прибора порог отсчитывается от эталона того же
    прибора. В обоих случаях из порога вычитается допуск. Если скилл холодного старта
    не посчитан, порог остаётся строгим: отсчёт от климатологии.

    Args:
        row: строка прогона с полями kind и, если есть, Skill_L0.
        tolerance: допуск, доля.

    Returns:
        Порог скилла, доля.
    """
    base = 0.0
    if row["kind"] == SCENARIO_INPUT:
        l0 = row.get("Skill_L0", float("nan"))
        if l0 is not None and np.isfinite(l0):
            base = min(0.0, float(l0))
    return base - float(tolerance)


def add_excess(rows):
    """Добавить рост MAE относительно нулевого уровня и превышение этого роста над искажением цели.

    Args:
        rows: строки сценария по уровням.

    Returns:
        Те же строки с новыми полями.
    """
    ref = {(r["set"], r["scenario"], r["model"], r["lead"]): r["MAE"]
           for r in rows if r["level"] == 0.0}
    for r in rows:
        mae0 = ref.get((r["set"], r["scenario"], r["model"], r["lead"]), float("nan"))
        r["dMAE"] = r["MAE"] - mae0
        r["excess"] = r["dMAE"] - r["distortion"]
    return rows


def _predictions(named, ds, r_damped, statistical, device):
    preds, aux = collect_predictions(named, ds, device=device)
    if statistical:
        preds = add_statistical_baselines(preds, aux, r_damped=r_damped)
    return preds, aux


def cold_start_sweep(named, base, cfg: RobustnessConfig, r_damped=None, statistical=True,
                     device="cpu", n_boot=None, boot_seed=0, set_name="internal"):
    """Скилл каждой модели на тех же окнах без истории.

    Окна, цель и эталон те же, что в сценариях; модели не видят ни одного часа истории.
    Эталон скилла - климатология станции. Это измерение качества климат-поля и
    одновременно пол проверки в сценариях отказа входа.

    Args:
        named: словарь из имени модели в модель.
        base: набор окон, общий для всех сценариев.
        cfg: настройки робастности.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        statistical: добавить статистические эталоны.
        device: устройство.
        n_boot: итераций бутстрапа по станциям; None значит взять из настроек.
        boot_seed: сид бутстрапа.
        set_name: имя набора в строках.

    Returns:
        Плоский список строк: по одной на модель и лид, с метриками и интервалами.
    """
    n_boot = cfg.bootstrap if n_boot is None else int(n_boot)
    ds = base.with_history(0)
    preds, aux = _predictions(named, ds, r_damped, statistical, device)
    rows = []
    for model, p in preds.items():
        ev = evaluation_for(p, aux)
        for lead in cfg.leads:
            summ = ev.restrict(leads=[lead]).summary(
                ci=n_boot > 0, n_boot=n_boot, seed=boot_seed, level=cfg.ci_level)
            r = dict(set=set_name, model=model, lead=int(lead), history=0,
                     n_windows=int(summ["n_windows"]), n_stations=int(summ["n_stations"]))
            rows.append(_add_metrics(r, summ))
    log.info("[%s] холодный старт: %d окон", set_name, len(ds))
    return rows


def robustness_sweep(named, base, cfg: RobustnessConfig, r_damped=None,
                     statistical=True, device="cpu", n_boot=None, boot_seed=0,
                     set_name="internal", cold=None):
    """Метрики всех моделей на каждом сценарии, уровне и лиде.

    Все модели оцениваются по сырым выходам, без калибровки. В сценариях свойства
    прибора эталон скилла - климатология того же прибора. Каждая строка несёт скилл
    холодного старта той же модели на том же лиде и порог проверки при допуске из
    настроек.

    Args:
        named: словарь из имени модели в модель.
        base: набор окон, общий для всех сценариев, уровней и моделей.
        cfg: настройки робастности.
        r_damped: коэффициенты затухающей персистентности; None значит без неё.
        statistical: добавить статистические эталоны.
        device: устройство.
        n_boot: итераций бутстрапа по станциям; None значит взять из настроек.
        boot_seed: сид бутстрапа.
        set_name: имя набора в строках.
        cold: готовые строки холодного старта; None значит посчитать их здесь.

    Returns:
        Плоский список строк: по одной на сценарий, уровень, модель и лид.
    """
    n_boot = cfg.bootstrap if n_boot is None else int(n_boot)
    if cold is None:
        cold = cold_start_sweep(named, base, cfg, r_damped=r_damped, statistical=statistical,
                                device=device, n_boot=0, set_name=set_name)
    l0 = {(r["model"], r["lead"]): r["Skill"] for r in cold if r["set"] == set_name}
    rows = []
    for spec in cfg.scenarios:
        y_ref = None
        for level in spec.levels:
            ds = RobustnessSet(base, spec.name, level, spec.params, qc=cfg.qc, seed=cfg.seed)
            preds, aux = _predictions(named, ds, r_damped, statistical, device)
            if y_ref is None:
                y_ref = aux["y"]
            dist = np.abs(aux["y"].astype(np.float64) - y_ref)
            skill_aux = reference_aux(aux)
            for model, p in preds.items():
                ev = evaluation_for(p, skill_aux)
                for lead in cfg.leads:
                    summ = ev.restrict(leads=[lead]).summary(
                        ci=n_boot > 0, n_boot=n_boot, seed=boot_seed, level=cfg.ci_level)
                    row = _row(set_name, spec, level, model, lead, summ)
                    row["distortion"] = float(wmean(dist[:, lead - 1],
                                                    aux["y_mask"][:, lead - 1]))
                    row["Skill_L0"] = float(l0.get((model, int(lead)), float("nan")))
                    row["floor"] = skill_floor(row, cfg.skill_tolerance)
                    rows.append(row)
            log.info("[%s] %s = %s: %d окон", set_name, spec.name,
                     level_label(spec.name, level), len(ds))
    return add_excess(rows)


def skill_violations(rows, tolerance, models=(MAIN_MODEL,)):
    """Строки сценариев с проверкой, где скилл модели из models ниже своего порога.

    Args:
        rows: строки прогона.
        tolerance: допуск, доля.
        models: имена проверяемых моделей.

    Returns:
        Список нарушивших строк.
    """
    return [r for r in rows if r["guard"] and r["model"] in models
            and np.isfinite(r["Skill"]) and r["Skill"] < skill_floor(r, tolerance)]


def check_skill_guard(rows, tolerance, models=(MAIN_MODEL,)):
    """Проверка скилла по всем сценариям с проверкой, уровням и лидам.

    Args:
        rows: строки прогона.
        tolerance: допуск, доля.
        models: имена проверяемых моделей.

    Returns:
        Число проверенных строк.

    Raises:
        RobustnessError: проверять нечего, или хотя бы в одной строке скилл ниже порога.
    """
    checked = [r for r in rows if r["guard"] and r["model"] in models]
    if not checked:
        raise RobustnessError(f"проверка скилла пуста: нет строк моделей {tuple(models)} "
                              f"в сценариях с проверкой")
    bad = skill_violations(rows, tolerance, models)
    if bad:
        lines = [f"  [{r['set']}] {r['model']}: {r['scenario']} = {r['level_label']}, "
                 f"лид {r['lead']} ч: Skill {r['Skill']:+.1%}, "
                 f"порог {skill_floor(r, tolerance):+.1%}" for r in bad[:20]]
        more = f"\n  ... и ещё {len(bad) - 20}" if len(bad) > 20 else ""
        raise RobustnessError(f"скилл ниже порога проверки в {len(bad)} случаях "
                              f"(допуск {tolerance:.0%}):\n" + "\n".join(lines) + more)
    return len(checked)


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def save_results(rows, out_dir, cfg: RobustnessConfig, meta=None, violations=(),
                 cold_start=()):
    """Результаты прогона в JSON и CSV.

    Args:
        rows: строки сценариев.
        out_dir: каталог результатов.
        cfg: настройки робастности.
        meta: сведения о прогоне: чекпойнты, режим QC, сид бутстрапа, наборы окон с ролями
            станций.
        violations: строки, нарушившие проверку.
        cold_start: строки холодного старта.

    Returns:
        Пара путей: robustness.json и robustness.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    clean = lambda rs: [{k: _clean(v) for k, v in r.items()} for r in rs]
    rows_c = clean(rows)
    blob = dict(config=cfg.to_dict(), meta=meta or {}, cold_start=clean(cold_start),
                rows=rows_c, violations=clean(violations))
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


def _guard_rule(spec, tolerance):
    """Словесное описание порога проверки для заголовка таблицы сценария."""
    if not spec.guard:
        return "проверки нет"
    if spec.rule.kind == SCENARIO_INPUT:
        return (f"порог: не хуже холодного старта, если он хуже климатологии, иначе не хуже "
                f"климатологии; допуск {tolerance:.0%}")
    return f"эталон - климатология того же прибора; допуск {tolerance:.0%}"


def print_cold_start(cold, cfg: RobustnessConfig, set_name="internal", model=MAIN_MODEL):
    """Строка скилла холодного старта модели по лидам, без проверки.

    Args:
        cold: строки холодного старта.
        cfg: настройки робастности.
        set_name: имя набора.
        model: имя модели.
    """
    by = {r["lead"]: r for r in cold if r["set"] == set_name and r["model"] == model}
    if not by:
        return
    print(f"\n=== [{set_name}] {model}: холодный старт, история 0 ч; скилл относительно "
          f"климатологии станции, без проверки ===")
    leads = [h for h in cfg.leads if h in by]
    print("".join(f"{'Sk@' + str(h):>9}" for h in leads))
    print("".join(f"{by[h]['Skill']:>+9.1%}" if np.isfinite(by[h]["Skill"]) else f"{'—':>9}"
                  for h in leads))


def print_report(rows, cfg: RobustnessConfig, set_name="internal", model=MAIN_MODEL, lead=24,
                 cold=()):
    """Таблица по каждому сценарию: скилл по лидам, метрики и порог на одном лиде.

    Восклицательный знак после скилла отмечает нарушение порога проверки.

    Args:
        rows: строки сценариев.
        cfg: настройки робастности.
        set_name: имя набора.
        model: имя модели.
        lead: лид, на котором печатаются метрики и порог.
        cold: строки холодного старта; печатаются отдельной строкой перед сценариями.
    """
    lead = lead if lead in cfg.leads else cfg.leads[len(cfg.leads) // 2]
    print_cold_start(cold, cfg, set_name=set_name, model=model)
    for spec in cfg.scenarios:
        rule = spec.rule
        tgt = "история и цель" if rule.target else "только история"
        print(f"\n=== [{set_name}] {spec.name}: {rule.title} ({KIND_RU[rule.kind]}; "
              f"искажается {tgt}; {_guard_rule(spec, cfg.skill_tolerance)}) ===")
        print(f"    уровень: {rule.unit}")
        head = f"{'уровень':>14} " + "".join(f"{'Sk@' + str(h):>10}" for h in cfg.leads)
        head += (f" {'порог@' + str(lead):>9} {'MAE@' + str(lead):>8} {'CRPS':>7}"
                 f" {'PICP90':>7} {'Sk макро':>9} {'ΔMAE':>7} {'искаж.':>7} {'превыш.':>8}")
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
                low = np.isfinite(sk) and sk < skill_floor(by[h], cfg.skill_tolerance)
                flag = "!" if spec.guard and low else " "
                line += f"{sk:>+9.1%}{flag}" if np.isfinite(sk) else f"{'—':>10}"
            floor = f"{skill_floor(r, cfg.skill_tolerance):>+9.1%}" if spec.guard else f"{'—':>9}"
            line += (f" {floor} {r['MAE']:>8.2f} {r['CRPS']:>7.2f} {r['PICP90']:>7.1%}"
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
    """Кривые MAE, CRPS, покрытия и скилла по уровню сценария с интервалами по станциям.

    Каждый лид - своя кривая. На панели скилла пунктиром того же цвета - порог проверки
    на этом лиде. Для смещения и дрейфа поверх штрихами - вариант, где искажён только
    вход.

    Args:
        rows: строки робастности.
        cfg: конфиг робастности.
        spec: сценарий.
        out_dir: каталог картинок.
        set_name: имя набора в имени файла.
        model: модель на графике.

    Returns:
        Путь к картинке.
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
            if metric == "Skill" and spec.guard:
                ax.axhline(skill_floor(sel[0], cfg.skill_tolerance), color=line.get_color(),
                           lw=1, ls=":",
                           label="порог проверки" if lead == cfg.leads[0] else None)
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
    """Скилл на одном лиде по доле от максимального уровня, все сценарии на одном графике.

    Горизонтальные линии: скилл холодного старта модели на этом лиде и два порога -
    для отказа входа и для свойства прибора. Если холодный старт не хуже климатологии,
    пороги совпадают.

    Args:
        rows: строки робастности.
        cfg: конфиг робастности.
        out_dir: каталог картинок.
        set_name: имя набора в имени файла.
        model: модель на графике.
        lead: лид, ч.

    Returns:
        Путь к картинке.
    """
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
    here = _select(rows, set=set_name, model=model, lead=lead)
    l0 = here[0].get("Skill_L0", float("nan")) if here else float("nan")
    if np.isfinite(l0):
        ax.axhline(l0, color="black", lw=1, ls="--", label="холодный старт")
        ax.axhline(min(0.0, l0) - cfg.skill_tolerance, color="red", lw=1, ls="-.",
                   label="порог отказа входа")
    ax.axhline(-cfg.skill_tolerance, color="red", lw=1, ls=":",
               label=f"порог свойства прибора, допуск {cfg.skill_tolerance:.0%}")
    ax.set_xlabel("доля от максимального уровня сценария")
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
                    "переобучения, кривые метрик и проверка скилла. Отказ входа: не хуже "
                    "своего холодного старта или климатологии, с допуском. Свойство "
                    "прибора: эталон - климатология того же прибора.")
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
    run_checklist(store, datasets=[base], checkpoints=all_ckpts)
    rec = load_run_record(args.ckpt)
    boot_seed = args.eval_seed if args.eval_seed is not None else int(rec["seeds"]["eval"])

    named = {MAIN_MODEL: load_model(args.ckpt)}
    for arch, c in baseline_ckpts.items():
        named[NEURAL_BASELINES[arch]] = load_model(c)
    r_damped = None
    if not args.no_statistical:
        r_damped = BL.fit_damped_persistence(
            {k: s for k, s in clims.items() if s["role"] == ROLE_TRAIN}, n_windows=20000)
    kw = dict(r_damped=r_damped, statistical=not args.no_statistical,
              device=args.device, n_boot=args.bootstrap, boot_seed=boot_seed)

    sets = {"internal": base}
    cold = cold_start_sweep(named, base, cfg, set_name="internal", **kw)
    rows = robustness_sweep(named, base, cfg, set_name="internal", cold=cold, **kw)
    if args.external_manifest:
        ext_store = get_store(args.external_manifest)
        ext = base_eval_set(ext_store.clims(), args.external_manifest, cfg,
                            roles=(ROLE_EXTERNAL,), time_key="test")
        run_checklist(ext_store, datasets=[ext])
        check_external(store, ext_store, checkpoints=all_ckpts)
        sets["external"] = ext
        cold_ext = cold_start_sweep(named, ext, cfg, set_name="external", **kw)
        cold = cold + cold_ext
        rows += robustness_sweep(named, ext, cfg, set_name="external", cold=cold_ext, **kw)

    for name in sets:
        print_report(rows, cfg, set_name=name, cold=cold)
        for p in plot_all(rows, cfg, args.out_dir, set_name=name):
            print("  ", p)
    bad = skill_violations(rows, cfg.skill_tolerance, cfg.guard_models)
    meta = dict(ckpt=args.ckpt, baselines=baseline_ckpts, outputs="raw",
                input_floor="холодный старт той же модели, если он хуже климатологии",
                instrument_reference="климатология, искажённая тем же прибором, что цель",
                qc=cfg.qc, boot_seed=boot_seed,
                sets={n: dict(station_roles=list(d.station_splits), n_windows=len(d),
                              n_stations=len({s for s, _t in d.items}))
                      for n, d in sets.items()})
    for p in save_results(rows, args.out_dir, cfg, meta=meta, violations=bad, cold_start=cold):
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
