"""Единый стенд оценки МАЯК.

Все числа считает ``mayak/metrics.py`` - здесь только сбор окон, сбор
предсказаний и печать. Содержит:

  * EvalSet — окна для оценки со стратифицированной подвыборкой (фиксированное
    число окон с каждой станции) и метаданными окон для разрезов;
  * сбор предсказаний МАЯК + нейробейзлайнов (GRU, DLinear, LRU, PatchTST) +
    статистических; перед сравнением проверяется, что все чекпойнты обучены по одному
    протоколу (mayak.lit.check_comparable);
  * таблицы в двух видах агрегирования — пуловом и макро - с доверительными
    интервалами блочного бутстрапа по станциям;
  * разрезы: по лидам, ролям станций, полным зонам Кёппена, сезонам, длине
    доступной истории и доле валидных часов в истории;
  * метрики надёжности: гистограмма PIT, диаграмма надёжности, острота против
    покрытия;
  * графики по каждой метрике, кривую холодного старта и проверку L=0;
  * отчёт о влиянии конформной калибровки (PICP/CRPS/MAE до и после);
  * покрытие по разрезам с вердиктами и критерий условной поправки; --save-preds
    сохраняет предсказания всех моделей,
    чтобы анализ калибровки шёл без повторного запуска моделей;
  * внешний тест на наблюдениях реальной сети (--external-manifest): те же
    таблицы и разрезы, разрезы внешнего теста (шаг отчётности, Δ высоты станции и
    ЦМР, канал давления) и сопоставление «внутренний тест против внешнего»;
  * проверки этапа A (поле), декомпозицию L=0, суточные амплитуды, ablation.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mayak import baselines as BL
from mayak.constants import H, L_MAX, QUANTILES
from mayak.data.dataset import (block_starts, footprint, history_len, norm_scale, slice_context,
                                slice_history, slice_target, station_qc_elev)
from mayak.data.masking import DEFAULT_TARGET_MASK, FilterStats, enforce_invariant
from mayak.data.qc import qc_window
from mayak.metrics import (FINE_LEADS, LEAD_BINS, NQ, Evaluation, apply_conformal, breakdown,
                           by_lead, calibrate_forecast, coverage, metric_table, pinball_crps,
                           seed_spread, skill, wmean)
from mayak.timeaxis import window_calendar, window_month
from mayak.zones import SEASON_RU, normalize_zone, season_of

Q = np.array(QUANTILES, np.float32)

HISTORY_BINS = ((0, 0, "L=0"), (1, 24, "L 1-24ч"), (25, 168, "L 25-168ч"),
                (169, L_MAX, f"L 169-{L_MAX}ч"))
HIST_VALID_BINS = ((-0.01, 0.5, "<50%"), (0.5, 0.8, "50-80%"),
                   (0.8, 0.95, "80-95%"), (0.95, 1.01, "95-100%"))
MIN_WINDOWS, MIN_STATIONS = 20, 2
PRESSURE_YES, PRESSURE_NO = "есть давление", "нет давления"
BOOTSTRAP = dict(n_boot=1000, seed=0, level=0.90)

NEURAL_BASELINES = {"gru": "GRU seq2seq", "dlinear": "DLinear", "lru": "LRU",
                    "patchtst": "PatchTST"}

BENCHMARK_NOTE = (
    "Эталон скилла — эмпирическая климатология самой станции, подогнанная по её\n"
    "многолетнему обучающему окну. При короткой истории эталон располагает бо́льшим\n"
    "объёмом информации о станции, чем модель: оценка строга в пользу эталона.\n"
    "Пуловая метрика — по всем парам «окно × лид» сразу, её доминируют станции\n"
    "с большой изменчивостью; макро — среднее по станциям, «типичная станция».")


def bin_label(value, bins):
    for lo, hi, name in bins:
        if lo <= value <= hi:
            return name
    return "прочее"


def stratified_items(per_station, max_windows=None, windows_per_station=None):
    """Стратифицированная подвыборка окон: фиксированное число окон со станции."""
    stations = [sid for sid, ts in per_station.items() if len(ts)]
    if not stations:
        return []
    k = windows_per_station
    if k is None and max_windows:
        k = max(1, int(max_windows) // len(stations))
    out = []
    for sid in stations:
        ts = list(per_station[sid])
        if k is not None and len(ts) > k:
            idx = np.unique(np.linspace(0, len(ts) - 1, k).round().astype(np.int64))
            ts = [ts[i] for i in idx]
        out += [(sid, int(t)) for t in ts]
    return out


class EvalSet(Dataset):
    """Окна для оценки.

    Attributes:
        items: пары из станции и часа начала горизонта.
    """
    def __init__(self, clims, station_splits=("train", "unseen_test"),
                 manifest="data/manifest.csv", time_key="test",
                 every_hours=72, L=None, max_windows=6000, windows_per_station=None,
                 target_mask=DEFAULT_TARGET_MASK):
        from mayak.data.splits import time_layout
        from mayak.data.store import read_manifest
        split_of = {r["id"]: r.get("split") for r in read_manifest(manifest)}
        self.station_splits, self.time_key = tuple(station_splits), time_key
        self.floor, self.roles = {}, {}
        self.filter_stats = FilterStats()
        per_station = {}
        for sid, s in clims.items():
            sp = split_of.get(sid)
            if sp not in station_splits:
                continue
            layout = time_layout(s["N"])
            self.floor[sid] = layout.history_floor(time_key)
            cand, ok = block_starts(layout, time_key, s["mask"][:, 0], every_hours,
                                    cfg=target_mask)
            self.filter_stats.add(len(cand), len(ok))
            self.roles[sid] = sp
            per_station[sid] = ok.tolist()
        self.filter_stats.report(f"eval/{time_key}")
        self.items = stratified_items(per_station, max_windows, windows_per_station)
        self.clims = clims
        self.L = L

    def __len__(self):
        return len(self.items)

    def footprints(self):
        for sid, t in self.items:
            lo, hi = footprint(t, self.L, self.floor[sid])
            yield dict(sid=sid, N=self.clims[sid]["N"], time_key=self.time_key,
                       lo=np.array([lo]), t=np.array([t]), hi=np.array([hi]))

    def station_attrs(self):
        if getattr(self, "_attrs", None) is None:
            from mayak.external import station_attributes
            sids = {sid for sid, _t in self.items}
            self._attrs = station_attributes({sid: self.clims[sid] for sid in sids})
        return self._attrs

    def window_meta(self):
        """Метки окон для разрезов; ``t`` - час начала горизонта (порядок окон во времени
        нужен офлайн-прогону адаптивной калибровки)."""
        sid_a, role, zone, season, hist, hvalid = [], [], [], [], [], []
        has_p, rep, egap = [], [], []
        attrs = self.station_attrs()
        for sid, t in self.items:
            s = self.clims[sid]
            L = history_len(self.L, t, self.floor[sid])
            m = s["mask"][t - L:t, 0] if L > 0 else np.zeros(0, np.float32)
            month = int(np.asarray(window_month(s["t0"], [t])).ravel()[0])
            sid_a.append(sid)
            role.append(self.roles[sid])
            zone.append(normalize_zone(s["koppen"]))
            season.append(SEASON_RU[season_of(month, s["lat"])])
            hist.append(L)
            hvalid.append(float((m > 0).mean()) if L > 0 else 0.0)
            mp = s["mask"][t - L:t, 1] if L > 0 else np.zeros(0, np.float32)
            has_p.append(PRESSURE_YES if (mp > 0).any() else PRESSURE_NO)
            rep.append(attrs[sid]["report_class"])
            egap.append(attrs[sid]["elev_gap_label"])
        return dict(station=np.array(sid_a, object), role=np.array(role, object),
                    zone=np.array(zone, object), season=np.array(season, object),
                    history=np.array(hist, np.int64), hist_valid=np.array(hvalid, np.float64),
                    has_pressure=np.array(has_p, object), report_class=np.array(rep, object),
                    elev_gap=np.array(egap, object),
                    t=np.array([t for _sid, t in self.items], np.int64))

    def raw_window(self, i):
        """Сырая история окна до QC.

        Args:
            i: номер окна.

        Returns:
            Словарь: значения и маска наличия истории формы (L_MAX, 3), контекст QC
            до истории, длина истории и высота для проверки давления.
        """
        sid, t = self.items[i]
        s = self.clims[sid]
        L = history_len(self.L, t, self.floor[sid])
        x, m = slice_history(s["raw"], s["present"], t, L)
        past = slice_context(s["raw"], s["present"], t, L, self.floor[sid])
        return dict(x=x, m=m, past=past, L=L, qc_elev=station_qc_elev(s))

    def __getitem__(self, i):
        sid, t = self.items[i]
        s = self.clims[sid]
        clim = s["clim"]
        t0 = s["t0"]
        k = np.arange(L_MAX)
        abs_h = t - L_MAX + k
        doy_h, hour_h = window_calendar(t0, abs_h)
        w = self.raw_window(i)
        x_hist, mask_hist = w["x"], w["m"]
        if w["L"] > 0:
            mask_hist, _ = qc_window(x_hist, mask_hist, elev=w["qc_elev"], past=w["past"])
            x_hist, mask_hist = enforce_invariant(x_hist, mask_hist)
        fut = np.arange(t, t + H)
        doy_f, hour_f = window_calendar(t0, fut)
        y, y_mask = slice_target(s["x"], s["mask"], t)
        mu_clim_fut = clim.predict(doy_f, hour_f).astype(np.float32)
        a_recent, _ = BL.recent_anomaly(x_hist[:, 0], mask_hist[:, 0], clim, L_MAX,
                                        int(t0) + int(t) - L_MAX)
        return {
            "lat": torch.tensor(s["lat"], dtype=torch.float32),
            "lon": torch.tensor(s["lon"], dtype=torch.float32),
            "elev": torch.tensor(s["elev"], dtype=torch.float32),
            "x_hist": torch.from_numpy(x_hist), "mask_hist": torch.from_numpy(mask_hist),
            "doy_hist": torch.from_numpy(doy_h.astype(np.float32)),
            "hour_hist": torch.from_numpy(hour_h.astype(np.float32)),
            "doy_fut": torch.from_numpy(doy_f.astype(np.float32)),
            "hour_fut": torch.from_numpy(hour_f.astype(np.float32)),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
            "norm_scale": torch.from_numpy(norm_scale(clim, doy_f, hour_f)),
            "mu_clim_fut": torch.from_numpy(mu_clim_fut),
            "sigma_clim": torch.tensor(clim.sigma, dtype=torch.float32),
            "a_recent": torch.tensor(a_recent, dtype=torch.float32),
        }


def coverage90(y, q, w):
    return coverage(y, q, w)


@torch.no_grad()
def gather(model, dataset, device="cpu", batch_size=128):
    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch_size)
    ys, yms, mus, qs, mucl = [], [], [], [], []
    a_rec, sig_cl, xh, mh = [], [], [], []
    for b in dl:
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        o = model(bb)
        ys.append(b["y"].numpy())
        yms.append(b["y_mask"].numpy())
        mus.append(o["mu"].cpu().numpy())
        qs.append(o["q"].cpu().numpy())
        mucl.append(b["mu_clim_fut"].numpy())
        a_rec.append(b["a_recent"].numpy())
        sig_cl.append(b["sigma_clim"].numpy())
        xh.append(b["x_hist"].numpy())
        mh.append(b["mask_hist"].numpy())
    cat = lambda L: np.concatenate(L, 0)
    return dict(y=cat(ys), y_mask=cat(yms), mu=cat(mus), q=cat(qs), mu_clim=cat(mucl),
                a_recent=cat(a_rec), sigma_clim=cat(sig_cl), x_hist=cat(xh), mask_hist=cat(mh))


@torch.no_grad()
def _gather_full(model, ds, device="cpu", batch_size=128):
    """Как gather, но дополнительно тащит o, r, e, sigma_c (нужны для L=0-проверки)."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size)
    keys = ["mu", "q", "o", "r", "e", "sigma_c"]
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


def collect_predictions(named_models, ds, device="cpu"):
    preds, aux = {}, None
    for name, mdl in named_models.items():
        D = gather(mdl, ds, device=device)
        if aux is None:
            aux = D
            aux["meta"] = ds.window_meta()
        preds[name] = dict(mu=D["mu"], q=D["q"])
    return preds, aux


def add_statistical_baselines(preds, aux, r_damped=None):
    mucl, sig = aux["mu_clim"], aux["sigma_clim"]
    mu_b, q_b = BL.climatology_forecast(mucl, sig)
    preds["Климатология"] = dict(mu=mu_b, q=q_b)
    if r_damped is not None:
        mu_b, q_b = BL.damped_persistence_forecast(aux["a_recent"], mucl, sig, r_damped)
        preds["Damped persistence"] = dict(mu=mu_b, q=q_b)
    mu_b, q_b = BL.seasonal_naive_forecast(aux["x_hist"], aux["mask_hist"], mucl, sig, period=24)
    preds["Seasonal-naive 24ч"] = dict(mu=mu_b, q=q_b)
    return preds


def evaluation_for(pred, aux, shift=None):
    """Предсказания одной модели → ``Evaluation``"""
    ev = Evaluation(y=aux["y"], mu=pred["mu"], q=pred["q"], mu_clim=aux["mu_clim"],
                    w=aux["y_mask"], station=aux["meta"]["station"])
    return ev.with_conformal(shift)


def evaluations(preds, aux, shift=None):
    return {name: evaluation_for(p, aux, shift) for name, p in preds.items()}


def build_tables(preds, aux, leads=FINE_LEADS):
    """Пуловые таблицы по лидам - вход для графиков ``plot_metric_curves``."""
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


def show(tbl):
    print(f"{'лид,ч':>6} {'MAE':>6} {'RMSE':>6} {'Skill':>7} {'CRPS':>6} "
          f"{'PICP80':>7} {'PICP90':>7} {'Wink90':>7}")
    for h, m in tbl.items():
        print(f"{h:>6} {m['MAE']:>6.2f} {m['RMSE']:>6.2f} {m['Skill']:>+7.1%} "
              f"{m['CRPS']:>6.2f} {m['PICP80']:>7.1%} {m['PICP90']:>7.1%} {m['Winkler90']:>7.2f}")


def print_lead_table(ev, leads=(1, 3, 6, 12, 24, 48, 72, 120, 168), ci=False, **kw):
    rows = {str(h): s for h, s in by_lead(ev, leads=leads, ci=ci, **kw).items()}
    print_rows(rows, label="лид, ч", ci=ci)


def all_breakdowns(ev, meta, leads=None, min_windows=MIN_WINDOWS, min_stations=MIN_STATIONS,
                   ci=False, **kw):
    hist = np.array([bin_label(int(v), HISTORY_BINS) for v in meta["history"]], object)
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    kwargs = dict(leads=leads, min_windows=min_windows, min_stations=min_stations, ci=ci, **kw)
    return {
        "роль станции": breakdown(ev, meta["role"], **kwargs),
        "зона Кёппена": breakdown(ev, meta["zone"], **kwargs),
        "сезон": breakdown(ev, meta["season"], **kwargs),
        "длина истории": breakdown(ev, hist, **kwargs),
        "валидность истории": breakdown(ev, hvalid, **kwargs),
    }


def external_breakdowns(ev, meta, leads=None, min_windows=MIN_WINDOWS,
                        min_stations=MIN_STATIONS, ci=False, **kw):
    hvalid = np.array([bin_label(float(v), HIST_VALID_BINS) for v in meta["hist_valid"]], object)
    kwargs = dict(leads=leads, min_windows=min_windows, min_stations=min_stations, ci=ci, **kw)
    return {
        "валидность истории": breakdown(ev, hvalid, **kwargs),
        "частота отчётности": breakdown(ev, meta["report_class"], **kwargs),
        "Δ высоты станция−ЦМР": breakdown(ev, meta["elev_gap"], **kwargs),
        "канал давления": breakdown(ev, meta["has_pressure"], **kwargs),
    }


def print_breakdowns(ev, meta, leads=None, ci=False, fn=None, **kw):
    fn = fn or all_breakdowns
    for name, rows in fn(ev, meta, leads=leads, ci=ci, **kw).items():
        print(f"\n--- разрез: {name} ---")
        if not rows:
            print("  (все страты меньше порога по числу окон или станций)")
            continue
        print_rows(rows, label=name, ci=ci)


def print_reliability(ev, lead_bins=LEAD_BINS):
    """Гистограмма PIT (по всему горизонту и по бинам лидов), надёжность, острота."""
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
    summaries = [ev.restrict(leads=[lead]).summary() for ev in evs_by_seed]
    print(f"\n--- разброс по {len(evs_by_seed)} сидам, лид {lead} ч ---")
    print(f"{'метрика':>10} {'среднее':>10} {'мин':>10} {'макс':>10} {'ст.откл.':>10}")
    for m, s in seed_spread(summaries).items():
        if not np.isfinite(s["mean"]):
            continue
        print(f"{m:>10} {s['mean']:>10.3f} {s['min']:>10.3f} {s['max']:>10.3f} {s['std']:>10.3f}")


def koppen_per_window(ds):
    return np.array([normalize_zone(ds.clims[sid]["koppen"]) for sid, _t in ds.items])


def zone_breakdown(preds, aux, koppen=None, leads=(24, 72), model="МАЯК",
                   min_windows=MIN_WINDOWS, min_stations=MIN_STATIONS):
    """Разрез по полным зонам Кёппена для одной модели: {зона: {лид: сводка}}."""
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
    """Диаграмма надёжности и кривая «острота против покрытия» одной картинкой."""
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
                           station_splits=("train", "unseen_test"),
                           shift=None, seed=0, L=None):
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
        if shift is not None:
            q, mu = calibrate_forecast(q, shift)
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


def evaluate_all(model, clims, manifest="data/manifest.csv", r_damped=None,
                 named_extra=None, ds=None, preds=None, aux=None, shift=None,
                 ci=True, bootstrap=BOOTSTRAP):
    """Главная таблица: все модели, обе агрегации, интервалы, разрезы для МАЯК."""
    if ds is None:
        ds = EvalSet(clims, manifest=manifest, time_key="test")
    if preds is None or aux is None:
        named = {"МАЯК": model}
        if named_extra:
            named.update(named_extra)
        preds, aux = collect_predictions(named, ds)
        preds = add_statistical_baselines(preds, aux, r_damped=r_damped)

    print(BENCHMARK_NOTE)
    evs = evaluations(preds, aux, shift=shift)
    for name, ev in evs.items():
        print(f"\n=== {name} ===")
        print_lead_table(ev, ci=ci, **(bootstrap if ci else {}))

    print("\n=== Общие метрики по всему горизонту ===")
    rows = {name: ev.summary(ci=ci, **(bootstrap if ci else {})) for name, ev in evs.items()}
    print_rows(rows, label="модель", ci=ci)

    print("\n=== Разрезы (МАЯК, лид 24 ч) ===")
    print_breakdowns(evs["МАЯК"], aux["meta"], leads=[24], ci=False)

    print("\n=== Надёжность (МАЯК) ===")
    print_reliability(evs["МАЯК"])
    print_calibration(evs["МАЯК"], aux["meta"], bootstrap=bootstrap if ci else None)
    return preds, aux, ds


def print_calibration(ev, meta, bootstrap=None, external=False):
    from dataclasses import replace
    from mayak.calibration import (conditional_gate, coverage_report, load_config,
                                   print_coverage_report)
    cfg = load_config()
    if bootstrap:
        cfg = replace(cfg, bootstrap=int(bootstrap["n_boot"]), seed=int(bootstrap["seed"]),
                      ci_level=float(bootstrap["level"]))
    else:
        cfg = replace(cfg, bootstrap=0)
    rep = coverage_report(ev, meta, cfg, external=external)
    print_coverage_report(rep, conditional_gate(rep, cfg),
                          title=f"\n=== {'[внешний] ' if external else ''}Покрытие по разрезам "
                                f"(МАЯК, весь горизонт) ===")
    return rep


def coldstart_curve(model, clims, manifest="data/manifest.csv",
                    Ls=(0, 6, 24, 72, 168, 336, 672)):
    print("\n=== Кривая холодного старта (Skill-24ч от L) ===")
    res = {}
    for L in Ls:
        ds = EvalSet(clims, manifest=manifest, time_key="test", L=L, every_hours=120)
        D = gather(model, ds)
        j = 24 - 1
        sk = skill(D["y"][:, j], D["mu"][:, j], D["mu_clim"][:, j], D["y_mask"][:, j])
        res[L] = float(sk)
        print(f"  L={L:>4} ч : Skill-24ч = {sk:+.1%}")
    if res.get(672, 0) > 0:
        print(f"  Доля при L=24ч от полного: {res.get(24, 0) / res[672]:.0%}  (цель ≥80%)")
    return res


def coldstart_L0_check(model, clims, manifest="data/manifest.csv", shift=None):
    ds = EvalSet(clims, manifest=manifest, time_key="test", L=0, every_hours=72)
    D = _gather_full(model, ds)
    o_abs = float(np.abs(D["o"]).mean())
    e_mean = float(np.abs(D["e"]).mean())
    dev = np.abs(D["sigma_c"] * (D["o"] + D["r"]))
    p_before = coverage90(D["y"], D["q"], D["y_mask"])
    diff_clim = float(np.abs(D["mu"] - D["mu_clim"]).mean())

    print(f"  mean|o| = {o_abs:.3f} σ   (аномалия включена? должно быть ≈0)")
    print(f"  mean e  = {e_mean:.3f}     (масса свидетельств; должно быть ≈0)")
    print(f"  |медиана − поле модели| = {dev.mean():.3f} °C  (макс {dev.max():.2f}); "
          f"= σ_c·(o+r), т.е. медиана почти равна якорю-полю")
    print(f"  |медиана − эмпирич. климатология| = {diff_clim:.3f} °C  "
          f"(ожидается мало: поле ≈ климатология, критерий этапа A)")
    print(f"  PICP-90 при L=0 = {p_before:.1%}  (цель 86–94%)", end="")
    if shift is not None:
        p_after = coverage90(D["y"], apply_conformal(D["q"], shift), D["y_mask"])
        print(f"  →  после конформной {p_after:.1%}")
    else:
        print()
    return dict(o_abs=o_abs, e_mean=e_mean, dev_deg=float(dev.mean()),
                picp90=p_before, diff_clim=diff_clim)


def calibration_quality_report(model, clims, shift, manifest="data/manifest.csv"):
    ds = EvalSet(clims, manifest=manifest, time_key="test", every_hours=72)
    D = gather(model, ds)
    y, q0, w = D["y"], D["q"], D["y_mask"]
    q1 = apply_conformal(q0, shift)
    print(f"{'бин лидов':>11} {'PICP90 до':>10} {'PICP90 после':>13} "
          f"{'CRPS до':>9} {'CRPS после':>11} {'MAEмед до':>10} {'MAEмед после':>13}")
    for a, b in LEAD_BINS:
        sl = slice(a - 1, b)
        ys, ws = y[:, sl], w[:, sl]
        p0 = coverage(ys, q0[:, sl], ws)
        p1 = coverage(ys, q1[:, sl], ws)
        c0 = wmean(pinball_crps(ys, q0[:, sl]), ws)
        c1 = wmean(pinball_crps(ys, q1[:, sl]), ws)
        m0 = wmean(np.abs(q0[:, sl, 3] - ys), ws)
        m1 = wmean(np.abs(q1[:, sl, 3] - ys), ws)
        print(f"{str(a) + '-' + str(b):>11} {p0:>10.1%} {p1:>13.1%} "
              f"{c0:>9.3f} {c1:>11.3f} {m0:>10.3f} {m1:>13.3f}")
    print(f"  MAE точечного прогноза mu (конформная таблица его НЕ трогает): "
          f"{wmean(np.abs(D['mu'] - y), w):.3f} °C")


def stage_a_field_check(model, clims, manifest="data/manifest.csv",
                        station_split="unseen_val", time_key="val"):
    """Критерий этапа A"""
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest,
                 time_key=time_key, L=0)
    D = gather(model, ds)
    y, mu, muc, w = D["y"], D["mu"], D["mu_clim"], D["y_mask"]
    mse_field = float(wmean((mu - y) ** 2, w))
    mse_clim = float(wmean((muc - y) ** 2, w))
    ratio = mse_field / max(mse_clim, 1e-9)
    bias = float(np.abs(mu - muc).mean())
    print(f"\n[Проверка этапа A] поле на {station_split} (L=0, окон: {len(ds)}):")
    print(f"  MSE поля         = {mse_field:7.3f}")
    print(f"  MSE климатологии = {mse_clim:7.3f}   ← эталон")
    print(f"  отношение        = {ratio:7.3f}   ← цель ≤ 1.05")
    print(f"  |поле − климат|  = {bias:7.3f} °C ← цель → 0")
    if ratio <= 1.05:
        print("  ИТОГ: OK — поле генерализует")
    else:
        print(
            "  ИТОГ: НЕ ПРОЙДЕНО — поле недоучено/переобучено на train; "
            "этап B на таком поле смысла мало")
    return dict(mse_field=mse_field, mse_clim=mse_clim, ratio=ratio, bias=bias)


@torch.no_grad()
def pure_field_check(model, clims, manifest="data/manifest.csv",
                     station_split="unseen_val", time_key="val"):
    """Чистое поле: field.coefficients(loc) БЕЗ паспорта (z=None) и БЕЗ r-головы."""
    from mayak.astro import astro_features
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest,
                 time_key=time_key, L=0)
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
    print(f"ЧИСТОЕ поле на {station_split}: ratio={e2 / max(ec, 1e-9):.3f}, "
          f"|поле−клим|={bias / n:.3f}°C  (окон: {len(ds)})")


@torch.no_grad()
def l0_decompose(model, clims, manifest="data/manifest.csv", station_split="train",
                 time_key="val"):
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest, time_key=time_key, L=0)
    model.eval()
    Z = []
    sr = oo = ee = 0.0
    n = 0
    for b in DataLoader(ds, batch_size=128):
        out = model(b)
        Z.append(out["z"])
        sr += float((out["sigma_c"] * out["r"]).abs().sum())
        oo += float(out["o"].abs().sum())
        ee += float(out["e"].abs().sum())
        n += out["mu"].numel()
    Z = torch.cat(Z, 0)
    print(f"[{station_split}] std(z) по станциям = {float(Z.std(0).mean()):.3f}  "
          f"(≈0 → прайор глобальный; >0 → прайор зависит от loc = меморизатор)")
    print(f"        |σ·r| = {sr / n:.3f}°C  (≈0 → r заглушён; >0 → r ещё активен и фитит)")
    print(f"        |o| = {oo / n:.3f}   e = {ee / Z.numel() * Z.shape[1]:.3f}  (ждём ≈0 при L=0)")


def compare_ablation(model_full, model_ablated, clims, manifest="data/manifest.csv", lead=24):
    ds = EvalSet(clims, manifest=manifest, time_key="test", every_hours=120)
    out = {}
    for tag, mdl in [("full", model_full), ("ablated", model_ablated)]:
        D = gather(mdl, ds)
        j = lead - 1
        out[tag] = skill(D["y"][:, j], D["mu"][:, j], D["mu_clim"][:, j], D["y_mask"][:, j])
    print(f"Skill-{lead}ч: full {out['full']:+.1%}  vs  ablated {out['ablated']:+.1%}  "
          f"(падение {out['full'] - out['ablated']:+.1%})")
    return out


def evaluate_external(named, external_manifest, store, r_damped=None, shift=None,
                      ci=True, bootstrap=BOOTSTRAP, checkpoints=(), conformal=None,
                      internal=None, transfer_level="group"):
    """Внешний тест: станции реальной сети (роль external_test), только их тестовое окно.

    named    - {имя: модель}, те же модели и чекпойнты, что во внутренней оценке;
    store    - основной набор (для чек-листа изоляции внешнего теста);
    internal - (preds, aux) внутренней оценки для сопоставления «внутренний против
               внешнего»; None - без сопоставления.
    Эталон для всех моделей один - климатология каждой внешней станции по её
    собственному обучающему окну (строится при сборке кэша внешнего набора).
    """
    from mayak.data.splits import ROLE_EXTERNAL, ROLE_TEST
    from mayak.data.store import get_store
    from mayak.external import print_transfer, transfer_table
    from mayak.leakage import check_external, run_checklist
    ext_store = get_store(external_manifest)
    ds = EvalSet(ext_store.clims(), station_splits=(ROLE_EXTERNAL,), manifest=external_manifest,
                 time_key="test")
    run_checklist(ext_store, datasets=[ds])
    check_external(store, ext_store, checkpoints=checkpoints, conformal=conformal)
    preds, aux = collect_predictions(named, ds)
    preds = add_statistical_baselines(preds, aux, r_damped=r_damped)
    kw = bootstrap if ci else {}

    print(f"\n########## ВНЕШНИЙ ТЕСТ: {len(ext_store.stations)} станций, "
          f"окон {len(ds)} ##########")
    print(BENCHMARK_NOTE)
    evs = evaluations(preds, aux, shift=shift)
    for name, ev in evs.items():
        print(f"\n=== [внешний] {name} ===")
        print_lead_table(ev, ci=ci, **kw)
    print("\n=== [внешний] Общие метрики по всему горизонту ===")
    print_rows({n: ev.summary(ci=ci, **kw) for n, ev in evs.items()}, label="модель", ci=ci)
    print("\n=== [внешний] Разрезы блока 5 (МАЯК, лид 24 ч) ===")
    print_breakdowns(evs["МАЯК"], aux["meta"], leads=[24])
    print("\n=== [внешний] Разрезы внешнего теста (МАЯК, лид 24 ч) ===")
    print_breakdowns(evs["МАЯК"], aux["meta"], leads=[24], fn=external_breakdowns)
    print("\n=== [внешний] Надёжность (МАЯК) ===")
    print_reliability(evs["МАЯК"])
    print_calibration(evs["МАЯК"], aux["meta"], bootstrap=kw or None, external=True)

    transfer = {}
    if internal is not None:
        p_int, a_int = internal
        for name in ("МАЯК", "Климатология"):
            if name not in p_int or name not in preds:
                continue
            ev_i = evaluation_for(p_int[name], a_int, shift)
            ev_e = evs[name]
            for tag, roles in (("все станции внутреннего теста", None),
                               ("только невиденные (unseen_test)", (ROLE_TEST,))):
                sel = (np.ones(len(ev_i.y), bool) if roles is None
                       else np.isin(a_int["meta"]["role"], roles))
                tbl = transfer_table(ev_i.restrict(windows=sel), a_int["meta"]["zone"],
                                     ev_e, aux["meta"]["zone"], level=transfer_level,
                                     n_boot=kw.get("n_boot", 0), seed=kw.get("seed", 0),
                                     ci_level=kw.get("level", 0.90))
                transfer[(name, tag)] = tbl
                print_transfer(tbl, title=f"\n=== Перенос: {name}, внешний против внутреннего "
                                          f"({tag}); Δ = внешний − внутренний ===")
    return preds, aux, ds, transfer


def print_parameter_counts(named):
    """Печатает таблицу числа параметров нейросетевых моделей.

    Args:
        named: словарь из имени строки таблицы в модель.

    Returns:
        Словарь из имени строки в полное число параметров модели.
    """
    from mayak.lit import parameter_counts
    counts = {name: parameter_counts(m)["total"] for name, m in named.items()}
    print("\n=== Число параметров ===")
    for name, n in counts.items():
        print(f"  {name:<40}{n:>12,}".replace(",", " "))
    return counts


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse
    from mayak.config import run_label
    from mayak.lit import SEED_FIELDS, check_comparable, load_model, load_run_record
    from mayak.protocol import ProtocolError
    ap = argparse.ArgumentParser(description="единый стенд оценки МАЯК")
    ap.add_argument("--ckpt", required=True, nargs="+",
                    help="чекпойнты МАЯК; несколько = прогоны с разными сидами")
    ap.add_argument("--manifest", default="data/manifest.csv")
    for arch, name in NEURAL_BASELINES.items():
        ap.add_argument(f"--{arch}-ckpt", default=None,
                        help=f"чекпойнт бейзлайна «{name}» "
                             f"(scripts/train.py --arch {arch})")
    ap.add_argument("--allow-protocol-mismatch", action="store_true",
                    help="не падать, если модели обучены по разным протоколам "
                         "(только для диагностики: такие таблицы несопоставимы)")
    ap.add_argument("--ablation-ckpt", nargs="*", default=[],
                    help="чекпойнты переобученных абляций МАЯК (тот же сид и протокол); "
                         "имя строки таблицы берётся из конфига в чекпойнте")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="сид оценки (бутстрап, примеры); по умолчанию - seeds.eval "
                         "из первого чекпойнта, иначе 0")
    ap.add_argument("--conformal", default=None, help="runs/conformal.npy (если есть)")
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
                    help="сохранить предсказания всех моделей (до калибровки) в DIR/internal.npz "
                         "и DIR/external.npz - для python -m mayak.calibration")
    args = ap.parse_args()

    from mayak.data.store import get_store
    from mayak.data.splits import ROLE_TRAIN
    from mayak.leakage import run_checklist
    baseline_ckpts = {a: getattr(args, f"{a}_ckpt") for a in NEURAL_BASELINES
                      if getattr(args, f"{a}_ckpt")}
    all_ckpts = [*args.ckpt, *baseline_ckpts.values(), *args.ablation_ckpt]
    try:
        check_comparable(args.ckpt[0], [*baseline_ckpts.values(), *args.ablation_ckpt])
        check_comparable(args.ckpt[0], args.ckpt[1:], ignore=SEED_FIELDS)
    except ProtocolError as e:
        if not args.allow_protocol_mismatch:
            raise
        print(f"ВНИМАНИЕ: {e}\nТаблицы ниже несопоставимы (--allow-protocol-mismatch).")

    store = get_store(args.manifest)
    clims = store.clims()
    ds = EvalSet(clims, manifest=args.manifest, time_key="test")
    run_checklist(store, datasets=[ds], conformal=args.conformal, checkpoints=all_ckpts)
    rec = load_run_record(args.ckpt[0])
    eval_seed = args.eval_seed
    if eval_seed is None:
        eval_seed = int(rec["seeds"]["eval"]) if rec else 0
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

    named_all = {"МАЯК": mayak, **named_extra}
    n_params = print_parameter_counts(named_all)
    preds, aux = collect_predictions(named_all, ds)
    preds = add_statistical_baselines(preds, aux, r_damped=r)

    shift = np.load(args.conformal) if args.conformal else None
    boot = dict(n_boot=args.bootstrap, seed=eval_seed, level=args.ci_level)
    info = dict(ckpt=args.ckpt, conformal=args.conformal, manifest=args.manifest,
                eval_seed=eval_seed, n_params=n_params)
    if args.save_preds:
        from mayak.calibration import save_predictions
        print("Предсказания:", save_predictions(os.path.join(args.save_preds, "internal.npz"),
                                                preds, aux, shift=shift, info=info))

    print("\n=== Таблицы метрик ===")
    evaluate_all(mayak, clims, args.manifest, r_damped=r,
                 named_extra=named_extra, ds=ds, preds=preds, aux=aux,
                 shift=shift, ci=args.bootstrap > 0, bootstrap=boot)

    if len(seeds) > 1:
        evs = [evaluation_for(dict(zip(("mu", "q"), _mu_q(m, ds))), aux, shift) for m in seeds[1:]]
        evs = [evaluation_for(preds["МАЯК"], aux, shift)] + evs
        print_seed_spread(evs)

    coldstart_curve(mayak, clims, args.manifest)

    tables = build_tables(preds, aux)
    print("\n=== Графики по метрикам ===")
    for p in plot_metric_curves(tables, args.out_dir):
        print("  ", p)

    ev_mayak = evaluation_for(preds["МАЯК"], aux, shift)
    print("  ", plot_reliability(ev_mayak, args.out_dir))
    print("  ", plot_pit(ev_mayak, args.out_dir))

    print("\n=== Разрез по зонам Кёппена (МАЯК) ===")
    print_zone_breakdown(zone_breakdown(preds, aux))

    print("\n=== Холодный старт L=0 ===")
    coldstart_L0_check(mayak, clims, args.manifest, shift=shift)

    if shift is not None:
        print("\n=== Влияние конформной калибровки ===")
        calibration_quality_report(mayak, clims, shift, args.manifest)

    print("\n=== Графики прогноз vs факт (примеры МАЯК) ===")
    plot_forecast_examples(mayak, clims, manifest=args.manifest,
                           n=args.n_examples, out_dir=args.out_dir, shift=shift, seed=eval_seed)
    plot_forecast_examples(mayak, clims, manifest=args.manifest, n=args.n_examples,
                           out_dir=args.out_dir + "/unseen",
                           station_splits=("unseen_test",), shift=shift, seed=eval_seed)
    plot_forecast_examples(mayak, clims, manifest=args.manifest, n=args.n_examples, L=0,
                           out_dir=args.out_dir, shift=shift, seed=eval_seed)
    plot_forecast_examples(mayak, clims, manifest=args.manifest, n=args.n_examples, L=0,
                           station_splits=("unseen_test",),
                           out_dir=args.out_dir + "/unseen", shift=shift, seed=eval_seed)
    print("\n=== Суточные амплитуды ===")
    plot_amplitude_scatter(mayak, clims, manifest=args.manifest, out_dir=args.out_dir)

    if args.external_manifest:
        p_ext, a_ext, _ds, _tr = evaluate_external(
            named_all, args.external_manifest, store, r_damped=r, shift=shift,
            ci=args.bootstrap > 0, bootstrap=boot, checkpoints=all_ckpts,
            conformal=args.conformal, internal=(preds, aux), transfer_level=args.transfer_zones)
        if args.save_preds:
            from mayak.calibration import save_predictions
            print("Предсказания:", save_predictions(
                os.path.join(args.save_preds, "external.npz"), p_ext, a_ext, shift=shift,
                info=dict(info, manifest=args.external_manifest)))


def _mu_q(model, ds):
    D = gather(model, ds)
    return D["mu"], D["q"]


if __name__ == "__main__":
    main()
