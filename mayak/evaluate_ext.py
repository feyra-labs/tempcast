"""Расширения стенда оценки (аддендум к §20–§21):
  * сбор предсказаний МАЯК + нейробейзлайнов (GRU/DLinear) + статистических бейзлайнов;
  * графики по каждой метрике (Skill/MAE/RMSE/CRPS/PICP90/Winkler) — линия на модель;
  * разрез метрик по климатическим зонам Кёппена;
  * проверка холодного старта L=0 (медиана = поле, интервалы калиброваны) — пункт 3;
  * отчёт о влиянии конформной калибровки (PICP/CRPS/MAE до и после) — пункт 4.
Зависит только от уже существующих модулей проекта."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from mayak.constants import H, QUANTILES
from mayak import baselines as BL
from mayak.evaluate import EvalSet, gather, metric_table, mse_clim_per_lead, pinball_crps

Q = np.array(QUANTILES, np.float32)
FINE_LEADS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168)
LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]


def _bin(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1


def apply_conformal(q, shift):
    q = np.array(q, np.float32, copy=True)
    for h in range(q.shape[-2]):
        q[..., h, :] += shift[_bin(h + 1)]
    return np.maximum.accumulate(q, axis=-1)


def coverage90(y, q):
    return float(((y >= q[..., 0]) & (y <= q[..., 6])).mean())


def collect_predictions(named_models, ds, device="cpu"):
    preds, aux = {}, None
    for name, mdl in named_models.items():
        D = gather(mdl, ds, device=device)
        if aux is None:
            aux = D
        preds[name] = dict(mu=D["mu"], q=D["q"])
    return preds, aux


def add_statistical_baselines(preds, aux, r_damped=None):
    mucl, sig = aux["mu_clim"], aux["sigma_clim"]
    mu_b, q_b = BL.climatology_forecast(mucl, sig)
    preds["Климатология"] = dict(mu=mu_b, q=q_b)
    if r_damped is not None:
        mu_b, q_b = BL.damped_persistence_forecast(aux["a_recent"], mucl, sig, r_damped)
        preds["Damped persistence"] = dict(mu=mu_b, q=q_b)
    mu_b, q_b = BL.seasonal_naive_forecast(aux["x_hist"], aux["mask_hist"], sig, period=24)
    preds["Seasonal-naive 24ч"] = dict(mu=mu_b, q=q_b)
    return preds


def build_tables(preds, aux, leads=FINE_LEADS):
    y = aux["y"]
    mse_clim = mse_clim_per_lead(y, aux["mu_clim"])
    return {name: metric_table(y, p["mu"], p["q"], mse_clim, leads=leads)
            for name, p in preds.items()}

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
        ax.set_xlabel("лид, ч"); ax.set_ylabel(met)
        ax.set_title(f"{met} по горизонту прогноза")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"metric_{met}.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        paths.append(p)
    return paths


def koppen_per_window(ds):
    return np.array([ds.clims[sid]["koppen"] for sid, _t, _seen in ds.items])


def zone_breakdown(preds, aux, koppen, leads=(24, 72), model="МАЯК", min_windows=20):
    y = aux["y"]; p = preds[model]; rows = {}
    for z in sorted(set(koppen.tolist())):
        m = koppen == z
        if int(m.sum()) < min_windows:
            continue
        d = {"n": int(m.sum())}
        for h in leads:
            j = h - 1
            e = p["mu"][m, j] - y[m, j]
            mc = ((aux["mu_clim"][m, j] - y[m, j]) ** 2).mean()
            lo, hi = p["q"][m, j, 0], p["q"][m, j, 6]
            d[h] = dict(skill=float(1 - (e ** 2).mean() / max(mc, 1e-9)),
                        mae=float(np.abs(e).mean()),
                        picp90=float(((y[m, j] >= lo) & (y[m, j] <= hi)).mean()))
        rows[z] = d
    return rows


def print_zone_breakdown(rows, leads=(24, 72)):
    hdr = f"{'зона':>6} {'окон':>6}"
    for h in leads:
        hdr += f" {'Sk@'+str(h)+'ч':>9} {'MAE@'+str(h):>9} {'P90@'+str(h):>9}"
    print(hdr)
    for z, d in rows.items():
        line = f"{z:>6} {d['n']:>6}"
        for h in leads:
            line += f" {d[h]['skill']:>+9.1%} {d[h]['mae']:>9.2f} {d[h]['picp90']:>9.1%}"
        print(line)


@torch.no_grad()
def _gather_full(model, ds, device="cpu", batch_size=128):
    """Как gather, но дополнительно тащит o, r, e, sigma_c (нужны для L=0-проверки)."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size)
    keys = ["mu", "q", "o", "r", "e", "sigma_c"]
    acc = {k: [] for k in keys}; acc["y"] = []; acc["mu_clim"] = []
    for b in dl:
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(bb)
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
        acc["y"].append(b["y"].numpy()); acc["mu_clim"].append(b["mu_clim_fut"].numpy())
    return {k: np.concatenate(v, 0) for k, v in acc.items()}


def coldstart_L0_check(model, clims, manifest="data/manifest.csv", shift=None):
    ds = EvalSet(clims, manifest=manifest, time_key="test", L=0, every_hours=72)
    D = _gather_full(model, ds)
    o_abs = float(np.abs(D["o"]).mean())
    e_mean = float(np.abs(D["e"]).mean())
    dev = np.abs(D["sigma_c"] * (D["o"] + D["r"]))
    p_before = coverage90(D["y"], D["q"])
    diff_clim = float(np.abs(D["mu"] - D["mu_clim"]).mean())

    print(f"  mean|o| = {o_abs:.3f} σ   (аномалия включена? должно быть ≈0)")
    print(f"  mean e  = {e_mean:.3f}     (масса свидетельств; должно быть ≈0)")
    print(f"  |медиана − поле модели| = {dev.mean():.3f} °C  (макс {dev.max():.2f}); "
          f"= σ_c·(o+r), т.е. медиана почти равна якорю-полю")
    print(f"  |медиана − эмпирич. климатология| = {diff_clim:.3f} °C  "
          f"(ожидается мало: поле ≈ климатология, критерий этапа A)")
    print(f"  PICP-90 при L=0 = {p_before:.1%}  (цель 86–94%)", end="")
    if shift is not None:
        p_after = coverage90(D["y"], apply_conformal(D["q"], shift))
        print(f"  →  после конформной {p_after:.1%}")
    else:
        print()
    return dict(o_abs=o_abs, e_mean=e_mean, dev_deg=float(dev.mean()),
                picp90=p_before, diff_clim=diff_clim)


def calibration_quality_report(model, clims, shift, manifest="data/manifest.csv"):
    ds = EvalSet(clims, manifest=manifest, time_key="test", every_hours=72)
    D = gather(model, ds)
    y, q0 = D["y"], D["q"]
    q1 = apply_conformal(q0, shift)
    print(f"{'бин лидов':>11} {'PICP90 до':>10} {'PICP90 после':>13} "
          f"{'CRPS до':>9} {'CRPS после':>11} {'MAEмед до':>10} {'MAEмед после':>13}")
    for a, b in LEAD_BINS:
        sl = slice(a - 1, b)
        lo0, hi0 = q0[:, sl, 0], q0[:, sl, 6]
        lo1, hi1 = q1[:, sl, 0], q1[:, sl, 6]
        p0 = ((y[:, sl] >= lo0) & (y[:, sl] <= hi0)).mean()
        p1 = ((y[:, sl] >= lo1) & (y[:, sl] <= hi1)).mean()
        c0 = pinball_crps(y[:, sl], q0[:, sl]).mean()
        c1 = pinball_crps(y[:, sl], q1[:, sl]).mean()
        m0 = np.abs(q0[:, sl, 3] - y[:, sl]).mean()
        m1 = np.abs(q1[:, sl, 3] - y[:, sl]).mean()
        print(f"{str(a)+'-'+str(b):>11} {p0:>10.1%} {p1:>13.1%} "
              f"{c0:>9.3f} {c1:>11.3f} {m0:>10.3f} {m1:>13.3f}")
    print(f"  MAE точечного прогноза mu (конформная таблица его НЕ трогает): "
          f"{np.abs(D['mu'] - y).mean():.3f} °C")


def main():
    import argparse
    from mayak.lit import LitMayak, LitBaseline
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="чекпойнт МАЯК (runs/stageB/best.ckpt)")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--gru-ckpt", default=None)
    ap.add_argument("--dlinear-ckpt", default=None)
    ap.add_argument("--conformal", default=None, help="runs/conformal.npy (если есть)")
    ap.add_argument("--out-dir", default="runs/plots")
    args = ap.parse_args()

    clims = BL.fit_climatologies(args.manifest)
    r = BL.fit_damped_persistence(clims, n_windows=20000)
    named = {"МАЯК": LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu").model}
    if args.gru_ckpt:
        named["GRU seq2seq"] = LitBaseline.load_from_checkpoint(args.gru_ckpt, map_location="cpu").model
    if args.dlinear_ckpt:
        named["DLinear"] = LitBaseline.load_from_checkpoint(args.dlinear_ckpt, map_location="cpu").model

    ds = EvalSet(clims, manifest=args.manifest, time_key="test")
    preds, aux = collect_predictions(named, ds)
    preds = add_statistical_baselines(preds, aux, r_damped=r)
    tables = build_tables(preds, aux)

    print("\n=== Графики по метрикам ===")
    for p in plot_metric_curves(tables, args.out_dir):
        print("  ", p)

    print("\n=== Разрез по зонам Кёппена (МАЯК) ===")
    print_zone_breakdown(zone_breakdown(preds, aux, koppen_per_window(ds)))

    shift = np.load(args.conformal) if args.conformal else None
    print("\n=== Холодный старт L=0 (пункт 3) ===")
    coldstart_L0_check(named["МАЯК"], clims, args.manifest, shift=shift)
    if shift is not None:
        print("\n=== Влияние конформной калибровки (пункт 4) ===")
        calibration_quality_report(named["МАЯК"], clims, shift, args.manifest)


if __name__ == "__main__":
    main()