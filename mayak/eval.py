import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mayak.constants import L_MAX, H, QUANTILES
from mayak import baselines as BL

Q = np.array(QUANTILES, np.float32)
SEASON = ["DJF", "MAM", "JJA", "SON"]
FINE_LEADS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168)
LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]


def _season(doy):
    return SEASON[int(((doy % 365.24) // 91.31)) % 4]


def _bin(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1


class EvalSet(Dataset):
    def __init__(self, clims, station_splits=("train", "unseen_test"),
                 manifest="data/manifest.csv", time_key="test",
                 every_hours=72, L=None, max_windows=6000):
        import csv, os
        from mayak.data.splits import time_bounds
        split_of = {}
        with open(manifest) as f:
            for r in csv.DictReader(f):
                split_of[r["id"]] = r["split"]
        self.items = []
        for sid, s in clims.items():
            sp = split_of.get(sid)
            if sp not in station_splits:
                continue
            lo, hi = time_bounds(s["N"])[time_key]
            for t in range(lo + 1, hi - H, every_hours):
                self.items.append((sid, t, "seen" if sp == "train" else "unseen"))
        if len(self.items) > max_windows:
            self.items = self.items[:: len(self.items) // max_windows][:max_windows]
        self.clims = clims; self.L = L

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, t, seen = self.items[i]
        s = self.clims[sid]; clim = s["clim"]; t0d, t0h = s["t0d"], s["t0h"]
        if self.L is None:
            L = min(t, L_MAX)
        elif self.L == 0:
            L = 0
        else:
            L = min(self.L, t, L_MAX)
        k = np.arange(L_MAX); abs_h = t - L_MAX + k
        doy_h = (t0d + (t0h + abs_h) / 24.0) % 365.24
        hour_h = (t0h + abs_h) % 24.0
        x_hist = np.zeros((L_MAX, 3), np.float32); mask_hist = np.zeros((L_MAX, 3), np.float32)
        if L > 0:
            src = np.arange(t - L, t)
            x_hist[L_MAX - L:] = s["x"][src]; mask_hist[L_MAX - L:] = s["mask"][src]
        fut = np.arange(t, t + H)
        doy_f = (t0d + (t0h + fut) / 24.0) % 365.24
        hour_f = (t0h + fut) % 24.0
        y = s["x"][fut, 0].astype(np.float32)
        mu_clim_fut = clim.predict(doy_f, hour_f).astype(np.float32)
        kh = np.arange(max(t - 24, 0), t); mh = s["mask"][kh, 0]
        if mh.sum() >= 1:
            dh = (t0d + (t0h + kh) / 24.0) % 365.24; hh = (t0h + kh) % 24.0
            a_recent = float(((s["x"][kh, 0] - clim.predict(dh, hh)) * mh).sum() / mh.sum())
        else:
            a_recent = 0.0
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
            "mu_clim_fut": torch.from_numpy(mu_clim_fut),
            "sigma_clim": torch.tensor(clim.sigma, dtype=torch.float32),
            "a_recent": torch.tensor(a_recent, dtype=torch.float32),
            "koppen_id": torch.tensor(hash(s["koppen"]) % 100, dtype=torch.long),
            "season_id": torch.tensor(_season(float(doy_f[0])).__hash__() % 4, dtype=torch.long),
            "seen": torch.tensor(0 if seen == "seen" else 1, dtype=torch.long),
        }


def pinball_crps(y, q):
    err = y[..., None] - q
    pin = np.maximum(Q * err, (Q - 1) * err)
    return 2.0 * pin.mean(axis=-1)


def metric_table(y, mu, q, mse_clim_lead, leads=(1, 3, 6, 12, 24, 48, 72, 120, 168)):
    out = {}
    crps = pinball_crps(y, q)
    for h in leads:
        j = h - 1
        e = mu[:, j] - y[:, j]
        mae = np.abs(e).mean(); rmse = np.sqrt((e ** 2).mean())
        skill = 1.0 - (e ** 2).mean() / max(mse_clim_lead[j], 1e-9)
        lo80, hi80 = q[:, j, 1], q[:, j, 5]
        lo90, hi90 = q[:, j, 0], q[:, j, 6]
        picp80 = ((y[:, j] >= lo80) & (y[:, j] <= hi80)).mean()
        picp90 = ((y[:, j] >= lo90) & (y[:, j] <= hi90)).mean()
        a = 0.10
        width = hi90 - lo90
        wink = (width
                + (2 / a) * (lo90 - y[:, j]) * (y[:, j] < lo90)
                + (2 / a) * (y[:, j] - hi90) * (y[:, j] > hi90)).mean()
        out[h] = dict(MAE=float(mae), RMSE=float(rmse), Skill=float(skill),
                      CRPS=float(crps[:, j].mean()), PICP80=float(picp80),
                      PICP90=float(picp90), Winkler90=float(wink))
    return out


@torch.no_grad()
def gather(model, dataset, device="cpu", batch_size=128):
    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch_size)
    ys, mus, qs, mucl, seen = [], [], [], [], []
    a_rec, sig_cl, xh, mh = [], [], [], []
    for b in dl:
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        o = model(bb)
        ys.append(b["y"].numpy()); mus.append(o["mu"].cpu().numpy()); qs.append(o["q"].cpu().numpy())
        mucl.append(b["mu_clim_fut"].numpy()); seen.append(b["seen"].numpy())
        a_rec.append(b["a_recent"].numpy()); sig_cl.append(b["sigma_clim"].numpy())
        xh.append(b["x_hist"].numpy()); mh.append(b["mask_hist"].numpy())
    cat = lambda L: np.concatenate(L, 0)
    return dict(y=cat(ys), mu=cat(mus), q=cat(qs), mu_clim=cat(mucl), seen=cat(seen),
                a_recent=cat(a_rec), sigma_clim=cat(sig_cl), x_hist=cat(xh), mask_hist=cat(mh))


def mse_clim_per_lead(y, mu_clim):
    return ((mu_clim - y) ** 2).mean(axis=0)


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


def show(tbl):
    print(f"{'лид,ч':>6} {'MAE':>6} {'RMSE':>6} {'Skill':>7} {'CRPS':>6} "
          f"{'PICP80':>7} {'PICP90':>7} {'Wink90':>7}")
    for h, m in tbl.items():
        print(f"{h:>6} {m['MAE']:>6.2f} {m['RMSE']:>6.2f} {m['Skill']:>+7.1%} "
              f"{m['CRPS']:>6.2f} {m['PICP80']:>7.1%} {m['PICP90']:>7.1%} {m['Winkler90']:>7.2f}")


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
    cols = 2; rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(idx):
        item = ds[int(i)]
        batch = {key: (v[None] if torch.is_tensor(v) else v) for key, v in item.items()}
        out = model(batch)
        mu = out["mu"][0].cpu().numpy(); q = out["q"][0].cpu().numpy()
        if shift is not None:
            q = apply_conformal(q, shift); mu = q[:, 3]
        y = item["y"].numpy(); muc = item["mu_clim_fut"].numpy()
        sid, _t, seen = ds.items[int(i)]; zone = ds.clims[sid]["koppen"]
        ax = axes[k]
        ax.fill_between(leads, q[:, 0], q[:, 6], alpha=0.2, color="tab:blue", label="90%-интервал")
        ax.plot(leads, y, color="black", lw=1.6, label="факт")
        ax.plot(leads, mu, color="tab:blue", lw=1.5, label="МАЯК (медиана)")
        ax.plot(leads, muc, color="tab:red", lw=1.0, ls="--", label="климатология")
        ax.set_title(f"{sid} · зона {zone} · {seen}", fontsize=9)
        ax.set_xlabel("лид, ч"); ax.set_ylabel("T, °C"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle("Прогноз МАЯК vs факт (примеры)", y=1.0, fontsize=12)
    fig.tight_layout()

    suff = "" if L is None else f"_L{L}"
    fig.suptitle(f"Прогноз МАЯК vs факт (примеры{', L='+str(L) if L is not None else ''})",
                 y=1.0, fontsize=12)
    p = os.path.join(out_dir, f"forecast_examples{suff}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("Сохранено:", p)
    return p


@torch.no_grad()
def _gather_full(model, ds, device="cpu", batch_size=128):
    """Как gather, но дополнительно тащит o, r, e, sigma_c"""
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


def coldstart_curve(model, clims, manifest="data/manifest.csv",
                    Ls=(0, 6, 24, 72, 168, 336, 672)):
    print("\n=== Кривая холодного старта (Skill-24ч от L) ===")
    res = {}
    for L in Ls:
        ds = EvalSet(clims, manifest=manifest, time_key="test", L=L, every_hours=120)
        D = gather(model, ds)
        j = 24 - 1
        mse_clim = ((D["mu_clim"][:, j] - D["y"][:, j]) ** 2).mean()
        e = D["mu"][:, j] - D["y"][:, j]
        sk = 1 - (e ** 2).mean() / max(mse_clim, 1e-9)
        res[L] = float(sk)
        print(f"  L={L:>4} ч : Skill-24ч = {sk:+.1%}")
    if res.get(672, 0) > 0:
        print(f"  Доля при L=24ч от полного: {res.get(24,0)/res[672]:.0%}  (цель ≥80%)")
    return res


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


def evaluate_all(model, clims, manifest="data/manifest.csv", r_damped=None,
                 named_extra=None, ds=None, preds=None, aux=None):
    if ds is None:
        ds = EvalSet(clims, manifest=manifest, time_key="test")
    if preds is None or aux is None:
        named = {"МАЯК": model}
        if named_extra:
            named.update(named_extra)
        preds, aux = collect_predictions(named, ds)
        preds = add_statistical_baselines(preds, aux, r_damped=r_damped)

    y = aux["y"]
    mse_clim = mse_clim_per_lead(y, aux["mu_clim"])
    for name, p in preds.items():
        print(f"\n=== {name} ===")
        show(metric_table(y, p["mu"], p["q"], mse_clim))

    j = 24 - 1
    mu_mayak = preds["МАЯК"]["mu"]
    for tag, mask in [("seen", aux["seen"] == 0), ("unseen", aux["seen"] == 1)]:
        if mask.sum() == 0:
            continue
        e = mu_mayak[mask, j] - y[mask, j]
        sk = 1 - (e ** 2).mean() / max(mse_clim[j], 1e-9)
        print(f"Skill-24ч [{tag}]: {sk:+.1%}  (окон: {int(mask.sum())})")
    return preds, aux, ds


def stage_a_field_check(model, clims, manifest="data/manifest.csv",
                        station_split="unseen_val", time_key="test"):
    """Критерий этапа A"""
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest,
                 time_key=time_key, L=0)
    D = gather(model, ds)
    y, mu, muc = D["y"], D["mu"], D["mu_clim"]
    mse_field = float(((mu - y) ** 2).mean())
    mse_clim = float(((muc - y) ** 2).mean())
    ratio = mse_field / max(mse_clim, 1e-9)
    bias = float(np.abs(mu - muc).mean())
    print(f"\n[Проверка этапа A] поле на {station_split} (L=0, окон: {len(ds)}):")
    print(f"  MSE поля         = {mse_field:7.3f}")
    print(f"  MSE климатологии = {mse_clim:7.3f}   ← эталон")
    print(f"  отношение        = {ratio:7.3f}   ← цель ≤ 1.05")
    print(f"  |поле − климат|  = {bias:7.3f} °C ← цель → 0")
    return dict(mse_field=mse_field, mse_clim=mse_clim, ratio=ratio, bias=bias)


@torch.no_grad()
def pure_field_check(model, clims, manifest="data/manifest.csv",
                     station_split="unseen_val", time_key="test"):
    """Чистое поле: field.coefficients(loc) БЕЗ паспорта (z=None) и БЕЗ r-головы."""
    from mayak.astro import astro_features
    from torch.utils.data import DataLoader
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest,
                 time_key=time_key, L=0)
    model.eval()
    e2 = ec = bias = 0.0; n = 0
    for b in DataLoader(ds, batch_size=128):
        lat, lon, elev = b["lat"], b["lon"], b["elev"]
        loc = model.loc(lat, lon, elev)
        astro_f = astro_features(b["doy_fut"], b["hour_fut"], lat[:, None], lon[:, None])
        coefs = model.field.coefficients(loc)
        mu_c, _, _ = model.field.evaluate(coefs, astro_f)
        y, muc = b["y"], b["mu_clim_fut"]
        e2 += float(((mu_c - y) ** 2).sum()); ec += float(((muc - y) ** 2).sum())
        bias += float((mu_c - muc).abs().sum()); n += y.numel()
    print(f"ЧИСТОЕ поле на {station_split}: ratio={e2/max(ec,1e-9):.3f}, "
          f"|поле−клим|={bias/n:.3f}°C  (окон: {len(ds)})")
    

@torch.no_grad()
def l0_decompose(model, clims, manifest="data/manifest.csv", station_split="train"):
    from torch.utils.data import DataLoader
    ds = EvalSet(clims, station_splits=(station_split,), manifest=manifest, time_key="test", L=0)
    model.eval()
    Z=[]; sr=oo=ee=0.0; n=0
    for b in DataLoader(ds, batch_size=128):
        out = model(b)
        Z.append(out["z"])
        sr += float((out["sigma_c"]*out["r"]).abs().sum())
        oo += float(out["o"].abs().sum())
        ee += float(out["e"].abs().sum())
        n  += out["mu"].numel()
    Z = torch.cat(Z, 0)
    print(f"[{station_split}] std(z) по станциям = {float(Z.std(0).mean()):.3f}  "
          f"(≈0 → прайор глобальный; >0 → прайор зависит от loc = меморизатор)")
    print(f"        |σ·r| = {sr/n:.3f}°C  (≈0 → r заглушён; >0 → r ещё активен и фитит)")
    print(f"        |o| = {oo/n:.3f}   e = {ee/Z.numel()*Z.shape[1]:.3f}  (ждём ≈0 при L=0)")


def diurnal_amplitude(series):
    H = series.shape[-1]; t = np.arange(H)
    c = np.cos(2 * np.pi * t / 24.0); s = np.sin(2 * np.pi * t / 24.0)
    a = (series * c).mean(-1); b = (series * s).mean(-1)
    return 2.0 * np.sqrt(a ** 2 + b ** 2)


def plot_amplitude_scatter(model, clims, manifest="data/manifest.csv", out_dir="runs/plots",
                           time_key="test", max_points=4000, seed=0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    D = gather(model, EvalSet(clims, manifest=manifest, time_key=time_key))
    ap = diurnal_amplitude(D["mu"]); ar = diurnal_amplitude(D["y"])
    idx = np.random.default_rng(seed).choice(len(ap), min(max_points, len(ap)), replace=False)
    ap, ar = ap[idx], ar[idx]
    slope = float(np.polyfit(ar, ap, 1)[0]); bias = float((ap - ar).mean())
    lim = float(max(ar.max(), ap.max())) * 1.05
    fig, axx = plt.subplots(figsize=(6, 6))
    axx.scatter(ar, ap, s=6, alpha=0.3)
    axx.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 (идеал)")
    axx.set_xlim(0, lim); axx.set_ylim(0, lim)
    axx.set_xlabel("факт: суточная амплитуда, °C")
    axx.set_ylabel("прогноз: суточная амплитуда, °C")
    axx.set_title(f"Суточная амплитуда\nнаклон={slope:.2f}, смещение={bias:+.2f}°C")
    axx.legend(); axx.grid(alpha=0.3)
    p = os.path.join(out_dir, "amplitude_scatter.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"Сохранено: {p}  (наклон {slope:.2f} — <1 значит модель ЗАНИЖАЕТ суточный ход)")
    return p


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
    ap.add_argument("--n-examples", type=int, default=10, help="число примеров прогноз vs факт")
    args = ap.parse_args()

    clims = BL.fit_climatologies(args.manifest)
    r = BL.fit_damped_persistence(clims, n_windows=20000)

    mayak = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu").model
    named_extra = {}
    if args.gru_ckpt:
        named_extra["GRU seq2seq"] = LitBaseline.load_from_checkpoint(
            args.gru_ckpt, map_location="cpu").model
    if args.dlinear_ckpt:
        named_extra["DLinear"] = LitBaseline.load_from_checkpoint(
            args.dlinear_ckpt, map_location="cpu").model

    ds = EvalSet(clims, manifest=args.manifest, time_key="test")
    named_all = {"МАЯК": mayak, **named_extra}
    preds, aux = collect_predictions(named_all, ds)
    preds = add_statistical_baselines(preds, aux, r_damped=r)

    print("\n=== Таблицы метрик ===")
    evaluate_all(mayak, clims, args.manifest, r_damped=r,
                 named_extra=named_extra, ds=ds, preds=preds, aux=aux)

    coldstart_curve(mayak, clims, args.manifest)

    tables = build_tables(preds, aux)
    print("\n=== Графики по метрикам ===")
    for p in plot_metric_curves(tables, args.out_dir):
        print("  ", p)

    print("\n=== Разрез по зонам Кёппена (МАЯК) ===")
    print_zone_breakdown(zone_breakdown(preds, aux, koppen_per_window(ds)))

    shift = np.load(args.conformal) if args.conformal else None
    print("\n=== Холодный старт L=0 (пункт 3) ===")
    coldstart_L0_check(mayak, clims, args.manifest, shift=shift)

    if shift is not None:
        print("\n=== Влияние конформной калибровки (пункт 4) ===")
        calibration_quality_report(mayak, clims, shift, args.manifest)

    print("\n=== Графики прогноз vs факт (примеры МАЯК) ===")
    plot_forecast_examples(mayak, clims, manifest=args.manifest,
                           n=args.n_examples, out_dir=args.out_dir, shift=shift)
    plot_forecast_examples(mayak, clims, manifest=args.manifest, n=args.n_examples,
                           out_dir=args.out_dir + "/unseen",
                           station_splits=("unseen_test",), shift=shift)
    plot_forecast_examples(mayak, clims, n=args.n_examples, L=0, out_dir="runs/plots")
    plot_forecast_examples(mayak, clims, n=args.n_examples, L=0, station_splits=("unseen_test",),
                       out_dir=args.out_dir + "/unseen")
    print("\n=== Суточные амплитуды ===")
    plot_amplitude_scatter(mayak, clims, manifest=args.manifest, out_dir=args.out_dir)


if __name__ == "__main__":
    main()