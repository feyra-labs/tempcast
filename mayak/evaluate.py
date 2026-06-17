import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mayak.constants import L_MAX, H, QUANTILES
from mayak import baselines as BL

Q = np.array(QUANTILES, np.float32)
SEASON = ["DJF", "MAM", "JJA", "SON"]


def _season(doy):
    return SEASON[int(((doy % 365.24) // 91.31)) % 4]


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

def evaluate_all(model, clims, manifest="data/manifest.csv", r_damped=None):
    ds = EvalSet(clims, manifest=manifest, time_key="test")
    D = gather(model, ds)
    y, mucl = D["y"], D["mu_clim"]
    mse_clim = mse_clim_per_lead(y, mucl)

    print("\n=== МАЯК ===")
    show(metric_table(y, D["mu"], D["q"], mse_clim))

    mu_b, q_b = BL.climatology_forecast(mucl, D["sigma_clim"])
    print("\n=== Климатология ===")
    show(metric_table(y, mu_b, q_b, mse_clim))

    if r_damped is not None:
        mu_b, q_b = BL.damped_persistence_forecast(D["a_recent"], mucl, D["sigma_clim"], r_damped)
        print("\n=== Damped persistence ===")
        show(metric_table(y, mu_b, q_b, mse_clim))

    mu_b, q_b = BL.seasonal_naive_forecast(D["x_hist"], D["mask_hist"], D["sigma_clim"], period=24)
    print("\n=== Seasonal-naive 24ч ===")
    show(metric_table(y, mu_b, q_b, mse_clim))

    j = 24 - 1
    for tag, mask in [("seen", D["seen"] == 0), ("unseen", D["seen"] == 1)]:
        if mask.sum() == 0:
            continue
        e = D["mu"][mask, j] - y[mask, j]
        sk = 1 - (e ** 2).mean() / max(mse_clim[j], 1e-9)
        print(f"Skill-24ч [{tag}]: {sk:+.1%}  (окон: {int(mask.sum())})")


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


def show(tbl):
    print(f"{'лид,ч':>6} {'MAE':>6} {'RMSE':>6} {'Skill':>7} {'CRPS':>6} "
          f"{'PICP80':>7} {'PICP90':>7} {'Wink90':>7}")
    for h, m in tbl.items():
        print(f"{h:>6} {m['MAE']:>6.2f} {m['RMSE']:>6.2f} {m['Skill']:>+7.1%} "
              f"{m['CRPS']:>6.2f} {m['PICP80']:>7.1%} {m['PICP90']:>7.1%} {m['Winkler90']:>7.2f}")
        
def compare_ablation(model_full, model_ablated, clims, manifest="data/manifest.csv", lead=24):
    ds = EvalSet(clims, manifest=manifest, time_key="test", every_hours=120)
    out = {}
    for tag, mdl in [("full", model_full), ("ablated", model_ablated)]:
        D = gather(mdl, ds); j = lead - 1
        mse_clim = ((D["mu_clim"][:, j] - D["y"][:, j]) ** 2).mean()
        out[tag] = 1 - ((D["mu"][:, j] - D["y"][:, j]) ** 2).mean() / max(mse_clim, 1e-9)
    print(f"Skill-{lead}ч: full {out['full']:+.1%}  vs  ablated {out['ablated']:+.1%}  "
          f"(падение {out['full']-out['ablated']:+.1%})")
    return out

def main():
    import argparse
    from mayak.lit import LitMayak
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()
    clims = BL.fit_climatologies(args.manifest)
    r = BL.fit_damped_persistence(clims, n_windows=20000)
    lit = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu")
    evaluate_all(lit.model, clims, args.manifest, r_damped=r)
    coldstart_curve(lit.model, clims, args.manifest)


if __name__ == "__main__":
    main()