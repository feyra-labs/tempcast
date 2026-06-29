"""Knockout-абляции обученной модели на инференсе (без переобучения)"""
import contextlib
import torch

from mayak.constants import GROUPS
from mayak.evaluate import EvalSet, gather, mse_clim_per_lead

VARIANTS = ["none", "no_r", "no_sun", "no_D", "no_S", "no_W",
            "no_compression", "no_passport"]


def _group_slices():
    names = ["R", "D", "S", "W"]; sl = {}; i = 0
    for nm, g in zip(names, GROUPS):
        sl[nm] = slice(i, i + g); i += g
    return sl


@contextlib.contextmanager
def knockout(model, which):
    """Контекст, в котором компонент 'which' выключен.
       none           — ничего (база)
       no_r           — r=0 (выключена нелинейная поправка голов)
       no_passport    — z=0 (паспорт молчит)
       no_compression — κ=0 (доказательное сжатие выключено)
       no_sun         — солнечные каналы энкодера и ковариаты голов занулены
       no_R/D/S/W      — занулены амплитуды соответствующей группы мод"""
    handles = []
    sl = _group_slices()
    prev_dc = getattr(model.readout, "disable_compression", False)
    try:
        if which == "no_compression":
            model.readout.disable_compression = True

        if which == "no_passport":
            handles.append(model.passport.register_forward_hook(
                lambda m, i, o: (torch.zeros_like(o[0]), o[1])))

        if which == "no_r":
            handles.append(model.heads.register_forward_hook(
                lambda m, i, o: (torch.zeros_like(o[0]), o[1], o[2])))

        if which in ("no_R", "no_D", "no_S", "no_W"):
            s = sl[which.split("_")[1]]
            def h_ro(m, i, o):
                a_re, a_im, e = o
                a_re = a_re.clone(); a_im = a_im.clone()
                a_re[:, s] = 0.0; a_im[:, s] = 0.0
                return a_re, a_im, e
            handles.append(model.readout.register_forward_hook(h_ro))

        if which == "no_sun":
            def pre_enc(m, args):
                (ch,) = args
                ch = ch.clone(); ch[:, 5:8] = 0.0
                return (ch,)
            handles.append(model.encoder.register_forward_pre_hook(pre_enc))
            def pre_heads(m, args):
                o, Eg, sun, ls, z, e = args
                return (o, Eg, torch.zeros_like(sun), ls, z, e)
            handles.append(model.heads.register_forward_pre_hook(pre_heads))

        yield
    finally:
        for hd in handles:
            hd.remove()
        model.readout.disable_compression = prev_dc


def _skill(mu, y, mse_clim, h):
    j = h - 1
    return float(1.0 - ((mu[:, j] - y[:, j]) ** 2).mean() / max(mse_clim[j], 1e-9))


def knockout_table(model, clims, manifest="data/manifest.csv", time_key="test",
                   leads=(6, 24, 72, 168), L=None,
                   station_splits=("train", "unseen_test"), variants=VARIANTS):
    kw = {} if L is None else {"L": L}
    ds = EvalSet(clims, station_splits=station_splits, manifest=manifest,
                 time_key=time_key, **kw)
    y = None; mse_clim = None; rows = {}
    for v in variants:
        with knockout(model, v):
            D = gather(model, ds)
        if y is None:
            y = D["y"]; mse_clim = mse_clim_per_lead(y, D["mu_clim"])
        rows[v] = {h: _skill(D["mu"], y, mse_clim, h) for h in leads}

    full = rows["none"]
    tag = "" if L is None else f"  (L={L})"
    print(f"\n=== Knockout-абляции{tag} ===")
    hdr = f"{'выключено':>16}" + "".join(f"{'Sk@'+str(h):>9}" for h in leads) \
          + "  |  " + "".join(f"{'Δ@'+str(h):>9}" for h in leads)
    print(hdr)
    for v in variants:
        sk = "".join(f"{rows[v][h]:>+9.1%}" for h in leads)
        dr = "" if v == "none" else "".join(f"{full[h]-rows[v][h]:>+9.1%}" for h in leads)
        print(f"{v:>16}{sk}  |  {dr}")
    return rows


def main():
    import argparse
    from mayak import baselines as BL
    from mayak.lit import LitMayak
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()
    clims = BL.fit_climatologies(args.manifest)
    m = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu").model
    knockout_table(m, clims, args.manifest)
    knockout_table(m, clims, args.manifest, L=0,
                   variants=["none", "no_passport", "no_sun"])
    print("\n>>> Только unseen-станции:")
    knockout_table(m, clims, args.manifest, station_splits=("unseen_test",))


if __name__ == "__main__":
    main()