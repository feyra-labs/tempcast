"""Knockout-абляции обученной модели на инференсе (без переобучения).

Диагностика, а не замена переобучению: флаги абляций в ModelConfig
обучают модель без компонента, а здесь компонент выключается у уже обученной.
Группы мод и индексы солнечных каналов берутся из конфига модели.
"""
import contextlib
import torch

from mayak.config import SOLAR_CHANNELS
from mayak.evaluate import EvalSet, gather
from mayak.metrics import skill_per_lead

BASE_VARIANTS = ["none", "no_r", "no_sun", "no_compression", "no_passport"]


def group_slices(model):
    """{имя группы: срез мод} по конфигу модели."""
    sl, i = {}, 0
    for name, size in zip(model.cfg.group_names, model.cfg.group_sizes):
        sl[name] = slice(i, i + size)
        i += size
    return sl


def variants_for(model):
    """Варианты knockout для модели: базовые + по одному на каждую группу мод."""
    groups = [f"no_{g}" for g in group_slices(model)] if model.cfg.n_groups > 1 else []
    return BASE_VARIANTS[:3] + groups + BASE_VARIANTS[3:]


@contextlib.contextmanager
def knockout(model, which):
    """Контекст, в котором компонент 'which' выключен.
       none           — ничего (база)
       no_r           — r=0 (выключена нелинейная поправка голов)
       no_passport    — z=0 (паспорт молчит)
       no_compression — κ=0 (доказательное сжатие выключено)
       no_sun         — солнечные каналы энкодера и ковариаты голов занулены
       no_<группа>    — занулены амплитуды соответствующей группы мод (группы из конфига)"""
    handles = []
    sl = group_slices(model)
    prev_compression = model.readout.compression
    try:
        if which == "no_compression":
            model.readout.compression = False

        if which == "no_passport":
            handles.append(model.passport.register_forward_hook(
                lambda m, i, o: (torch.zeros_like(o[0]), o[1])))

        if which == "no_r":
            handles.append(model.heads.register_forward_hook(
                lambda m, i, o: (torch.zeros_like(o[0]), o[1], o[2])))

        if which.startswith("no_") and which[3:] in sl:
            s = sl[which[3:]]

            def h_ro(m, i, o):
                a_re, a_im, e = o
                a_re = a_re.clone()
                a_im = a_im.clone()
                a_re[:, s] = 0.0
                a_im[:, s] = 0.0
                return a_re, a_im, e

            handles.append(model.readout.register_forward_hook(h_ro))

        if which == "no_sun":
            idx = model.channel_indices(SOLAR_CHANNELS)

            def pre_enc(m, args):
                (ch,) = args
                ch = ch.clone()
                ch[:, idx] = 0.0
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
        model.readout.compression = prev_compression


def knockout_table(model, clims, manifest="data/manifest.csv", time_key="test",
                   leads=(6, 24, 72, 168), L=None,
                   station_splits=("train", "unseen_test"), variants=None):
    variants = variants_for(model) if variants is None else variants
    kw = {} if L is None else {"L": L}
    ds = EvalSet(clims, station_splits=station_splits, manifest=manifest,
                 time_key=time_key, **kw)
    rows = {}
    for v in variants:
        with knockout(model, v):
            D = gather(model, ds)
        sk = skill_per_lead(D["y"], D["mu"], D["mu_clim"], D["y_mask"])
        rows[v] = {h: float(sk[h - 1]) for h in leads}

    full = rows["none"]
    tag = "" if L is None else f"  (L={L})"
    print(f"\n=== Knockout-абляции{tag} ===")
    hdr = f"{'выключено':>16}" + "".join(f"{'Sk@' + str(h):>9}" for h in leads) \
          + "  |  " + "".join(f"{'Δ@' + str(h):>9}" for h in leads)
    print(hdr)
    for v in variants:
        sk = "".join(f"{rows[v][h]:>+9.1%}" for h in leads)
        dr = "" if v == "none" else "".join(f"{full[h] - rows[v][h]:>+9.1%}" for h in leads)
        print(f"{v:>16}{sk}  |  {dr}")
    return rows


def main():
    import argparse
    from mayak.lit import load_model
    ap = argparse.ArgumentParser(description="knockout-абляции обученной модели")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()
    from mayak.data.store import get_store
    from mayak.leakage import run_checklist
    store = get_store(args.manifest)
    clims = store.clims()
    run_checklist(store, datasets=[EvalSet(clims, manifest=args.manifest, time_key="test")],
                  checkpoints=[args.ckpt])
    m = load_model(args.ckpt)
    knockout_table(m, clims, args.manifest)
    knockout_table(m, clims, args.manifest, L=0,
                   variants=["none", "no_passport", "no_sun"])
    print("\n>>> Только unseen-станции:")
    knockout_table(m, clims, args.manifest, station_splits=("unseen_test",))


if __name__ == "__main__":
    main()
