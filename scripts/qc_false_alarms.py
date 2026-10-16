"""Доля часов, которые причинный QC бракует на рядах набора.

На реанализе артефактов нет, поэтому каждая отметка, кроме пропуска, - ложное
срабатывание. Ряды записаны так, как их пишет прибор: температура и влажность целыми
числами, давление в десятых. Прибор отчитывается раз в час, поэтому ряды проверяются
почасовыми, без прореживания. Скрипт печатает доли по каналам и кодам, худшие станции и
пишет всё в JSON. По этим числам подбираются пороги QC.
"""
import argparse
import json
import os

import numpy as np

from mayak.data.qc import CHANNELS, DEFAULT_QC, QC_CODES, QCCode, causal_codes
from mayak.data.recording import record_values
from mayak.data.store import get_store


def station_rates(raw, present, elev):
    """Доли часов с каждым кодом среди часов, где значение есть.

    Args:
        raw: значения, записанные прибором, форма (N, 3).
        present: маска наличия, форма (N, 3).
        elev: высота станции для проверки давления, м.

    Returns:
        Пара: словарь долей по каналу и коду и число часов со значением по каналам.
    """
    x = record_values(raw)
    p = np.array(present, np.uint8)
    codes = causal_codes(x, p, elev=elev)
    have = p > 0
    out = {}
    for j, ch in enumerate(CHANNELS):
        n = int(have[:, j].sum())
        bad = codes[have[:, j], j]
        out[f"{ch}/any"] = float((bad != 0).mean()) if n else 0.0
        for c in QC_CODES:
            if c != QCCode.MISSING:
                out[f"{ch}/{c.name}"] = float(((bad & c) > 0).mean()) if n else 0.0
    return out, have.sum(0).tolist()


def main():
    ap = argparse.ArgumentParser(
        description="ложные срабатывания причинного QC",
        epilog="пример: python scripts/qc_false_alarms.py --manifest data/manifest.csv "
               "--out runs/qc_false_alarms.json")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--out", default="runs/qc_false_alarms.json")
    args = ap.parse_args()
    store = get_store(args.manifest)
    sids = sorted(store.stations)[:args.max_stations]
    per, weights = {}, {}
    for sid in sids:
        s = store.stations[sid]
        elev = s["dem_elev"] if s.get("dem_elev") is not None else s["elev"]
        per[sid], weights[sid] = station_rates(s["raw"], s["present"], elev)
    keys = next(iter(per.values())).keys()
    pooled = {}
    for k in keys:
        j = CHANNELS.index(k.split("/")[0])
        w = np.array([weights[sid][j] for sid in sids], np.float64)
        v = np.array([per[sid][k] for sid in sids])
        pooled[k] = float((v * w).sum() / max(w.sum(), 1.0))
    worst = {ch: sorted(sids, key=lambda sid: -per[sid][f"{ch}/any"])[:5] for ch in CHANNELS}
    result = dict(config=DEFAULT_QC.to_dict(), lookback_hours=DEFAULT_QC.lookback_hours,
                  pooled=pooled,
                  worst={ch: {sid: per[sid][f"{ch}/any"] for sid in ids}
                         for ch, ids in worst.items()},
                  stations=per)
    print(f"\nстанций {len(sids)}: доля забракованных часов, %")
    for ch in CHANNELS:
        parts = ", ".join(f"{c.name} {100 * pooled[f'{ch}/{c.name}']:.3f}"
                          for c in QC_CODES if c != QCCode.MISSING
                          and pooled[f"{ch}/{c.name}"] > 0)
        print(f"  {ch:2s}: всего {100 * pooled[f'{ch}/any']:.3f}  ({parts or 'нет'}); "
              f"худшая станция {worst[ch][0]} {100 * per[worst[ch][0]][f'{ch}/any']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\nЗаписано:", args.out)


if __name__ == "__main__":
    main()
