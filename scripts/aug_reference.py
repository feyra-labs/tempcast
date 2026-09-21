"""Эталонный набор окон с аугментациями.

Строит чистое окно и то же окно с каждым видом аугментации отдельно
(mayak.data.augment.REFERENCE_CASES), прогоняет QC окна и записывает:

* <out>/aug_reference.npz  - x, маска и коды QC до и после для каждого случая;
* <out>/aug_reference.json - таблица «случай → ожидаемые коды, новые коды,
  попадание, доля побочных кодов».

    python scripts/aug_reference.py --out runs/aug_reference
"""
import argparse
import json
import os
import sys

import numpy as np


def build(out, seed=0):
    from mayak.data.augment import EXPECTED_QC, qc_effect, reference_windows, window_codes
    from mayak.data.masking import enforce_invariant
    os.makedirs(out, exist_ok=True)
    arrays, table = {}, []
    for case, name, before, after, ch, rows in reference_windows(seed):
        eff = qc_effect(name, before, after, ch, rows)
        for tag, w in (("before", before), ("after", after)):
            x, m = enforce_invariant(w.x, w.m)
            arrays[f"{case}/{tag}/x"] = x
            arrays[f"{case}/{tag}/mask"] = m
            arrays[f"{case}/{tag}/codes"] = window_codes(w)
        table.append(dict(case=case, augmentation=name, params=after.applied[name],
                          channel=ch, rows=rows,
                          expected=sorted(c.name for c in EXPECTED_QC[name]), **eff))
    np.savez_compressed(os.path.join(out, "aug_reference.npz"), **arrays)
    with open(os.path.join(out, "aug_reference.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seed=seed, cases=table), f, ensure_ascii=False, indent=1)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/aug_reference", help="каталог результата")
    ap.add_argument("--seed", type=int, default=0, help="сид чистого окна")
    a = ap.parse_args(argv)
    table = build(a.out, a.seed)
    bad = 0
    for r in table:
        ok = (r["hit"] is not False) and r["side_frac"] < 0.002
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {r['case']:15s} ожидается {r['expected'] or '-'}, "
              f"новые {r['new'] or '-'}")
    print(f"записано в {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
