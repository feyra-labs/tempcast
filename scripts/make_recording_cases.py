"""Эталонные случаи записи значений прибором.

Значения ровно посередине между соседними значениями сетки, в том числе отрицательные,
значения рядом с серединой и значения у границ физических диапазонов. Для каждого
значения записан результат правила записи. По этому файлу сверяются Python и порт на
Rust. Запускать осознанно: только когда меняется правило записи.
"""
import argparse
import json
import os

import numpy as np

from mayak.data.recording import record_channel

FORMAT = 1
CHANNELS = ("T", "P", "RH")
INPUTS = {
    "T": [-90.5, -89.5, -3.5, -2.5, -1.5, -0.5, -0.4, -0.0, 0.0, 0.4, 0.5, 1.5, 2.5,
          3.5, 12.49, 12.51, 59.5, 60.5, 0.49999997, 0.50000006, -0.50000006, 21.7, -7.3],
    "P": [299.95, 300.05, 1013.25, 1013.35, 1013.15, 999.95, 1000.05, 1100.05, 1100.15,
          850.45, 850.55, 701.25, 701.75, 0.25, 0.35],
    "RH": [-0.5, 0.5, 1.5, 2.5, 55.5, 56.5, 98.5, 99.5, 100.5, 101.5, 42.4, 42.6],
}


def cases():
    """Случаи по каналам: вход и результат, оба как float32.

    Returns:
        Список словарей с полями channel, x, recorded.
    """
    out = []
    for j, name in enumerate(CHANNELS):
        x = np.asarray(INPUTS[name], np.float32)
        rec = record_channel(x, j)
        out += [dict(channel=name, x=float(a), recorded=float(b)) for a, b in zip(x, rec)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="эталонные случаи записи значений прибором",
        epilog="запуск: python scripts/make_recording_cases.py; после пересоздания - "
               "cargo test --release --manifest-path runtime-rs/Cargo.toml")
    ap.add_argument("--out", default=os.path.join("tests", "data", "recording", "cases.json"))
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dict(format=FORMAT, cases=cases()), fh, ensure_ascii=False, indent=1)
    print("Записано:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
