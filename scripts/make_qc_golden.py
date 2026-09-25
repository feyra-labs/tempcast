"""Эталонные векторы кодов причинного QC.

Каждый случай - короткий ряд T, P, RH с пропусками и артефактами: выбросы, скачки,
возврат к прежнему уровню, залипание, насыщение влажности, градусы Фаренгейта,
давление на уровне моря, отчёты раз в 2 и 3 часа. Температура и влажность записаны
целыми числами, давление - десятыми. Коды каждого часа посчитаны пакетным причинным QC.
По этому файлу сверяются пакетный QC и поток на Python и порт на Rust.

Запускать осознанно: только когда меняются правила или пороги QC.
"""
import argparse
import json
import math
import os

import numpy as np

from mayak.data.qc import DEFAULT_QC, PHYS, causal_codes, station_pressure_expected

FORMAT = 1


def _base(n, seed, amp, t0, syn_sd, rh_mean, elev):
    """Правдоподобный почасовой ряд с суточным ходом и синоптикой."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    ph = 2 * np.pi * ((k % 24) - 15) / 24
    d = amp * (np.cos(ph) + 0.3 * np.cos(2 * ph + 0.8)) / 1.3
    rho = math.exp(-1 / 60)
    e = rng.standard_normal(n) * syn_sd * math.sqrt(1 - rho ** 2)
    syn = np.zeros(n)
    for i in range(1, n):
        syn[i] = rho * syn[i - 1] + e[i]
    T = t0 + d + syn + 0.3 * rng.standard_normal(n)
    P = station_pressure_expected(elev) + syn + 0.15 * rng.standard_normal(n)
    RH = np.clip(rh_mean - 2.5 * d + 3 * rng.standard_normal(n), 3, 100)
    return np.stack([T, P, RH], -1), np.ones((n, 3), bool), rng


def _record(x, present):
    """Запись целых T и RH и давления в десятых, как у прибора."""
    x = x.copy()
    x[:, 0] = np.round(x[:, 0])
    x[:, 2] = np.round(x[:, 2])
    x[:, 1] = np.round(x[:, 1] * 10) / 10
    return x.astype(np.float32), present


def cases():
    out = []

    x, p, rng = _base(600, 1, 9.0, 22.0, 2.0, 30, 200.0)
    x[200, 0] += 25
    x[300:303, 0] += 18
    x[410, 1] -= 30
    x[450, 2] += 60
    x[500:, 0] += 14
    p[rng.random(p.shape) < 0.04] = False
    out.append(("spikes_jumps", 200.0, *_record(x, p)))

    x, p, rng = _base(600, 2, 5.0, 8.0, 3.0, 70, 200.0)
    x[100:130, 0], x[100:130, 2] = x[100, 0], x[100, 2]
    x[200:290, 0] = x[200, 0]
    x[320:360, 1] = x[320, 1]
    x[400:490, 2] = 100.0
    out.append(("stuck", 200.0, *_record(x, p)))

    x, p, rng = _base(700, 3, 5.0, 4.0, 2.0, 70, 200.0)
    x[450:520, 0] = x[450:520, 0] * 1.8 + 32.0
    x[600:640, 0] = x[600:640, 0] * 1.8 + 32.0 + 20
    out.append(("fahrenheit", 200.0, *_record(x, p)))

    x, p, rng = _base(500, 4, 4.0, 2.0, 2.0, 70, 1600.0)
    x[250:, 1] += 1013.25 - station_pressure_expected(1600.0)
    x[100, 1] = 1200.0
    x[120, 0] = 75.0
    out.append(("sea_level_pressure", 1600.0, *_record(x, p)))

    for step, seed in ((2, 5), (3, 6)):
        x, p, rng = _base(600, seed, 7.0, 15.0, 3.0, 55, 200.0)
        keep = np.zeros(len(x), bool)
        keep[::step] = True
        p &= keep[:, None]
        p[rng.random(p.shape) < 0.05] = False
        x[301 - 301 % step, 0] += 22
        x[400:490, 0] = x[400, 0]
        x[400:490, 2] = x[400, 2]
        out.append((f"sparse_{step}h", 200.0, *_record(x, p)))

    x, p, rng = _base(400, 7, 2.0, -20.0, 5.0, 80, 0.0)
    x[50:60] = np.nan
    x[70, 2] = -5.0
    x[90:] = x[90:] + rng.standard_normal((310, 3)) * [4, 3, 10]
    out.append(("no_elevation_noisy", None, *_record(x, p)))
    return out


def generate(path):
    doc = dict(format=FORMAT, config=dict(DEFAULT_QC.to_dict(),
                                          lookback_hours=DEFAULT_QC.lookback_hours),
               phys={c: list(v) for c, v in PHYS.items()}, cases=[])
    for name, elev, x, present in cases():
        present = present & np.isfinite(x)
        codes = causal_codes(x, present, elev=elev)
        vals = [[float(x[i, j]) if present[i, j] else None for j in range(3)]
                for i in range(len(x))]
        doc["cases"].append(dict(name=name, elev=elev, x=vals, codes=codes.tolist()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    return doc


def main():
    ap = argparse.ArgumentParser(
        description="эталонные векторы кодов причинного QC",
        epilog="запуск: python scripts/make_qc_golden.py; после пересоздания - "
               "cargo test --release --manifest-path runtime-rs/Cargo.toml")
    ap.add_argument("--out", default="tests/data/qc_causal/golden.json")
    args = ap.parse_args()
    doc = generate(args.out)
    for c in doc["cases"]:
        codes = np.array(c["codes"], np.uint8)
        print(f"  {c['name']:20s} часов {len(codes):4d}, с кодами кроме пропуска "
              f"{int(((codes & 0xFE) > 0).any(1).sum()):4d}")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
