"""Пересоздание эталонных сценариев хоста устройства: tests/data/runtime_golden.

    python scripts/make_runtime_golden.py [--out tests/data/runtime_golden]

Запускать осознанно - только когда меняется поведение эталона: модель, потоковый
рантайм, хост, калибровка, календарь, формат состояния. Тест свежести эталона падает,
если эталон устарел. После пересоздания: cargo test --release в runtime-rs и коммит
каталога целиком.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="эталонные сценарии хоста устройства")
    ap.add_argument("--out", default="tests/data/runtime_golden")
    args = ap.parse_args()
    from mayak.runtime.golden import generate
    doc = generate(args.out)
    for sc in doc["scenarios"]:
        ops = [ev["op"] if ev["op"] != "cmd" else ev["line"].split()[0] for ev in sc["events"]]
        n = {k: ops.count(k) for k in ("obs", "forecast", "restart", "status", "state")}
        print(f"  {sc['name']:12s} {sc['precision']:4s} наблюдений {n['obs']:5d}, выпусков "
              f"{n['forecast']:4d}, перезапусков {n['restart']:2d}")
    print(f"  мин. запас решения ACI: {doc['aci_margin_min']:.2e}")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
