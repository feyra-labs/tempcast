"""Пересоздание эталонных векторов рантайма: tests/data/runtime_golden.

    python scripts/make_runtime_golden.py [--out tests/data/runtime_golden]

Запускать осознанно - только когда меняется поведение эталона (модель, потоковый
рантайм, калибровка, календарь, формат состояния). Тест
tests/test_block14_runtime.py::test_golden_is_fresh падает, если эталон устарел.
После пересоздания: cargo test --release в runtime-rs и коммит каталога целиком.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="эталонные векторы компилируемого рантайма")
    ap.add_argument("--out", default="tests/data/runtime_golden")
    args = ap.parse_args()
    from mayak.runtime.golden import generate
    doc = generate(args.out)
    for sc in doc["scenarios"]:
        n_fc = sum(ev["op"] == "forecast" for ev in sc["events"])
        print(f"  {sc['name']:9s} событий {len(sc['events']):4d}, выпусков {n_fc:3d}, "
              f"итог {sc['final']}")
    print(f"  мин. запас решения ACI: {doc['aci_margin_min']:.2e}")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
