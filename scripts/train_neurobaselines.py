"""Обучение нейробейзлайнов по тому же протоколу, что и МАЯК.

Эквивалентно `python scripts/train.py --arch <имя>` для каждого имени из --models:
та же функция запуска (mayak.protocol.run_protocol), те же флаги протокола.

Запуск полный:
    python scripts/train_neurobaselines.py --models gru dlinear --accelerator gpu
Отладка:
    python scripts/train_neurobaselines.py --models gru dlinear --steps-a 200 --steps-b 1000 \\
        --batch 32 --windows 2000 --workers 0 --accelerator cpu --precision 32 --val-every 200
"""
import argparse
import logging

from mayak.protocol import ARCH_NAMES, add_protocol_args, protocol_from_args, run_protocol


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["gru", "dlinear"],
                    choices=[a for a in ARCH_NAMES if a != "mayak"])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--out-root", default="runs")
    add_protocol_args(ap)
    args = ap.parse_args()
    protocol = protocol_from_args(args)

    for name in args.models:
        print(f">>> Обучение бейзлайна: {name}")
        journal = run_protocol(name, args.manifest, protocol, out_root=args.out_root,
                               accelerator=args.accelerator)
        print(f"    лучший чекпойнт: {journal['final_ckpt']}")


if __name__ == "__main__":
    main()
