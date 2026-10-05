r"""Обучение нейробейзлайнов по тому же протоколу, что и МАЯК.

Эквивалентно `python scripts/train.py --arch <имя>` для каждого имени из --models:
та же функция запуска, те же флаги протокола и тот же подбор скорости обучения.

Этап 1 сравнения: подбор скорости обучения по сетке протокола, затем полный прогон:
    python scripts/train_neurobaselines.py --lr-search --accelerator gpu  # все бейзлайны
    python scripts/train_neurobaselines.py --models lru patchtst --lr-search --accelerator gpu
Отладка:
    python scripts/train_neurobaselines.py --models lru --steps-a 200 --steps-b 1000 \
        --batch 32 --windows 2000 --workers 0 --accelerator cpu --val-every 200 \
        --lr-search --lr-search-steps 400
"""
import argparse
import logging

from mayak.protocol import (ARCH_NAMES, ProtocolError, add_protocol_args, protocol_from_args,
                            run_protocol)
from mayak.stages import Launch
from mayak.tuning import add_tuning_args, check_search, format_tuning, tuning_from_args


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    neural = [a for a in ARCH_NAMES if a != "mayak"]
    ap.add_argument("--models", nargs="+", default=neural, choices=neural)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--out-root", default="runs")
    add_protocol_args(ap)
    add_tuning_args(ap, inherit=False)
    args = ap.parse_args()
    try:
        protocol = protocol_from_args(args)
        tuning = tuning_from_args(args)
        if tuning.lr_search:
            check_search(protocol, Launch())
    except (ValueError, ProtocolError) as e:
        ap.error(str(e))

    for name in args.models:
        print(f">>> Обучение бейзлайна: {name}")
        journal = run_protocol(name, args.manifest, protocol, out_root=args.out_root,
                               accelerator=args.accelerator, tuning=tuning)
        for line in format_tuning(journal.get("tuning")):
            print(f"    {line}")
        print(f"    лучший чекпойнт: {journal['final_ckpt']}")


if __name__ == "__main__":
    main()
