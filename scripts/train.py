"""Обучение одной архитектуры по единому протоколу (mayak/protocol.py).

Конфиг запуска разных моделей отличается только именем архитектуры (--arch): этапы,
шаги, батч, поток окон, оптимизатор, расписание, ранняя остановка, выбор чекпойнта
и EMA весов общие. Чекпойнты выбираются по val/loss на валидационных станциях
в валидационном окне; перед каждым этапом прогоняется чек-лист антиутечек.
Журнал прогона: <out-root>/<arch>/protocol.json. Диагностика поля после этапа A -
отдельный скрипт scripts/diagnose_stage_a.py.

Запуск отладка:
    python scripts/train.py --arch mayak --steps-a 200 --steps-b 1000 --batch 32 \\
        --windows 2000 --workers 0 --accelerator cpu --precision 32

Запуск полный (для бейзлайнов — те же флаги, другое --arch):
    python scripts/train.py --arch mayak --accelerator gpu
    python scripts/train.py --arch gru --accelerator gpu
    python scripts/train.py --arch dlinear --accelerator gpu
"""
import argparse
import logging

from mayak.protocol import ARCH_NAMES, add_protocol_args, protocol_from_args, run_protocol


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=ARCH_NAMES, default="mayak")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--out-root", default="runs")
    add_protocol_args(ap)
    return ap


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = make_parser().parse_args()
    journal = run_protocol(args.arch, args.manifest, protocol_from_args(args),
                           out_root=args.out_root, accelerator=args.accelerator)
    for st in journal["stages"]:
        print(f"Лучшая модель этапа {st['name']}:", st["best_ckpt"])
    if args.arch == "mayak":
        print(f"    диагностика поля: python scripts/diagnose_stage_a.py "
              f"--ckpt {journal['stages'][0]['best_ckpt']}")
    print("Итоговый чекпойнт:", journal["final_ckpt"])


if __name__ == "__main__":
    main()
