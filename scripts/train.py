"""Обучение одной архитектуры по единому протоколу.

Конфиг запуска разных моделей отличается только именем архитектуры: этапы, шаги, батч,
поток окон, оптимизатор, расписание, ранняя остановка, выбор чекпойнта и усреднение
весов общие. Чекпойнты выбираются по val/loss на валидационных станциях в валидационном
окне; перед каждым этапом прогоняется чек-лист антиутечек.

Этапы A и B можно запускать отдельно. После этапа A в каталоге этапа лежат лучший
чекпойнт, кандидаты после каждой валидации, отчёт о поле и графики. Человек смотрит
отчёт и графики и сам выбирает, с какого чекпойнта запускать этап B. Примеры команд
выводит справка.
"""
import argparse
import logging
import os
import shlex
import sys

from mayak.config import ABLATION_NAMES, Ablations, ModelConfig, run_label
from mayak.protocol import (ARCH_NAMES, ProtocolError, add_protocol_args, protocol_from_args,
                            run_protocol)
from mayak.stages import (FIELD_CURRICULUM, GATE_EXIT_CODE, INIT_EXIT_CODE, LAUNCH_FIELDS,
                          GateError, InitCheckpointError, add_launch_args, launch_from_args,
                          plan_stages)

# Флаги, которые описывают этот запуск, а не прогон: подсказка следующей команды
# собирается без них.
NOT_REPEATED = set(LAUNCH_FIELDS) | {"tag", "help"}

EXAMPLES = """\
Журнал прогона: <out-root>/<tag>/protocol.json, по умолчанию tag равен --arch.
После этапа A в <out-root>/<tag>/stageA/: best.ckpt, candidates/step*.ckpt, report.json,
report/*.png.

Отладка:
    python scripts/train.py --arch mayak --steps-a 200 --steps-b 1000 --batch 32 \\
        --windows 2000 --workers 0 --accelerator cpu --precision 32

Полный прогон одной командой (для бейзлайнов те же флаги, другое --arch):
    python scripts/train.py --arch mayak --accelerator gpu

По этапам, с ручным решением между ними:
    python scripts/train.py --arch mayak --accelerator gpu --stages A
    python scripts/stage_report.py a --run runs/mayak          # по желанию: пересчёт отчёта
    python scripts/train.py --arch mayak --accelerator gpu --stages B \\
        --init-from runs/mayak/stageA/candidates/step006000.ckpt --require-gate 1.05

Пробный этап B с нескольких кандидатов и их сравнение:
    python scripts/train.py --arch mayak --accelerator gpu --stages B --probe-steps 20000 \\
        --init-from runs/mayak/stageA/candidates/step004000.ckpt --tag mayak-probe-a4000
    python scripts/stage_report.py b --runs runs/mayak-probe-a4000 runs/mayak-probe-a10000

Внутренности поля МАЯК после этапа A:
    python scripts/diagnose_stage_a.py --ckpt runs/mayak/stageA/best.ckpt

Абляции переобучением (прогон в runs/mayak-<флаги>):
    python scripts/train.py --arch mayak --ablate no_compression --accelerator gpu

Бейзлайны, их источники и отличия от оригиналов: MODELS.md. Композиция конфигов,
переопределения и групповые запуски: python scripts/run.py --help.
"""


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCH_NAMES, default="mayak")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--tag", default=None,
                    help="имя каталога прогона; по умолчанию имя архитектуры или варианта "
                         "абляции")
    ap.add_argument("--ablate", nargs="*", default=[], choices=ABLATION_NAMES,
                    help="флаги абляций МАЯК (переобучение без компонента)")
    add_protocol_args(ap)
    add_launch_args(ap)
    return ap


def base_command(ap, args):
    """Команда обучения с теми же флагами прогона, без флагов этого запуска.

    Args:
        ap: разборщик аргументов.
        args: разобранные аргументы.

    Returns:
        Список слов команды.
    """
    words = ["python", "scripts/train.py"]
    for action in ap._actions:
        if not action.option_strings or action.dest in NOT_REPEATED:
            continue
        value = getattr(args, action.dest)
        if value == ap.get_default(action.dest):
            continue
        words.append(action.option_strings[0])
        words += [str(v) for v in value] if isinstance(value, list) else [str(value)]
    return words


def print_next_steps(ap, args, journal, protocol):
    """Печатает итог запуска и подсказку, как продолжить.

    Args:
        ap: разборщик аргументов.
        args: разобранные аргументы.
        journal: журнал прогона.
        protocol: протокол обучения.
    """
    from mayak.stage_report import format_field_report, read_report
    names = [s.name for s in protocol.stages]
    for st in journal["stages"]:
        print(f"Лучшая модель этапа {st['name']}:", st["best_ckpt"])
        if st.get("probe_steps"):
            print(f"    пробный запуск: {st['steps_done']} шагов из "
                  f"{protocol.stages[names.index(st['name'])].steps}")
    last = journal["stages"][-1]
    if last.get("report"):
        print()
        print("\n".join(format_field_report(read_report(last["report"]))))
        print(f"Отчёт: {last['report']}")
        if last.get("plots"):
            print(f"Графики: {os.path.dirname(last['plots'][0])}")
    idx = names.index(last["name"])
    stage = protocol.stages[idx]
    if journal["final_ckpt"]:
        print("Итоговый чекпойнт:", journal["final_ckpt"])
        return
    if last.get("probe_steps"):
        print("Пробный запуск итогового чекпойнта не даёт. Сравнение запусков:")
        print(f"    python scripts/stage_report.py b --stage {stage.name} --runs "
              f"{os.path.join(args.out_root, '<тег>')} ...")
        return
    nxt = names[idx + 1]
    cmd = base_command(ap, args)
    cand_dir = os.path.join(os.path.dirname(last["best_ckpt"]), "candidates")
    print(f"\nЭтап {stage.name} готов. Посмотрите отчёт и графики и выберите чекпойнт для "
          f"этапа {nxt}: лучший по val/loss или любой кандидат из {cand_dir}.")
    print(f"Этап {nxt} с выбранного чекпойнта:")
    print("    " + shlex.join(cmd + ["--stages", nxt, "--init-from", last["best_ckpt"]]))
    if stage.curriculum == FIELD_CURRICULUM:
        print("    (добавьте --require-gate 1.05, чтобы не начинать этап при плохом поле)")
    print(f"Пробный этап {nxt} с нескольких кандидатов для сравнения, по тегу на кандидата:")
    print("    " + shlex.join(cmd + ["--stages", nxt, "--init-from", "<кандидат>",
                                     "--probe-steps", "<шаги>", "--tag", "<тег>"]))
    print(f"    python scripts/stage_report.py b --stage {nxt} --runs <каталоги пробных "
          f"запусков>")
    if args.arch == "mayak" and stage.curriculum == FIELD_CURRICULUM:
        print(f"Внутренности поля МАЯК: python scripts/diagnose_stage_a.py --ckpt "
              f"{last['best_ckpt']}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = make_parser()
    args = ap.parse_args(argv)
    try:
        launch = launch_from_args(args)
        protocol = protocol_from_args(args)
        plan_stages(protocol, launch)
    except (ValueError, ProtocolError) as e:
        ap.error(str(e))
    model_config, tag = None, args.arch
    if args.ablate:
        if args.arch != "mayak":
            ap.error("--ablate применим только к --arch mayak")
        model_config = ModelConfig(ablations=Ablations(**{n: True for n in args.ablate}))
        tag = run_label(model_config)
    tag = args.tag or tag
    try:
        journal = run_protocol(args.arch, args.manifest, protocol,
                               out_root=args.out_root, accelerator=args.accelerator, tag=tag,
                               model_config=model_config, launch=launch)
    except InitCheckpointError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(INIT_EXIT_CODE)
    except GateError as e:
        print(f"\nВорота закрыты: {e}", file=sys.stderr)
        sys.exit(GATE_EXIT_CODE)
    print_next_steps(ap, args, journal, protocol)


if __name__ == "__main__":
    main()
