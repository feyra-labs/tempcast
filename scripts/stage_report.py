"""Отчёты для ручного решения о переходе от этапа A к этапу B.

Команда a пересчитывает отчёт о поле для лучшего чекпойнта этапа и всех его кандидатов
и заново строит графики: например, с другим числом примеров, с порогом ворот или после
того, как отчёт внутри обучения не построился. Отчёт считается на том же наборе
валидации, по которому выбирался чекпойнт; набор и кэш данных сверяются с журналом.
Разрыв обобщения считается по окнам обучающих станций, построенным по правилу набора
валидации.

Команда b сравнивает запуски этапа B, начатые с разных чекпойнтов этапа A, обычно
пробные: кривые валидации на одних осях, значение на общем для всех шаге, pinball по
длинам истории на лучшем шаге и числа поля у стартового чекпойнта каждого запуска.

Примеры команд выводит справка.
"""
import argparse
import json
import logging
import os
import sys

from mayak.stage_report import N_EXAMPLES

EXAMPLES = """\
Примеры:
    python scripts/stage_report.py a --run runs/mayak
    python scripts/stage_report.py a --run runs/mayak --examples 10 --gate 1.05
    python scripts/stage_report.py b --runs runs/mayak-probe-a4000 runs/mayak-probe-a10000 \\
        --out runs/plots/stageB_runs.png
"""


def _stage_entry(journal, name):
    return next((s for s in reversed(journal.get("stages", [])) if s.get("name") == name), None)


def cmd_a(args):
    from mayak.config import RunConfig
    from mayak.data.datamodule import train_stations_set, validation_set
    from mayak.data.store import get_store
    from mayak.leakage import run_checklist
    from mayak.protocol import CONFIG_FILE, Protocol, read_journal
    from mayak.stage_report import describe_checkpoint, format_field_report, write_stage_report
    from mayak.stages import FIELD_CURRICULUM

    journal = read_journal(args.run)
    protocol = Protocol.from_dict(journal["protocol"])
    stage = next((s for s in protocol.stages if s.name == args.stage), None)
    if stage is None or stage.curriculum != FIELD_CURRICULUM:
        sys.exit(f"этап {args.stage}: отчёт о поле считается только для этапа холодного старта")
    entry = _stage_entry(journal, stage.name)
    if entry is None:
        sys.exit(f"{args.run}: в журнале нет этапа {stage.name}")
    with open(os.path.join(args.run, CONFIG_FILE)) as f:
        data_cfg = RunConfig.from_dict(json.load(f)).data
    manifest = journal["manifest"]
    store = get_store(manifest, cache_root=data_cfg.cache_root)
    if journal.get("data_key") and store.key != journal["data_key"]:
        sys.exit(f"данные изменились: прогон обучен на кэше {journal['data_key']}, сейчас "
                 f"{store.key}; отчёт на других данных не имеет смысла")
    ds = validation_set(store, manifest, data_cfg, stage.curriculum)
    if ds.fingerprint() != entry["val_set"]:
        sys.exit(f"набор валидации {ds.fingerprint()} не совпадает с набором выбора "
                 f"{entry['val_set']}")
    paths = [c["ckpt"] for c in entry.get("candidates") or []]
    missing = [p for p in [entry["best_ckpt"], *paths] if not os.path.isfile(p)]
    if missing:
        sys.exit(f"нет файлов чекпойнтов: {missing}")
    gap_ds = train_stations_set(store, manifest, data_cfg, stage.curriculum)
    run_checklist(store, datasets=[ds, gap_ds], checkpoints=[entry["best_ckpt"], *paths])
    report, path, plots = write_stage_report(
        os.path.dirname(entry["best_ckpt"]), stage.name, journal["arch"],
        describe_checkpoint(entry["best_ckpt"]), [describe_checkpoint(p) for p in paths], ds,
        train_dataset=gap_ds, device=args.device, seed=journal["seeds"]["eval"],
        threshold=args.gate, metrics_csv=entry.get("metrics_csv"), n_examples=args.examples)
    print("\n".join(format_field_report(report)))
    print(f"Отчёт: {path}")
    for p in plots:
        print(f"График: {p}")


def cmd_b(args):
    from mayak.stage_report import format_stage_runs, plot_stage_runs, stage_runs
    rows, warnings = stage_runs(args.runs, stage=args.stage)
    if not rows:
        sys.exit("\n".join(warnings) or "нет запусков для сравнения")
    print("\n".join(format_stage_runs(rows, warnings)))
    print(f"График: {plot_stage_runs(rows, args.out)}")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("a", help="отчёт о поле по чекпойнтам этапа холодного старта")
    a.add_argument("--run", required=True, help="каталог прогона, например runs/mayak")
    a.add_argument("--stage", default="A")
    a.add_argument("--device", default="cpu")
    a.add_argument("--gate", type=float, default=None,
                   help="порог отношения MSE для пометки кандидатов; на обучение не влияет")
    a.add_argument("--examples", type=int, default=N_EXAMPLES,
                   help="число окон с примерами прогнозов")
    a.set_defaults(fn=cmd_a)
    b = sub.add_parser("b", help="сравнение запусков этапа, начатых с разных чекпойнтов")
    b.add_argument("--runs", nargs="+", required=True, help="каталоги прогонов")
    b.add_argument("--stage", default="B")
    b.add_argument("--out", default="runs/plots/stageB_runs.png")
    b.set_defaults(fn=cmd_b)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = make_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
