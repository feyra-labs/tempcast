"""Пересчёт отчёта этапа A, по которому человек выбирает чекпойнт для старта этапа B.

Отчёт о поле считается заново для лучшего чекпойнта этапа холодного старта и всех его
кандидатов, графики строятся заново: например, с другим числом примеров или после того,
как отчёт внутри обучения не построился. Отчёт считается на том же наборе валидации, по
которому выбирался чекпойнт; набор и кэш данных сверяются с журналом. Разрыв обобщения
считается по окнам обучающих станций, построенным по правилу набора валидации.

Отчёт ничего не решает: числа и графики нужны человеку, который выбирает кандидата.

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
    python scripts/stage_report.py --run runs/mayak
    python scripts/stage_report.py --run runs/mayak-no_solar --examples 10 --device cuda
    python scripts/stage_report.py --run runs/mayak/lr_search/lr0.001
"""


def _stage_entry(journal, name):
    return next((s for s in reversed(journal.get("stages", [])) if s.get("name") == name), None)


def report(args):
    """Пересчитывает отчёт этапа холодного старта прогона и печатает его.

    Args:
        args: разобранные аргументы командной строки.
    """
    from mayak.config import RunConfig
    from mayak.data.datamodule import train_stations_set, validation_set
    from mayak.data.store import get_store
    from mayak.leakage import run_checklist
    from mayak.protocol import CONFIG_FILE, Protocol, read_journal
    from mayak.stage_report import describe_checkpoint, format_field_report, write_stage_report
    from mayak.stages import FIELD_CURRICULUM

    journal = read_journal(args.run)
    protocol = Protocol.from_dict(journal["protocol"])
    stage = next((s for s in protocol.stages if s.curriculum == FIELD_CURRICULUM), None)
    if stage is None:
        sys.exit(f"{args.run}: в протоколе нет этапа холодного старта ({FIELD_CURRICULUM})")
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
    rep, path, plots = write_stage_report(
        os.path.dirname(entry["best_ckpt"]), stage.name, journal["arch"],
        describe_checkpoint(entry["best_ckpt"]), [describe_checkpoint(p) for p in paths], ds,
        train_dataset=gap_ds, device=args.device, seed=journal["seeds"]["eval"],
        metrics_csv=entry.get("metrics_csv"), n_examples=args.examples)
    print("\n".join(format_field_report(rep)))
    print(f"Отчёт: {path}")
    for p in plots:
        print(f"График: {p}")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="каталог прогона, например runs/mayak")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--examples", type=int, default=N_EXAMPLES,
                    help="число окон с примерами прогнозов")
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
