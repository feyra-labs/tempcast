"""Диагностика климат-поля после этапа холодного старта.

Числа поля и разрыв обобщения считает та же функция, что отчёт этапа, на тех же наборах
окон: валидационные станции в валидационном окне и обучающие станции в том же окне, при
нулевой истории. Набор, конфиг данных и сид бутстрапа берутся из чекпойнта, поэтому
числа совпадают с записью чекпойнта в отчёте этапа. У МАЯК дополнительно печатаются
чистое поле без паспорта и поправки голов, разложение выхода при нулевой истории и нормы
частот признаков координат на обоих наборах.

Запуск:
    python scripts/diagnose_stage_a.py --ckpt runs/mayak/stageA/best.ckpt
"""
import argparse
import logging
import sys

from torch.utils.data import DataLoader


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch

    from mayak import stage_report as SR
    from mayak.config import DataConfig
    from mayak.data.datamodule import train_stations_set, validation_set
    from mayak.data.store import get_store
    from mayak.evaluate import l0_decompose, pure_field_check
    from mayak.leakage import SELECTION_KEY, run_checklist
    from mayak.lit import load_model
    from mayak.protocol import Protocol
    from mayak.stages import FIELD_CURRICULUM, STAGE_KEY

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="чекпойнт этапа A (runs/mayak/stageA/best.ckpt)")
    ap.add_argument("--device", default="cpu",
                    help="устройство прогона; отчёт внутри обучения идёт на cuda, если обучение "
                         "шло на видеокарте")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    protocol = Protocol.from_dict(hp["protocol"])
    stage = next((s for s in protocol.stages if s.name == hp.get("stage")), None)
    if stage is None or stage.curriculum != FIELD_CURRICULUM:
        sys.exit(f"{args.ckpt}: этап {hp.get('stage')!r} не этап холодного старта")
    data_cfg = DataConfig.from_dict(hp["data_config"])
    manifest = data_cfg.manifest
    store = get_store(manifest, cache_root=data_cfg.cache_root)
    data_key = (ck.get(STAGE_KEY) or {}).get("data_key")
    if data_key and store.key != data_key:
        sys.exit(f"данные изменились: чекпойнт обучен на кэше {data_key}, сейчас {store.key}")
    val_ds = validation_set(store, manifest, data_cfg, stage.curriculum)
    digest = (ck.get(SELECTION_KEY) or {}).get("windows_digest")
    if digest != val_ds.fingerprint():
        sys.exit(f"набор валидации {val_ds.fingerprint()} не совпадает с набором выбора "
                 f"чекпойнта {digest}")
    train_ds = train_stations_set(store, manifest, data_cfg, stage.curriculum)
    run_checklist(store, datasets=[val_ds, train_ds], checkpoints=[args.ckpt])

    items = SR.report_items(SR.describe_checkpoint(args.ckpt), [])
    report, _ = SR.build_field_report(items, val_ds, arch=hp["arch"], stage=stage.name,
                                      train_dataset=train_ds, device=args.device,
                                      seed=protocol.resolved_seeds()["eval"], n_examples=0)
    print("\n".join(SR.format_field_report(report)))
    if hp["arch"] != "mayak":
        return

    model = load_model(args.ckpt)
    for ds in (val_ds, train_ds):
        pure_field_check(model, ds)
        l0_decompose(model, ds)
    w = model.loc.W.norm(dim=0)
    print("W норма  макс:", float(w.max()), " среднее:", float(w.mean()))
    with torch.no_grad():
        zs = [model(b)["z"].abs().mean().item() for b in DataLoader(val_ds, batch_size=128)]
    print(f"|z| при L=0 ({'/'.join(val_ds.station_splits)}, окно {val_ds.time_key}):",
          sum(zs) / max(len(zs), 1))


if __name__ == "__main__":
    main()
