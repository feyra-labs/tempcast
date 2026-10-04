r"""Экспорт модели для устройства: графы ONNX и манифест.

    python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \
        --conformal runs/conformal.npy --aci --out runtime/model

Каталог --out целиком - то, что копируется на устройство:
``python -m mayak.runtime.run_inference --model runtime/model ...``. Архитектура берётся
из конфига в чекпойнте; манифест хранит конфиг, размеры, пределы QC, формат состояния,
параметры калибровки и пороги смены координат прибора. Конформная таблица должна быть
подогнана по тому же чекпойнту. Каждый граф при экспорте сверяется с PyTorch.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="экспорт графов ONNX для устройства")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runtime/model")
    ap.add_argument("--conformal", default=None,
                    help="конформная таблица с записью о подгонке рядом (runs/conformal.npy), "
                         "подогнанная по экспортируемому чекпойнту")
    ap.add_argument("--aci", action="store_true", help="записать параметры ACI в манифест")
    ap.add_argument("--calibration-config", default=None,
                    help="YAML с параметрами ACI (по умолчанию conf/calibration/default.yaml)")
    ap.add_argument("--runtime-config", default=None,
                    help="YAML с порогами смены координат (по умолчанию "
                         "conf/runtime/default.yaml)")
    args = ap.parse_args()

    from mayak.lit import load_model
    from mayak.runtime.graphs import export_graphs
    from mayak.runtime.site import load_runtime_config
    aci = None
    if args.aci:
        from mayak.calibration import load_config
        aci = load_config(args.calibration_config).aci()
    try:
        man = export_graphs(load_model(args.ckpt), args.out, conformal=args.conformal, aci=aci,
                            checkpoint=args.ckpt, runtime=load_runtime_config(args.runtime_config))
    except ValueError as e:
        ap.error(str(e))
    print("Экспортировано:", args.out)
    cal = man["calibration"]
    print(f"  конформная таблица: {'есть' if cal['conformal'] else 'нет'}")
    for name, err in man["export_check_max_rel"].items():
        print(f"  {name:9s} расхождение ONNX и PyTorch в единицах масштаба выхода {err:.2e}")
    print(f"  состояние на диске: {man['state']['nbytes']} Б (v{man['state']['version']})")
    rt = man["runtime"]
    print(f"  уточнение координат: до {rt['site_max_dlat_deg']:g}° по широте, "
          f"{rt['site_max_dlon_deg']:g}° по долготе, {rt['site_max_delev_m']:g} м по высоте")


if __name__ == "__main__":
    main()
