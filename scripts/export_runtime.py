"""Экспорт модели для компилируемого рантайма: графы ONNX и манифест.

    python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \\
        --conformal runs/conformal_int8.npy --aci --int8 --out runtime/model

Каталог --out целиком - то, что копируется на устройство вместе с бинарником
runtime-rs (mayak-rt run --model runtime/model ...). Архитектура берётся из
конфига в чекпойнте; манифест хранит конфиг, размеры, пределы QC, формат состояния
и параметры калибровки. Конформная таблица должна быть подогнана на той точности, которую
объявляет экспорт: с --int8 - на int8-графах этого же экспорта, без него - на fp32.
Каждый граф при экспорте сверяется с PyTorch.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="экспорт графов ONNX для mayak-rt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runtime/model")
    ap.add_argument("--conformal", default=None,
                    help="конформная таблица с записью о подгонке рядом (runs/conformal.npy). "
                         "Без --int8 нужна таблица fp32, с --int8 - таблица, подогнанная на "
                         "int8-графах этого экспорта: python scripts/calibrate.py "
                         "--precision int8 --model-dir <каталог экспорта>")
    ap.add_argument("--aci", action="store_true", help="записать параметры ACI в манифест")
    ap.add_argument("--calibration-config", default=None,
                    help="YAML с параметрами ACI (по умолчанию conf/calibration/default.yaml)")
    ap.add_argument("--int8", action="store_true", help="дополнительно int8-копии графов")
    args = ap.parse_args()

    from mayak.lit import load_model
    from mayak.runtime.graphs import export_graphs
    aci = None
    if args.aci:
        from mayak.calibration import load_config
        aci = load_config(args.calibration_config).aci()
    try:
        man = export_graphs(load_model(args.ckpt), args.out, conformal=args.conformal, aci=aci,
                            int8=args.int8, checkpoint=args.ckpt)
    except ValueError as e:
        ap.error(str(e))
    print("Экспортировано:", args.out)
    cal = man["calibration"]
    print(f"  конформная таблица: {cal['precision'] if cal['conformal'] else 'нет'}")
    for name, err in man["export_check_max_rel"].items():
        print(f"  {name:9s} расхождение ONNX и PyTorch в единицах масштаба выхода {err:.2e}")
    print(f"  состояние на диске: {man['state']['nbytes']} Б (v{man['state']['version']})")


if __name__ == "__main__":
    main()
