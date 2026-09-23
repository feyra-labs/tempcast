"""Экспорт модели для компилируемого рантайма: четыре графа ONNX + манифест.

    python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \\
        --conformal runs/conformal.npy --aci --int8 --out runtime/model

Каталог --out целиком - то, что копируется на устройство вместе с бинарником
runtime-rs (mayak-rt run --model runtime/model ...). Архитектура берётся из
конфига в чекпойнте; манифест хранит конфиг, размеры, пределы QC, формат состояния
и параметры калибровки. Каждый граф при экспорте сверяется с PyTorch.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description="экспорт четырёх графов ONNX для mayak-rt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runtime/model")
    ap.add_argument("--conformal", default=None, help="runs/conformal.npy (бины лидов × квантили)")
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
    man = export_graphs(load_model(args.ckpt), args.out, conformal=args.conformal, aci=aci,
                        int8=args.int8)
    print("Экспортировано:", args.out)
    for name, err in man["export_check_max_abs"].items():
        print(f"  {name:9s} max|ONNX − PyTorch| = {err:.2e}")
    print(f"  состояние на диске: {man['state']['nbytes']} Б (v{man['state']['version']})")


if __name__ == "__main__":
    main()
