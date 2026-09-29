"""Хост устройства на Python: построчный протокол на stdin, ответы JSON на stdout.

Протокол, формат и выбор файлов состояния, откат к климатологии точки - те же, что у
компилируемого рантайма, и проверяются одними эталонными сценариями. Модель берётся
либо из каталога экспорта графов ONNX, либо из чекпойнта PyTorch.

Запуск:
    python -m mayak.runtime.run_inference --model runtime/model \\
        --lat 52.37 --lon 4.90 --elev -2 --state-dir runtime --aci
    python -m mayak.runtime.run_inference --ckpt runs/mayak/stageB/best.ckpt \\
        --conformal runs/conformal.npy --lat 52.37 --lon 4.90 --elev -2 --aci

Команды: ``obs <секунды UTC> <T> <P> <RH>``, ``forecast [<секунды UTC>]``, ``status``.
Значение наблюдения - число, ``-`` (нет данных) или ``nan``. Если модель не удалось
поднять, в том числе не удался расчёт климатологии точки при старте, процесс
завершается с кодом 1.
"""
import argparse
import sys


def build_parser():
    """Разбор аргументов командной строки."""
    ap = argparse.ArgumentParser(description="хост устройства МАЯК на Python")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="каталог экспорта графов ONNX с манифестом")
    src.add_argument("--ckpt", help="чекпойнт PyTorch")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--state-dir", default="runtime",
                    help="каталог двух чередующихся файлов состояния")
    ap.add_argument("--aci", action="store_true",
                    help="адаптивная калибровка интервалов по собственным промахам прибора")
    ap.add_argument("--no-conformal", action="store_true",
                    help="не применять конформную таблицу")
    ap.add_argument("--int8", action="store_true", help="int8-графы экспорта")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--conformal", default=None,
                    help="конформная таблица с записью о подгонке рядом, только с --ckpt")
    ap.add_argument("--calibration-config", default=None,
                    help="YAML с параметрами ACI для --ckpt (по умолчанию "
                         "conf/calibration/default.yaml)")
    return ap


def open_runtime(args):
    """Потоковый рантайм по аргументам командной строки.

    Args:
        args: разобранные аргументы.

    Returns:
        Потоковый рантайм.
    """
    if args.model:
        from mayak.runtime.graphs import runtime_from_export
        return runtime_from_export(args.model, args.lat, args.lon, args.elev,
                                   precision="int8" if args.int8 else "fp32",
                                   threads=args.threads, conformal=not args.no_conformal,
                                   aci=args.aci)
    from mayak.lit import load_model
    from mayak.runtime.streaming import StreamingMayak
    aci = None
    if args.aci:
        from mayak.calibration import load_config
        aci = load_config(args.calibration_config).aci()
    conformal = None if args.no_conformal else args.conformal
    return StreamingMayak(load_model(args.ckpt).eval(), args.lat, args.lon, args.elev,
                          conformal=conformal, aci=aci)


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.ckpt and (args.int8 or args.threads != 1):
        ap.error("--int8 и --threads относятся к графам экспорта: нужен --model")
    if args.model and args.conformal:
        ap.error("--conformal относится к чекпойнту: у экспорта таблица лежит в манифесте")
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="mayak-rt: %(message)s")
    from mayak.runtime.host import Host, StateStore
    try:
        rt = open_runtime(args)
    except Exception as e:
        print(f"mayak-rt: {e}", file=sys.stderr)
        return 1
    host = Host(rt, StateStore(args.state_dir))
    host.restore()
    host.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
