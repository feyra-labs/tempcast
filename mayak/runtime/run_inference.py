r"""Хост устройства на Python: построчный протокол на stdin, ответы JSON на stdout.

Модель берётся либо из каталога экспорта графов ONNX, либо из чекпойнта PyTorch: тогда
те же графы исполняются в PyTorch. Для каталога экспорта нужны только numpy и
onnxruntime; чекпойнт открывается только при установленных зависимостях обучения
(дополнительная группа ``train``).

Запуск:
    python -m mayak.runtime.run_inference --model runtime/model \
        --lat 52.37 --lon 4.90 --elev -2 --state-dir runtime --aci
    python -m mayak.runtime.run_inference --ckpt runs/mayak/stageB/best.ckpt \
        --conformal runs/conformal.npy --lat 52.37 --lon 4.90 --elev -2 --aci

Команды: ``obs <секунды UTC> <T> <P> <RH>``, ``forecast [<секунды UTC>]``, ``status``.
Пороги смены координат при перезапуске берутся из манифеста экспорта, а для чекпойнта -
из конфига рантайма.
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
    src.add_argument("--ckpt", help="чекпойнт PyTorch; нужны зависимости обучения (extra train)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, required=True,
                    help="высота точки над уровнем моря, м")
    ap.add_argument("--state-dir", default="runtime",
                    help="каталог двух чередующихся файлов состояния")
    ap.add_argument("--aci", action="store_true",
                    help="адаптивная калибровка интервалов по собственным промахам прибора")
    ap.add_argument("--no-conformal", action="store_true",
                    help="не применять конформную таблицу")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--conformal", default=None,
                    help="конформная таблица с записью о подгонке рядом, только с --ckpt")
    ap.add_argument("--calibration-config", default=None,
                    help="YAML с параметрами ACI для --ckpt (по умолчанию "
                         "conf/calibration/default.yaml)")
    ap.add_argument("--runtime-config", default=None,
                    help="YAML с порогами смены координат для --ckpt (по умолчанию "
                         "conf/runtime/default.yaml); у экспорта пороги лежат в манифесте")
    return ap


def open_runtime(args):
    """Устройство по аргументам командной строки.

    Args:
        args: разобранные аргументы.

    Returns:
        Устройство на графах экспорта или на графах PyTorch из чекпойнта.
    """
    if args.model:
        from mayak.runtime.backend import runtime_from_export
        return runtime_from_export(args.model, args.lat, args.lon, args.elev,
                                   threads=args.threads, conformal=not args.no_conformal,
                                   aci=args.aci)
    from mayak.export import TorchBackend
    from mayak.lit import load_model
    from mayak.runtime.device import Device
    aci = None
    if args.aci:
        from mayak.calibration import load_config
        aci = load_config(args.calibration_config).aci()
    from mayak.runtime.site import load_runtime_config
    conformal = None if args.no_conformal else args.conformal
    model = load_model(args.ckpt).eval()
    return Device(TorchBackend(model), model.cfg, args.lat, args.lon, args.elev,
                  conformal=conformal, aci=aci,
                  runtime_cfg=load_runtime_config(args.runtime_config))


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.ckpt and args.threads != 1:
        ap.error("--threads относится к графам экспорта: нужен --model")
    if args.model and args.conformal:
        ap.error("--conformal относится к чекпойнту: у экспорта таблица лежит в манифесте")
    if args.model and args.runtime_config:
        ap.error("--runtime-config относится к чекпойнту: у экспорта пороги лежат в манифесте")
    # Ответы и сообщения идут в UTF-8 на любой системе: иначе на Windows с однобайтовой
    # кодовой страницей вывода сообщение по-русски роняет хост.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="mayak: %(message)s")
    from mayak.runtime.host import Host, StateStore
    try:
        rt = open_runtime(args)
    except Exception as e:
        print(f"mayak: {e}", file=sys.stderr)
        return 1
    host = Host(rt, StateStore(args.state_dir))
    host.restore()
    host.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
