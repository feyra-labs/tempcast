r"""Хост устройства на Python: построчный протокол на stdin, ответы JSON на stdout.

Модель берётся из каталога экспорта (``scripts/export_runtime.py``): графы ONNX, конформная
таблица, параметры адаптивной калибровки и пороги смены координат при перезапуске - из его
манифеста. Нужны только numpy и onnxruntime.

Запуск:
    python -m mayak.runtime.run_inference --model runtime/model \
        --lat 52.37 --lon 4.90 --elev -2 --state-dir runtime --aci

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
    ap.add_argument("--model", required=True, help="каталог экспорта графов ONNX с манифестом")
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
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
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
        from mayak.runtime.backend import runtime_from_export
        rt = runtime_from_export(args.model, args.lat, args.lon, args.elev,
                                 threads=args.threads, conformal=not args.no_conformal,
                                 aci=args.aci)
    except Exception as e:
        print(f"mayak: {e}", file=sys.stderr)
        return 1
    host = Host(rt, StateStore(args.state_dir))
    host.restore()
    host.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
