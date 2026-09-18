"""Офлайн-сборка кэша станций: QC + климатология.

Все остальные шаги читают готовый кэш и никогда не пересчитывают QC.
Повторный запуск при неизменных источниках и правилах = только хеширование.

Запуск:
    python scripts/build_cache.py --manifest data/manifest.csv --jobs 8
"""
import argparse
import json
import logging
import os
import time

from mayak.data.store import build_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache-root", default=None, help="по умолчанию <каталог манифеста>/cache")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true", help="пересобрать, даже если ключ совпал")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t = time.perf_counter()
    path, built = build_cache(args.manifest, args.cache_root, jobs=args.jobs, force=args.force)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    print(("собран" if built else "уже актуален"), f"за {time.perf_counter() - t:.1f} с:", path)
    print("исключено станций:", len(meta["excluded"]))
    bad = {k: round(v, 5) for k, v in meta["qc_total"].items() if v > 0}
    print("доли кодов QC по набору:", bad or "нет отбраковки")


if __name__ == "__main__":
    main()
