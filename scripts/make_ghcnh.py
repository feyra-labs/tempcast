"""Сборка набора внешнего теста из скачанных наблюдений GHCNh.

Скачанные файлы (scripts/fetch_ghcnh.py) → разбор T, Td, станционного давления и
их штатных кодов качества → почасовая сетка без интерполяции (ближайший к целому
часу отчёт в пределах допуска) → влажность из T и Td формулой модели →
<out>/stations/<id>.npz + <out>/manifest.csv (роль external_test) +
<out>/selection_report.csv.

Метаданные станции:
  * координаты - из списка станций;
  * высота, которую видит модель (elev) - из цифровой модели рельефа в точке
    станции, как у обучающих точек; заявленная высота станции - station_elev,
    их расхождение проверяет станционный QC и показывает разрез оценки;
  * зона Кёппена - полная, из растровой карты (Beck et al.).

ЦМР (--dem):
  open-meteo      Elevation API Open-Meteo (Copernicus DEM GLO-90, до 100 точек за
                  запрос; ответы кэшируются в <out>/dem_cache.json). Если обучающие
                  высоты пришли из Open-Meteo, это та же модель рельефа;
  <путь к .tif>   локальный растр или VRT-мозаика в WGS84;
  station         без ЦМР: высота станции (явное отклонение от плана, пишется в лог).

Рядом с манифестом пишется запись о том, каким кодом собраны файлы станций:
сборка кэша откажется работать, если этот код потом изменится, а набор не
пересоберут.

Затем - кэш и QC тем же модулем, что для обучения:
    python scripts/build_cache.py --manifest data/ghcnh/manifest.csv

Запуск:
    python scripts/make_ghcnh.py --raw data/ghcnh/raw --out data/ghcnh \\
        --koppen Beck_KG_V1_present_0p0083.tif --dem open-meteo
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

from mayak.data.ghcnh import build_external_dataset, read_station_list
from mayak.data.store import write_source_build, command_line

log = logging.getLogger("make_ghcnh")

BUILD_CODE = {
    "mayak.astro": ("rh_from_dewpoint",),
    "mayak.constants": ("H", "L_MAX", "MAGNUS_A", "MAGNUS_B"),
    "mayak.data.ghcnh": None,
    "mayak.data.rasters": None,
    "mayak.data.splits": None,
    "mayak.timeaxis": None,
    "mayak.zones": ("KOPPEN_ZONES", "KG_TIF_CODE", "UNKNOWN_ZONE"),
    "scripts/make_ghcnh.py": None,
}

OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"


class OpenMeteoElevation:
    """Высота из Elevation API Open-Meteo с дисковым кэшем. prefetch - пакетами по 100."""

    def __init__(self, cache_path, open_url=urllib.request.urlopen, batch=100, pause=0.5):
        self.cache_path = cache_path
        self.open_url, self.batch, self.pause = open_url, batch, pause
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    @staticmethod
    def key(lat, lon):
        return f"{float(lat):.4f},{float(lon):.4f}"

    def prefetch(self, points):
        todo = [p for p in points if self.key(*p) not in self.cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            q = urllib.parse.urlencode({
                "latitude": ",".join(f"{la:.4f}" for la, _ in chunk),
                "longitude": ",".join(f"{lo:.4f}" for _, lo in chunk)})
            with self.open_url(f"{OPEN_METEO_ELEVATION}?{q}", timeout=60) as r:
                elev = json.loads(r.read().decode())["elevation"]
            for p, h in zip(chunk, elev):
                self.cache[self.key(*p)] = h
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
            time.sleep(self.pause)

    def __call__(self, lat, lon):
        k = self.key(lat, lon)
        if k not in self.cache:
            self.prefetch([(lat, lon)])
        h = self.cache.get(k)
        return None if h is None or not np.isfinite(h) else float(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/ghcnh/raw")
    ap.add_argument("--out", default="data/ghcnh")
    ap.add_argument("--stations", default=None,
                    help="таблица станций; по умолчанию <raw>/selected_stations.csv")
    ap.add_argument("--koppen", default="Beck_KG_V1_present_0p0083.tif")
    ap.add_argument("--dem", default="open-meteo", help="open-meteo | <путь к растру> | station")
    ap.add_argument("--tol-minutes", type=int, default=20,
                    help="допуск от целого часа, мин (ближайший отчёт; дальше - дыра)")
    ap.add_argument("--min-train-years", type=int, default=None,
                    help="предотбор по длине ряда; по умолчанию EXTERNAL_MIN_TRAIN_YEARS")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from mayak.data.rasters import dem_reader, koppen_reader
    stations = read_station_list(args.stations or os.path.join(args.raw, "selected_stations.csv"))
    os.makedirs(args.out, exist_ok=True)

    if args.dem == "open-meteo":
        dem = OpenMeteoElevation(os.path.join(args.out, "dem_cache.json"))
        dem.prefetch(list(zip(stations["lat"], stations["lon"])))
    elif args.dem == "station":
        log.warning("--dem station: высота модели = заявленная высота станции, а не ЦМР; "
                    "это отклонение от плана, разрез по Δ высоты будет пустым")
        by_id = {(round(a, 4), round(b, 4)): h for a, b, h in
                 zip(stations["lat"], stations["lon"], stations["elev"])}
        dem = lambda la, lo: by_id.get((round(la, 4), round(lo, 4)))
    else:
        dem = dem_reader(args.dem)
    koppen = koppen_reader(args.koppen)

    rows, report = build_external_dataset(args.raw, args.out, stations, dem, koppen,
                                          tol_minutes=args.tol_minutes,
                                          min_train_years=args.min_train_years)
    if args.dem == "station":
        import csv
        mpath = os.path.join(args.out, "manifest.csv")
        with open(mpath) as f:
            mrows = list(csv.DictReader(f))
        for r in mrows:
            r["dem_elev"] = ""
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mrows[0]) if mrows else ["id"])
            w.writeheader()
            w.writerows(mrows)
    write_source_build(args.out, "make_ghcnh",
                       command_line(["python", "scripts/make_ghcnh.py", *sys.argv[1:]]), BUILD_CODE)
    reasons = {}
    for r in report:
        if r["status"] != "included":
            k = r["reason"].split(":")[0][:40]
            reasons[k] = reasons.get(k, 0) + 1
    print(f"включено станций: {len(rows)} из {len(report)}; причины исключения: {reasons}")
    print(f"манифест: {os.path.join(args.out, 'manifest.csv')}")
    print(f"отчёт:    {os.path.join(args.out, 'selection_report.csv')}")
    print("далее:    python scripts/build_cache.py "
          f"--manifest {os.path.join(args.out, 'manifest.csv')}")


if __name__ == "__main__":
    main()
