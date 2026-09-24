"""Скачивание наблюдений GHCNh (NOAA/NCEI) для внешнего теста.

Отбор станций по списку станций (координаты, страна, тип сети, явный список id),
затем скачивание файлов по станциям. Скачанное кэшируется: готовый файл повторно
не запрашивается; ответ 404 запоминается (``<файл>.absent``); оборванная загрузка
докачивается с места обрыва (заголовок Range) из ``<файл>.part``; сетевые ошибки
повторяются с экспоненциальной паузой. Журнал - ``<out>/download_log.csv``.

Фильтр по длине ряда и наличию переменных применяется к скачанным данным в
``scripts/make_ghcnh.py``: в списке станций периода наблюдений нет.

Раскладка файлов (документация GHCNh 1.1.0):
  by-year: <base>/by-year/<ГОД>/<psv|parquet>/GHCNh_<ID>_<ГОД>.<psv|parquet>
  por:     <base>/by-station/GHCNh_<ID>_por.psv    (весь период, сотни МБ)
Зеркало в открытом реестре AWS (registry.opendata.aws/noaa-ghcnh) - через --base-url.

Запуск:
    python scripts/fetch_ghcnh.py --out data/ghcnh/raw --years 2015-2024 \\
        --bbox 35 60 -10 40 --max-stations 300 --jobs 8
"""
import argparse
import csv
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mayak.data.ghcnh import read_station_list

log = logging.getLogger("fetch_ghcnh")

BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access"
STATION_LIST_URL = ("https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
                    "hourly/doc/ghcnh-station-list.csv")
CHUNK = 1 << 20


class Absent(Exception):
    """Файла на сервере нет (404) - это ответ, а не ошибка сети."""


def file_urls(base, sid, layout, years=(), fmt="psv"):
    """[(url, имя файла)] для станции."""
    base = base.rstrip("/")
    if layout == "por":
        name = f"GHCNh_{sid}_por.psv"
        return [(f"{base}/by-station/{name}", name)]
    return [(f"{base}/by-year/{y}/{fmt}/GHCNh_{sid}_{y}.{fmt}", f"GHCNh_{sid}_{y}.{fmt}")
            for y in years]


def _open(url, start, timeout, open_url):
    req = urllib.request.Request(url)
    if start:
        req.add_header("Range", f"bytes={start}-")
    return open_url(req, timeout=timeout)


def download(url, dest, retries=5, timeout=60, backoff=2.0, open_url=urllib.request.urlopen,
             sleep=time.sleep):
    """Скачать url в dest с докачкой и повторами. Возвращает 'cached'|'absent'|'done'."""
    if os.path.exists(dest):
        return "cached"
    if os.path.exists(dest + ".absent"):
        return "absent"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    last = None
    for attempt in range(retries + 1):
        start = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            with _open(url, start, timeout, open_url) as resp:
                status = getattr(resp, "status", 200)
                mode = "ab" if (start and status == 206) else "wb"
                total = resp.headers.get("Content-Length") if resp.headers else None
                expect = (int(total) + (start if mode == "ab" else 0)) if total else None
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
            got = os.path.getsize(part)
            if expect is not None and got < expect:
                raise ConnectionError(f"обрыв: получено {got} из {expect} байт")
            os.replace(part, dest)
            return "done"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(dest + ".absent", "w").close()
                return "absent"
            if e.code == 416 and os.path.exists(part):
                os.replace(part, dest)
                return "done"
            last = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
        if attempt < retries:
            pause = backoff ** attempt
            log.info("%s: %s, повтор через %.0f с (%d/%d)", url, last, pause, attempt + 1, retries)
            sleep(pause)
    raise ConnectionError(f"{url}: не удалось скачать за {retries + 1} попыток: {last}")


def parse_years(spec):
    if not spec:
        return []
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def select_stations(df, bbox=None, countries=None, networks=None, ids=None, max_stations=None,
                    seed=0):
    """Фильтр списка станций. bbox = (lat_min, lat_max, lon_min, lon_max)."""
    sel = df
    if ids:
        sel = sel[sel["id"].isin(set(ids))]
    if bbox:
        la0, la1, lo0, lo1 = bbox
        sel = sel[(sel["lat"] >= la0) & (sel["lat"] <= la1)
                  & (sel["lon"] >= lo0) & (sel["lon"] <= lo1)]
    if countries:
        sel = sel[sel["id"].str[:2].isin(set(countries))]
    if networks:
        sel = sel[sel["id"].str[2].isin(set(networks))]
    sel = sel.sort_values("id").reset_index(drop=True)
    if max_stations and len(sel) > max_stations:
        idx = np.sort(np.random.default_rng(seed).choice(len(sel), max_stations, replace=False))
        sel = sel.iloc[idx].reset_index(drop=True)
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/ghcnh/raw")
    ap.add_argument("--station-list", default=STATION_LIST_URL,
                    help="URL или локальный путь к ghcnh-station-list.csv|.txt")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--layout", choices=("by-year", "por"), default="by-year")
    ap.add_argument("--format", choices=("psv", "parquet"), default="psv")
    ap.add_argument("--years", default="", help="для by-year: 2015-2024 или 2019,2020")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--countries", nargs="*", help="коды FIPS: первые два символа id")
    ap.add_argument("--networks", nargs="*", help="код сети: третий символ id (W, M, I, …)")
    ap.add_argument("--ids", default=None, help="файл со списком id по одному в строке")
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.makedirs(args.out, exist_ok=True)
    sl = args.station_list
    if sl.startswith(("http://", "https://")):
        local = os.path.join(args.out, os.path.basename(sl))
        download(sl, local, retries=args.retries, timeout=args.timeout)
        sl = local
    stations = read_station_list(sl)
    ids = None
    if args.ids:
        with open(args.ids) as f:
            ids = [line.strip() for line in f if line.strip()]
    sel = select_stations(stations, args.bbox, args.countries, args.networks, ids,
                          args.max_stations, args.seed)
    years = parse_years(args.years)
    if args.layout == "by-year" and not years:
        ap.error("для --layout by-year нужен --years")
    sel.to_csv(os.path.join(args.out, "selected_stations.csv"), index=False)
    log.info("станций в списке %d, отобрано %d", len(stations), len(sel))

    jobs = [(sid, url, os.path.join(args.out, sid, name)) for sid in sel["id"]
            for url, name in file_urls(args.base_url, sid, args.layout, years, args.format)]

    def run(job):
        sid, url, dest = job
        try:
            return sid, url, download(url, dest, args.retries, args.timeout), ""
        except Exception as e:
            return sid, url, "failed", str(e)

    with ThreadPoolExecutor(args.jobs) as ex, \
            open(os.path.join(args.out, "download_log.csv"), "a", newline="") as f:
        w = csv.writer(f)
        counts = {}
        for sid, url, status, err in ex.map(run, jobs):
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), sid, url, status, err])
            counts[status] = counts.get(status, 0) + 1
    log.info("файлов: %s", counts)
    if counts.get("failed"):
        log.warning("есть неудачные загрузки - перезапустите ту же команду: "
                    "готовое не перекачивается, оборванное докачивается")


if __name__ == "__main__":
    main()
