"""Скачивание почасовых рядов ERA5 для обучающих точек из архивного API Open-Meteo.

Каждая точка сохраняется в отдельный файл вместе с подписью запроса. Готовые
файлы при повторном запуске не запрашиваются. Запросы идут по очереди, пачками по
нескольку точек, с паузой между ними. Временные сбои повторяются с растущей
паузой. Если запрос так и не прошёл, скрипт останавливается, и его запускают ещё
раз позже.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from mayak.data.era5 import (ARCHIVE_URL, DEFAULT_END, DEFAULT_YEARS, ApiError, default_period,
                             make_raw, parse_date, parse_response, raw_matches, raw_path, read_raw,
                             request_params, request_signature, write_raw)
from mayak.provenance import git_info

log = logging.getLogger("fetch_era5")

FETCH_VERSION = "1"
RAW_DIR = "era5"
META_NAME = "fetch_meta.json"
LOG_NAME = "fetch_log.csv"
RETRY_CODES = (429, 500, 502, 503, 504)


def fetch_batch(points, signature, base_url=ARCHIVE_URL, retries=5, backoff=30.0,
                timeout=180.0):
    """Один запрос к API по пачке точек с повторами при временных сбоях.

    Повторяются сетевые ошибки, таймауты и ответы HTTP 429 и 5xx. Пауза перед
    первым повтором равна backoff и каждый раз удваивается.

    Args:
        points: список словарей с полями id, lat, lon.
        signature: подпись запроса.
        base_url: адрес эндпоинта архива.
        retries: число повторов после первой попытки.
        backoff: пауза перед первым повтором, с.
        timeout: таймаут одного запроса, с.

    Returns:
        Список ответов API по точкам в порядке пачки, уже проверенных разбором.

    Raises:
        ApiError: запрос отвергнут окончательно или ответ не разбирается.
        ConnectionError: запрос не прошёл за все попытки.
    """
    query = urllib.parse.urlencode(request_params([p["lat"] for p in points],
                                                  [p["lon"] for p in points], signature))
    req = urllib.request.Request(f"{base_url}?{query}")
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            parse_response(payload, len(points))
            return payload if isinstance(payload, list) else [payload]
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES:
                raise ApiError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError,
                UnicodeDecodeError) as e:
            last = str(e)
        if attempt < retries:
            pause = backoff * 2 ** attempt
            log.info("сбой запроса (%s), повтор через %.0f с (%d/%d)", last, pause,
                     attempt + 1, retries)
            time.sleep(pause)
    raise ConnectionError(f"запрос не прошёл за {retries + 1} попыток: {last}")


def classify(points, raw_dir, signature, refetch=False):
    """Какие точки уже скачаны с теми же параметрами, а какие надо качать.

    Args:
        points: список точек.
        raw_dir: каталог скачанных точек.
        signature: подпись текущего запроса.
        refetch: перекачивать файлы с другими параметрами вместо ошибки.

    Returns:
        Кортеж из списка готовых точек и списка точек к скачиванию.

    Raises:
        ValueError: есть файлы, скачанные с другими параметрами, а refetch не задан.
    """
    done, todo, clash = [], [], []
    for p in points:
        path = raw_path(raw_dir, p["id"])
        if not os.path.exists(path):
            todo.append(p)
            continue
        try:
            why = raw_matches(read_raw(path), p, signature)
        except (OSError, ValueError, EOFError) as e:
            why = f"файл не читается: {e}"
        if not why:
            done.append(p)
        elif refetch:
            todo.append(p)
        else:
            clash.append(f"{p['id']}: {why}")
    if clash:
        raise ValueError(
            f"{len(clash)} файлов в {raw_dir} скачаны не для этих точек или с другими "
            "параметрами: " + "; ".join(clash[:5]) + ("; …" if len(clash) > 5 else "")
            + ". Укажите другой --out или добавьте --refetch.")
    return done, todo


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_all(points, raw_dir, signature, batch=2, pause=1.0, base_url=ARCHIVE_URL, retries=5,
              backoff=30.0, timeout=180.0, refetch=False, log_path=None):
    """Скачать все недостающие точки.

    Args:
        points: список словарей с полями id, lat, lon.
        raw_dir: каталог скачанных точек.
        signature: подпись запроса.
        batch: число точек в запросе.
        pause: пауза между запросами, с.
        base_url: адрес эндпоинта архива.
        retries: число повторов при временном сбое.
        backoff: пауза перед первым повтором, с.
        timeout: таймаут запроса, с.
        refetch: перекачивать файлы, скачанные с другими параметрами.
        log_path: журнал запросов в формате CSV; None отключает журнал.

    Returns:
        Словарь счётчиков cached, done, failed, pending и stopped с причиной
        остановки или пустой строкой.
    """
    cached, todo = classify(points, raw_dir, signature, refetch)
    counts = dict(cached=len(cached), done=0, failed=0, pending=0, stopped="")
    size = max(1, int(batch))
    batches = [todo[i:i + size] for i in range(0, len(todo), size)]
    log_file = open(log_path, "a", newline="") if log_path else None
    writer = csv.writer(log_file) if log_file else None
    try:
        for k, chunk in enumerate(batches):
            ids = " ".join(str(p["id"]) for p in chunk)
            if k:
                time.sleep(pause)
            try:
                payloads = fetch_batch(chunk, signature, base_url, retries, backoff, timeout)
            except ApiError as e:
                counts["failed"] += len(chunk)
                status, message = "failed", str(e)
                log.warning("точки %s: %s", ids, e)
            except ConnectionError as e:
                counts["pending"] = sum(len(b) for b in batches[k:])
                counts["stopped"] = str(e)
                if writer:
                    writer.writerow([_now_iso(), ids, "stopped", str(e)])
                break
            else:
                stamp = _now_iso()
                for p, payload in zip(chunk, payloads):
                    write_raw(raw_path(raw_dir, p["id"]), make_raw(p, signature, payload, stamp))
                counts["done"] += len(chunk)
                status, message = "done", ""
                log.info("скачано %d из %d точек", counts["cached"] + counts["done"],
                         len(points))
            if writer:
                writer.writerow([_now_iso(), ids, status, message])
                log_file.flush()
    finally:
        if log_file:
            log_file.close()
    return counts


def read_points(path):
    """Таблица обучающих точек.

    Args:
        path: путь к CSV с колонками id, lat, lon, koppen.

    Returns:
        Список словарей в порядке файла.

    Raises:
        ValueError: нет нужных колонок или повторяются id.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"id", "lat", "lon"}
    if rows and not need <= set(rows[0]):
        raise ValueError(f"{path}: нужны колонки {sorted(need)}, есть {sorted(rows[0])}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: повторяются id точек")
    return rows


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_meta(path, signature, base_url, points_path, n_points, counts, previous=None):
    """Запись параметров скачивания рядом с данными.

    Args:
        path: путь к файлу параметров.
        signature: подпись запроса.
        base_url: адрес эндпоинта архива.
        points_path: путь к таблице точек.
        n_points: число точек в таблице.
        counts: счётчики скачивания.
        previous: прежнее содержимое файла, из него сохраняется дата первого скачивания.
    """
    now = _now_iso()
    info = git_info()
    meta = dict(
        source="Open-Meteo Historical Weather API", base_url=base_url,
        endpoint=signature["endpoint"], model=signature["model"],
        variables=signature["variables"], start_date=signature["start_date"],
        end_date=signature["end_date"], timezone=signature["timezone"], request=signature,
        points_file=os.path.abspath(points_path), points_sha256=_sha256(points_path),
        n_points=n_points, downloaded=counts["cached"] + counts["done"],
        failed=counts["failed"], pending=counts["pending"],
        first_fetch_utc=(previous or {}).get("first_fetch_utc", now), last_fetch_utc=now,
        script=dict(name="fetch_era5", version=FETCH_VERSION, git_commit=info.get("commit"),
                    git_dirty=info.get("dirty")),
        licence="Open-Meteo data: CC BY 4.0. ERA5: Copernicus Climate Change Service.")
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", default="data/points.csv", help="таблица точек")
    ap.add_argument("--out", default="data", help="корень набора данных")
    ap.add_argument("--end", default=DEFAULT_END,
                    help="последний день периода; не позже дня перед тестовым годом "
                         "внешнего теста")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS, help="длина периода, лет")
    ap.add_argument("--start", default=None, help="первый день периода вместо --years")
    ap.add_argument("--base-url", default=ARCHIVE_URL,
                    help="эндпоинт архива, например своего сервера Open-Meteo")
    ap.add_argument("--batch", type=int, default=2, help="точек в одном запросе")
    ap.add_argument("--pause", type=float, default=1.0, help="пауза между запросами, с")
    ap.add_argument("--retries", type=int, default=5, help="повторов при временном сбое")
    ap.add_argument("--backoff", type=float, default=30.0,
                    help="пауза перед первым повтором, дальше удваивается, с")
    ap.add_argument("--timeout", type=float, default=180.0, help="таймаут запроса, с")
    ap.add_argument("--refetch", action="store_true",
                    help="перекачать точки, скачанные с другими параметрами")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start, end = (parse_date(args.start), parse_date(args.end)) if args.start else \
        default_period(args.end, args.years)
    if start > end:
        ap.error(f"начало периода {start} позже конца {end}")
    signature = request_signature(start, end)
    points = read_points(args.points)
    raw_dir = os.path.join(args.out, RAW_DIR)
    meta_path = os.path.join(args.out, META_NAME)
    os.makedirs(raw_dir, exist_ok=True)

    previous = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("request") != signature and not args.refetch:
            sys.exit(f"{meta_path}: данные скачаны с другими параметрами "
                     f"({previous.get('start_date')}..{previous.get('end_date')}, "
                     f"модель {previous.get('model')}). Укажите другой --out или --refetch.")

    log.info("период %s..%s, модель %s, точек %d", start, end, signature["model"], len(points))
    try:
        counts = fetch_all(points, raw_dir, signature, args.batch, args.pause, args.base_url,
                           args.retries, args.backoff, args.timeout, args.refetch,
                           os.path.join(raw_dir, LOG_NAME))
    except ValueError as e:
        sys.exit(str(e))
    write_meta(meta_path, signature, args.base_url, args.points, len(points), counts, previous)

    print(f"готово {counts['cached'] + counts['done']} из {len(points)} "
          f"(было {counts['cached']}, скачано {counts['done']}, ошибок {counts['failed']}, "
          f"осталось {counts['pending']})")
    if counts["stopped"]:
        print(f"остановлено: {counts['stopped']}")
    if counts["cached"] + counts["done"] < len(points):
        print("запустите ту же команду ещё раз позже: готовые точки не перекачиваются")
        sys.exit(1)
    print(f"параметры скачивания: {meta_path}")
    print(f"далее: python scripts/make_era5.py --data {args.out}")


if __name__ == "__main__":
    main()
