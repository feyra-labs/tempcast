"""Офлайн-кэш станций и единая точка загрузки.

Сборка:  исходные файлы станций → QC (поточечный и оконный) → станционные проверки →
         отбор станций → сплиты → климатология → кэш и отчёт QC на диске.
Загрузка: кэш → StationStore в памяти, один объект на процесс.

Исходный файл станции (``stations/<id>.npz``): T, P, RH, valid, t0_utc_h и
необязательные ``flag`` — штатные флаги источника «подозрительно» формы (N,) или
(N, 3), и ``Td`` - точка росы (N,), если источник даёт её отдельно.
Необязательная колонка манифеста ``dem_elev`` — высота из цифровой модели рельефа.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from mayak.data.climatology import CLIM_VERSION, Climatology
from mayak.data.qc import (DEFAULT_QC, QC_CODE_DOC, QC_VERSION, STATION_CHECKS,
                           code_fractions, qc_station, station_checks, station_selection)
from mayak.data.splits import SPLITS_VERSION, TIME_BOUNDS, time_bounds
from mayak.timeaxis import CALENDAR_VERSION, legacy_t0, window_calendar

log = logging.getLogger(__name__)

CACHE_FORMAT = "4"   # Увеличивать при изменении; входит в ключ кэша.
CLIM_PARAMS = dict(n_year=3, n_day=3, scale_n_year=2, scale_n_day=2, min_valid=24 * 30)
CLIM_BASIS = ("n_year", "n_day", "scale_n_year", "scale_n_day")


def new_climatology():
    """Пустая климатология с базисом из CLIM_PARAMS"""
    return Climatology(**{k: CLIM_PARAMS[k] for k in CLIM_BASIS})


def read_manifest(manifest):
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"в манифесте {manifest} повторяются id станций")
    return rows


def source_path(manifest, sid):
    return os.path.join(os.path.dirname(os.path.abspath(manifest)), "stations", f"{sid}.npz")


def read_source(path):
    """Исходный файл станции → dict(T, P, RH, valid, t0[, flag][, Td])."""
    with np.load(path) as d:
        out = dict(T=d["T"], P=d["P"], RH=d["RH"], valid=d["valid"])
        for opt in ("flag", "Td"):
            if opt in d:
                out[opt] = d[opt]
        if "t0_utc_h" in d:
            out["t0"] = int(d["t0_utc_h"])
        else:
            log.warning("%s: старый формат (t0_doy/t0_hour) — пересоберите источник; "
                        "год условный, календарь приближённый", path)
            out["t0"] = legacy_t0(float(d["t0_doy"]), float(d["t0_hour"]))
    return out


def file_sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _opt_float(v):
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def qc_meta(row):
    """Метаданные станции, которые читает QC: долгота, высота, высота из ЦМР.

    Ровно эти поля входят в ключ кэша: от них зависят станционные проверки и
    проверка давления. Широта, зона и роль в QC не участвуют и в ключ не входят.
    """
    return dict(lon=_opt_float(row.get("lon")), elev=_opt_float(row.get("elev")),
                dem_elev=_opt_float(row.get("dem_elev")))


def qc_elev(meta):
    """Высота для проверки давления: из ЦМР, если есть, иначе заявленная."""
    return meta["dem_elev"] if meta.get("dem_elev") is not None else meta.get("elev")


def key_payload(manifest, rows=None, qc_cfg=DEFAULT_QC):
    rows = read_manifest(manifest) if rows is None else rows
    stations = sorted((r["id"], file_sha256(source_path(manifest, r["id"])),
                       sorted(qc_meta(r).items())) for r in rows)
    return {
        "format": CACHE_FORMAT, "qc": [QC_VERSION, qc_cfg.to_dict()],
        "calendar": CALENDAR_VERSION,
        "clim": [CLIM_VERSION, CLIM_PARAMS], "splits": [SPLITS_VERSION, TIME_BOUNDS],
        "stations": stations,
    }


def cache_key(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def default_cache_root(manifest):
    return os.path.join(os.path.dirname(os.path.abspath(manifest)), "cache")


def process_station(path, meta=None, qc_cfg=DEFAULT_QC):
    """QC + станционные проверки + отбор + климатология одной станции (в пуле процессов).

    Возвращает dict с данными станции либо с ``error``; в обоих случаях - ``report``
    (строка отчёта QC) и ``reasons`` [(правило, текст)].
    """
    meta = meta or {}
    src = read_source(path)
    x, mask, codes = qc_station(src["T"], src["P"], src["RH"], src["valid"],
                                Td=src.get("Td"), flag=src.get("flag"),
                                elev=qc_elev(meta), cfg=qc_cfg)
    n = x.shape[0]
    checks = station_checks(x, mask, src["t0"], lon=meta.get("lon"), elev=meta.get("elev"),
                            dem_elev=meta.get("dem_elev"), cfg=qc_cfg,
                            raw_T=src["T"], codes=codes)
    reasons = station_selection(n, mask, checks, qc_cfg)
    report = dict(n_hours=n, **code_fractions(codes, mask),
                  **{f"check/{k}": v["status"] for k, v in checks.items()},
                  **{f"check/{k}/value": v["value"] for k, v in checks.items()})
    try:
        lo, hi = time_bounds(n)["train"]
    except ValueError as e:
        reasons.append(("splits", f"сплиты: {e}"))
    if not reasons:
        k = np.arange(lo, hi)
        doy, hour = window_calendar(src["t0"], k)
        try:
            clim = new_climatology().fit(
                doy.astype(np.float64), hour.astype(np.float64), x[lo:hi, 0], mask[lo:hi, 0],
                min_valid=CLIM_PARAMS["min_valid"])
        except ValueError as e:
            reasons.append(("climatology", f"климатология: {e}"))
    base = dict(report=report, reasons=reasons, checks=checks)
    if reasons:
        return dict(base, error="; ".join(text for _, text in reasons))
    return dict(base, x=x, mask=mask, codes=codes, t0=src["t0"], beta=clim.beta,
                sigma=clim.sigma, scale_beta=clim.scale_beta, clim_fit=[int(lo), int(hi)])


def _process_station_args(args):
    return process_station(*args)


def qc_summary(results, rows, all_codes, all_mask, qc_cfg=DEFAULT_QC):
    by_rule = {}
    checks = {name: {"pass": 0, "fail": 0, "skip": 0} for name in STATION_CHECKS}
    for res in results:
        for rule, _ in res["reasons"]:
            by_rule[rule] = by_rule.get(rule, 0) + 1
        for name, c in res["checks"].items():
            checks[name][c["status"]] += 1
    return dict(
        version=QC_VERSION, config=qc_cfg.to_dict(), fingerprint=qc_cfg.fingerprint(),
        codes=QC_CODE_DOC,
        stations_total=len(rows), stations_included=sum("error" not in r for r in results),
        excluded_by_rule=dict(sorted(by_rule.items())),
        station_checks=checks,
        fractions_included=code_fractions(all_codes, all_mask),
    )


def build_cache(manifest, cache_root=None, jobs=1, force=False, qc_cfg=DEFAULT_QC):
    rows = read_manifest(manifest)
    t_start = time.perf_counter()
    payload = key_payload(manifest, rows, qc_cfg)
    key = cache_key(payload)
    root = cache_root or default_cache_root(manifest)
    final = os.path.join(root, key)
    log.info("кэш: ключ %s (хеширование источников %.1f с)", key, time.perf_counter() - t_start)
    if os.path.isdir(final) and not force:
        return final, False

    args = [(source_path(manifest, r["id"]), qc_meta(r), qc_cfg) for r in rows]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(_process_station_args, args, chunksize=4))
    else:
        results = [process_station(*a) for a in args]

    xs, ms, cs, betas, scale_betas, index, excluded, report = [], [], [], [], [], [], {}, []
    off = 0
    for r, res in zip(rows, results):
        status = "excluded" if "error" in res else "included"
        report.append(dict(id=r["id"], status=status, reason=res.get("error", ""),
                           **res["report"]))
        if "error" in res:
            excluded[r["id"]] = res["error"]
            continue
        n = res["x"].shape[0]
        xs.append(res["x"])
        ms.append(res["mask"])
        cs.append(res["codes"])
        betas.append(res["beta"])
        scale_betas.append(res["scale_beta"])
        index.append(dict(id=r["id"], offset=off, n=n, t0_utc_h=res["t0"], clim_sigma=res["sigma"],
                          clim_fit=res["clim_fit"],
                          qc_checks={k: v["status"] for k, v in res["checks"].items()}))
        off += n
    if not index:
        raise RuntimeError(f"ни одна станция не прошла сборку кэша: {excluded}")
    for sid, why in excluded.items():
        log.warning("станция %s исключена: %s", sid, why)

    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".tmp-{key}-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    np.save(os.path.join(tmp, "x.npy"), np.concatenate(xs))
    np.save(os.path.join(tmp, "mask.npy"), np.concatenate(ms))
    np.save(os.path.join(tmp, "qc.npy"), np.concatenate(cs))
    np.save(os.path.join(tmp, "clim_beta.npy"), np.stack(betas))
    np.save(os.path.join(tmp, "clim_scale_beta.npy"), np.stack(scale_betas))
    with open(os.path.join(tmp, "index.json"), "w") as f:
        json.dump(index, f)
    all_codes, all_mask = np.concatenate(cs), np.concatenate(ms)
    summary = qc_summary(results, rows, all_codes, all_mask, qc_cfg)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(dict(key=key, payload=payload, excluded=excluded,
                       qc_total=code_fractions(all_codes), qc=summary,
                       build_seconds=time.perf_counter() - t_start), f, indent=1, ensure_ascii=False)
    write_qc_report(os.path.join(tmp, "qc_report.csv"), report)

    if force:
        shutil.rmtree(final, ignore_errors=True)
    try:
        os.rename(tmp, final)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("кэш собран за %.1f с: %d станций, %d исключено → %s",
             time.perf_counter() - t_start, len(index), len(excluded), final)
    return final, True


def write_qc_report(path, report):
    fields = []
    for row in report:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report)


@dataclass
class StationStore:
    key: str
    path: str
    stations: dict = field(default_factory=dict)

    def by_role(self, role):
        return [s for s in self.stations.values() if s["role"] == role]

    def clims(self):
        return self.stations


def load_cache(path, rows, mmap=False):
    mode = "r" if mmap else None
    x = np.load(os.path.join(path, "x.npy"), mmap_mode=mode)
    mask = np.load(os.path.join(path, "mask.npy"), mmap_mode=mode)
    qc = np.load(os.path.join(path, "qc.npy"), mmap_mode=mode)
    beta = np.load(os.path.join(path, "clim_beta.npy"))
    scale_beta = np.load(os.path.join(path, "clim_scale_beta.npy"))
    basis = {k: CLIM_PARAMS[k] for k in CLIM_BASIS}
    with open(os.path.join(path, "index.json")) as f:
        index = json.load(f)
    meta_of = {r["id"]: r for r in rows}
    store = StationStore(key=os.path.basename(path), path=path)
    for i, it in enumerate(index):
        r = meta_of[it["id"]]
        a, n = it["offset"], it["n"]
        store.stations[it["id"]] = dict(
            id=it["id"], lat=float(r["lat"]), lon=float(r["lon"]), elev=float(r["elev"]),
            dem_elev=_opt_float(r.get("dem_elev")), koppen=r["koppen"], role=r.get("split"),
            qc_checks=it.get("qc_checks", {}),
            x=x[a:a + n], mask=mask[a:a + n], qc=qc[a:a + n], N=n, t0=int(it["t0_utc_h"]),
            clim_fit=tuple(it["clim_fit"]),
            clim=Climatology.from_params(beta[i], it["clim_sigma"], scale_beta[i], **basis))
    return store


_STORES: dict = {}


def _stat_fingerprint(manifest, rows):
    paths = [manifest] + [source_path(manifest, r["id"]) for r in rows]
    return tuple((st.st_mtime_ns, st.st_size, st.st_ino) for st in map(os.stat, paths))


def get_store(manifest, cache_root=None, mmap=False, build_if_missing=True, jobs=1, rebuild=False):
    manifest = os.path.abspath(manifest)
    rows = read_manifest(manifest)
    memo = (manifest, cache_root, mmap)
    fp = _stat_fingerprint(manifest, rows)
    hit = _STORES.get(memo)
    if hit is not None and hit[0] == fp and not rebuild:
        return hit[1]
    root = cache_root or default_cache_root(manifest)
    path = os.path.join(root, cache_key(key_payload(manifest, rows)))
    if rebuild or not os.path.isdir(path):
        if not build_if_missing:
            raise FileNotFoundError(f"кэш {path} не найден: запустите "
                                    f"python scripts/build_cache.py --manifest {manifest}")
        path, _ = build_cache(manifest, root, jobs=jobs, force=rebuild)
    store = load_cache(path, rows, mmap=mmap)
    _STORES[memo] = (fp, store)
    return store
