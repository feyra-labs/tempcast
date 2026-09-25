"""Офлайн-кэш станций и единая точка загрузки.

Сборка: исходные файлы станций, центрированный QC, станционные проверки, отбор станций,
сплиты, климатология, затем кэш и отчёт QC на диске. Кроме очищенного ряда кэш хранит
сырые значения до QC и маску наличия от источника: из них история окна проходит тот же
причинный QC, что на устройстве.

Загрузка: кэш читается в StationStore, один объект на процесс.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from mayak.codehash import code_digests, unit_digest
from mayak.data.climatology import Climatology
from mayak.data.qc import (DEFAULT_QC, QC_CODE_DOC, STATION_CHECKS, code_fractions, presence,
                           qc_station, station_checks, station_selection)
from mayak.data.splits import (EXTERNAL_MIN_TRAIN_YEARS, ROLE_EXTERNAL, TIME_LAYOUT, full_years,
                               layout_fingerprint, time_layout)
from mayak.timeaxis import legacy_t0, window_calendar

log = logging.getLogger(__name__)

CLIM_PARAMS = dict(n_year=3, n_day=3, scale_n_year=2, scale_n_day=2, min_valid=24 * 30)
CLIM_BASIS = ("n_year", "n_day", "scale_n_year", "scale_n_day")

CACHE_CODE = {
    "mayak.constants": ("H", "L_MAX"),
    "mayak.data.climatology": None,
    "mayak.data.masking": None,
    "mayak.data.qc": None,
    "mayak.data.splits": None,
    "mayak.data.store": ("CLIM_PARAMS", "CLIM_BASIS", "new_climatology", "read_source",
                         "_opt_float", "qc_meta", "qc_elev", "process_station",
                         "_process_station_args", "qc_summary", "write_qc_report",
                         "build_cache"),
    "mayak.timeaxis": None,
}

CACHE_CODE_IGNORED = ("mayak.codehash", "check_sources", "read_manifest", "source_path",
                      "key_payload", "cache_key", "default_cache_root", "previous_build",
                      "rebuild_reasons", "log")
KEY_PARTS = {"sources": "источники", "qc_config": "конфиг QC",
             "clim_params": "параметры климатологии", "layout": "раскладка сплитов",
             "code": "код правил"}
SOURCE_BUILD_NAME = "source_build.json"


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
    """Метаданные станции, которые читает QC: долгота, заявленная высота, высота из ЦМР."""
    declared = _opt_float(row.get("station_elev"))
    meta = dict(lon=_opt_float(row.get("lon")),
                elev=declared if declared is not None else _opt_float(row.get("elev")),
                dem_elev=_opt_float(row.get("dem_elev")))
    if str(row.get("split") or "") == ROLE_EXTERNAL:
        meta["min_train_years"] = EXTERNAL_MIN_TRAIN_YEARS
    return meta


def qc_elev(meta):
    """Высота для проверки давления: из ЦМР, если есть, иначе заявленная."""
    return meta["dem_elev"] if meta.get("dem_elev") is not None else meta.get("elev")


def key_payload(manifest, rows=None, qc_cfg=DEFAULT_QC):
    """Всё, от чего зависит содержимое кэша, по частям.

    Args:
        manifest: путь к манифесту.
        rows: строки манифеста, если они уже прочитаны.
        qc_cfg: пороги контроля качества.

    Returns:
        Словарь частей ключа: источники, конфиг QC, параметры климатологии,
        раскладка сплитов и отпечатки кода правил.
    """
    rows = read_manifest(manifest) if rows is None else rows
    stations = sorted((r["id"], file_sha256(source_path(manifest, r["id"])),
                       sorted(qc_meta(r).items())) for r in rows)
    return {
        "sources": stations,
        "qc_config": qc_cfg.to_dict(),
        "clim_params": dict(CLIM_PARAMS),
        "layout": dict(params=dict(TIME_LAYOUT), code=layout_fingerprint()),
        "code": code_digests(CACHE_CODE),
    }


def cache_key(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def default_cache_root(manifest):
    return os.path.join(os.path.dirname(os.path.abspath(manifest)), "cache")


def previous_build(root, manifest):
    """Метаданные самой свежей готовой сборки того же манифеста.

    Args:
        root: каталог кэшей.
        manifest: абсолютный путь к манифесту.

    Returns:
        Словарь метаданных сборки или None, если готовых сборок нет.
    """
    if not os.path.isdir(root):
        return None
    best, best_time = None, -1
    for name in os.listdir(root):
        path = os.path.join(root, name, "meta.json")
        if name.startswith(".") or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if meta.get("manifest") not in (None, manifest):
            continue
        when = os.stat(path).st_mtime_ns
        if when > best_time:
            best, best_time = meta, when
    return best


def _short(ids, limit=5):
    return ", ".join(ids[:limit]) + (", …" if len(ids) > limit else "")


def _changed_keys(old, new):
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}.{sub}" for sub in _changed_keys(a, b)]
        else:
            out.append(k)
    return out


def _sources_change(old, new):
    a = {s[0]: (s[1], s[2]) for s in old}
    b = {s[0]: (s[1], s[2]) for s in new}
    both = sorted(a.keys() & b.keys())
    content = [i for i in both if a[i][0] != b[i][0]]
    groups = (("добавлены", sorted(b.keys() - a.keys())),
              ("удалены", sorted(a.keys() - b.keys())),
              ("изменено содержимое", content),
              ("изменены метаданные", [i for i in both if i not in content and a[i] != b[i]]))
    return "; ".join(f"{title} {len(ids)} ({_short(ids)})" for title, ids in groups if ids)


def rebuild_reasons(old, new):
    """Какие части ключа кэша изменились с прошлой сборки.

    Args:
        old: части ключа прошлой сборки или None, если её нет.
        new: части ключа текущей сборки.

    Returns:
        Список строк, по одной на изменившуюся часть ключа.
    """
    if old is None:
        return ["первая сборка: готовых сборок этого манифеста нет"]
    if not set(KEY_PARTS) & set(old):
        return ["прошлая сборка сделана с ключом другого состава, части не сравнить"]
    new = json.loads(json.dumps(new))
    out = []
    for part, title in KEY_PARTS.items():
        a, b = old.get(part), new.get(part)
        if a == b:
            continue
        if a is None:
            detail = "в прошлой сборке этой части ключа нет"
        elif part == "sources":
            detail = _sources_change(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            detail = "изменились " + ", ".join(_changed_keys(a, b))
        else:
            detail = "изменилось значение"
        out.append(f"{title}: {detail}")
    return out or ["ключ совпадает с прошлой сборкой"]


def command_line(args):
    """Команда одной строкой, как её набирают в оболочке этой системы.

    Args:
        args: аргументы команды, первой идёт программа.

    Returns:
        Строка с кавычками по правилам Windows или POSIX.
    """
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def write_source_build(out_dir, builder, command, spec):
    """Записывает рядом с манифестом, каким кодом собраны файлы станций.

    Args:
        out_dir: каталог набора, где лежат манифест и файлы станций.
        builder: короткое имя сборщика для сообщений.
        command: команда, которой набор пересобирается.
        spec: словарь: единица кода и кортеж имён определений, None означает весь файл.

    Returns:
        Путь к записанному файлу.
    """
    code = {unit: dict(names=None if names is None else sorted(names),
                       digest=unit_digest(unit, names)) for unit, names in sorted(spec.items())}
    path = os.path.join(out_dir, SOURCE_BUILD_NAME)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(builder=builder, command=command, code=code), f, ensure_ascii=False,
                  indent=1)
    os.replace(tmp, path)
    return path


def check_sources(manifest):
    """Проверяет, что файлы станций собраны тем же кодом, что сейчас в репозитории.

    Args:
        manifest: путь к манифесту.

    Raises:
        RuntimeError: код сборщика изменился после сборки файлов станций. В
            сообщении перечислены изменившиеся части и команда пересборки.
    """
    folder = os.path.dirname(os.path.abspath(manifest))
    path = os.path.join(folder, SOURCE_BUILD_NAME)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    changed = []
    for unit, item in sorted(rec.get("code", {}).items()):
        try:
            now = unit_digest(unit, item.get("names"))
        except (OSError, ImportError, ValueError, SyntaxError):
            now = None
        if now != item.get("digest"):
            changed.append(unit)
    if changed:
        raise RuntimeError(f"файлы станций в {folder} собраны другим кодом сборщика "
                           f"{rec.get('builder')} (изменились: {', '.join(changed)}); "
                           f"пересоберите их: {rec.get('command')}")


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
    raw = np.stack([src["T"], src["P"], src["RH"]], axis=-1).astype(np.float32)
    present = presence(raw, src["valid"])
    raw = np.where(present > 0, raw, 0.0).astype(np.float32)
    n = x.shape[0]
    checks = station_checks(x, mask, src["t0"], lon=meta.get("lon"), elev=meta.get("elev"),
                            dem_elev=meta.get("dem_elev"), cfg=qc_cfg,
                            raw_T=src["T"], codes=codes)
    reasons = station_selection(n, mask, checks, qc_cfg)
    report = dict(n_hours=n, **code_fractions(codes, mask),
                  **{f"check/{k}": v["status"] for k, v in checks.items()},
                  **{f"check/{k}/value": v["value"] for k, v in checks.items()})
    try:
        lo, hi = time_layout(n).span("train")
    except ValueError as e:
        reasons.append(("splits", f"сплиты: {e}"))
    need_years = int(meta.get("min_train_years") or 0)
    if need_years and not reasons:
        ny = full_years(mask[:, 0], src["t0"], lo, hi)
        report["train_full_years"] = ny
        if ny < need_years:
            reasons.append(("clim_years", f"в обучающем окне {ny} полных лет < {need_years}: "
                                          f"климатология станции неустойчива"))
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
    return dict(base, x=x, mask=mask, codes=codes, raw=raw, present=present, t0=src["t0"],
                beta=clim.beta,
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
        config=qc_cfg.to_dict(), fingerprint=qc_cfg.fingerprint(),
        codes=QC_CODE_DOC,
        stations_total=len(rows), stations_included=sum("error" not in r for r in results),
        excluded_by_rule=dict(sorted(by_rule.items())),
        station_checks=checks,
        fractions_included=code_fractions(all_codes, all_mask),
    )


def build_cache(manifest, cache_root=None, jobs=1, force=False, qc_cfg=DEFAULT_QC):
    """Собирает кэш станций, если для текущего ключа его ещё нет.

    Args:
        manifest: путь к манифесту.
        cache_root: каталог кэшей; по умолчанию каталог cache рядом с манифестом.
        jobs: число процессов обработки станций.
        force: пересобрать, даже если кэш с таким ключом уже есть.
        qc_cfg: пороги контроля качества.

    Returns:
        Пара: путь к каталогу кэша и признак, что он собран этим вызовом.

    Raises:
        RuntimeError: файлы станций собраны устаревшим кодом сборщика или ни одна
            станция не прошла сборку.
    """
    check_sources(manifest)
    rows = read_manifest(manifest)
    t_start = time.perf_counter()
    payload = key_payload(manifest, rows, qc_cfg)
    key = cache_key(payload)
    root = cache_root or default_cache_root(manifest)
    final = os.path.join(root, key)
    log.info("кэш: ключ %s (хеширование источников %.1f с)", key, time.perf_counter() - t_start)
    if os.path.isdir(final) and not force:
        return final, False
    previous = previous_build(root, os.path.abspath(manifest))
    if os.path.isdir(final):
        reasons = ["пересборка по требованию при том же ключе"]
    else:
        reasons = rebuild_reasons(None if previous is None else previous.get("payload"), payload)
    for reason in reasons:
        log.info("кэш: причина сборки - %s", reason)

    args = [(source_path(manifest, r["id"]), qc_meta(r), qc_cfg) for r in rows]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(_process_station_args, args, chunksize=4))
    else:
        results = [process_station(*a) for a in args]

    xs, ms, cs, raws, pres = [], [], [], [], []
    betas, scale_betas, index, excluded, report = [], [], [], {}, []
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
        raws.append(res["raw"])
        pres.append(res["present"])
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
    np.save(os.path.join(tmp, "raw.npy"), np.concatenate(raws))
    np.save(os.path.join(tmp, "present.npy"), np.concatenate(pres))
    np.save(os.path.join(tmp, "clim_beta.npy"), np.stack(betas))
    np.save(os.path.join(tmp, "clim_scale_beta.npy"), np.stack(scale_betas))
    with open(os.path.join(tmp, "index.json"), "w") as f:
        json.dump(index, f)
    all_codes, all_mask = np.concatenate(cs), np.concatenate(ms)
    summary = qc_summary(results, rows, all_codes, all_mask, qc_cfg)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(key=key, manifest=os.path.abspath(manifest),
                       rebuild=dict(previous_key=None if previous is None else previous.get("key"),
                                    reasons=reasons),
                       payload=payload, excluded=excluded,
                       qc_total=code_fractions(all_codes), qc=summary,
                       build_seconds=time.perf_counter() - t_start),
                  f, indent=1, ensure_ascii=False)
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
    raw = np.load(os.path.join(path, "raw.npy"), mmap_mode=mode)
    present = np.load(os.path.join(path, "present.npy"), mmap_mode=mode)
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
            dem_elev=_opt_float(r.get("dem_elev")), station_elev=_opt_float(r.get("station_elev")),
            report_every=_opt_float(r.get("report_every")),
            koppen=r["koppen"], role=r.get("split"),
            qc_checks=it.get("qc_checks", {}),
            x=x[a:a + n], mask=mask[a:a + n], qc=qc[a:a + n], raw=raw[a:a + n],
            present=present[a:a + n], N=n, t0=int(it["t0_utc_h"]),
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
    check_sources(manifest)
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
