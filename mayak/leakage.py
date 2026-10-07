"""Исполняемый чек-лист антиутечек.

Проверки вызываются перед обучением и перед оценкой:

1. раскладка ряда каждой станции упорядочена, блоки не пусты, а год валидации и
   калибровки отделён от обучения и теста зазором не короче горизонта и полной
   истории вместе;
2. климатология каждой станции подогнана только по её обучающему окну;
3. цель каждого окна датасета лежит целиком в одном блоке своего временного окна;
   история окон валидации и калибровки может заходить в зазор и в соседние блоки,
   но не в обучение и тест; история тестовых окон не заходит в обучение, валидацию
   и калибровку; история обучающих окон не выходит за обучение;
4. конформная таблица построена только по калибровочным блокам и только на
   валидационных станциях, длины истории её окон взяты из распределения куррикулума,
   таблица разбита по бинам лидов и тем же бинам длины истории, что в коде, для каждого
   бина длины истории записано, своя у него строка или маргинальная, а поправка
   медианы равна нулю;
5. чекпойнт выбран по метрике на валидационных станциях в валидационных блоках, а
   длины истории окон валидации взяты из распределения куррикулума, а не одной длиной;
6. окна внешних станций - только в тестовом окне; внешние станции не встречаются в
   основном наборе под другой ролью, в записи о выборе чекпойнта и в метаданных
   конформной таблицы; самый поздний час обучающих окон основного набора раньше
   самого раннего часа, который читают тестовые окна внешнего набора.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np

from mayak.constants import H
from mayak.data.splits import (BLOCK_KEYS, HISTORY_FORBIDDEN, MIN_BLOCK_HOURS, MIN_GAP_HOURS,
                               ROLE_EXTERNAL, ROLE_TRAIN, ROLE_VAL, TIME_KEYS,
                               TIME_LAYOUT, layout_fingerprint, time_layout)
from mayak.timeaxis import from_utc_hour

log = logging.getLogger(__name__)

SELECTION_KEY = "mayak_selection"          # ключ записи о выборе чекпойнта в .ckpt
SELECTION_TIME_KEY, CONFORMAL_TIME_KEY = "val", "calib"
ALLOWED_ROLES = {"train": {ROLE_TRAIN}, "calib": {ROLE_VAL}}
EXTERNAL_TIME_KEYS = {"test"}


class LeakageError(RuntimeError):
    """Нарушение чек-листа антиутечек."""


def _fail(msg):
    raise LeakageError(msg)


def _split_state():
    """Правила сплитов, при которых построен артефакт.

    Совпадение правил проверяется по параметрам раскладки и отпечатку её кода.

    Returns:
        Словарь для записи рядом с артефактом.
    """
    return dict(time_layout=dict(TIME_LAYOUT), layout_code=layout_fingerprint())


def _check_split_state(rec, what):
    """Проверяет, что артефакт построен при текущих правилах сплитов.

    Args:
        rec: запись о правилах из артефакта.
        what: название артефакта для сообщения.

    Raises:
        LeakageError: параметры раскладки или её код отличаются от текущих.
    """
    diff = []
    if rec.get("time_layout") != dict(TIME_LAYOUT):
        diff.append(f"параметры раскладки {rec.get('time_layout')}, сейчас {dict(TIME_LAYOUT)}")
    if rec.get("layout_code") != layout_fingerprint():
        diff.append(f"код раскладки {rec.get('layout_code')}, сейчас {layout_fingerprint()}")
    if diff:
        _fail(f"{what}: построено при других правилах сплитов: {'; '.join(diff)}; пересоберите")


def _check_station_roles(stations, store, role, what):
    bad = {sid: store.stations[sid]["role"] if sid in store.stations else None
           for sid in stations
           if sid not in store.stations or store.stations[sid]["role"] != role}
    if bad:
        _fail(f"{what}: станции не из роли {role}: {dict(list(bad.items())[:5])}")
    if not stations:
        _fail(f"{what}: пустой список станций")


def check_time_layout(n_hours):
    """Проверяет раскладку ряда заданной длины, не полагаясь на то, как она построена.

    Args:
        n_hours: длина ряда, ч.

    Returns:
        Проверенная раскладка.

    Raises:
        LeakageError: блок выходит за ряд, пуст или короче одного окна; блоки
            валидации и калибровки не идут подряд или не чередуются; год валидации и
            калибровки ближе допустимого к обучению или тесту.
    """
    lay = time_layout(n_hours)
    for k in TIME_KEYS:
        if not lay.blocks[k]:
            _fail(f"у окна {k} нет блоков")
        for lo, hi in lay.blocks[k]:
            if lo < 0 or hi > n_hours:
                _fail(f"блок {k} [{lo}, {hi}) выходит за ряд длины {n_hours}")
            if hi - lo < MIN_BLOCK_HOURS:
                _fail(f"блок {k} [{lo}, {hi}) короче одного окна ({MIN_BLOCK_HOURS} ч)")
    inner = sorted((lo, hi, k) for k in BLOCK_KEYS for lo, hi in lay.blocks[k])
    if len(lay.blocks["val"]) != len(lay.blocks["calib"]):
        _fail(f"блоков валидации {len(lay.blocks['val'])}, калибровки "
              f"{len(lay.blocks['calib'])}: должно быть поровну")
    for i, (lo, hi, k) in enumerate(inner):
        if k != BLOCK_KEYS[i % 2]:
            _fail(f"блоки валидации и калибровки не чередуются: блок {i} [{lo}, {hi}) - {k}")
        if i and lo != inner[i - 1][1]:
            _fail(f"между блоками {i - 1} и {i} валидации и калибровки разрыв или наложение")
    for name, gap in (("обучением и валидацией", inner[0][0] - lay.span("train")[1]),
                      ("калибровкой и тестом", lay.span("test")[0] - inner[-1][1])):
        if gap < MIN_GAP_HOURS:
            _fail(f"зазор между {name} {gap} ч меньше горизонта и полной истории вместе "
                  f"({MIN_GAP_HOURS} ч)")
    return lay


def check_climatology(store, deep=False):
    """Окно подгонки климатологии каждой станции совпадает с её обучающим окном.

    Глубокая проверка дополнительно переподгоняет климатологию по обучающему окну и
    сверяет коэффициенты: ловит кэш, в котором записанное окно не соответствует
    подгонке.

    Args:
        store: хранилище станций.
        deep: делать ли глубокую проверку.

    Raises:
        LeakageError: окно подгонки или коэффициенты не совпадают.
    """
    for sid, s in store.stations.items():
        want = tuple(time_layout(s["N"]).span("train"))
        got = tuple(s.get("clim_fit") or ())
        if got != want:
            _fail(f"климатология {sid}: подогнана по окну {got or 'неизвестно'}, "
                  f"а обучающее окно {want}")
        if deep:
            from mayak.data.store import CLIM_PARAMS, new_climatology
            from mayak.timeaxis import window_calendar
            lo, hi = want
            d, h = window_calendar(s["t0"], np.arange(lo, hi))
            ref = new_climatology().fit(
                d.astype(np.float64), h.astype(np.float64), s["x"][lo:hi, 0],
                s["mask"][lo:hi, 0], min_valid=CLIM_PARAMS["min_valid"])
            got = s["clim"]
            same = (np.allclose(ref.beta, got.beta, rtol=1e-6, atol=1e-6)
                    and np.allclose(ref.scale_beta, got.scale_beta, rtol=1e-6, atol=1e-6))
            if not same:
                _fail(f"климатология {sid}: коэффициенты среднего или масштаба не "
                      f"воспроизводятся по обучающему окну {want}")


def check_windows(datasets, store=None):
    """Проверяет, что окна датасетов не нарушают правил раскладки и ролей.

    Args:
        datasets: датасеты окон.
        store: набор станций; если задан, проверяются ещё и роли станций.

    Returns:
        Число проверенных окон.

    Raises:
        LeakageError: цель окна не лежит целиком в одном блоке своего окна, длина цели
            не равна горизонту, история заходит в запретное окно или выходит за ряд,
            либо станция чужой роли читается в обучении, калибровке или вне теста.
    """
    n = 0
    for ds in datasets:
        name = type(ds).__name__
        if not hasattr(ds, "footprints"):
            _fail(f"{name}: не умеет перечислять свои окна (нет footprints())")
        for fp in ds.footprints():
            key = fp["time_key"]
            if key not in TIME_KEYS:
                _fail(f"{name}: неизвестное временное окно {key!r}")
            if "t" not in fp:
                _fail(f"{name}: окна станции {fp['sid']} без начала горизонта (нет поля t)")
            lay = time_layout(fp["N"])
            flo, ft, fhi = (np.asarray(fp[k], np.int64) for k in ("lo", "t", "hi"))
            _check_window_arrays(name, fp["sid"], key, lay, flo, ft, fhi)
            if store is not None:
                role = store.stations[fp["sid"]]["role"]
                if key in ALLOWED_ROLES and role not in ALLOWED_ROLES[key]:
                    _fail(f"{name}: станция {fp['sid']} роли {role} в окне {key} "
                          f"(допустимы {sorted(ALLOWED_ROLES[key])})")
                if role == ROLE_EXTERNAL and key not in EXTERNAL_TIME_KEYS:
                    _fail(f"{name}: станция внешнего теста {fp['sid']} в окне {key}; "
                          f"внешние станции читаются только в {sorted(EXTERNAL_TIME_KEYS)}")
            n += flo.size
    return n


def _check_window_arrays(name, sid, key, lay, lo, t, hi):
    def first(bad):
        i = int(np.flatnonzero(bad)[0])
        return f"[{int(lo[i])}, {int(t[i])}, {int(hi[i])}) (всего {int(bad.sum())} окон)"

    bad = hi - t != H
    if bad.any():
        _fail(f"{name}: окно станции {sid} {first(bad)}: длина цели не равна горизонту {H} ч")
    bad = (lo > t) | (lo < 0) | (hi > lay.n_hours)
    if bad.any():
        _fail(f"{name}: окно станции {sid} {first(bad)} выходит за ряд или история после цели")
    bad = lay.block_index(key, t, hi) < 0
    if bad.any():
        _fail(f"{name}: цель окна станции {sid} {first(bad)} не лежит целиком в одном "
              f"блоке окна {key}")
    for other in HISTORY_FORBIDDEN[key]:
        bad = lay.overlaps((other,), lo, t)
        if bad.any():
            _fail(f"{name}: история окна {key} станции {sid} {first(bad)} заходит в окно "
                  f"{other}")


def selection_record(val_ds, monitor, scores=None):
    """Запись о том, на чём выбирался чекпойнт. Кладётся в .ckpt под SELECTION_KEY.

    Args:
        val_ds: набор окон валидации.
        monitor: имя метрики выбора.
        scores: валидационные числа на момент сохранения; None - не писать.

    Returns:
        Словарь: метрика, роль станций, временное окно, станции, правила сплитов,
        правило длины истории, число окон и отпечаток набора.
    """
    stations = sorted({fp["sid"] for fp in val_ds.footprints()})
    roles = tuple(val_ds.station_splits)
    rec = dict(monitor=monitor, station_role=roles[0] if len(roles) == 1 else list(roles),
               time_key=val_ds.time_key, stations=stations, **_split_state(),
               history=val_ds.history_spec(), windows=len(val_ds),
               windows_digest=val_ds.fingerprint())
    if scores:
        rec["scores"] = dict(sorted(scores.items()))
    return rec


def check_selection_record(rec, store, what="чекпойнт"):
    if not rec:
        _fail(f"{what}: нет записи о выборе ({SELECTION_KEY}) — неизвестно, на каких "
              f"данных выбран чекпойнт")
    if not str(rec.get("monitor") or "").startswith("val/"):
        _fail(f"{what}: выбран по метрике {rec.get('monitor')!r}, а не по val/*")
    if rec.get("station_role") != ROLE_VAL or rec.get("time_key") != SELECTION_TIME_KEY:
        _fail(f"{what}: выбран на станциях {rec.get('station_role')!r} в окне "
              f"{rec.get('time_key')!r}; нужно {ROLE_VAL!r} / {SELECTION_TIME_KEY!r}")
    _check_split_state(rec, what)
    _check_station_roles(rec.get("stations", []), store, ROLE_VAL, what)
    history = rec.get("history") or {}
    if history.get("curriculum") is None:
        _fail(f"{what}: выбран на окнах с одной длиной истории {history.get('L')!r}; длины "
              f"истории окон валидации должны следовать куррикулуму этапа")


def check_checkpoint(path, store):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    check_selection_record(ckpt.get(SELECTION_KEY), store, what=f"чекпойнт {path}")


def conformal_meta_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def file_digest(path):
    """Отпечаток содержимого файла.

    Args:
        path: путь к файлу.

    Returns:
        Первые 16 шестнадцатеричных знаков SHA-256 содержимого.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def history_bins_record(history_bins=None):
    """Бины длины истории таблицы в том виде, в каком они пишутся в запись о подгонке.

    Args:
        history_bins: бины длины истории; None - бины из кода.

    Returns:
        Список пар границ, ч.
    """
    from mayak.constants import HISTORY_BINS
    return [[int(b[0]), int(b[1])] for b in (history_bins or HISTORY_BINS)]


def conformal_record(ds, checkpoint=None, history_fit=None):
    """Запись о том, на чём подогнана конформная таблица.

    Args:
        ds: калибровочный набор окон.
        checkpoint: путь к чекпойнту модели; если файл есть, пишется и его отпечаток.
        history_fit: сведения о подгонке по бинам длины истории: подпись и границы бина,
            число окон и признак маргинальной строки.

    Returns:
        Словарь для записи рядом с таблицей.
    """
    stations = sorted({fp["sid"] for fp in ds.footprints()})
    digest = file_digest(checkpoint) if checkpoint and os.path.isfile(checkpoint) else None
    return dict(station_roles=sorted(ds.station_splits), time_key=ds.time_key,
                stations=stations, checkpoint=checkpoint, checkpoint_digest=digest,
                history=ds.history_spec(),
                history_bins=history_bins_record(), history_fit=history_fit,
                windows=len(ds), windows_digest=ds.fingerprint(), **_split_state())


def save_conformal(path, shift, record):
    """Сохраняет таблицу поправок и рядом с ней запись о подгонке.

    Args:
        path: путь к таблице, файл numpy.
        shift: таблица поправок по бинам лидов и длины истории.
        record: запись о подгонке; ложится в файл с тем же именем и суффиксом meta.json.

    Raises:
        ValueError: таблица другой формы или сдвигает медиану.
    """
    from mayak.metrics import check_conformal_shape
    check_conformal_shape(shift)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, shift)
    with open(conformal_meta_path(path), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)


def load_conformal(path):
    """Таблица поправок вместе с записью о подгонке.

    Args:
        path: путь к таблице.

    Returns:
        Пара: таблица float32 и запись о подгонке.

    Raises:
        LeakageError: рядом с таблицей нет записи о подгонке.
        ValueError: таблица другой формы или сдвигает медиану.
    """
    from mayak.metrics import check_conformal_shape
    meta = conformal_meta_path(path)
    if not os.path.exists(meta):
        _fail(f"конформная таблица {path}: нет метаданных {meta} — неизвестно, на каких "
              f"данных и по какому чекпойнту она подогнана")
    with open(meta) as f:
        rec = json.load(f)
    shift = np.load(path).astype(np.float32)
    try:
        check_conformal_shape(shift)
    except ValueError as e:
        raise ValueError(f"конформная таблица {path}: {e}") from None
    return shift, rec


def check_conformal(path, store):
    what = f"конформная таблица {path}"
    try:
        _shift, rec = load_conformal(path)
    except ValueError as e:
        _fail(f"{what}: {e}")
    if rec.get("station_roles") != [ROLE_VAL] or rec.get("time_key") != CONFORMAL_TIME_KEY:
        _fail(f"{what}: подогнана на ролях {rec.get('station_roles')} в окне "
              f"{rec.get('time_key')!r}; нужно [{ROLE_VAL!r}] / {CONFORMAL_TIME_KEY!r}")
    _check_split_state(rec, what)
    _check_station_roles(rec.get("stations", []), store, ROLE_VAL, what)
    if (rec.get("history") or {}).get("curriculum") is None:
        _fail(f"{what}: подогнана на окнах с одной длиной истории "
              f"{(rec.get('history') or {}).get('L')!r}; длины истории калибровочных окон "
              f"должны следовать куррикулуму")
    if rec.get("history_bins") != history_bins_record():
        _fail(f"{what}: подогнана по бинам длины истории {rec.get('history_bins')}, в коде "
              f"{history_bins_record()}; подгоните таблицу заново")
    fit = rec.get("history_fit")
    if not isinstance(fit, list) or len(fit) != len(history_bins_record()):
        _fail(f"{what}: в записи о подгонке нет сведений по бинам длины истории "
              f"(history_fit); подгоните таблицу заново")


def train_last_hour(store):
    """Самый поздний час, который читают обучающие окна основного набора.

    Обучающие окна берутся только со станций с ролью обучения и целиком лежат в их
    обучающем временном окне, поэтому самый поздний час - последний час этого окна.

    Args:
        store: основной набор.

    Returns:
        Час от эпохи UTC или None, если в наборе нет обучающих станций.
    """
    ends = [int(s["t0"]) + time_layout(s["N"]).span("train")[1] - 1
            for s in store.by_role(ROLE_TRAIN)]
    return max(ends) if ends else None


def station_test_first_hour(t0, n_hours):
    """Самый ранний час, который читают тестовые окна одной станции.

    Считается по самому раннему окну: горизонт с первого часа тестового окна, полная
    история и часы перед ней, которые смотрит причинный QC.

    Args:
        t0: абсолютный час первой строки ряда станции.
        n_hours: длина ряда, ч.

    Returns:
        Час от эпохи UTC.

    Raises:
        ValueError: ряд слишком короток для раскладки.
    """
    from mayak.data.dataset import footprint
    lay = time_layout(n_hours)
    lo, _hi = footprint(lay.span("test")[0], None, lay.history_floor("test"))
    return int(t0) + int(lo)


def external_first_hour(external_store):
    """Самый ранний час, который читают тестовые окна внешнего набора.

    Args:
        external_store: набор внешнего теста.

    Returns:
        Час от эпохи UTC или None, если набор пуст.
    """
    starts = [station_test_first_hour(s["t0"], s["N"])
              for s in external_store.stations.values()]
    return min(starts) if starts else None


def _utc_text(hour):
    return str(from_utc_hour(hour)) + ":00 UTC"


def check_external_calendar(store, external_store):
    """Тестовые окна внешнего набора не пересекаются по календарю с обучением.

    Иначе модель проверялась бы на тех же днях погоды, по которым училась в соседних
    точках реанализа. Сравниваются абсолютные часы, а не индексы рядов: ряды станций
    двух наборов начинаются в разные дни.

    Args:
        store: основной набор.
        external_store: набор внешнего теста.

    Returns:
        Словарь с последним часом обучения и первым часом внешнего теста, текстом;
        None, если в основном наборе нет обучающих станций и сравнивать не с чем.

    Raises:
        LeakageError: обучение заканчивается не раньше начала внешнего теста.
    """
    last, first = train_last_hour(store), external_first_hour(external_store)
    if last is None or first is None:
        return None
    if last >= first:
        _fail(f"внешний тест пересекается с обучением по календарю: обучающие окна "
              f"основного набора читают часы до {_utc_text(last)}, а тестовые окна внешнего "
              f"начинаются с {_utc_text(first)}; скачайте внешний набор за более поздний "
              f"период или сдвиньте период обучения раньше")
    return dict(train_last=_utc_text(last), external_first=_utc_text(first),
                margin_hours=int(first - last - 1))


def check_external(store, external_store, checkpoints=(), conformal=None):
    """Внешний тест изолирован от обучения, выбора чекпойнта и калибровки.

    Близость внешней станции к обучающей точке ошибкой не считается: она показывается
    отдельным разрезом оценки, а здесь попадает только в сводку.

    Args:
        store: основной набор: обучение, валидация, калибровка, внутренний тест.
        external_store: набор внешнего теста.
        checkpoints: пути чекпойнтов, записи о выборе которых проверяются.
        conformal: путь конформной таблицы, метаданные которой проверяются; None значит
            без таблицы.

    Returns:
        Сводка: число внешних станций, чекпойнтов, наличие таблицы, календарный запас
        между обучением и внешним тестом, число внешних станций по бинам расстояния до
        обучающей точки и медиана этого расстояния.

    Raises:
        LeakageError: набор пуст; станция не из роли внешнего теста или встречается в
            основном наборе под другой ролью; внешняя станция участвовала в выборе
            чекпойнта или в подгонке таблицы; тест пересекается с обучением по календарю.
    """
    from mayak.external import nearest_train_km, train_distance_label
    ext = external_store.stations
    if not ext:
        _fail("внешний тест: пустой набор станций")
    wrong = {sid: s["role"] for sid, s in ext.items() if s["role"] != ROLE_EXTERNAL}
    if wrong:
        _fail(f"внешний тест: станции не из роли {ROLE_EXTERNAL}: {dict(list(wrong.items())[:5])}")
    if external_store is not store:
        clash = {sid: store.stations[sid]["role"] for sid in ext
                 if sid in store.stations and store.stations[sid]["role"] != ROLE_EXTERNAL}
        if clash:
            _fail(f"внешний тест: станции есть в основном наборе под другой ролью: "
                  f"{dict(list(clash.items())[:5])}")
    ext_ids = set(ext)
    import torch
    for c in checkpoints:
        rec = torch.load(c, map_location="cpu", weights_only=False).get(SELECTION_KEY) or {}
        hit = sorted(ext_ids & set(rec.get("stations", [])))
        if hit:
            _fail(f"чекпойнт {c}: выбран с участием станций внешнего теста {hit[:5]}")
    if conformal:
        with open(conformal_meta_path(conformal)) as f:
            rec = json.load(f)
        hit = sorted(ext_ids & set(rec.get("stations", [])))
        if hit or ROLE_EXTERNAL in rec.get("station_roles", []):
            _fail(f"конформная таблица {conformal}: подогнана с участием внешнего теста {hit[:5]}")
    calendar = check_external_calendar(store, external_store)
    dist = nearest_train_km(store, external_store)
    by_bin = {}
    for km in dist.values():
        name = train_distance_label(km)
        by_bin[name] = by_bin.get(name, 0) + 1
    summary = dict(external_stations=len(ext), checkpoints=len(checkpoints),
                   conformal=bool(conformal), calendar=calendar, train_distance=by_bin,
                   median_nearest_train_km=float(np.median(list(dist.values()))) if dist else None)
    log.info("внешний тест изолирован: %s", summary)
    return summary


def run_checklist(store, datasets=(), conformal=None, checkpoints=(), deep=False,
                  external_store=None):
    for s in store.stations.values():
        check_time_layout(s["N"])
    check_climatology(store, deep=deep)
    n_windows = check_windows(datasets, store)
    if conformal:
        check_conformal(conformal, store)
    for c in checkpoints:
        check_checkpoint(c, store)
    summary = dict(stations=len(store.stations), windows=n_windows,
                   conformal=bool(conformal), checkpoints=len(checkpoints))
    if external_store is not None:
        summary["external"] = check_external(store, external_store, checkpoints=checkpoints,
                                             conformal=conformal)
    log.info("чек-лист антиутечек пройден: %s", summary)
    return summary
