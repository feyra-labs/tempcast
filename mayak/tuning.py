"""Подбор скорости обучения и запись о нём.

Этап 1 сравнения моделей. Каждая нейросеть, включая МАЯК, проходит одну и ту же сетку
скоростей обучения протокола (``lr_grid``). На каждом значении идёт прогон с полными
этапами протокола, кроме последнего, и последним этапом, укороченным до
``lr_search_steps`` шагов со своим расписанием WSD, которое до начала спада совпадает с
расписанием полного этапа. Внутри прогона сетки каждый этап стартует с лучшего по
метрике выбора чекпойнта предыдущего. Подбор отмечает значение с наименьшей метрикой
выбора на последнем этапе (``selected_lr``, при равенстве - меньшее) и то, лежит ли оно
на краю сетки (``edge``), и пишет запись в журнал каталога прогона.

Дальше правило одно для всех моделей, без решений человека
(``mayak.protocol.run_protocol``):

* при ``edge: true`` запуск завершается ``ProtocolError``: сетку расширяют и подбор
  повторяют для всех моделей;
* иначе в том же запуске идёт последний этап полной длины со скоростью ``selected_lr``
  со старта с лучшего чекпойнта предпоследнего этапа прогона сетки с этой скоростью.

Абляции, повторы основной модели с другими сидами и прогон без аугментаций берут
скорость обучения и запись о подборе из журнала прогона основной модели
(``run.lr_from``), подбор не повторяют и проходят все этапы протокола подряд. Запись с
``edge: true`` не наследуется.

Этап 2 сравнения - необязательная дополнительная настройка МАЯК любыми
гиперпараметрами по валидации. Её прогон помечается в записи, лежит в своём каталоге и в
сравнение на равных не входит.

Запись о подборе пишется в журнал прогона и в каждый его чекпойнт. Стенд оценки
сравнивает модели только тогда, когда у каждой есть запись о подборе этапа 1 с той же
сеткой, тем же числом шагов этапов и той же метрикой выбора.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, fields, replace
from typing import Optional

from mayak.protocol import JOURNAL, LR_FIELDS, SEED_FIELDS, Protocol, ProtocolError, protocol_diff
from mayak.stages import jsonable

log = logging.getLogger(__name__)

TUNING_KEY = "mayak_tuning"
SEARCH_DIR = "lr_search"
PHASE_EQUAL, PHASE_EXTRA = 1, 2
EXTRA_SUFFIX = "-tuned"
EXTRA_ARCH = "mayak"
GRID_CONFIG = "conf/train/default.yaml"


@dataclass(frozen=True)
class Tuning:
    """Как прогон получает скорость обучения. В протокол не входит.

    Attributes:
        lr_search: пройти сетку скоростей обучения протокола, записать подбор в журнал
            каталога прогона и обучить последний этап полной длины с выбранной скоростью.
        lr_from: каталог прогона основной модели или его журнал: оттуда берутся
            скорость обучения и запись о подборе, подбор не повторяется.
        extra_tuning: прогон дополнительной настройки МАЯК, этап 2 сравнения.
    """
    lr_search: bool = False
    lr_from: Optional[str] = None
    extra_tuning: bool = False

    def __post_init__(self):
        for name in ("lr_search", "extra_tuning"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} должен быть bool, получено {getattr(self, name)!r}")
        if self.lr_from is not None:
            path = str(self.lr_from).strip()
            object.__setattr__(self, "lr_from", path or None)
        if self.lr_search and self.lr_from:
            raise ValueError("подбор скорости обучения и готовая скорость из другого прогона "
                             "(lr_from) взаимоисключающие")
        if self.extra_tuning and self.lr_from:
            raise ValueError("дополнительная настройка (этап 2) не наследует подбор этапа 1: "
                             "задайте скорость обучения явно или подберите её")

    @classmethod
    def coerce(cls, value):
        """Настройки подбора из None, словаря или готового объекта.

        Args:
            value: None, словарь с полями настроек или сами настройки.

        Returns:
            Настройки подбора.

        Raises:
            ValueError: в словаре есть неизвестный ключ.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        d = dict(value)
        known = {f.name for f in fields(cls)}
        extra = sorted(set(d) - known)
        if extra:
            raise ValueError(f"неизвестные настройки подбора {extra}; есть {sorted(known)}")
        return cls(**d)

    @property
    def active(self):
        """Пишется ли запись о подборе."""
        return bool(self.lr_search or self.lr_from or self.extra_tuning)

    @property
    def phase(self):
        """Этап сравнения прогона."""
        return PHASE_EXTRA if self.extra_tuning else PHASE_EQUAL


TUNING_FIELDS = tuple(f.name for f in fields(Tuning))


def tuning_from_config(section):
    """Настройки подбора из секции конфига, где рядом лежат и другие ключи.

    Args:
        section: словарь секции; берутся только ключи настроек подбора.

    Returns:
        Настройки подбора.
    """
    section = dict(section or {})
    return Tuning.coerce({k: section[k] for k in TUNING_FIELDS if k in section})


def search_protocol(protocol, lr):
    """Протокол прогона на одном значении сетки.

    Args:
        protocol: протокол полного прогона.
        lr: скорость обучения.

    Returns:
        Протокол с этой скоростью обучения и последним этапом на ``lr_search_steps``
        шагов.
    """
    last = protocol.stages[-1]
    stages = (*protocol.stages[:-1], replace(last, steps=int(protocol.lr_search_steps)))
    return replace(protocol, lr=float(lr), stages=stages)


def check_search(protocol):
    """Проверяет, что подбор скорости обучения можно провести по этому протоколу.

    Args:
        protocol: протокол полного прогона.

    Raises:
        ProtocolError: в протоколе один этап, и последнему этапу не с чего стартовать;
            последний этап при подборе не короче полного.
    """
    if len(protocol.stages) < 2:
        raise ProtocolError("подбор скорости обучения: в протоколе один этап, а последний "
                            "этап после подбора стартует с чекпойнта предыдущего этапа "
                            "прогона сетки")
    last = protocol.stages[-1]
    if protocol.lr_search_steps > last.steps:
        raise ProtocolError(f"подбор скорости обучения: этап {last.name} на "
                            f"{protocol.lr_search_steps} шагов длиннее полного ({last.steps})")


def lr_label(lr):
    """Имя подкаталога прогона на одном значении сетки.

    Args:
        lr: скорость обучения.

    Returns:
        Строка вида lr0.001.
    """
    return f"lr{float(lr):g}"


def select_lr(results, stage):
    """Значение сетки с наименьшей метрикой выбора на этапе.

    Args:
        results: записи по значениям сетки со скоростью обучения и метрикой по этапам.
        stage: имя этапа, по которому идёт выбор.

    Returns:
        Скорость обучения; при равенстве метрики - меньшая.

    Raises:
        ProtocolError: ни одно значение не дало конечной метрики.
    """
    scored = [(r["val_loss"].get(stage), r["lr"]) for r in results]
    finite = [(float(v), float(lr)) for v, lr in scored if v is not None and math.isfinite(v)]
    if not finite:
        raise ProtocolError(f"подбор скорости обучения: ни одно значение сетки не дало "
                            f"конечной метрики выбора на этапе {stage}")
    return min(finite)[1]


def run_lr_search(protocol, train, tag):
    """Проходит сетку скоростей обучения протокола.

    Args:
        protocol: протокол полного прогона.
        train: функция от протокола и имени каталога прогона, которая обучает модель и
            возвращает журнал прогона и путь к нему.
        tag: имя каталога полного прогона; прогоны сетки лежат в его подкаталоге.

    Returns:
        Запись о подборе: сетка, число шагов этапов, метрика выбора, метрика выбора и
        лучший чекпойнт по этапам на каждом значении, пути к журналам, значение с
        наименьшей метрикой (``selected_lr``) и лежит ли оно на краю сетки (``edge``).
    """
    results = []
    for lr in protocol.lr_grid:
        p = search_protocol(protocol, lr)
        log.info("подбор скорости обучения: lr %g (%d из %d)", lr, len(results) + 1,
                 len(protocol.lr_grid))
        journal, path = train(p, f"{tag}/{SEARCH_DIR}/{lr_label(lr)}")
        results.append(dict(lr=float(lr), journal=path,
                            best_ckpts={s["name"]: s["best_ckpt"] for s in journal["stages"]},
                            val_loss={s["name"]: s["best_score"] for s in journal["stages"]}))
    short = search_protocol(protocol, protocol.lr_grid[0])
    selected = select_lr(results, short.stages[-1].name)
    grid = list(protocol.lr_grid)
    return jsonable(dict(grid=grid, stage_steps={s.name: s.steps for s in short.stages},
                         monitor=protocol.monitor, results=results, selected_lr=selected,
                         edge=selected in (grid[0], grid[-1])))


def edge_error(search):
    """Текст ошибки для подбора с минимумом на краю сетки.

    Args:
        search: запись о подборе по сетке.

    Returns:
        Строка с правилом: расширить сетку и повторить подбор всех моделей.
    """
    return (f"подбор скорости обучения: минимум {search['monitor']} на краю сетки "
            f"{search['grid']} (lr {float(search['selected_lr']):g}), оптимум может лежать "
            f"за её пределами. Правило: расширьте train.lr_grid в {GRID_CONFIG} и повторите "
            f"подбор всех моделей")


def search_start(search, stage_name):
    """Чекпойнт, с которого после подбора стартует последний этап.

    Args:
        search: запись о подборе по сетке.
        stage_name: имя предпоследнего этапа протокола.

    Returns:
        Пара: лучший по метрике выбора чекпойнт этапа ``stage_name`` прогона сетки со
        скоростью ``selected_lr`` и путь к журналу этого прогона.
    """
    lr = float(search["selected_lr"])
    row = next(r for r in search["results"] if float(r["lr"]) == lr)
    return row["best_ckpts"][stage_name], row["journal"]


def read_source_journal(path):
    """Журнал прогона по пути к каталогу прогона или к самому журналу.

    Args:
        path: каталог прогона или путь к protocol.json.

    Returns:
        Пара: журнал и абсолютный путь к нему.

    Raises:
        ProtocolError: журнала нет.
    """
    jp = path if os.path.isfile(path) else os.path.join(path, JOURNAL)
    if not os.path.isfile(jp):
        raise ProtocolError(f"журнал прогона {jp} не найден")
    with open(jp) as f:
        return json.load(f), os.path.abspath(jp)


def inherit_search(path, arch, protocol):
    """Запись о подборе этапа 1 из журнала прогона основной модели.

    Args:
        path: каталог прогона основной модели или его журнал.
        arch: архитектура этого прогона.
        protocol: протокол этого прогона.

    Returns:
        Пара: запись о подборе и абсолютный путь к журналу, из которого она взята.

    Raises:
        ProtocolError: журнала нет; в нём нет своего подбора этапа 1; минимум подбора на
            краю сетки; архитектура другая; протоколы различаются не только скоростью
            обучения и сидами.
    """
    journal, jp = read_source_journal(path)
    rec = journal.get("tuning") or {}
    if (rec.get("phase") != PHASE_EQUAL or not rec.get("lr_search")
            or rec.get("inherited_from")):
        raise ProtocolError(f"{jp}: в прогоне нет своего подбора скорости обучения этапа 1; "
                            f"укажите прогон основной модели, запущенный с подбором")
    if rec["lr_search"].get("edge"):
        raise ProtocolError(f"{jp}: скорость обучения не наследуется, "
                            f"{edge_error(rec['lr_search'])}")
    if journal.get("arch") != arch:
        raise ProtocolError(f"{jp}: подбор шёл для архитектуры {journal.get('arch')!r}, "
                            f"этот прогон - {arch!r}")
    diff = protocol_diff(Protocol.from_dict(journal["protocol"]), protocol,
                         ignore=(*LR_FIELDS, *SEED_FIELDS))
    if diff:
        raise ProtocolError(f"{jp}: протокол прогона с подбором отличается от этого: {diff}")
    return rec["lr_search"], jp


def tuning_record(tuning, lr_search=None, inherited_from=None):
    """Запись о подборе для журнала и чекпойнтов.

    Args:
        tuning: настройки подбора.
        lr_search: запись о подборе по сетке; None, если подбора не было.
        inherited_from: журнал, из которого взят подбор; None, если подбор свой.

    Returns:
        Словарь: этап сравнения, запись о подборе и её источник.
    """
    return jsonable(dict(phase=tuning.phase, lr_search=lr_search, inherited_from=inherited_from))


def check_run_dir(journal_path, tuning):
    """Отказывает, если каталог прогона занят прогоном другого этапа сравнения.

    Прогон дополнительной настройки МАЯК и прогоны этапа 1 лежат в разных каталогах.

    Args:
        journal_path: путь к журналу в каталоге прогона.
        tuning: настройки подбора этого прогона.

    Raises:
        ProtocolError: в каталоге лежит журнал прогона другого этапа сравнения.
    """
    if not os.path.isfile(journal_path):
        return
    with open(journal_path) as f:
        old = (json.load(f).get("tuning") or {}).get("phase")
    new = tuning.phase if tuning.active else None
    if PHASE_EXTRA in (old, new) and old != new:
        raise ProtocolError(f"каталог {os.path.dirname(journal_path)} занят прогоном "
                            f"{describe_phase(old)}, этот прогон - {describe_phase(new)}: "
                            f"они лежат в разных каталогах")


def describe_phase(phase):
    """Название этапа сравнения для сообщений.

    Args:
        phase: этап сравнения или None, если записи о подборе нет.

    Returns:
        Строка.
    """
    if phase == PHASE_EXTRA:
        return "этапа 2 сравнения (доп. настройка МАЯК)"
    if phase == PHASE_EQUAL:
        return "этапа 1 сравнения"
    return "без записи о подборе"


def resolve(arch, protocol, tuning, model_config, tag, train):
    """Протокол прогона и запись о подборе.

    Запись о подборе берётся из своего подбора по сетке (``run.lr_search``) или из
    журнала прогона основной модели (``run.lr_from``). Скорость обучения прогона с
    записью о подборе - ``selected_lr``.

    Args:
        arch: архитектура.
        protocol: протокол прогона.
        tuning: настройки подбора.
        model_config: конфиг архитектуры.
        tag: имя каталога прогона.
        train: функция обучения прогона на одном значении сетки, как в run_lr_search.

    Returns:
        Пара: протокол со скоростью обучения прогона и запись о подборе (None, если её
        нет).

    Raises:
        ProtocolError: дополнительная настройка не для МАЯК; подбор для абляции;
            подбор невозможен по этому протоколу; источник подбора не подходит.
    """
    if tuning.extra_tuning and arch != EXTRA_ARCH:
        raise ProtocolError(f"дополнительная настройка (этап 2) только для {EXTRA_ARCH}")
    abl = getattr(model_config, "ablations", None)
    if tuning.lr_search and abl is not None and abl.active() and not tuning.extra_tuning:
        raise ProtocolError("абляции берут скорость обучения основного МАЯК: вместо подбора "
                            "укажите его прогон (run.lr_from)")
    search = inherited = None
    if tuning.lr_search:
        check_search(protocol)
        search = run_lr_search(protocol, train, tag)
    elif tuning.lr_from:
        search, inherited = inherit_search(tuning.lr_from, arch, protocol)
    if search is None:
        return protocol, (tuning_record(tuning) if tuning.active else None)
    protocol = replace(protocol, lr=float(search["selected_lr"]))
    return protocol, tuning_record(tuning, search, inherited)


def equal_terms_problems(record, protocol):
    """Почему чекпойнт не входит в сравнение на равных.

    Args:
        record: запись о подборе из чекпойнта или None.
        protocol: протокол чекпойнта.

    Returns:
        Список причин; пустой, если чекпойнт входит в сравнение.
    """
    if not record:
        return ["нет записи о подборе скорости обучения: прогон запущен без run.lr_search "
                "и без run.lr_from"]
    if record.get("phase") != PHASE_EQUAL:
        return [f"прогон {describe_phase(record.get('phase'))} в сравнение на равных не "
                f"входит: для него --tuned-ckpt"]
    search = record.get("lr_search")
    if not search:
        return ["нет подбора скорости обучения по сетке"]
    lr = float(protocol.lr)
    if lr not in [float(v) for v in search.get("grid") or ()]:
        return [f"скорость обучения {lr:g} не из сетки подбора {search.get('grid')}"]
    target = search.get("selected_lr", math.nan)
    if float(target) != lr:
        return [f"скорость обучения {lr:g} не равна выбранной подбором {target}"]
    return []


def search_terms(record):
    """Условия подбора, которые у сравниваемых моделей должны совпадать.

    Args:
        record: запись о подборе или None.

    Returns:
        Словарь: сетка, число шагов этапов и метрика выбора.
    """
    search = (record or {}).get("lr_search") or {}
    return {k: search.get(k) for k in ("grid", "stage_steps", "monitor")}


def format_tuning(record):
    """Сводка подбора для печати.

    Args:
        record: запись о подборе или None.

    Returns:
        Список строк; пустой, если записи нет.
    """
    if not record:
        return []
    lines = [f"подбор скорости обучения, {describe_phase(record.get('phase'))}:"]
    if record.get("inherited_from"):
        lines.append(f"  взят из {record['inherited_from']}")
    search = record.get("lr_search")
    if not search:
        lines.append("  без подбора по сетке")
        return lines
    steps = ", ".join(f"{k} {v}" for k, v in search["stage_steps"].items())
    lines.append(f"  сетка {search['grid']}, шаги этапов: {steps}, выбор по {search['monitor']}")
    for r in search["results"]:
        cells = ", ".join(f"{k} {'—' if v is None else f'{v:.5f}'}"
                          for k, v in r["val_loss"].items())
        mark = "  <- минимум" if r["lr"] == search.get("selected_lr") else ""
        lines.append(f"  lr {r['lr']:g}: {cells}{mark}")
    if search.get("edge"):
        lines.append(f"  ВНИМАНИЕ: минимум на краю сетки (lr {search['selected_lr']:g}): "
                     f"оптимум может лежать за её пределами, этап B не запускается")
    else:
        lines.append(f"  скорость обучения прогона: {float(search['selected_lr']):g}")
    return lines


__all__ = ["EXTRA_SUFFIX", "GRID_CONFIG", "PHASE_EQUAL", "PHASE_EXTRA", "SEARCH_DIR",
           "TUNING_FIELDS", "TUNING_KEY", "Tuning", "check_run_dir", "check_search",
           "describe_phase", "edge_error", "equal_terms_problems", "format_tuning",
           "inherit_search", "lr_label", "read_source_journal", "resolve", "run_lr_search",
           "search_protocol", "search_start", "search_terms", "select_lr",
           "tuning_from_config", "tuning_record"]
