"""Подбор скорости обучения и запись о нём.

Этап 1 сравнения моделей. Каждая нейросеть, включая МАЯК, проходит одну и ту же сетку
скоростей обучения протокола (``lr_grid``). На каждом значении идёт прогон с полными
этапами протокола, кроме последнего, и последним этапом, укороченным до
``lr_search_steps`` шагов со своим расписанием WSD, которое до начала спада совпадает с
расписанием полного этапа. Внутри прогона сетки последний этап стартует с лучшего по
метрике выбора чекпойнта предыдущего, у всех моделей одинаково, а у этапа холодного
старта сохраняются кандидаты. Подбор отмечает значение с наименьшей метрикой выбора на
последнем этапе (при равенстве - меньшее) и то, лежит ли оно на краю сетки, пишет запись
в журнал каталога прогона и на этом заканчивается: полного прогона он не делает.

Дальше решает человек, у всех моделей одинаково:

1. подтверждает скорость обучения или выбирает другое значение сетки явным ``train.lr``;
   тогда в запись добавляется ``chosen_lr``;
2. выбирает чекпойнт предпоследнего этапа среди кандидатов прогона сетки с этой
   скоростью (``lr_search/lr<X>/stageA/``); этот этап заново не обучается;
3. запускает последний этап в том же каталоге прогона: ``run.stages=[B]
   run.init_from=<кандидат>``. Запись о подборе берётся из журнала каталога и остаётся
   собственной.

Абляции и повторы основной модели с другими сидами берут скорость обучения и запись о
подборе из журнала прогона основной модели (``run.lr_from``) и подбор не повторяют, а
этапы проходят по тому же пути: этап A, выбор кандидата человеком, этап B.

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
from mayak.stages import (CANDIDATE_DIR, InitCheckpointError, Launch, jsonable, plan_stages,
                          stage_dir_name)

log = logging.getLogger(__name__)

TUNING_KEY = "mayak_tuning"
SEARCH_DIR = "lr_search"
PHASE_EQUAL, PHASE_EXTRA = 1, 2
EXTRA_SUFFIX = "-tuned"
EXTRA_ARCH = "mayak"
# Прогоны сетки: все этапы, кандидаты этапа холодного старта сохраняются, чтобы человек
# выбрал из них старт последнего этапа.
SEARCH_LAUNCH = Launch()


@dataclass(frozen=True)
class Tuning:
    """Как прогон получает скорость обучения. В протокол не входит.

    Attributes:
        lr_search: пройти сетку скоростей обучения протокола и записать подбор в журнал
            каталога прогона; полного прогона нет.
        lr_from: каталог прогона основной модели или его журнал: оттуда берутся
            скорость обучения и запись о подборе, подбор не повторяется.
        extra_tuning: прогон дополнительной настройки МАЯК, этап 2 сравнения.
        lr_explicit: скорость обучения протокола задана человеком явно (``train.lr`` в
            командной строке). При запуске этапа после подбора она заменяет выбранную
            подбором и должна входить в сетку; без подбора - обычная скорость прогона.
    """
    lr_search: bool = False
    lr_from: Optional[str] = None
    extra_tuning: bool = False
    lr_explicit: bool = False

    def __post_init__(self):
        for name in ("lr_search", "extra_tuning", "lr_explicit"):
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
        if self.lr_explicit and self.lr_search:
            raise ValueError("подбор перебирает сетку train.lr_grid: явный train.lr с ним не "
                             "сочетается")
        if self.lr_explicit and self.lr_from:
            raise ValueError("скорость обучения берётся из прогона run.lr_from: явный train.lr "
                             "с ним не сочетается")

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


def check_search(protocol, launch):
    """Проверяет, что подбор скорости обучения можно провести в этом запуске.

    Args:
        protocol: протокол полного прогона.
        launch: настройки запуска.

    Raises:
        ProtocolError: запуск не по всем этапам; в протоколе один этап, и выбирать
            старт последнего этапа не из чего; последний этап при подборе не короче
            полного.
    """
    plan = plan_stages(protocol, launch)
    if len(plan) != len(protocol.stages):
        raise ProtocolError("подбор скорости обучения идёт по всем этапам протокола: "
                            "run.stages и run.init_from с ним не сочетаются")
    if len(protocol.stages) < 2:
        raise ProtocolError("подбор скорости обучения: в протоколе один этап, а после подбора "
                            "человек выбирает чекпойнт предыдущего этапа для старта "
                            "последнего")
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


def search_stage_dir(run_dir, lr, stage_name):
    """Каталог этапа прогона сетки с этой скоростью обучения.

    Args:
        run_dir: каталог прогона.
        lr: скорость обучения.
        stage_name: имя этапа.

    Returns:
        Путь вида ``<run_dir>/lr_search/lr0.001/stageA``.
    """
    return os.path.join(run_dir, SEARCH_DIR, lr_label(lr), stage_dir_name(stage_name))


def search_lr(search):
    """Скорость обучения прогона по записи о подборе.

    Args:
        search: запись о подборе по сетке.

    Returns:
        ``chosen_lr``, если человек выбрал значение явно, иначе ``selected_lr``.
    """
    return float(search["chosen_lr"] if "chosen_lr" in search else search["selected_lr"])


def choose_lr(search, lr=None):
    """Запись о подборе с решением человека о скорости обучения.

    Прежнее решение из записи не переносится: оно описывало прежний запуск.

    Args:
        search: запись о подборе по сетке.
        lr: скорость обучения, заданная явно; None - принять выбранную подбором.

    Returns:
        Копия записи; с явной скоростью - с полем ``chosen_lr``.

    Raises:
        ProtocolError: явная скорость не входит в сетку подбора.
    """
    out = {k: v for k, v in search.items() if k != "chosen_lr"}
    if lr is None:
        return out
    lr = float(lr)
    if lr not in [float(v) for v in out["grid"]]:
        raise ProtocolError(f"train.lr={lr:g} не из сетки подбора {out['grid']}: этап стартует "
                            f"с чекпойнта прогона сетки, поэтому скорость обучения - одно из "
                            f"её значений")
    out["chosen_lr"] = lr
    return jsonable(out)


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


def own_search(journal_path):
    """Собственная запись о подборе по сетке из журнала каталога прогона.

    Args:
        journal_path: путь к журналу в каталоге прогона.

    Returns:
        Запись о подборе с этапом сравнения; None, если журнала нет, подбора по сетке в
        нём нет или он взят из другого прогона.
    """
    if not os.path.isfile(journal_path):
        return None
    with open(journal_path) as f:
        rec = json.load(f).get("tuning") or {}
    if rec.get("lr_search") and not rec.get("inherited_from"):
        return rec
    return None


def search_init(path, run_dir, lr, stage_name):
    """Проверяет, откуда стартует этап, запущенный после подбора в каталоге прогона.

    Чекпойнт предыдущего этапа берётся из прогона сетки с этой скоростью обучения: из
    каталога этапа или его кандидатов. Чекпойнт из каталога этапа самого прогона тоже
    годится: так идут следующие этапы протокола длиннее двух этапов.

    Args:
        path: чекпойнт инициализации.
        run_dir: каталог прогона.
        lr: скорость обучения этого запуска.
        stage_name: имя этапа, чекпойнт которого нужен.

    Returns:
        True - чекпойнт из прогона сетки; False - из каталога этапа самого прогона.

    Raises:
        InitCheckpointError: чекпойнт лежит в другом месте, например в прогоне сетки с
            другой скоростью обучения.
    """
    where = os.path.realpath(os.path.dirname(str(path or "")))

    def inside(stage_dir):
        stage_dir = os.path.realpath(stage_dir)
        return where in (stage_dir, os.path.join(stage_dir, CANDIDATE_DIR))

    search_dir = search_stage_dir(run_dir, lr, stage_name)
    if inside(search_dir):
        return True
    if inside(os.path.join(run_dir, stage_dir_name(stage_name))):
        return False
    raise InitCheckpointError(
        f"чекпойнт {path} не подходит: после подбора этап стартует с чекпойнта этапа "
        f"{stage_name} прогона сетки со скоростью обучения этого запуска {lr:g}, то есть из "
        f"{search_dir} или {os.path.join(search_dir, CANDIDATE_DIR)}. Для другого значения "
        f"сетки задайте train.lr=<значение> и возьмите чекпойнт из его каталога")


def inherit_search(path, arch, protocol):
    """Запись о подборе этапа 1 из журнала прогона основной модели.

    Args:
        path: каталог прогона основной модели или его журнал.
        arch: архитектура этого прогона.
        protocol: протокол этого прогона.

    Returns:
        Пара: запись о подборе и абсолютный путь к журналу, из которого она взята.

    Raises:
        ProtocolError: журнала нет; в нём нет своего подбора этапа 1; архитектура
            другая; протоколы различаются не только скоростью обучения и сидами.
    """
    journal, jp = read_source_journal(path)
    rec = journal.get("tuning") or {}
    if (rec.get("phase") != PHASE_EQUAL or not rec.get("lr_search")
            or rec.get("inherited_from")):
        raise ProtocolError(f"{jp}: в прогоне нет своего подбора скорости обучения этапа 1; "
                            f"укажите прогон основной модели, запущенный с подбором")
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


def resolve(arch, protocol, tuning, launch, model_config, tag, train, journal_path):
    """Протокол запуска и запись о подборе.

    Запись о подборе берётся из одного из трёх источников: свой подбор по сетке
    (``run.lr_search``), журнал прогона основной модели (``run.lr_from``) или
    собственная запись в журнале каталога прогона, если этот запуск начинается не с
    первого этапа протокола. В последнем случае запись остаётся собственной, а явный
    ``train.lr`` пишется в неё как ``chosen_lr``.

    Args:
        arch: архитектура.
        protocol: протокол прогона.
        tuning: настройки подбора.
        launch: настройки запуска.
        model_config: конфиг архитектуры.
        tag: имя каталога прогона.
        train: функция обучения прогона на одном значении сетки, как в run_lr_search.
        journal_path: путь к журналу в каталоге прогона.

    Returns:
        Тройка: протокол со скоростью обучения прогона, запись о подборе (None, если её
        нет) и признак запуска после подбора в этом каталоге.

    Raises:
        ProtocolError: дополнительная настройка не для МАЯК; подбор для абляции;
            подбор невозможен в этом запуске; источник подбора не подходит; явная
            скорость не из сетки; запуск затёр бы собственную запись о подборе в журнале
            каталога.
    """
    if tuning.extra_tuning and arch != EXTRA_ARCH:
        raise ProtocolError(f"дополнительная настройка (этап 2) только для {EXTRA_ARCH}")
    abl = getattr(model_config, "ablations", None)
    if tuning.lr_search and abl is not None and abl.active() and not tuning.extra_tuning:
        raise ProtocolError("абляции берут скорость обучения основного МАЯК: вместо подбора "
                            "укажите его прогон (run.lr_from)")
    first = plan_stages(protocol, launch)[0][0]
    own = own_search(journal_path)
    search = inherited = None
    after_search = False
    if tuning.lr_search:
        check_search(protocol, launch)
        search = run_lr_search(protocol, train, tag)
    elif tuning.lr_from:
        search, inherited = inherit_search(tuning.lr_from, arch, protocol)
    elif first > 0 and own is not None and own.get("phase") == tuning.phase:
        search = choose_lr(own["lr_search"], protocol.lr if tuning.lr_explicit else None)
        after_search = True
    if own is not None and (search is None or inherited):
        raise ProtocolError(
            f"в журнале {journal_path} лежит подбор скорости обучения "
            f"{describe_phase(own.get('phase'))}, и этот запуск его затёр бы. После подбора "
            f"этап A заново не обучается: запустите следующий этап с кандидата прогона сетки "
            f"(run.stages=[B] run.init_from=<{SEARCH_DIR}/lr<X>/stageA/...>) или задайте "
            f"другой run.tag")
    if search is None:
        return protocol, (tuning_record(tuning) if tuning.active else None), False
    protocol = replace(protocol, lr=search_lr(search))
    return protocol, tuning_record(tuning, search, inherited), after_search


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
    if "chosen_lr" in search:
        target, who = search["chosen_lr"], "выбранной человеком из сетки"
    else:
        target, who = search.get("selected_lr", math.nan), "выбранной подбором"
    if float(target) != lr:
        return [f"скорость обучения {lr:g} не равна {who} {target}"]
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
        marks = [m for m, key in (("минимум", "selected_lr"), ("выбор человека", "chosen_lr"))
                 if search.get(key) is not None and r["lr"] == search[key]]
        mark = f"  <- {', '.join(marks)}" if marks else ""
        lines.append(f"  lr {r['lr']:g}: {cells}{mark}")
    if search.get("edge"):
        lines.append(f"  ВНИМАНИЕ: минимум на краю сетки (lr {search['selected_lr']:g}): "
                     f"оптимум может лежать за её пределами")
    lines.append(f"  скорость обучения прогона: {search_lr(search):g}")
    return lines


__all__ = ["EXTRA_SUFFIX", "PHASE_EQUAL", "PHASE_EXTRA", "SEARCH_DIR", "SEARCH_LAUNCH",
           "TUNING_FIELDS", "TUNING_KEY", "Tuning", "check_run_dir", "check_search",
           "choose_lr", "describe_phase", "equal_terms_problems", "format_tuning",
           "inherit_search", "lr_label", "own_search", "read_source_journal", "resolve",
           "run_lr_search", "search_init", "search_lr", "search_protocol",
           "search_stage_dir", "search_terms", "select_lr", "tuning_from_config",
           "tuning_record"]
