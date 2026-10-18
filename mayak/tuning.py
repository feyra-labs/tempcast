"""Подбор скорости обучения и запись о нём."""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, fields, replace
from typing import Optional

from mayak.protocol import (CONFIG_FILE, JOURNAL, LR_FIELDS, SEED_FIELDS, Protocol,
                            ProtocolError, protocol_diff)
from mayak.stages import jsonable

log = logging.getLogger(__name__)

TUNING_KEY = "mayak_tuning"
SEARCH_DIR = "lr_search"
PHASE_EQUAL, PHASE_EXTRA = 1, 2
EXTRA_SUFFIX = "-tuned"
EXTRA_ARCH = "mayak"
GRID_CONFIG = "conf/train/default.yaml"
# Правило выбора по сетке, пишется в запись о подборе: сосед снизу от минимума в пределах
# допуска, край - только при отрыве от соседа больше допуска.
SELECT_RULE = "neighbor_below_tol+edge_tol"


@dataclass(frozen=True)
class Tuning:
    """Как прогон получает скорость обучения. В протокол не входит."""
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
        """Настройки подбора из None, словаря или готового объекта."""
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
    """Настройки подбора из секции конфига, где рядом лежат и другие ключи."""
    section = dict(section or {})
    return Tuning.coerce({k: section[k] for k in TUNING_FIELDS if k in section})


def search_protocol(protocol, lr):
    """Протокол прогона на одном значении сетки."""
    last = protocol.stages[-1]
    stages = (*protocol.stages[:-1], replace(last, steps=int(protocol.lr_search_steps)))
    return replace(protocol, lr=float(lr), stages=stages)


def check_search(protocol):
    """Проверяет, что подбор скорости обучения можно провести по этому протоколу."""
    if len(protocol.stages) < 2:
        raise ProtocolError("подбор скорости обучения: в протоколе один этап, а последний "
                            "этап после подбора стартует с чекпойнта предыдущего этапа "
                            "прогона сетки")
    last = protocol.stages[-1]
    if protocol.lr_search_steps > last.steps:
        raise ProtocolError(f"подбор скорости обучения: этап {last.name} на "
                            f"{protocol.lr_search_steps} шагов длиннее полного ({last.steps})")


def lr_label(lr):
    """Имя подкаталога прогона на одном значении сетки."""
    return f"lr{float(lr):g}"


def _scored(results, stage):
    """Значения сетки с конечной метрикой выбора, по возрастанию скорости обучения."""
    scored = [(float(r["lr"]), r["val_loss"].get(stage)) for r in results]
    finite = sorted((lr, float(v)) for lr, v in scored if v is not None and math.isfinite(v))
    if not finite:
        raise ProtocolError(f"подбор скорости обучения: ни одно значение сетки не дало "
                            f"конечной метрики выбора на этапе {stage}")
    i = min(range(len(finite)), key=lambda k: (finite[k][1], finite[k][0]))
    return finite, i


def _within(v, best, tol):
    """Метрика не хуже минимума больше чем на долю ``tol`` от его модуля."""
    return v <= best + float(tol) * abs(best)


def select_lr(results, stage, tol=0.0):
    """Скорость обучения, выбранная подбором."""
    finite, i = _scored(results, stage)
    best = finite[i][1]
    if i > 0 and _within(finite[i - 1][1], best, tol):
        return finite[i - 1][0]
    return finite[i][0]


def search_edge(results, stage, grid, tol=0.0):
    """Упирается ли минимум подбора в край сетки."""
    finite, i = _scored(results, stage)
    lr, best = finite[i]
    grid = sorted(float(v) for v in grid)
    if lr not in (grid[0], grid[-1]):
        return False
    if len(grid) < 2:
        return True
    inner = grid[1] if lr == grid[0] else grid[-2]
    neighbor = next((v for g, v in finite if g == inner), None)
    if neighbor is None:
        return True
    return not _within(neighbor, best, tol)


def run_lr_search(protocol, train, tag):
    """Проходит сетку скоростей обучения протокола."""
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
    stage = short.stages[-1].name
    best = select_lr(results, stage)
    selected = select_lr(results, stage, protocol.lr_tol)
    grid = list(protocol.lr_grid)
    edge = search_edge(results, stage, grid, protocol.lr_tol)
    if selected != best:
        log.info("подбор скорости обучения: минимум при lr %g, соседнее снизу lr %g в "
                 "пределах допуска %g - выбрано оно", best, selected, protocol.lr_tol)
    if edge:
        log.warning("подбор скорости обучения: минимум %s на краю сетки %s (lr %g) и лучше "
                    "соседа больше чем на допуск %g; сетка не расширяется, выбрано lr %g",
                    protocol.monitor, grid, best, protocol.lr_tol, selected)
    elif best in (grid[0], grid[-1]):
        log.info("подбор скорости обучения: минимум при крайнем lr %g, но соседнее значение "
                 "в пределах допуска %g: кривая у края плоская",
                 best, protocol.lr_tol)
    return jsonable(dict(grid=grid, stage_steps={s.name: s.steps for s in short.stages},
                         monitor=protocol.monitor, lr_tol=protocol.lr_tol, results=results,
                         min_lr=best, selected_lr=selected, edge=edge,
                         select_rule=SELECT_RULE))


SELECTION_ONLY = ("lr_grid", "lr_tol")


def _without_grid(cfg):
    """Конфиг прогона без полей, которые влияют только на выбор по сетке."""
    cfg = dict(cfg)
    cfg["train"] = {k: v for k, v in dict(cfg.get("train") or {}).items()
                    if k not in SELECTION_ONLY}
    return cfg


def reusable_grid_run(run_dir, arch, expected_cfg, data_key):
    """Журнал завершённого прогона значения сетки, если его можно взять без обучения."""
    jp, cp = os.path.join(run_dir, JOURNAL), os.path.join(run_dir, CONFIG_FILE)
    if not (os.path.isfile(jp) and os.path.isfile(cp)):
        return None

    def skip(reason):
        log.info("подбор скорости обучения: %s обучается заново: %s", run_dir, reason)

    try:
        with open(jp, encoding="utf-8") as f:
            journal = json.load(f)
        with open(cp, encoding="utf-8") as f:
            saved_cfg = json.load(f)
    except (OSError, ValueError) as e:
        return skip(f"журнал или конфиг не читается ({e})")
    if journal.get("arch") != arch:
        return skip(f"архитектура {journal.get('arch')!r}, нужна {arch!r}")
    if journal.get("data_key") != data_key:
        return skip("другой ключ кэша данных")
    want = jsonable(_without_grid(expected_cfg))
    have = jsonable(_without_grid(saved_cfg))
    if want != have:
        diff = sorted(f"{sec}.{k}" for sec in set(want) | set(have)
                      for k in set(want.get(sec) or {}) | set(have.get(sec) or {})
                      if (want.get(sec) or {}).get(k) != (have.get(sec) or {}).get(k))
        return skip(f"конфиг отличается: {', '.join(diff) or 'структура'}")
    stages = [dict(name=s["name"], steps=int(s["steps"])) for s in expected_cfg["train"]["stages"]]
    done = journal.get("stages") or []
    if [s.get("name") for s in done] != [s["name"] for s in stages]:
        return skip("прогон не завершён")
    for s, d in zip(stages, done):
        if int(d.get("steps_done") or 0) < s["steps"]:
            return skip(f"этап {s['name']} прошёл {d.get('steps_done')} из {s['steps']} шагов")
        if not d.get("best_ckpt") or not os.path.isfile(d["best_ckpt"]):
            return skip(f"нет чекпойнта этапа {s['name']}")
        if d.get("best_score") is None:
            return skip(f"нет метрики выбора этапа {s['name']}")
    log.info("подбор скорости обучения: %s уже обучен с тем же конфигом, берётся с диска",
             run_dir)
    return journal


def search_start(search, stage_name):
    """Чекпойнт, с которого после подбора стартует последний этап."""
    lr = float(search["selected_lr"])
    row = next(r for r in search["results"] if float(r["lr"]) == lr)
    return row["best_ckpts"][stage_name], row["journal"]


def read_source_journal(path):
    """Журнал прогона по пути к каталогу прогона или к самому журналу."""
    jp = path if os.path.isfile(path) else os.path.join(path, JOURNAL)
    if not os.path.isfile(jp):
        raise ProtocolError(f"журнал прогона {jp} не найден")
    with open(jp, encoding="utf-8") as f:
        return json.load(f), os.path.abspath(jp)


def inherit_search(path, arch, protocol):
    """Запись о подборе этапа 1 из журнала прогона основной модели."""
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
    """Запись о подборе для журнала и чекпойнтов."""
    return jsonable(dict(phase=tuning.phase, lr_search=lr_search, inherited_from=inherited_from))


def check_run_dir(journal_path, tuning):
    """Отказывает, если каталог прогона занят прогоном другого этапа сравнения."""
    if not os.path.isfile(journal_path):
        return
    with open(journal_path, encoding="utf-8") as f:
        old = (json.load(f).get("tuning") or {}).get("phase")
    new = tuning.phase if tuning.active else None
    if PHASE_EXTRA in (old, new) and old != new:
        raise ProtocolError(f"каталог {os.path.dirname(journal_path)} занят прогоном "
                            f"{describe_phase(old)}, этот прогон - {describe_phase(new)}: "
                            f"они лежат в разных каталогах")


def describe_phase(phase):
    """Название этапа сравнения для сообщений."""
    if phase == PHASE_EXTRA:
        return "этапа 2 сравнения (доп. настройка МАЯК)"
    if phase == PHASE_EQUAL:
        return "этапа 1 сравнения"
    return "без записи о подборе"


def resolve(arch, protocol, tuning, model_config, tag, train):
    """Протокол прогона и запись о подборе."""
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
    """Почему чекпойнт не входит в сравнение на равных."""
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
    """Условия подбора, которые у сравниваемых моделей должны совпадать."""
    search = (record or {}).get("lr_search") or {}
    return {k: search.get(k) for k in ("grid", "stage_steps", "monitor", "lr_tol")}


def format_tuning(record):
    """Сводка подбора для печати."""
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
    tol = search.get("lr_tol")
    lines.append(f"  сетка {search['grid']}, шаги этапов: {steps}, выбор по {search['monitor']}"
                 + ("" if tol is None else f", допуск {float(tol):g}"))
    best = search.get("min_lr", search.get("selected_lr"))
    for r in search["results"]:
        cells = ", ".join(f"{k} {'—' if v is None else f'{v:.5f}'}"
                          for k, v in r["val_loss"].items())
        marks = [m for m, lr in (("минимум", best), ("выбрано", search.get("selected_lr")))
                 if r["lr"] == lr]
        mark = f"  <- {', '.join(marks)}" if marks else ""
        lines.append(f"  lr {r['lr']:g}: {cells}{mark}")
    if search.get("edge"):
        lines.append(f"  ВНИМАНИЕ: минимум на краю сетки (lr {float(best):g}): "
                     f"оптимум может лежать за её пределами, сетка не расширялась")
    lines.append(f"  скорость обучения прогона: {float(search['selected_lr']):g}")
    return lines


__all__ = ["EXTRA_SUFFIX", "GRID_CONFIG", "PHASE_EQUAL", "PHASE_EXTRA", "SEARCH_DIR",
           "TUNING_FIELDS", "TUNING_KEY", "Tuning", "check_run_dir", "check_search",
           "describe_phase", "equal_terms_problems", "format_tuning",
           "search_edge",
           "inherit_search", "lr_label", "read_source_journal", "resolve", "reusable_grid_run",
           "run_lr_search","search_protocol", "search_start", "search_terms", "select_lr",
           "tuning_from_config", "tuning_record"]
