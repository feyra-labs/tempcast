"""Раздельный запуск этапов обучения и переход между ними.

Этапы протокола можно запускать по одному. Этап, который в протоколе идёт не первым,
стартует с явно указанного чекпойнта предыдущего этапа. Перед стартом чекпойнт
проверяется целиком, и все найденные отличия перечисляются в одном исключении: та же
архитектура, тот же конфиг модели и данных, тот же протокол, та же запись о подборе
скорости обучения, тот же кэш данных, тот же набор валидации, чекпойнт выбран на
валидационных станциях по правилам чек-листа антиутечек.

Какие этапы запускать, с какого чекпойнта стартовать и сохранение кандидатов в протокол
не входят. Протокол один на все модели и сравнивается между чекпойнтами, а эти настройки
описывают только то, как человек проводит обучение по шагам. Поэтому этап B, запущенный
отдельной командой, получает тот же протокол, что и в прогоне одной командой, и с теми же
сидами даёт тот же чекпойнт.

На этапе холодного старта после каждой валидации сохраняется кандидат. Решение, какой
кандидат передать следующему этапу, принимает человек по отчёту и графикам этапа; код
только считает и печатает числа и ничего не решает за человека. После подбора скорости
обучения кандидаты берутся из прогона сетки с выбранной скоростью, этап холодного старта
заново не обучается (``mayak.tuning``).
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

from mayak.protocol import ProtocolError

log = logging.getLogger(__name__)

STAGE_KEY = "mayak_stage"
FIELD_CURRICULUM = "L0"
CANDIDATE_DIR = "candidates"
MAX_LISTED_DIFFS = 25
INIT_EXIT_CODE = 2


class InitCheckpointError(ProtocolError):
    """Чекпойнт, с которого должен стартовать этап, не подходит этому запуску."""


def stage_dir_name(name):
    """Имя каталога этапа внутри каталога прогона.

    Args:
        name: имя этапа.

    Returns:
        Строка вида stageA.
    """
    return f"stage{name}"


def parse_stage_list(value, allow_empty=False):
    """Список имён этапов из строки через запятую или пробел либо из списка.

    Args:
        value: строка вроде «A,B» или «A B», либо список строк.
        allow_empty: разрешить пустой список.

    Returns:
        Кортеж имён без пустых элементов, в исходном порядке.

    Raises:
        ValueError: список пуст, а пустой не разрешён.
    """
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = [str(v).strip() for v in value]
    out = tuple(v for v in items if v)
    if not out and not allow_empty:
        raise ValueError("пустой список этапов")
    return out


@dataclass(frozen=True)
class Launch:
    """Что запустить в этом вызове обучения. В протокол не входит.

    Attributes:
        stages: имена запускаемых этапов подряд в порядке протокола; None значит все
            этапы протокола, как одной командой.
        init_from: чекпойнт предыдущего этапа, с которого стартует первый запускаемый
            этап. Обязателен, если этот этап в протоколе не первый.
        candidates: этапы, у которых чекпойнт сохраняется после каждой валидации;
            None значит этапы холодного старта, пустой кортеж отключает сохранение.
    """
    stages: Optional[tuple] = None
    init_from: Optional[str] = None
    candidates: Optional[tuple] = None

    def __post_init__(self):
        if self.stages is not None:
            object.__setattr__(self, "stages", parse_stage_list(self.stages))
        if self.candidates is not None:
            object.__setattr__(self, "candidates",
                               parse_stage_list(self.candidates, allow_empty=True))
        if self.init_from is not None:
            path = str(self.init_from).strip()
            object.__setattr__(self, "init_from", path or None)

    @classmethod
    def coerce(cls, value):
        """Настройки запуска из None, словаря или готового объекта.

        Args:
            value: None, словарь с полями настроек или сами настройки.

        Returns:
            Настройки запуска.

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
            raise ValueError(f"неизвестные настройки запуска {extra}; есть {sorted(known)}")
        return cls(**d)

    def to_dict(self):
        """Словарь, пригодный для записи в JSON."""
        return jsonable(asdict(self))


LAUNCH_FIELDS = tuple(f.name for f in fields(Launch))


def launch_from_config(section):
    """Настройки запуска из секции конфига, где рядом лежат и другие ключи.

    Args:
        section: словарь секции; берутся только ключи настроек запуска.

    Returns:
        Настройки запуска.
    """
    section = dict(section or {})
    return Launch.coerce({k: section[k] for k in LAUNCH_FIELDS if k in section})


def plan_stages(protocol, launch):
    """Проверяет выбор этапов и возвращает запускаемые этапы с их номерами в протоколе.

    Args:
        protocol: протокол обучения.
        launch: настройки запуска.

    Returns:
        Список пар из номера этапа в протоколе и самого этапа, в порядке запуска.

    Raises:
        ProtocolError: неизвестный или повторённый этап; этапы идут не подряд или не в
            порядке протокола; нет чекпойнта инициализации для этапа, который в
            протоколе не первый, или он задан для первого; неизвестный этап в списке
            кандидатов.
    """
    names = [s.name for s in protocol.stages]
    wanted = list(launch.stages) if launch.stages is not None else list(names)
    unknown = [n for n in wanted if n not in names]
    if unknown:
        raise ProtocolError(f"неизвестные этапы {unknown}; в протоколе {names}")
    if len(set(wanted)) != len(wanted):
        raise ProtocolError(f"этапы повторяются: {wanted}")
    idx = [names.index(n) for n in wanted]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ProtocolError(f"этапы {wanted} должны идти подряд в порядке протокола {names}")
    first = idx[0]
    if first > 0 and not launch.init_from:
        raise ProtocolError(f"этап {names[first]} в протоколе не первый: укажите чекпойнт этапа "
                            f"{names[first - 1]}, с которого он стартует (run.init_from)")
    if first == 0 and launch.init_from:
        raise ProtocolError(f"этап {names[0]} первый в протоколе и стартует с нуля: чекпойнт "
                            f"инициализации ему не нужен")
    bad = [n for n in (launch.candidates or ()) if n not in names]
    if bad:
        raise ProtocolError(f"кандидаты для неизвестных этапов {bad}; в протоколе {names}")
    return [(i, protocol.stages[i]) for i in idx]


def candidate_stages(protocol, launch):
    """Этапы, у которых сохраняется чекпойнт после каждой валидации.

    Args:
        protocol: протокол обучения.
        launch: настройки запуска.

    Returns:
        Кортеж имён этапов.
    """
    if launch.candidates is None:
        return tuple(s.name for s in protocol.stages if s.curriculum == FIELD_CURRICULUM)
    return tuple(launch.candidates)


def jsonable(obj):
    """Значение, которое JSON записывает и читает обратно без изменений.

    Кортежи становятся списками, числа numpy и тензоры обычными числами, ключи словарей
    строками, а бесконечности и нечисла значением None: иначе прочитанный журнал не
    совпал бы с записанным.

    Args:
        obj: значение.

    Returns:
        Значение из словарей, списков, строк, чисел, логических значений и None.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def _normalized(value):
    return json.loads(json.dumps(jsonable(value), sort_keys=True))


def config_diff(where, old, new):
    """Отличия двух конфигов, по одной строке на поле.

    Args:
        where: имя конфига в начале каждой строки.
        old: значение из чекпойнта.
        new: значение этого запуска.

    Returns:
        Список строк вида «модель.encoder_width: в чекпойнте 16, в запуске 64».
    """
    old, new = _normalized(old), _normalized(new)
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            if k not in old:
                out.append(f"{where}.{k}: нет в чекпойнте, в запуске {new[k]!r}")
            elif k not in new:
                out.append(f"{where}.{k}: в чекпойнте {old[k]!r}, нет в запуске")
            else:
                out += config_diff(f"{where}.{k}", old[k], new[k])
        return out
    if old != new:
        return [f"{where}: в чекпойнте {old!r}, в запуске {new!r}"]
    return []


# Путь к манифесту и каталог кэша зависят от того, откуда запущена команда. Совпадение
# данных проверяется по ключу кэша: он строится по содержимому манифеста и источников.
DATA_PATH_FIELDS = ("manifest", "cache_root")


def _data_fields(d):
    return {k: v for k, v in dict(d or {}).items() if k not in DATA_PATH_FIELDS}


def init_mismatches(ck, arch, protocol, model_config, data_config, stage, data_key,
                    val_digest, tuning=None):
    """Все причины, по которым чекпойнт не годится для старта следующего этапа.

    Args:
        ck: загруженный чекпойнт, словарь.
        arch: архитектура этого запуска.
        protocol: протокол этого запуска, словарь.
        model_config: конфиг модели этого запуска, словарь.
        data_config: конфиг данных этого запуска, словарь.
        stage: этап, чекпойнт которого нужен.
        data_key: ключ кэша данных этого запуска.
        val_digest: отпечаток набора валидации этого этапа в этом запуске.
        tuning: запись о подборе скорости обучения этого запуска или None.

    Returns:
        Список причин; пустой, если чекпойнт подходит.
    """
    from mayak.leakage import SELECTION_KEY
    from mayak.tuning import TUNING_KEY
    hp = ck.get("hyper_parameters") or {}
    out = []
    got_arch = hp.get("arch")
    if got_arch != arch:
        out.append(f"архитектура: в чекпойнте {got_arch!r}, в запуске {arch!r}")
    if hp.get("stage") != stage.name:
        out.append(f"этап: чекпойнт этапа {hp.get('stage')!r}, нужен чекпойнт этапа "
                   f"{stage.name!r}")
    if got_arch == arch:
        if hp.get("model_config") is None:
            out.append("модель: в чекпойнте нет конфига модели")
        else:
            out += config_diff("модель", hp["model_config"], model_config)
    if hp.get("data_config") is None:
        out.append("данные: в чекпойнте нет конфига данных")
    else:
        out += config_diff("данные", _data_fields(hp["data_config"]), _data_fields(data_config))
    if "protocol" not in hp:
        out.append("протокол: в чекпойнте нет протокола обучения")
    else:
        from mayak.protocol import Protocol
        try:
            got = Protocol.from_dict(hp["protocol"]).to_dict()
        except (ProtocolError, TypeError, ValueError) as e:
            out.append(f"протокол: протокол в чекпойнте не читается: {e}")
        else:
            out += config_diff("протокол", got, protocol)
    if _normalized(ck.get(TUNING_KEY)) != _normalized(tuning):
        out.append("запись о подборе скорости обучения в чекпойнте другая, чем у этого "
                   "запуска: укажите те же run.lr_from, run.extra_tuning и train.lr, что у "
                   "предыдущего этапа")
    rec = ck.get(STAGE_KEY)
    if not rec:
        out.append("нет записи об этапе: чекпойнт сохранён до раздельного запуска этапов, "
                   "обучите этап заново")
    elif rec.get("data_key") != data_key:
        out.append(f"данные: чекпойнт обучен на кэше {rec.get('data_key')!r}, сейчас кэш "
                   f"{data_key!r}")
    sel = ck.get(SELECTION_KEY) or {}
    if sel.get("windows_digest") != val_digest:
        out.append(f"набор валидации этапа {stage.name}: в чекпойнте "
                   f"{sel.get('windows_digest')!r}, в запуске {val_digest!r}")
    got_curriculum = (sel.get("history") or {}).get("curriculum")
    if got_curriculum != stage.curriculum:
        out.append(f"чекпойнт выбран на окнах с куррикулумом {got_curriculum!r}, у этапа "
                   f"{stage.name} куррикулум {stage.curriculum!r}")
    return out


def _format_problems(path, problems):
    shown = problems[:MAX_LISTED_DIFFS]
    more = len(problems) - len(shown)
    lines = [f"чекпойнт {path} не подходит для старта этапа:"] + [f"  - {p}" for p in shown]
    if more > 0:
        lines.append(f"  и ещё {more}")
    return "\n".join(lines)


def _commit_warnings(ck):
    from mayak.lit import RUN_KEY
    from mayak.provenance import git_info
    old = (((ck.get(RUN_KEY) or {}).get("provenance") or {}).get("git") or {})
    now = git_info()
    out = []
    if old.get("commit") and now.get("commit") and old["commit"] != now["commit"]:
        out.append(f"код изменился: чекпойнт обучен на коммите {old['commit'][:12]}, "
                   f"сейчас {now['commit'][:12]}")
    if old.get("dirty"):
        out.append("чекпойнт обучен на коде с незакоммиченными правками")
    return out


def _is_best_in_journal(journal_path, stage_name, digest, step):
    """Был ли чекпойнт лучшим по метрике выбора в своём прогоне.

    Args:
        journal_path: журнал прогона, в котором сохранён чекпойнт.
        stage_name: имя этапа.
        digest: отпечаток файла чекпойнта.
        step: шаг, на котором он сохранён.

    Returns:
        True или False, если журнал знает этот файл; None, если журнала нет или файл в
        нём не упомянут.
    """
    if not journal_path or not os.path.isfile(journal_path):
        return None
    with open(journal_path) as f:
        journal = json.load(f)
    entry = next((s for s in reversed(journal.get("stages", []))
                  if s.get("name") == stage_name), None)
    if entry is None:
        return None
    known = {entry.get("best_digest")} | {c.get("digest") for c in entry.get("candidates", [])}
    if digest not in known:
        return None
    return step == entry.get("best_step")


def inspect_init_checkpoint(path, arch, protocol, model_config, data_config, stage, store,
                            val_digest, tuning=None):
    """Проверяет чекпойнт, с которого стартует этап, и возвращает запись о нём.

    Args:
        path: путь к чекпойнту.
        arch: архитектура этого запуска.
        protocol: протокол этого запуска.
        model_config: конфиг модели этого запуска.
        data_config: конфиг данных этого запуска.
        stage: этап, чекпойнт которого нужен.
        store: набор станций этого запуска.
        val_digest: отпечаток набора валидации этого этапа в этом запуске.
        tuning: запись о подборе скорости обучения этого запуска или None.

    Returns:
        Словарь: путь, отпечаток файла, этап, шаг, каталог и журнал прогона, в котором
        чекпойнт сохранён, был ли он лучшим по метрике выбора и предупреждения.

    Raises:
        InitCheckpointError: файла нет, или чекпойнт не подходит; в сообщении все
            найденные отличия.
    """
    import torch

    from mayak.leakage import SELECTION_KEY, LeakageError, check_selection_record, file_digest
    if not path or not os.path.isfile(path):
        raise InitCheckpointError(f"чекпойнт инициализации {path!r} не найден")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    problems = init_mismatches(ck, arch=arch, protocol=protocol.to_dict(),
                               model_config=model_config.to_dict(),
                               data_config=data_config.to_dict(), stage=stage,
                               data_key=store.key, val_digest=val_digest, tuning=tuning)
    try:
        check_selection_record(ck.get(SELECTION_KEY), store, what="запись о выборе")
    except LeakageError as e:
        problems.append(f"чек-лист антиутечек: {e}")
    if problems:
        raise InitCheckpointError(_format_problems(path, problems))
    rec = ck[STAGE_KEY]
    digest = file_digest(path)
    step = int(ck.get("global_step", rec.get("step", -1)))
    warnings = _commit_warnings(ck)
    for w in warnings:
        log.warning("чекпойнт инициализации %s: %s", path, w)
    return dict(ckpt=os.path.abspath(path), digest=digest, stage=stage.name, step=step,
                run_dir=rec.get("run_dir"), journal=rec.get("journal"),
                is_best=_is_best_in_journal(rec.get("journal"), stage.name, digest, step),
                warnings=warnings, report=None, report_file=None)


def load_init_weights(module, path):
    """Загружает в модуль веса чекпойнта предыдущего этапа.

    Args:
        module: обучаемый модуль этапа.
        path: путь к чекпойнту.

    Raises:
        InitCheckpointError: веса не подходят к модели.
    """
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    try:
        module.load_state_dict(sd)
    except RuntimeError as e:
        raise InitCheckpointError(f"веса чекпойнта {path} не подходят к модели этапа: {e}") from e


def stage_record(stage, index, data_key, run_dir, journal, init_from=None):
    """Запись об этапе, которая кладётся в каждый его чекпойнт.

    Args:
        stage: этап.
        index: номер этапа в протоколе.
        data_key: ключ кэша данных.
        run_dir: каталог прогона.
        journal: путь к журналу прогона.
        init_from: откуда этап стартовал: путь, отпечаток, этап и шаг; None для первого.

    Returns:
        Словарь без шага сохранения: его добавляет сохранение.
    """
    return jsonable(dict(stage=stage.name, index=int(index), curriculum=stage.curriculum,
                         data_key=data_key, run_dir=os.path.abspath(run_dir),
                         journal=os.path.abspath(journal), init_from=init_from))


def lineage(prev):
    """Короткая запись о том, откуда стартовал этап, для чекпойнта.

    Args:
        prev: запись о чекпойнте предыдущего этапа или None.

    Returns:
        Путь, отпечаток, этап и шаг; None для этапа, который стартует с нуля.
    """
    if prev is None:
        return None
    return jsonable({k: prev.get(k) for k in ("ckpt", "digest", "stage", "step")})


def init_summary(prev, report_summary=None):
    """Запись для журнала о том, с какого чекпойнта стартовал этап.

    Args:
        prev: запись о чекпойнте предыдущего этапа или None.
        report_summary: сводка отчёта о поле этого чекпойнта или None.

    Returns:
        Словарь для журнала или None для этапа, который стартует с нуля.
    """
    if prev is None:
        return None
    keys = ("ckpt", "digest", "stage", "step", "is_best", "run_dir", "journal", "report_file",
            "warnings")
    out = {k: prev.get(k) for k in keys}
    out["report"] = report_summary
    return jsonable(out)


JOURNAL_IDENTITY = ("arch", "model_class", "protocol", "manifest", "seeds", "config_file",
                    "augment", "data_key", "tuning")


def start_journal(path, base, protocol, first, init, run_dir):
    """Журнал прогона перед запуском этапов.

    Если первый запускаемый этап стартует с чекпойнта, сохранённого в этом же каталоге
    прогона, и лежащий там журнал описывает тот же прогон, записи предыдущих этапов
    переносятся из него. Иначе журнал начинается заново, а откуда стартовал этап, видно
    по записи о чекпойнте инициализации.

    Args:
        path: путь к журналу.
        base: общие поля журнала этого запуска.
        protocol: протокол обучения.
        first: номер первого запускаемого этапа в протоколе.
        init: запись о чекпойнте инициализации или None.
        run_dir: каталог прогона.

    Returns:
        Журнал без записей о запускаемых этапах.
    """
    journal = dict(base, stages=[], final_ckpt=None)
    if first == 0 or init is None or not os.path.isfile(path):
        return journal
    if os.path.realpath(init.get("run_dir") or "") != os.path.realpath(run_dir):
        return journal
    with open(path) as f:
        old = json.load(f)
    if any(_normalized(old.get(k)) != _normalized(base.get(k)) for k in JOURNAL_IDENTITY):
        log.warning("журнал %s описывает другой прогон: записи прежних этапов не переносятся",
                    path)
        return journal
    before = {s.name for s in protocol.stages[:first]}
    journal["stages"] = [s for s in old.get("stages", []) if s.get("name") in before]
    for k in ("param_groups", "n_params", "n_params_by_module"):
        if k in old:
            journal[k] = old[k]
    return journal


def warn_stale(stage_dir):
    """Предупреждает, если в каталоге этапа остались чекпойнты прежнего запуска.

    Args:
        stage_dir: каталог этапа.
    """
    if not os.path.isdir(stage_dir):
        return
    old = [n for n in os.listdir(stage_dir) if n.endswith(".ckpt")]
    cand = os.path.join(stage_dir, CANDIDATE_DIR)
    if os.path.isdir(cand):
        old += [n for n in os.listdir(cand) if n.endswith(".ckpt")]
    if old:
        log.warning("в %s уже есть чекпойнты прежнего запуска (%d); новые получат суффикс "
                    "версии, журнал и отчёт укажут только на новые", stage_dir, len(old))


__all__ = ["CANDIDATE_DIR", "FIELD_CURRICULUM", "INIT_EXIT_CODE", "InitCheckpointError",
           "LAUNCH_FIELDS", "Launch", "STAGE_KEY", "candidate_stages", "config_diff",
           "init_mismatches", "init_summary", "inspect_init_checkpoint", "jsonable",
           "launch_from_config", "lineage", "load_init_weights", "parse_stage_list",
           "plan_stages", "stage_dir_name", "stage_record", "start_journal", "warn_stale"]
