"""Карточки бейзлайнов: источник, что взято, чем отличается от оригинала и почему.

Формат:

* ``sources`` — статьи (авторы, название, место публикации, arXiv) и книги;
* ``code``    — эталонные реализации, с которыми сверялась наша;
* ``taken``   — что именно взято из оригинала;
* ``differences`` — каждое отличие вместе с причиной. Отличие без причины —
  ошибка конструирования карточки.
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = ("statistical", "neural")


class CardError(ValueError):
    """Неполная карточка бейзлайна."""


@dataclass(frozen=True)
class Source:
    authors: str
    title: str
    venue: str
    ref: str = ""

    def render(self):
        ref = f", {self.ref}" if self.ref else ""
        return f"{self.authors}. «{self.title}». {self.venue}{ref}."


@dataclass(frozen=True)
class Difference:
    what: str
    why: str


@dataclass(frozen=True)
class BaselineCard:
    """Описание бейзлайна.

    key     — ключ реестра (для нейробейзлайнов совпадает с именем архитектуры);
    name    — имя строки в таблицах оценки;
    kind    — ``statistical`` (без обучения весов) или ``neural``.
    """
    key: str
    name: str
    kind: str
    summary: str
    sources: tuple
    taken: tuple
    differences: tuple
    code: tuple = ()
    notes: tuple = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise CardError(f"{self.key}: kind={self.kind!r}, допустимо {KINDS}")
        if not self.sources:
            raise CardError(f"{self.key}: нет источника")
        if not self.taken:
            raise CardError(f"{self.key}: не указано, что взято из оригинала")
        for d in self.differences:
            if not isinstance(d, Difference) or not d.what.strip() or not d.why.strip():
                raise CardError(f"{self.key}: отличие без описания или без причины: {d!r}")

    @property
    def refs(self):
        """Все ссылки карточки: arXiv-идентификаторы и адреса кода."""
        return tuple(s.ref for s in self.sources if s.ref) + tuple(self.code)

    def render(self, heading="Карточка бейзлайна"):
        """Текст для докстринга и для mayak/baselines/README.md (разметка Markdown-совместима)."""
        lines = [f"{heading}: {self.name}", "", self.summary, "", "Источник:"]
        lines += [f"  - {s.render()}" for s in self.sources]
        if self.code:
            lines += ["", "Эталонная реализация:"]
            lines += [f"  - {c}" for c in self.code]
        lines += ["", "Что взято:"]
        lines += [f"  - {t}" for t in self.taken]
        lines += ["", "Отличия от оригинала и их причины:"]
        if self.differences:
            lines += [f"  - {d.what}.\n    Причина: {d.why}." for d in self.differences]
        else:
            lines += ["  - нет"]
        if self.notes:
            lines += ["", "Замечания:"]
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


REGISTRY: dict = {}


def describe(card):
    """Декоратор: регистрирует карточку и дописывает её в докстринг объекта."""
    if card.key in REGISTRY and REGISTRY[card.key] != card:
        raise CardError(f"карточка {card.key!r} уже зарегистрирована")
    REGISTRY[card.key] = card

    def deco(obj):
        base = (obj.__doc__ or "").rstrip()
        obj.__doc__ = (base + "\n\n" if base else "") + card.render()
        obj.CARD = card
        return obj

    return deco


def card_for(key):
    if key not in REGISTRY:
        raise KeyError(f"нет карточки бейзлайна {key!r}; есть {tuple(REGISTRY)}")
    return REGISTRY[key]


def display_name(key):
    return card_for(key).name


DOC_HEADER = """# Бейзлайны

Файл собран из карточек в коде (`mayak/baselines/`) командой
`python scripts/baseline_cards.py --out mayak/baselines/README.md`; руками не править —
тест `tests/test_block11_baselines.py` сверяет его с карточками.

Все нейробейзлайны обучаются одной функцией `mayak.protocol.run_protocol` по
единому протоколу (блок 4): одинаковые шаги, батч, поток окон, оптимизатор,
расписание, ранняя остановка, выбор чекпойнта и EMA весов; функция потерь —
общий pinball, нормированный на климатологический масштаб из батча. Все модели
оцениваются `mayak.evaluate` на одних и тех же окнах одними и теми же метриками.
Квантили у всех нейробейзлайнов строятся одинаково: медиана `mu`, масштаб `sigma`
и монотонные по уровню смещения, общие для всех окон и свои для каждого лида.
"""


def markdown(order=None):
    """mayak/baselines/README.md целиком."""
    keys = list(order or REGISTRY)
    parts = [DOC_HEADER]
    for kind, title in (("statistical", "Статистические эталоны"),
                        ("neural", "Обучаемые бейзлайны")):
        cards = [REGISTRY[k] for k in keys if REGISTRY[k].kind == kind]
        if not cards:
            continue
        parts.append(f"## {title}\n")
        for c in cards:
            body = c.render(heading="###").split("\n", 1)
            parts.append(f"### {c.name} (`{c.key}`)\n{body[1]}\n")
    return "\n".join(parts).rstrip() + "\n"


__all__ = ["BaselineCard", "CardError", "Difference", "KINDS", "REGISTRY", "Source",
           "card_for", "describe", "display_name", "markdown"]
