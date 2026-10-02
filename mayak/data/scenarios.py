"""Сценарии робастности: преобразования окна для обученной модели.

Сценарий - это преобразование одного окна аугментаций, а не отдельная копия
датасета. Контракт каждого сценария - класс, искажается ли цель, границы уровня,
параметры - задан в конфиге робастности; здесь только реализация.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from mayak.config import SCENARIO_INPUT, SCENARIO_INSTRUMENT, SCENARIO_RULES, ScenarioRule
from mayak.constants import L_MAX
from mayak.data.augment import P, RH, AugWindow, apply_one, drift_offset, drift_target

DROP_CHANNEL_LABELS = ("все каналы", "без P", "без RH", "без P и RH")


def _dropout(w: AugWindow, level, rng, p):
    """Доля ``level`` часов истории пропадает целиком (все каналы)."""
    u = rng.random(L_MAX)
    w.m[u < level] = 0.0


def _gap(w, level, rng, p):
    """Последние ``level`` часов перед выпуском прогноза - без данных."""
    n = int(level)
    if n > 0:
        w.m[L_MAX - n:] = 0.0


def _offset(w, level, rng, p):
    """Постоянное смещение температуры: история и цель, та же функция, что в обучении."""
    apply_one(w, "offset", dict(b=float(level)))


def _offset_input(w, level, rng, p):
    """То же смещение, но цель не трогается: незамеченное смещение прибора."""
    apply_one(w, "offset", dict(b=float(level), target=False))


def _drift_params(level, target):
    return dict(rate=[float(level), 0.0, 0.0], walk=False, seed=0, target=target)


def _drift(w, level, rng, p):
    """Дрейф температуры со скоростью ``level`` в сутки той же функцией, что в обучении.

    История нарастает от нуля до текущего смещения, на горизонте цель смещается дальше с
    той же скоростью.
    """
    apply_one(w, "drift", _drift_params(level, True))


def _drift_input(w, level, rng, p):
    """Тот же дрейф, но цель не трогается: незамеченный дрейф прибора."""
    apply_one(w, "drift", _drift_params(level, False))


def _scale(w, level, rng, p):
    """Ошибка масштаба температуры той же функцией, что в обучении: история и цель."""
    apply_one(w, "scale", dict(k=[1.0 + float(level), 1.0, 1.0]))


def _noise(w, level, rng, p):
    """Гауссов шум; его разброс у температуры равен уровню.

    У давления и влажности разброс задан в отношении ``sd_ratio``. Влажность после
    шума обрезается в допустимый диапазон.
    """
    z = rng.standard_normal((L_MAX, 3)).astype(np.float32)
    sd = np.float32(level) * np.asarray(p["sd_ratio"], np.float32)
    w.x[:] = w.x + z * sd * (w.m > 0)
    ok = w.m[:, RH] > 0
    w.x[:, RH] = np.where(ok, np.clip(w.x[:, RH], 0.0, 100.0), w.x[:, RH])


def _spikes(w, level, rng, p):
    """Одиночные выбросы: на доле валидных точек, равной уровню, значение сдвинуто на ``magnitude``.

    Направление сдвига каждой точки случайное.
    """
    u = rng.random((L_MAX, 3))
    sign = np.where(rng.random((L_MAX, 3)) < 0.5, -1.0, 1.0).astype(np.float32)
    hit = (u < level) & (w.m > 0)
    w.x[:] = w.x + np.where(hit, sign * np.asarray(p["magnitude"], np.float32), 0.0)


def _freeze(w, level, rng, p):
    """Замерзание: последние ``level`` часов канал повторяет значение момента отказа.

    Значение момента отказа - последнее валидное до начала замерзания; если его нет
    (замерзание длиннее истории) - первое валидное внутри участка. Маска не меняется:
    замёрзший датчик продолжает отчитываться, но только там, где отчитывался.
    """
    n = int(level)
    if n <= 0:
        return
    i0 = max(L_MAX - n, w.h0)
    for ch in p["channels"]:
        rows = np.flatnonzero(w.m[w.h0:, ch] > 0) + w.h0
        if rows.size == 0:
            continue
        before, inside = rows[rows < i0], rows[rows >= i0]
        if inside.size == 0:
            continue
        v = w.x[before[-1] if before.size else inside[0], ch]
        w.x[inside, ch] = v


def _drop_channel(w, level, rng, p):
    """0 - ничего, 1 - нет давления, 2 - нет влажности, 3 - нет обоих, на всей истории."""
    lv = int(level)
    if lv in (1, 3):
        w.m[:, P] = 0.0
    if lv in (2, 3):
        w.m[:, RH] = 0.0


def _history(w, level, rng, p):
    """История укорочена на ``level`` самых старых часов; 672 - холодный старт."""
    cut = int(level)
    if cut <= 0:
        return
    new_l = max(0, w.L - cut)
    w.m[:L_MAX - new_l] = 0.0
    w.L = new_l


def _coords(w, level, rng, p):
    """Ошибка координат на ``level`` градусов в случайном направлении."""
    th = rng.uniform(0.0, 2.0 * np.pi)
    lat = w.lat + level * np.cos(th)
    lon = w.lon + level * np.sin(th)
    w.lat = float(min(90.0, max(-90.0, lat)))
    w.lon = float(lon if -180.0 <= lon < 180.0 else (lon + 180.0) % 360.0 - 180.0)


def _elev(w, level, rng, p):
    """Ошибка высоты на ``level`` метров со случайным знаком (и в метаданных QC)."""
    d = level if rng.random() < 0.5 else -level
    w.elev = float(w.elev + d)
    if w.qc_elev is not None:
        w.qc_elev = float(w.qc_elev + d)


@dataclass(frozen=True)
class ScenarioDef:
    name: str
    rule: ScenarioRule
    fn: Callable
    stream: int

    @property
    def kind(self):
        return self.rule.kind

    @property
    def target(self):
        return self.rule.target


_FN = {"dropout": _dropout, "gap": _gap, "noise": _noise, "spikes": _spikes,
       "freeze": _freeze, "drop_channel": _drop_channel, "history": _history,
       "coords": _coords, "elev": _elev, "offset": _offset, "drift": _drift,
       "scale": _scale, "offset_input": _offset_input, "drift_input": _drift_input}
# Номер подпотока случайных чисел. Вариант «только вход» делит подпоток с основным.
_STREAM = {"dropout": 1, "gap": 2, "offset": 3, "offset_input": 3, "drift": 4,
           "drift_input": 4, "scale": 5, "noise": 6, "spikes": 7, "freeze": 8,
           "drop_channel": 9, "history": 10, "coords": 11, "elev": 12}
_DITHER_STREAM = 14
DITHER_SCENARIOS = frozenset({"noise", "offset", "offset_input", "drift", "drift_input",
                              "scale"})
assert set(_FN) == set(_STREAM) == set(SCENARIO_RULES)
for _r in SCENARIO_RULES.values():
    assert _r.kind in (SCENARIO_INPUT, SCENARIO_INSTRUMENT)
    assert _r.variant_of is None or _r.variant_of in SCENARIO_RULES

SCENARIOS = {n: ScenarioDef(n, SCENARIO_RULES[n], _FN[n], _STREAM[n]) for n in SCENARIO_RULES}


def variants_of(name):
    """Варианты «искажён только вход» основного сценария.

    Args:
        name: имя основного сценария.

    Returns:
        Кортеж имён вариантов.
    """
    return tuple(n for n, d in SCENARIOS.items() if d.rule.variant_of == name)


def scenario_rng(seed, name, index):
    """Генератор сценария на окне; одинаковый на всех уровнях.

    Args:
        seed: сид прогона робастности.
        name: имя сценария.
        index: номер окна.

    Returns:
        Генератор случайных чисел.
    """
    return np.random.default_rng([int(seed), SCENARIOS[name].stream, int(index)])


def dither_rng(seed, index):
    """Генератор шума непрерывности окна; общий для всех сценариев и уровней.

    Args:
        seed: сид прогона робастности.
        index: номер окна.

    Returns:
        Генератор случайных чисел.
    """
    return np.random.default_rng([int(seed), _DITHER_STREAM, int(index)])


def apply_scenario(w: AugWindow, name, level, rng, params=None):
    """Применить сценарий к окну на месте и проверить контракт сценария.

    Args:
        w: окно.
        name: имя сценария.
        level: уровень деградации.
        rng: генератор сценария.
        params: параметры сценария; по умолчанию - из правила сценария.

    Returns:
        То же окно.

    Raises:
        RuntimeError: сценарий изменил маску цели или, будучи отказом входа, саму цель.
    """
    d = SCENARIOS[name]
    p = {**d.rule.params, **(params or {})}
    y0, ym0 = w.y.copy(), w.y_mask.copy()
    d.fn(w, float(level), rng, p)
    if not np.array_equal(w.y_mask, ym0):
        raise RuntimeError(f"сценарий {name} изменил маску цели: сценарии не решают, "
                           f"какие часы цели оцениваются")
    if not d.target and not np.array_equal(w.y, y0):
        raise RuntimeError(f"сценарий {name} класса «отказ входа» изменил цель: такой "
                           f"сценарий портит только вход модели")
    w.applied[f"scenario:{name}"] = float(level)
    return w


def instrument_reference(name, level, w: AugWindow, mu_clim, rng, params=None):
    """Эталон скилла на горизонте: климатология, записанная тем же прибором, что цель.

    Если сценарий искажает цель, эталон искажается тем же преобразованием. Иначе
    знаменатель скилла растёт вместе с поломкой прибора, и скилл улучшается оттого, что
    прибор сломан. Преобразование не описывается отдельно, его выполняет сам сценарий:
    он применяется к пробному окну с той же длиной истории и теми же метаданными, пустой
    историей и климатологией на месте цели. Если сценарий цель не трогает, эталон равен
    климатологии.

    Args:
        name: имя сценария.
        level: уровень сценария.
        w: окно до применения сценария; из него берутся длина истории и метаданные.
        mu_clim: климатология на часах горизонта, форма (H,).
        rng: генератор случайных чисел сценария, заведённый так же, как для цели.
        params: параметры сценария; None значит значения по умолчанию.

    Returns:
        Эталон формы (H,), float32. До целых градусов он не округляется: это среднее,
        а не отдельная запись прибора.
    """
    ref = np.array(mu_clim, np.float32)
    if not SCENARIOS[name].rule.target:
        return ref
    probe = AugWindow(x=np.zeros_like(w.x), m=np.zeros_like(w.m), y=ref,
                      y_mask=np.ones(ref.shape, np.float32), L=w.L, hour=w.hour,
                      lat=w.lat, lon=w.lon, elev=w.elev, qc_elev=w.qc_elev)
    apply_scenario(probe, name, level, rng, params)
    return probe.y


def level_label(name, level):
    """Подпись уровня для таблиц и осей.

    Args:
        name: имя сценария.
        level: уровень деградации.

    Returns:
        Строка подписи.
    """
    if name == "drop_channel":
        return DROP_CHANNEL_LABELS[int(level)]
    if SCENARIOS[name].rule.integer:
        return str(int(level))
    return f"{level:g}"


__all__ = ["DITHER_SCENARIOS", "DROP_CHANNEL_LABELS", "SCENARIOS", "ScenarioDef",
           "apply_scenario", "dither_rng", "drift_offset", "drift_target", "instrument_reference",
           "level_label", "scenario_rng", "variants_of"]
