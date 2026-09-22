"""Сценарии робастности: преобразования окна для обученной модели.

Сценарий - это преобразование одного окна (``AugWindow`` из ``mayak.data.augment``),
а не отдельная копия датасета. Контракт каждого сценария - ``SCENARIO_RULES`` в
``mayak/config.py`` (класс, искажается ли цель, границы уровня, параметры), здесь -
только реализация.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mayak.config import SCENARIO_INPUT, SCENARIO_INSTRUMENT, SCENARIO_RULES, ScenarioRule
from mayak.constants import H, L_MAX
from mayak.data.augment import P, RH, T, AugWindow, apply_one
from mayak.data.qc import CHANNELS, PHYS

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
    """Постоянное смещение T - та же функция, что аугментация обучения: история и цель."""
    apply_one(w, "offset", dict(b=float(level)))


def _offset_input(w, level, rng, p):
    w.x[:, T] += np.float32(level) * (w.m[:, T] > 0)


def drift_history(w, rate_per_day):
    """Дрейф на часах истории: 0 в первом часе фактической истории (прибор откалиброван)
    и rate·(k − h0)/24 далее. (L_MAX,) float32."""
    k = np.arange(L_MAX, dtype=np.float64) - w.h0
    return (np.float64(rate_per_day) / 24.0 * np.maximum(k, 0.0)).astype(np.float32)


def drift_target(w, rate_per_day):
    """Продолжение того же дрейфа на горизонте: rate·(L + j)/24, j = 0..H−1."""
    j = w.L + np.arange(H, dtype=np.float64)
    return (np.float64(rate_per_day) / 24.0 * j).astype(np.float32)


def _drift(w, level, rng, p):
    w.x[:, T] += drift_history(w, level) * (w.m[:, T] > 0)
    w.y[:] = w.y + drift_target(w, level) * (w.y_mask > 0)


def _drift_input(w, level, rng, p):
    w.x[:, T] += drift_history(w, level) * (w.m[:, T] > 0)


def _scale(w, level, rng, p):
    """Ошибка масштаба T: множитель 1 + level и на истории, и на цели."""
    k = np.float32(1.0 + level)
    w.x[:, T] = np.where(w.m[:, T] > 0, w.x[:, T] * k, w.x[:, T])
    w.y[:] = np.where(w.y_mask > 0, w.y * k, w.y)


def _noise(w, level, rng, p):
    """Гауссов шум: σ_T = level, σ_P и σ_RH - в отношении ``sd_ratio``; RH насыщается."""
    z = rng.standard_normal((L_MAX, 3)).astype(np.float32)
    sd = np.float32(level) * np.asarray(p["sd_ratio"], np.float32)
    w.x[:] = w.x + z * sd * (w.m > 0)
    ok = w.m[:, RH] > 0
    w.x[:, RH] = np.where(ok, np.clip(w.x[:, RH], 0.0, 100.0), w.x[:, RH])


def _spikes(w, level, rng, p):
    """Одиночные выбросы: на доле ``level`` валидных точек значение сдвинуто на ±magnitude."""
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
assert set(_FN) == set(_STREAM) == set(SCENARIO_RULES)
for _r in SCENARIO_RULES.values():
    assert _r.kind in (SCENARIO_INPUT, SCENARIO_INSTRUMENT)
    assert _r.variant_of is None or _r.variant_of in SCENARIO_RULES

SCENARIOS = {n: ScenarioDef(n, SCENARIO_RULES[n], _FN[n], _STREAM[n]) for n in SCENARIO_RULES}


def variants_of(name):
    """Варианты «искажён только вход» основного сценария name."""
    return tuple(n for n, d in SCENARIOS.items() if d.rule.variant_of == name)


def scenario_rng(seed, name, index):
    """Генератор сценария name на окне index - одинаковый на всех уровнях."""
    return np.random.default_rng([int(seed), SCENARIOS[name].stream, int(index)])


def apply_scenario(w: AugWindow, name, level, rng, params=None):
    """Применить сценарий name уровня level к окну w (на месте) и проверить контракт.

    params - параметры сценария (по умолчанию - из ``SCENARIO_RULES``).
    """
    d = SCENARIOS[name]
    p = {**d.rule.params, **(params or {})}
    y0, ym0 = w.y.copy(), w.y_mask.copy()
    d.fn(w, float(level), rng, p)
    if not np.array_equal(w.y_mask, ym0):
        raise RuntimeError(f"сценарий {name} изменил маску цели (блок 1.6)")
    if not d.target and not np.array_equal(w.y, y0):
        raise RuntimeError(f"сценарий {name} класса «отказ входа» изменил цель (блок 12.2)")
    w.applied[f"scenario:{name}"] = float(level)
    return w


def point_qc_mask(x, m):
    """Поточечный QC рантайма (``mayak.data.qc.point_qc``) на всём окне сразу:
    значение вне физического диапазона канала - невалидно. Маска (N, 3) float32."""
    x = np.asarray(x)
    lo = np.array([PHYS[c][0] for c in CHANNELS], np.float32)
    hi = np.array([PHYS[c][1] for c in CHANNELS], np.float32)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(x) & (x >= lo) & (x <= hi)
    return ((np.asarray(m) > 0) & ok).astype(np.float32)


def level_label(name, level):
    """Подпись уровня для таблиц и осей."""
    if name == "drop_channel":
        return DROP_CHANNEL_LABELS[int(level)]
    if SCENARIOS[name].rule.integer:
        return str(int(level))
    return f"{level:g}"


__all__ = ["DROP_CHANNEL_LABELS", "SCENARIOS", "ScenarioDef", "apply_scenario", "drift_history",
           "drift_target", "level_label", "point_qc_mask", "scenario_rng", "variants_of"]
