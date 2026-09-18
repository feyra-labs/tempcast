"""Единая конвенция маски валидности (блок 1).

Инвариант хранения (1.1): для любого массива значений ``x`` рядом лежит
маска ``m`` той же формы (по каналам), и ``x == 0`` везде, где ``m == 0``.
Инвариант восстанавливается принудительно в одной точке —
:func:`enforce_invariant` — после контроля качества и после любой
аугментации. Ни один потребитель не читает значения без маски.

Правило отбора окон (1.5): окно ``[t, t + H)`` отбрасывается, если доля
валидных часов цели ниже ``min_valid_frac`` либо в первых ``first_day_hours``
часах нет ни одного валидного. Правило одинаково для обучения и оценки.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetMaskConfig:
    """Параметры отбора окон по маске цели.

    Живёт отдельным датаклассом до блока 6, где войдёт в конфигурацию данных.
    """
    min_valid_frac: float = 0.5
    first_day_hours: int = 24

    def __post_init__(self):
        if not 0.0 <= self.min_valid_frac <= 1.0:
            raise ValueError(f"min_valid_frac вне [0, 1]: {self.min_valid_frac}")
        if self.first_day_hours < 1:
            raise ValueError(f"first_day_hours < 1: {self.first_day_hours}")


DEFAULT_TARGET_MASK = TargetMaskConfig()


def enforce_invariant(x, mask):
    """Бинаризует маску и обнуляет значения там, где маска равна нулю.

    Работает с numpy и torch, возвращает новую пару (x, mask) того же типа.
    Нечисловые значения (NaN, inf) под нулевой маской тоже обнуляются.
    """
    if torch.is_tensor(x):
        m = (mask > 0).to(x.dtype)
        return torch.where(m > 0, x, torch.zeros_like(x)), m
    m = (np.asarray(mask) > 0).astype(np.float32)
    x = np.where(m > 0, x, 0.0).astype(np.float32)
    return x, m


def target_window_ok(mask_T, starts, horizon, cfg: TargetMaskConfig = DEFAULT_TARGET_MASK):
    """Векторная проверка правила 1.5 для набора стартов окон.

    mask_T  — маска канала температуры станции, форма (N,);
    starts  — индексы начала горизонта t (цель — часы [t, t + horizon));
    Возвращает булев массив той же длины, что starts.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=bool)
    if starts.min() < 0 or starts.max() + horizon > len(mask_T):
        raise ValueError("окно цели выходит за пределы ряда")
    c = np.concatenate([[0], np.cumsum(np.asarray(mask_T) > 0, dtype=np.int64)])
    n_valid = c[starts + horizon] - c[starts]
    first = min(cfg.first_day_hours, horizon)
    n_first = c[starts + first] - c[starts]
    need = math.ceil(cfg.min_valid_frac * horizon - 1e-9)
    return (n_valid >= need) & (n_first > 0)


@dataclass
class FilterStats:
    """Счётчик отбраковки окон: сколько кандидатов рассмотрено и сколько принято."""
    candidates: int = 0
    kept: int = 0
    stations_total: int = 0
    stations_dropped: int = 0

    def add(self, n_candidates, n_kept):
        self.candidates += int(n_candidates)
        self.kept += int(n_kept)
        self.stations_total += 1
        if n_kept == 0:
            self.stations_dropped += 1

    @property
    def rejected_frac(self):
        return 1.0 - self.kept / self.candidates if self.candidates else 0.0

    def report(self, name):
        log.info("%s: отбраковано %.2f%% окон по маске цели (%d из %d); "
                 "станций без единого годного окна: %d из %d",
                 name, 100 * self.rejected_frac, self.candidates - self.kept,
                 self.candidates, self.stations_dropped, self.stations_total)
