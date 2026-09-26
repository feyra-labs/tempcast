"""Бейзлайны прогноза.

Статистические эталоны: климатология, затухающая персистентность, сезонно-наивный
прогноз. Обучаемые: GRU, DLinear, LRU, PatchTST; все обучаются по единому протоколу.
"""
from mayak.baselines.lru import (LRUForecaster, lru_recurrent, lru_scan_associative,
                                 lru_scan_chunked)
from mayak.baselines.neural import DLinear, GRUSeq2Seq, LeadHead, recurrent_inputs
from mayak.baselines.neural import median_centered_offsets
from mayak.baselines.patchtst import PatchTST
from mayak.baselines.statistical import (DAMPED_VAR_FLOOR, RECENT_HOURS, RECENT_MIN_VALID, ZQ,
                                         climatology_forecast, damped_coefficients,
                                         damped_persistence_forecast, fit_climatologies,
                                         fit_damped_persistence, quantiles_from_normal,
                                         recent_anomaly, seasonal_naive_forecast)

NEURAL = {"gru": GRUSeq2Seq, "dlinear": DLinear, "lru": LRUForecaster, "patchtst": PatchTST}
STATISTICAL = ("climatology", "damped_persistence", "seasonal_naive")
ORDER = STATISTICAL + tuple(NEURAL)
SIZE_BAND = (0.75, 1.5)

__all__ = ["DAMPED_VAR_FLOOR", "DLinear", "GRUSeq2Seq", "LRUForecaster", "LeadHead", "NEURAL",
           "ORDER", "PatchTST", "RECENT_HOURS", "RECENT_MIN_VALID", "SIZE_BAND", "STATISTICAL",
           "ZQ", "climatology_forecast", "damped_coefficients", "damped_persistence_forecast",
           "fit_climatologies", "fit_damped_persistence", "lru_recurrent",
           "lru_scan_associative", "lru_scan_chunked", "median_centered_offsets",
           "quantiles_from_normal", "recent_anomaly", "recurrent_inputs",
           "seasonal_naive_forecast"]
