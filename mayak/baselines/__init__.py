"""Бейзлайны прогноза.

Статистические эталоны (``statistical``): климатология, затухающая персистентность,
сезонно-наивный прогноз. Обучаемые (``neural``, ``lru``, ``patchtst``): GRU seq2seq,
DLinear, LRU, PatchTST — обучаются по единому протоколу (mayak/protocol.py).

У каждого бейзлайна есть карточка (``cards``): источник, что взято, отличия от
оригинала и их причины. Карточка дописана в докстринг класса или функции и
собирается в mayak/baselines/README.md (scripts/baseline_cards.py).
"""
from mayak.baselines.cards import REGISTRY as CARDS
from mayak.baselines.cards import BaselineCard, card_for, display_name, markdown
from mayak.baselines.lru import (LRUForecaster, lru_recurrent, lru_scan_associative,
                                 lru_scan_chunked)
from mayak.baselines.neural import (DLinear, GRUSeq2Seq, _median_centered_offsets,
                                    median_centered_offsets)
from mayak.baselines.patchtst import PatchTST
from mayak.baselines.statistical import (DAMPED_VAR_FLOOR, RECENT_HOURS, RECENT_MIN_VALID, ZQ,
                                         climatology_forecast, damped_coefficients,
                                         damped_persistence_forecast, fit_climatologies,
                                         fit_damped_persistence, quantiles_from_normal,
                                         recent_anomaly, seasonal_naive_forecast)

NEURAL = {"gru": GRUSeq2Seq, "dlinear": DLinear, "lru": LRUForecaster, "patchtst": PatchTST}
STATISTICAL = ("climatology", "damped_persistence", "seasonal_naive")
ORDER = STATISTICAL + tuple(NEURAL)

__all__ = ["BaselineCard", "CARDS", "DAMPED_VAR_FLOOR", "DLinear", "GRUSeq2Seq", "LRUForecaster",
           "NEURAL", "ORDER", "PatchTST", "RECENT_HOURS", "RECENT_MIN_VALID", "STATISTICAL", "ZQ",
           "card_for", "climatology_forecast", "damped_coefficients", "damped_persistence_forecast",
           "display_name", "fit_climatologies", "fit_damped_persistence", "lru_recurrent",
           "lru_scan_associative", "lru_scan_chunked", "markdown", "median_centered_offsets",
           "quantiles_from_normal", "recent_anomaly", "seasonal_naive_forecast"]
