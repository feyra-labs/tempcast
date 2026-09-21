"""Контракт данных и физические константы МАЯК."""
import torch

H = 168  # горизонт прогноза, ч
L_MAX = 672  # максимальная длина истории, ч

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
NQ = len(QUANTILES)

MAGNUS_A = 17.625
MAGNUS_B = 243.04


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """x такой, что softplus(x) = y"""
    return torch.log(torch.expm1(y))
