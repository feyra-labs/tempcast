"""Слои МАЯК и общие для них функции."""
import torch


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Обратная функция к softplus: сырой параметр, который даёт заданное значение."""
    return torch.log(torch.expm1(y))
