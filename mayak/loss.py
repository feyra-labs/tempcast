"""Функция потерь - одна на все модели.

Функция потерь складывается из общей части и регуляризаторов архитектуры, которые от
цели не зависят. Общая часть - pinball по семи квантилям, где ошибка каждой пары окна и
лида делится на нормировочный масштаб из батча: климатологический масштаб остатка
станции на часах горизонта, посчитанный датасетом по данным.
"""
import torch

from mayak.constants import QUANTILES

NORM_SCALE_CLAMP = (0.8, 12.0)


def masked_mean(x, w):
    w = w.to(x.dtype).expand_as(x)
    num = torch.where(w > 0, x * w, torch.zeros_like(x)).sum()
    return num / w.sum().clamp_min(1e-8)


def pinball_terms(q, y, y_mask, scale):
    """Нормированный pinball каждой пары окна и лида и её вес.

    Args:
        q: квантили прогноза, форма (B, H, 7).
        y: цель, форма (B, H).
        y_mask: маска цели, форма (B, H).
        scale: нормировочный масштаб, форма (B, H).

    Returns:
        Пара тензоров формы (B, H): pinball, усреднённый по квантилям, и вес пары -
        единица у валидной цели, ноль у пропущенной.
    """
    taus = torch.tensor(QUANTILES, dtype=q.dtype, device=q.device)
    m = (y_mask > 0).to(q.dtype)
    y = torch.where(m > 0, y, torch.zeros_like(y))
    err = (y[..., None] - q) / scale[..., None]
    per_pair = torch.maximum(taus * err, (taus - 1.0) * err).mean(-1)
    return per_pair, m


def loss_scale(batch):
    """Нормировка функции потерь: масштаб из батча, обрезанный общими пределами.

    Масштаб приходит из данных, а не из выхода модели, поэтому градиент через него не
    идёт.

    Args:
        batch: батч с ключом ``norm_scale``, форма (B, H).

    Returns:
        Тензор формы (B, H).

    Raises:
        KeyError: в батче нет ``norm_scale``.
    """
    if "norm_scale" not in batch:
        raise KeyError("в батче нет norm_scale: нормировка функции потерь приходит из датасета "
                       "(климатологический масштаб), а не из выхода модели")
    return batch["norm_scale"].detach().clamp(*NORM_SCALE_CLAMP)


def forecast_terms(out, batch):
    """Общая часть функции потерь по парам окна и лида, до усреднения.

    Args:
        out: выход модели с квантилями под ключом q.
        batch: батч с целью, её маской и нормировочным масштабом.

    Returns:
        Пара тензоров формы (B, H): нормированный pinball пары и её вес.
    """
    q = out["q"]
    return pinball_terms(q, batch["y"], batch["y_mask"], loss_scale(batch).to(q.dtype))


def forecast_loss(out, batch):
    """Общая часть функции потерь всех моделей.

    Зависит только от квантилей прогноза, цели, её маски и нормировочного масштаба.

    Args:
        out: выход модели с квантилями под ключом q.
        batch: батч с целью, её маской и нормировочным масштабом.

    Returns:
        Скаляр: средний нормированный pinball по валидным парам окна и лида.
    """
    per_pair, m = forecast_terms(out, batch)
    return masked_mean(per_pair, m)


def mayak_regularizers(out):
    """Регуляризаторы МАЯК, не зависящие от цели.

    Три члена: расхождение паспорта с приором, энергия мод и притяжение масштаба
    интервала к климатологическому там, где энергия мод почти нулевая. Отдельного
    штрафа за поправку голов нет: без истории она равна нулю по устройству голов.

    Args:
        out: выход модели с ключами kl, a_re, a_im, Eg и ratio.

    Returns:
        Скаляр, который прибавляется к общей части функции потерь.
    """
    kl = out["kl"]
    energy = (out["a_re"] ** 2 + out["a_im"] ** 2).mean()
    dead = (out["Eg"].sum(-1) < 0.05).float()
    anchor = ((out["ratio"] - 1.0) ** 2 * dead).mean()
    return 1e-3 * kl + 1e-4 * energy + 1e-2 * anchor
