"""Функция потерь — одна на все модели.

    loss = forecast_loss(out, batch) + регуляризаторы архитектуры (не зависят от цели)

forecast_loss — pinball по 7 квантилям, где ошибка каждой пары «окно × лид» делится
на нормировочный масштаб ``batch["norm_scale"]``: климатологический масштаб остатка
станции на часах горизонта, посчитанный датасетом по данным (mayak/data/climatology.py).
"""
import torch

from mayak.constants import QUANTILES

NORM_SCALE_CLAMP = (0.8, 12.0)


def masked_mean(x, w):
    w = w.to(x.dtype).expand_as(x)
    num = torch.where(w > 0, x * w, torch.zeros_like(x)).sum()
    return num / w.sum().clamp_min(1e-8)


def pinball(q, y, y_mask, scale):
    taus = torch.tensor(QUANTILES, dtype=q.dtype, device=q.device)
    m = (y_mask > 0).to(q.dtype)
    y = torch.where(m > 0, y, torch.zeros_like(y))
    err = (y[..., None] - q) / scale[..., None]
    per_pair = torch.maximum(taus * err, (taus - 1.0) * err).mean(-1)
    return masked_mean(per_pair, m)


def loss_scale(batch):
    """Нормировка функции потерь: данные батча, обрезанные общими константами."""
    if "norm_scale" not in batch:
        raise KeyError("в батче нет norm_scale: нормировка функции потерь приходит из датасета "
                       "(климатологический масштаб), а не из выхода модели")
    return batch["norm_scale"].detach().clamp(*NORM_SCALE_CLAMP)


def forecast_loss(out, batch):
    """Общая часть функции потерь всех моделей: зависит только от q, y, y_mask, norm_scale."""
    q = out["q"]
    return pinball(q, batch["y"], batch["y_mask"], loss_scale(batch).to(q.dtype))


def mayak_regularizers(out):
    """Регуляризаторы МАЯК: расхождение паспорта с прайором, энергия мод, якорные члены."""
    kl = out["kl"]
    energy = (out["a_re"] ** 2 + out["a_im"] ** 2).mean()
    dead = (out["Eg"].sum(-1) < 0.05).float()
    anchor = ((out["ratio"] - 1.0) ** 2 * dead).mean()
    r_anchor = (out["r"] ** 2 * dead).mean()
    return 1e-3 * kl + 1e-4 * energy + 1e-2 * anchor + 0.1 * r_anchor


def mayak_loss(out, batch):
    return forecast_loss(out, batch) + mayak_regularizers(out)


def pinball_loss(out, batch):
    """Нейробейзлайны: регуляризаторов нет, только общая часть."""
    return forecast_loss(out, batch)
