"""Функция потерь, общая для всех моделей."""
import torch

from mayak.constants import QUANTILES

NORM_SCALE_CLAMP = (0.8, 12.0)


def masked_mean(x, w):
    w = w.to(x.dtype).expand_as(x)
    num = torch.where(w > 0, x * w, torch.zeros_like(x)).sum()
    return num / w.sum().clamp_min(1e-8)


def pinball_terms(q, y, y_mask, scale):
    """Нормированный pinball каждой пары окна и лида и её вес."""
    taus = torch.tensor(QUANTILES, dtype=q.dtype, device=q.device)
    m = (y_mask > 0).to(q.dtype)
    y = torch.where(m > 0, y, torch.zeros_like(y))
    err = (y[..., None] - q) / scale[..., None]
    per_pair = torch.maximum(taus * err, (taus - 1.0) * err).mean(-1)
    return per_pair, m


def loss_scale(batch):
    """Нормировка функции потерь: масштаб из батча, обрезанный общими пределами."""
    if "norm_scale" not in batch:
        raise KeyError("в батче нет norm_scale: нормировка функции потерь приходит из датасета "
                       "(климатологический масштаб), а не из выхода модели")
    return batch["norm_scale"].detach().clamp(*NORM_SCALE_CLAMP)


def forecast_terms(out, batch):
    """Общая часть функции потерь по парам окна и лида, до усреднения."""
    q = out["q"]
    return pinball_terms(q, batch["y"], batch["y_mask"], loss_scale(batch).to(q.dtype))


def forecast_loss(out, batch):
    """Общая часть функции потерь всех моделей."""
    per_pair, m = forecast_terms(out, batch)
    return masked_mean(per_pair, m)


def mayak_regularizers(out):
    """Регуляризаторы МАЯК, не зависящие от цели."""
    kl = out["kl"]
    energy = (out["a_re"] ** 2 + out["a_im"] ** 2).mean()
    dead = (out["Eg"].sum(-1) < 0.05).float()
    anchor = ((out["ratio"] - 1.0) ** 2 * dead).mean()
    return 1e-3 * kl + 1e-4 * energy + 1e-2 * anchor
