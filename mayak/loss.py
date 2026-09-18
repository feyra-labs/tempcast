"""Функции потерь МАЯК и нейробейзлайнов.

Pinball считается только по валидным часам цели: среднее с весами маски,
нормированное на сумму весов.
"""
import torch
from mayak.constants import QUANTILES

SIGMA_CLAMP = (0.8, 12.0)


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


def _scale(out):
    return out["sigma_c"].detach().clamp(*SIGMA_CLAMP)


def mayak_loss(out, y, y_mask):
    loss = pinball(out["q"], y, y_mask, _scale(out))

    kl = out["kl"]
    energy = (out["a_re"] ** 2 + out["a_im"] ** 2).mean()
    dead = (out["Eg"].sum(-1) < 0.05).float()
    anchor = ((out["ratio"] - 1.0) ** 2 * dead).mean()
    r_anchor = (out["r"] ** 2 * dead).mean()

    return loss + 1e-3 * kl + 1e-4 * energy + 1e-2 * anchor + 0.1 * r_anchor


def pinball_loss(out, y, y_mask):
    """Только pinball по 7 квантилям в аномальной шкале для нейробейзлайнов"""
    return pinball(out["q"], y, y_mask, _scale(out))
