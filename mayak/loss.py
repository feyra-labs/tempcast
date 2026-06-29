import torch
from mayak.constants import QUANTILES


def mayak_loss(out, y):
    sg = out["sigma_c"].detach().clamp(0.8, 12.0)
    taus = torch.tensor(QUANTILES, dtype=y.dtype, device=y.device)

    err = (y[..., None] - out["q"]) / sg[..., None]
    pinball = torch.maximum(taus * err, (taus - 1.0) * err).mean()

    kl = out["kl"]

    energy = (out["a_re"] ** 2 + out["a_im"] ** 2).mean()

    dead = (out["Eg"].sum(-1) < 0.05).float()
    anchor = ((out["ratio"] - 1.0) ** 2 * dead).mean()
    r_anchor = (out["r"] ** 2 * dead).mean()

    return pinball + 1e-3 * kl + 1e-4 * energy + 1e-2 * anchor + 0.1 * r_anchor

def pinball_loss(out, y):
    """Только pinball по 7 квантилям в аномальной шкале — для нейробейзлайнов"""
    sg = out["sigma_c"].detach().clamp(0.8, 12.0)
    taus = torch.tensor(QUANTILES, dtype=y.dtype, device=y.device)
    err = (y[..., None] - out["q"]) / sg[..., None]
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()