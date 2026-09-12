"""Детерминированные признаки Солнца, календаря и термодинамики"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from mayak.constants import MAGNUS_A, MAGNUS_B


def astro_features(doy, hour_utc, lat_deg, lon_deg):
    """Солнечно-календарные признаки"""
    B = 2 * math.pi * (doy - 81.0) / 364.0
    eot_min = 9.87 * torch.sin(2 * B) - 7.53 * torch.cos(B) - 1.5 * torch.sin(B)

    t_sol = (hour_utc + lon_deg / 15.0 + eot_min / 60.0) % 24.0

    decl = -0.40928 * torch.cos(2 * math.pi * (doy + 10.0) / 365.24)

    phi = lat_deg * math.pi / 180.0
    hra = (t_sol - 12.0) * (math.pi / 12.0)
    cz = (torch.sin(phi) * torch.sin(decl)
          + torch.cos(phi) * torch.cos(decl) * torch.cos(hra))

    dphase = 2 * math.pi * t_sol / 24.0
    yphase = 2 * math.pi * doy / 365.24
    return (torch.sin(dphase), torch.cos(dphase), cz, F.relu(cz),
            torch.sin(yphase), torch.cos(yphase))


def dewpoint_c(T, RH):
    """Точка росы по Магнусу"""
    rh = RH.clamp(1.0, 100.0)
    gamma = (torch.log(rh / 100.0) + MAGNUS_A * T / (MAGNUS_B + T))
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


def rh_from_dewpoint(T, Td):
    """Обратная формула Магнуса."""
    gamma_T = MAGNUS_A * T / (MAGNUS_B + T)
    gamma_Td = MAGNUS_A * Td / (MAGNUS_B + Td)
    rh = 100.0 * np.exp(gamma_Td - gamma_T)
    return np.clip(rh, 0.0, 100.0).astype(np.float32)
