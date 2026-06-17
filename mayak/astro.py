"""Детерминированные признаки Солнца, календаря и термодинамики"""
import math
import torch
import torch.nn.functional as F


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
    g = torch.log(rh / 100.0) + 17.625 * T / (243.04 + T)
    return 243.04 * g / (17.625 - g)