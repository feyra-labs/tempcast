"""Эмпирическая климатология станции гармонической регрессией по (doy, hour)."""
import math

import numpy as np

ABS_TO_SD = math.sqrt(math.pi / 2.0)
SCALE_FLOOR_FRAC = 0.2


def _design(doy, hour, n_year=3, n_day=3):
    """Матрица признаков: константа + годовые + суточные + смешанные гармоники"""
    cols = [np.ones_like(doy)]
    wy, wd = 2 * np.pi / 365.24, 2 * np.pi / 24.0
    for k in range(1, n_year + 1):
        cols += [np.cos(k * wy * doy), np.sin(k * wy * doy)]
    for k in range(1, n_day + 1):
        cols += [np.cos(k * wd * hour), np.sin(k * wd * hour)]
    cols += [np.cos(wy * doy) * np.cos(wd * hour), np.cos(wy * doy) * np.sin(wd * hour)]
    return np.stack(cols, axis=-1)


class Climatology:
    """Гармоническая климатология одной станции: среднее, скалярный σ и масштаб остатка."""

    @classmethod
    def from_params(cls, beta, sigma, scale_beta, n_year=3, n_day=3,
                    scale_n_year=2, scale_n_day=2):
        """Восстановление из кэша без повторной подгонки."""
        c = cls(n_year, n_day, scale_n_year, scale_n_day)
        c.beta = np.asarray(beta, np.float64)
        c.sigma = float(sigma)
        c.scale_beta = None if scale_beta is None else np.asarray(scale_beta, np.float64)
        return c

    def __init__(self, n_year=3, n_day=3, scale_n_year=2, scale_n_day=2):
        self.n_year, self.n_day = n_year, n_day
        self.scale_n_year, self.scale_n_day = scale_n_year, scale_n_day
        self.beta = None
        self.sigma = None
        self.scale_beta = None

    def fit(self, doy, hour, T, mask, min_valid=24 * 30):
        m = mask > 0
        if m.sum() < min_valid:
            raise ValueError(f"мало валидных часов для климатологии: {int(m.sum())} < {min_valid}")
        d, h = doy[m], hour[m]
        A = _design(d, h, self.n_year, self.n_day)
        y = T[m]
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.beta
        self.sigma = float(np.sqrt(np.mean(resid ** 2)) + 1e-6)
        As = _design(d, h, self.scale_n_year, self.scale_n_day)
        self.scale_beta, *_ = np.linalg.lstsq(As, np.abs(resid) * ABS_TO_SD, rcond=None)
        return self

    def predict(self, doy, hour):
        A = _design(np.asarray(doy, float), np.asarray(hour, float), self.n_year, self.n_day)
        return A @ self.beta

    def scale(self, doy, hour):
        """Масштаб остатка в моменты (doy, hour), °C; всегда ≥ SCALE_FLOOR_FRAC · σ > 0."""
        if self.scale_beta is None:
            raise RuntimeError("у климатологии нет масштаба остатка — пересоберите кэш "
                               "(python scripts/build_cache.py)")
        A = _design(np.asarray(doy, float), np.asarray(hour, float),
                    self.scale_n_year, self.scale_n_day)
        return np.maximum(A @ self.scale_beta, SCALE_FLOOR_FRAC * self.sigma)
