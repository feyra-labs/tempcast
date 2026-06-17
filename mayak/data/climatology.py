"""Эмпирическая климатология гармонической регрессией по doy, hour"""
import numpy as np

def _design(doy, hour, n_year=3, n_day=3):
    """Матрица признаков: константа + годовые + суточные + смешанные гармоники"""
    cols = [np.ones_like(doy)]
    wy, wd = 2 * np.pi / 365.24, 2 * np.pi / 24.0
    for k in range(1, n_year + 1):
        cols += [np.cos(k * wy * doy), np.sin(k * wy * doy)]
    for k in range(1, n_day + 1):
        cols += [np.cos(k * wd * hour), np.sin(k * wd * hour)]
    # смешанные члены: суточная амплитуда «дышит» с сезоном
    cols += [np.cos(wy * doy) * np.cos(wd * hour), np.cos(wy * doy) * np.sin(wd * hour)]
    return np.stack(cols, axis=-1)

class Climatology:
    """Гармоническая климатология одной станции + остаток"""
    def __init__(self, n_year=3, n_day=3):
        self.n_year, self.n_day = n_year, n_day
        self.beta = None
        self.sigma = None 

    def fit(self, doy, hour, T, mask):
        m = mask > 0
        A = _design(doy[m], hour[m], self.n_year, self.n_day)
        y = T[m]
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.beta
        self.sigma = float(np.sqrt(np.mean(resid ** 2)) + 1e-6)
        return self

    def predict(self, doy, hour):
        A = _design(np.asarray(doy, float), np.asarray(hour, float), self.n_year, self.n_day)
        return A @ self.beta