"""Точечная выборка из растров: зона Кёппена и высота из цифровой модели рельефа.

Одна реализация на оба сборщика (реанализ и наблюдения реальной сети), чтобы зона
и высота обучающих и внешних станций снимались одинаково. Чтение - через
``rasterio.DatasetReader.sample``: растр не загружается целиком, поэтому работает и
глобальная мозаика ЦМР (например, VRT над тайлами Copernicus GLO-90).
"""
from __future__ import annotations

import math

from mayak.zones import KG_TIF_CODE, UNKNOWN_ZONE


class RasterSampler:
    """Значение первого (или заданного) канала растра в точке (lat, lon)."""

    def __init__(self, path, band=1):
        import rasterio
        self.path = str(path)
        self.ds = rasterio.open(self.path)
        self.band = int(band)
        self.nodata = self.ds.nodata

    def __call__(self, lat, lon):
        lat, lon = float(lat), float(lon)
        b = self.ds.bounds
        if not (b.left <= lon < b.right and b.bottom < lat <= b.top):
            return None
        val = next(self.ds.sample([(lon, lat)], indexes=self.band))[0]
        val = float(val)
        if (self.nodata is not None and val == float(self.nodata)) or not math.isfinite(val):
            return None
        return val

    def close(self):
        self.ds.close()


def koppen_reader(path):
    """Карта Бека и соавт. (коды 1..30) → функция (lat, lon) → полная зона Кёппена."""
    sampler = RasterSampler(path)

    def get(lat, lon):
        v = sampler(lat, lon)
        if v is None or v <= 0:
            return UNKNOWN_ZONE
        return KG_TIF_CODE.get(int(round(v)), UNKNOWN_ZONE)

    return get


def dem_reader(path):
    """Растр ЦМР → функция (lat, lon) → высота, м, или None."""
    return RasterSampler(path)


__all__ = ["RasterSampler", "dem_reader", "koppen_reader"]
