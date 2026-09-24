"""Выбор обучающих точек на суше: равномерная решётка на сфере и гарантия покрытия зон."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from mayak.zones import KG_TIF_CODE

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
ANTARCTICA_LAT = -60.0
GOLDEN_ANGLE_DEG = 180.0 * (3.0 - math.sqrt(5.0))

DEFAULT_N_POINTS = 350
DEFAULT_MIN_PER_ZONE = 3
DEFAULT_MIN_DIST_KM = 150.0
DEFAULT_GRID_STEP = 0.1
DEFAULT_SEED = 0

POINTS_FIELDS = ("id", "lat", "lon", "koppen")


@dataclass
class ZoneGrid:
    """Коды зон Кёппена на регулярной сетке широт и долгот.

    Attributes:
        codes: коды зон формы (n_lat, n_lon), строки идут с севера на юг. Ноль
            означает море, отсутствие данных или код вне таблицы зон.
        top: широта северного края сетки, градусы.
        left: долгота западного края сетки, градусы.
        step: шаг сетки, градусы.
    """

    codes: np.ndarray
    top: float
    left: float
    step: float

    @property
    def shape(self):
        return self.codes.shape

    def lat_centers(self):
        """Широты центров строк сетки.

        Returns:
            Массив формы (n_lat,), градусы.
        """
        return self.top - (np.arange(self.codes.shape[0]) + 0.5) * self.step

    def lon_centers(self):
        """Долготы центров столбцов сетки.

        Returns:
            Массив формы (n_lon,), градусы.
        """
        return self.left + (np.arange(self.codes.shape[1]) + 0.5) * self.step

    def cell_of(self, lat, lon):
        """Ячейки сетки, в которые попадают точки.

        Args:
            lat: широты, градусы.
            lon: долготы, градусы.

        Returns:
            Кортеж из номеров строк, номеров столбцов и признака того, что точка
            лежит внутри сетки. Для точек вне сетки номера обрезаются до края.
        """
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        n_lat, n_lon = self.codes.shape
        row = np.floor((self.top - lat) / self.step).astype(np.int64)
        col = np.floor((lon - self.left) / self.step).astype(np.int64)
        inside = (row >= 0) & (row < n_lat) & (col >= 0) & (col < n_lon)
        return np.clip(row, 0, n_lat - 1), np.clip(col, 0, n_lon - 1), inside

    def code_at(self, lat, lon):
        """Код зоны в точках, ноль вне сетки.

        Args:
            lat: широты, градусы.
            lon: долготы, градусы.

        Returns:
            Массив кодов той же формы, что и входы.
        """
        row, col, inside = self.cell_of(lat, lon)
        return np.where(inside, self.codes[row, col], 0)

    @classmethod
    def from_raster(cls, path, step=DEFAULT_GRID_STEP):
        """Сетка зон, снятая с растра Кёппена в центрах ячеек заданного шага.

        Значение ячейки берётся из того пикселя растра, в который попадает её
        центр. Растр читается по строкам и целиком в память не загружается.

        Args:
            path: путь к растру с кодами зон в географических координатах.
            step: шаг сетки, градусы.

        Returns:
            Сетка зон с тем же охватом, что у растра.

        Raises:
            ValueError: растр не в географических координатах или шаг не положителен.
        """
        import rasterio
        from rasterio.windows import Window

        if step <= 0:
            raise ValueError(f"шаг сетки должен быть положительным: {step}")
        with rasterio.open(path) as ds:
            if ds.crs is not None and not ds.crs.is_geographic:
                raise ValueError(f"{path}: растр не в географических координатах ({ds.crs})")
            b = ds.bounds
            n_lat = int(math.floor((b.top - b.bottom) / step + 1e-9))
            n_lon = int(math.floor((b.right - b.left) / step + 1e-9))
            grid = cls(np.zeros((n_lat, n_lon), np.uint8), float(b.top), float(b.left),
                       float(step))
            res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
            src_row = np.floor((b.top - grid.lat_centers()) / res_y).astype(np.int64)
            src_col = np.floor((grid.lon_centers() - b.left) / res_x).astype(np.int64)
            src_row = np.clip(src_row, 0, ds.height - 1)
            src_col = np.clip(src_col, 0, ds.width - 1)
            nodata = ds.nodata
            for i, r in enumerate(src_row):
                line = ds.read(1, window=Window(0, int(r), ds.width, 1))[0][src_col]
                line = line.astype(np.float64)
                bad = ~np.isfinite(line)
                if nodata is not None:
                    bad |= line == float(nodata)
                code = np.where(bad, 0, np.rint(line)).astype(np.int64)
                known = np.isin(code, list(KG_TIF_CODE))
                grid.codes[i] = np.where(known, code, 0).astype(np.uint8)
        return grid


def fibonacci_lattice(n, lon_offset=0.0):
    """Узлы решётки Фибоначчи на сфере.

    Узлы идут от северного полюса к южному и почти равномерно покрывают площадь.

    Args:
        n: число узлов.
        lon_offset: поворот решётки вокруг оси Земли, градусы.

    Returns:
        Кортеж из широт и долгот узлов, массивы формы (n,), градусы. Долготы лежат
        в полуинтервале от минус 180 до 180.
    """
    k = np.arange(int(n), dtype=np.float64) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * k / int(n)))
    lon = (k * GOLDEN_ANGLE_DEG + float(lon_offset)) % 360.0 - 180.0
    return lat, lon


def unit_vectors(lat, lon):
    """Точки сферы в виде единичных векторов, для быстрых поисков соседей.

    Args:
        lat: широты, градусы.
        lon: долготы, градусы.

    Returns:
        Массив формы (n, 3).
    """
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1)


def chord_of_km(dist_km):
    """Длина хорды единичной сферы для расстояния по поверхности Земли.

    Args:
        dist_km: расстояние по дуге большого круга, км.

    Returns:
        Длина хорды между двумя точками на этом расстоянии.
    """
    return 2.0 * math.sin(float(dist_km) / (2.0 * EARTH_RADIUS_KM))


def km_of_chord(chord):
    """Расстояние по поверхности Земли для длины хорды единичной сферы.

    Args:
        chord: длина хорды, скаляр или массив.

    Returns:
        Расстояние по дуге большого круга, км.
    """
    c = np.clip(np.asarray(chord, np.float64) / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(c)


@dataclass
class Selection:
    """Результат выбора точек.

    Attributes:
        lat: широты точек, градусы.
        lon: долготы точек, градусы.
        codes: коды зон Кёппена точек.
        source: откуда взялась точка: lattice для узла решётки, zone для
            точки, добранной ради покрытия зоны.
        n_lattice: число узлов решётки на всей сфере.
        zones_with_land: коды зон, в которых есть суша в разрешённых широтах.
        relaxed: коды зон, первая точка которых поставлена ближе минимального
            расстояния, потому что иначе зона осталась бы без точек.
        short: зоны, не добравшие минимума, и сколько точек в них получилось.
    """

    lat: np.ndarray
    lon: np.ndarray
    codes: np.ndarray
    source: np.ndarray
    n_lattice: int
    zones_with_land: list
    relaxed: list = field(default_factory=list)
    short: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.lat.size)

    def zone_counts(self):
        """Число точек по кодам зон.

        Returns:
            Словарь из кода зоны в число точек, для всех зон с сушей.
        """
        out = {int(z): 0 for z in self.zones_with_land}
        for c in self.codes:
            out[int(c)] = out.get(int(c), 0) + 1
        return out

    def rows(self):
        """Строки выходной таблицы точек.

        Returns:
            Список словарей с полями id, lat, lon, koppen в порядке выбора.
        """
        width = max(4, len(str(max(len(self) - 1, 0))))
        return [dict(id=f"p{i:0{width}d}", lat=round(float(a), 4), lon=round(float(b), 4),
                     koppen=KG_TIF_CODE[int(c)])
                for i, (a, b, c) in enumerate(zip(self.lat, self.lon, self.codes))]


class _Selector:
    """Выбор точек при заданном числе узлов решётки.

    Пулы кандидатов по зонам не зависят от числа узлов, поэтому считаются один раз
    и переиспользуются при поиске размера решётки.
    """

    def __init__(self, grid, min_per_zone, min_dist_km, include_antarctica, lon_offset):
        self.grid = grid
        self.min_per_zone = int(min_per_zone)
        self.chord = chord_of_km(min_dist_km) if min_dist_km > 0 else 0.0
        self.lon_offset = float(lon_offset)
        lat_c = grid.lat_centers()
        allowed_rows = np.ones_like(lat_c, bool) if include_antarctica else lat_c >= ANTARCTICA_LAT
        self.land = (grid.codes > 0) & allowed_rows[:, None]
        self.zones = sorted(int(z) for z in np.unique(grid.codes[self.land]))
        weight = np.cos(np.radians(lat_c))[:, None]
        total = float(np.sum(weight * np.ones(grid.shape[1])[None, :]))
        self.land_frac = float(np.sum(weight * self.land)) / max(total, 1e-12)
        self._pools = {}

    def pool(self, zone):
        """Ячейки зоны: плоские номера и единичные векторы центров."""
        if zone not in self._pools:
            flat = np.flatnonzero(self.land & (self.grid.codes == zone))
            row, col = np.unravel_index(flat, self.grid.shape)
            lat, lon = self.grid.lat_centers()[row], self.grid.lon_centers()[col]
            self._pools[zone] = (flat, lat, lon, unit_vectors(lat, lon))
        return self._pools[zone]

    def run(self, n_lattice):
        grid = self.grid
        node_lat, node_lon = fibonacci_lattice(n_lattice, self.lon_offset)
        row, col, inside = grid.cell_of(node_lat, node_lon)
        keep = inside & self.land[row, col]
        flat = row[keep] * grid.shape[1] + col[keep]
        _, first = np.unique(flat, return_index=True)
        flat = flat[np.sort(first)]
        r, c = np.unravel_index(flat, grid.shape)
        lat, lon = grid.lat_centers()[r], grid.lon_centers()[c]
        codes = grid.codes[r, c].astype(np.int64)

        chosen_xyz, chosen = [], []
        for i, v in enumerate(unit_vectors(lat, lon)):
            if chosen_xyz and self.chord > 0:
                d = np.min(np.linalg.norm(np.asarray(chosen_xyz) - v, axis=1))
                if d < self.chord:
                    continue
            chosen_xyz.append(v)
            chosen.append(i)
        out_lat, out_lon = list(lat[chosen]), list(lon[chosen])
        out_code = list(codes[chosen])
        source = ["lattice"] * len(chosen)

        node_tree = cKDTree(unit_vectors(node_lat, node_lon))
        taken = set(int(x) for x in flat[chosen])
        relaxed, short = [], {}
        for zone in self.zones:
            have = sum(1 for z in out_code if z == zone)
            if have >= self.min_per_zone:
                continue
            p_flat, p_lat, p_lon, p_xyz = self.pool(zone)
            score, _ = node_tree.query(p_xyz)
            order = np.lexsort((p_flat, score))
            while have < self.min_per_zone:
                if chosen_xyz:
                    dist, _ = cKDTree(np.asarray(chosen_xyz)).query(p_xyz[order])
                else:
                    dist = np.full(order.size, np.inf)
                free = ~np.isin(p_flat[order], np.fromiter(taken, np.int64, len(taken)))
                ok = free & (dist >= self.chord)
                if ok.any():
                    j = order[np.flatnonzero(ok)[0]]
                elif have == 0 and free.any():
                    j = order[np.flatnonzero(free)[np.argmax(dist[free])]]
                    relaxed.append(zone)
                else:
                    short[zone] = have
                    break
                taken.add(int(p_flat[j]))
                chosen_xyz.append(p_xyz[j])
                out_lat.append(p_lat[j])
                out_lon.append(p_lon[j])
                out_code.append(zone)
                source.append("zone")
                have += 1
        return Selection(np.asarray(out_lat, np.float64), np.asarray(out_lon, np.float64),
                         np.asarray(out_code, np.int64), np.asarray(source), int(n_lattice),
                         list(self.zones), relaxed, short)


def select_points(grid, n_points=DEFAULT_N_POINTS, min_per_zone=DEFAULT_MIN_PER_ZONE,
                  min_dist_km=DEFAULT_MIN_DIST_KM, include_antarctica=False, seed=DEFAULT_SEED,
                  max_evals=400):
    """Выбор обучающих точек на суше.

    Размер решётки подбирается так, чтобы вместе с точками, добранными ради
    покрытия зон, получилось ровно n_points точек. Если точного совпадения
    нет, берётся ближайший размер и пишется предупреждение. Сид задаёт поворот
    решётки вокруг оси Земли, поэтому один и тот же сид даёт один и тот же выбор.

    Args:
        grid: сетка зон Кёппена.
        n_points: сколько точек нужно всего.
        min_per_zone: минимум точек в каждой зоне, где есть суша.
        min_dist_km: минимальное расстояние между точками, км.
        include_antarctica: брать ли сушу южнее 60 градусов южной широты.
        seed: сид поворота решётки.
        max_evals: предел числа проверенных размеров решётки.

    Returns:
        Выбор точек с отчётом о покрытии зон.

    Raises:
        ValueError: на сетке нет суши в разрешённых широтах.
    """
    lon_offset = float(np.random.default_rng(seed).uniform(0.0, 360.0))
    sel = _Selector(grid, min_per_zone, min_dist_km, include_antarctica, lon_offset)
    if not sel.zones:
        raise ValueError("на карте зон нет суши в разрешённых широтах")
    target = int(n_points)
    n = max(target, int(round(target / max(sel.land_frac, 1e-6))))
    tried = {}
    best = None
    for _ in range(int(max_evals)):
        if n in tried:
            break
        res = sel.run(n)
        tried[n] = len(res)
        if best is None or (abs(len(res) - target), len(res) < target) < \
                (abs(len(best) - target), len(best) < target):
            best = res
        if len(res) == target:
            break
        step = int(round((target - len(res)) / max(sel.land_frac, 1e-6)))
        nxt = n + (step if step != 0 else (1 if len(res) < target else -1))
        if nxt in tried or nxt < 1:
            nxt = _nearest_untried(n, tried)
        n = nxt
    if len(best) != target:
        log.warning("точно %d точек не получилось: выбрано %d (решётка %d узлов)",
                    target, len(best), best.n_lattice)
    for zone in best.relaxed:
        log.warning("зона %s: единственная точка ближе %.0f км к соседней, иначе зона "
                    "осталась бы без точек", KG_TIF_CODE[zone], min_dist_km)
    for zone, have in best.short.items():
        log.warning("зона %s: %d точек из %d, больше не помещается при расстоянии %.0f км",
                    KG_TIF_CODE[zone], have, min_per_zone, min_dist_km)
    return best


def _nearest_untried(n, tried):
    """Ближайший к n размер решётки, который ещё не проверяли."""
    for d in range(1, 10 * n + 1):
        for cand in (n + d, n - d):
            if cand >= 1 and cand not in tried:
                return cand
    return n + 1


def min_pairwise_km(lat, lon):
    """Наименьшее расстояние между двумя точками набора.

    Args:
        lat: широты, градусы.
        lon: долготы, градусы.

    Returns:
        Расстояние в км; бесконечность, если точек меньше двух.
    """
    if np.size(lat) < 2:
        return math.inf
    d, _ = cKDTree(unit_vectors(lat, lon)).query(unit_vectors(lat, lon), k=2)
    return float(km_of_chord(d[:, 1].min()))


__all__ = ["ANTARCTICA_LAT", "DEFAULT_GRID_STEP", "DEFAULT_MIN_DIST_KM", "DEFAULT_MIN_PER_ZONE",
           "DEFAULT_N_POINTS", "DEFAULT_SEED", "POINTS_FIELDS", "Selection", "ZoneGrid",
           "chord_of_km", "fibonacci_lattice", "km_of_chord", "min_pairwise_km",
           "select_points", "unit_vectors"]
