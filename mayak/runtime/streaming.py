"""Потоковый рантайм МАЯК для устройства.

Прогноз потока равен пакетному прогнозу по последним часам истории модели при любом
моменте выпуска. Пакетное окно истории делится на две части.

* Хвост - поздние часы окна. Признаки энкодера для них не зависят от того, где окно
  начинается, и совпадают у пакета и потока. Поток держит их вклад в моды скользящей
  суммой: на каждом часе вклад нового часа добавляется, вклад часа, вышедшего из
  хвоста, вычитается. Раз в сутки сумма пересчитывается точно по кольцу вкладов, чтобы
  ошибка округления от вычитания не копилась.
* Край - ранние часы окна. У пакета энкодер видит перед ними нули, а каналы с лагом не
  знают часов до окна, поэтому их признаки зависят от момента выпуска. Вклад края
  пересчитывается при каждом выпуске пакетным проходом по часам края.

Паспорт тоже считается при выпуске: по кольцу строк суточного накопителя за всю
историю, сутки заканчиваются в момент выпуска.

Шаг часа стоит один потактовый шаг энкодера и сумма по модам; выпуск - проход энкодера
по краю окна, суточные сводки и паспорт.

Персистентное состояние - только сырое окно и заголовок. В заголовке абсолютный час
последнего шага, число часов окна после холодного старта, множители адаптивной
калибровки по бинам лидов и координаты точки, для которой состояние записано. Состояние
прежней версии с одним множителем читается: множитель переносится во все бины. В окне для каждого
часа записанные прибором значения до отбраковки, маска наличия и маска годности после
причинного контроля качества. Температура и влажность занимают по байту со знаком,
давление - два байта в десятых гектопаскаля, маски упакованы по битам. Моды, кольца,
буфер энкодера и кольцо контроля качества восстанавливаются из окна одним пакетным
проходом при загрузке.

Холодный старт - это окно из пустых часов, как у пакета при короткой истории. Простой
заполняется пустыми часами, простой не короче окна опустошает окно. Множители
калибровки при этом сохраняются: они относятся к прибору, а не к истории.

Калибровка интервалов: квантили модели, затем конформная таблица, затем адаптивный
множитель своего бина лидов. Строка конформной таблицы выбирается по длине истории
выпуска: числу часов после холодного старта, но не больше истории модели. Множители
подстраиваются онлайн, если рантайм создан с параметрами адаптивной калибровки:
каждый валидный час температуры сверяется с кольцом по часам-мишеням, где для каждого
бина лидов лежит последний выпуск, чей лид до этого часа попадает в бин. Кольцо живёт
только в памяти.

Смена точки. Состояние помнит координаты и высоту, для которых оно записано. При загрузке
они сравниваются с текущими. Сдвиг в пределах порогов рантайма - уточнение метаданных:
окно сохраняется, всё модельное состояние восстанавливается из него уже для новой точки,
множитель калибровки сохраняется, в лог пишется предупреждение. Сдвиг больше порога -
прибор перенесён: окно опустошается, множитель калибровки обнуляется, абсолютный час
последнего шага сохраняется. Контроль качества новых часов всегда берёт текущую высоту.

Момент выпуска - последний шаг. До первого шага, в том числе после старта без
состояния, момент выпуска задаёт вызывающий по часам устройства, и прогноз строится по
пустому окну, которое этим часом заканчивается. Само состояние при этом не меняется.

Откат. При старте граф точки возвращает таблицу климатологии: среднее и масштаб
климат-поля с паспортом холодного старта на каждый час года. Если выпуск не удался или
дал нечисловые квантили, прогноз - квантили нормального распределения с этими средним
и масштабом на часах горизонта. Таблица своя у каждой точки, поэтому откат - это
климатология той точки, где стоит прибор.
"""
import logging
from typing import NamedTuple

import numpy as np

from mayak.config import CHANNEL_MAX_LAG
from mayak.data.qc import PHYS, CausalQC, qc_window
from mayak.data.recording import RECORD_SCALE, record_values
from mayak.leakage import load_conformal
from mayak.metrics import (LEAD_BINS, ACIParams, AdaptiveCalibration, apply_adaptive,
                           apply_conformal, check_conformal_shape)
from mayak.runtime.site import (SITE_MOVED, SITE_REFINED, as_site, describe_gap,
                                load_runtime_config, site_change)
from mayak.timeaxis import hour_of_year, window_calendar

log = logging.getLogger(__name__)

STATE_MAGIC = b"MYK"
STATE_VERSION = 5
N_LEAD_BINS = len(LEAD_BINS)
STATE_HEADER = np.dtype([("magic", "S3"), ("version", "u1"), ("filled", "<u2"),
                         ("reserved", "<u2"), ("last_hour", "<i8"),
                         ("aci_theta", "<f4", (N_LEAD_BINS,)),
                         ("lat", "<f4"), ("lon", "<f4"), ("elev", "<f4")])
STATE_HEADER_V4 = np.dtype([("magic", "S3"), ("version", "u1"), ("filled", "<u2"),
                            ("reserved", "<u2"), ("last_hour", "<i8"), ("aci_theta", "<f4"),
                            ("lat", "<f4"), ("lon", "<f4"), ("elev", "<f4")])
STATE_HEADERS = {4: STATE_HEADER_V4, STATE_VERSION: STATE_HEADER}
NO_HOUR = int(np.iinfo(np.int64).min)
RESYNC_HOURS = 24
CTX = CHANNEL_MAX_LAG + 1

RAW_CHANNELS = ("T", "P", "RH")
STORE_DTYPES = (np.dtype("i1"), np.dtype("<u2"), np.dtype("i1"))
STORE_SCALE = np.asarray(RECORD_SCALE, np.float64)
STORE_LO = np.array([-128.0, 0.0, -128.0])
STORE_HI = np.array([127.0, 65535.0, 127.0])
PHYS_LO = np.array([PHYS[c][0] for c in RAW_CHANNELS], np.float32)
PHYS_HI = np.array([PHYS[c][1] for c in RAW_CHANNELS], np.float32)
HOURS_OF_YEAR = 366 * 24


class StateSnapshot(NamedTuple):
    """Разобранное состояние с диска.

    Attributes:
        filled: сколько часов окна прошло после холодного старта.
        last_hour: абсолютный час последнего шага или None, если шагов не было.
        theta: множители адаптивной калибровки в логарифме по бинам лидов.
        site: широта, долгота и высота, для которых записано окно.
        raw: значения на сетке хранения, форма (W, 3), от старых часов к новым.
        present: маска наличия, форма (W, 3).
        valid: маска годности, форма (W, 3).
    """
    filled: int
    last_hour: int | None
    theta: tuple
    site: tuple
    raw: np.ndarray
    present: np.ndarray
    valid: np.ndarray


def parse_state(raw, window):
    """Разбор байт состояния с проверками целостности.

    Args:
        raw: байты состояния.
        window: длина окна модели, часы.

    Returns:
        Разобранное состояние.

    Состояние версии 4 хранит один множитель калибровки: он переносится во все бины
    лидов.

    Raises:
        ValueError: байты не состояние этого формата, не подходят длине окна или
            повреждены.
    """
    raw = bytes(raw)
    if len(raw) < len(STATE_MAGIC) + 1 or raw[:len(STATE_MAGIC)] != STATE_MAGIC:
        raise ValueError("не состояние МАЯК: нет заголовка")
    version = raw[len(STATE_MAGIC)]
    if version not in STATE_HEADERS:
        raise ValueError(f"версия состояния {version}, рантайм читает "
                         f"{sorted(STATE_HEADERS)}; прежние версии не хранят сырое окно "
                         f"целиком, нужен холодный старт")
    header = STATE_HEADERS[version]
    want = window_nbytes(window, version)
    if len(raw) != want:
        raise ValueError(f"состояние {len(raw)} Б не соответствует конфигу модели "
                         f"(ожидалось {want} Б)")
    hdr = np.frombuffer(raw, header, count=1)[0]
    theta = tuple(float(v) for v in np.broadcast_to(hdr["aci_theta"], (N_LEAD_BINS,)))
    filled, last = int(hdr["filled"]), int(hdr["last_hour"])
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"повреждённый множитель калибровки в состоянии: {list(theta)}")
    if filled > window or (last == NO_HOUR and filled):
        raise ValueError(f"повреждённый заголовок состояния: filled={filled}, окно {window}, "
                         f"последний час {last}")
    x, present, valid = decode_window(raw[header.itemsize:], window)
    if np.any(valid > present):
        raise ValueError("повреждённое окно: годный час без значения")
    ok = valid > 0
    if np.any(ok & ((x < PHYS_LO) | (x > PHYS_HI))):
        raise ValueError("повреждённое окно: годное значение вне физического диапазона")
    site = (float(hdr["lat"]), float(hdr["lon"]), float(hdr["elev"]))
    return StateSnapshot(filled, None if last == NO_HOUR else last, theta, site, x, present,
                         valid)


def check_climatology(clim_mu, clim_sig):
    """Проверка таблицы климатологии точки, которую вернул граф старта.

    Без годной таблицы откат невозможен, поэтому негодная таблица - ошибка
    конфигурации, а не повод работать дальше.

    Args:
        clim_mu: среднее климат-поля на каждый час года.
        clim_sig: масштаб климат-поля на каждый час года.

    Returns:
        Пара массивов float32 формы (8784,).

    Raises:
        RuntimeError: таблица не той длины, с нечисловыми значениями или с
            неположительным масштабом.
    """
    mu = np.asarray(clim_mu, np.float32).ravel()
    sig = np.asarray(clim_sig, np.float32).ravel()
    if mu.size != HOURS_OF_YEAR or sig.size != HOURS_OF_YEAR:
        raise RuntimeError(f"граф старта: таблица климатологии на {mu.size} и {sig.size} ч, "
                           f"нужно {HOURS_OF_YEAR}; графы экспортированы другой версией")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sig)) and np.all(sig > 0)):
        raise RuntimeError("граф старта: таблица климатологии точки негодна (нечисловые "
                           "значения или неположительный масштаб); это ошибка конфигурации "
                           "модели или координат")
    return mu, sig


def to_store(x):
    """Значения на сетке хранения окна.

    Сетка та же, что у записи прибора; значения за пределами типов хранения
    обрезаются. Значение вне физического диапазона после обрезки остаётся вне него,
    поэтому решения контроля качества по сохранённому окну не меняются.

    Args:
        x: записанные значения, форма (..., 3).

    Returns:
        Массив float32 той же формы.
    """
    q = np.clip(np.rint(np.asarray(x, np.float64) * STORE_SCALE), STORE_LO, STORE_HI)
    return (q / STORE_SCALE).astype(np.float32)


def mask_bytes(n_hours):
    """Размер одной битовой маски окна в байтах.

    Args:
        n_hours: длина окна, часы.

    Returns:
        Число байт.
    """
    return (3 * n_hours + 7) // 8


def window_nbytes(window, version=STATE_VERSION):
    """Размер сериализованного состояния для длины окна.

    Args:
        window: длина окна, часы.
        version: версия формата состояния.

    Returns:
        Число байт.
    """
    W = int(window)
    return (STATE_HEADERS[version].itemsize + W * sum(d.itemsize for d in STORE_DTYPES)
            + 2 * mask_bytes(W))


def state_nbytes(cfg):
    """Размер сериализованного состояния для конфига модели.

    Args:
        cfg: конфиг модели.

    Returns:
        Число байт.
    """
    return window_nbytes(cfg.stream_window)


def encode_window(raw, present, valid):
    """Сырое окно в байты состояния.

    Args:
        raw: значения на сетке хранения, форма (W, 3), от старых часов к новым.
        present: маска наличия, форма (W, 3).
        valid: маска годности, форма (W, 3).

    Returns:
        Байты окна.
    """
    p = np.asarray(present) > 0
    q = np.where(p, np.rint(np.asarray(raw, np.float64) * STORE_SCALE), 0.0)
    parts = [q[:, c].astype(STORE_DTYPES[c]).tobytes() for c in range(3)]
    for mk in (p, np.asarray(valid) > 0):
        parts.append(np.packbits(mk.ravel(), bitorder="little").tobytes())
    return b"".join(parts)


def decode_window(buf, n_hours):
    """Байты окна в значения и маски.

    Args:
        buf: байты окна.
        n_hours: длина окна, часы.

    Returns:
        Тройка: значения float32 (W, 3) и маски наличия и годности uint8 (W, 3).
    """
    off, cols = 0, []
    for c in range(3):
        d = STORE_DTYPES[c]
        q = np.frombuffer(buf, d, count=n_hours, offset=off).astype(np.float64)
        cols.append((q / STORE_SCALE[c]).astype(np.float32))
        off += n_hours * d.itemsize
    masks = []
    nb = mask_bytes(n_hours)
    for _ in range(2):
        bits = np.unpackbits(np.frombuffer(buf, np.uint8, count=nb, offset=off),
                             bitorder="little")[:3 * n_hours]
        masks.append(bits.reshape(n_hours, 3).astype(np.uint8))
        off += nb
    return np.stack(cols, axis=-1), masks[0], masks[1]


class StreamingMayak:
    """Потоковый рантайм на PyTorch.

    Args:
        model: модель в режиме eval.
        lat: широта точки.
        lon: долгота точки.
        elev: высота точки, м.
        conformal: таблица поправок по бинам лидов и длины истории, путь к ней с записью
            о подгонке рядом или None.
        aci: параметры адаптивной калибровки, True для параметров по умолчанию или None,
            если множители калибровки не подстраиваются.
        runtime_cfg: параметры хоста с порогами смены точки или None - конфиг рантайма
            по умолчанию.

    Attributes:
        last_hour: абсолютный час UTC последнего шага или None до первого шага.
        filled: сколько часов окна прошло после холодного старта, не больше длины окна.
        cal: адаптивная калибровка по бинам лидов: множители, счётчики и кольцо.
        idle_hours: сколько пустых часов подставлено за простой в этом процессе.
        loaded_site: координаты и высота из загруженного состояния или None.
        site_change: исход сравнения точки загруженного состояния с текущей: та же
            точка, уточнение или перенос; None, если состояние не загружалось.
        clim_mu: среднее климат-поля точки на каждый час года, для отката.
        clim_sig: масштаб климат-поля точки на каждый час года, для отката.
        fallbacks: сколько выпусков в этом процессе заменено откатом.
    """

    def __init__(self, model, lat, lon, elev, conformal=None, aci=None, runtime_cfg=None):
        from mayak.runtime.graphs import TorchBackend
        self.m = model.eval()
        self._setup(TorchBackend(model), model.cfg, lat, lon, elev, conformal, aci, runtime_cfg)

    def _setup(self, backend, cfg, lat, lon, elev, conformal, aci, runtime_cfg=None):
        self.b, self.cfg = backend, cfg
        self.window, self.history = cfg.stream_window, cfg.max_history
        self.edge, self.tail = cfg.stream_edge, cfg.stream_tail
        self.horizon, self.n_modes = cfg.horizon, cfg.n_modes
        self.lat, self.lon, self.elev = float(lat), float(lon), float(elev)
        f = lambda v: np.array([[v]], np.float32)
        self._lat, self._lon = f(lat), f(lon)
        self.loc, c_mu, c_sig, c_def, self.z0, clim_mu, clim_sig = backend.run(
            "init", self._lat, self._lon, f(elev))
        self.coefs = [c_mu, c_sig, c_def]
        self.clim_mu, self.clim_sig = check_climatology(clim_mu, clim_sig)
        from mayak.baselines.statistical import ZQ
        self.zq = np.asarray(ZQ, np.float32)
        self.conformal = self._conformal(conformal)
        self.aci = ACIParams() if aci is True else aci
        self.cal = AdaptiveCalibration(self.aci, self.horizon, LEAD_BINS)
        self.qc = CausalQC(elev=self.elev)
        self.runtime_cfg = load_runtime_config() if runtime_cfg is None else runtime_cfg
        self.loaded_site = None
        self.site_change = None
        self.idle_hours = 0
        self.fallbacks = 0
        self.reset_calibration()
        self.reset()

    def reset_calibration(self, theta=0.0):
        """Сброс адаптивной калибровки прибора: множители, счётчики и кольцо.

        Args:
            theta: новый логарифм множителя: одно число на все бины лидов или по числу
                на бин.
        """
        self.cal.reset(theta)

    @property
    def theta(self):
        """Логарифмы множителей адаптивной калибровки по бинам лидов."""
        return tuple(self.cal.theta)

    @property
    def aci_updates(self):
        """Число обратных связей по бинам лидов с последнего сброса калибровки."""
        return tuple(self.cal.updates)

    @property
    def aci_misses(self):
        """Число промахов по бинам лидов с последнего сброса калибровки."""
        return tuple(self.cal.misses)

    @property
    def history_length(self):
        """Длина истории выпуска: часы после холодного старта, не больше истории модели."""
        return min(int(self.filled), int(self.history))

    def reset(self, last_hour=None):
        """Холодный старт: окно состоит из пустых часов. Множители калибровки сохраняются.

        Args:
            last_hour: час, которым заканчивается пустое окно. None - момент ещё не
                известен, окно строится при первом шаге.
        """
        W, M = self.window, self.n_modes
        self.cal.clear()
        self.qc.reset()
        self.raw = np.zeros((W, 3), np.float32)
        self.present = np.zeros((W, 3), np.uint8)
        self.valid = np.zeros((W, 3), np.uint8)
        self.filled = 0
        self.last_hour = None
        self.modes = [np.zeros((1, M), np.float32) for _ in range(3)]
        self.u_ring = np.zeros((self.tail, 2 * M), np.float32)
        self.v_ring = np.zeros(self.tail, np.float32)
        self.rows = np.zeros((self.history, 4), np.float32)
        pads = (self.cfg.encoder_kernel - 1) * sum(self.cfg.encoder_dilations)
        self.enc_buf = np.zeros((1, self.cfg.encoder_width, pads), np.float32)
        if last_hour is not None:
            self.last_hour = int(last_hour)
            self._rebuild(seed_qc=False)

    @property
    def site(self):
        """Текущие широта, долгота и высота так, как они пишутся в состояние."""
        return as_site((self.lat, self.lon, self.elev))

    @property
    def memory_nbytes(self):
        """Размер колец, буфера энкодера и таблицы климатологии в памяти, байт.

        Кольцо адаптивной калибровки считается, если она включена. На диск ничего из
        этого не пишется: всё восстанавливается из окна и графа старта, кольцо
        калибровки после перезапуска пусто.
        """
        return int(self.u_ring.nbytes + self.v_ring.nbytes + self.rows.nbytes
                   + self.enc_buf.nbytes + self.clim_mu.nbytes + self.clim_sig.nbytes
                   + self.cal.nbytes)

    def _hours(self, n):
        """Абсолютные часы последних n часов окна, от старых к новым."""
        return self.last_hour - n + 1 + np.arange(n, dtype=np.int64)

    def _window_arrays(self, hours):
        """Значения, маски годности и календарь часов окна."""
        pos = hours % self.window
        m = self.valid[pos].astype(np.float32)
        x = np.where(m > 0, self.raw[pos], 0.0).astype(np.float32)
        doy, hour = window_calendar(0, hours)
        return x, m, doy, hour

    def _window_pass(self, x, m, hours):
        """Пакетный проход графом полного окна по часам окна."""
        doy, hour = window_calendar(0, hours)
        return self.b.run("window", x[None], m[None], doy[None], hour[None], self._lat,
                          self._lon, *self.coefs)

    def _rebuild(self, seed_qc):
        """Всё модельное состояние из сырого окна одним пакетным проходом."""
        W = self.window
        hours = self._hours(W)
        x, m, _, _ = self._window_arrays(hours)
        enc, u, v, rows, n_re, n_im, e = self._window_pass(x, m, hours)
        self.enc_buf, self.modes = enc, [n_re, n_im, e]
        if self.tail:
            slot = hours[W - self.tail:] % self.tail
            self.u_ring[slot], self.v_ring[slot] = u[0], v[0]
        self.rows[hours[W - self.history:] % self.history] = rows[0]
        if seed_qc:
            pos = hours[-self.qc.size:] % W
            self.qc.seed(self.raw[pos], self.present[pos])

    def step(self, T, P, RH, hour):
        """Новый час наблюдений.

        Пропущенные часы между прошлым шагом и этим заполняются пустыми. Простой не
        короче окна - холодный старт.

        Args:
            T: температура; None или NaN - значения нет.
            P: давление; None или NaN - значения нет.
            RH: влажность; None или NaN - значения нет.
            hour: абсолютный час UTC, целое число часов от эпохи.

        Returns:
            Коды причинного контроля качества этого часа, uint8 формы (3,).

        Raises:
            ValueError: час не позже последнего шага.
        """
        hour = int(hour)
        if self.last_hour is None:
            self.reset(hour - 1)
        elif hour <= self.last_hour:
            raise ValueError(f"час {hour} не позже последнего шага {self.last_hour}")
        else:
            gap = hour - self.last_hour - 1
            if gap >= self.window:
                log.info("простой %d ч не короче окна %d ч: холодный старт", gap, self.window)
                self.reset(hour - 1)
            for h in range(self.last_hour + 1, hour):
                self._push((None, None, None), h)
            self.idle_hours += gap
        return self._push((T, P, RH), hour)

    def _push(self, values, hour):
        xj, codes = self.qc.push(values)
        raw, present = self.qc.latest()
        valid = (codes == 0).astype(np.uint8)
        if valid[0]:
            self.cal.feedback(float(xj[0]), hour)
        j = hour % self.window
        self.raw[j] = np.where(present > 0, to_store(raw), 0.0)
        self.present[j], self.valid[j] = present, valid
        self._ingest(hour)
        return codes

    def _ingest(self, hour):
        """Один час окна через граф шага: буфер энкодера, сумма мод, кольца."""
        self.last_hour = hour
        self.filled = min(self.window, self.filled + 1)
        x, m, _, _ = self._window_arrays(self._hours(CTX))
        doy, hr = window_calendar(0, np.array([hour]))
        M = self.n_modes
        if self.tail:
            slot = hour % self.tail
            u_old, v_old = self.u_ring[slot][None], np.array([[self.v_ring[slot]]], np.float32)
        else:
            u_old, v_old = np.zeros((1, 2 * M), np.float32), np.zeros((1, 1), np.float32)
        enc, n_re, n_im, e, u, v, row = self.b.run(
            "step", x[None], m[None], doy[None], hr[None], self._lat, self._lon, *self.coefs,
            self.enc_buf, *self.modes, u_old, v_old)
        self.enc_buf, self.modes = enc, [n_re, n_im, e]
        if self.tail:
            self.u_ring[slot], self.v_ring[slot] = u[0], v[0, 0]
        self.rows[hour % self.history] = row[0]
        if self.tail and (hour + 1) % RESYNC_HOURS == 0:
            self.resync()

    def resync(self):
        """Точная сумма мод по кольцу вкладов хвоста вместо скользящей."""
        slot = self._hours(self.tail) % self.tail
        self.modes = list(self.b.run("resync", self.u_ring[slot][None], self.v_ring[slot][None]))

    @property
    def aci_coverage(self):
        """Фактическое покрытие по обратной связи в каждом бине лидов с последнего сброса."""
        return self.cal.coverage()

    def issue_hour(self, now_hour=None):
        """Момент выпуска: последний шаг, а до первого шага - текущий час устройства.

        Args:
            now_hour: текущий абсолютный час UTC по часам устройства или None.

        Returns:
            Абсолютный час, после которого начинается горизонт.

        Raises:
            ValueError: шагов не было и текущий час не задан.
        """
        if self.last_hour is not None:
            return self.last_hour
        if now_hour is None:
            raise ValueError("нет ни одного шага и не задан текущий час: момент выпуска не "
                             "определён")
        return int(now_hour)

    def _empty_history(self, last):
        """Строки накопителя, сумма мод и край истории пустого окна, которое кончается часом last.

        Состояние рантайма не меняется.
        """
        W, L, E = self.window, self.history, self.edge
        hours = last - W + 1 + np.arange(W, dtype=np.int64)
        zeros = np.zeros((W, 3), np.float32)
        _enc, _u, _v, rows, n_re, n_im, e = self._window_pass(zeros, zeros, hours)
        de, he = window_calendar(0, hours[W - L:W - L + E])
        return rows, [n_re, n_im, e], (zeros[:E], zeros[:E], de, he)

    def raw_forecast(self, now_hour=None):
        """Квантили модели до калибровки на часы после момента выпуска.

        Args:
            now_hour: текущий абсолютный час UTC по часам устройства; нужен только до
                первого шага, тогда прогноз строится по пустому окну.

        Returns:
            Массив float32 формы (H, число квантилей).

        Raises:
            ValueError: момент выпуска не определён.
            FloatingPointError: выход графа выпуска не конечен.
        """
        last = self.issue_hour(now_hour)
        L, E = self.history, self.edge
        if self.last_hour is None:
            rows, modes, (xe, me, de, he) = self._empty_history(last)
        else:
            hours = self._hours(L)
            rows, modes = self.rows[hours % L][None], self.modes
            xe, me, de, he = self._window_arrays(hours[:E])
        df, hf = window_calendar(0, last + 1 + np.arange(self.horizon))
        (q,) = self.b.run("issue", self.loc, self._lat, self._lon, *self.coefs, rows,
                          *modes, xe[None], me[None], de[None], he[None], df[None], hf[None])
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("выход графа выпуска не конечен")
        return q[0]

    def forecast(self, now_hour=None):
        """Выпуск на часы после момента выпуска.

        Args:
            now_hour: текущий абсолютный час UTC по часам устройства; нужен только до
                первого шага.

        Returns:
            Пара: квантили float32 формы (H, число квантилей) и медиана формы (H,).
        """
        q = self.raw_forecast(now_hour)
        if self.conformal is not None:
            q = apply_conformal(q, self.conformal, self.history_length)
        self.cal.record(self.issue_hour(now_hour), q)
        return apply_adaptive(q, self.theta)

    def climatology_forecast(self, last):
        """Откат: квантили нормального распределения по таблице климатологии точки.

        Args:
            last: абсолютный час, после которого начинается горизонт.

        Returns:
            Пара: квантили float32 формы (H, число квантилей) и медиана формы (H,).
        """
        idx = hour_of_year(int(last) + 1 + np.arange(self.horizon, dtype=np.int64))
        mu, sig = self.clim_mu[idx], self.clim_sig[idx]
        q = mu[:, None] + self.zq[None, :] * sig[:, None]
        return q.astype(np.float32), mu.copy()

    def safe_forecast(self, now_hour=None):
        """Выпуск, а при любой ошибке выпуска - откат к климатологии точки.

        Args:
            now_hour: текущий абсолютный час UTC по часам устройства; нужен только до
                первого шага.

        Returns:
            Тройка: квантили формы (H, число квантилей), медиана формы (H,) и признак
            отката.

        Raises:
            ValueError: момент выпуска не определён, откатываться не к чему.
        """
        try:
            q, mu = self.forecast(now_hour)
            return q, mu, False
        except Exception as e:
            last = self.issue_hour(now_hour)
            log.warning("откат к климатологии: %s", e)
            self.fallbacks += 1
            q, mu = self.climatology_forecast(last)
            return q, mu, True

    def warm_start(self, x_hist, mask_hist, last_hour):
        """Прогрев по сырой истории: то же состояние, что после шага по каждому часу.

        Значения записываются так, как их пишет прибор, и проходят причинный контроль
        качества по всей переданной истории.

        Args:
            x_hist: сырые значения, форма (L, 3).
            mask_hist: маска наличия, форма (L, 3).
            last_hour: абсолютный час UTC последнего часа истории.
        """
        self.reset()
        x = np.asarray(x_hist, np.float32).reshape(-1, 3)
        present = ((np.asarray(mask_hist) > 0) & np.isfinite(x)).astype(np.uint8)
        n = x.shape[0]
        if n == 0:
            return
        raw = np.where(present > 0, record_values(np.where(present > 0, x, 0.0)), 0.0)
        valid, _ = qc_window(raw, present, elev=self.elev)
        take = min(self.window, n)
        self.last_hour = int(last_hour)
        pos = self._hours(take) % self.window
        self.raw[pos] = np.where(present > 0, to_store(raw), 0.0)[-take:]
        self.present[pos] = present[-take:]
        self.valid[pos] = (valid[-take:] > 0).astype(np.uint8)
        self.filled = take
        self._rebuild(seed_qc=True)

    @staticmethod
    def _conformal(conformal):
        """Таблица поправок для выходов модели.

        Args:
            conformal: None, путь к таблице с записью о подгонке рядом или сама таблица.

        Returns:
            Таблица float32 по бинам лидов и длины истории или None.

        Raises:
            ValueError: таблица старого формата, другой формы или сдвигает медиану.
        """
        if conformal is None:
            return None
        if not isinstance(conformal, str):
            return check_conformal_shape(conformal)
        shift, _rec = load_conformal(conformal)
        return shift

    @property
    def state_nbytes(self):
        """Размер сериализованного состояния для конфига этой модели, байт."""
        return state_nbytes(self.cfg)

    def serialize(self):
        """Состояние в байтах: заголовок и сырое окно от старых часов к новым."""
        W = self.window
        hdr = np.zeros((), STATE_HEADER)
        hdr["magic"], hdr["version"] = STATE_MAGIC, STATE_VERSION
        hdr["filled"] = self.filled
        hdr["last_hour"] = NO_HOUR if self.last_hour is None else self.last_hour
        hdr["aci_theta"] = np.asarray(self.theta, np.float32)
        hdr["lat"], hdr["lon"], hdr["elev"] = self.lat, self.lon, self.elev
        pos = np.arange(W) if self.last_hour is None else self._hours(W) % W
        return hdr.tobytes() + encode_window(self.raw[pos], self.present[pos], self.valid[pos])

    def load_state(self, raw):
        """Состояние с диска и восстановление всего остального одним проходом по окну.

        Args:
            raw: байты состояния.

        Если состояние записано для другой точки, сдвиг в пределах порогов рантайма -
        уточнение: окно сохраняется и пересчитывается для новой точки. Больше порога -
        перенос: окно пустое, множители калибровки нулевые, момент последнего шага
        сохраняется. Состояние версии 4 читается, его множитель идёт во все бины лидов.

        Raises:
            ValueError: байты не состояние этого формата, не подходят конфигу модели или
                повреждены. Рантайм тогда остаётся в холодном старте.
        """
        snap = parse_state(raw, self.window)
        kind, gap = site_change(snap.site, self.site, self.runtime_cfg)
        self.reset()
        self.loaded_site, self.site_change = snap.site, kind
        if kind == SITE_MOVED:
            log.warning("прибор перенесён: %s; холодный старт, множитель калибровки сброшен",
                        describe_gap(gap, self.runtime_cfg))
            self.reset_calibration(0.0)
            try:
                self.reset(snap.last_hour)
            except Exception:
                self.reset()
                raise
            return
        if kind == SITE_REFINED:
            log.warning("координаты уточнены: %s; окно пересчитано для новой точки, множитель "
                        "калибровки сохранён", describe_gap(gap, self.runtime_cfg))
        self.reset_calibration(snap.theta)
        if snap.last_hour is None:
            return
        W = self.window
        self.last_hour = snap.last_hour
        pos = self._hours(W) % W
        self.raw[pos], self.present[pos], self.valid[pos] = snap.raw, snap.present, snap.valid
        self.filled = snap.filled
        try:
            self._rebuild(seed_qc=True)
        except Exception:
            self.reset()
            raise


__all__ = ["CTX", "HOURS_OF_YEAR", "NO_HOUR", "N_LEAD_BINS", "RAW_CHANNELS", "RESYNC_HOURS",
           "STATE_HEADER", "STATE_HEADERS", "STATE_HEADER_V4", "STATE_VERSION", "StateSnapshot",
           "StreamingMayak", "check_climatology", "decode_window", "encode_window", "mask_bytes",
           "parse_state", "state_nbytes", "to_store", "window_nbytes"]
