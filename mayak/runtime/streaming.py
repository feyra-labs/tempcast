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
последнего шага, число часов окна после холодного старта, множитель адаптивной
калибровки и координаты точки, для которой состояние записано. В окне для каждого
часа записанные прибором значения до отбраковки, маска наличия и маска годности после
причинного контроля качества. Температура и влажность занимают по байту со знаком,
давление - два байта в десятых гектопаскаля, маски упакованы по битам. Моды, кольца,
буфер энкодера и кольцо контроля качества восстанавливаются из окна одним пакетным
проходом при загрузке.

Холодный старт - это окно из пустых часов, как у пакета при короткой истории. Простой
заполняется пустыми часами, простой не короче окна опустошает окно. Множитель
калибровки при этом сохраняется: он относится к прибору, а не к истории.

Калибровка интервалов: квантили модели, затем конформная таблица, затем адаптивный
множитель. Множитель подстраивается онлайн, если рантайм создан с параметрами
адаптивной калибровки: каждый валидный час температуры сверяется с последним
выпущенным прогнозом на том лиде, который приходится на этот час.
"""
import logging

import numpy as np

from mayak.config import CHANNEL_MAX_LAG
from mayak.data.qc import PHYS, CausalQC, qc_window
from mayak.data.recording import RECORD_SCALE, record_values
from mayak.leakage import load_conformal, precision_mismatch
from mayak.metrics import (ACIParams, aci_score, apply_adaptive, apply_conformal,
                           check_median_free)
from mayak.timeaxis import window_calendar

log = logging.getLogger(__name__)

STATE_MAGIC = b"MYK"
STATE_VERSION = 4
STATE_HEADER = np.dtype([("magic", "S3"), ("version", "u1"), ("filled", "<u2"),
                         ("reserved", "<u2"), ("last_hour", "<i8"), ("aci_theta", "<f4"),
                         ("lat", "<f4"), ("lon", "<f4"), ("elev", "<f4")])
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


def state_nbytes(cfg):
    """Размер сериализованного состояния для конфига модели.

    Args:
        cfg: конфиг модели.

    Returns:
        Число байт.
    """
    W = cfg.stream_window
    return STATE_HEADER.itemsize + W * sum(d.itemsize for d in STORE_DTYPES) + 2 * mask_bytes(W)


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
        conformal: таблица поправок, путь к ней с записью о подгонке рядом или None.
        aci: параметры адаптивной калибровки, True для параметров по умолчанию или None,
            если множитель калибровки не подстраивается.

    Attributes:
        last_hour: абсолютный час UTC последнего шага или None до первого шага.
        filled: сколько часов окна прошло после холодного старта, не больше длины окна.
        theta: множитель адаптивной калибровки в логарифме.
        idle_hours: сколько пустых часов подставлено за простой в этом процессе.
        loaded_site: координаты и высота из загруженного состояния или None.
    """

    def __init__(self, model, lat, lon, elev, conformal=None, aci=None):
        from mayak.runtime.graphs import TorchBackend
        self.m = model.eval()
        self._setup(TorchBackend(model), model.cfg, lat, lon, elev, conformal, aci)

    def _setup(self, backend, cfg, lat, lon, elev, conformal, aci):
        self.b, self.cfg = backend, cfg
        self.window, self.history = cfg.stream_window, cfg.max_history
        self.edge, self.tail = cfg.stream_edge, cfg.stream_tail
        self.horizon, self.n_modes = cfg.horizon, cfg.n_modes
        self.lat, self.lon, self.elev = float(lat), float(lon), float(elev)
        f = lambda v: np.array([[v]], np.float32)
        self._lat, self._lon = f(lat), f(lon)
        self.loc, *coefs, self.z0 = backend.run("init", self._lat, self._lon, f(elev))
        self.coefs = coefs
        self.conformal = self._conformal(conformal)
        self.aci = ACIParams() if aci is True else aci
        self.qc = CausalQC(elev=self.elev)
        self.loaded_site = None
        self.idle_hours = 0
        self.reset_calibration()
        self.reset()

    def reset_calibration(self, theta=0.0):
        """Сброс адаптивной калибровки прибора: множитель и счётчики обратной связи."""
        self.theta = float(np.float32(theta)) if self.aci is None else self.aci.clip(theta)
        self.aci_updates = 0
        self.aci_misses = 0
        self._pending = None

    def reset(self, last_hour=None):
        """Холодный старт: окно состоит из пустых часов. Множитель калибровки сохраняется.

        Args:
            last_hour: час, которым заканчивается пустое окно. None - момент ещё не
                известен, окно строится при первом шаге.
        """
        W, M = self.window, self.n_modes
        self._pending = None
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
    def memory_nbytes(self):
        """Размер колец и буфера энкодера в памяти, байт. На диск они не пишутся."""
        return int(self.u_ring.nbytes + self.v_ring.nbytes + self.rows.nbytes
                   + self.enc_buf.nbytes)

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

    def _rebuild(self, seed_qc):
        """Всё модельное состояние из сырого окна одним пакетным проходом."""
        W = self.window
        hours = self._hours(W)
        x, m, doy, hour = self._window_arrays(hours)
        enc, u, v, rows, n_re, n_im, e = self.b.run(
            "window", x[None], m[None], doy[None], hour[None], self._lat, self._lon, *self.coefs)
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
        self._push((T, P, RH), hour)

    def _push(self, values, hour):
        xj, codes = self.qc.push(values)
        raw, present = self.qc.latest()
        valid = (codes == 0).astype(np.uint8)
        if self.aci is not None and valid[0]:
            self._aci_feedback(float(xj[0]), hour)
        j = hour % self.window
        self.raw[j] = np.where(present > 0, to_store(raw), 0.0)
        self.present[j], self.valid[j] = present, valid
        self._ingest(hour)

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

    def _aci_feedback(self, y, hour):
        """Сверка валидной температуры часа с последним выпущенным прогнозом."""
        p = self._pending
        if p is None:
            return
        k = hour - p["first"]
        if k < 0 or k >= len(p["q"]) or k <= p["last"]:
            return
        p["last"] = k
        score = float(aci_score(y, p["q"][k], self.aci.interval))
        self.theta, miss = self.aci.step(self.theta, score)
        self.aci_updates += 1
        self.aci_misses += int(miss)

    @property
    def aci_coverage(self):
        """Фактическое покрытие по обратной связи с момента последнего сброса калибровки."""
        if not self.aci_updates:
            return float("nan")
        return 1.0 - self.aci_misses / self.aci_updates

    def raw_forecast(self):
        """Квантили модели до калибровки на часы после последнего шага.

        Returns:
            Массив float32 формы (H, число квантилей).

        Raises:
            ValueError: не было ни одного шага, момент выпуска не определён.
        """
        if self.last_hour is None:
            raise ValueError("нет ни одного шага: момент выпуска не определён")
        L, E = self.history, self.edge
        hours = self._hours(L)
        rows = self.rows[hours % L][None]
        xe, me, de, he = self._window_arrays(hours[:E])
        df, hf = window_calendar(0, self.last_hour + 1 + np.arange(self.horizon))
        (q,) = self.b.run("issue", self.loc, self._lat, self._lon, *self.coefs, rows,
                          *self.modes, xe[None], me[None], de[None], he[None], df[None],
                          hf[None])
        return q[0]

    def forecast(self):
        """Выпуск на часы после последнего шага.

        Returns:
            Пара: квантили float32 формы (H, число квантилей) и медиана формы (H,).
        """
        q = self.raw_forecast()
        if self.conformal is not None:
            q = apply_conformal(q, self.conformal)
        if self.aci is not None:
            self._pending = dict(first=self.last_hour + 1, q=np.array(q, np.float32), last=-1)
        return apply_adaptive(q, self.theta)

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
        """Таблица поправок, которую можно применять к выходам этой модели.

        Модель здесь считает во fp32. Таблица, подогнанная на другой точности, не
        применяется, причина пишется в лог.

        Args:
            conformal: None, путь к таблице с записью о подгонке рядом или сама таблица.

        Returns:
            Таблица float32 или None.
        """
        if conformal is None:
            return None
        if not isinstance(conformal, str):
            shift = np.asarray(conformal, np.float32)
            check_median_free(shift)
            return shift
        shift, rec = load_conformal(conformal)
        why = precision_mismatch(rec, "fp32")
        if why:
            log.warning("конформная таблица %s не применяется: %s", conformal, why)
            return None
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
        hdr["aci_theta"] = self.theta
        hdr["lat"], hdr["lon"], hdr["elev"] = self.lat, self.lon, self.elev
        pos = np.arange(W) if self.last_hour is None else self._hours(W) % W
        return hdr.tobytes() + encode_window(self.raw[pos], self.present[pos], self.valid[pos])

    def load_state(self, raw):
        """Состояние с диска и восстановление всего остального одним проходом по окну.

        Args:
            raw: байты состояния.

        Raises:
            ValueError: байты не состояние этого формата, не подходят конфигу модели или
                повреждены. Рантайм тогда остаётся в холодном старте.
        """
        raw = bytes(raw)
        if len(raw) < len(STATE_MAGIC) + 1 or raw[:len(STATE_MAGIC)] != STATE_MAGIC:
            raise ValueError("не состояние МАЯК: нет заголовка")
        version = raw[len(STATE_MAGIC)]
        if version != STATE_VERSION:
            raise ValueError(f"версия состояния {version}, рантайм читает {STATE_VERSION}; "
                             f"прежние версии не хранят сырое окно целиком, нужен холодный "
                             f"старт")
        if len(raw) != self.state_nbytes:
            raise ValueError(f"состояние {len(raw)} Б не соответствует конфигу модели "
                             f"(ожидалось {self.state_nbytes} Б)")
        hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
        theta, filled, last = float(hdr["aci_theta"]), int(hdr["filled"]), int(hdr["last_hour"])
        W = self.window
        if not np.isfinite(theta):
            raise ValueError(f"повреждённый множитель калибровки в состоянии: {theta}")
        if filled > W or (last == NO_HOUR and filled):
            raise ValueError(f"повреждённый заголовок состояния: filled={filled}, окно {W}, "
                             f"последний час {last}")
        x, present, valid = decode_window(raw[STATE_HEADER.itemsize:], W)
        if np.any(valid > present):
            raise ValueError("повреждённое окно: годный час без значения")
        ok = valid > 0
        if np.any(ok & ((x < PHYS_LO) | (x > PHYS_HI))):
            raise ValueError("повреждённое окно: годное значение вне физического диапазона")
        self.reset()
        self.reset_calibration(theta)
        self.loaded_site = (float(hdr["lat"]), float(hdr["lon"]), float(hdr["elev"]))
        if last == NO_HOUR:
            return
        self.last_hour = last
        pos = self._hours(W) % W
        self.raw[pos], self.present[pos], self.valid[pos] = x, present, valid
        self.filled = filled
        self._rebuild(seed_qc=True)


def safe_forecast(stream, mu_clim_fut, sigma_clim):
    """Выпуск с откатом к климатологии при любой ошибке.

    Args:
        stream: потоковый рантайм.
        mu_clim_fut: климатическое среднее на часах горизонта, форма (H,).
        sigma_clim: климатический разброс.

    Returns:
        Пара: квантили и медиана.
    """
    try:
        return stream.forecast()
    except Exception:
        from mayak.baselines import quantiles_from_normal
        mu = np.asarray(mu_clim_fut, np.float32)
        q = quantiles_from_normal(mu, np.full(mu.shape[-1], sigma_clim, np.float32))
        return q, mu


__all__ = ["CTX", "NO_HOUR", "RAW_CHANNELS", "RESYNC_HOURS", "STATE_HEADER", "STATE_VERSION",
           "StreamingMayak", "decode_window", "encode_window", "mask_bytes", "safe_forecast",
           "state_nbytes", "to_store"]
