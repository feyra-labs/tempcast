"""Рантайм МАЯК на устройстве: часовой цикл и пакетный выпуск."""
import logging
from typing import NamedTuple

import numpy as np

from mayak.data.qc import PHYS, CausalQC
from mayak.data.recording import RECORD_SCALE
from mayak.data.window import issue_calendar, place_history, valid_history_hours
from mayak.leakage import load_conformal
from mayak.metrics import (LEAD_BINS, ZQ, ACIParams, AdaptiveCalibration, apply_adaptive,
                           apply_conformal, check_conformal_shape)
from mayak.runtime.site import (SITE_MOVED, SITE_REFINED, as_site, describe_gap,
                                load_runtime_config, site_change)
from mayak.timeaxis import hour_of_year

log = logging.getLogger(__name__)

STATE_MAGIC = b"MYK"
STATE_VERSION = 1
N_LEAD_BINS = len(LEAD_BINS)
STATE_HEADER = np.dtype([("magic", "S3"), ("version", "u1"), ("filled", "<u2"),
                         ("reserved", "<u2"), ("last_hour", "<i8"),
                         ("aci_theta", "<f4", (N_LEAD_BINS,)),
                         ("lat", "<f4"), ("lon", "<f4"), ("elev", "<f4")])
NO_HOUR = int(np.iinfo(np.int64).min)

RAW_CHANNELS = ("T", "P", "RH")
STORE_DTYPES = (np.dtype("i1"), np.dtype("<u2"), np.dtype("i1"))
STORE_SCALE = np.asarray(RECORD_SCALE, np.float64)
STORE_LO = np.array([-128.0, 0.0, -128.0])
STORE_HI = np.array([127.0, 65535.0, 127.0])
PHYS_LO = np.array([PHYS[c][0] for c in RAW_CHANNELS], np.float32)
PHYS_HI = np.array([PHYS[c][1] for c in RAW_CHANNELS], np.float32)
HOURS_OF_YEAR = 366 * 24
FORECAST_INPUTS = ("lat", "lon", "elev", "x_hist", "mask_hist", "doy_hist", "hour_hist",
                   "doy_fut", "hour_fut")
CLIMATOLOGY_INPUTS = ("lat", "lon", "elev")


class StateSnapshot(NamedTuple):
    """Разобранное состояние с диска."""
    filled: int
    last_hour: int | None
    theta: tuple
    site: tuple
    raw: np.ndarray
    present: np.ndarray
    valid: np.ndarray


def parse_state(raw, window):
    """Разбор байт состояния с проверками целостности."""
    raw = bytes(raw)
    if len(raw) < len(STATE_MAGIC) + 1 or raw[:len(STATE_MAGIC)] != STATE_MAGIC:
        raise ValueError("не состояние МАЯК: нет заголовка")
    version = raw[len(STATE_MAGIC)]
    if version != STATE_VERSION:
        raise ValueError(f"версия состояния {version}, рантайм читает {STATE_VERSION}")
    want = window_nbytes(window)
    if len(raw) != want:
        raise ValueError(f"состояние {len(raw)} Б не соответствует конфигу модели "
                         f"(ожидалось {want} Б)")
    hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
    theta = tuple(float(v) for v in hdr["aci_theta"])
    filled, last = int(hdr["filled"]), int(hdr["last_hour"])
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"повреждённый множитель калибровки в состоянии: {list(theta)}")
    if filled > window or (last == NO_HOUR and filled):
        raise ValueError(f"повреждённый заголовок состояния: filled={filled}, окно {window}, "
                         f"последний час {last}")
    x, present, valid = decode_window(raw[STATE_HEADER.itemsize:], window)
    if np.any(valid > present):
        raise ValueError("повреждённое окно: годный час без значения")
    ok = valid > 0
    if np.any(ok & ((x < PHYS_LO) | (x > PHYS_HI))):
        raise ValueError("повреждённое окно: годное значение вне физического диапазона")
    site = (float(hdr["lat"]), float(hdr["lon"]), float(hdr["elev"]))
    return StateSnapshot(filled, None if last == NO_HOUR else last, theta, site, x, present,
                         valid)


def check_climatology(clim_mu, clim_sig):
    """Проверка таблицы климатологии точки, которую вернул граф климатологии."""
    mu = np.asarray(clim_mu, np.float32).ravel()
    sig = np.asarray(clim_sig, np.float32).ravel()
    if mu.size != HOURS_OF_YEAR or sig.size != HOURS_OF_YEAR:
        raise RuntimeError(f"граф климатологии: таблица климатологии на {mu.size} и {sig.size} ч, "
                           f"нужно {HOURS_OF_YEAR}; графы экспортированы другой версией")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sig)) and np.all(sig > 0)):
        raise RuntimeError("граф климатологии: таблица климатологии точки негодна (нечисловые "
                           "значения или неположительный масштаб); это ошибка конфигурации "
                           "модели или координат")
    return mu, sig


def to_store(x):
    """Значения на сетке хранения окна."""
    q = np.clip(np.rint(np.asarray(x, np.float64) * STORE_SCALE), STORE_LO, STORE_HI)
    return (q / STORE_SCALE).astype(np.float32)


def mask_bytes(n_hours):
    """Размер одной битовой маски окна в байтах."""
    return (3 * n_hours + 7) // 8


def window_nbytes(window):
    """Размер сериализованного состояния для длины окна."""
    W = int(window)
    return (STATE_HEADER.itemsize + W * sum(d.itemsize for d in STORE_DTYPES)
            + 2 * mask_bytes(W))


def state_nbytes(cfg):
    """Размер сериализованного состояния для конфига модели."""
    return window_nbytes(cfg.device_window)


def encode_window(raw, present, valid):
    """Сырое окно в байты состояния."""
    p = np.asarray(present) > 0
    q = np.where(p, np.rint(np.asarray(raw, np.float64) * STORE_SCALE), 0.0)
    parts = [q[:, c].astype(STORE_DTYPES[c]).tobytes() for c in range(3)]
    for mk in (p, np.asarray(valid) > 0):
        parts.append(np.packbits(mk.ravel(), bitorder="little").tobytes())
    return b"".join(parts)


def decode_window(buf, n_hours):
    """Байты окна в значения и маски."""
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


class Device:
    """Устройство: часовой цикл, пакетный выпуск, откат и состояние на диске."""

    def __init__(self, backend, cfg, lat, lon, elev, conformal=None, aci=None,
                 runtime_cfg=None):
        self.b, self.cfg = backend, cfg
        self.window, self.history = cfg.device_window, cfg.max_history
        self.horizon = cfg.horizon
        self.lat, self.lon, self.elev = float(lat), float(lon), float(elev)
        clim_mu, clim_sig = backend.run("climatology", self._site_inputs())
        self.clim_mu, self.clim_sig = check_climatology(clim_mu, clim_sig)
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

    def _site_inputs(self):
        """Координаты и высота точки как входы графов, форма (1,) каждая."""
        f = lambda v: np.array([v], np.float32)
        return dict(lat=f(self.lat), lon=f(self.lon), elev=f(self.elev))

    def reset_calibration(self, theta=0.0):
        """Сброс адаптивной калибровки прибора: множители, счётчики и кольцо."""
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

    @property
    def valid_hours(self):
        """Часы с валидной температурой во входе модели; по ним выбирается строка таблицы."""
        if self.last_hour is None:
            return 0
        return valid_history_hours(self.model_inputs()["mask_hist"][0])

    def reset(self, last_hour=None):
        """Холодный старт: окно состоит из пустых часов. Множители калибровки сохраняются."""
        W = self.window
        self.cal.clear()
        self.qc.reset()
        self.raw = np.zeros((W, 3), np.float32)
        self.present = np.zeros((W, 3), np.uint8)
        self.valid = np.zeros((W, 3), np.uint8)
        self.filled = 0
        self.last_hour = None if last_hour is None else int(last_hour)

    @property
    def site(self):
        """Текущие широта, долгота и высота так, как они пишутся в состояние."""
        return as_site((self.lat, self.lon, self.elev))

    @property
    def memory_nbytes(self):
        """Размер окна, таблицы климатологии и кольца калибровки в памяти, байт."""
        return int(self.raw.nbytes + self.present.nbytes + self.valid.nbytes
                   + self.clim_mu.nbytes + self.clim_sig.nbytes + self.cal.nbytes)

    def _hours(self, n):
        """Абсолютные часы последних n часов окна, от старых к новым."""
        return self.last_hour - n + 1 + np.arange(n, dtype=np.int64)

    def step(self, T, P, RH, hour):
        """Новый час наблюдений: контроль качества, запись в окно, обратная связь калибровки."""
        hour = int(hour)
        if self.last_hour is not None:
            if hour <= self.last_hour:
                raise ValueError(f"час {hour} не позже последнего шага {self.last_hour}")
            gap = hour - self.last_hour - 1
            if gap >= self.window:
                log.info("простой %d ч не короче окна %d ч: холодный старт", gap, self.window)
                self.reset()
            else:
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
        self.last_hour = hour
        self.filled = min(self.window, self.filled + 1)
        return codes

    def issue_hour(self, now_hour=None):
        """Момент выпуска: последний шаг, а до первого шага - текущий час устройства."""
        if self.last_hour is not None:
            return self.last_hour
        if now_hour is None:
            raise ValueError("нет ни одного шага и не задан текущий час: момент выпуска не "
                             "определён")
        return int(now_hour)

    def model_inputs(self, now_hour=None):
        """Входы графа прогноза по окну, как у окна оценки с той же историей."""
        last = self.issue_hour(now_hour)
        n = self.history_length
        pos = (last - n + 1 + np.arange(n, dtype=np.int64)) % self.window
        x, m = place_history(self.raw[pos], self.valid[pos], self.history)
        doy_h, hour_h, doy_f, hour_f = issue_calendar(0, last + 1, self.history, self.horizon)
        return dict(self._site_inputs(), x_hist=x[None], mask_hist=m[None],
                    doy_hist=doy_h[None], hour_hist=hour_h[None], doy_fut=doy_f[None],
                    hour_fut=hour_f[None])

    def raw_forecast(self, now_hour=None):
        """Квантили модели до калибровки на часы после момента выпуска."""
        return self._run_forecast(self.model_inputs(now_hour))

    def _run_forecast(self, inputs):
        """Проход графа прогноза по готовым входам; квантили формы (H, число квантилей)."""
        (q,) = self.b.run("forecast", inputs)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("выход графа прогноза не конечен")
        return q[0]

    def forecast(self, now_hour=None):
        """Выпуск на часы после момента выпуска."""
        inputs = self.model_inputs(now_hour)
        q = self._run_forecast(inputs)
        if self.conformal is not None:
            q = apply_conformal(q, self.conformal, valid_history_hours(inputs["mask_hist"][0]))
        self.cal.record(self.issue_hour(now_hour), q)
        return apply_adaptive(q, self.theta)

    def climatology_forecast(self, last):
        """Откат: квантили нормального распределения по таблице климатологии точки."""
        idx = hour_of_year(int(last) + 1 + np.arange(self.horizon, dtype=np.int64))
        mu, sig = self.clim_mu[idx], self.clim_sig[idx]
        q = mu[:, None] + self.zq[None, :] * sig[:, None]
        return q.astype(np.float32), mu.copy()

    def safe_forecast(self, now_hour=None):
        """Выпуск, а при любой ошибке выпуска - откат к климатологии точки."""
        try:
            q, mu = self.forecast(now_hour)
            return q, mu, False
        except Exception as e:
            last = self.issue_hour(now_hour)
            log.warning("откат к климатологии: %s", e)
            self.fallbacks += 1
            q, mu = self.climatology_forecast(last)
            return q, mu, True

    @staticmethod
    def _conformal(conformal):
        """Таблица поправок для выходов модели."""
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
        """Состояние с диска: окно, момент последнего шага, множители калибровки."""
        snap = parse_state(raw, self.window)
        kind, gap = site_change(snap.site, self.site, self.runtime_cfg)
        self.reset()
        self.loaded_site, self.site_change = snap.site, kind
        if kind == SITE_MOVED:
            log.warning("прибор перенесён: %s; холодный старт, множитель калибровки сброшен",
                        describe_gap(gap, self.runtime_cfg))
            self.reset_calibration(0.0)
            self.reset(snap.last_hour)
            return
        if kind == SITE_REFINED:
            log.warning("координаты уточнены: %s; окно сохранено, прогноз строится для новой "
                        "точки, множитель калибровки сохранён",
                        describe_gap(gap, self.runtime_cfg))
        self.reset_calibration(snap.theta)
        if snap.last_hour is None:
            return
        W = self.window
        self.last_hour = snap.last_hour
        pos = self._hours(W) % W
        self.raw[pos], self.present[pos], self.valid[pos] = snap.raw, snap.present, snap.valid
        self.filled = snap.filled
        pos = self._hours(self.qc.size) % W
        self.qc.seed(self.raw[pos], self.present[pos])


__all__ = ["CLIMATOLOGY_INPUTS", "FORECAST_INPUTS", "HOURS_OF_YEAR", "NO_HOUR", "N_LEAD_BINS",
           "RAW_CHANNELS", "STATE_HEADER", "STATE_VERSION",
           "Device", "StateSnapshot", "check_climatology", "decode_window", "encode_window",
           "mask_bytes", "parse_state", "state_nbytes", "to_store", "window_nbytes"]
