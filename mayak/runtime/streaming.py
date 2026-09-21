"""Потоковый рантайм МАЯК для устройства.

Все размеры (число мод, размер паспорта, число суток сводок, длина окна) берутся из
конфига модели (model.cfg), а не из глобальных констант: рантайм работает с любой
конфигурацией и любой абляцией, с которой обучена модель.

Стоимость часа. ``step`` не пересчитывает энкодер по окну: каналы часа
считаются по хвосту из CHANNEL_MAX_LAG + 1 часов, энкодер делает один потактовый шаг
по кольцевым буферам блоков (SynopticEncoder.step), моды - один шаг O(M). Ни одна
операция шага не зависит от длины рецептивного поля.

Состояние делится на две части.

* Персистентное (``serialize`` / ``load_state``): заголовок, моды
  (n_re, n_im, e), паспорт z, суточные сводки и их маска, сырое окно наблюдений
  (``stream_window`` часов, фиксированная точка uint16 на канал) и его маска, курсор
  (сколько часов окна заполнено, сколько часов накоплено в текущих сутках, час года
  первой и последней позиции окна).
* Эфемерное: кольцевые буферы энкодера, календарь окна, накопители текущих суток.
  На диск не пишется; ``load_state`` восстанавливает его одним пакетным проходом
  энкодера по сохранённому окну (цена платится один раз при старте).
"""
import logging

import numpy as np
import torch

from mayak.astro import astro_features
from mayak.config import CHANNEL_MAX_LAG
from mayak.data.qc import PHYS, point_qc
from mayak.metrics import I_MED, apply_conformal

log = logging.getLogger(__name__)

STATE_MAGIC = b"MYK"
STATE_VERSION = 2
STATE_HEADER = np.dtype([("magic", "S3"), ("version", "u1"), ("filled", "<u2"),
                         ("hours_in_day", "u1"), ("reserved", "u1"),
                         ("hoy_first", "<u2"), ("hoy_last", "<u2")])
YEAR_HOURS = (365 * 24, 366 * 24)

RAW_CHANNELS = ("T", "P", "RH")
RAW_LO = np.array([PHYS[c][0] for c in RAW_CHANNELS], np.float64)
RAW_HI = np.array([PHYS[c][1] for c in RAW_CHANNELS], np.float64)
RAW_STEP = (RAW_HI - RAW_LO) / 65534.0


def encode_raw(x, m):
    q = np.rint((np.clip(np.asarray(x, np.float64), RAW_LO, RAW_HI) - RAW_LO) / RAW_STEP)
    return np.where(np.asarray(m) > 0, q, 0).astype("<u2")


def decode_raw(q, m):
    return ((RAW_LO + q.astype(np.float64) * RAW_STEP) * (np.asarray(m) > 0)).astype(np.float32)


def hour_of_year(doy):
    """doy по конвенции mayak.timeaxis (день года с нуля, с долей суток) → час года."""
    return int(round(float(doy) * 24.0))


class StreamingMayak:
    def __init__(self, model, lat, lon, elev, conformal=None):
        self.m = model.eval()
        cfg = model.cfg
        self.window = cfg.stream_window
        self.n_modes, self.dz, self.n_days = cfg.n_modes, cfg.passport_dim, cfg.history_days
        self.ctx = CHANNEL_MAX_LAG + 1
        self.lat, self.lon, self.elev = float(lat), float(lon), float(elev)
        self._lat_t = torch.tensor([[self.lat]])
        self._lon_t = torch.tensor([[self.lon]])

        if conformal is None:
            self.conformal = None
        elif isinstance(conformal, str):
            self.conformal = np.load(conformal).astype(np.float32)
        else:
            self.conformal = np.asarray(conformal, np.float32)

        with torch.no_grad():
            self.loc = self.m.loc(torch.tensor([lat]), torch.tensor([lon]),
                                  torch.tensor([elev]))
            self.base_coefs = self.m.field.coefficients(self.loc)
            tau, omega, kappa = self.m.readout.constants()
        self.tau, self.omega, self.kappa = tau, omega, kappa
        self.reset()

    def reset(self):
        """Холодный старт: история пуста (L = 0)."""
        M, W = self.n_modes, self.window
        self.n_re = torch.zeros(1, M)
        self.n_im = torch.zeros(1, M)
        self.e = torch.zeros(1, M)
        self.raw_x = np.zeros((W, 3), np.float32)
        self.raw_m = np.zeros((W, 3), np.float32)
        self.raw_doy = np.zeros(W, np.float32)
        self.raw_hour = np.zeros(W, np.float32)
        self.head = 0
        self.filled = 0
        self.enc = self.m.encoder.init_state(1)
        self.day_summ = torch.zeros(1, self.n_days, 6)
        self.day_mask = torch.zeros(1, self.n_days)
        self._reset_day()
        self._last_hoy = None
        self.calendar_breaks = 0
        self.z = self._recompute_passport()

    def _reset_day(self):
        self._day = np.zeros((4, 24), np.float32)
        self._hours_in_day = 0

    def _recompute_passport(self):
        with torch.no_grad():
            z, _ = self.m.passport(self.loc, self.day_summ, self.day_mask, sample=False)
        return z

    @property
    def encoder_state_nbytes(self):
        """Размер кольцевых буферов энкодера в памяти (на диск не пишутся), байт."""
        return self.enc.nbytes

    def _ordered(self, n=None):
        """Индексы кольца для последних n часов окна в хронологическом порядке."""
        n = self.window if n is None else n
        return (self.head - n + np.arange(n)) % self.window

    def _channels(self, idx):
        """Каналы энкодера по позициям окна idx → (ch, aT, vt, dP24, vp24)."""
        x = torch.from_numpy(self.raw_x[idx])[None]
        mk = torch.from_numpy(self.raw_m[idx])[None]
        doy = torch.from_numpy(self.raw_doy[idx])[None]
        hour = torch.from_numpy(self.raw_hour[idx])[None]
        astro_h = astro_features(doy, hour, self._lat_t, self._lon_t)
        mu0, sg0, df0 = self.m.field.evaluate(self.base_coefs, astro_h)
        ch, aT, vt = self.m.build_channels(x, mk, astro_h, mu0, sg0, df0)
        return ch, aT, vt, self.m.channel(ch, "dP24"), self.m.lag_valid(mk[..., 1], 24)

    def _check_calendar(self, doy):
        hoy = hour_of_year(doy)
        prev = self._last_hoy
        if prev is not None and hoy != prev + 1 and not (hoy == 0 and prev + 1 in YEAR_HOURS):
            self.calendar_breaks += 1
            log.warning("календарь потока не непрерывен: час года %d после %d. Рантайм "
                        "ждёт ровно один вызов step на час; пропуск датчика - None, "
                        "а не пропуск шага", hoy, prev)
        self._last_hoy = hoy

    @staticmethod
    def _qc_point(T, P, RH):
        """Поточечный QC часа - та же функция и те же пределы, что в mayak.data.qc."""
        return point_qc(T, P, RH)

    @torch.no_grad()
    def _ingest(self, x, m, doy, hour):
        """Один час уже прошедших QC наблюдений: окно → каналы → энкодер → моды → сутки."""
        self._check_calendar(doy)
        j = self.head
        self.raw_m[j] = m
        self.raw_x[j] = np.where(m > 0, x, 0.0)
        self.raw_doy[j], self.raw_hour[j] = doy, hour
        self.head = (j + 1) % self.window
        self.filled = min(self.window, self.filled + 1)

        ch, aT, vt, dp24, vp24 = self._channels(self._ordered(self.ctx))
        feat = self.m.encoder.step(ch[..., -1], self.enc)
        self.n_re, self.n_im, self.e = self.m.readout.step(
            (self.n_re, self.n_im, self.e), feat, vt[:, -1])
        self._accumulate_day(aT[0, -1], dp24[0, -1], vt[0, -1], vp24[0, -1])

    def _accumulate_day(self, aT, dp24, vt, vp24):
        self._day[:, self._hours_in_day] = (float(aT), float(dp24), float(vt), float(vp24))
        self._hours_in_day += 1
        if self._hours_in_day < 24:
            return
        d = torch.from_numpy(self._day)
        summ, has = self.m.daily_summaries(d[0:1], d[1:2], d[2:3], d[3:4])
        self.day_summ = torch.cat([self.day_summ[:, 1:], summ], dim=1)
        self.day_mask = torch.cat([self.day_mask[:, 1:], has], dim=1)
        self.z = self._recompute_passport()
        self._reset_day()

    def step(self, T, P, RH, doy, hour):
        """Новый час наблюдений. None или значение вне физического диапазона - пропуск."""
        xj, mj = self._qc_point(T, P, RH)
        self._ingest(xj, mj, doy, hour)

    @torch.no_grad()
    def warm_start(self, x_hist, mask_hist, doy_hist, hour_hist):
        """Прогрев по истории (L, 3): то же состояние, что после L вызовов step.

        Энкодер и сводки считаются одним пакетным проходом, моды - потактово O(M).
        Сутки считаются от начала истории, неполный остаток идёт в накопители.
        """
        self.reset()
        mk = np.asarray(mask_hist, np.float32)
        x = np.where(mk > 0, np.asarray(x_hist, np.float32), 0.0).astype(np.float32)
        doy = np.asarray(doy_hist, np.float32)
        hour = np.asarray(hour_hist, np.float32)
        L = x.shape[0]
        if L == 0:
            return
        take = min(self.window, L)
        self.raw_x[:take], self.raw_m[:take] = x[-take:], mk[-take:]
        self.raw_doy[:take], self.raw_hour[:take] = doy[-take:], hour[-take:]
        self.head, self.filled = take % self.window, take
        self._last_hoy = hour_of_year(doy[-1])

        xt, mt = torch.from_numpy(x)[None], torch.from_numpy(mk)[None]
        astro_h = astro_features(torch.from_numpy(doy)[None], torch.from_numpy(hour)[None],
                                 self._lat_t, self._lon_t)
        mu0, sg0, df0 = self.m.field.evaluate(self.base_coefs, astro_h)
        ch, aT, vt = self.m.build_channels(xt, mt, astro_h, mu0, sg0, df0)
        dp24, vp24 = self.m.channel(ch, "dP24"), self.m.lag_valid(mt[..., 1], 24)

        feats, self.enc = self.m.encoder.prefill(ch)
        state = (self.n_re, self.n_im, self.e)
        for k in range(L):
            state = self.m.readout.step(state, feats[:, k], vt[:, k])
        self.n_re, self.n_im, self.e = state

        D = L // 24
        if D:
            summ, has = self.m.daily_summaries(aT[:, :D * 24], dp24[:, :D * 24],
                                               vt[:, :D * 24], vp24[:, :D * 24])
            self.day_summ = torch.cat([self.day_summ, summ], dim=1)[:, -self.n_days:]
            self.day_mask = torch.cat([self.day_mask, has], dim=1)[:, -self.n_days:]
        r = L - D * 24
        if r:
            self._day[:, :r] = torch.stack([aT[0, -r:], dp24[0, -r:], vt[0, -r:],
                                            vp24[0, -r:]]).numpy()
        self._hours_in_day = r
        self.z = self._recompute_passport()

    @torch.no_grad()
    def forecast(self, doy_fut, hour_fut):
        a_re, a_im = self.m.readout.normalize(self.n_re, self.n_im, self.e)
        doy = torch.as_tensor(doy_fut, dtype=torch.float32)[None]
        hour = torch.as_tensor(hour_fut, dtype=torch.float32)[None]
        astro_f = astro_features(doy, hour, self._lat_t, self._lon_t)
        out = self.m.issue(self.loc, self.z, a_re, a_im, self.e, astro_f)
        q, mu = out["q"], out["mu"]

        q = q[0].numpy()
        mu = mu[0].numpy()
        if self.conformal is not None:
            q = apply_conformal(q, self.conformal)
            mu = q[:, I_MED]
        return q, mu

    def serialize(self):
        idx = self._ordered()
        n = self.filled
        hdr = np.zeros((), STATE_HEADER)
        hdr["magic"], hdr["version"] = STATE_MAGIC, STATE_VERSION
        hdr["filled"], hdr["hours_in_day"] = n, self._hours_in_day
        if n:
            hdr["hoy_first"] = hour_of_year(self.raw_doy[idx[-n]]) % 65536
            hdr["hoy_last"] = hour_of_year(self.raw_doy[idx[-1]]) % 65536
        return b"".join([
            hdr.tobytes(),
            self.n_re.numpy().astype(np.float32).tobytes(),
            self.n_im.numpy().astype(np.float32).tobytes(),
            self.e.numpy().astype(np.float32).tobytes(),
            self.z.numpy().astype(np.float32).tobytes(),
            self.day_summ.numpy().astype(np.float16).tobytes(),
            self.day_mask.numpy().astype(np.float16).tobytes(),
            encode_raw(self.raw_x[idx], self.raw_m[idx]).tobytes(),
            (self.raw_m[idx] > 0).astype(np.uint8).tobytes(),
        ])

    @property
    def state_nbytes(self):
        """Размер сериализованного состояния для конфига этой модели, байт."""
        M, D, W = self.n_modes, self.n_days, self.window
        return STATE_HEADER.itemsize + 4 * (3 * M + self.dz) + 2 * (D * 6 + D) + 2 * W * 3 + W * 3

    def load_state(self, raw):
        """Состояние с диска + восстановление эфемерной части одним проходом по окну."""
        raw = bytes(raw)
        if len(raw) < STATE_HEADER.itemsize or raw[:len(STATE_MAGIC)] != STATE_MAGIC:
            raise ValueError("не состояние МАЯК формата v2 (нет заголовка); состояния, "
                             "записанные до блока 7, не поддерживаются - нужен чистый старт")
        hdr = np.frombuffer(raw, STATE_HEADER, count=1)[0]
        if int(hdr["version"]) != STATE_VERSION:
            raise ValueError(f"версия состояния {int(hdr['version'])}, рантайм читает "
                             f"{STATE_VERSION}")
        if len(raw) != self.state_nbytes:
            raise ValueError(f"состояние {len(raw)} Б не соответствует конфигу модели "
                             f"(ожидалось {self.state_nbytes} Б)")
        filled, hid = int(hdr["filled"]), int(hdr["hours_in_day"])
        if filled > self.window or hid > 23 or hid > filled:
            raise ValueError(f"повреждённый курсор состояния: filled={filled}, "
                             f"часов в сутках {hid}, окно {self.window}")
        doy, hour = self._window_calendar(filled, int(hdr["hoy_first"]), int(hdr["hoy_last"]))

        off = STATE_HEADER.itemsize

        def take(shape, dtype):
            nonlocal off
            cnt = int(np.prod(shape))
            arr = np.frombuffer(raw, dtype=dtype, count=cnt, offset=off).reshape(shape).copy()
            off += cnt * np.dtype(dtype).itemsize
            return arr

        M, D, W = self.n_modes, self.n_days, self.window
        self.reset()
        self.n_re = torch.from_numpy(take((1, M), np.float32))
        self.n_im = torch.from_numpy(take((1, M), np.float32))
        self.e = torch.from_numpy(take((1, M), np.float32))
        self.z = torch.from_numpy(take((1, self.dz), np.float32))
        self.day_summ = torch.from_numpy(take((1, D, 6), np.float16).astype(np.float32))
        self.day_mask = torch.from_numpy(take((1, D), np.float16).astype(np.float32))
        q = take((W, 3), "<u2")
        self.raw_m = take((W, 3), np.uint8).astype(np.float32)
        self.raw_x = decode_raw(q, self.raw_m)
        self.raw_doy[W - filled:], self.raw_hour[W - filled:] = doy, hour
        self.head, self.filled = 0, filled
        self._hours_in_day = hid
        self._last_hoy = int(hdr["hoy_last"]) if filled else None
        self._rebuild_from_window()

    @staticmethod
    def _window_calendar(n, hoy_first, hoy_last):
        """Календарь n последних часов окна по часу года первой и последней позиции.

        Часы окна идут подряд, поэтому hoy_first + n − 1 − hoy_last равно нулю без
        перехода через Новый год и длине года в часах (8760 или 8784) с переходом.
        """
        if n == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        span = hoy_first + n - 1 - hoy_last
        if span != 0 and span not in YEAR_HOURS:
            raise ValueError(f"календарь окна несогласован: час года {hoy_first} … "
                             f"{hoy_last} на {n} ч")
        hoy = hoy_first + np.arange(n, dtype=np.int64)
        if span:
            hoy = np.where(hoy >= span, hoy - span, hoy)
        return (hoy / 24.0).astype(np.float32), (hoy % 24).astype(np.float32)

    @torch.no_grad()
    def _rebuild_from_window(self):
        """Кольцевые буферы энкодера и накопители текущих суток по сохранённому окну.

        Буферы после часа t зависят только от последних (RF − 1) + CHANNEL_MAX_LAG часов
        (см. ModelConfig.stream_window).
        """
        n = self.filled
        if n == 0:
            return
        ch, aT, vt, dp24, vp24 = self._channels(self._ordered(n))
        _, self.enc = self.m.encoder.prefill(ch)
        h = self._hours_in_day
        if h:
            self._day[:, :h] = torch.stack([aT[0, -h:], dp24[0, -h:], vt[0, -h:],
                                            vp24[0, -h:]]).numpy()


def safe_forecast(stream, doy_fut, hour_fut, mu_clim_fut, sigma_clim):
    try:
        return stream.forecast(doy_fut, hour_fut)
    except Exception:
        from mayak.baselines import quantiles_from_normal
        mu = np.asarray(mu_clim_fut, np.float32)
        q = quantiles_from_normal(mu, np.full(mu.shape[-1], sigma_clim, np.float32))
        return q, mu
