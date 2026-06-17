"""Потоковый рантайм МАЯК для устройства"""
import numpy as np
import torch

from mayak.constants import L_MAX, H, M, DZ
from mayak.astro import astro_features
from mayak.data.qc import PHYS

RF = 256
LEAD_BINS = [(1, 6), (7, 24), (25, 72), (73, 168)]

def _lead_bin(h1):
    for i, (a, b) in enumerate(LEAD_BINS):
        if a <= h1 <= b:
            return i
    return len(LEAD_BINS) - 1

class StreamingMayak:
    def __init__(self, model, lat, lon, elev, conformal=None):
        self.m = model.eval()
        self.lat = float(lat); self.lon = float(lon); self.elev = float(elev)

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
        self.n_re = torch.zeros(1, M); self.n_im = torch.zeros(1, M); self.e = torch.zeros(1, M)
        self.buf_x = np.zeros((RF, 3), np.float32)
        self.buf_m = np.zeros((RF, 3), np.float32)
        self.buf_doy = np.zeros(RF, np.float32)
        self.buf_hour = np.zeros(RF, np.float32)
        self.filled = 0
        self.day_summ = torch.zeros(1, 28, 6); self.day_mask = torch.zeros(1, 28)
        self._cur_day_aT = []; self._cur_day_dp = []; self._hours_in_day = 0
        self.z = self._recompute_passport()

    def _recompute_passport(self):
        with torch.no_grad():
            z, _ = self.m.passport(self.loc, self.day_summ, self.day_mask, sample=False)
        return z

    def _features_over_buffer(self):
        n = RF
        x = torch.from_numpy(self.buf_x)[None]
        mk = torch.from_numpy(self.buf_m)[None]
        doy = torch.from_numpy(self.buf_doy)[None]
        hour = torch.from_numpy(self.buf_hour)[None]
        with torch.no_grad():
            astro_h = astro_features(doy, hour, torch.tensor([[self.lat]]),
                                     torch.tensor([[self.lon]]))
            mu0, sg0, df0 = self.m.field.evaluate(self.base_coefs, astro_h)
            ch, aT, vt = self.m.build_channels(x, mk, astro_h, mu0, sg0, df0)
            feats = self.m.encoder(ch)                    # (1,RF,C)
        return feats[:, -1], aT[:, -1], vt[:, -1], ch[:, 3, -1]

    @staticmethod
    def _qc_point(T, P, RH):
        out = np.zeros(3, np.float32); mask = np.zeros(3, np.float32)
        for j, (name, val) in enumerate([("T", T), ("P", P), ("RH", RH)]):
            lo, hi = PHYS[name]
            if val is not None and lo <= val <= hi and np.isfinite(val):
                out[j] = val; mask[j] = 1.0
        return out, mask

    def warm_start(self, x_hist, mask_hist, doy_hist, hour_hist):
        x = torch.as_tensor(x_hist, dtype=torch.float32)[None]
        mk = torch.as_tensor(mask_hist, dtype=torch.float32)[None]
        doy = torch.as_tensor(doy_hist, dtype=torch.float32)[None]
        hour = torch.as_tensor(hour_hist, dtype=torch.float32)[None]
        with torch.no_grad():
            astro_h = astro_features(doy, hour, torch.tensor([[self.lat]]),
                                     torch.tensor([[self.lon]]))
            mu0, sg0, df0 = self.m.field.evaluate(self.base_coefs, astro_h)
            ch, aT, vt = self.m.build_channels(x, mk, astro_h, mu0, sg0, df0)
            self.day_summ, self.day_mask = self.m.daily_summaries(aT, ch[:, 3], vt)
            self.z = self._recompute_passport()
            feats = self.m.encoder(ch)
            state = (self.n_re, self.n_im, self.e)
            for k in range(feats.shape[1]):
                state = self.m.readout.step(state, feats[:, k], vt[:, k])
            self.n_re, self.n_im, self.e = state
        L = x_hist.shape[0]
        take = min(RF, L)
        self.buf_x[-take:] = np.asarray(x_hist)[-take:]
        self.buf_m[-take:] = np.asarray(mask_hist)[-take:]
        self.buf_doy[-take:] = np.asarray(doy_hist)[-take:]
        self.buf_hour[-take:] = np.asarray(hour_hist)[-take:]
        self.filled = take

    def step(self, T, P, RH, doy, hour):
        xj, mj = self._qc_point(T, P, RH)
        self.buf_x[:-1] = self.buf_x[1:]; self.buf_x[-1] = xj
        self.buf_m[:-1] = self.buf_m[1:]; self.buf_m[-1] = mj
        self.buf_doy[:-1] = self.buf_doy[1:]; self.buf_doy[-1] = doy
        self.buf_hour[:-1] = self.buf_hour[1:]; self.buf_hour[-1] = hour
        self.filled = min(RF, self.filled + 1)
        feat, aT, vt, dp = self._features_over_buffer()
        with torch.no_grad():
            self.n_re, self.n_im, self.e = self.m.readout.step(
                (self.n_re, self.n_im, self.e), feat, vt)
        self._cur_day_aT.append(float(aT)); self._cur_day_dp.append(float(dp))
        self._hours_in_day += 1
        if self._hours_in_day >= 24:
            a = torch.tensor(self._cur_day_aT); p = torch.tensor(self._cur_day_dp)
            summ = torch.tensor([a.mean(), a.max(), a.min(), p.mean(), len(a) / 24.0, 1.0])
            self.day_summ = torch.cat([self.day_summ[:, 1:], summ[None, None]], dim=1)
            self.day_mask = torch.cat([self.day_mask[:, 1:], torch.ones(1, 1)], dim=1)
            self.z = self._recompute_passport()
            self._cur_day_aT, self._cur_day_dp, self._hours_in_day = [], [], 0

    @torch.no_grad()
    def forecast(self, doy_fut, hour_fut):
        a_re = self.n_re / (self.e + self.kappa)
        a_im = self.n_im / (self.e + self.kappa)
        doy = torch.as_tensor(doy_fut, dtype=torch.float32)[None]
        hour = torch.as_tensor(hour_fut, dtype=torch.float32)[None]
        astro_f = astro_features(doy, hour, torch.tensor([[self.lat]]),
                                 torch.tensor([[self.lon]]))
        coefs = self.m.field.coefficients(self.loc, self.z)
        mu_c, sigma_c, _ = self.m.field.evaluate(coefs, astro_f)
        o, Eg = self.m.propagator(a_re, a_im, self.z, self.tau, self.omega)
        sun_fut = torch.stack([astro_f[0], astro_f[1], astro_f[3]], dim=-1)
        r, ratio, off = self.m.heads(o, Eg, sun_fut, torch.log(sigma_c), self.z, self.e)
        mu = mu_c + sigma_c * (o + r)
        q = mu[..., None] + (sigma_c * ratio)[..., None] * off

        q = q[0].numpy()
        mu = mu[0].numpy()
        if self.conformal is not None:
            for h in range(H):
                q[h] += self.conformal[_lead_bin(h + 1)]
            q = np.maximum.accumulate(q, axis=-1)
            mu = q[:, 3]
        return q, mu

    def serialize(self):
        return b"".join([
            self.n_re.numpy().astype(np.float32).tobytes(),
            self.n_im.numpy().astype(np.float32).tobytes(),
            self.e.numpy().astype(np.float32).tobytes(),
            self.z.numpy().astype(np.float32).tobytes(),
            self.day_summ.numpy().astype(np.float16).tobytes(),
            self.day_mask.numpy().astype(np.float16).tobytes(),
            self.buf_x.astype(np.float16).tobytes(),
            (self.buf_m > 0).astype(np.uint8).tobytes(),
        ])

    def load_state(self, raw):
        off = 0
        def take(shape, dtype):
            nonlocal off
            cnt = int(np.prod(shape))
            arr = np.frombuffer(raw, dtype=dtype, count=cnt, offset=off).reshape(shape).copy()
            off += cnt * np.dtype(dtype).itemsize
            return arr
        self.n_re = torch.from_numpy(take((1, M), np.float32))
        self.n_im = torch.from_numpy(take((1, M), np.float32))
        self.e = torch.from_numpy(take((1, M), np.float32))
        self.z = torch.from_numpy(take((1, DZ), np.float32))
        self.day_summ = torch.from_numpy(take((1, 28, 6), np.float16).astype(np.float32))
        self.day_mask = torch.from_numpy(take((1, 28), np.float16).astype(np.float32))
        self.buf_x = take((RF, 3), np.float16).astype(np.float32)
        self.buf_m = take((RF, 3), np.uint8).astype(np.float32)


def safe_forecast(stream, doy_fut, hour_fut, mu_clim_fut, sigma_clim):
    try:
        return stream.forecast(doy_fut, hour_fut)
    except Exception:
        from mayak.baselines import quantiles_from_normal
        mu = np.asarray(mu_clim_fut, np.float32)
        q = quantiles_from_normal(mu, np.full(H, sigma_clim, np.float32))
        return q, mu