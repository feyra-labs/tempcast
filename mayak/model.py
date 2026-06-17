import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import M, H
from mayak.astro import astro_features, dewpoint_c
from mayak.modules.loc import LocEncoder
from mayak.modules.field import ClimateField
from mayak.modules.passport import Fingerprint
from mayak.modules.encoder import SynopticEncoder
from mayak.modules.readout import LaplaceReadout
from mayak.modules.propagator import ModalPropagator
from mayak.modules.heads import Heads


class MAYAK(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = LocEncoder()
        self.field = ClimateField()
        self.passport = Fingerprint()
        self.encoder = SynopticEncoder()
        self.readout = LaplaceReadout()
        self.propagator =  ModalPropagator()
        self.heads = Heads()

    @staticmethod
    def build_channels(x, mask, astro_h, mu_c, sigma_c, defc):
        T, P, RH = x[..., 0], x[..., 1], x[..., 2]
        vt, vp, vr = mask[..., 0], mask[..., 1], mask[..., 2]

        aT = ((T - mu_c) / sigma_c).clamp(-8, 8) * vt
        Td = dewpoint_c(T, RH)
        adef = (((T - Td).clamp(min=0.0) - defc) / sigma_c).clamp(-8, 8) * vt * vr

        def dP(p, vpm, k, scale):
            ps = F.pad(p, (k, 0))[..., :p.shape[-1]]
            vs = F.pad(vpm, (k, 0))[..., :p.shape[-1]]
            return (((p - ps) / scale).clamp(-4, 4)) * vpm * vs

        sin_d, cos_d, _, czp, sin_y, cos_y = astro_h
        ch = torch.stack([
            aT,
            adef,
            dP(P, vp, 3, 3.0),
            dP(P, vp, 24, 8.0),
            (RH / 100.0 - 0.5) * vr,
            sin_d, cos_d, czp,
            sin_y, cos_y,
            vt, vp, vr,
        ], dim=1)
        return ch, aT, vt
    
    @staticmethod
    def daily_summaries(aT, adP24, vt):
        Bsz = aT.shape[0]
        a = aT.view(Bsz, 28, 24)
        v = vt.view(Bsz, 28, 24)
        p = adP24.view(Bsz, 28, 24)
        n = v.sum(-1)
        mean = (a * v).sum(-1) / n.clamp(min=1.0)
        mx = a.masked_fill(v < 0.5, -1e4).amax(-1)
        mn = a.masked_fill(v < 0.5, 1e4).amin(-1)
        has = (n > 0).float()
        mx, mn = mx * has, mn * has
        mp = (p * v).sum(-1) / n.clamp(min=1.0)
        s = torch.stack([mean, mx, mn, mp, n / 24.0, has], dim=-1)
        return s, has
    
    def forward(self, batch):
        lat, lon, elev = batch["lat"], batch["lon"], batch["elev"]
        x, mask = batch["x_hist"], batch["mask_hist"]
        loc = self.loc(lat, lon, elev)

        astro_h = astro_features(batch["doy_hist"], batch["hour_hist"],
                                 lat[:, None], lon[:, None])
        astro_f = astro_features(batch["doy_fut"], batch["hour_fut"],
                                 lat[:, None], lon[:, None])

        mu0, sg0, df0 = self.field.evaluate(self.field.coefficients(loc), astro_h)
        ch, aT, vt = self.build_channels(x, mask, astro_h, mu0, sg0, df0)

        summ, day_mask = self.daily_summaries(aT, ch[:, 3], vt)
        z, kl = self.passport(loc, summ, day_mask, sample=self.training)

        feats = self.encoder(ch)
        a_re, a_im, e = self.readout(feats, vt)

        mu_c, sigma_c, _ = self.field.evaluate(self.field.coefficients(loc, z), astro_f)

        tau, omega, _ = self.readout.constants()
        o, Eg = self.propagator(a_re, a_im, z, tau, omega)
        sun_fut = torch.stack([astro_f[0], astro_f[1], astro_f[3]], dim=-1)

        r, ratio, off = self.heads(o, Eg, sun_fut, torch.log(sigma_c), z, e)
        mu = mu_c + sigma_c * (o + r)
        q = mu[..., None] + (sigma_c * ratio)[..., None] * off
        return {"q": q, "mu": mu, "sigma_c": sigma_c, "o": o, "r": r,
                "ratio": ratio, "Eg": Eg, "a_re": a_re, "a_im": a_im,
                "e": e, "kl": kl, "z": z}
    

# SMOKE тест модели
if __name__ == "__main__":
    import math
    from mayak.constants import L_MAX
    from mayak.loss import mayak_loss

    torch.manual_seed(0)
    Bsz = 2
    model = MAYAK()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_par/1e3:.1f} тыс.")

    hours = torch.arange(L_MAX, dtype=torch.float32)
    doy_h = ((hours / 24.0 + 120.0) % 365.24).expand(Bsz, -1).clone()
    hour_h = (hours % 24.0).expand(Bsz, -1).clone()
    hf = torch.arange(1, H + 1, dtype=torch.float32)
    doy_f = (((L_MAX + hf) / 24.0 + 120.0) % 365.24).expand(Bsz, -1).clone()
    hour_f = ((L_MAX + hf) % 24.0).expand(Bsz, -1).clone()

    T = 12 + 6 * torch.sin(2 * math.pi * hours / 24.0) + torch.randn(Bsz, L_MAX)
    P = 1013 + 3 * torch.randn(Bsz, L_MAX)
    RH = (70 + 10 * torch.randn(Bsz, L_MAX)).clamp(5, 100)
    x = torch.stack([T, P, RH], dim=-1)

    mask = torch.ones(Bsz, L_MAX, 3)
    mask[1, : L_MAX - 30] = 0.0
    y = 12 + 6 * torch.sin(2 * math.pi * (L_MAX + hf) / 24.0) + torch.randn(Bsz, H)

    batch = dict(lat=torch.tensor([52.4, -1.3]), lon=torch.tensor([4.9, 36.8]),
                 elev=torch.tensor([-2.0, 1600.0]), x_hist=x, mask_hist=mask,
                 doy_hist=doy_h, hour_hist=hour_h, doy_fut=doy_f, hour_fut=hour_f)

    out = model(batch)
    loss = mayak_loss(out, y)
    loss.backward()
    assert (out["q"].diff(dim=-1) >= 0).all(), "квантили обязаны быть монотонны"
    print("q:", tuple(out["q"].shape), "| e:", out["e"].mean(-1).tolist(),
          "| loss:", float(loss))

    state = (torch.zeros(Bsz, M), torch.zeros(Bsz, M), torch.zeros(Bsz, M))
    loc = model.loc(batch["lat"], batch["lon"], batch["elev"])
    a_h = astro_features(doy_h, hour_h, batch["lat"][:, None], batch["lon"][:, None])
    mu0, sg0, df0 = model.field.evaluate(model.field.coefficients(loc), a_h)
    feats = model.encoder(model.build_channels(x, mask, a_h, mu0, sg0, df0)[0])
    vt = mask[..., 0]
    for k in range(L_MAX):
        state = model.readout.step(state, feats[:, k], vt[:, k])
    print("Потоковая масса e после 672 шагов:", state[2].mean(-1).tolist())