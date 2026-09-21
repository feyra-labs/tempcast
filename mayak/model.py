import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.astro import astro_features, dewpoint_c
from mayak.config import N_DAILY_SUMMARY, ModelConfig
from mayak.modules.loc import LocEncoder
from mayak.modules.field import ClimateField, ConstantAnchor
from mayak.modules.passport import Fingerprint
from mayak.modules.encoder import SynopticEncoder
from mayak.modules.readout import LaplaceReadout
from mayak.modules.propagator import ModalPropagator
from mayak.modules.heads import Heads
from mayak.loss import mayak_regularizers


class MAYAK(nn.Module):
    """МАЯК: прогноз = климат-поле + σ·(аномалия из затухающих мод + поправка).

    Вся архитектура задаётся ModelConfig (mayak/config.py): размеры, группы мод,
    квантили, флаги абляций. Производные размерности (каналы энкодера, вход голов,
    число групп) вычисляются из конфига. Флаги абляций меняют только эту модель:
    потоковый рантайм и оценка вызывают те же методы (build_channels, issue,
    readout.normalize) и получают абляцию автоматически.
    """
    NO_WD_SUFFIX = ("raw_tau", "p_w", "p_k", "anchor_mu", "anchor_sig", "anchor_def")
    FIELD_WEIGHT_DECAY = 0.5

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        elif not isinstance(cfg, ModelConfig):
            cfg = ModelConfig.from_dict(cfg)
        self.cfg = cfg
        abl = cfg.ablations
        self.loc = LocEncoder(cfg.loc_freqs, cfg.loc_freq_scale, cfg.loc_freq_max, cfg.loc_seed)
        self.field = (ConstantAnchor() if abl.no_anchor
                      else ClimateField(self.loc.out_dim, cfg.passport_dim, cfg.field_hidden))
        self.passport = Fingerprint(cfg.passport_dim, cfg.passport_hidden, N_DAILY_SUMMARY,
                                    enabled=not abl.no_passport)
        self.encoder = SynopticEncoder(cfg.n_channels, cfg.encoder_width, cfg.encoder_dilations,
                                       cfg.encoder_kernel, cfg.encoder_norm_groups)
        self.readout = LaplaceReadout(cfg.encoder_width, cfg.effective_mode_groups,
                                      cfg.tau_bounds, compression=not abl.no_compression)
        self.propagator = ModalPropagator(cfg.n_modes, cfg.passport_dim, cfg.group_sizes,
                                          cfg.horizon)
        self.heads = Heads(cfg.passport_dim, cfg.n_groups, cfg.n_solar_head, cfg.quantiles,
                           cfg.heads_hidden, cfg.heads_z_proj)
        assert self.heads.in_dim == cfg.heads_in_dim
        self._ch_index = {n: i for i, n in enumerate(cfg.channel_names)}

    def regularization(self, out):
        return mayak_regularizers(out)

    def optim_groups(self, weight_decay):
        """Группы параметров для оптимизатора протокола: [{name, params, weight_decay}]."""
        no_decay, field, rest = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.split(".")[-1] in self.NO_WD_SUFFIX:
                no_decay.append(p)
            elif n.startswith("field."):
                field.append(p)
            else:
                rest.append(p)
        return [dict(name="rest", params=rest, weight_decay=weight_decay),
                dict(name="field", params=field, weight_decay=self.FIELD_WEIGHT_DECAY),
                dict(name="no_decay", params=no_decay, weight_decay=0.0)]

    @staticmethod
    def lag_valid(v, k):
        vs = F.pad(v, (k, 0))[..., :v.shape[-1]]
        return v * vs

    def build_channels(self, x, mask, astro_h, mu_c, sigma_c, defc):
        """Входные каналы энкодера в порядке cfg.channel_names → (ch, aT, vt)."""
        T, P, RH = x[..., 0], x[..., 1], x[..., 2]
        vt, vp, vr = mask[..., 0], mask[..., 1], mask[..., 2]

        aT = ((T - mu_c) / sigma_c).clamp(-8, 8) * vt
        Td = dewpoint_c(T, RH)
        adef = (((T - Td).clamp(min=0.0) - defc) / sigma_c).clamp(-8, 8) * vt * vr

        def dP(p, vpm, k, scale):
            ps = F.pad(p, (k, 0))[..., :p.shape[-1]]
            return (((p - ps) / scale).clamp(-4, 4)) * MAYAK.lag_valid(vpm, k)

        sin_d, cos_d, _, czp, sin_y, cos_y = astro_h
        all_ch = dict(aT=aT, adef=adef, dP3=dP(P, vp, 3, 3.0), dP24=dP(P, vp, 24, 8.0),
                      rh=(RH / 100.0 - 0.5) * vr, sin_d=sin_d, cos_d=cos_d, czp=czp,
                      sin_y=sin_y, cos_y=cos_y, vt=vt, vp=vp, vr=vr)
        ch = torch.stack([all_ch[n] for n in self.cfg.channel_names], dim=1)
        return ch, aT, vt

    def channel(self, ch, name):
        """Канал по имени из тензора (B, n_ch, …) - без жёстких индексов."""
        return ch[:, self._ch_index[name]]

    def channel_indices(self, names):
        return [self._ch_index[n] for n in names if n in self._ch_index]

    def solar_future(self, astro_f):
        """Солнечные ковариаты голов на горизонте: (B, H, n_solar_head)."""
        if self.cfg.n_solar_head == 0:
            return astro_f[0].new_zeros(*astro_f[0].shape, 0)
        return torch.stack([astro_f[0], astro_f[1], astro_f[3]], dim=-1)

    def issue(self, loc, z, a_re, a_im, e, astro_f):
        """Выпуск прогноза из состояния мод и паспорта. Общий для пакета и рантайма."""
        mu_c, sigma_c, _ = self.field.evaluate(self.field.coefficients(loc, z), astro_f)
        tau, omega, _ = self.readout.constants()
        o, Eg = self.propagator(a_re, a_im, z, tau, omega)
        r, ratio, off = self.heads(o, Eg, self.solar_future(astro_f), torch.log(sigma_c), z, e)
        mu = mu_c + sigma_c * (o + r)
        q = mu[..., None] + (sigma_c * ratio)[..., None] * off
        return dict(q=q, mu=mu, sigma_c=sigma_c, o=o, r=r, ratio=ratio, Eg=Eg)

    @staticmethod
    def daily_summaries(aT, adP24, vt, vp24):
        """Суточные сводки по 24-часовым блокам.

        aT, vt   — аномалия T и её маска;
        adP24    — канал разности давления за 24 ч, vp24 — его маска.
        Среднее каждого канала нормируется на число валидных часов своего канала.
        """
        Bsz, Lh = aT.shape
        D = Lh // 24
        a = aT.reshape(Bsz, D, 24)
        v = vt.reshape(Bsz, D, 24)
        p = adP24.reshape(Bsz, D, 24)
        vp = vp24.reshape(Bsz, D, 24)
        n = v.sum(-1)
        mean = (a * v).sum(-1) / n.clamp(min=1.0)
        mx = a.masked_fill(v < 0.5, -1e4).amax(-1)
        mn = a.masked_fill(v < 0.5, 1e4).amin(-1)
        has = (n > 0).float()
        mx, mn = mx * has, mn * has
        mp = (p * vp).sum(-1) / vp.sum(-1).clamp(min=1.0)
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

        summ, day_mask = self.daily_summaries(aT, self.channel(ch, "dP24"), vt,
                                              self.lag_valid(mask[..., 1], 24))
        z, kl = self.passport(loc, summ, day_mask, sample=self.training)

        feats = self.encoder(ch)
        a_re, a_im, e = self.readout(feats, vt)

        out = self.issue(loc, z, a_re, a_im, e, astro_f)
        out.update(a_re=a_re, a_im=a_im, e=e, kl=kl, z=z)
        return out
