import torch
import torch.nn as nn

from mayak.astro import astro_features
from mayak.config import N_DAILY_SUMMARY, SOLAR_CHANNELS, ModelConfig
from mayak.features import dewpoint_deficit, future_channels, history_channels, lag_valid
from mayak.modules.loc import LocEncoder
from mayak.modules.field import ClimateField
from mayak.modules.passport import Fingerprint
from mayak.modules.encoder import SynopticEncoder
from mayak.modules.readout import LaplaceReadout
from mayak.modules.propagator import ModalPropagator
from mayak.modules.heads import Heads
from mayak.loss import mayak_regularizers

DAY_LAG = 24
ATC_SCALE = 5.0
ATC_CLAMP = 8.0
P_MODE_SCALE = 3.0


class MAYAK(nn.Module):
    """МАЯК: климат-поле плюс аномалия из затухающих мод и поправка.

    Аномалия мод погоды и поправка измеряются в единицах климатологического разброса
    точки без паспорта - того же, что нормирует историю, - и переводятся в градусы
    умножением на него. Вклад квазипостоянных мод переводится в градусы постоянным
    масштабом ``P_MODE_SCALE``: устойчивое смещение станции - константа плюс суточная
    составляющая с фиксированным периодом 24 ч; масштаб перевода не зависит ни от часа
    суток, ни от паспорта. Паспорт задаёт ширину интервала через разброс поля с
    паспортом и подстраивает постоянные времени мод и частоты мод погоды. Формула - в
    описании модели.

    Без квазипостоянной группы (абляция ``no_persistent``) слагаемого с
    ``P_MODE_SCALE`` в медиане нет, смещение станции может идти только через моды
    погоды. Без поправки голов (абляция ``no_correction``) поправка равна нулю.

    Вся архитектура задаётся конфигом модели: размеры, группы мод, квантили, флаги
    абляций. Производные размерности - каналы энкодера, вход голов, число групп -
    вычисляются из конфига. Флаги абляций меняют только методы этой модели, поэтому
    любой код, который строит каналы, выпускает прогноз и нормирует моды её методами,
    получает абляцию автоматически.

    Args:
        cfg: конфиг модели; None - значения по умолчанию.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        elif not isinstance(cfg, ModelConfig):
            cfg = ModelConfig.from_dict(cfg)
        self.cfg = cfg
        abl = cfg.ablations
        self.loc = LocEncoder(cfg.loc_freqs, cfg.loc_freq_scale, cfg.loc_freq_max, cfg.loc_seed)
        self.field = ClimateField(self.loc.out_dim, cfg.passport_dim, cfg.field_hidden)
        self.passport = Fingerprint(cfg.passport_dim, cfg.passport_hidden, N_DAILY_SUMMARY,
                                    enabled=not abl.no_passport)
        self.encoder = SynopticEncoder(cfg.n_channels, cfg.encoder_width, cfg.encoder_dilations,
                                       cfg.encoder_kernel, cfg.encoder_norm_groups)
        self.readout = LaplaceReadout(cfg.encoder_width, cfg.effective_mode_groups,
                                      cfg.mode_tau_bounds, compression=not abl.no_compression,
                                      fixed_freq=cfg.persistent_modes)
        self.propagator = ModalPropagator(cfg.n_modes, cfg.passport_dim, cfg.group_sizes,
                                          cfg.horizon, cfg.mode_tau_bounds,
                                          cfg.persistent_modes)
        self.heads = Heads(cfg.passport_dim, cfg.n_groups, cfg.n_solar_head, cfg.quantiles,
                           cfg.heads_hidden, cfg.heads_z_proj, cfg.evidence_modes,
                           correction=not abl.no_correction)
        assert self.heads.in_dim == cfg.heads_in_dim
        self.has_persistent = any(cfg.persistent_modes)
        self._ch_index = {n: i for i, n in enumerate(cfg.channel_names)}

    def regularization(self, out):
        return mayak_regularizers(out)

    def decay_exceptions(self, weight_decay):
        """Отличия от общего правила весового затухания.

        Веса слоёв климат-поля, которые по общему правилу получают затухание, получают
        затухание поля из конфига вместо базового. Исключение - веса FiLM
        (``field.film.*``): через них паспорт модулирует разброс поля, и сильное
        затухание поля гасило бы эту модуляцию. Они идут с базовым затуханием по общему
        правилу, иначе абляция ``no_passport`` сравнивалась бы с моделью, где модуляция
        уже задавлена. Смещения поля, как и все остальные параметры вне правила,
        остаются без затухания.

        Args:
            weight_decay: базовое весовое затухание протокола.

        Returns:
            Список из одной группы ``field``.
        """
        return [dict(name="field", weight_decay=self.cfg.field_weight_decay,
                     match=lambda name, by_rule: (by_rule and name.startswith("field.")
                                                  and not name.startswith("field.film.")))]

    def build_channels(self, x, mask, astro_h, mu_c, sigma_c, defc):
        """Входные каналы энкодера в порядке имён каналов конфига.

        Каналы, не зависящие от поля, берутся из общего построителя признаков. Здесь
        добавляются только аномалии относительно поля: температуры и дефицита точки росы в
        единицах климатологического разброса и температуры в фиксированной нормировке
        ``ATC_SCALE``. В последнем канале постоянное смещение прибора постоянно и не
        промодулировано суточным ходом разброса.

        Args:
            x: наблюдения, форма (B, L, 3).
            mask: маски наличия, форма (B, L, 3).
            astro_h: солнечно-календарные признаки часов истории.
            mu_c: среднее поля на часах истории, форма (B, L).
            sigma_c: климатологический разброс поля, форма (B, L).
            defc: дефицит точки росы по полю, форма (B, L).

        Returns:
            Тройка: каналы формы (B, n_ch, L), аномалия температуры и маска температуры,
            обе формы (B, L).
        """
        shared = history_channels(x, mask, astro_h)
        vt, vr = shared["vt"], shared["vr"]
        aT = ((x[..., 0] - mu_c) / sigma_c).clamp(-8, 8) * vt
        aTc = ((x[..., 0] - mu_c) / ATC_SCALE).clamp(-ATC_CLAMP, ATC_CLAMP) * vt
        adef = ((dewpoint_deficit(x, mask) - defc) / sigma_c).clamp(-8, 8) * vt * vr
        all_ch = dict(shared, aT=aT, aTc=aTc, adef=adef)
        ch = torch.stack([all_ch[n] for n in self.cfg.channel_names], dim=1)
        return ch, aT, vt

    def channel(self, ch, name):
        """Канал по имени, без жёстких номеров.

        Args:
            ch: каналы, форма (B, n_ch, ...).
            name: имя канала.

        Returns:
            Тензор канала, форма (B, ...).
        """
        return ch[:, self._ch_index[name]]

    def solar_future(self, astro_f):
        """Солнечные ковариаты голов на часах горизонта.

        Args:
            astro_f: солнечно-календарные признаки часов горизонта.

        Returns:
            Тензор формы (B, H, n_solar_head); при выключенных солнечных признаках
            последняя ось пустая.
        """
        if self.cfg.n_solar_head == 0:
            return astro_f[0].new_zeros(*astro_f[0].shape, 0)
        fut = future_channels(astro_f)
        return torch.stack([fut[n] for n in SOLAR_CHANNELS], dim=-1)

    def issue(self, loc, z, coefs, a_re, a_im, e, astro_f):
        """Выпуск прогноза из состояния мод и паспорта.

        Поле на часах горизонта считается дважды: без паспорта - его разброс переводит
        аномалию мод погоды и поправку в градусы, как при нормировке истории, - и с
        паспортом - его разброс задаёт ширину интервала. Среднее у обоих одно.

        Args:
            loc: признаки точки, форма (B, loc_dim).
            z: паспорт станции, форма (B, dz).
            coefs: коэффициенты климат-поля точки без паспорта, те же, что нормировали
                историю.
            a_re: действительные амплитуды мод, форма (B, M).
            a_im: мнимые амплитуды мод, форма (B, M).
            e: масса свидетельств по модам, форма (B, M).
            astro_f: солнечно-календарные признаки часов горизонта.

        Returns:
            Словарь: квантили (B, H, число квантилей), медиана, среднее поля ``mu_c``,
            разброс поля с паспортом ``sigma_c`` и без него ``sigma_0``, аномалия мод
            погоды ``o``, вклад квазипостоянных мод ``o_p`` - постоянная и суточная
            составляющие смещения станции, нулевой без квазипостоянной группы, -
            поправка, масштаб интервала и энергии групп мод.
        """
        mu_c, sigma_0, _ = self.field.evaluate(coefs, astro_f)
        _, sigma_z, _ = self.field.evaluate(self.field.coefficients(loc, z), astro_f)
        tau, omega, _ = self.readout.constants()
        o, o_p, Eg = self.propagator(a_re, a_im, z, tau, omega)
        r, ratio, off = self.heads(o, Eg, self.solar_future(astro_f), torch.log(sigma_z), z, e)
        mu = mu_c + P_MODE_SCALE * o_p if self.has_persistent else mu_c
        mu = mu + sigma_0 * (o + r)
        q = mu[..., None] + (sigma_z * ratio)[..., None] * off
        return dict(q=q, mu=mu, mu_c=mu_c, sigma_c=sigma_z, sigma_0=sigma_0, o=o, o_p=o_p,
                    r=r, ratio=ratio, Eg=Eg)

    @staticmethod
    def daily_summaries(aT, adP24, vt, vp24):
        """Суточные сводки по суткам истории.

        Среднее каждого канала нормируется на число валидных часов своего канала.

        Args:
            aT: аномалия температуры, форма (B, L); L кратно 24.
            adP24: изменение давления за сутки, форма (B, L).
            vt: маска температуры, форма (B, L).
            vp24: маска изменения давления, форма (B, L).

        Returns:
            Пара: сводки формы (B, L / 24, 6) - среднее, максимум и минимум аномалии,
            среднее изменение давления, доля валидных часов и признак данных - и маска
            суток с данными формы (B, L / 24).
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

        coefs = self.field.coefficients(loc)
        mu0, sg0, df0 = self.field.evaluate(coefs, astro_h)
        ch, aT, vt = self.build_channels(x, mask, astro_h, mu0, sg0, df0)

        summ, day_mask = self.daily_summaries(aT, self.channel(ch, "dP24"), vt,
                                              lag_valid(mask[..., 1], DAY_LAG))
        z, kl = self.passport(loc, summ, day_mask, sample=self.training)

        feats = self.encoder(ch)
        a_re, a_im, e = self.readout(feats, vt)

        out = self.issue(loc, z, coefs, a_re, a_im, e, astro_f)
        out.update(a_re=a_re, a_im=a_im, e=e, kl=kl, z=z)
        return out
