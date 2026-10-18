"""Конфигурация проекта: модель, данные, обучение."""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

from mayak.constants import H, L_MAX, QUANTILES
from mayak.data.masking import TargetMaskConfig
from mayak.data.splits import ROLE_TEST, ROLES, TIME_LAYOUT
from mayak.metrics import ACIParams, interval_indices
from mayak.protocol import DEFAULT_PROTOCOL, Protocol, Seeds

TrainConfig = Protocol


class ConfigError(ValueError):
    """Недопустимая или несовместимая конфигурация."""


def _floats(v):
    if isinstance(v, (int, float)):
        return (float(v),)
    return tuple(float(x) for x in v)


def _ints(v):
    return tuple(int(x) for x in v)


def _strict_kwargs(cls, d, where):
    if d is None:
        return {}
    if dataclasses.is_dataclass(d):
        d = dataclasses.asdict(d)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ConfigError(f"{where}: неизвестные ключи {unknown}; допустимы {sorted(names)}")
    return dict(d)


def to_jsonable(obj):
    """JSON-совместимая копия датакласса или вложенных кортежей."""
    return json.loads(json.dumps(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
                                 default=lambda o: dataclasses.asdict(o)))


@dataclass(frozen=True)
class ModeGroup:
    """Группа затухающих мод."""
    name: str
    tau0: tuple
    period: tuple = (0.0,)
    tau_bounds: Optional[tuple] = None

    def __post_init__(self):
        tau0, period = _floats(self.tau0), _floats(self.period)
        if not tau0:
            raise ConfigError(f"группа мод {self.name!r}: пустая")
        if len(period) == 1:
            period = period * len(tau0)
        if len(period) != len(tau0):
            raise ConfigError(f"группа мод {self.name!r}: {len(tau0)} постоянных времени, "
                              f"но {len(period)} периодов")
        if any(p < 0 for p in period) or any(t <= 0 for t in tau0):
            raise ConfigError(f"группа мод {self.name!r}: τ₀ > 0 и период ≥ 0")
        if self.tau_bounds is not None:
            bounds = _floats(self.tau_bounds)
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
                raise ConfigError(f"группа мод {self.name!r}: tau_bounds = {bounds}: нужна пара "
                                  f"0 < τ_min < τ_max")
            object.__setattr__(self, "tau_bounds", bounds)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "tau0", tau0)
        object.__setattr__(self, "period", period)

    @property
    def size(self):
        return len(self.tau0)

    def bounds(self, default):
        """Пределы постоянных времени группы."""
        return self.tau_bounds if self.tau_bounds is not None else tuple(default)


PERSISTENT_GROUP = "P"
PERSISTENT_PERIODS = (0.0, 24.0)
DEFAULT_MODE_GROUPS = (
    ModeGroup("R", (3, 6, 12, 24, 48, 96, 168, 240), (0.0,)),
    ModeGroup("D", (12, 24, 48, 96, 168, 240), (24.0,)),
    ModeGroup("S", (12, 24, 72, 168), (12.0,)),
    ModeGroup("W", (24, 48, 72, 120, 168, 240), (60, 84, 108, 132, 156, 192)),
    ModeGroup(PERSISTENT_GROUP, (2000.0, 2000.0), PERSISTENT_PERIODS, (720.0, 8760.0)),
)

ENCODER_CHANNELS = ("aT", "aTc", "adef", "dP3", "dP24", "rh", "sin_d", "cos_d", "czp",
                    "sin_y", "cos_y", "vt", "vp", "vr")
SOLAR_CHANNELS = ("sin_d", "cos_d", "czp")
N_SOLAR_HEAD = 3
N_DAILY_SUMMARY = 6


@dataclass(frozen=True)
class Ablations:
    """Флаги абляций. Меняют только поведение модели, кроме ``no_offset_aug``."""
    no_compression: bool = False
    no_passport: bool = False
    no_solar: bool = False
    no_mode_groups: bool = False
    no_offset_aug: bool = False
    no_correction: bool = False
    no_persistent: bool = False

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, bool):
                raise ConfigError(f"флаг абляции {f.name} должен быть bool, получено {v!r}")

    def active(self):
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


ABLATION_NAMES = tuple(f.name for f in fields(Ablations))


@dataclass(frozen=True)
class ModelConfig:
    """Архитектура МАЯК."""
    arch: str = "mayak"
    horizon: int = H
    max_history: int = L_MAX
    quantiles: tuple = QUANTILES
    mode_groups: tuple = DEFAULT_MODE_GROUPS
    tau_bounds: tuple = (3.0, 240.0)
    passport_dim: int = 16
    passport_hidden: int = 32
    encoder_width: int = 64
    encoder_dilations: tuple = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32)
    encoder_kernel: int = 3
    encoder_norm_groups: int = 4
    loc_freqs: int = 48
    loc_freq_scale: float = 3.0
    loc_freq_max: float = 8.0
    loc_seed: int = 7
    field_hidden: int = 128
    heads_hidden: int = 48
    heads_z_proj: int = 4
    field_weight_decay: float = 0.5
    ablations: Ablations = Ablations()

    def __post_init__(self):
        s = object.__setattr__
        if self.arch != "mayak":
            raise ConfigError(f"ModelConfig описывает МАЯК, arch={self.arch!r}")
        groups = tuple(g if isinstance(g, ModeGroup) else ModeGroup(**_strict_kwargs(
            ModeGroup, g, "model.mode_groups[]")) for g in self.mode_groups)
        s(self, "mode_groups", groups)
        s(self, "quantiles", _floats(self.quantiles))
        s(self, "tau_bounds", _floats(self.tau_bounds))
        s(self, "encoder_dilations", _ints(self.encoder_dilations))
        if not isinstance(self.ablations, Ablations):
            s(self, "ablations", Ablations(**_strict_kwargs(Ablations, self.ablations,
                                                            "model.ablations")))
        for name in ("horizon", "max_history", "passport_dim", "passport_hidden", "encoder_width",
                     "encoder_kernel", "encoder_norm_groups", "loc_freqs", "field_hidden",
                     "heads_hidden", "heads_z_proj", "loc_seed"):
            s(self, name, int(getattr(self, name)))
        for name in ("loc_freq_scale", "loc_freq_max", "field_weight_decay"):
            s(self, name, float(getattr(self, name)))
        self._validate()

    def _validate(self):
        q = self.quantiles
        if any(not 0.0 < x < 1.0 for x in q) or any(b <= a for a, b in zip(q, q[1:])):
            raise ConfigError(f"квантили должны строго возрастать внутри (0, 1): {q}")
        if 0.5 not in q:
            raise ConfigError(f"в наборе квантилей нет медианы 0.5: {q}")
        if self.horizon < 1:
            raise ConfigError("horizon < 1")
        if self.max_history < 24 or self.max_history % 24:
            raise ConfigError(f"max_history = {self.max_history}: нужно кратное 24 и ≥ 24 "
                              f"(суточные сводки паспорта)")
        lo, hi = self.tau_bounds if len(self.tau_bounds) == 2 else (None, None)
        if lo is None or not 0 < lo < hi:
            raise ConfigError(f"tau_bounds = {self.tau_bounds}: нужна пара 0 < τ_min < τ_max")
        for g in self.mode_groups:
            g_lo, g_hi = g.bounds(self.tau_bounds)
            bad = [t for t in g.tau0 if not g_lo <= t <= g_hi]
            if bad:
                raise ConfigError(f"группа мод {g.name!r}: τ₀ {bad} вне [{g_lo}, {g_hi}]")
        names = [g.name for g in self.mode_groups]
        if not names or len(set(names)) != len(names):
            raise ConfigError(f"имена групп мод пусты или повторяются: {names}")
        for g in self.mode_groups:
            if g.name == PERSISTENT_GROUP and not set(g.period) <= set(PERSISTENT_PERIODS):
                raise ConfigError(f"группа мод {g.name!r} квазипостоянная: периоды её мод "
                                  f"только из {PERSISTENT_PERIODS}, получено {g.period}")
        if names == [PERSISTENT_GROUP]:
            raise ConfigError(f"кроме квазипостоянной группы {PERSISTENT_GROUP!r} нужна хотя "
                              f"бы одна группа мод: по ней считается масса свидетельств голов")
        if self.ablations.no_persistent and PERSISTENT_GROUP not in names:
            raise ConfigError(f"абляция no_persistent: в mode_groups нет квазипостоянной "
                              f"группы {PERSISTENT_GROUP!r}, убирать нечего")
        if self.ablations.no_mode_groups:
            own = [g.name for g in self.mode_groups
                   if g.name != PERSISTENT_GROUP and g.tau_bounds is not None]
            if own:
                raise ConfigError(f"абляция no_mode_groups: у групп погоды {own} собственные "
                                  f"tau_bounds; слитая группа погоды получает одну пару пределов "
                                  f"- общие tau_bounds модели, - поэтому собственные пределы "
                                  f"групп потерялись бы молча. Уберите их (null) или перенесите "
                                  f"в общие tau_bounds")
        if not self.encoder_dilations or min(self.encoder_dilations) < 1:
            raise ConfigError("дилатации энкодера должны быть ≥ 1")
        if self.encoder_kernel < 2:
            raise ConfigError("ядро энкодера < 2")
        if self.encoder_width % self.encoder_norm_groups:
            raise ConfigError(f"ширина энкодера {self.encoder_width} не делится на число "
                              f"групп нормализации {self.encoder_norm_groups}")
        for name in ("passport_dim", "passport_hidden", "encoder_width", "loc_freqs",
                     "field_hidden", "heads_hidden", "heads_z_proj"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} < 1")
        if not (math.isfinite(self.field_weight_decay) and self.field_weight_decay >= 0.0):
            raise ConfigError(f"field_weight_decay = {self.field_weight_decay}: нужно конечное "
                              f"число не меньше нуля")

    @property
    def effective_mode_groups(self):
        """Группы мод с учётом абляций ``no_persistent`` и ``no_mode_groups``."""
        groups = self.mode_groups
        if self.ablations.no_persistent:
            groups = tuple(g for g in groups if g.name != PERSISTENT_GROUP)
        if not self.ablations.no_mode_groups:
            return groups
        weather = tuple(g for g in groups if g.name != PERSISTENT_GROUP)
        kept = tuple(g for g in groups if g.name == PERSISTENT_GROUP)
        merged = ModeGroup("all", tuple(t for g in weather for t in g.tau0),
                           tuple(p for g in weather for p in g.period))
        return (merged, *kept)

    @property
    def mode_tau_bounds(self):
        """Пределы постоянной времени каждой моды, ч, в порядке мод модели."""
        return tuple(g.bounds(self.tau_bounds) for g in self.effective_mode_groups
                     for _ in range(g.size))

    @property
    def persistent_modes(self):
        """Моды квазипостоянной группы."""
        return tuple(g.name == PERSISTENT_GROUP for g in self.effective_mode_groups
                     for _ in range(g.size))

    @property
    def evidence_modes(self):
        """Моды, по которым головы считают среднюю массу свидетельств."""
        return tuple(not f for f in self.persistent_modes)

    @property
    def n_modes(self):
        """Число мод модели с учётом абляций."""
        return sum(g.size for g in self.effective_mode_groups)

    @property
    def group_sizes(self):
        return tuple(g.size for g in self.effective_mode_groups)

    @property
    def n_groups(self):
        return len(self.effective_mode_groups)

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    @property
    def channel_names(self):
        if self.ablations.no_solar:
            return tuple(c for c in ENCODER_CHANNELS if c not in SOLAR_CHANNELS)
        return ENCODER_CHANNELS

    @property
    def n_channels(self):
        return len(self.channel_names)

    @property
    def n_solar_head(self):
        return 0 if self.ablations.no_solar else N_SOLAR_HEAD

    @property
    def heads_in_dim(self):
        """Ширина входа голов."""
        return 1 + self.n_groups + 1 + self.n_solar_head + 1 + self.heads_z_proj + 1 + 1

    @property
    def device_window(self):
        """Длина сырого окна наблюдений, которое устройство хранит на диске, часы."""
        from mayak.data.qc import DEFAULT_QC
        need = max(self.max_history, DEFAULT_QC.lookback_hours + 1)
        return 8 * math.ceil(need / 8)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


@dataclass(frozen=True)
class GRUConfig:
    """Бейзлайн GRU."""
    arch: str = "gru"
    horizon: int = H
    quantiles: tuple = QUANTILES
    hidden: int = 96
    layers: int = 2
    head_hidden: int = 128

    def __post_init__(self):
        if self.arch != "gru":
            raise ConfigError(f"GRUConfig: arch={self.arch!r}")
        object.__setattr__(self, "quantiles", _floats(self.quantiles))
        for name in ("horizon", "hidden", "layers", "head_hidden"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"GRUConfig.{name} < 1")
            object.__setattr__(self, name, v)

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


@dataclass(frozen=True)
class DLinearConfig:
    """Бейзлайн DLinear."""
    arch: str = "dlinear"
    horizon: int = H
    quantiles: tuple = QUANTILES
    input_len: int = 336
    kernel: int = 25

    def __post_init__(self):
        if self.arch != "dlinear":
            raise ConfigError(f"DLinearConfig: arch={self.arch!r}")
        object.__setattr__(self, "quantiles", _floats(self.quantiles))
        if self.kernel < 1 or self.kernel % 2 == 0:
            raise ConfigError(f"ядро скользящего среднего DLinear должно быть нечётным: "
                              f"{self.kernel}")

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


@dataclass(frozen=True)
class LRUConfig:
    """Бейзлайн с линейной рекуррентной памятью без разложения на якорь и аномалию."""
    arch: str = "lru"
    horizon: int = H
    quantiles: tuple = QUANTILES
    max_history: int = L_MAX
    d_model: int = 64
    d_state: int = 64
    layers: int = 3
    dropout: float = 0.0
    tau_bounds: tuple = (3.0, 240.0)
    min_period: float = 12.0
    head_hidden: int = 128

    def __post_init__(self):
        s = object.__setattr__
        if self.arch != "lru":
            raise ConfigError(f"LRUConfig: arch={self.arch!r}")
        s(self, "quantiles", _floats(self.quantiles))
        s(self, "tau_bounds", _floats(self.tau_bounds))
        for name in ("horizon", "max_history", "d_model", "d_state", "layers", "head_hidden"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"LRUConfig.{name} < 1")
            s(self, name, v)
        s(self, "dropout", float(self.dropout))
        s(self, "min_period", float(self.min_period))
        if len(self.tau_bounds) != 2 or not 0 < self.tau_bounds[0] < self.tau_bounds[1]:
            raise ConfigError(f"LRUConfig.tau_bounds = {self.tau_bounds}: нужна пара "
                              f"0 < τ_min < τ_max")
        if self.min_period <= 2.0:
            raise ConfigError(f"LRUConfig.min_period = {self.min_period}: период ≤ 2 ч "
                              f"неразличим на часовой сетке")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"LRUConfig.dropout = {self.dropout} вне [0, 1)")

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    @property
    def r_min(self):
        return math.exp(-1.0 / self.tau_bounds[0])

    @property
    def r_max(self):
        return math.exp(-1.0 / self.tau_bounds[1])

    @property
    def max_phase(self):
        return 2.0 * math.pi / self.min_period

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


@dataclass(frozen=True)
class PatchTSTConfig:
    """Бейзлайн PatchTST, малая конфигурация."""
    arch: str = "patchtst"
    horizon: int = H
    quantiles: tuple = QUANTILES
    input_len: int = 504
    patch_len: int = 24
    stride: int = 24
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    layers: int = 2
    dropout: float = 0.2
    attn_dropout: float = 0.0
    head_dropout: float = 0.0

    def __post_init__(self):
        s = object.__setattr__
        if self.arch != "patchtst":
            raise ConfigError(f"PatchTSTConfig: arch={self.arch!r}")
        s(self, "quantiles", _floats(self.quantiles))
        for name in ("horizon", "input_len", "patch_len", "stride", "d_model", "n_heads", "d_ff",
                     "layers"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"PatchTSTConfig.{name} < 1")
            s(self, name, v)
        for name in ("dropout", "attn_dropout", "head_dropout"):
            v = float(getattr(self, name))
            if not 0.0 <= v < 1.0:
                raise ConfigError(f"PatchTSTConfig.{name} = {v} вне [0, 1)")
            s(self, name, v)
        if self.patch_len > self.input_len:
            raise ConfigError(f"патч {self.patch_len} ч длиннее входа {self.input_len} ч")
        if (self.input_len - self.patch_len) % self.stride:
            raise ConfigError(f"вход {self.input_len} ч не делится на патчи {self.patch_len} ч "
                              f"с шагом {self.stride} ч: последние часы окна не попали бы "
                              f"ни в один патч")
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model {self.d_model} не делится на n_heads {self.n_heads}")

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    @property
    def n_patches(self):
        """Число патчей окна."""
        return (self.input_len - self.patch_len) // self.stride + 1

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


MODEL_CONFIGS = {"mayak": ModelConfig, "gru": GRUConfig, "dlinear": DLinearConfig,
                 "lru": LRUConfig, "patchtst": PatchTSTConfig}


def model_config_for(arch, cfg=None):
    """Конфиг архитектуры в виде датакласса."""
    if arch not in MODEL_CONFIGS:
        raise ConfigError(f"неизвестная архитектура {arch!r}; есть {tuple(MODEL_CONFIGS)}")
    cls = MODEL_CONFIGS[arch]
    if cfg is None:
        return cls()
    if isinstance(cfg, cls):
        return cfg
    if dataclasses.is_dataclass(cfg):
        raise ConfigError(f"конфиг {type(cfg).__name__} не подходит архитектуре {arch!r}")
    d = dict(cfg)
    if d.setdefault("arch", arch) != arch:
        raise ConfigError(f"конфиг модели для {d['arch']!r}, а архитектура {arch!r}")
    return cls.from_dict(d)


def model_config_from_dict(d):
    """Конфиг архитектуры по словарю."""
    d = dict(d or {})
    return model_config_for(d.get("arch", "mayak"), d)


def check_pipeline_compat(cfg):
    """Проверка, что модель совместима с контрактом данных."""
    errs = []
    if cfg.horizon != H:
        errs.append(f"horizon={cfg.horizon}, конвейер данных - {H}")
    if tuple(cfg.quantiles) != tuple(float(q) for q in QUANTILES):
        errs.append(f"quantiles={cfg.quantiles}, модуль метрик - {QUANTILES}")
    if hasattr(cfg, "max_history") and cfg.max_history != L_MAX:
        errs.append(f"длина истории {cfg.max_history}, окна датасетов - {L_MAX}")
    if hasattr(cfg, "input_len") and not 1 <= cfg.input_len <= L_MAX:
        errs.append(f"длина входа {cfg.input_len} вне [1, {L_MAX}] - окна датасетов")
    if errs:
        raise ConfigError("конфиг модели несовместим с контрактом данных: "
                          + "; ".join(errs) + ". Модель это поддерживает, но сплиты, кэш и "
                                              "метрики построены под контракт - меняйте его там.")
    return cfg


def _pair(v, name, cast=float):
    v = tuple(cast(x) for x in v)
    if len(v) != 2 or v[0] > v[1]:
        raise ConfigError(f"{name} = {v}: нужна пара lo ≤ hi")
    return v


AUGMENT_PROB_FIELDS = {
    "offset": "offset_prob", "noise": "noise_prob", "rh_dewpoint": "rh_dewpoint_prob",
    "stuck": "stuck_prob", "dropout": "dropout_prob", "gap": "gap_prob",
    "drop_channel": "drop_channel_prob",
}


@dataclass(frozen=True)
class AugmentConfig:
    """Аугментации обучающих окон, имитирующие реальный прибор."""
    profile: str = "default"
    offset_prob: float = 0.4
    offset_min: float = 0.1
    offset_max: float = 2.0
    offset_diurnal_frac: float = 0.5
    offset_diurnal_max: float = 1.5
    offset_peak_hours: tuple = (13.0, 16.0)
    noise_prob: float = 1.0
    noise_sd: tuple = (0.2, 0.3, 2.0)
    rh_dewpoint_prob: float = 0.1
    stuck_prob: float = 0.15
    stuck_hours: tuple = (6, 48)
    dropout_prob: float = 0.3
    dropout_max_rate: float = 0.2
    gap_prob: float = 0.5
    gap_max_count: int = 3
    gap_max_len: int = 96
    drop_channel_prob: float = 0.28
    drop_channel_weights: tuple = (0.46, 0.46, 0.08)

    def __post_init__(self):
        s = object.__setattr__
        if self.profile not in AUGMENT_PROFILES:
            raise ConfigError(f"неизвестный профиль аугментаций {self.profile!r}; "
                              f"есть {tuple(AUGMENT_PROFILES)}")
        v = _floats(self.noise_sd)
        if len(v) != 3 or min(v) < 0:
            raise ConfigError(f"noise_sd - по одному неотрицательному значению на канал "
                              f"T, P, RH: {v}")
        s(self, "noise_sd", v)
        v = _floats(self.drop_channel_weights)
        if len(v) != 3 or min(v) < 0 or sum(v) <= 0:
            raise ConfigError(f"drop_channel_weights - три неотрицательные доли вариантов "
                              f"без P, без RH, без обоих с положительной суммой: {v}")
        s(self, "drop_channel_weights", v)
        v = _pair(self.stuck_hours, "stuck_hours", int)
        if v[0] < 1:
            raise ConfigError("stuck_hours: длительность ≥ 1 ч")
        s(self, "stuck_hours", v)
        v = _pair(self.offset_peak_hours, "offset_peak_hours")
        if v[0] < 0.0 or v[1] > 24.0:
            raise ConfigError(f"offset_peak_hours = {v}: часы суток в [0, 24]")
        s(self, "offset_peak_hours", v)
        for name in (*AUGMENT_PROB_FIELDS.values(), "offset_diurnal_frac", "dropout_max_rate"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ConfigError(f"{name} = {v} вне [0, 1]")
            s(self, name, v)
        for name in ("gap_max_count", "gap_max_len"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"{name} ≥ 1")
            s(self, name, v)
        for name in ("offset_min", "offset_max", "offset_diurnal_max"):
            v = float(getattr(self, name))
            if v < 0:
                raise ConfigError(f"{name} ≥ 0")
            s(self, name, v)
        if self.offset_max == 0.0:
            s(self, "offset_min", 0.0)
        elif self.offset_min > self.offset_max:
            raise ConfigError(f"offset_min {self.offset_min} > offset_max {self.offset_max}")

    @classmethod
    def from_profile(cls, name="default", **overrides):
        """Конфиг из именованного профиля с явными переопределениями."""
        if name not in AUGMENT_PROFILES:
            raise ConfigError(f"неизвестный профиль аугментаций {name!r}; "
                              f"есть {tuple(AUGMENT_PROFILES)}")
        kw = {**AUGMENT_PROFILES[name], **overrides}
        kw.pop("profile", None)
        return cls(**_strict_kwargs(cls, {"profile": name, **kw}, "data.augment"))

    @classmethod
    def from_dict(cls, d=None):
        """Конфиг из словаря: сначала профиль, потом остальные ключи поверх него."""
        d = _strict_kwargs(cls, d, "data.augment")
        return cls.from_profile(d.pop("profile", "default"), **d)

    def deviations(self):
        """Поля, которые отличаются от объявленного профиля."""
        ref = AugmentConfig.from_profile(self.profile)
        return {f.name: (getattr(ref, f.name), getattr(self, f.name)) for f in fields(self)
                if getattr(ref, f.name) != getattr(self, f.name)}

    def summary(self):
        return dict(profile=self.profile,
                    deviations={k: [to_jsonable(a), to_jsonable(b)]
                                for k, (a, b) in self.deviations().items()},
                    enabled=sorted(n for n, f in AUGMENT_PROB_FIELDS.items()
                                   if getattr(self, f) > 0))


AUGMENT_PROFILES = {
    "default": {},
    "none": {f: 0.0 for f in AUGMENT_PROB_FIELDS.values()},
}

@dataclass(frozen=True)
class DataConfig:
    """Данные прогона: источник, правила окон, аугментации, набор валидации."""
    manifest: str = "data/manifest.csv"
    cache_root: Optional[str] = None
    time_layout: dict = field(default_factory=lambda: dict(TIME_LAYOUT))
    target_mask: TargetMaskConfig = TargetMaskConfig()
    val_every_hours: int = 24
    val_windows_per_station: int = 96
    val_seed: int = 0
    augment: AugmentConfig = AugmentConfig()
    window_qc: bool = True

    def __post_init__(self):
        if not isinstance(self.target_mask, TargetMaskConfig):
            object.__setattr__(self, "target_mask", TargetMaskConfig(**_strict_kwargs(
                TargetMaskConfig, self.target_mask, "data.target_mask")))
        if not isinstance(self.augment, AugmentConfig):
            object.__setattr__(self, "augment", AugmentConfig.from_dict(self.augment))
        tl = dict(self.time_layout)
        if tl != dict(TIME_LAYOUT):
            raise ConfigError(f"data.time_layout = {tl} расходится с контрактом сплитов "
                              f"{dict(TIME_LAYOUT)}: раскладка входит в ключ кэша и "
                              f"задаётся только в коде")
        object.__setattr__(self, "time_layout", tl)
        if self.val_every_hours < 1 or self.val_windows_per_station < 1:
            raise ConfigError("val_every_hours и val_windows_per_station должны быть не меньше 1")
        if self.val_seed < 0:
            raise ConfigError(f"val_seed = {self.val_seed}: сид не может быть отрицательным")
        if not isinstance(self.window_qc, bool):
            raise ConfigError(f"window_qc должен быть bool, получено {self.window_qc!r}")

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        """Конфиг данных из словаря."""
        return cls(**_strict_kwargs(cls, d, "data"))


@dataclass(frozen=True)
class RunConfig:
    """Полная конфигурация прогона: модель, данные и протокол обучения."""
    model: Any = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: Protocol = DEFAULT_PROTOCOL

    def __post_init__(self):
        if not dataclasses.is_dataclass(self.model):
            object.__setattr__(self, "model", model_config_from_dict(self.model))
        if not isinstance(self.data, DataConfig):
            object.__setattr__(self, "data", DataConfig.from_dict(self.data))
        if not isinstance(self.train, Protocol):
            object.__setattr__(self, "train", Protocol.from_dict(dict(self.train)))

    @property
    def arch(self):
        return self.model.arch

    def resolved(self):
        """Конфигурация, где эффекты абляций перенесены между секциями."""
        abl = getattr(self.model, "ablations", None)
        data = self.data
        aug = data.augment
        if abl is not None and abl.no_offset_aug and (aug.offset_max or aug.offset_diurnal_max):
            data = replace(data, augment=replace(aug, offset_min=0.0, offset_max=0.0,
                                                 offset_diurnal_max=0.0))
        return replace(self, data=data)

    def to_dict(self):
        return dict(model=self.model.to_dict(), data=self.data.to_dict(),
                    train=self.train.to_dict())

    @classmethod
    def from_dict(cls, d):
        d = _strict_kwargs(cls, d, "run")
        return cls(**d)


def run_label(model_cfg):
    """Короткое имя варианта модели для таблиц."""
    abl = getattr(model_cfg, "ablations", None)
    act = abl.active() if abl is not None else ()
    return model_cfg.arch + ("" if not act else "-" + "+".join(act))


SCENARIO_INSTRUMENT, SCENARIO_INPUT = "instrument", "input"
ROBUSTNESS_QC = ("none", "device")
ROBUSTNESS_TIME_KEYS = ("val", "test")


@dataclass(frozen=True)
class ScenarioRule:
    """Контракт сценария робастности."""
    kind: str
    target: bool
    title: str
    unit: str
    max_level: Optional[float] = None
    integer: bool = False
    params: dict = field(default_factory=dict)
    guard: bool = True
    variant_of: Optional[str] = None


SCENARIO_RULES = {
    "dropout": ScenarioRule(SCENARIO_INPUT, False, "случайные пропуски истории",
                            "доля потерянных часов", max_level=1.0),
    "gap": ScenarioRule(SCENARIO_INPUT, False, "блочный пропуск последних часов",
                        "ч без данных перед выпуском", max_level=float(L_MAX), integer=True),
    "noise": ScenarioRule(SCENARIO_INPUT, False, "шум измерений", "σ шума T, °C",
                          params=dict(sd_ratio=(1.0, 1.5, 10.0))),
    "spikes": ScenarioRule(SCENARIO_INPUT, False, "одиночные выбросы",
                           "доля часов с выбросом", max_level=1.0,
                           params=dict(magnitude=(20.0, 25.0, 60.0))),
    "freeze": ScenarioRule(SCENARIO_INPUT, False, "замерзание датчика",
                           "ч повторения одного значения", max_level=float(L_MAX),
                           integer=True, params=dict(channels=(0, 1, 2))),
    "drop_channel": ScenarioRule(SCENARIO_INPUT, False, "отсутствие канала",
                                 "0 - все, 1 - без P, 2 - без RH, 3 - без P и RH",
                                 max_level=3.0, integer=True),
    "history": ScenarioRule(SCENARIO_INPUT, False, "сокращение истории",
                            "убрано ч истории (672 - холодный старт)",
                            max_level=float(L_MAX), integer=True),
    "coords": ScenarioRule(SCENARIO_INPUT, False, "ошибка координат", "градусы"),
    "elev": ScenarioRule(SCENARIO_INPUT, False, "ошибка высоты", "м"),
    "offset": ScenarioRule(SCENARIO_INSTRUMENT, True, "постоянное смещение T", "°C"),
    "drift": ScenarioRule(SCENARIO_INSTRUMENT, True, "медленный дрейф T", "°C/сутки"),
    "scale": ScenarioRule(SCENARIO_INSTRUMENT, True, "ошибка масштаба T", "|k − 1|"),
    "offset_input": ScenarioRule(SCENARIO_INSTRUMENT, False,
                                 "незамеченное смещение прибора (только вход)", "°C",
                                 guard=False, variant_of="offset"),
    "drift_input": ScenarioRule(SCENARIO_INSTRUMENT, False,
                                "незамеченный дрейф прибора (только вход)", "°C/сутки",
                                guard=False, variant_of="drift"),
}

DEFAULT_SCENARIOS = (
    dict(name="dropout", levels=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0)),
    dict(name="gap", levels=(0, 3, 6, 12, 24, 48, 96, 168, 336, 672)),
    dict(name="offset", levels=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)),
    dict(name="offset_input", levels=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)),
    dict(name="drift", levels=(0.0, 0.02, 0.05, 0.1, 0.2)),
    dict(name="drift_input", levels=(0.0, 0.02, 0.05, 0.1, 0.2)),
    dict(name="scale", levels=(0.0, 0.01, 0.02, 0.05, 0.1)),
    dict(name="noise", levels=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)),
    dict(name="spikes", levels=(0.0, 0.005, 0.01, 0.03, 0.1)),
    dict(name="freeze", levels=(0, 6, 24, 72, 168, 336, 672)),
    dict(name="drop_channel", levels=(0, 1, 2, 3)),
    dict(name="history", levels=(0, 168, 336, 504, 600, 648, 666, 672)),
    dict(name="coords", levels=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)),
    dict(name="elev", levels=(0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)),
)


def _norm_param(where, key, default, value):
    """Значение параметра сценария, приведённое к типу и длине значения по умолчанию."""
    if isinstance(default, tuple):
        try:
            v = tuple(value)
        except TypeError as err:
            raise ConfigError(
                f"{where}.{key}: ожидался список из {len(default)} значений") from err
        if len(v) != len(default) and not (key == "channels" and 1 <= len(v) <= 3):
            raise ConfigError(f"{where}.{key}: {len(v)} значений, нужно {len(default)}")
        cast = int if all(isinstance(d, int) for d in default) else float
        return tuple(cast(x) for x in v)
    return type(default)(value)


@dataclass(frozen=True)
class ScenarioSpec:
    """Один сценарий робастности из конфига."""
    name: str
    levels: tuple
    params: dict = field(default_factory=dict)
    guard: Optional[bool] = None

    def __post_init__(self):
        s = object.__setattr__
        where = f"robustness.scenarios[{self.name}]"
        rule = SCENARIO_RULES.get(self.name)
        if rule is None:
            raise ConfigError(f"неизвестный сценарий {self.name!r}; есть {tuple(SCENARIO_RULES)}")
        lv = tuple(float(v) for v in _floats(self.levels))
        if not lv:
            raise ConfigError(f"{where}: пустой список уровней")
        if lv[0] != 0.0:
            raise ConfigError(f"{where}: первый уровень должен быть 0 (точка отсчёта без "
                              f"деградации), получено {lv[0]}")
        if any(b <= a for a, b in zip(lv, lv[1:])):
            raise ConfigError(f"{where}: уровни должны строго возрастать: {lv}")
        if rule.max_level is not None and lv[-1] > rule.max_level:
            raise ConfigError(f"{where}: уровень {lv[-1]} больше допустимого {rule.max_level}")
        if rule.integer and not all(v.is_integer() for v in lv):
            raise ConfigError(f"{where}: уровни - целые числа, получено {lv}")
        s(self, "levels", lv)
        params = dict(self.params or {})
        unknown = sorted(set(params) - set(rule.params))
        if unknown:
            raise ConfigError(f"{where}: неизвестные параметры {unknown}; "
                              f"допустимы {sorted(rule.params)}")
        s(self, "params", {k: _norm_param(where, k, d, params.get(k, d))
                           for k, d in rule.params.items()})
        if "channels" in self.params and not set(self.params["channels"]) <= {0, 1, 2}:
            raise ConfigError(f"{where}.channels: каналы 0 (T), 1 (P), 2 (RH)")
        s(self, "guard", rule.guard if self.guard is None else bool(self.guard))

    @property
    def rule(self):
        return SCENARIO_RULES[self.name]

    @classmethod
    def from_dict(cls, d):
        return d if isinstance(d, cls) else cls(**_strict_kwargs(cls, d, "robustness.scenarios"))


def default_scenarios():
    return tuple(ScenarioSpec(**d) for d in DEFAULT_SCENARIOS)


@dataclass(frozen=True)
class RobustnessConfig:
    """Прогон робастности обученной модели без переобучения."""
    roles: tuple = (ROLE_TEST,)
    time_key: str = "test"
    every_hours: int = 72
    windows_per_station: int = 20
    qc: str = "device"
    leads: tuple = (1, 6, 24, 72, 168)
    skill_tolerance: float = 0.05
    guard_models: tuple = ("МАЯК",)
    seed: int = 0
    bootstrap: int = 1000
    ci_level: float = 0.90
    scenarios: tuple = field(default_factory=default_scenarios)

    def __post_init__(self):
        s = object.__setattr__
        roles = tuple(str(r) for r in ([self.roles] if isinstance(self.roles, str)
                                       else self.roles))
        if not roles or not set(roles) <= set(ROLES):
            raise ConfigError(f"robustness.roles = {roles}; допустимы {ROLES} "
                              f"(внешний тест - отдельным манифестом)")
        s(self, "roles", roles)
        if self.time_key not in ROBUSTNESS_TIME_KEYS:
            raise ConfigError(f"robustness.time_key = {self.time_key!r}; "
                              f"допустимо {ROBUSTNESS_TIME_KEYS}")
        if self.qc not in ROBUSTNESS_QC:
            raise ConfigError(f"robustness.qc = {self.qc!r}; допустимо {ROBUSTNESS_QC}")
        leads = tuple(sorted(set(_ints(self.leads))))
        if not leads or leads[0] < 1 or leads[-1] > H:
            raise ConfigError(f"robustness.leads = {leads}: лиды в [1, {H}]")
        s(self, "leads", leads)
        for name in ("every_hours", "windows_per_station", "seed", "bootstrap"):
            s(self, name, int(getattr(self, name)))
        if self.every_hours < 1 or self.windows_per_station < 1 or self.bootstrap < 0:
            raise ConfigError("every_hours, windows_per_station ≥ 1; bootstrap ≥ 0")
        tol = float(self.skill_tolerance)
        if not 0.0 <= tol < 1.0:
            raise ConfigError(f"robustness.skill_tolerance = {tol} вне [0, 1)")
        s(self, "skill_tolerance", tol)
        lvl = float(self.ci_level)
        if not 0.0 < lvl < 1.0:
            raise ConfigError(f"robustness.ci_level = {lvl} вне (0, 1)")
        s(self, "ci_level", lvl)
        gm = tuple(str(m) for m in ([self.guard_models] if isinstance(self.guard_models, str)
                                    else self.guard_models))
        if not gm:
            raise ConfigError("robustness.guard_models пуст: проверке скилла нечего проверять")
        s(self, "guard_models", gm)
        sc = tuple(ScenarioSpec.from_dict(d) for d in self.scenarios)
        if not sc:
            raise ConfigError("robustness.scenarios пуст")
        names = [x.name for x in sc]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ConfigError(f"robustness.scenarios: повторяются {dup}")
        s(self, "scenarios", sc)

    def scenario(self, name):
        for sc in self.scenarios:
            if sc.name == name:
                return sc
        raise KeyError(name)

    def select(self, names):
        """Конфиг только с выбранными сценариями."""
        names = list(names)
        have = {sc.name for sc in self.scenarios}
        missing = sorted(set(names) - have)
        if missing:
            raise ConfigError(f"в конфиге нет сценариев {missing}; есть {sorted(have)}")
        return replace(self, scenarios=tuple(sc for sc in self.scenarios if sc.name in names))

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "robustness"))


COVERAGE_DIMS_INTERNAL = ("зона Кёппена", "длина истории", "валидность истории")
COVERAGE_DIMS_EXTERNAL = ("зона Кёппена", "длина истории", "валидность истории",
                          "расстояние до обучающей точки", "Δ высоты станция−ЦМР",
                          "канал давления")
COVERAGE_DIMS = tuple(dict.fromkeys(COVERAGE_DIMS_INTERNAL + COVERAGE_DIMS_EXTERNAL))
CALIBRATION_NOMINALS = (0.8, 0.9)


@dataclass(frozen=True)
class CalibrationConfig:
    """Подгонка конформной таблицы, анализ калибровки и адаптивная калибровка на устройстве."""
    nominal: float = 0.90
    tolerance: float = 0.04
    min_windows: int = 20
    min_stations: int = 2
    bootstrap: int = 1000
    ci_level: float = 0.90
    seed: int = 0
    conditional_dims: tuple = ("зона Кёппена", "длина истории")
    sharpness_range: tuple = (0.25, 4.0)
    sharpness_points: int = 33
    aci_gamma: float = 0.005
    aci_max_factor: float = 4.0
    aci_stations: int = 8
    aci_hours: int = 720
    fit_every_hours: int = 24
    fit_windows_per_station: Optional[int] = None
    fit_seed: int = 0
    fit_min_windows: int = 200

    def __post_init__(self):
        s = object.__setattr__
        for name in ("fit_every_hours", "fit_seed", "fit_min_windows", "aci_stations",
                     "aci_hours"):
            s(self, name, int(getattr(self, name)))
        if self.fit_every_hours < 1 or self.fit_seed < 0 or self.fit_min_windows < 1:
            raise ConfigError("calibration.fit_every_hours ≥ 1, calibration.fit_seed ≥ 0 и "
                              "calibration.fit_min_windows ≥ 1")
        if self.aci_stations < 1 or self.aci_hours < 1:
            raise ConfigError("calibration.aci_stations ≥ 1 и calibration.aci_hours ≥ 1")
        if self.fit_windows_per_station is not None:
            s(self, "fit_windows_per_station", int(self.fit_windows_per_station))
            if self.fit_windows_per_station < 1:
                raise ConfigError("calibration.fit_windows_per_station: None (все окна) или ≥ 1")
        nominal = float(self.nominal)
        if not any(abs(nominal - n) < 1e-9 for n in CALIBRATION_NOMINALS):
            raise ConfigError(f"calibration.nominal = {nominal}; допустимо {CALIBRATION_NOMINALS}")
        interval_indices(nominal)
        s(self, "nominal", nominal)
        tol = float(self.tolerance)
        if not 0.0 < tol < 0.5:
            raise ConfigError(f"calibration.tolerance = {tol} вне (0, 0.5)")
        s(self, "tolerance", tol)
        for name in ("min_windows", "min_stations", "bootstrap", "seed", "sharpness_points"):
            s(self, name, int(getattr(self, name)))
        if self.min_windows < 1 or self.min_stations < 1 or self.bootstrap < 0:
            raise ConfigError("min_windows, min_stations ≥ 1; bootstrap ≥ 0")
        lvl = float(self.ci_level)
        if not 0.0 < lvl < 1.0:
            raise ConfigError(f"calibration.ci_level = {lvl} вне (0, 1)")
        s(self, "ci_level", lvl)
        dims = tuple(str(d) for d in ([self.conditional_dims]
                                      if isinstance(self.conditional_dims, str)
                                      else self.conditional_dims))
        unknown = sorted(set(dims) - set(COVERAGE_DIMS))
        if unknown:
            raise ConfigError(f"calibration.conditional_dims: неизвестные разрезы {unknown}; "
                              f"есть {list(COVERAGE_DIMS)}")
        s(self, "conditional_dims", dims)
        rng = _floats(self.sharpness_range)
        if len(rng) != 2 or not 0.0 < rng[0] < 1.0 < rng[1]:
            raise ConfigError(f"calibration.sharpness_range = {rng}: нужно (a, b), 0 < a < 1 < b")
        s(self, "sharpness_range", rng)
        if self.sharpness_points < 3:
            raise ConfigError("calibration.sharpness_points ≥ 3")
        s(self, "aci_gamma", float(self.aci_gamma))
        s(self, "aci_max_factor", float(self.aci_max_factor))
        try:
            self.aci()
        except ValueError as e:
            raise ConfigError(f"calibration.aci_*: {e}") from None

    def aci(self):
        """Параметры адаптивной калибровки на устройстве."""
        return ACIParams(target=round(1.0 - self.nominal, 10), gamma=self.aci_gamma,
                         max_factor=self.aci_max_factor)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "calibration"))


@dataclass(frozen=True)
class RuntimeConfig:
    """Параметры хоста устройства, которые не относятся к модели."""
    site_max_dlat_deg: float = 0.5
    site_max_dlon_deg: float = 0.5
    site_max_delev_m: float = 100.0

    def __post_init__(self):
        for f in fields(self):
            v = float(getattr(self, f.name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ConfigError(f"runtime.{f.name} = {v}: нужно конечное число не меньше нуля")
            object.__setattr__(self, f.name, v)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "runtime"))


__all__ = ["ABLATION_NAMES", "AUGMENT_PROB_FIELDS", "AUGMENT_PROFILES", "Ablations",
           "AugmentConfig", "CALIBRATION_NOMINALS", "COVERAGE_DIMS",
           "COVERAGE_DIMS_EXTERNAL", "COVERAGE_DIMS_INTERNAL", "CalibrationConfig", "ConfigError",
           "DEFAULT_MODE_GROUPS",
           "DLinearConfig", "DataConfig", "ENCODER_CHANNELS", "GRUConfig", "LRUConfig",
           "MODEL_CONFIGS", "ModeGroup", "ModelConfig", "PERSISTENT_GROUP",
           "PERSISTENT_PERIODS",
           "PatchTSTConfig",
           "DEFAULT_SCENARIOS", "ROBUSTNESS_QC", "RobustnessConfig", "RunConfig", "RuntimeConfig",
           "SCENARIO_INPUT", "SCENARIO_INSTRUMENT", "SCENARIO_RULES", "SOLAR_CHANNELS",
           "ScenarioRule", "ScenarioSpec", "Seeds", "TrainConfig",
           "check_pipeline_compat", "default_scenarios", "model_config_for",
           "model_config_from_dict", "run_label",
           "to_jsonable"]
