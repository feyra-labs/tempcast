"""Конфигурация проекта: модель, данные, обучение.

Три датакласса верхнего уровня:

* ``ModelConfig`` (МАЯК), ``GRUConfig``, ``DLinearConfig`` - архитектура. Всё, что
  раньше лежало глобальными константами и числами в конструкторах модулей;
* ``DataConfig`` - пути, временные окна, пороги масок, параметры аугментаций;
* ``TrainConfig`` - это ``mayak.protocol.Protocol`` (единый протокол
  обучения): шаги, батч, оптимизатор, расписание, ранняя остановка, сиды.

``RunConfig`` собирает три части в один объект - его полностью разрешённая форма
пишется рядом с чекпойнтом и внутри него.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

from mayak.constants import H, L_MAX, QUANTILES
from mayak.data.masking import TargetMaskConfig
from mayak.data.splits import TIME_BOUNDS
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
    """Датакласс / кортежи → JSON-совместимые словари и списки."""
    return json.loads(json.dumps(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
                                 default=lambda o: dataclasses.asdict(o)))

@dataclass(frozen=True)
class ModeGroup:
    """Группа затухающих мод.

    tau0   - начальные постоянные времени, ч (по одной на моду; их число = размер группы);
    period - начальные периоды колебаний, ч; 0 - мода без колебаний (чистая релаксация).
             Один элемент распространяется на всю группу.
    """
    name: str
    tau0: tuple
    period: tuple = (0.0,)

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
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "tau0", tau0)
        object.__setattr__(self, "period", period)

    @property
    def size(self):
        return len(self.tau0)


DEFAULT_MODE_GROUPS = (
    ModeGroup("R", (3, 6, 12, 24, 48, 96, 168, 240), (0.0,)),
    ModeGroup("D", (12, 24, 48, 96, 168, 240), (24.0,)),
    ModeGroup("S", (12, 24, 72, 168), (12.0,)),
    ModeGroup("W", (24, 48, 72, 120, 168, 240), (60, 84, 108, 132, 156, 192)),
)

ENCODER_CHANNELS = ("aT", "adef", "dP3", "dP24", "rh", "sin_d", "cos_d", "czp",
                    "sin_y", "cos_y", "vt", "vp", "vr")
SOLAR_CHANNELS = ("sin_d", "cos_d", "czp")
N_SOLAR_HEAD = 3
N_DAILY_SUMMARY = 6
UNSTRUCTURED_PERIODS = (12.0, 240.0)


@dataclass(frozen=True)
class Ablations:
    """Флаги абляций. Меняют только поведение модели (кроме no_offset_aug, см. ниже).

    no_anchor       - прогноз без климат-поля: C(h), σ(h) и дефицит точки росы - обучаемые
                      глобальные константы, одинаковые для всех станций и моментов;
    no_compression  - без доказательного сжатия: считывание мод делится на массу
                      свидетельств e (не меньше EVIDENCE_FLOOR), а не на e + κ;
    no_passport     - паспорт станции выключен: z ≡ 0, KL = 0;
    no_solar        - солнечные признаки убраны из входа энкодера и из ковариат голов
                      (гармонический базис климат-поля не трогается - это часть якоря);
    no_mode_groups  - групповой структуры нет: M мод одной группой с однородной
                      инициализацией (τ₀ и периоды лог-равномерно), одна групповая энергия;
    no_offset_aug   - без аугментации постоянного смещения температуры. Это свойство
                      потока данных, но флаг живёт здесь, чтобы абляция задавалась в одном
                      месте; RunConfig.resolved переносит его в DataConfig.augment.
    """
    no_anchor: bool = False
    no_compression: bool = False
    no_passport: bool = False
    no_solar: bool = False
    no_mode_groups: bool = False
    no_offset_aug: bool = False

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
    encoder_width: int = 48
    encoder_dilations: tuple = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32)
    encoder_kernel: int = 3
    encoder_norm_groups: int = 4
    loc_freqs: int = 24
    loc_freq_scale: float = 12.0
    loc_freq_max: float = 30.0
    loc_seed: int = 7
    field_hidden: int = 48
    heads_hidden: int = 48
    heads_z_proj: int = 4
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
        for name in ("loc_freq_scale", "loc_freq_max"):
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
            bad = [t for t in g.tau0 if not lo <= t <= hi]
            if bad:
                raise ConfigError(f"группа мод {g.name!r}: τ₀ {bad} вне [{lo}, {hi}]")
        names = [g.name for g in self.mode_groups]
        if not names or len(set(names)) != len(names):
            raise ConfigError(f"имена групп мод пусты или повторяются: {names}")
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

    @property
    def effective_mode_groups(self):
        """Группы мод с учётом абляции no_mode_groups."""
        if not self.ablations.no_mode_groups:
            return self.mode_groups
        m = self.n_modes
        lo, hi = self.tau_bounds
        p_lo, p_hi = UNSTRUCTURED_PERIODS
        geom = lambda a, b: tuple(a * (b / a) ** (i / max(m - 1, 1)) for i in range(m))
        return (ModeGroup("all", geom(lo, hi), geom(p_lo, p_hi)),)

    @property
    def n_modes(self):
        return sum(g.size for g in self.mode_groups)

    @property
    def group_sizes(self):
        return tuple(g.size for g in self.effective_mode_groups)

    @property
    def group_names(self):
        return tuple(g.name for g in self.effective_mode_groups)

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
        """o + энергии групп + их сумма + солнце + log σ + проекция z + доля лида + log(1+e)."""
        return 1 + self.n_groups + 1 + self.n_solar_head + 1 + self.heads_z_proj + 1 + 1

    @property
    def receptive_field(self):
        """Рецептивное поле энкодера, ч: (k − 1)·Σd + 1 (для k = 3 - удвоенная сумма + 1)."""
        return (self.encoder_kernel - 1) * sum(self.encoder_dilations) + 1

    @property
    def history_days(self):
        return self.max_history // 24

    @property
    def stream_buffer(self):
        """Длина буфера потокового рантайма: рецептивное поле, округлённое вверх до 16 ч."""
        return 16 * math.ceil(self.receptive_field / 16)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


@dataclass(frozen=True)
class GRUConfig:
    """GRU seq2seq-бейзлайн."""
    arch: str = "gru"
    horizon: int = H
    quantiles: tuple = QUANTILES
    hidden: int = 96
    layers: int = 2
    mu_hidden: int = 256
    sigma_hidden: int = 128

    def __post_init__(self):
        if self.arch != "gru":
            raise ConfigError(f"GRUConfig: arch={self.arch!r}")
        object.__setattr__(self, "quantiles", _floats(self.quantiles))

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
    """DLinear-бейзлайн."""
    arch: str = "dlinear"
    horizon: int = H
    quantiles: tuple = QUANTILES
    input_len: int = L_MAX
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


MODEL_CONFIGS = {"mayak": ModelConfig, "gru": GRUConfig, "dlinear": DLinearConfig}


def model_config_for(arch, cfg=None):
    """Конфиг архитектуры arch из None / словаря / датакласса."""
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
    """Словарь с ключом arch → конфиг соответствующей архитектуры."""
    d = dict(d or {})
    return model_config_for(d.get("arch", "mayak"), d)


def check_pipeline_compat(cfg):
    """Модель совместима с контрактом данных (горизонт, история, квантили)."""
    errs = []
    if cfg.horizon != H:
        errs.append(f"horizon={cfg.horizon}, конвейер данных - {H}")
    if tuple(cfg.quantiles) != tuple(float(q) for q in QUANTILES):
        errs.append(f"quantiles={cfg.quantiles}, модуль метрик - {QUANTILES}")
    hist = getattr(cfg, "max_history", getattr(cfg, "input_len", L_MAX))
    if hist != L_MAX:
        errs.append(f"длина истории {hist}, окна датасетов - {L_MAX}")
    if errs:
        raise ConfigError("конфиг модели несовместим с контрактом данных (mayak/constants.py): "
                          + "; ".join(errs) + ". Модель это поддерживает, но сплиты, кэш и "
                          "метрики построены под контракт - меняйте его там.")
    return cfg


@dataclass(frozen=True)
class AugmentConfig:
    """Аугментации обучающих окон. Значения по умолчанию - прежний набор (профиль base).

    gap_prob / gap_max_len     - блочный пропуск в истории: вероятность и длина до, ч;
    drop_humidity_prob         - вся история без влажности;
    drop_pressure_prob         - вся история без давления;
    noise_sd                   - шум по каналам T, P, RH (°C, гПа, %);
    offset_max                 - постоянное смещение T в [−a, a] для истории и цели; 0 - нет;
    coord_jitter_deg           - дрожание широты и долготы, градусы.
    """
    profile: str = "base"
    gap_prob: float = 0.3
    gap_max_len: int = 24
    drop_humidity_prob: float = 0.1
    drop_pressure_prob: float = 0.1
    noise_sd: tuple = (0.2, 0.5, 2.0)
    offset_max: float = 0.7
    coord_jitter_deg: float = 0.4

    def __post_init__(self):
        object.__setattr__(self, "noise_sd", _floats(self.noise_sd))
        if len(self.noise_sd) != 3:
            raise ConfigError(f"noise_sd - по одному на канал T, P, RH: {self.noise_sd}")
        for name in ("gap_prob", "drop_humidity_prob", "drop_pressure_prob"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ConfigError(f"{name} вне [0, 1]")
        if self.gap_max_len < 1 or self.offset_max < 0 or self.coord_jitter_deg < 0:
            raise ConfigError("gap_max_len ≥ 1, offset_max ≥ 0, coord_jitter_deg ≥ 0")


@dataclass(frozen=True)
class DataConfig:
    manifest: str = "data/manifest.csv"
    cache_root: Optional[str] = None
    time_bounds: dict = field(default_factory=lambda: dict(TIME_BOUNDS))
    target_mask: TargetMaskConfig = TargetMaskConfig()
    val_every_hours: int = 72
    val_max_windows: int = 8000
    augment: AugmentConfig = AugmentConfig()

    def __post_init__(self):
        if not isinstance(self.target_mask, TargetMaskConfig):
            object.__setattr__(self, "target_mask", TargetMaskConfig(**_strict_kwargs(
                TargetMaskConfig, self.target_mask, "data.target_mask")))
        if not isinstance(self.augment, AugmentConfig):
            object.__setattr__(self, "augment", AugmentConfig(**_strict_kwargs(
                AugmentConfig, self.augment, "data.augment")))
        tb = dict(self.time_bounds)
        if tb != dict(TIME_BOUNDS):
            raise ConfigError(f"data.time_bounds = {tb} расходится с контрактом сплитов "
                              f"{dict(TIME_BOUNDS)} (mayak/data/splits.py, входит в ключ кэша)")
        object.__setattr__(self, "time_bounds", tb)
        if self.val_every_hours < 1 or self.val_max_windows < 1:
            raise ConfigError("val_every_hours и val_max_windows ≥ 1")

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "data"))


@dataclass(frozen=True)
class RunConfig:
    """Полная конфигурация прогона: модель + данные + обучение."""
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
        """Перенос межсекционных эффектов абляций. Идемпотентно."""
        abl = getattr(self.model, "ablations", None)
        data = self.data
        if abl is not None and abl.no_offset_aug and data.augment.offset_max != 0.0:
            data = replace(data, augment=replace(data.augment, offset_max=0.0))
        return replace(self, data=data)

    def to_dict(self):
        return dict(model=self.model.to_dict(), data=self.data.to_dict(),
                    train=self.train.to_dict())

    @classmethod
    def from_dict(cls, d):
        d = _strict_kwargs(cls, d, "run")
        return cls(**d)


def run_label(model_cfg):
    """Короткое имя варианта модели для таблиц: arch[-абляции]."""
    abl = getattr(model_cfg, "ablations", None)
    act = abl.active() if abl is not None else ()
    return model_cfg.arch + ("" if not act else "-" + "+".join(act))


__all__ = ["ABLATION_NAMES", "Ablations", "AugmentConfig", "ConfigError", "DEFAULT_MODE_GROUPS",
           "DLinearConfig", "DataConfig", "ENCODER_CHANNELS", "GRUConfig", "MODEL_CONFIGS",
           "ModeGroup", "ModelConfig", "RunConfig", "SOLAR_CHANNELS", "Seeds", "TrainConfig",
           "check_pipeline_compat", "model_config_for", "model_config_from_dict", "run_label",
           "to_jsonable"]
