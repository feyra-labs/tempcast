"""Конфигурация проекта: модель, данные, обучение.

Три датакласса верхнего уровня:

* ``ModelConfig`` (МАЯК), ``GRUConfig``, ``DLinearConfig``, ``LRUConfig``,
  ``PatchTSTConfig`` - архитектура. Всё, что раньше лежало глобальными константами
  и числами в конструкторах модулей;
* ``DataConfig`` - пути, временные окна, пороги масок, параметры аугментаций, QC окна;
* ``TrainConfig`` - это ``mayak.protocol.Protocol`` (единый протокол
  обучения): шаги, батч, оптимизатор, расписание, ранняя остановка, сиды;
* ``RobustnessConfig`` - сценарии робастности обученной модели (блок 12): какие
  преобразования окна, на каких уровнях деградации, на каких станциях и лидах,
  допуск проверки скилла. Читается ``mayak.robustness``, в обучение не входит.

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
from mayak.data.splits import ROLE_TEST, ROLES, TIME_BOUNDS
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
CHANNEL_MAX_LAG = 24
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
        """Рецептивное поле энкодера, округлённое вверх до 16 ч.

        Нижняя граница истории, которую видит выход энкодера. Сырое окно рантайма
        длиннее на лаг входных каналов - см. stream_window.
        """
        return 16 * math.ceil(self.receptive_field / 16)

    @property
    def stream_window(self):
        """Длина сырого окна наблюдений в потоковом рантайме, ч (кратно 16)."""
        return 16 * math.ceil((self.receptive_field - 1 + CHANNEL_MAX_LAG) / 16)

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


@dataclass(frozen=True)
class LRUConfig:
    """Linear Recurrent Unit - бейзлайн «линейная память без
    разложения на якорь и аномалию».

    d_model    - ширина слоя (вход/выход каждого LRU-блока);
    d_state    - число комплексных собственных значений диагональной рекуррентности;
    layers     - число блоков (pre-LayerNorm → LRU → GELU → GLU → остаток);
    tau_bounds - диапазон постоянных времени при инициализации, ч:
                 |λ| ∈ [exp(−1/τ_min), exp(−1/τ_max)] (r_min, r_max оригинала). По
                 умолчанию - тот же диапазон, что у мод МАЯК: модели отличаются
                 разложением, а не априорной памятью;
    min_period - наименьший начальный период колебаний, ч: фаза arg λ ∈ [0, 2π/min_period]
                 (max_phase оригинала);
    head_hidden - ширина MLP-головы, общей для всех лидов;
    scan       - развёртка рекуррентности: ``chunked`` (блочный ассоциативный скан, по
                 умолчанию), ``associative`` (скан Хиллиса-Стила по всей длине) или
                 ``recurrent`` (наивный цикл по часам - эталон для тестов);
    chunk      - длина блока для ``chunked``.
    """
    arch: str = "lru"
    horizon: int = H
    quantiles: tuple = QUANTILES
    max_history: int = L_MAX
    d_model: int = 64
    d_state: int = 128
    layers: int = 4
    dropout: float = 0.0
    tau_bounds: tuple = (3.0, 240.0)
    min_period: float = 12.0
    head_hidden: int = 128
    scan: str = "chunked"
    chunk: int = 32

    def __post_init__(self):
        s = object.__setattr__
        if self.arch != "lru":
            raise ConfigError(f"LRUConfig: arch={self.arch!r}")
        s(self, "quantiles", _floats(self.quantiles))
        s(self, "tau_bounds", _floats(self.tau_bounds))
        for name in ("horizon", "max_history", "d_model", "d_state", "layers", "head_hidden",
                     "chunk"):
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
        if self.scan not in LRU_SCANS:
            raise ConfigError(f"LRUConfig.scan = {self.scan!r}; допустимо {LRU_SCANS}")

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


LRU_SCANS = ("chunked", "associative", "recurrent")
PATCH_PADDINGS = ("end", "none")
PATCHTST_NORMS = ("batch", "layer")


@dataclass(frozen=True)
class PatchTSTConfig:
    """PatchTST. Значения по умолчанию - конфигурация авторов для
    набора Weather (scripts/PatchTST/weather.sh): патч 16, шаг 8, паддинг «end»,
    3 слоя, d_model 128, 16 голов, d_ff 256, dropout 0.2, head_dropout 0, RevIN без
    аффинных параметров (affine=0 по умолчанию в run_longExp.py).

    input_len       - длина входа (контракт данных: L_MAX);
    patch_len, stride, padding_patch - разбиение на патчи;
    revin_min_valid - сколько валидных часов нужно для статистики экземпляра; меньше -
                      нормировка (0, 1) (история пуста или почти пуста).
    """
    arch: str = "patchtst"
    horizon: int = H
    quantiles: tuple = QUANTILES
    input_len: int = L_MAX
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    layers: int = 3
    dropout: float = 0.2
    attn_dropout: float = 0.0
    head_dropout: float = 0.0
    res_attention: bool = True
    norm: str = "batch"
    revin: bool = True
    revin_min_valid: int = 2

    def __post_init__(self):
        s = object.__setattr__
        if self.arch != "patchtst":
            raise ConfigError(f"PatchTSTConfig: arch={self.arch!r}")
        s(self, "quantiles", _floats(self.quantiles))
        for name in ("horizon", "input_len", "patch_len", "stride", "d_model", "n_heads", "d_ff",
                     "layers", "revin_min_valid"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"PatchTSTConfig.{name} < 1")
            s(self, name, v)
        for name in ("dropout", "attn_dropout", "head_dropout"):
            v = float(getattr(self, name))
            if not 0.0 <= v < 1.0:
                raise ConfigError(f"PatchTSTConfig.{name} = {v} вне [0, 1)")
            s(self, name, v)
        for name in ("res_attention", "revin"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"PatchTSTConfig.{name} должен быть bool")
        if self.padding_patch not in PATCH_PADDINGS:
            raise ConfigError(f"PatchTSTConfig.padding_patch = {self.padding_patch!r}; "
                              f"допустимо {PATCH_PADDINGS}")
        if self.norm not in PATCHTST_NORMS:
            raise ConfigError(f"PatchTSTConfig.norm = {self.norm!r}; допустимо {PATCHTST_NORMS}")
        if self.patch_len > self.input_len:
            raise ConfigError(f"патч {self.patch_len} ч длиннее входа {self.input_len} ч")
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model {self.d_model} не делится на n_heads {self.n_heads}")

    @property
    def n_quantiles(self):
        return len(self.quantiles)

    @property
    def n_patches(self):
        """(L − P) // S + 1 (+1 при паддинге «end» повторением последнего значения S раз)."""
        n = (self.input_len - self.patch_len) // self.stride + 1
        return n + (1 if self.padding_patch == "end" else 0)

    def to_dict(self):
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d=None):
        return cls(**_strict_kwargs(cls, d, "model"))


MODEL_CONFIGS = {"mayak": ModelConfig, "gru": GRUConfig, "dlinear": DLinearConfig,
                 "lru": LRUConfig, "patchtst": PatchTSTConfig}


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


def _pair(v, name, cast=float):
    v = tuple(cast(x) for x in v)
    if len(v) != 2 or v[0] > v[1]:
        raise ConfigError(f"{name} = {v}: нужна пара lo ≤ hi")
    return v


AUGMENT_PROB_FIELDS = {
    "coords": "coords_prob", "scale": "scale_prob", "drift": "drift_prob",
    "offset": "offset_prob", "noise": "noise_prob", "spike": "spike_prob",
    "stuck": "stuck_prob", "units": "units_prob", "quantize": "quant_prob",
    "dropout": "dropout_prob", "gap": "gap_prob", "sparse": "sparse_prob",
    "outage": "outage_prob", "drop_pressure": "drop_pressure_prob",
    "drop_humidity": "drop_humidity_prob",
}


@dataclass(frozen=True)
class AugmentConfig:
    """Аугментации обучающих окон, имитирующие реальный прибор.

    Значения по умолчанию - профиль ``aggressive``: профиль обучения для переноса на
    реальные наблюдения. Именованные профили - ``AUGMENT_PROFILES`` (отличия от
    умолчаний), собираются ``AugmentConfig.from_profile``. Каждая аугментация
    включается с вероятностью ``*_prob`` и берёт параметры из своего диапазона.
    Кортежи из трёх элементов - по каналам (T, P, RH); пары - диапазон [lo, hi].

    Инструмент (к валидным точкам истории):
      scale_max        - множитель 1 ± U(0, a) на канал;
      drift_max        - дрейф за историю: к началу истории ±a, к последнему часу 0;
      drift_rw_frac    - доля случайного блуждания (остальное - линейный дрейф);
      offset_min/max   - постоянное смещение T, |b| log-равномерно в [min, max] (при
                         min = 0 - равномерно в [0, max]); **и к истории, и к цели**;
                         offset_max = 0 - выключено (абляция no_offset_aug);
      noise_sd         - гауссов шум по каналам (°C, гПа, %);
      quant_f_frac     - доля квантования T целыми °F (остальное - шаг 0.1 °C).
    Грубые ошибки (к истории):
      spike_max_count, spike_min/max - число выбросов 1..n и их величина по каналам;
      stuck_hours      - залипание значения канала на [lo, hi] ч;
      units_hours      - подмена единиц на участке [lo, hi] ч: T в °F либо (с долей
                         units_p_frac) давление, приведённое к уровню моря.
    Доступность (к маске истории):
      dropout_max_rate - одиночные пропуски: доля часов U(0, a);
      gap_max_count, gap_max_len - блочные пропуски: число 1..n, длина 1..len ч;
      sparse_every     - регулярная отчётность: валиден каждый k-й час, k из набора;
      outage_hours     - выпадение канала в середине истории на [lo, hi] ч;
      drop_pressure/humidity - канал отсутствует на всей истории.
    Метаданные:
      coord_jitter_deg - дрожание широты и долготы U(±a), градусы;
      elev_jitter_m    - дрожание высоты N(0, a), м.
    """
    profile: str = "aggressive"
    scale_prob: float = 0.3
    scale_max: tuple = (0.02, 0.001, 0.05)
    drift_prob: float = 0.3
    drift_max: tuple = (1.5, 1.5, 6.0)
    drift_rw_frac: float = 0.5
    offset_prob: float = 0.8
    offset_min: float = 0.1
    offset_max: float = 3.0
    noise_prob: float = 1.0
    noise_sd: tuple = (0.2, 0.3, 2.0)
    quant_prob: float = 0.5
    quant_f_frac: float = 0.4
    spike_prob: float = 0.2
    spike_max_count: int = 3
    spike_min: tuple = (12.0, 15.0, 40.0)
    spike_max: tuple = (30.0, 40.0, 80.0)
    stuck_prob: float = 0.15
    stuck_hours: tuple = (12, 96)
    units_prob: float = 0.05
    units_hours: tuple = (24, 96)
    units_p_frac: float = 0.3
    dropout_prob: float = 0.3
    dropout_max_rate: float = 0.2
    gap_prob: float = 0.5
    gap_max_count: int = 3
    gap_max_len: int = 96
    sparse_prob: float = 0.15
    sparse_every: tuple = (3, 6)
    outage_prob: float = 0.2
    outage_hours: tuple = (24, 240)
    drop_humidity_prob: float = 0.15
    drop_pressure_prob: float = 0.15
    coords_prob: float = 1.0
    coord_jitter_deg: float = 0.4
    elev_jitter_m: float = 50.0

    def __post_init__(self):
        s = object.__setattr__
        if self.profile not in AUGMENT_PROFILES:
            raise ConfigError(f"неизвестный профиль аугментаций {self.profile!r}; "
                              f"есть {tuple(AUGMENT_PROFILES)}")
        for name in ("scale_max", "drift_max", "noise_sd", "spike_min", "spike_max"):
            v = _floats(getattr(self, name))
            if len(v) != 3 or min(v) < 0:
                raise ConfigError(f"{name} - по одному неотрицательному значению на канал "
                                  f"T, P, RH: {v}")
            s(self, name, v)
        if any(a > b for a, b in zip(self.spike_min, self.spike_max)):
            raise ConfigError(f"spike_min {self.spike_min} > spike_max {self.spike_max}")
        for name in ("stuck_hours", "units_hours", "outage_hours"):
            v = _pair(getattr(self, name), name, int)
            if v[0] < 1:
                raise ConfigError(f"{name}: длительность ≥ 1 ч")
            s(self, name, v)
        every = _ints(self.sparse_every)
        if not every or min(every) < 2:
            raise ConfigError(f"sparse_every - шаги отчётности ≥ 2 ч: {every}")
        s(self, "sparse_every", every)
        for name in (*AUGMENT_PROB_FIELDS.values(), "drift_rw_frac", "quant_f_frac",
                     "units_p_frac", "dropout_max_rate"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ConfigError(f"{name} = {v} вне [0, 1]")
            s(self, name, v)
        for name in ("spike_max_count", "gap_max_count", "gap_max_len"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"{name} ≥ 1")
            s(self, name, v)
        for name in ("offset_min", "offset_max", "coord_jitter_deg", "elev_jitter_m"):
            v = float(getattr(self, name))
            if v < 0:
                raise ConfigError(f"{name} ≥ 0")
            s(self, name, v)
        if self.offset_max == 0.0:
            s(self, "offset_min", 0.0)
        elif self.offset_min > self.offset_max:
            raise ConfigError(f"offset_min {self.offset_min} > offset_max {self.offset_max}")

    @classmethod
    def from_profile(cls, name="aggressive", **overrides):
        """Именованный профиль + явные переопределения."""
        if name not in AUGMENT_PROFILES:
            raise ConfigError(f"неизвестный профиль аугментаций {name!r}; "
                              f"есть {tuple(AUGMENT_PROFILES)}")
        kw = {**AUGMENT_PROFILES[name], **overrides}
        kw.pop("profile", None)
        return cls(**_strict_kwargs(cls, {"profile": name, **kw}, "data.augment"))

    @classmethod
    def from_dict(cls, d=None):
        """Словарь → конфиг: профиль из ключа profile (по умолчанию aggressive), затем
        остальные ключи поверх него."""
        d = _strict_kwargs(cls, d, "data.augment")
        return cls.from_profile(d.pop("profile", "aggressive"), **d)

    @classmethod
    def only(cls, name, profile="aggressive"):
        if name not in AUGMENT_PROB_FIELDS:
            raise ConfigError(f"нет аугментации {name!r}; есть {tuple(AUGMENT_PROB_FIELDS)}")
        base = cls.from_profile(profile)
        probs = {f: 0.0 for f in AUGMENT_PROB_FIELDS.values()}
        probs[AUGMENT_PROB_FIELDS[name]] = 1.0
        return replace(base, **probs)

    def deviations(self):
        """Поля, отличающиеся от объявленного профиля: {поле: (в профиле, фактически)}."""
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
    "aggressive": {},
    "soft": dict(
        scale_prob=0.15, scale_max=(0.01, 0.0005, 0.03),
        drift_prob=0.15, drift_max=(0.7, 1.0, 3.0),
        offset_max=1.5,
        noise_sd=(0.15, 0.2, 1.5),
        quant_prob=0.3, quant_f_frac=0.2,
        spike_prob=0.05, spike_max_count=1,
        stuck_prob=0.05, stuck_hours=(6, 48),
        units_prob=0.01,
        dropout_prob=0.2, dropout_max_rate=0.1,
        gap_prob=0.3, gap_max_count=2, gap_max_len=48,
        sparse_prob=0.05, outage_prob=0.1, outage_hours=(24, 120),
        drop_humidity_prob=0.1, drop_pressure_prob=0.1,
        coord_jitter_deg=0.2, elev_jitter_m=20.0),
    "base": dict(
        scale_prob=0.0, drift_prob=0.0,
        offset_prob=1.0, offset_min=0.0, offset_max=0.7,
        noise_prob=1.0, noise_sd=(0.2, 0.5, 2.0), quant_prob=0.0,
        spike_prob=0.0, stuck_prob=0.0, units_prob=0.0,
        dropout_prob=0.0, gap_prob=0.3, gap_max_count=1, gap_max_len=24,
        sparse_prob=0.0, outage_prob=0.0,
        drop_humidity_prob=0.1, drop_pressure_prob=0.1,
        coords_prob=1.0, coord_jitter_deg=0.4, elev_jitter_m=0.0),
    "none": {f: 0.0 for f in AUGMENT_PROB_FIELDS.values()},
}

ZONE_WEIGHTINGS = {"uniform": 0.0, "inv_sqrt": 0.5, "inv": 1.0}


@dataclass(frozen=True)
class DataConfig:
    manifest: str = "data/manifest.csv"
    cache_root: Optional[str] = None
    time_bounds: dict = field(default_factory=lambda: dict(TIME_BOUNDS))
    target_mask: TargetMaskConfig = TargetMaskConfig()
    val_every_hours: int = 72
    val_max_windows: int = 8000
    augment: AugmentConfig = AugmentConfig()
    window_qc: bool = True
    zone_weighting: str = "inv_sqrt"
    zone_weight_cap: float = 0.0

    def __post_init__(self):
        if self.zone_weighting not in ZONE_WEIGHTINGS:
            raise ConfigError(f"zone_weighting = {self.zone_weighting!r}; "
                              f"допустимо {sorted(ZONE_WEIGHTINGS)}")
        cap = float(self.zone_weight_cap)
        if cap != 0.0 and cap < 1.0:
            raise ConfigError(f"zone_weight_cap = {cap}: 0 (выкл.) или ≥ 1")
        object.__setattr__(self, "zone_weight_cap", cap)
        if not isinstance(self.target_mask, TargetMaskConfig):
            object.__setattr__(self, "target_mask", TargetMaskConfig(**_strict_kwargs(
                TargetMaskConfig, self.target_mask, "data.target_mask")))
        if not isinstance(self.augment, AugmentConfig):
            object.__setattr__(self, "augment", AugmentConfig.from_dict(self.augment))
        tb = dict(self.time_bounds)
        if tb != dict(TIME_BOUNDS):
            raise ConfigError(f"data.time_bounds = {tb} расходится с контрактом сплитов "
                              f"{dict(TIME_BOUNDS)} (mayak/data/splits.py, входит в ключ кэша)")
        object.__setattr__(self, "time_bounds", tb)
        if self.val_every_hours < 1 or self.val_max_windows < 1:
            raise ConfigError("val_every_hours и val_max_windows ≥ 1")
        if not isinstance(self.window_qc, bool):
            raise ConfigError(f"window_qc должен быть bool, получено {self.window_qc!r}")

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
            data = replace(data, augment=replace(data.augment, offset_min=0.0, offset_max=0.0))
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


SCENARIO_INSTRUMENT, SCENARIO_INPUT = "instrument", "input"
ROBUSTNESS_QC = ("none", "point", "window")
ROBUSTNESS_TIME_KEYS = ("val", "test")


@dataclass(frozen=True)
class ScenarioRule:
    """Контракт сценария робастности; реализация - ``mayak.data.scenarios``.

    kind      - ``instrument`` («свойство прибора») или ``input`` («отказ входа»);
    target    - искажается ли и цель (для свойства прибора - да, кроме вариантов
                «незамеченное смещение прибора», где искажён только вход);
    max_level - верхняя граница уровня деградации (None - без границы);
    integer   - уровень - целое число (часы, номер варианта);
    params    - фиксированные параметры сценария и их значения по умолчанию;
    guard     - участвует ли сценарий в проверке «скилл не ниже −допуска»;
    variant_of - для варианта «искажён только вход» - имя основного сценария.
    """
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
    """Значение параметра сценария к типу и длине значения по умолчанию."""
    if isinstance(default, tuple):
        try:
            v = tuple(value)
        except TypeError:
            raise ConfigError(f"{where}.{key}: ожидался список из {len(default)} значений")
        if len(v) != len(default) and not (key == "channels" and 1 <= len(v) <= 3):
            raise ConfigError(f"{where}.{key}: {len(v)} значений, нужно {len(default)}")
        cast = int if all(isinstance(d, int) for d in default) else float
        return tuple(cast(x) for x in v)
    return type(default)(value)


@dataclass(frozen=True)
class ScenarioSpec:
    """Один сценарий робастности из конфига.

    name   - ключ ``SCENARIO_RULES``;
    levels - уровни деградации по возрастанию; первый обязан быть 0 - сценарий с
             нулевым параметром не меняет данные и служит точкой отсчёта кривой;
    params - переопределения фиксированных параметров (остальные - по умолчанию);
    guard  - None - по умолчанию сценария (``ScenarioRule.guard``).
    """
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
    """Прогон робастности обученной модели без переобучения.

    roles, time_key     - станции и временное окно внутреннего набора (внешний тест
                          всегда external_test × test);
    every_hours, windows_per_station - шаг кандидатов и стратифицированная подвыборка
                          окон (одинаковое число окон с каждой станции);
    qc                  - контроль качества после сценария: ``point`` - поточечный, как
                          в рантайме на устройстве; ``window`` - оконный, как при
                          обучении; ``none`` - без QC;
    leads               - лиды кривых и проверки скилла;
    skill_tolerance     - допуск: скилл ``guard_models`` не ниже −skill_tolerance ни в
                          одном сценарии с ``guard`` и ни на одном уровне и лиде;
    seed                - сид случайных чисел сценариев (общие для всех уровней);
    bootstrap, ci_level - блочный бутстрап по станциям (блок 5); 0 - без интервалов.
    """
    roles: tuple = (ROLE_TEST,)
    time_key: str = "test"
    every_hours: int = 72
    windows_per_station: int = 20
    qc: str = "point"
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
        """Подмножество сценариев по именам (порядок конфига сохраняется)."""
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


__all__ = ["ABLATION_NAMES", "AUGMENT_PROB_FIELDS", "AUGMENT_PROFILES", "Ablations",
           "AugmentConfig", "CHANNEL_MAX_LAG", "ConfigError",
           "DEFAULT_MODE_GROUPS",
           "DLinearConfig", "DataConfig", "ENCODER_CHANNELS", "GRUConfig", "LRUConfig",
           "LRU_SCANS", "MODEL_CONFIGS", "ModeGroup", "ModelConfig", "PatchTSTConfig",
           "DEFAULT_SCENARIOS", "ROBUSTNESS_QC", "RobustnessConfig", "RunConfig",
           "SCENARIO_INPUT", "SCENARIO_INSTRUMENT", "SCENARIO_RULES", "SOLAR_CHANNELS",
           "ScenarioRule", "ScenarioSpec", "Seeds", "TrainConfig",
           "check_pipeline_compat", "default_scenarios", "model_config_for",
           "model_config_from_dict", "run_label",
           "to_jsonable"]
