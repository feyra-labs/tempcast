"""Конфигурация проекта: модель, данные, обучение.

Датаклассы верхнего уровня:

* ModelConfig (МАЯК), GRUConfig, DLinearConfig, LRUConfig,
  PatchTSTConfig - архитектура;
* DataConfig - пути, временные окна, пороги масок, параметры аугментаций, QC окна;
* протокол обучения - шаги, батч, оптимизатор, расписание, ранняя остановка, сиды;
  он один на все архитектуры;
* RobustnessConfig - сценарии робастности обученной модели: какие
  преобразования окна, на каких уровнях деградации, на каких станциях и лидах,
  допуск проверки скилла. Читается проверкой робастности, в обучение не входит.
* CalibrationConfig - анализ калибровки: номинал и допуск покрытия,
  пороги страт, бутстрап, критерий условной поправки, сетка кривой «острота против
  покрытия», параметры адаптивной калибровки на устройстве. Читается анализом
  калибровки и рантаймом, в обучение не входит.
* RuntimeConfig - параметры хоста устройства вне модели: пороги, в пределах которых
  смена координат и высоты при перезапуске считается уточнением метаданных, а не
  переносом прибора. Пишется в манифест графов, в обучение не входит.

RunConfig собирает три части в один объект - его полностью разрешённая форма
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
    """JSON-совместимая копия датакласса или вложенных кортежей.

    Args:
        obj: датакласс или значение из словарей, списков и кортежей.

    Returns:
        То же содержимое из словарей, списков и простых значений.
    """
    return json.loads(json.dumps(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
                                 default=lambda o: dataclasses.asdict(o)))


@dataclass(frozen=True)
class ModeGroup:
    """Группа затухающих мод.

    Attributes:
        name: имя группы.
        tau0: начальные постоянные времени, ч, по одной на моду; их число - размер
            группы.
        period: начальные периоды колебаний, ч; ноль - мода без колебаний, чистая
            релаксация. Один элемент распространяется на всю группу.
        tau_bounds: собственные пределы постоянных времени группы, ч; None - общие
            пределы модели. Их учитывают проверка конфига, считывание мод и пропагатор.
    """
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
        """Пределы постоянных времени группы.

        Args:
            default: общие пределы модели, ч.

        Returns:
            Пара (τ_min, τ_max), ч: собственные пределы группы, если они заданы, иначе
            общие.
        """
        return self.tau_bounds if self.tau_bounds is not None else tuple(default)


PERSISTENT_GROUP = "P"
DEFAULT_MODE_GROUPS = (
    ModeGroup("R", (3, 6, 12, 24, 48, 96, 168, 240), (0.0,)),
    ModeGroup("D", (12, 24, 48, 96, 168, 240), (24.0,)),
    ModeGroup("S", (12, 24, 72, 168), (12.0,)),
    ModeGroup("W", (24, 48, 72, 120, 168, 240), (60, 84, 108, 132, 156, 192)),
    ModeGroup(PERSISTENT_GROUP, (2000.0,), (0.0,), (720.0, 8760.0)),
)

ENCODER_CHANNELS = ("aT", "aTc", "adef", "dP3", "dP24", "rh", "sin_d", "cos_d", "czp",
                    "sin_y", "cos_y", "vt", "vp", "vr")
SOLAR_CHANNELS = ("sin_d", "cos_d", "czp")
N_SOLAR_HEAD = 3
N_DAILY_SUMMARY = 6
UNSTRUCTURED_PERIODS = (12.0, 240.0)


@dataclass(frozen=True)
class Ablations:
    """Флаги абляций. Меняют только поведение модели, кроме ``no_offset_aug``.

    Attributes:
        no_compression: без доказательного сжатия: считывание мод нормируется только
            массой свидетельств с нижней границей, без добавки силы сжатия.
        no_passport: паспорт станции выключен: он нулевой, штраф расхождения с приором
            тоже нулевой.
        no_solar: солнечные признаки убраны из входа энкодера и из входа голов.
            Гармонический базис климат-поля не трогается: это часть якоря.
        no_mode_groups: групповой структуры нет: все моды, кроме квазипостоянной группы,
            одной группой, постоянные времени и периоды при инициализации идут
            равномерно в логарифме. Квазипостоянная группа остаётся отдельной со своими
            пределами, поэтому групповых энергий две.
        no_offset_aug: без аугментации постоянного смещения температуры. Это свойство
            потока данных, но флаг живёт здесь, чтобы абляция задавалась в одном
            месте; полная конфигурация прогона переносит его в аугментации данных.
    """
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
    """Архитектура МАЯК.

    Производные размеры: число мод и групп, каналы энкодера, ширина входа голов и
    рецептивное поле - вычисляются из этих полей.

    Attributes:
        arch: имя архитектуры, всегда ``mayak``.
        horizon: горизонт прогноза, ч.
        max_history: наибольшая длина истории, ч, кратно суткам.
        quantiles: уровни квантилей по возрастанию, среди них медиана.
        mode_groups: группы затухающих мод. Группа с именем ``P`` - квазипостоянная:
            её мода не колеблется и затухает медленнее всех, через неё идёт устойчивое
            смещение станции в градусах. В среднюю массу свидетельств голов она не
            входит.
        tau_bounds: общие пределы постоянных времени мод, ч; группа может задать свои.
        passport_dim: размер паспорта станции.
        passport_hidden: ширина скрытого слоя кодировщика паспорта.
        encoder_width: ширина признаков энкодера истории.
        encoder_dilations: дилатации слоёв энкодера по порядку.
        encoder_kernel: ядро свёрток энкодера.
        encoder_norm_groups: число групп канальной нормализации энкодера.
        loc_freqs: число случайных частот признаков координат.
        loc_freq_scale: разброс случайных частот.
        loc_freq_max: наибольшая длина вектора частоты.
        loc_seed: сид случайных частот.
        field_hidden: ширина скрытых слоёв климат-поля.
        heads_hidden: ширина скрытых слоёв голов.
        heads_z_proj: размер проекции паспорта на вход голов.
        ablations: флаги абляций.
    """
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
    loc_freq_scale: float = 12.0
    loc_freq_max: float = 30.0
    loc_seed: int = 7
    field_hidden: int = 128
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
            g_lo, g_hi = g.bounds(self.tau_bounds)
            bad = [t for t in g.tau0 if not g_lo <= t <= g_hi]
            if bad:
                raise ConfigError(f"группа мод {g.name!r}: τ₀ {bad} вне [{g_lo}, {g_hi}]")
        names = [g.name for g in self.mode_groups]
        if not names or len(set(names)) != len(names):
            raise ConfigError(f"имена групп мод пусты или повторяются: {names}")
        for g in self.mode_groups:
            if g.name == PERSISTENT_GROUP and any(p > 0 for p in g.period):
                raise ConfigError(f"группа мод {g.name!r} квазипостоянная: её моды не "
                                  f"колеблются, периоды должны быть нулевыми, получено "
                                  f"{g.period}")
        if names == [PERSISTENT_GROUP]:
            raise ConfigError(f"кроме квазипостоянной группы {PERSISTENT_GROUP!r} нужна хотя "
                              f"бы одна группа мод: по ней считается масса свидетельств голов")
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
        """Группы мод с учётом абляции ``no_mode_groups``.

        Абляция снимает структуру только с мод погоды: они сливаются в одну группу с
        однородной инициализацией в общих пределах модели. Квазипостоянная группа
        остаётся отдельной группой после неё, со своими начальными постоянными времени и
        пределами: она описывает смещение станции, а не погоду.
        """
        if not self.ablations.no_mode_groups:
            return self.mode_groups
        kept = tuple(g for g in self.mode_groups if g.name == PERSISTENT_GROUP)
        m = self.n_modes - sum(g.size for g in kept)
        lo, hi = self.tau_bounds
        p_lo, p_hi = UNSTRUCTURED_PERIODS
        geom = lambda a, b: tuple(a * (b / a) ** (i / max(m - 1, 1)) for i in range(m))
        return (ModeGroup("all", geom(lo, hi), geom(p_lo, p_hi)), *kept)

    @property
    def mode_tau_bounds(self):
        """Пределы постоянной времени каждой моды, ч, в порядке мод модели.

        Returns:
            Кортеж пар (τ_min, τ_max), по паре на моду.
        """
        return tuple(g.bounds(self.tau_bounds) for g in self.effective_mode_groups
                     for _ in range(g.size))

    @property
    def persistent_modes(self):
        """Моды квазипостоянной группы.

        Их вклад в медиану переводится в градусы постоянным масштабом, а не
        климатологическим разбросом точки: через них идёт устойчивое смещение станции.

        Returns:
            Кортеж флагов по модам в порядке мод модели.
        """
        return tuple(g.name == PERSISTENT_GROUP for g in self.effective_mode_groups
                     for _ in range(g.size))

    @property
    def evidence_modes(self):
        """Моды, по которым головы считают среднюю массу свидетельств.

        Масса квазипостоянной моды за полную историю в разы больше массы остальных и
        растёт почти линейно с длиной истории, поэтому в среднее она не входит: вес
        поправки и вход голов описывают свидетельства о погоде.

        Returns:
            Кортеж флагов по модам в порядке мод модели.
        """
        return tuple(not f for f in self.persistent_modes)

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
        """Ширина входа голов.

        Вход голов: поправка к полю, энергии групп мод и их сумма, солнечные признаки,
        логарифм климатологического разброса, проекция паспорта, доля лида в
        горизонте и логарифм массы свидетельств.
        """
        return 1 + self.n_groups + 1 + self.n_solar_head + 1 + self.heads_z_proj + 1 + 1

    @property
    def receptive_field(self):
        """Рецептивное поле энкодера, ч.

        Каждый слой добавляет своё ядро без единицы, умноженное на дилатацию; к сумме
        по слоям прибавляется сам текущий час. При ядре 3 это удвоенная сумма дилатаций
        плюс один.
        """
        return (self.encoder_kernel - 1) * sum(self.encoder_dilations) + 1

    @property
    def history_days(self):
        return self.max_history // 24

    @property
    def device_window(self):
        """Длина сырого окна наблюдений, которое устройство хранит на диске, часы.

        Окно вмещает всю историю модели и прошлое, которое нужно причинному контролю
        качества, плюс текущий час. Длина кратна восьми, чтобы битовые маски окна
        занимали целое число байт.

        Returns:
            Число часов.
        """
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
    """Бейзлайн GRU.

    Attributes:
        hidden: размер скрытого состояния.
        layers: число слоёв GRU.
        head_hidden: ширина скрытых слоёв головы, общей для всех лидов.
    """
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
    """Бейзлайн DLinear.

    Attributes:
        input_len: длина входа в часах, последние часы истории окна. Не больше
            максимальной длины истории.
        kernel: ядро скользящего среднего в часах, нечётное.
    """
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
    """Бейзлайн с линейной рекуррентной памятью без разложения на якорь и аномалию.

    Attributes:
        d_model: ширина входа и выхода каждого блока.
        d_state: число комплексных собственных чисел рекуррентности в блоке.
        layers: число блоков.
        dropout: доля прореживания внутри блока.
        tau_bounds: диапазон постоянных времени при инициализации, часы. По умолчанию
            тот же, что у мод основной модели: модели отличаются разложением, а не
            априорной памятью.
        min_period: наименьший начальный период колебаний, часы.
        head_hidden: ширина скрытых слоёв головы, общей для всех лидов.
        scan: развёртка рекуррентности. ``chunked`` - блоками с параллельным переносом
            между блоками, по умолчанию; ``associative`` - параллельный скан по всей
            длине; ``recurrent`` - простой цикл по часам, эталон для тестов.
        chunk: длина блока для ``chunked``.
    """
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
    """Бейзлайн PatchTST, малая конфигурация.

    Патчи - целые сутки без перекрытия, заканчивающиеся в момент выпуска.

    Attributes:
        input_len: длина входа в часах, последние часы истории окна. Не больше
            максимальной длины истории.
        patch_len: длина патча в часах.
        stride: шаг между началами патчей в часах.
        padding_patch: ``end`` - добавить патч из повторов последнего часа; ``none`` -
            без добавки.
        d_model: ширина представления патча.
        n_heads: число голов внимания.
        d_ff: ширина перцептрона в слое энкодера.
        layers: число слоёв энкодера.
        dropout: прореживание в энкодере.
        attn_dropout: прореживание весов внимания.
        head_dropout: прореживание на выходе головы медианы.
        res_attention: прибавлять логиты внимания предыдущего слоя.
        norm: нормализация в слоях энкодера, ``batch`` или ``layer``.
        revin: нормировать окно его средним и разбросом.
        revin_min_valid: сколько валидных часов нужно для статистики окна; при меньшем
            числе нормировка не применяется.
    """
    arch: str = "patchtst"
    horizon: int = H
    quantiles: tuple = QUANTILES
    input_len: int = 504
    patch_len: int = 24
    stride: int = 24
    padding_patch: str = "none"
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    layers: int = 2
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
        """Число патчей окна, с учётом добавочного патча при дополнении ``end``."""
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
    """Конфиг архитектуры в виде датакласса.

    Args:
        arch: имя архитектуры.
        cfg: None, словарь или датакласс конфига; None значит значения по умолчанию.

    Returns:
        Датакласс конфига этой архитектуры.

    Raises:
        ConfigError: архитектура неизвестна или конфиг ей не подходит.
    """
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
    """Конфиг архитектуры по словарю.

    Args:
        d: словарь полей; ключ ``arch`` выбирает архитектуру, без него - МАЯК.

    Returns:
        Датакласс конфига.
    """
    d = dict(d or {})
    return model_config_for(d.get("arch", "mayak"), d)


def check_pipeline_compat(cfg):
    """Проверка, что модель совместима с контрактом данных.

    Горизонт и набор квантилей должны совпадать с контрактом. Модель, читающая историю
    целиком, должна ждать ровно максимальную длину истории. Модель с фиксированной
    длиной входа берёт последние часы окна, и её вход не может быть длиннее истории.

    Args:
        cfg: конфиг любой архитектуры.

    Returns:
        Тот же конфиг.

    Raises:
        ConfigError: конфиг расходится с контрактом данных.
    """
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
    "coords": "coords_prob", "scale": "scale_prob", "drift": "drift_prob",
    "offset": "offset_prob", "noise": "noise_prob", "spike": "spike_prob",
    "stuck": "stuck_prob", "units": "units_prob", "rh_dewpoint": "rh_dewpoint_prob",
    "dropout": "dropout_prob", "gap": "gap_prob", "outage": "outage_prob",
    "drop_pressure": "drop_pressure_prob", "drop_humidity": "drop_humidity_prob",
}


@dataclass(frozen=True)
class AugmentConfig:
    """Аугментации обучающих окон, имитирующие реальный прибор.

    Значения по умолчанию - профиль ``aggressive``: профиль обучения для переноса на
    реальные наблюдения. Именованные профили хранят только отличия от умолчаний и
    собираются методом ``from_profile``. Каждая аугментация включается со своей
    вероятностью и берёт параметры из своего диапазона. Тройки значений - по каналам
    температуры, давления и влажности; пары - нижняя и верхняя граница диапазона.

    Смещение, масштаб и дрейф - свойства самого датчика, поэтому искажают и историю, и
    цель. Шум искажает только историю. Запись температуры и влажности целыми числами -
    не аугментация: она применяется к каждому окну после всех аугментаций. Если окно
    искажается свойствами прибора или влажностью из точки росы, записанным значениям
    перед искажением возвращается непрерывность: к ним прибавляется равномерный шум в
    пределах полушага записи. Без искажений запись возвращает то же значение.

    Attributes:
        profile: имя профиля, от которого отсчитываются отличия.
        scale_prob: вероятность ошибки масштаба.
        scale_max: наибольшее отклонение множителя канала от единицы; фактическое
            отклонение случайно, от нуля до этого значения, в любую сторону.
        drift_prob: вероятность дрейфа.
        drift_rate_max: наибольшая скорость дрейфа по каналам: °C, гПа и % в сутки;
            фактическая скорость случайна, от нуля до этого значения, в любую сторону.
            Смещение растёт с этой скоростью от калибровки прибора через всю историю. На
            горизонте цели оно растёт дальше при истории не короче
            ``DRIFT_GROWTH_MIN_HISTORY`` из ``mayak.data.augment`` и держится на уровне
            последнего часа истории при более короткой.
        drift_age_max: наибольший возраст калибровки прибора к первому часу истории, ч;
            фактический возраст - целое число часов, равномерно от нуля до этого
            значения. Начало окна с калибровкой датчика не связано, поэтому смещение в
            первом часе истории не нулевое и от длины истории не зависит.
        drift_rw_frac: доля нарастания дрейфа на истории случайным блужданием,
            остальное - линейно; на горизонте нарастание всегда линейное.
        offset_prob: вероятность постоянного смещения температуры.
        offset_min: нижняя граница величины смещения. Величина выбирается равномерно в
            логарифме между границами, а при нулевой нижней границе - равномерно.
        offset_max: верхняя граница величины смещения; ноль выключает аугментацию.
        noise_prob: вероятность шума.
        noise_sd: разброс гауссова шума по каналам: °C, гПа, %.
        rh_dewpoint_prob: вероятность того, что влажность восстановлена из целых
            температуры и точки росы и прыгает на несколько процентов, как в
            наблюдениях реальной сети.
        spike_prob: вероятность одиночных выбросов в истории.
        spike_max_count: наибольшее число выбросов, не меньше одного.
        spike_min: наименьшая величина выброса по каналам.
        spike_max: наибольшая величина выброса по каналам.
        stuck_prob: вероятность залипания канала.
        stuck_hours: пределы длительности залипания, ч.
        units_prob: вероятность того, что вместо станционного давления пишется давление,
            приведённое к уровню моря. Температура всегда в градусах Цельсия: это
            обязанность владельца прибора.
        units_hours: пределы длительности такого участка, ч.
        dropout_prob: вероятность одиночных пропусков.
        dropout_max_rate: наибольшая доля пропущенных часов; фактическая доля
            случайна, от нуля до неё.
        gap_prob: вероятность блочных пропусков.
        gap_max_count: наибольшее число блочных пропусков.
        gap_max_len: наибольшая длина блочного пропуска, ч.
        outage_prob: вероятность выпадения канала в середине истории.
        outage_hours: пределы длительности выпадения, ч.
        drop_humidity_prob: вероятность того, что влажности нет на всей истории.
        drop_pressure_prob: вероятность того, что давления нет на всей истории.
        coords_prob: вероятность дрожания метаданных точки.
        coord_jitter_deg: наибольший сдвиг широты и долготы, градусы; сдвиг равномерный
            в обе стороны.
        elev_jitter_m: разброс нормального дрожания высоты, м.
    """
    profile: str = "aggressive"
    scale_prob: float = 0.3
    scale_max: tuple = (0.02, 0.001, 0.05)
    drift_prob: float = 0.3
    drift_rate_max: tuple = (0.05, 0.05, 0.2)
    drift_age_max: int = 336
    drift_rw_frac: float = 0.5
    offset_prob: float = 0.8
    offset_min: float = 0.1
    offset_max: float = 3.0
    noise_prob: float = 1.0
    noise_sd: tuple = (0.2, 0.3, 2.0)
    rh_dewpoint_prob: float = 0.1
    spike_prob: float = 0.2
    spike_max_count: int = 3
    spike_min: tuple = (12.0, 15.0, 40.0)
    spike_max: tuple = (30.0, 40.0, 80.0)
    stuck_prob: float = 0.15
    stuck_hours: tuple = (12, 96)
    units_prob: float = 0.05
    units_hours: tuple = (24, 96)
    dropout_prob: float = 0.3
    dropout_max_rate: float = 0.2
    gap_prob: float = 0.5
    gap_max_count: int = 3
    gap_max_len: int = 96
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
        for name in ("scale_max", "drift_rate_max", "noise_sd", "spike_min", "spike_max"):
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
        for name in (*AUGMENT_PROB_FIELDS.values(), "drift_rw_frac", "dropout_max_rate"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ConfigError(f"{name} = {v} вне [0, 1]")
            s(self, name, v)
        for name in ("spike_max_count", "gap_max_count", "gap_max_len"):
            v = int(getattr(self, name))
            if v < 1:
                raise ConfigError(f"{name} ≥ 1")
            s(self, name, v)
        if int(self.drift_age_max) < 0:
            raise ConfigError(f"drift_age_max = {self.drift_age_max}: возраст калибровки ≥ 0 ч")
        s(self, "drift_age_max", int(self.drift_age_max))
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
        """Конфиг из именованного профиля с явными переопределениями.

        Args:
            name: имя профиля.
            **overrides: поля, которые заменяют значения профиля.

        Returns:
            Конфиг аугментаций.

        Raises:
            ConfigError: профиль неизвестен или среди полей есть лишние.
        """
        if name not in AUGMENT_PROFILES:
            raise ConfigError(f"неизвестный профиль аугментаций {name!r}; "
                              f"есть {tuple(AUGMENT_PROFILES)}")
        kw = {**AUGMENT_PROFILES[name], **overrides}
        kw.pop("profile", None)
        return cls(**_strict_kwargs(cls, {"profile": name, **kw}, "data.augment"))

    @classmethod
    def from_dict(cls, d=None):
        """Конфиг из словаря: сначала профиль, потом остальные ключи поверх него.

        Args:
            d: словарь полей; ключ ``profile`` выбирает профиль, без него -
                ``aggressive``. None - профиль по умолчанию.

        Returns:
            Конфиг аугментаций.

        Raises:
            ConfigError: неизвестный ключ или ключ прежнего дрейфа.
        """
        if d is not None and "drift_max" in d:
            raise ConfigError("data.augment.drift_max больше не поддерживается: дрейф задаётся "
                              "скоростью в сутки (drift_rate_max), а не смещением в момент "
                              "выпуска. Конфиг с этим ключом принадлежит прогону, обученному с "
                              "прежним дрейфом; переобучите модель")
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
        """Поля, которые отличаются от объявленного профиля.

        Returns:
            Словарь: имя поля и пара значений - в профиле и фактическое.
        """
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
        drift_prob=0.15, drift_rate_max=(0.025, 0.035, 0.1),
        offset_max=1.5,
        noise_sd=(0.15, 0.2, 1.5),
        rh_dewpoint_prob=0.05,
        spike_prob=0.05, spike_max_count=1,
        stuck_prob=0.05, stuck_hours=(6, 48),
        units_prob=0.01,
        dropout_prob=0.2, dropout_max_rate=0.1,
        gap_prob=0.3, gap_max_count=2, gap_max_len=48,
        outage_prob=0.1, outage_hours=(24, 120),
        drop_humidity_prob=0.1, drop_pressure_prob=0.1,
        coord_jitter_deg=0.2, elev_jitter_m=20.0),
    "base": dict(
        scale_prob=0.0, drift_prob=0.0,
        offset_prob=1.0, offset_min=0.0, offset_max=0.7,
        noise_prob=1.0, noise_sd=(0.2, 0.5, 2.0), rh_dewpoint_prob=0.0,
        spike_prob=0.0, stuck_prob=0.0, units_prob=0.0,
        dropout_prob=0.0, gap_prob=0.3, gap_max_count=1, gap_max_len=24,
        outage_prob=0.0,
        drop_humidity_prob=0.1, drop_pressure_prob=0.1,
        coords_prob=1.0, coord_jitter_deg=0.4, elev_jitter_m=0.0),
    "none": {f: 0.0 for f in AUGMENT_PROB_FIELDS.values()},
}

ZONE_WEIGHTINGS = {"uniform": 0.0, "inv_sqrt": 0.5, "inv": 1.0}


@dataclass(frozen=True)
class DataConfig:
    """Данные прогона: источник, правила окон, аугментации, набор валидации.

    Attributes:
        manifest: путь к манифесту станций.
        cache_root: каталог кэша; None - рядом с манифестом.
        time_layout: раскладка ряда; должна совпадать с раскладкой в коде.
        target_mask: правило годности цели окна.
        val_every_hours: шаг между кандидатами в начала горизонта валидации, ч.
        val_windows_per_station: сколько окон валидации берётся с каждой станции; станция
            с меньшим числом кандидатов отдаёт все свои. Этим числом задаётся размер
            набора валидации, проход валидации идёт по всему набору.
        val_seed: сид выбора длины истории окон валидации. Не зависит от сидов протокола,
            поэтому набор одинаков у всех архитектур и повторов.
        augment: аугментации обучающих окон.
        window_qc: причинный QC истории обучающих окон.
        zone_weighting: вес станции при сэмплировании по её зоне.
        zone_weight_cap: верхняя граница веса станции относительно среднего; 0 - без неё.
    """
    manifest: str = "data/manifest.csv"
    cache_root: Optional[str] = None
    time_layout: dict = field(default_factory=lambda: dict(TIME_LAYOUT))
    target_mask: TargetMaskConfig = TargetMaskConfig()
    val_every_hours: int = 24
    val_windows_per_station: int = 96
    val_seed: int = 0
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
        """Конфиг данных из словаря.

        Args:
            d: словарь с полями конфига; None - значения по умолчанию.

        Returns:
            Конфиг данных.

        Raises:
            ConfigError: неизвестный ключ или ключ прежнего набора валидации.
        """
        if d is not None and "val_max_windows" in d:
            raise ConfigError("data.val_max_windows больше не поддерживается: размер набора "
                              "валидации задаётся числом окон на станцию "
                              "(val_windows_per_station). Конфиг с этим ключом принадлежит "
                              "прогону, выбранному по прежней валидации; переобучите модель")
        return cls(**_strict_kwargs(cls, d, "data"))


@dataclass(frozen=True)
class RunConfig:
    """Полная конфигурация прогона: модель, данные и протокол обучения.

    Attributes:
        model: конфиг архитектуры.
        data: конфиг данных.
        train: протокол обучения.
    """
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
        """Конфигурация, где эффекты абляций перенесены между секциями.

        Абляция без смещения температуры выключает аугментацию смещения в данных.
        Повторный вызов ничего не меняет.

        Returns:
            Новая полная конфигурация.
        """
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
    """Короткое имя варианта модели для таблиц.

    Args:
        model_cfg: конфиг архитектуры.

    Returns:
        Имя архитектуры, а при включённых абляциях - ещё их имена через дефис.
    """
    abl = getattr(model_cfg, "ablations", None)
    act = abl.active() if abl is not None else ()
    return model_cfg.arch + ("" if not act else "-" + "+".join(act))


SCENARIO_INSTRUMENT, SCENARIO_INPUT = "instrument", "input"
ROBUSTNESS_QC = ("none", "device")
ROBUSTNESS_TIME_KEYS = ("val", "test")


@dataclass(frozen=True)
class ScenarioRule:
    """Контракт сценария робастности.

    Attributes:
        kind: ``instrument`` - свойство прибора, или ``input`` - отказ входа.
        target: искажается ли и цель. У свойства прибора - да, кроме вариантов
            «незамеченное смещение прибора», где искажён только вход.
        title: название сценария для отчётов.
        unit: единица уровня деградации для подписей.
        max_level: верхняя граница уровня деградации; None - без границы.
        integer: уровень - целое число: часы или номер варианта.
        params: фиксированные параметры сценария и их значения по умолчанию.
        guard: участвует ли сценарий в проверке скилла.
        variant_of: у варианта «искажён только вход» - имя основного сценария.
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
    """Один сценарий робастности из конфига.

    Attributes:
        name: имя сценария из таблицы правил сценариев.
        levels: уровни деградации по возрастанию. Первый обязан быть нулём: сценарий с
            нулевым параметром не меняет данные и служит точкой отсчёта кривой.
        params: переопределения фиксированных параметров; остальные - по умолчанию.
        guard: участвует ли сценарий в проверке скилла; None - как задано правилом
            сценария.
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

    Attributes:
        roles: роли станций внутреннего набора. Внешний тест всегда берёт станции
            внешнего теста в тестовом окне.
        time_key: временное окно внутреннего набора.
        every_hours: шаг между кандидатами в моменты выпуска, ч.
        windows_per_station: сколько окон берётся с каждой станции.
        qc: контроль качества после сценария: ``device`` - причинный QC прибора, общий
            с обучением и оценкой; ``none`` - только маска наличия.
        leads: лиды кривых и проверки скилла.
        skill_tolerance: допуск проверки скилла охраняемых моделей в сценариях с
            проверкой. В отказе входа порог отсчитывается от холодного старта модели,
            если он хуже климатологии, иначе от климатологии; в свойстве прибора - от
            климатологии того же прибора.
        guard_models: модели, скилл которых проверяется.
        seed: сид случайных чисел сценариев, общий для всех уровней.
        bootstrap: число повторов блочного бутстрапа по станциям; 0 - без интервалов.
        ci_level: уровень интервалов бутстрапа.
        scenarios: сценарии и их уровни.
    """
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
        """Конфиг только с выбранными сценариями.

        Args:
            names: имена сценариев; порядок берётся из конфига.

        Returns:
            Новый конфиг.

        Raises:
            ConfigError: какого-то имени нет в конфиге.
        """
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


COVERAGE_DIMS_INTERNAL = ("роль станции", "зона Кёппена", "длина истории", "валидность истории")
COVERAGE_DIMS_EXTERNAL = ("зона Кёппена", "длина истории", "валидность истории",
                          "расстояние до обучающей точки", "Δ высоты станция−ЦМР",
                          "канал давления")
COVERAGE_DIMS = tuple(dict.fromkeys(COVERAGE_DIMS_INTERNAL + COVERAGE_DIMS_EXTERNAL))
CALIBRATION_NOMINALS = (0.8, 0.9)


@dataclass(frozen=True)
class CalibrationConfig:
    """Подгонка конформной таблицы, анализ калибровки и адаптивная калибровка на устройстве.

    Attributes:
        nominal: номинал центрального интервала, покрытие которого разбирается.
        tolerance: существенное отклонение покрытия, доля; 0.04 даёт полосу 86-94 %.
        min_windows: страты с меньшим числом окон не показываются.
        min_stations: страты с меньшим числом станций не показываются.
        bootstrap: число повторов блочного бутстрапа по станциям; 0 - без интервалов,
            тогда вердикты опираются только на допуск.
        ci_level: уровень интервалов бутстрапа.
        seed: сид бутстрапа.
        conditional_dims: разрезы, по которым допустима условная конформная поправка. Она
            рекомендуется, только если страты этих разрезов систематически отличаются от
            покрытия набора в целом.
        sharpness_range: крайние множители ширины для кривой «острота против покрытия».
        sharpness_points: число множителей на этой кривой.
        aci_gamma: шаг адаптивной калибровки устройства.
        aci_max_factor: граница множителя ширины адаптивной калибровки; целевая доля
            промахов равна единице минус номинал.
        aci_stations: на скольких тестовых станциях идёт офлайн-прогон адаптивной
            калибровки с ежечасным выпуском.
        aci_hours: длина непрерывного периода этого прогона на каждой станции, ч.
        fit_every_hours: шаг между кандидатами в начала горизонта калибровочного набора, ч.
        fit_windows_per_station: сколько окон калибровочного набора берётся с каждой
            станции; None - все окна.
        fit_seed: сид выбора длины истории окон калибровочного набора по куррикулуму.
        fit_min_windows: наименьшее число окон бина длины истории, при котором у бина в
            конформной таблице своя строка; иначе строка маргинальная.
    """
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
    """Параметры хоста устройства, которые не относятся к модели.

    Состояние на диске помнит координаты и высоту, для которых оно записано. Если при
    перезапуске они отличаются от текущих не больше порогов, это уточнение метаданных:
    окно сохраняется, прогноз строится уже для новой точки. Больше любого порога - прибор
    перенесён, окно и калибровка начинаются заново. Пороги взяты по порядку дрожания
    метаданных в аугментациях обучения.

    Attributes:
        site_max_dlat_deg: наибольшая разница широты для уточнения, градусы.
        site_max_dlon_deg: наибольшая разница долготы по кратчайшей дуге для
            уточнения, градусы.
        site_max_delev_m: наибольшая разница высоты для уточнения, метры.
    """
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
           "LRU_SCANS", "MODEL_CONFIGS", "ModeGroup", "ModelConfig", "PERSISTENT_GROUP",
           "PatchTSTConfig",
           "DEFAULT_SCENARIOS", "ROBUSTNESS_QC", "RobustnessConfig", "RunConfig", "RuntimeConfig",
           "SCENARIO_INPUT", "SCENARIO_INSTRUMENT", "SCENARIO_RULES", "SOLAR_CHANNELS",
           "ScenarioRule", "ScenarioSpec", "Seeds", "TrainConfig",
           "check_pipeline_compat", "default_scenarios", "model_config_for",
           "model_config_from_dict", "run_label",
           "to_jsonable"]
