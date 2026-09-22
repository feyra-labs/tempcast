from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np

from mayak.constants import H, QUANTILES

Q = np.array(QUANTILES, np.float32)
NQ = len(QUANTILES)
I_LO90, I_LO80, I_MED, I_HI80, I_HI90 = 0, 1, 3, 5, 6
EPS = 1e-9

LEAD_BINS = ((1, 6), (7, 24), (25, 72), (73, 168))
FINE_LEADS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168)

METRICS = ("MAE", "RMSE", "Skill", "CRPS", "PICP80", "PICP90", "Winkler90", "Width90")
_MEAN_METRIC = {"ae": "MAE", "crps": "CRPS", "cov80": "PICP80", "cov90": "PICP90",
                "wink90": "Winkler90", "width90": "Width90"}
_TERMS = ("ae", "se", "se_clim", "crps", "cov80", "cov90", "wink90", "width90")


def _central_intervals():
    """Симметричные центральные интервалы из набора квантилей: (номинал, i_lo, i_hi).\n"""
    out = []
    for i in range(NQ // 2):
        j = NQ - 1 - i
        if abs(float(Q[i]) + float(Q[j]) - 1.0) < 1e-6:
            out.append((round(float(Q[j]) - float(Q[i]), 6), i, j))
    return tuple(out)


CENTRAL_INTERVALS = _central_intervals()


def interval_indices(nominal):
    """Номинал центрального интервала → (i_lo, i_hi) в наборе квантилей."""
    for nom, i, j in CENTRAL_INTERVALS:
        if abs(nom - float(nominal)) < 1e-6:
            return i, j
    raise ValueError(f"номинал {nominal} не является центральным интервалом набора квантилей; "
                     f"есть {[nom for nom, _i, _j in CENTRAL_INTERVALS]}")


def wmean(x, w, axis=None):
    """Среднее с весами маски: нормировка на сумму весов, а не на число элементов."""
    x = np.asarray(x, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), x.shape)
    num = np.where(w > 0, x * w, 0.0).sum(axis=axis)
    den = w.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def pinball_crps(y, q):
    err = y[..., None] - q
    pin = np.maximum(Q * err, (Q - 1) * err)
    return 2.0 * pin.mean(axis=-1)


def inside(y, lo, hi):
    return ((y >= lo) & (y <= hi)).astype(np.float64)


def winkler(y, lo, hi, alpha):
    return ((hi - lo)
            + (2 / alpha) * (lo - y) * (y < lo)
            + (2 / alpha) * (y - hi) * (y > hi))


def pair_terms(y, mu, q, mu_clim):
    """Все числители метрик на парах «окно × лид»: (N, H) массивы."""
    y = np.asarray(y, np.float64)
    mu = np.asarray(mu, np.float64)
    q = np.asarray(q, np.float64)
    mu_clim = np.asarray(mu_clim, np.float64)
    e = mu - y
    lo90, hi90 = q[..., I_LO90], q[..., I_HI90]
    return dict(
        ae=np.abs(e), se=e ** 2, se_clim=(mu_clim - y) ** 2,
        crps=pinball_crps(y, q),
        cov80=inside(y, q[..., I_LO80], q[..., I_HI80]),
        cov90=inside(y, lo90, hi90),
        wink90=winkler(y, lo90, hi90, 0.10),
        width90=hi90 - lo90,
    )


def _metrics_from_sums(sums, den):
    """Взвешенные суммы слагаемых и сумма весов → словарь метрик."""
    den = np.asarray(den, np.float64)
    ok = den > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {k: np.where(ok, np.asarray(v, np.float64) / np.where(ok, den, 1.0), np.nan)
                for k, v in sums.items()}
        out = {name: mean[k] for k, name in _MEAN_METRIC.items()}
        out["RMSE"] = np.sqrt(mean["se"])
        sc = np.asarray(sums["se_clim"], np.float64)
        good = ok & (sc > EPS)
        out["Skill"] = np.where(good,
                                1.0 - np.asarray(sums["se"], np.float64) / np.where(good, sc, 1.0),
                                np.nan)
    return out


def _scalar_metrics(sums, den):
    m = _metrics_from_sums({k: float(v) for k, v in sums.items()}, float(den))
    return {k: float(m[k]) for k in METRICS}


def lead_mask(leads, horizon=H):
    """Спецификация лидов → булева маска длины ``horizon``."""
    if leads is None:
        return None
    a = np.asarray(leads)
    if a.dtype == bool:
        if a.shape != (horizon,):
            raise ValueError(f"булева маска лидов длины {a.shape}, нужно ({horizon},)")
        return a
    m = np.zeros(horizon, bool)
    idx = a.astype(np.int64) - 1
    if idx.size and (idx.min() < 0 or idx.max() >= horizon):
        raise ValueError(f"лиды вне [1, {horizon}]: {a.min()}..{a.max()}")
    m[idx] = True
    return m


def lead_bin_index(h1, lead_bins=LEAD_BINS):
    """Час лида (1-based) → индекс бина лидов."""
    for i, (a, b) in enumerate(lead_bins):
        if a <= h1 <= b:
            return i
    return len(lead_bins) - 1


def lead_bin_of(horizon=H, lead_bins=LEAD_BINS):
    """Вектор (horizon,) с индексом бина для каждого лида."""
    return np.array([lead_bin_index(h + 1, lead_bins) for h in range(horizon)], np.int64)


def conformal_table(shift, horizon=H, lead_bins=LEAD_BINS):
    """Таблица поправок по бинам → поправка на каждый лид, (horizon, NQ)."""
    shift = np.asarray(shift, np.float32)
    if shift.ndim != 2 or shift.shape[1] != NQ:
        raise ValueError(f"таблица поправок формы {shift.shape}, нужно (бины, {NQ})")
    if shift.shape[0] != len(lead_bins):
        raise ValueError(f"таблица поправок на {shift.shape[0]} бинов, "
                         f"а бинов лидов {len(lead_bins)}")
    return shift[lead_bin_of(horizon, lead_bins)]


def apply_conformal(q, shift, lead_bins=LEAD_BINS):
    """Сдвиг квантилей конформной поправкой с восстановлением монотонности."""
    q = np.asarray(q, np.float32)
    table = conformal_table(shift, q.shape[-2], lead_bins)
    return np.maximum.accumulate(q + table, axis=-1)


def apply_adaptive(q, theta=0.0):
    """Адаптивная поправка: все квантили растягиваются вокруг медианы в e^θ раз.

    Возвращает (q, mu), mu - медиана поправленных квантилей. Медиана поправкой не
    меняется; при θ = 0 квантили возвращаются как есть, без арифметики (пакет и поток
    совпадают до бита). Растяжение с положительным множителем сохраняет порядок;
    ``maximum.accumulate`` - защита для входа, который уже был немонотонным.
    """
    theta = float(theta)
    if not math.isfinite(theta):
        raise ValueError(f"θ адаптивной поправки не конечно: {theta}")
    q = np.asarray(q, np.float32)
    if theta != 0.0:
        med = q[..., I_MED:I_MED + 1]
        q = np.maximum.accumulate(med + np.float32(math.exp(theta)) * (q - med), axis=-1)
    return q, q[..., I_MED].copy()


def calibrate_forecast(q, shift=None, theta=0.0, lead_bins=LEAD_BINS):
    """Единственная точка применения калибровки в проекте → (q, mu).

    Порядок фиксирован: сплит-конформная таблица по бинам лидов (подогнана офлайн на
    калибровочном окне), затем адаптивный множитель e^θ (подстраивается онлайн на
    устройстве). Медиана берётся из итоговых квантилей. Оценка, графики и рантайм
    вызывают эту функцию (рантайм - те же две ступени по отдельности, потому что ему
    нужен промежуточный результат для обратной связи).
    """
    if shift is not None:
        q = apply_conformal(q, shift, lead_bins)
    return apply_adaptive(q, theta)


def aci_score(y, q, interval=(I_LO90, I_HI90)):
    """Нормированный выход факта за интервал: 0 - на медиане, 1 - ровно на границе.

    Для y выше медианы - (y − med) / (q_hi − med), ниже - (med − y) / (med − q_lo).
    Факт внутри интервала, растянутого ``apply_adaptive`` в e^θ раз, тогда и только
    тогда, когда оценка ≤ e^θ. Вырожденная половина интервала (нулевая ширина) даёт
    +inf для любого факта, кроме самой медианы.
    """
    i, j = interval
    y = np.asarray(y, np.float64)
    q = np.asarray(q, np.float64)
    med = q[..., I_MED]
    u = y - med
    d = np.where(u >= 0, q[..., j] - med, med - q[..., i])
    au = np.abs(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d > 0, au / np.where(d > 0, d, 1.0), np.inf)
    s = np.where(au == 0, 0.0, s)
    return np.where(np.isfinite(y) & np.isfinite(med), s, np.nan)


def _f32(x):
    """Округление до float32: θ хранится в состоянии рантайма как float32, и онлайн-путь
    с перезапусками совпадает с непрерывным до бита."""
    return float(np.float32(x))


@dataclass(frozen=True)
class ACIParams:
    """Адаптивная конформная калибровка на устройстве.

    Источник: Gibbs, Candès, «Adaptive Conformal Inference Under Distribution Shift»,
    NeurIPS 2021, arXiv:2106.00170 - онлайн-подстройка по ошибкам покрытия
    err_t ∈ {0, 1} с малым шагом γ: параметр сдвигается на γ·(err_t − α).

    Что взято: правило обновления и его гарантия. Сумма обновлений телескопируется:
    θ_T − θ_0 = γ·Σ(err_t − α), поэтому средняя доля промахов на любом потоке отличается
    от α не больше чем на |θ_T − θ_0| / (γ·T) - без предположений о распределении.

    Отличие от оригинала: подстраивается не номинальный уровень α_t, а логарифм
    множителя ширины интервала θ (интервал растягивается вокруг медианы в e^θ раз;
    ``apply_adaptive``). Это та же схема в форме отслеживания квантиля оценки
    (quantile tracking; Angelopoulos, Candès, Tibshirani, «Conformal PID Control for
    Time Series Prediction», NeurIPS 2023, arXiv:2307.16895). Причина: у модели семь
    квантилей, уровни вне [5 %, 95 %] ей недоступны, и при α_t ≤ 0 оригинал требует
    бесконечного интервала. В θ-форме каждый промах расширяет интервал на один и тот же
    относительный шаг независимо от текущего уровня, а само состояние - одно число.

    Границы: θ ∈ [−ln f, ln f], f = ``max_factor``. Нужны, чтобы поток сплошных
    промахов (отказ прибора, подмена единиц) не разгонял θ без предела. Пока граница не
    достигнута, гарантия выше выполняется точно; упоры в границу считаются и
    показываются в отчётах.

    target - α, целевая доля промахов центрального интервала уровня 1 − α (он должен
    быть в наборе квантилей); gamma - шаг γ (0.005 - значение из статьи Gibbs, Candès).
    """
    target: float = 0.10
    gamma: float = 0.005
    max_factor: float = 4.0

    def __post_init__(self):
        object.__setattr__(self, "target", float(self.target))
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "max_factor", float(self.max_factor))
        interval_indices(1.0 - self.target)
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"шаг ACI γ = {self.gamma} вне (0, 1)")
        if not (math.isfinite(self.max_factor) and self.max_factor > 1.0):
            raise ValueError(f"max_factor = {self.max_factor}: нужен конечный множитель > 1")

    @property
    def interval(self):
        return interval_indices(1.0 - self.target)

    @property
    def theta_min(self):
        return _f32(-math.log(self.max_factor))

    @property
    def theta_max(self):
        return _f32(math.log(self.max_factor))

    def clip(self, theta):
        return min(max(_f32(theta), self.theta_min), self.theta_max)

    def update(self, theta, miss):
        """θ после одного наблюдения: θ + γ·(err − α), с упором в границы."""
        return self.clip(float(theta) + self.gamma * (float(miss) - self.target))

    def step(self, theta, score):
        """Одна обратная связь: (новое θ, был ли промах) по оценке ``aci_score``."""
        miss = bool(score > math.exp(theta))
        return self.update(theta, miss), miss


def aci_run(scores, params, theta0=0.0):
    """Прогон ACI по одному потоку оценок в порядке времени.

    NaN - обратной связи нет (факт невалиден): θ не меняется. Возвращает θ до каждого
    наблюдения (им и выпущен интервал), промахи (NaN там, где связи нет), итоговое θ и
    число упоров в границы.
    """
    scores = np.asarray(scores, np.float64).ravel()
    theta = params.clip(theta0)
    before = np.empty(len(scores), np.float64)
    miss = np.full(len(scores), np.nan)
    clipped = 0
    lo, hi = params.theta_min, params.theta_max
    for k, s in enumerate(scores.tolist()):
        before[k] = theta
        if s != s:
            continue
        theta, m = params.step(theta, s)
        miss[k] = m
        clipped += theta in (lo, hi)
    return dict(theta=before, miss=miss, theta_end=theta, clipped=int(clipped))


def aci_effective_level(theta, target=0.10):
    """Номинал, которому соответствует растянутый интервал при нормальной форме прогноза:
    интервал уровня 1 − α, растянутый в e^θ раз, - это уровень 2Φ(z·e^θ) − 1."""
    nd = NormalDist()
    z = nd.inv_cdf(1.0 - float(target) / 2.0)
    return 2.0 * nd.cdf(z * math.exp(float(theta))) - 1.0


SHARPNESS_RANGE = (0.25, 4.0)
SHARPNESS_POINTS = 33


def sharpness_scales(lo=SHARPNESS_RANGE[0], hi=SHARPNESS_RANGE[1], n=SHARPNESS_POINTS):
    """Логарифмическая сетка множителей ширины, всегда содержит 1 (выход модели)."""
    s = np.exp(np.linspace(math.log(lo), math.log(hi), int(n)))
    return np.unique(np.concatenate([s, [1.0]]))


def width_at_coverage(coverage, width, target):
    """Ширина, при которой фактическое покрытие достигает target (линейная интерполяция
    по кривой, упорядоченной по множителю). NaN, если кривая target не достигает."""
    coverage = np.asarray(coverage, np.float64)
    width = np.asarray(width, np.float64)
    ok = np.isfinite(coverage) & np.isfinite(width)
    coverage, width = coverage[ok], width[ok]
    idx = np.flatnonzero(coverage >= target)
    if coverage.size == 0 or idx.size == 0 or coverage[0] > target:
        return float("nan")
    k = int(idx[0])
    if k == 0 or coverage[k] == coverage[k - 1]:
        return float(width[k])
    f = (target - coverage[k - 1]) / (coverage[k] - coverage[k - 1])
    return float(width[k - 1] + f * (width[k] - width[k - 1]))


def fit_conformal_shift(y, q, w, lead_bins=LEAD_BINS):
    """Сплит-конформные поправки по бинам лидов: квантиль остатка на валидных часах."""
    shift = np.zeros((len(lead_bins), q.shape[-1]), np.float32)
    for bi, (a, b) in enumerate(lead_bins):
        sl = slice(a - 1, b)
        resid = (y[:, sl, None] - q[:, sl, :]).reshape(-1, q.shape[-1])
        resid = resid[np.asarray(w)[:, sl].reshape(-1) > 0]
        if len(resid) == 0:
            raise ValueError(f"в бине лидов {a}-{b} нет ни одного валидного часа")
        for qi, tau in enumerate(QUANTILES):
            shift[bi, qi] = np.quantile(resid[:, qi], tau)
    return shift


def quantile_ci(samples, level=0.90):
    """{метрика: выборка} → {метрика: (нижняя, верхняя)} центральный интервал уровня level."""
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    out = {}
    for k, v in samples.items():
        v = np.asarray(v, np.float64)
        v = v[np.isfinite(v)]
        out[k] = ((float(np.quantile(v, lo)), float(np.quantile(v, hi)))
                  if v.size else (float("nan"), float("nan")))
    return out


@dataclass(frozen=True)
class Evaluation:
    """Предсказания одной модели на наборе окон плюс веса и привязка к станциям."""
    y: np.ndarray
    mu: np.ndarray
    q: np.ndarray
    mu_clim: np.ndarray
    w: np.ndarray
    station: np.ndarray
    _terms: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        n, h = self.y.shape
        for name, a, shape in (("mu", self.mu, (n, h)), ("mu_clim", self.mu_clim, (n, h)),
                               ("w", self.w, (n, h)), ("q", self.q, (n, h, NQ)),
                               ("station", self.station, (n,))):
            if tuple(np.shape(a)) != shape:
                raise ValueError(f"{name}: форма {np.shape(a)}, ожидалась {shape}")
        object.__setattr__(self, "w", (np.asarray(self.w, np.float64) > 0).astype(np.float64))
        object.__setattr__(self, "station", np.asarray(self.station))
        if self._terms is None:
            object.__setattr__(self, "_terms", pair_terms(self.y, self.mu, self.q, self.mu_clim))

    @property
    def horizon(self):
        return self.y.shape[1]

    def restrict(self, windows=None, leads=None):
        """Подвыборка окон и/или лидов. Возвращает новую оценку с теми же данными."""
        w = self.w
        if windows is not None:
            sel = np.asarray(windows)
            if sel.dtype != bool:
                mask = np.zeros(len(self.y), bool)
                mask[sel.astype(np.int64)] = True
                sel = mask
            w = w * sel[:, None]
        lm = lead_mask(leads, self.horizon)
        if lm is not None:
            w = w * lm[None, :]
        return replace(self, w=w)

    def with_calibration(self, shift=None, theta=0.0, lead_bins=LEAD_BINS):
        """Оценка после калибровки ``calibrate_forecast``: медиана берётся из квантилей."""
        if shift is None and float(theta) == 0.0:
            return self
        q, mu = calibrate_forecast(self.q, shift, theta, lead_bins)
        return Evaluation(y=self.y, mu=mu, q=q, mu_clim=self.mu_clim,
                          w=self.w, station=self.station)

    def with_conformal(self, shift, lead_bins=LEAD_BINS):
        """Оценка с применённой конформной поправкой: медиана берётся из квантилей."""
        return self.with_calibration(shift, 0.0, lead_bins)


    def counts(self):
        used = self.w > 0
        win = used.any(1)
        return dict(n_windows=int(win.sum()),
                    n_stations=int(len(np.unique(self.station[win]))) if win.any() else 0,
                    n_pairs=int(used.sum()))

    def station_sums(self):
        """(станции, суммы слагаемых по станциям, суммы весов). Основа макро и бутстрапа."""
        st, inv = np.unique(self.station, return_inverse=True)
        n = len(st)
        w = self.w
        den = np.bincount(inv, weights=w.sum(1), minlength=n)
        sums = {k: np.bincount(inv, weights=np.where(w > 0, v * w, 0.0).sum(1), minlength=n)
                for k, v in self._terms.items()}
        return st, sums, den

    def pooled(self):
        """Метрики по всем валидным парам сразу."""
        _st, sums, den = self.station_sums()
        return _scalar_metrics({k: v.sum() for k, v in sums.items()}, den.sum())

    def per_station(self):
        """(станции, {метрика: массив по станциям}) - станции без валидных пар отброшены."""
        st, sums, den = self.station_sums()
        keep = den > 0
        m = _metrics_from_sums({k: v[keep] for k, v in sums.items()}, den[keep])
        return st[keep], {k: np.asarray(m[k], np.float64) for k in METRICS}

    def macro(self):
        """Метрики на каждой станции отдельно, затем простое среднее по станциям."""
        _st, per = self.per_station()
        with np.errstate(invalid="ignore"):
            return {k: (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"))
                    for k, v in per.items()}

    def bootstrap_samples(self, n_boot=1000, seed=0):
        """Выборки блочного бутстрапа по станциям: (пуловые, макро) {метрика: (n_boot,)}. """
        st, sums, den = self.station_sums()
        keep = den > 0
        st, den = st[keep], den[keep]
        sums = {k: v[keep] for k, v in sums.items()}
        n = len(st)
        if n == 0:
            return None
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=int(n_boot)).astype(np.float64)

        boot_pooled = _metrics_from_sums({k: counts @ v for k, v in sums.items()}, counts @ den)
        per = _metrics_from_sums(sums, den)
        boot_macro = {}
        for k in METRICS:
            v = np.asarray(per[k], np.float64)
            good = np.isfinite(v)
            num = counts @ np.where(good, v, 0.0)
            cnt = counts @ good.astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_macro[k] = np.where(cnt > 0, num / np.where(cnt > 0, cnt, 1.0), np.nan)
        return ({k: np.asarray(boot_pooled[k], np.float64) for k in METRICS}, boot_macro)

    def bootstrap_ci(self, n_boot=1000, seed=0, level=0.90):
        samples = self.bootstrap_samples(n_boot=n_boot, seed=seed)
        empty = {k: (float("nan"), float("nan")) for k in METRICS}
        if samples is None:
            return dict(pooled=empty, macro=empty, n_boot=0, level=level, n_stations=0)
        boot_pooled, boot_macro = samples
        n = int((self.station_sums()[2] > 0).sum())
        return dict(pooled=quantile_ci(boot_pooled, level), macro=quantile_ci(boot_macro, level),
                    n_boot=int(n_boot), level=float(level), n_stations=n)

    def summary(self, ci=False, n_boot=1000, seed=0, level=0.90):
        """Пуловая и макро-метрики, число окон и станций, при ``ci`` - интервалы."""
        out = dict(pooled=self.pooled(), macro=self.macro(), **self.counts())
        if ci:
            out["ci"] = self.bootstrap_ci(n_boot=n_boot, seed=seed, level=level)
        return out

    def pit_histogram(self):
        """Гистограмма PIT по бинам, задаваемым квантилями: наблюдаемые доли и ожидаемые."""
        b = (self.y[..., None] >= self.q).sum(-1)  # 0..NQ
        w = self.w
        tot = w.sum()
        obs = np.array([np.where((b == i) & (w > 0), w, 0.0).sum() for i in range(NQ + 1)])
        exp = np.diff(np.concatenate([[0.0], np.asarray(Q, np.float64), [1.0]]))
        return dict(observed=(obs / tot if tot > 0 else obs * np.nan), expected=exp,
                    n=int((w > 0).sum()))

    def pit_by_lead_bin(self, lead_bins=LEAD_BINS):
        return {f"{a}-{b}": self.restrict(leads=np.arange(a, b + 1)).pit_histogram()
                for a, b in lead_bins}

    def reliability(self):
        """Диаграмма надёжности: номинальный уровень против фактического для всех квантилей."""
        emp = np.array([float(wmean((self.y <= self.q[..., i]).astype(np.float64), self.w))
                        for i in range(NQ)])
        return dict(nominal=np.asarray(Q, np.float64), empirical=emp)

    def sharpness_coverage(self):
        """Острота против покрытия: средняя ширина интервала и фактическое покрытие.

        below / above - доли факта ниже нижней и выше верхней границы: при одинаковом
        покрытии они различают «узкий интервал» (промахи с обеих сторон) и «сдвиг»
        (промахи с одной стороны).
        """
        rows = []
        for nominal, i, j in CENTRAL_INTERVALS:
            lo, hi = self.q[..., i], self.q[..., j]
            rows.append(dict(nominal=float(nominal),
                             coverage=float(wmean(inside(self.y, lo, hi), self.w)),
                             width=float(wmean(hi - lo, self.w)),
                             below=float(wmean((self.y < lo).astype(np.float64), self.w)),
                             above=float(wmean((self.y > hi).astype(np.float64), self.w))))
        return rows

    def sharpness_curve(self, scales=None, nominal=0.9, lead_bins=LEAD_BINS):
        """Кривая «острота против покрытия» по уже собранным предсказаниям.

        Интервал уровня ``nominal`` растягивается вокруг медианы в s раз для каждого s из
        ``scales`` (та же операция, что ``apply_adaptive``) - получается непрерывная кривая
        «фактическое покрытие → средняя ширина». s = 1 - выход модели как есть. Кривые
        разных моделей сравниваются при одинаковом фактическом покрытии
        (``width_at_coverage``), а не при одинаковом номинале. Возвращает
        {"весь горизонт" | "a-b": dict(scale, coverage, width)}.
        """
        scales = sharpness_scales() if scales is None else np.asarray(scales, np.float64)
        i, j = interval_indices(nominal)
        panels = {"весь горизонт": self.w}
        for a, b in lead_bins:
            panels[f"{a}-{b}"] = self.w * lead_mask(np.arange(a, b + 1), self.horizon)[None, :]
        out = {k: dict(scale=scales.copy(), coverage=np.full(len(scales), np.nan),
                       width=np.full(len(scales), np.nan)) for k in panels}
        for n, sc in enumerate(scales):
            qs, _ = apply_adaptive(self.q, math.log(sc))
            lo, hi = qs[..., i].astype(np.float64), qs[..., j].astype(np.float64)
            cov, wid = inside(self.y, lo, hi), hi - lo
            for k, w in panels.items():
                out[k]["coverage"][n] = float(wmean(cov, w))
                out[k]["width"][n] = float(wmean(wid, w))
        return out


def by_lead(ev, leads=FINE_LEADS, ci=False, **kw):
    """Таблица по лидам: {час лида: сводка}. Знаменатель скилла - на том же лиде."""
    return {int(h): ev.restrict(leads=[h]).summary(ci=ci, **kw) for h in leads}


def by_lead_bin(ev, lead_bins=LEAD_BINS, ci=False, **kw):
    return {f"{a}-{b}": ev.restrict(leads=np.arange(a, b + 1)).summary(ci=ci, **kw)
            for a, b in lead_bins}


def breakdown(ev, keys, leads=None, min_windows=20, min_stations=2, ci=False, **kw):
    """Разрез по меткам окон ``keys`` (N,)."""
    keys = np.asarray(keys)
    if len(keys) != len(ev.y):
        raise ValueError(f"меток {len(keys)}, а окон {len(ev.y)}")
    rows = {}
    for k in sorted({str(v) for v in keys.tolist()}):
        sub = ev.restrict(windows=(keys.astype(str) == k), leads=leads)
        c = sub.counts()
        if c["n_windows"] < min_windows or c["n_stations"] < min_stations:
            continue
        rows[k] = sub.summary(ci=ci, **kw)
    return rows


def spread(values):
    """Разброс числа по сидам: среднее, минимум, максимум, стандартное отклонение."""
    v = np.asarray([x for x in np.asarray(values, np.float64).ravel() if np.isfinite(x)])
    if v.size == 0:
        return dict(mean=float("nan"), min=float("nan"), max=float("nan"),
                    std=float("nan"), n=0)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size))


def seed_spread(summaries, key="pooled"):
    """{метрика: разброс} по нескольким прогонам одной архитектуры с разными сидами."""
    return {m: spread([s[key][m] for s in summaries]) for m in METRICS}


def skill(y, mu, mu_clim, w):
    """Скилл относительно климатологии на выбранных парах. Знаменатель - те же пары."""
    y = np.asarray(y, np.float64)
    sums, den = _sums_of(y, mu, mu_clim, w)
    return float(1.0 - sums["se"] / max(sums["se_clim"], EPS))


def _sums_of(y, mu, mu_clim, w):
    w = (np.asarray(w, np.float64) > 0).astype(np.float64)
    se = (np.asarray(mu, np.float64) - y) ** 2
    se_c = (np.asarray(mu_clim, np.float64) - y) ** 2
    return dict(se=float((se * w).sum()), se_clim=float((se_c * w).sum())), float(w.sum())


def skill_per_lead(y, mu, mu_clim, w):
    """Скилл отдельно на каждом лиде: и числитель, и знаменатель - по этому лиду."""
    mse = wmean((np.asarray(mu) - np.asarray(y)) ** 2, w, axis=0)
    mse_c = wmean((np.asarray(mu_clim) - np.asarray(y)) ** 2, w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - mse / np.maximum(mse_c, EPS)


def coverage(y, q, w, lo=I_LO90, hi=I_HI90):
    return float(wmean(inside(y, q[..., lo], q[..., hi]), w))


def metric_table(y, mu, q, mu_clim, w, leads=(1, 3, 6, 12, 24, 48, 72, 120, 168),
                 station=None):
    y = np.asarray(y)
    if station is None:
        station = np.zeros(len(y), np.int64)
    ev = Evaluation(y=np.asarray(y, np.float64), mu=np.asarray(mu, np.float64),
                    q=np.asarray(q, np.float64), mu_clim=np.asarray(mu_clim, np.float64),
                    w=np.asarray(w, np.float64), station=np.asarray(station))
    out = {}
    for h in leads:
        sub = ev.restrict(leads=[h])
        m = sub.pooled()
        out[h] = dict(MAE=m["MAE"], RMSE=m["RMSE"], Skill=m["Skill"], CRPS=m["CRPS"],
                      PICP80=m["PICP80"], PICP90=m["PICP90"], Winkler90=m["Winkler90"],
                      n_valid=int((np.asarray(w)[:, h - 1] > 0).sum()))
    return out


__all__ = ["ACIParams", "CENTRAL_INTERVALS", "Evaluation", "FINE_LEADS", "LEAD_BINS", "METRICS",
           "NQ", "Q", "SHARPNESS_POINTS", "SHARPNESS_RANGE", "aci_effective_level", "aci_run",
           "aci_score", "apply_adaptive", "apply_conformal", "breakdown", "by_lead",
           "by_lead_bin", "calibrate_forecast", "conformal_table", "coverage",
           "fit_conformal_shift", "inside", "interval_indices", "lead_bin_index", "lead_bin_of",
           "lead_mask", "metric_table", "pair_terms", "pinball_crps", "seed_spread",
           "sharpness_scales", "skill", "skill_per_lead", "spread", "width_at_coverage",
           "winkler", "wmean"]
