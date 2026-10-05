from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np

from mayak.constants import H, HISTORY_BINS, QUANTILES

Q = np.array(QUANTILES, np.float32)
NQ = len(QUANTILES)
# Квантили стандартного нормального распределения на уровнях QUANTILES.
ZQ = np.array([NormalDist().inv_cdf(q) for q in QUANTILES], np.float32)
I_LO90, I_LO80, I_MED, I_HI80, I_HI90 = 0, 1, 3, 5, 6
EPS = 1e-9
NO_HOUR = int(np.iinfo(np.int64).min)

LEAD_BINS = ((1, 6), (7, 24), (25, 72), (73, 168))
FINE_LEADS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168)

METRICS = ("MAE", "RMSE", "Skill", "CRPS", "PICP80", "PICP90", "Winkler90", "Width90")
_MEAN_METRIC = {"ae": "MAE", "crps": "CRPS", "cov80": "PICP80", "cov90": "PICP90",
                "wink90": "Winkler90", "width90": "Width90"}
_TERMS = ("ae", "se", "se_clim", "crps", "cov80", "cov90", "wink90", "width90")


def _central_intervals():
    """Симметричные центральные интервалы из набора квантилей.

    Returns:
        Кортеж троек: номинал интервала, номер нижнего и номер верхнего квантиля.
    """
    out = []
    for i in range(NQ // 2):
        j = NQ - 1 - i
        if abs(float(Q[i]) + float(Q[j]) - 1.0) < 1e-6:
            out.append((round(float(Q[j]) - float(Q[i]), 6), i, j))
    return tuple(out)


CENTRAL_INTERVALS = _central_intervals()


def interval_indices(nominal):
    """Номера квантилей центрального интервала заданного номинала.

    Args:
        nominal: номинал интервала, например 0.9.

    Returns:
        Пара номеров: нижний и верхний квантиль.

    Raises:
        ValueError: такого центрального интервала в наборе квантилей нет.
    """
    for nom, i, j in CENTRAL_INTERVALS:
        if abs(nom - float(nominal)) < 1e-6:
            return i, j
    raise ValueError(f"номинал {nominal} не является центральным интервалом набора квантилей; "
                     f"есть {[nom for nom, _i, _j in CENTRAL_INTERVALS]}")


def wmean(x, w, axis=None):
    """Среднее с весами маски: нормировка на сумму весов, а не на число элементов.

    Args:
        x: значения.
        w: веса той же формы; нулевой вес исключает значение.
        axis: ось усреднения; None - по всем элементам.

    Returns:
        Среднее; NaN там, где сумма весов нулевая.
    """
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
    """Слагаемые всех метрик на каждой паре окна и лида.

    Args:
        y: факт, форма (N, H).
        mu: точечный прогноз, форма (N, H).
        q: квантили, форма (N, H, число квантилей).
        mu_clim: прогноз климатологии, форма (N, H).

    Returns:
        Словарь массивов формы (N, H): абсолютная и квадратичная ошибка, квадратичная
        ошибка климатологии, CRPS, попадание в интервалы 80 и 90 %, оценка Винклера и
        ширина интервала 90 %.
    """
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
    """Словарь метрик по взвешенным суммам слагаемых и сумме весов."""
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
    """Булева маска лидов по их списку.

    Args:
        leads: None, булева маска длины горизонта или номера лидов с единицы.
        horizon: длина горизонта, ч.

    Returns:
        Булева маска длины горизонта; None, если лиды не заданы.

    Raises:
        ValueError: маска не той длины или лид вне горизонта.
    """
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
    """Номер бина лидов для часа лида.

    Args:
        h1: час лида, считая с единицы.
        lead_bins: бины лидов, пары границ включительно.

    Returns:
        Номер бина; лиды за последним бином относятся к последнему.
    """
    for i, (a, b) in enumerate(lead_bins):
        if a <= h1 <= b:
            return i
    return len(lead_bins) - 1


def lead_bin_of(horizon=H, lead_bins=LEAD_BINS):
    """Номер бина для каждого лида горизонта.

    Args:
        horizon: длина горизонта, ч.
        lead_bins: бины лидов.

    Returns:
        Массив int64 длины горизонта.
    """
    return np.array([lead_bin_index(h + 1, lead_bins) for h in range(horizon)], np.int64)


def _lead_bin_spans(lead_bin, n_bins):
    """Первый лид и число лидов каждого бина в пределах горизонта.

    Границы берутся по фактическим лидам горизонта, а не по объявленным бинам: лиды за
    последним бином относятся к последнему, а бин, который горизонт обрезает, короче
    объявленного. У бина без лидов на горизонте первый лид ноль и ширина единица; записей
    у него всё равно нет.

    Args:
        lead_bin: номер бина для каждого лида горизонта.
        n_bins: число бинов.

    Returns:
        Пара массивов int64 по бинам: первый лид, считая с единицы, и число лидов.

    Raises:
        ValueError: лиды какого-то бина идут не подряд.
    """
    lead_bin = np.asarray(lead_bin, np.int64)
    lo, width = np.zeros(n_bins, np.int64), np.ones(n_bins, np.int64)
    for b in range(n_bins):
        leads = np.flatnonzero(lead_bin == b) + 1
        if not leads.size:
            continue
        if leads[-1] - leads[0] + 1 != leads.size:
            raise ValueError(f"лиды бина {b} идут не подряд: {leads.tolist()}")
        lo[b], width[b] = leads[0], leads.size
    return lo, width


def check_history_bins(history_bins=HISTORY_BINS):
    """Проверяет, что бины длины истории идут подряд с нуля без пропусков и наложений.

    Args:
        history_bins: бины длины истории: границы включительно, ч, и подпись.

    Returns:
        Верхние границы бинов, массив int64.

    Raises:
        ValueError: бинов нет, первый бин не начинается с нуля, между бинами пропуск
            или наложение.
    """
    if not history_bins:
        raise ValueError("бинов длины истории нет")
    prev = -1
    for lo, hi, *_name in history_bins:
        if int(lo) != prev + 1 or int(hi) < int(lo):
            raise ValueError(f"бины длины истории {[tuple(b[:2]) for b in history_bins]}: "
                             f"нужны подряд с нуля, без пропусков и наложений")
        prev = int(hi)
    return np.array([int(b[1]) for b in history_bins], np.int64)


def history_bin_of(history, history_bins=HISTORY_BINS):
    """Номер бина длины истории для каждой длины.

    Args:
        history: длина истории, ч; число или массив.
        history_bins: бины длины истории.

    Returns:
        Номера бинов int64 той же формы; длина за последним бином относится к нему.

    Raises:
        ValueError: длина истории отрицательная или бины негодны.
    """
    his = check_history_bins(history_bins)
    h = np.asarray(history, np.int64)
    if np.any(h < 0):
        raise ValueError(f"длина истории отрицательная: {h.min()} ч")
    return np.minimum(np.searchsorted(his, h, side="left"), len(his) - 1).astype(np.int64)


def history_bin_index(L, history_bins=HISTORY_BINS):
    """Номер бина длины истории для одной длины.

    Args:
        L: длина истории, ч.
        history_bins: бины длины истории.

    Returns:
        Номер бина.
    """
    return int(history_bin_of(int(L), history_bins))


def order_around_median(q):
    """Восстанавливает порядок квантилей, не трогая медиану.

    Квантили выше медианы подтягиваются вверх до соседа слева, квантили ниже медианы
    опускаются до соседа справа. Медиана остаётся ровно той, что была на входе. Пропуск
    значения распространяется от медианы наружу.

    Args:
        q: квантили, последняя ось - набор квантилей.

    Returns:
        Упорядоченные квантили float32 той же формы.
    """
    q = np.asarray(q, np.float32)
    lo = np.minimum.accumulate(q[..., I_MED::-1], axis=-1)[..., ::-1]
    hi = np.maximum.accumulate(q[..., I_MED:], axis=-1)
    return np.concatenate([lo[..., :-1], hi], axis=-1)


def check_median_free(shift):
    """Проверяет, что таблица поправок не сдвигает медиану.

    Args:
        shift: таблица поправок, форма (число бинов лидов, число квантилей).

    Raises:
        ValueError: поправка медианы хотя бы в одном бине не равна нулю.
    """
    med = np.asarray(shift, np.float32)[..., I_MED]
    if np.any(med != 0.0):
        raise ValueError(f"поправка медианы в таблице не нулевая ({med.tolist()}): такая "
                         f"таблица сдвигает точечный прогноз; подгоните таблицу заново")


def check_conformal_shape(shift, lead_bins=LEAD_BINS, history_bins=HISTORY_BINS):
    """Проверяет форму конформной таблицы и нулевую поправку медианы.

    Args:
        shift: таблица поправок.
        lead_bins: бины лидов таблицы.
        history_bins: бины длины истории таблицы.

    Returns:
        Таблица float32 формы (число бинов лидов, число бинов длины истории, число
        квантилей).

    Raises:
        ValueError: таблица старого формата без бинов длины истории, другой формы или с
            ненулевой поправкой медианы.
    """
    shift = np.asarray(shift, np.float32)
    want = (len(lead_bins), len(history_bins), NQ)
    if shift.ndim == 2:
        raise ValueError(f"таблица поправок старого формата {shift.shape}: бины лидов × "
                         f"квантили, без бинов длины истории. Теперь таблица - бины лидов × "
                         f"бины длины истории × квантили {want}; подгоните таблицу заново: "
                         f"python scripts/calibrate.py")
    if shift.shape != want:
        raise ValueError(f"таблица поправок формы {shift.shape}, нужно {want}: бины лидов × "
                         f"бины длины истории × квантили")
    check_median_free(shift)
    return shift


def conformal_table(shift, history, horizon=H, lead_bins=LEAD_BINS, history_bins=HISTORY_BINS):
    """Поправки таблицы, развёрнутые на каждый лид, для заданной длины истории.

    Args:
        shift: таблица поправок, форма (число бинов лидов, число бинов длины истории,
            число квантилей).
        history: длина истории прогноза, ч; число или массив длин по прогнозам.
        horizon: число лидов.
        lead_bins: бины лидов таблицы.
        history_bins: бины длины истории таблицы.

    Returns:
        Поправки float32, форма (*форма history, horizon, число квантилей).

    Raises:
        ValueError: неверная форма таблицы, ненулевая поправка медианы или
            отрицательная длина истории.
    """
    shift = check_conformal_shape(shift, lead_bins, history_bins)
    hb = history_bin_of(history, history_bins)
    per_lead = np.moveaxis(shift[lead_bin_of(horizon, lead_bins)], 1, 0)
    return np.ascontiguousarray(per_lead[hb])


def apply_conformal(q, shift, history, lead_bins=LEAD_BINS, history_bins=HISTORY_BINS):
    """Конформная поправка квантилей: меняется ширина интервалов, медиана остаётся.

    Строка таблицы выбирается по фактической длине истории прогноза.

    Args:
        q: квантили, форма (..., лиды, число квантилей).
        shift: таблица поправок по бинам лидов и длины истории с нулевой поправкой
            медианы.
        history: длина истории, ч: одно число для всех прогнозов или массив формы
            q.shape[:-2].
        lead_bins: бины лидов таблицы.
        history_bins: бины длины истории таблицы.

    Returns:
        Поправленные и упорядоченные квантили float32.

    Raises:
        ValueError: длина истории не задана или не по одной на прогноз; таблица негодна.
    """
    if history is None:
        raise ValueError("конформная таблица применяется по длине истории прогноза, а она "
                         "не задана")
    q = np.asarray(q, np.float32)
    hist = np.asarray(history)
    if hist.ndim and hist.shape != q.shape[:-2]:
        raise ValueError(f"длин истории {hist.shape}, а прогнозов {q.shape[:-2]}: нужна длина "
                         f"истории каждого прогноза")
    table = conformal_table(shift, hist, q.shape[-2], lead_bins, history_bins)
    return order_around_median(q + table)


def apply_adaptive(q, theta=0.0, lead_bins=LEAD_BINS):
    """Адаптивная поправка: квантили растягиваются вокруг медианы.

    Множитель растяжения - экспонента от параметра: один на все лиды или свой на каждый
    бин лидов. Медиана поправкой не меняется. Лиды с нулевым параметром возвращаются как
    есть, без арифметики, чтобы устройство и офлайн-оценка совпадали до бита. Растяжение с
    положительным множителем сохраняет порядок; упорядочивание вокруг медианы - защита
    для входа, который уже был немонотонным.

    Args:
        q: квантили, форма (..., лиды, число квантилей).
        theta: логарифм множителя ширины: число или по значению на бин лидов.
        lead_bins: бины лидов параметров.

    Returns:
        Пара: поправленные квантили и их медиана.

    Raises:
        ValueError: параметр не конечен или число параметров не равно числу бинов.
    """
    th = np.asarray(theta, np.float64)
    if not np.all(np.isfinite(th)):
        raise ValueError(f"θ адаптивной поправки не конечно: {theta}")
    q = np.asarray(q, np.float32)
    if th.ndim == 0:
        if float(th) != 0.0:
            med = q[..., I_MED:I_MED + 1]
            q = order_around_median(med + np.float32(math.exp(float(th))) * (q - med))
        return q, q[..., I_MED].copy()
    if th.shape != (len(lead_bins),):
        raise ValueError(f"θ по бинам лидов формы {th.shape}, а бинов {len(lead_bins)}")
    per_lead = th[lead_bin_of(q.shape[-2], lead_bins)]
    nz = per_lead != 0.0
    if nz.any():
        k = np.array([math.exp(v) for v in per_lead[nz].tolist()], np.float32)
        sel = q[..., nz, :]
        med = sel[..., I_MED:I_MED + 1]
        q = q.copy()
        q[..., nz, :] = order_around_median(med + k[:, None] * (sel - med))
    return q, q[..., I_MED].copy()


def calibrate_forecast(q, shift=None, theta=0.0, history=None, lead_bins=LEAD_BINS,
                       history_bins=HISTORY_BINS):
    """Калибровка квантилей прогноза в фиксированном порядке.

    Сначала сплит-конформная таблица по бинам лидов и длины истории, подогнанная офлайн
    на калибровочном окне, затем адаптивный множитель ширины по бинам лидов, который
    подстраивается онлайн на устройстве. Медиана берётся из итоговых квантилей.

    Args:
        q: квантили модели, форма (..., лиды, число квантилей).
        shift: конформная таблица; None - без неё.
        theta: логарифм адаптивного множителя ширины: число или по бинам лидов.
        history: длина истории прогноза, ч; нужна, если задана таблица.
        lead_bins: бины лидов.
        history_bins: бины длины истории таблицы.

    Returns:
        Пара: откалиброванные квантили и их медиана.
    """
    if shift is not None:
        q = apply_conformal(q, shift, history, lead_bins, history_bins)
    return apply_adaptive(q, theta, lead_bins)


def aci_score_bounds(y, lo, med, hi):
    """Нормированный выход факта за интервал по его границам и медиане.

    Args:
        y: факт.
        lo: нижняя граница интервала.
        med: медиана.
        hi: верхняя граница интервала.

    Returns:
        Оценка той же формы, что факт; NaN там, где факт или медиана не конечны.
    """
    y = np.asarray(y, np.float64)
    lo, med, hi = (np.asarray(v, np.float64) for v in (lo, med, hi))
    u = y - med
    d = np.where(u >= 0, hi - med, med - lo)
    au = np.abs(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d > 0, au / np.where(d > 0, d, 1.0), np.inf)
    s = np.where(au == 0, 0.0, s)
    return np.where(np.isfinite(y) & np.isfinite(med), s, np.nan)


def aci_score(y, q, interval=(I_LO90, I_HI90)):
    """Нормированный выход факта за интервал: 0 на медиане, 1 ровно на границе.

    Расстояние факта от медианы делится на ширину той половины интервала, в которую
    факт попал. Факт лежит внутри интервала, растянутого адаптивной поправкой, тогда и
    только тогда, когда оценка не больше множителя растяжения. Половина интервала
    нулевой ширины даёт бесконечность для любого факта, кроме самой медианы.

    Args:
        y: факт.
        q: квантили, последняя ось - набор квантилей.
        interval: номера нижнего и верхнего квантиля интервала.

    Returns:
        Оценка той же формы, что факт; NaN там, где факт или медиана не конечны.
    """
    i, j = interval
    q = np.asarray(q, np.float64)
    return aci_score_bounds(y, q[..., i], q[..., I_MED], q[..., j])


def _f32(x):
    """Округление до float32.

    Параметр адаптивной калибровки хранится в состоянии рантайма как float32, поэтому
    онлайн-путь с перезапусками совпадает с непрерывным до бита.
    """
    return float(np.float32(x))


@dataclass(frozen=True)
class ACIParams:
    """Адаптивная конформная калибровка на устройстве.

    После каждого валидного часа параметр сдвигается на малый шаг: вверх при промахе
    интервала, вниз при попадании, так что в среднем доля промахов тянется к целевой.
    Сумма сдвигов равна шагу, умноженному на сумму отклонений промахов от цели. Отсюда
    гарантия без предположений о распределении: средняя доля промахов на любом потоке
    отличается от цели не больше чем на изменение параметра, делённое на шаг и на число
    часов.

    Подстраивается не номинальный уровень интервала, а логарифм множителя ширины:
    интервал растягивается вокруг медианы в экспоненту от параметра раз. У модели семь
    квантилей, уровни за пределами пяти и девяноста пяти процентов ей недоступны, а
    подстройка уровня при серии промахов требовала бы бесконечного интервала. В форме
    множителя каждый промах расширяет интервал на один и тот же относительный шаг, и всё
    состояние - одно число.

    Параметр ограничен с обеих сторон логарифмом ``max_factor``, чтобы поток сплошных
    промахов при отказе прибора не разгонял его без предела. Пока граница не достигнута,
    гарантия выполняется точно; упоры в границу считаются и показываются в отчётах.

    Attributes:
        target: целевая доля промахов центрального интервала. Интервал с уровнем, равным
            единице минус цель, должен быть в наборе квантилей.
        gamma: шаг подстройки.
        max_factor: наибольший множитель ширины интервала, больше единицы.
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
        """Параметр после одного наблюдения.

        Промах сдвигает параметр вверх, попадание - вниз; сдвиг равен шагу подстройки,
        умноженному на отклонение промаха от целевой доли. Результат упирается в
        границы.

        Args:
            theta: параметр до наблюдения.
            miss: был ли промах.

        Returns:
            Новый параметр, округлённый до float32.
        """
        return self.clip(float(theta) + self.gamma * (float(miss) - self.target))

    def step(self, theta, score):
        """Одна обратная связь по нормированному выходу факта за интервал.

        Args:
            theta: параметр до наблюдения.
            score: нормированный выход факта за интервал.

        Returns:
            Пара: новый параметр и признак промаха.
        """
        miss = bool(score > math.exp(theta))
        return self.update(theta, miss), miss


def aci_run(scores, params, theta0=0.0):
    """Прогон адаптивной калибровки по одному потоку оценок в порядке времени.

    NaN в оценках - обратной связи нет, факт невалиден, параметр не меняется.

    Args:
        scores: нормированные выходы факта за интервал по порядку времени.
        params: параметры адаптивной калибровки.
        theta0: начальный параметр.

    Returns:
        Словарь: параметр до каждого наблюдения, с которым и выпущен интервал; промахи,
        NaN там, где связи нет; итоговый параметр и число упоров в границы.
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


class AdaptiveCalibration:
    """Адаптивная калибровка прибора по бинам лидов.

    У каждого бина лидов свой логарифм множителя ширины. Обратная связь идёт через
    кольцо по часам-мишеням: для каждого из следующих часов горизонта и каждого бина
    лежат медиана и границы интервала одной записи.

    Выпуск пишет бин в кольцо не всегда, а только когда сумма момента выпуска и первого
    лида бина делится на ширину бина; тогда он пишет все лиды бина. Записи таких выпусков
    стыкуются без наложений: каждый час получает ровно одну запись на бин, а её лид
    пробегает лиды бина подряд и повторяется с периодом, равным ширине бина. Так при
    ежечасном выпуске множитель бина подстраивается по всем его лидам поровну, а не по
    самому короткому. При выпуске реже бин получает обратную связь только с выпусков,
    попавших на такие часы.

    Валидный час сверяется со всеми записями на него, и каждый бин обновляется этим часом
    не больше одного раза.

    Записи берутся после конформной таблицы и до множителя, как и прежде. Кольцо живёт
    только в памяти и после перезапуска пусто.

    Логика общая для рантайма устройства и офлайн-прогона калибровки.

    Args:
        params: параметры адаптивной калибровки или None: тогда множители только
            хранятся, не подстраиваются, и кольца нет.
        horizon: длина горизонта, ч.
        lead_bins: бины лидов.

    Attributes:
        lead_bin: номер бина для каждого лида горизонта.
        bin_lo: первый лид каждого бина в пределах горизонта, считая с единицы.
        bin_width: число лидов каждого бина в пределах горизонта.
        theta: логарифмы множителей по бинам лидов, значения float32.
        updates: число обратных связей по бинам с последнего сброса.
        misses: число промахов по бинам с последнего сброса.
        clipped: число обновлений, упёршихся в границу множителя, с последнего сброса.
    """

    def __init__(self, params=None, horizon=H, lead_bins=LEAD_BINS):
        self.params = params
        self.horizon = int(horizon)
        self.lead_bins = tuple(tuple(b) for b in lead_bins)
        self.lead_bin = lead_bin_of(self.horizon, self.lead_bins)
        nb = len(self.lead_bins)
        self.bin_lo, self.bin_width = _lead_bin_spans(self.lead_bin, nb)
        self.ring_hour = None if params is None else np.full((self.horizon, nb), NO_HOUR,
                                                             np.int64)
        self.ring = None if params is None else np.zeros((self.horizon, nb, 3), np.float32)
        self.reset()

    @property
    def n_bins(self):
        return len(self.lead_bins)

    @property
    def nbytes(self):
        """Размер кольца в памяти, байт; без параметров кольца нет."""
        return 0 if self.ring is None else int(self.ring_hour.nbytes + self.ring.nbytes)

    def reset(self, theta=0.0):
        """Новые множители, нулевые счётчики и пустое кольцо.

        Args:
            theta: логарифм множителя: одно число на все бины или по числу на бин.

        Raises:
            ValueError: параметр не конечен или не по одному на бин.
        """
        th = np.asarray(theta, np.float64)
        if th.ndim and th.shape != (self.n_bins,):
            raise ValueError(f"θ по бинам лидов формы {th.shape}, а бинов {self.n_bins}")
        th = np.broadcast_to(th, (self.n_bins,))
        if not np.all(np.isfinite(th)):
            raise ValueError(f"θ адаптивной поправки не конечно: {th.tolist()}")
        clip = _f32 if self.params is None else self.params.clip
        self.theta = [clip(float(v)) for v in th]
        self.updates = [0] * self.n_bins
        self.misses = [0] * self.n_bins
        self.clipped = 0
        self.clear()

    def clear(self):
        """Пустое кольцо; множители и счётчики остаются."""
        if self.ring_hour is not None:
            self.ring_hour.fill(NO_HOUR)

    def record(self, after_hour, q):
        """Записать выпуск в кольцо.

        Пишутся только бины, для которых сумма момента выпуска и первого лида бина
        делится на ширину бина; у такого бина - все его лиды.

        Args:
            after_hour: абсолютный час, после которого начинается горизонт выпуска.
            q: квантили выпуска после конформной таблицы и до множителя, форма
                (горизонт, число квантилей).
        """
        if self.params is None:
            return
        q = np.asarray(q, np.float32)
        i, j = self.params.interval
        t = int(after_hour)
        b = self.lead_bin
        lead = np.flatnonzero((t + self.bin_lo[b]) % self.bin_width[b] == 0)
        if not lead.size:
            return
        hours = t + 1 + lead
        slot, b = hours % self.horizon, b[lead]
        self.ring_hour[slot, b] = hours
        self.ring[slot, b] = np.stack([q[lead, i], q[lead, I_MED], q[lead, j]], -1)

    def scores(self, y, hour):
        """Нормированные выходы факта за интервал по записям кольца на этот час.

        Кольцо не меняется.

        Args:
            y: факт часа.
            hour: абсолютный час.

        Returns:
            Список пар: номер бина лидов и оценка.
        """
        if self.params is None:
            return []
        s = int(hour) % self.horizon
        return [(b, float(aci_score_bounds(y, *self.ring[s, b].tolist())))
                for b in range(self.n_bins) if int(self.ring_hour[s, b]) == int(hour)]

    def feedback(self, y, hour):
        """Обратная связь валидного факта часа: каждый бин с записью на этот час - один раз.

        Args:
            y: факт часа.
            hour: абсолютный час.
        """
        if self.params is None:
            return
        s = int(hour) % self.horizon
        for b, score in self.scores(y, hour):
            self.ring_hour[s, b] = NO_HOUR
            if score != score:
                continue
            self.theta[b], miss = self.params.step(self.theta[b], score)
            self.updates[b] += 1
            self.misses[b] += int(miss)
            self.clipped += self.theta[b] in (self.params.theta_min, self.params.theta_max)

    def coverage(self):
        """Фактическое покрытие по обратной связи в каждом бине; NaN без обратной связи."""
        return tuple(1.0 - m / u if u else float("nan")
                     for u, m in zip(self.updates, self.misses))


def aci_effective_level(theta, target=0.10):
    """Номинал, которому соответствует растянутый интервал при нормальной форме прогноза.

    Половина исходного интервала в единицах разброса растягивается в множитель раз, и
    уровень считается по нормальному распределению заново.

    Args:
        theta: логарифм множителя ширины.
        target: целевая доля промахов исходного интервала.

    Returns:
        Номинал растянутого интервала.
    """
    nd = NormalDist()
    z = nd.inv_cdf(1.0 - float(target) / 2.0)
    return 2.0 * nd.cdf(z * math.exp(float(theta))) - 1.0


SHARPNESS_RANGE = (0.25, 4.0)
SHARPNESS_POINTS = 33


def sharpness_scales(lo=SHARPNESS_RANGE[0], hi=SHARPNESS_RANGE[1], n=SHARPNESS_POINTS):
    """Логарифмическая сетка множителей ширины; в ней всегда есть единица - выход модели.

    Args:
        lo: наименьший множитель.
        hi: наибольший множитель.
        n: число точек сетки.

    Returns:
        Возрастающий массив множителей.
    """
    s = np.exp(np.linspace(math.log(lo), math.log(hi), int(n)))
    return np.unique(np.concatenate([s, [1.0]]))


def width_at_coverage(coverage, width, target):
    """Ширина, при которой фактическое покрытие достигает цели.

    Кривая упорядочена по множителю; между её точками ширина интерполируется линейно.

    Args:
        coverage: фактическое покрытие в точках кривой.
        width: средняя ширина в тех же точках.
        target: целевое покрытие.

    Returns:
        Ширина; NaN, если кривая цели не достигает.
    """
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


def _fit_lead_bins(y, q, w, lead_bins, what):
    """Поправки по бинам лидов на одном наборе окон; None, если какой-то бин пуст."""
    shift = np.zeros((len(lead_bins), q.shape[-1]), np.float32)
    for bi, (a, b) in enumerate(lead_bins):
        sl = slice(a - 1, b)
        resid = (y[:, sl, None] - q[:, sl, :]).reshape(-1, q.shape[-1])
        resid = resid[np.asarray(w)[:, sl].reshape(-1) > 0]
        if len(resid) == 0:
            if what is None:
                return None
            raise ValueError(f"{what}: в бине лидов {a}-{b} нет ни одного валидного часа")
        for qi, tau in enumerate(QUANTILES):
            if qi != I_MED:
                shift[bi, qi] = np.quantile(resid[:, qi], tau)
    return shift


def fit_conformal_shift(y, q, w, history, lead_bins=LEAD_BINS, history_bins=HISTORY_BINS,
                        min_windows=1):
    """Сплит-конформные поправки по бинам лидов и бинам длины истории.

    Поправка квантиля в ячейке - квантиль того же уровня от остатков факта относительно
    этого квантиля на валидных часах окон ячейки. Строка бина длины истории подгоняется
    по окнам этого бина. Если окон бина с валидными часами цели меньше порога или в
    какой-то ячейке нет ни одного валидного часа, бин получает маргинальную строку -
    поправки по всем окнам набора сразу. Поправка медианы равна нулю по построению:
    таблица меняет только ширину интервалов, точечный прогноз остаётся прогнозом модели.

    Args:
        y: факт, форма (N, H).
        q: квантили модели, форма (N, H, число квантилей).
        w: веса часов цели, форма (N, H); учитываются только положительные.
        history: фактическая длина истории каждого окна, ч, форма (N,).
        lead_bins: бины лидов.
        history_bins: бины длины истории.
        min_windows: наименьшее число окон бина длины истории, при котором у бина своя
            строка.

    Returns:
        Пара: таблица float32 формы (число бинов лидов, число бинов длины истории, число
        квантилей) со столбцом медианы из нулей и сведения о подгонке - по словарю на бин
        длины истории: подпись, границы, число окон и признак маргинальной строки.

    Raises:
        ValueError: длин истории не по одной на окно или в каком-то бине лидов нет ни
            одного валидного часа во всём наборе.
    """
    y, q, w = np.asarray(y), np.asarray(q), np.asarray(w)
    hist = np.asarray(history, np.int64)
    if hist.shape != (len(y),):
        raise ValueError(f"длин истории {hist.shape}, а окон {len(y)}")
    marginal = _fit_lead_bins(y, q, w, lead_bins, "весь калибровочный набор")
    hb = history_bin_of(hist, history_bins)
    shift = np.repeat(marginal[:, None, :], len(history_bins), axis=1)
    rows = []
    for k, (lo, hi, name) in enumerate(history_bins):
        sel = hb == k
        n = int((w[sel] > 0).any(axis=1).sum()) if sel.any() else 0
        own = _fit_lead_bins(y[sel], q[sel], w[sel], lead_bins, None) \
            if n >= max(1, int(min_windows)) else None
        if own is not None:
            shift[:, k] = own
        rows.append(dict(bin=name, lo=int(lo), hi=int(hi), windows=n, marginal=own is None))
    return shift, rows


def quantile_ci(samples, level=0.90):
    """Центральные интервалы по выборкам бутстрапа.

    Args:
        samples: словарь: метрика и её выборка.
        level: уровень интервала.

    Returns:
        Словарь: метрика и пара границ интервала.
    """
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
    """Предсказания одной модели на наборе окон, веса и привязка окон к станциям.

    Attributes:
        y: факт, форма (N, H).
        mu: точечный прогноз, форма (N, H).
        q: квантили, форма (N, H, число квантилей).
        mu_clim: прогноз климатологии, форма (N, H).
        w: веса пар окна и лида; при создании приводятся к нулю и единице.
        station: станция каждого окна, форма (N,).
    """
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
        """Оценка на подвыборке окон, лидов или того и другого.

        Args:
            windows: номера окон или булева маска окон; None - все окна.
            leads: лиды в любом виде, который понимает маска лидов; None - все.

        Returns:
            Новая оценка с теми же данными и обнулёнными весами вне подвыборки.
        """
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

    def with_calibration(self, shift=None, theta=0.0, history=None, lead_bins=LEAD_BINS,
                         history_bins=HISTORY_BINS):
        """Оценка после калибровки квантилей; медиана берётся из квантилей.

        Args:
            shift: конформная таблица; None - без неё.
            theta: логарифм адаптивного множителя ширины: число или по бинам лидов.
            history: фактическая длина истории каждого окна, ч; нужна с таблицей.
            lead_bins: бины лидов.
            history_bins: бины длины истории таблицы.

        Returns:
            Новая оценка; без калибровки - та же.
        """
        if shift is None and not np.any(np.asarray(theta, np.float64)):
            return self
        q, mu = calibrate_forecast(self.q, shift, theta, history, lead_bins, history_bins)
        return Evaluation(y=self.y, mu=mu, q=q, mu_clim=self.mu_clim,
                          w=self.w, station=self.station)

    def with_conformal(self, shift, history, lead_bins=LEAD_BINS, history_bins=HISTORY_BINS):
        """Оценка с конформной поправкой; медиана берётся из квантилей.

        Args:
            shift: конформная таблица; None - без неё.
            history: фактическая длина истории каждого окна, ч.
            lead_bins: бины лидов таблицы.
            history_bins: бины длины истории таблицы.

        Returns:
            Новая оценка.
        """
        return self.with_calibration(shift, 0.0, history, lead_bins, history_bins)


    def counts(self):
        used = self.w > 0
        win = used.any(1)
        return dict(n_windows=int(win.sum()),
                    n_stations=int(len(np.unique(self.station[win]))) if win.any() else 0,
                    n_pairs=int(used.sum()))

    def station_sums(self):
        """Суммы слагаемых метрик по станциям - основа макро-метрик и бутстрапа.

        Returns:
            Тройка: станции, словарь сумм слагаемых по станциям и суммы весов.
        """
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
        """Метрики на каждой станции отдельно.

        Returns:
            Пара: станции с валидными парами и словарь метрика - массив по этим станциям.
        """
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
        """Выборки блочного бутстрапа по станциям.

        Args:
            n_boot: число повторов.
            seed: сид генератора.

        Returns:
            Пара словарей метрика - выборка длины ``n_boot``: пуловые и макро-метрики.
            None, если ни у одной станции нет валидных пар.
        """
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
        """Сводка оценки.

        Args:
            ci: считать ли интервалы блочного бутстрапа.
            n_boot: число повторов бутстрапа.
            seed: сид бутстрапа.
            level: уровень интервалов.

        Returns:
            Словарь: пуловые и макро-метрики, число окон, станций и пар, при ``ci`` -
            интервалы.
        """
        out = dict(pooled=self.pooled(), macro=self.macro(), **self.counts())
        if ci:
            out["ci"] = self.bootstrap_ci(n_boot=n_boot, seed=seed, level=level)
        return out

    def pit_histogram(self):
        """Гистограмма PIT по бинам между соседними квантилями.

        Returns:
            Словарь: наблюдаемые и ожидаемые доли факта в бинах и число пар.
        """
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
        """Диаграмма надёжности: номинальный уровень против фактического для всех квантилей.

        Returns:
            Словарь с массивами номинальных и фактических уровней.
        """
        emp = np.array([float(wmean((self.y <= self.q[..., i]).astype(np.float64), self.w))
                        for i in range(NQ)])
        return dict(nominal=np.asarray(Q, np.float64), empirical=emp)

    def sharpness_coverage(self):
        """Острота против покрытия: средняя ширина интервала и фактическое покрытие.

        Доли факта ниже нижней и выше верхней границы при одинаковом покрытии
        различают узкий интервал с промахами с обеих сторон и сдвиг с промахами с одной.

        Returns:
            Список строк по центральным интервалам: номинал, покрытие, средняя ширина,
            доля факта ниже и доля выше интервала.
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

        Интервал заданного номинала растягивается вокруг медианы в каждый множитель
        сетки адаптивной поправкой - получается непрерывная зависимость средней ширины
        от фактического покрытия. Единичный множитель - выход модели как есть. Кривые
        разных моделей сравниваются при одинаковом фактическом покрытии, а не при
        одинаковом номинале.

        Args:
            scales: множители ширины; None - сетка по умолчанию.
            nominal: номинал интервала.
            lead_bins: бины лидов для отдельных панелей.

        Returns:
            Словарь: панель («весь горизонт» или бин лидов) и её множители, покрытия и
            ширины.
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
    """Сводки по отдельным лидам. Знаменатель скилла - на том же лиде.

    Args:
        ev: оценка одной модели.
        leads: лиды таблицы.
        ci: считать ли интервалы бутстрапа.
        **kw: параметры сводки.

    Returns:
        Словарь: час лида и сводка.
    """
    return {int(h): ev.restrict(leads=[h]).summary(ci=ci, **kw) for h in leads}


def by_lead_bin(ev, lead_bins=LEAD_BINS, ci=False, **kw):
    return {f"{a}-{b}": ev.restrict(leads=np.arange(a, b + 1)).summary(ci=ci, **kw)
            for a, b in lead_bins}


def ordered_labels(keys, order=None):
    """Метки разреза в порядке показа.

    Args:
        keys: метки окон.
        order: желаемый порядок меток. Метки, которых нет в наборе, пропускаются;
            метки набора, которых нет в порядке, идут следом по алфавиту.

    Returns:
        Список различных меток в порядке показа.
    """
    present = {str(v) for v in np.asarray(keys).tolist()}
    head = [str(k) for k in (order or ()) if str(k) in present]
    return head + sorted(present - set(head))


def breakdown(ev, keys, leads=None, min_windows=20, min_stations=2, ci=False, order=None, **kw):
    """Разрез оценки по меткам окон.

    Args:
        ev: оценка одной модели.
        keys: метка каждого окна, форма (N,).
        leads: лиды, на которых считаются метрики; None значит весь горизонт.
        min_windows: страта с меньшим числом окон не показывается.
        min_stations: страта с меньшим числом станций не показывается.
        ci: считать интервалы бутстрапа по станциям.
        order: порядок меток; по умолчанию алфавитный.
        **kw: параметры бутстрапа.

    Returns:
        Словарь из метки страты в её сводку.

    Raises:
        ValueError: число меток не совпадает с числом окон.
    """
    keys = np.asarray(keys)
    if len(keys) != len(ev.y):
        raise ValueError(f"меток {len(keys)}, а окон {len(ev.y)}")
    rows = {}
    for k in ordered_labels(keys, order):
        sub = ev.restrict(windows=(keys.astype(str) == k), leads=leads)
        c = sub.counts()
        if c["n_windows"] < min_windows or c["n_stations"] < min_stations:
            continue
        rows[k] = sub.summary(ci=ci, **kw)
    return rows


def spread(values):
    """Разброс числа по сидам.

    Args:
        values: значения по сидам; не конечные пропускаются.

    Returns:
        Словарь: среднее, минимум, максимум, стандартное отклонение и число значений.
    """
    v = np.asarray([x for x in np.asarray(values, np.float64).ravel() if np.isfinite(x)])
    if v.size == 0:
        return dict(mean=float("nan"), min=float("nan"), max=float("nan"),
                    std=float("nan"), n=0)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size))


def seed_spread(summaries, key="pooled"):
    """Разброс метрик по прогонам одной архитектуры с разными сидами.

    Args:
        summaries: сводки прогонов.
        key: какие метрики брать из сводки: ``pooled`` или ``macro``.

    Returns:
        Словарь: метрика и её разброс.
    """
    return {m: spread([s[key][m] for s in summaries]) for m in METRICS}


def skill(y, mu, mu_clim, w):
    """Скилл относительно климатологии на выбранных парах; знаменатель - те же пары.

    Args:
        y: факт.
        mu: точечный прогноз.
        mu_clim: прогноз климатологии.
        w: веса пар.

    Returns:
        Единица минус отношение квадратичной ошибки прогноза к ошибке климатологии.
    """
    y = np.asarray(y, np.float64)
    sums, den = _sums_of(y, mu, mu_clim, w)
    return float(1.0 - sums["se"] / max(sums["se_clim"], EPS))


def _sums_of(y, mu, mu_clim, w):
    w = (np.asarray(w, np.float64) > 0).astype(np.float64)
    se = (np.asarray(mu, np.float64) - y) ** 2
    se_c = (np.asarray(mu_clim, np.float64) - y) ** 2
    return dict(se=float((se * w).sum()), se_clim=float((se_c * w).sum())), float(w.sum())


def skill_per_lead(y, mu, mu_clim, w):
    """Скилл отдельно на каждом лиде: и числитель, и знаменатель - по этому лиду.

    Args:
        y: факт, форма (N, H).
        mu: точечный прогноз, форма (N, H).
        mu_clim: прогноз климатологии, форма (N, H).
        w: веса пар, форма (N, H).

    Returns:
        Массив длины H.
    """
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


__all__ = ["ACIParams", "AdaptiveCalibration", "CENTRAL_INTERVALS", "Evaluation", "FINE_LEADS",
           "HISTORY_BINS", "LEAD_BINS", "METRICS", "NO_HOUR", "NQ", "Q", "SHARPNESS_POINTS",
           "SHARPNESS_RANGE", "ZQ", "aci_effective_level", "aci_run", "aci_score",
           "aci_score_bounds", "apply_adaptive", "apply_conformal", "breakdown", "by_lead",
           "by_lead_bin",
           "calibrate_forecast", "check_conformal_shape", "check_history_bins",
           "check_median_free", "conformal_table", "coverage", "fit_conformal_shift",
           "history_bin_index", "history_bin_of", "inside", "interval_indices",
           "lead_bin_index", "lead_bin_of", "lead_mask", "metric_table", "order_around_median",
           "ordered_labels", "pair_terms", "pinball_crps", "seed_spread", "sharpness_scales",
           "skill", "skill_per_lead", "spread", "width_at_coverage", "winkler", "wmean"]
