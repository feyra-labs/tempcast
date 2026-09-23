"""Статистические эталоны: климатология, затухающая персистентность, сезонно-наивный."""
import numpy as np
from scipy.stats import norm

from mayak.baselines.cards import BaselineCard, Difference, Source, describe
from mayak.constants import H, QUANTILES
from mayak.data.splits import time_bounds
from mayak.data.store import get_store
from mayak.timeaxis import window_calendar

ZQ = norm.ppf(np.array(QUANTILES)).astype(np.float32)

RECENT_HOURS = 24
RECENT_MIN_VALID = 6
DAMPED_VAR_FLOOR = 0.02

FPP3 = Source("R. J. Hyndman, G. Athanasopoulos",
              "Forecasting: Principles and Practice, 3rd ed.", "OTexts, 2021",
              "https://otexts.com/fpp3/")

CLIMATOLOGY_CARD = BaselineCard(
    key="climatology", name="Климатология", kind="statistical",
    summary="Прогноз — многолетняя климатология самой станции на часах горизонта; "
            "эталон знаменателя скилл-скора.",
    sources=(Source("R. J. Hyndman, G. Athanasopoulos",
                    "Forecasting: Principles and Practice, 3rd ed., гл. 5.8 "
                    "(skill score относительно эталонного прогноза)", "OTexts, 2021",
                    "https://otexts.com/fpp3/"),
             Source("D. S. Wilks", "Statistical Methods in the Atmospheric Sciences, 4th ed., "
                    "гл. 9 (климатологический эталон, skill score)", "Elsevier, 2019")),
    taken=("климатология как эталонный прогноз и знаменатель скилла SS = 1 − MSE/MSE_clim",),
    differences=(
        Difference("климатология гармоническая (годовые × суточные гармоники среднего и "
                   "масштаба), а не среднее по календарному часу за многолетний период",
                   "у станций 3–10 лет данных: среднее по (день года, час) даёт "
                   "единицы наблюдений на ячейку и шумный эталон"),
        Difference("интервал нормальный с климатологическим масштабом остатка",
                   "тот же масштаб нормирует функцию потерь (блок 4); отдельная "
                   "эмпирическая квантильная климатология ввела бы второй эталон"),
    ),
    notes=("Эталон построен по многолетнему обучающему окну станции и при короткой "
           "истории знает о ней больше модели — оценка строга в пользу эталона.",),
)

DAMPED_CARD = BaselineCard(
    key="damped_persistence", name="Damped persistence", kind="statistical",
    summary="Аномалия относительно климатологии затухает к нулю с коэффициентом, "
            "своим для каждого лида: μ(h) = C(h) + r_h·ā.",
    sources=(Source("D. S. Wilks", "Statistical Methods in the Atmospheric Sciences, 4th ed., "
                    "гл. 10 (персистентность и AR(1) как эталоны)", "Elsevier, 2019"),
             Source("R. J. Hyndman, G. Athanasopoulos",
                    "Forecasting: Principles and Practice, 3rd ed., гл. 9 (AR(p))",
                    "OTexts, 2021", "https://otexts.com/fpp3/")),
    taken=("прогноз аномалии, затухающей к климатологии",
           "коэффициент затухания r_h оценивается по обучающей выборке отдельно для "
           "каждого лида h (прямая стратегия)",
           "интервал σ_clim·√(1 − r_h²) — условная дисперсия AR(1) с единичной "
           "безусловной дисперсией аномалии"),
    differences=(
        Difference("предиктор ā — средняя аномалия за последние 24 ч (не меньше 6 "
                   "валидных часов), а не аномалия последнего часа",
                   "аномалия одного часа несёт ошибку суточного хода климатологии и "
                   "шум прибора; при пропуске последнего часа эталон иначе не определён"),
        Difference("r_h — МНК через ноль по парам (ā, a(t+h)) с обрезкой в [0, 1], а не "
                   "φ^h одного параметра AR(1)",
                   "прямая оценка по лиду не навязывает экспоненциальное затухание и "
                   "(до обрезки) минимизирует MSE эталона на обучающей выборке; для "
                   "процесса AR(1) с предиктором-последним-часом она сходится к φ^h"),
        Difference("снизу дисперсия ограничена 0.02·σ_clim²",
                   "при r_h → 1 интервал иначе схлопывается в точку"),
    ),
)

SEASONAL_NAIVE_CARD = BaselineCard(
    key="seasonal_naive", name="Seasonal-naive 24ч", kind="statistical",
    summary="Прогноз на лид h — последнее наблюдённое значение в тот же час суток: "
            "ŷ(T+h) = y(T+h − m(k+1)), m = 24, k = ⌊(h−1)/m⌋.",
    sources=(Source("R. J. Hyndman, G. Athanasopoulos",
                    "Forecasting: Principles and Practice, 3rd ed., гл. 5.2 "
                    "(seasonal naive method)", "OTexts, 2021", "https://otexts.com/fpp3/"),),
    taken=("повтор последнего наблюдённого сезона периода m = 24 ч на весь горизонт",),
    differences=(
        Difference("если час суток в последних сутках невалиден, берётся тот же час из "
                   "более ранних суток истории; если его нет нигде — климатология",
                   "в истории есть пропуски (маска), а формула учебника предполагает "
                   "полный ряд"),
        Difference("интервал — нормальный с климатологическим масштабом, а не по "
                   "остаткам сезонно-наивного метода",
                   "остатки по истории окна недоступны при короткой истории; "
                   "одинаковый интервал у всех статистических эталонов"),
    ),
)


def recent_anomaly(xT, mT, clim, t, t0,
                   hours=RECENT_HOURS, min_valid=RECENT_MIN_VALID):
    """Средняя аномалия температуры за ``hours`` часов перед моментом t.

    Только по валидным часам; меньше ``min_valid`` валидных — (0, False).
    """
    kh = np.arange(max(t - hours, 0), t)
    mh = (mT[kh] > 0).astype(np.float64)
    if mh.sum() < min_valid:
        return 0.0, False
    doy, hour = window_calendar(t0, kh)
    a = ((xT[kh] - clim.predict(doy, hour)) * mh).sum() / mh.sum()
    return float(a), True


def fit_climatologies(manifest, force=False):
    """Климатологии всех станций манифеста из кэша, без повторного QC и подгонки.
    Возвращает {id: запись станции} с полями clim, x, mask, N, t0, lat, lon, elev, koppen.
    """
    return get_store(manifest, rebuild=force).clims()


def quantiles_from_normal(mu, sigma):
    mu = np.asarray(mu, np.float32)
    sigma = np.asarray(sigma, np.float32)
    return mu[..., None] + ZQ * sigma[..., None]


@describe(CLIMATOLOGY_CARD)
def climatology_forecast(mu_clim_fut, sigma_clim):
    """μ(h) = C(h); q_k(h) = C(h) + z_k·σ_clim."""
    mu = np.asarray(mu_clim_fut, np.float32)
    sig = np.broadcast_to(np.asarray(sigma_clim, np.float32).reshape(-1, 1), mu.shape)
    return mu, quantiles_from_normal(mu, sig)


def damped_coefficients(Sxx, Sxy):
    """r_h = Σ ā·a_h / Σ ā² (МНК через ноль), обрезанные в [0, 1]; при Σ ā² ≈ 0 — 0."""
    Sxx = np.asarray(Sxx, np.float64)
    Sxy = np.asarray(Sxy, np.float64)
    r = np.where(Sxx > 1e-6, Sxy / np.maximum(Sxx, 1e-6), 0.0)
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def fit_damped_persistence(clims, n_windows=20000, seed=0):
    """Коэффициенты r_h по обучающему окну станций ``clims`` (передавать обучающие).

    На каждой станции — ``n_windows / число станций`` случайных моментов t в её
    обучающем окне; пара (ā(t), a(t+h)) входит в сумму лида h, только если час t+h
    валиден (маска цели, блок 1).
    """
    rng = np.random.default_rng(seed)
    sids = list(clims.keys())
    Sxx = np.zeros(H)
    Sxy = np.zeros(H)
    per = max(1, n_windows // len(sids))
    for sid in sids:
        s = clims[sid]
        lo, hi = time_bounds(s["N"])["train"]
        if hi - lo < 24 + H + 1:
            continue
        clim = s["clim"]
        t0 = s["t0"]
        xT, mT = s["x"][:, 0], s["mask"][:, 0]
        for _ in range(per):
            t = int(rng.integers(lo + 24, hi - H))
            anom_recent, ok = recent_anomaly(xT, mT, clim, t, t0)
            if not ok:
                continue
            kf = np.arange(t, t + H)
            mf = (mT[kf] > 0).astype(np.float64)
            doy_f, hour_f = window_calendar(t0, kf)
            anom_fut = (xT[kf] - clim.predict(doy_f, hour_f)) * mf
            Sxx += anom_recent ** 2 * mf
            Sxy += anom_recent * anom_fut
    return damped_coefficients(Sxx, Sxy)


@describe(DAMPED_CARD)
def damped_persistence_forecast(a_recent, mu_clim_fut, sigma_clim, r):
    """Затухающая персистентность аномалии.

        μ(h)   = C(h) + r_h · ā
        σ(h)   = σ_clim · √max(1 − r_h², 0.02)
        q_k(h) = μ(h) + z_k · σ(h)

    ā — ``recent_anomaly`` (средняя аномалия за 24 ч), r_h — ``fit_damped_persistence``.
    Для AR(1)-аномалии a(t+1) = φ·a(t) + ε с Var a = σ_clim² и предиктором a(t)
    это в точности условное среднее и условный разброс при r_h = φ^h.
    """
    a = np.asarray(a_recent, np.float32)[:, None]
    r = np.asarray(r, np.float32)
    mu = np.asarray(mu_clim_fut, np.float32) + r[None, :] * a
    sig = np.asarray(sigma_clim, np.float32)[:, None] * np.sqrt(
        np.clip(1 - r[None, :] ** 2, DAMPED_VAR_FLOOR, 1.0))
    return mu, quantiles_from_normal(mu, sig)


@describe(SEASONAL_NAIVE_CARD)
def seasonal_naive_forecast(x_hist, mask_hist, mu_clim_fut, sigma_clim, period=24):
    """Сезонно-наивный прогноз с периодом ``period`` по истории окна.

    История выровнена по правому краю: последний час истории — момент t−1, лид h
    (с нуля) — момент t+h. Слот суток j = h mod period; для слота берутся последние
    сутки истории, в которых этот слот валиден; если таких нет — C(h).
    """
    B, Lh = x_hist.shape[:2]
    D = Lh // period
    T = x_hist[:, Lh - D * period:, 0].reshape(B, D, period)[:, ::-1]
    v = (mask_hist[:, Lh - D * period:, 0] > 0.5).reshape(B, D, period)[:, ::-1]
    has = v.any(axis=1)
    k = v.argmax(axis=1)
    last = np.take_along_axis(T, k[:, None, :], axis=1)[:, 0]
    slot = np.arange(H) % period
    mu = np.where(has[:, slot], last[:, slot], np.asarray(mu_clim_fut, np.float32))
    mu = mu.astype(np.float32)
    sig = np.broadcast_to(np.asarray(sigma_clim, np.float32).reshape(-1, 1), mu.shape)
    return mu, quantiles_from_normal(mu, sig)
