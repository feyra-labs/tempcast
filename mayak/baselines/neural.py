"""Нейробейзлайны GRU seq2seq и DLinear и общая квантильная параметризация."""
import torch
import torch.nn as nn

from mayak.baselines.cards import BaselineCard, Difference, Source, describe
from mayak.config import DLinearConfig, GRUConfig

T_SCALE, P_REF, P_SCALE, RH_REF, RH_SCALE = 30.0, 1013.0, 50.0, 50.0, 50.0
COORD_SCALE = (90.0, 180.0, 1000.0)

QUANTILE_HEAD = Difference(
    "квантильная голова на 7 квантилей (медиана, масштаб, монотонные смещения по лиду) "
    "вместо точечного прогноза; функция потерь — общий нормированный pinball",
    "проект сравнивает вероятностные прогнозы одной функцией потерь (блоки 1, 4); "
    "точечная модель не даёт ни CRPS, ни покрытия")


def median_centered_offsets(gaps_param, nq, device):
    """Монотонные смещения квантилей (H, nq) с нулём на медиане."""
    gaps = torch.nn.functional.softplus(gaps_param)
    offs = torch.cat([torch.zeros(gaps.shape[0], 1, device=device), torch.cumsum(gaps, -1)], -1)
    return offs - offs[:, nq // 2:nq // 2 + 1]


_median_centered_offsets = median_centered_offsets


def normalized_obs(x):
    """(B, L, 3) T, P, RH → фиксированная нормировка (не зависит от окна)."""
    T, P, RH = x[..., 0], x[..., 1], x[..., 2]
    return torch.stack([T / T_SCALE, (P - P_REF) / P_SCALE, (RH - RH_REF) / RH_SCALE], dim=-1)


def coord_features(batch, device):
    """(B, 3): широта, долгота, высота в долях масштаба."""
    coord = torch.stack([batch["lat"], batch["lon"], batch["elev"]], -1)
    return coord / torch.tensor(COORD_SCALE, device=device)


GRU_CARD = BaselineCard(
    key="gru", name="GRU seq2seq", kind="neural",
    summary="Рекуррентный кодировщик истории (GRU) и прямая голова, отображающая "
            "последнее скрытое состояние сразу в весь горизонт.",
    sources=(Source("K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, "
                    "H. Schwenk, Y. Bengio",
                    "Learning Phrase Representations using RNN Encoder-Decoder for "
                    "Statistical Machine Translation", "EMNLP 2014", "arXiv:1406.1078"),
             Source("I. Sutskever, O. Vinyals, Q. V. Le",
                    "Sequence to Sequence Learning with Neural Networks", "NeurIPS 2014",
                    "arXiv:1409.3215")),
    taken=("блок GRU (Cho и соавт., 2014) — двухслойный кодировщик истории",
           "схема «кодировщик → вектор фиксированной длины → прогноз» (Sutskever и "
           "соавт., 2014)"),
    differences=(
        Difference("декодирование не авторегрессионное: последнее состояние "
                   "кодировщика отображается MLP-головой сразу во все 168 лидов "
                   "(direct multi-horizon); «seq2seq» в названии — только кодировщик",
                   "авторегрессионный декодер на 168 шагов требует teacher forcing "
                   "и накапливает ошибку; прямое отображение — стандартное упрощение "
                   "в многогоризонтном прогнозе, и оно названо здесь явно"),
        Difference("на входе каждого часа: T, P, RH с фиксированной нормировкой, "
                   "умноженные на маску, сама маска и координаты станции",
                   "вход с маской валидности (блок 1): пропуск отличим от нуля"),
        QUANTILE_HEAD,
    ),
)


@describe(GRU_CARD)
class GRUSeq2Seq(nn.Module):
    """GRU-кодировщик с прямой головой на весь горизонт (конфиг - GRUConfig)."""
    N_INPUT = 3 + 3 + 3

    def __init__(self, cfg=None):
        super().__init__()
        cfg = GRUConfig() if cfg is None else cfg
        if not isinstance(cfg, GRUConfig):
            cfg = GRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.gru = nn.GRU(input_size=self.N_INPUT, hidden_size=cfg.hidden,
                          num_layers=cfg.layers, batch_first=True)
        self.head_mu = nn.Sequential(nn.Linear(cfg.hidden + 3, cfg.mu_hidden), nn.GELU(),
                                     nn.Linear(cfg.mu_hidden, cfg.horizon))
        self.head_sig = nn.Sequential(nn.Linear(cfg.hidden + 3, cfg.sigma_hidden), nn.GELU(),
                                      nn.Linear(cfg.sigma_hidden, cfg.horizon))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def forward(self, batch):
        x = batch["x_hist"]; m = batch["mask_hist"]
        xn = normalized_obs(x)
        coord = coord_features(batch, x.device)
        coord_seq = coord[:, None, :].expand(-1, x.shape[1], -1)

        inp = torch.cat([xn * m, m, coord_seq], dim=-1)

        h, _ = self.gru(inp)
        last = h[:, -1]
        feat = torch.cat([last, coord], dim=-1)
        mu = self.head_mu(feat)
        log_sig = self.head_sig(feat).clamp(-2, 4)
        sig = torch.exp(log_sig)
        offs = median_centered_offsets(self.gaps, self.nq, x.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}


DLINEAR_CARD = BaselineCard(
    key="dlinear", name="DLinear", kind="neural",
    summary="Разложение ряда температуры скользящим средним на тренд и остаток и "
            "два независимых линейных отображения «история → горизонт», результаты "
            "суммируются.",
    sources=(Source("A. Zeng, M. Chen, L. Zhang, Q. Xu",
                    "Are Transformers Effective for Time Series Forecasting?", "AAAI 2023",
                    "arXiv:2205.13504"),),
    code=("https://github.com/cure-lab/LTSF-Linear (models/DLinear.py)",),
    taken=("разложение скользящим средним с ядром 25 и паддингом краёв повторением "
           "крайнего значения (moving_avg + series_decomp оригинала)",
           "два независимых nn.Linear по временной оси: Linear_Trend и Linear_Seasonal, "
           "выход — их сумма; инициализация весов — по умолчанию PyTorch, как в оригинале",
           "нормализации входа нет (RevIN и вычитания последнего значения в DLinear нет; "
           "с ними это NLinear/RevIN-вариант — другая модель семейства LTSF-Linear)"),
    differences=(
        Difference("одноканальный вход: только температура (режим univariate «S» "
                   "оригинала, individual не применим)",
                   "прогнозируется одна температура, а в DLinear каналы не смешиваются — "
                   "давление и влажность не могли бы повлиять на прогноз T"),
        Difference("длина входа 672 ч вместо 336 в основных таблицах статьи",
                   "контракт данных проекта — 28 суток истории у всех моделей; статья "
                   "показывает, что DLinear выигрывает от длинного окна (до 720)"),
        Difference("пропуски истории заполняются нулём (T·mask) до разложения",
                   "в оригинале пропусков нет; ноль — единственное заполнение без "
                   "интерполяции, а решение о маске модели принимает сама (блок 1)"),
        QUANTILE_HEAD,
        Difference("масштаб интервала — обучаемый параметр на лид, не зависящий от входа",
                   "у DLinear нет нелинейностей, из которых можно было бы взять "
                   "условный разброс, не меняя модель"),
    ),
)


@describe(DLINEAR_CARD)
class DLinear(nn.Module):
    """DLinear: разложение скользящим средним и два линейных отображения по времени
    (конфиг - DLinearConfig: длина входа и ядро скользящего среднего)."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = DLinearConfig() if cfg is None else cfg
        if not isinstance(cfg, DLinearConfig):
            cfg = DLinearConfig.from_dict(cfg)
        self.cfg = cfg
        self.k = cfg.kernel
        self.nq = cfg.n_quantiles
        self.input_len = cfg.input_len
        self.lin_trend = nn.Linear(cfg.input_len, cfg.horizon)
        self.lin_resid = nn.Linear(cfg.input_len, cfg.horizon)
        self.log_sig = nn.Parameter(torch.zeros(cfg.horizon))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def decompose(self, T):
        """(B, L) → (тренд, остаток): скользящее среднее с паддингом повторением краёв."""
        pad = self.k // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(T[:, None], (pad, pad), mode="replicate"),
            kernel_size=self.k, stride=1)[:, 0]
        return trend, T - trend

    def point(self, T):
        trend, resid = self.decompose(T)
        return self.lin_trend(trend) + self.lin_resid(resid)

    def forward(self, batch):
        T = (batch["x_hist"][..., 0] * batch["mask_hist"][..., 0])[:, -self.input_len:]
        mu = self.point(T)
        sig = torch.exp(self.log_sig).clamp(0.3, 12)[None].expand_as(mu)
        offs = median_centered_offsets(self.gaps, self.nq, T.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}
