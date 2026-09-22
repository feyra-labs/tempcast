"""Linear Recurrent Unit — бейзлайн «линейная память без разложения»."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.baselines.cards import BaselineCard, Difference, Source, describe
from mayak.baselines.neural import (QUANTILE_HEAD, T_SCALE, coord_features, median_centered_offsets,
                                    normalized_obs)
from mayak.config import LRUConfig

N_CALENDAR = 4
N_INPUT = 3 + 3 + N_CALENDAR + 3
N_LEAD = N_CALENDAR + 3 + 1
DAYS_IN_YEAR = 365.25
RECURRENT_PARAMS = ("nu_log", "theta_log", "gamma_log", "B_re", "B_im")


def calendar_features(doy, hour):
    """(…,) день года и час UTC → (…, 4): sin/cos суточной и годовой фазы."""
    wd = 2 * math.pi * hour / 24.0
    wy = 2 * math.pi * doy / DAYS_IN_YEAR
    return torch.stack([torch.sin(wd), torch.cos(wd), torch.sin(wy), torch.cos(wy)], dim=-1)


def lambda_polar(nu_log, theta_log):
    """(модуль-логарифм, фаза): λ = exp(−ν + iθ), ν = exp(ν_log) > 0, θ = exp(θ_log)."""
    return torch.exp(nu_log), torch.exp(theta_log)


def lambda_power(nu, theta, p):
    """λ^p в полярной форме: exp(−pν)·(cos pθ + i sin pθ). p (…,) × состояние (N,)."""
    p = p[..., None].to(nu.dtype)
    mag = torch.exp(-p * nu)
    return mag * torch.cos(p * theta), mag * torch.sin(p * theta)


def _shift(x, k):
    """x[:, t] → x[:, t − k] с нулями при t < k (ось времени — 1)."""
    return F.pad(x[:, :-k], (0, 0, k, 0))


def lru_scan_associative(a_re, a_im, u_re, u_im):
    """h_t = a ⊙ h_{t−1} + u_t, h_{−1} = 0 — скан Хиллиса–Стила, ⌈log₂ L⌉ шагов.

    a (N,) не зависит от времени; u (B, L, N). После шага с k = 2^j в h[t] собрана сумма
    по окну длины 2k, и a^k возводится в квадрат. Умножаются только |a| ≤ 1 — без
    переполнения на любой длине.
    """
    h_re, h_im = u_re, u_im
    L = h_re.shape[1]
    k = 1
    while k < L:
        s_re, s_im = _shift(h_re, k), _shift(h_im, k)
        h_re, h_im = h_re + a_re * s_re - a_im * s_im, h_im + a_re * s_im + a_im * s_re
        a_re, a_im = a_re * a_re - a_im * a_im, 2 * a_re * a_im
        k *= 2
    return h_re, h_im


def lru_scan_chunked(nu, theta, u_re, u_im, chunk=32):
    """Та же рекуррентность блоками длины ``chunk``.

    Внутри блока h = T·u, T[i, j] = λ^{i−j} при i ≥ j (тёплицева, степени — в полярной
    форме, без накопления произведений); перенос состояния между блоками — скан
    Хиллиса–Стила с коэффициентом λ^C. Память — O(B·L·N), а не O(B·L·N·log L).
    """
    B, L, N = u_re.shape
    C = min(int(chunk), L)
    pad = (-L) % C
    if pad:
        u_re, u_im = F.pad(u_re, (0, 0, pad, 0)), F.pad(u_im, (0, 0, pad, 0))
    nC = (L + pad) // C
    u_re, u_im = u_re.reshape(B, nC, C, N), u_im.reshape(B, nC, C, N)

    idx = torch.arange(C, device=u_re.device)
    diff = idx[:, None] - idx[None, :]
    k_re, k_im = lambda_power(nu, theta, diff.clamp_min(0))
    low = (diff >= 0).to(k_re.dtype)[..., None]
    k_re, k_im = k_re * low, k_im * low
    y_re = torch.einsum("ijn,bcjn->bcin", k_re, u_re) - torch.einsum("ijn,bcjn->bcin", k_im, u_im)
    y_im = torch.einsum("ijn,bcjn->bcin", k_re, u_im) + torch.einsum("ijn,bcjn->bcin", k_im, u_re)

    if nC > 1:
        aC_re, aC_im = lambda_power(nu, theta, torch.tensor(float(C), device=u_re.device))
        s_re, s_im = lru_scan_associative(aC_re, aC_im, y_re[:, :, -1], y_im[:, :, -1])
        s_re, s_im = _shift(s_re, 1), _shift(s_im, 1)
        p_re, p_im = lambda_power(nu, theta, (idx + 1).to(nu.dtype))
        y_re = y_re + p_re * s_re[:, :, None] - p_im * s_im[:, :, None]
        y_im = y_im + p_re * s_im[:, :, None] + p_im * s_re[:, :, None]
    y_re, y_im = y_re.reshape(B, nC * C, N), y_im.reshape(B, nC * C, N)
    return y_re[:, pad:], y_im[:, pad:]


def lru_recurrent(a_re, a_im, u_re, u_im):
    """Наивная рекуррентность циклом по часам — эталон для проверки сканов."""
    B, L, N = u_re.shape
    h_re = u_re.new_zeros(B, N)
    h_im = u_re.new_zeros(B, N)
    out_re, out_im = [], []
    for t in range(L):
        h_re, h_im = a_re * h_re - a_im * h_im + u_re[:, t], a_re * h_im + a_im * h_re + u_im[:, t]
        out_re.append(h_re)
        out_im.append(h_im)
    return torch.stack(out_re, 1), torch.stack(out_im, 1)


class LRULayer(nn.Module):
    """Один LRU (рекуррентное ядро), инициализация — как в minimal-LRU."""

    def __init__(self, d_model, d_state, r_min, r_max, max_phase, scan="chunked", chunk=32):
        super().__init__()
        self.scan, self.chunk = scan, chunk
        u1, u2 = torch.rand(d_state), torch.rand(d_state)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(max_phase * u2.clamp_min(1e-4))
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        mod = torch.exp(-torch.exp(nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - mod ** 2)))
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(2 * d_model))
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(2 * d_model))
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.randn(d_model))

    def eigenvalues(self):
        """(|λ|, arg λ) — для проверок и журнала."""
        nu, theta = lambda_polar(self.nu_log, self.theta_log)
        return torch.exp(-nu), theta

    def states(self, x, scan=None):
        """(B, L, d_model) → комплексные состояния (h_re, h_im), (B, L, d_state).

        Считается в точности параметров (float32 при обучении), вне autocast."""
        scan = scan or self.scan
        x = x.to(self.B_re.dtype)
        g = torch.exp(self.gamma_log)[:, None]
        u_re, u_im = x @ (self.B_re * g).T, x @ (self.B_im * g).T
        nu, theta = lambda_polar(self.nu_log, self.theta_log)
        if scan == "chunked":
            return lru_scan_chunked(nu, theta, u_re, u_im, self.chunk)
        a_re, a_im = torch.exp(-nu) * torch.cos(theta), torch.exp(-nu) * torch.sin(theta)
        if scan == "associative":
            return lru_scan_associative(a_re, a_im, u_re, u_im)
        if scan == "recurrent":
            return lru_recurrent(a_re, a_im, u_re, u_im)
        raise ValueError(f"неизвестная развёртка {scan!r}")

    def forward(self, x, scan=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            h_re, h_im = self.states(x, scan)
            return h_re @ self.C_re.T - h_im @ self.C_im.T + self.D * x.to(self.D.dtype)


class LRUBlock(nn.Module):
    """pre-LayerNorm → LRU → GELU → dropout → GLU → dropout → остаток (SequenceLayer)."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lru = LRULayer(cfg.d_model, cfg.d_state, cfg.r_min, cfg.r_max, cfg.max_phase,
                            cfg.scan, cfg.chunk)
        self.out1 = nn.Linear(cfg.d_model, cfg.d_model)
        self.out2 = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, scan=None):
        z = self.lru(self.norm(x), scan).to(x.dtype)
        z = self.drop(F.gelu(z))
        z = self.out1(z) * torch.sigmoid(self.out2(z))
        return x + self.drop(z)


LRU_CARD = BaselineCard(
    key="lru", name="LRU", kind="neural",
    summary="Стек блоков Linear Recurrent Unit над историей наблюдений: обучаемая "
            "линейная память с комплексными диагональными модами, но без разложения "
            "прогноза на климатический якорь и аномалию. Прогноз на каждый лид — "
            "MLP-голова над последним выходом стека, координатами и календарём лида.",
    sources=(Source("A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, "
                    "S. De", "Resurrecting Recurrent Neural Networks for Long Sequences",
                    "ICML 2023", "arXiv:2303.06349"),),
    code=("https://github.com/NicolasZucchet/minimal-LRU (lru/model.py) — эталонная "
          "JAX-реализация, на которую ссылаются авторы; официальной реализации авторы "
          "не публиковали",),
    taken=("комплексная диагональная рекуррентность h_t = λ ⊙ h_{t−1} + γ ⊙ B u_t, "
           "y_t = Re(C h_t) + D ⊙ u_t",
           "экспоненциальная параметризация λ = exp(−exp(ν) + i·exp(θ)) и "
           "инициализация |λ|² ~ U[r_min², r_max²], arg λ ~ U[0, max_phase] "
           "(nu_init, theta_init)",
           "нормализация входа по модулю: γ = √(1 − |λ|²) при инициализации, далее "
           "обучаемый γ_log (gamma_log_init)",
           "инициализация B ~ N(0, 1/(2·d_model)), C ~ N(0, 1/d_state), D ~ N(0, 1)",
           "блок SequenceLayer: нормализация → LRU → GELU → dropout → GLU "
           "out1(x)·σ(out2(x)) → dropout → остаток",
           "развёртка ассоциативным сканом (у авторов — jax.lax.associative_scan)",
           "нулевое затухание весов для ν, θ, γ, B (группа «ssm» в minimal-LRU)"),
    differences=(
        Difference("r_min, r_max и max_phase заданы через постоянные времени "
                   "[3, 240] ч и наименьший период 12 ч — тот же диапазон, что у мод "
                   "МАЯК (в minimal-LRU по умолчанию r_min = 0, r_max = 1, "
                   "max_phase = 2π)",
                   "эксперимент проверяет разложение, а не априорную память: обе модели "
                   "стартуют с одинаковыми постоянными времени"),
        Difference("LayerNorm перед блоком (pre-norm), а не BatchNorm после",
                   "BatchNorm смешивает статистику по батчу и по времени, то есть "
                   "не причинна (то же решение для энкодера МАЯК, блок 7); pre-norm "
                   "устойчивее на 4 слоях без подбора"),
        Difference("скорость обучения рекуррентных параметров не уменьшена "
                   "(в minimal-LRU lr_factor = 0.5)",
                   "единый протокол (блок 4): одна скорость обучения на все модели; "
                   "архитектура определяет только группы весового затухания"),
        Difference("блочный скан: внутри блока из 32 ч — тёплицева свёртка степенями λ "
                   "в полярной форме, между блоками — скан Хиллиса–Стила",
                   "в PyTorch нет associative_scan; полный скан Хиллиса–Стила на 672 ч "
                   "хранит для градиента O(L·log L) промежуточных тензоров, блочный — "
                   "O(L); результат тот же (проверяется тестом против цикла)"),
        Difference("вход часа: T, P, RH с фиксированной нормировкой, умноженные на маску, "
                   "маска, календарь (sin/cos суток и года) и координаты; выход — "
                   "последний шаг стека, а не среднее по времени",
                   "вход с маской валидности и те же сведения о точке и времени, что "
                   "у МАЯК; задача — прогноз из конца истории, а не классификация"),
        Difference("прямая многогоризонтная голова: общий для всех лидов MLP над "
                   "[последний выход стека, координаты, календарь лида, доля лида]",
                   "без авторегрессии и без климатического якоря: всё, что модель знает "
                   "о сезоне и суточном ходе в будущем, она выводит сама"),
        QUANTILE_HEAD,
    ),
    notes=("Рекуррентность LRU сама по себе даёт O(d_state) на новый час в потоке; "
           "потоковый рантайм для LRU не строится — это бейзлайн точности.",),
)


@describe(LRU_CARD)
class LRUForecaster(nn.Module):
    """LRU-бейзлайн (конфиг - LRUConfig).

    Группы весового затухания: ``recurrent`` (ν, θ, γ, B всех блоков) — без затухания,
    ``other`` — базовое затухание протокола.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = LRUConfig() if cfg is None else cfg
        if not isinstance(cfg, LRUConfig):
            cfg = LRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.embed = nn.Linear(N_INPUT, cfg.d_model)
        self.blocks = nn.ModuleList([LRUBlock(cfg) for _ in range(cfg.layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Linear(cfg.d_model + N_LEAD, cfg.head_hidden), nn.GELU(),
                                  nn.Linear(cfg.head_hidden, cfg.head_hidden), nn.GELU(),
                                  nn.Linear(cfg.head_hidden, 2))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def inputs(self, batch):
        """(B, L, 13): наблюдения·маска, маска, календарь часа, координаты."""
        x, m = batch["x_hist"], batch["mask_hist"]
        L = x.shape[1]
        coord = coord_features(batch, x.device)
        cal = calendar_features(batch["doy_hist"][:, -L:], batch["hour_hist"][:, -L:])
        return torch.cat([normalized_obs(x) * m, m, cal.to(x.dtype),
                          coord[:, None, :].expand(-1, L, -1)], dim=-1)

    def encode(self, batch, scan=None):
        z = self.embed(self.inputs(batch))
        for blk in self.blocks:
            z = blk(z, scan)
        return self.out_norm(z)

    def forward(self, batch, scan=None):
        x = batch["x_hist"]
        last = self.encode(batch, scan)[:, -1]
        B, Hh = x.shape[0], self.horizon
        coord = coord_features(batch, x.device)
        lead = (torch.arange(Hh, device=x.device, dtype=x.dtype) + 1) / Hh
        feat = torch.cat([last[:, None, :].expand(-1, Hh, -1),
                          calendar_features(batch["doy_fut"], batch["hour_fut"]).to(x.dtype),
                          coord[:, None, :].expand(-1, Hh, -1),
                          lead[None, :, None].expand(B, -1, -1)], dim=-1)
        o = self.head(feat)
        mu = T_SCALE * o[..., 0]
        sig = torch.exp(o[..., 1].clamp(-2, 4))
        offs = median_centered_offsets(self.gaps, self.nq, x.device)
        q = mu[..., None] + sig[..., None] * offs[None]
        return {"q": q, "mu": mu, "sigma": sig}

    def optim_groups(self, weight_decay):
        rec, other = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (rec if name.rsplit(".", 1)[-1] in RECURRENT_PARAMS else other).append(p)
        return [dict(name="recurrent", params=rec, weight_decay=0.0),
                dict(name="other", params=other, weight_decay=weight_decay)]
