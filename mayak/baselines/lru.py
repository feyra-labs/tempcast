"""Бейзлайн с линейной рекуррентной памятью без разложения на якорь и аномалию.

Стек блоков с диагональной комплексной линейной рекуррентностью читает историю. Его
последний выход - сводка истории, из которой общая по лидам голова строит прогноз.
Климат-поля нет: всё, что модель знает о сезоне и суточном ходе на горизонте, она
выводит сама из календаря, координат и истории.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.baselines.neural import N_RECURRENT_INPUT, LeadHead, recurrent_inputs
from mayak.config import LRUConfig

RECURRENT_PARAMS = ("nu_log", "theta_log", "gamma_log", "B_re", "B_im")


def lambda_polar(nu_log, theta_log):
    """Скорость затухания и угловая частота собственных чисел из сырых параметров.

    Обе величины положительны при любых сырых значениях: это экспоненты от них.

    Args:
        nu_log: логарифм скорости затухания, форма (N,).
        theta_log: логарифм угловой частоты, форма (N,).

    Returns:
        Пара тензоров формы (N,): скорость затухания за час и поворот за час в радианах.
    """
    return torch.exp(nu_log), torch.exp(theta_log)


def lambda_power(nu, theta, p):
    """Собственные числа в степени p, вещественная и мнимая части.

    Степень считается сразу в полярной форме, без накопления произведений.

    Args:
        nu: скорость затухания за час, форма (N,).
        theta: поворот за час, форма (N,).
        p: показатели степени, произвольная форма.

    Returns:
        Пара тензоров формы (..., N).
    """
    p = p[..., None].to(nu.dtype)
    mag = torch.exp(-p * nu)
    return mag * torch.cos(p * theta), mag * torch.sin(p * theta)


def _shift(x, k):
    """Сдвиг по оси времени на k часов в прошлое с нулями в начале."""
    return F.pad(x[:, :-k], (0, 0, k, 0))


def lru_scan_associative(a_re, a_im, u_re, u_im):
    """Линейная рекуррентность с постоянным коэффициентом, параллельный скан.

    Каждое состояние равно предыдущему, умноженному на коэффициент, плюс вход часа;
    состояние до первого часа нулевое. Скан удваивает шаг на каждой итерации и делает
    число итераций, равное логарифму длины по основанию два. Возводятся в степень только
    коэффициенты с модулем не больше единицы, поэтому переполнения нет на любой длине.

    Args:
        a_re: вещественная часть коэффициента, форма (N,).
        a_im: мнимая часть коэффициента, форма (N,).
        u_re: вещественная часть входа, форма (B, L, N).
        u_im: мнимая часть входа, форма (B, L, N).

    Returns:
        Пара тензоров формы (B, L, N): вещественная и мнимая части состояний.
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
    """Та же рекуррентность, развёрнутая блоками фиксированной длины.

    Внутри блока состояния получаются одной свёрткой со степенями собственных чисел.
    Перенос состояния между блоками - параллельный скан с коэффициентом, равным
    собственному числу в степени длины блока. Для градиента хранится память, линейная
    по длине окна, а не с лишним логарифмическим множителем, как у скана по всей длине.

    Args:
        nu: скорость затухания за час, форма (N,).
        theta: поворот за час, форма (N,).
        u_re: вещественная часть входа, форма (B, L, N).
        u_im: мнимая часть входа, форма (B, L, N).
        chunk: длина блока в часах.

    Returns:
        Пара тензоров формы (B, L, N): вещественная и мнимая части состояний.
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
    """Та же рекуррентность простым циклом по часам, эталон для проверки сканов."""
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
    """Рекуррентное ядро: диагональная комплексная линейная рекуррентность.

    Вход часа проецируется в комплексное состояние, состояние затухает и поворачивается
    на каждом часе, выход - вещественная часть проекции состояния плюс вход, умноженный
    на обучаемый вектор. Модули собственных чисел при инициализации равномерно заполняют
    заданный диапазон по квадрату модуля, фазы - заданный диапазон частот. Проекция входа
    масштабируется так, чтобы разброс состояния не зависел от скорости затухания.

    Args:
        d_model: размер входа и выхода.
        d_state: число комплексных собственных чисел.
        r_min: наименьший модуль собственного числа при инициализации.
        r_max: наибольший модуль собственного числа при инициализации.
        max_phase: наибольший поворот за час при инициализации, радианы.
        scan: способ развёртки по умолчанию.
        chunk: длина блока для блочной развёртки.
    """

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
        """Модули и фазы собственных чисел, для проверок и журнала."""
        nu, theta = lambda_polar(self.nu_log, self.theta_log)
        return torch.exp(-nu), theta

    def states(self, x, scan=None):
        """Комплексные состояния рекуррентности.

        Считаются в точности параметров, без понижения точности при смешанном обучении.

        Args:
            x: вход, форма (B, L, d_model).
            scan: способ развёртки; None - способ из конструктора.

        Returns:
            Пара тензоров формы (B, L, d_state).

        Raises:
            ValueError: неизвестный способ развёртки.
        """
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
    """Блок стека: нормализация, рекуррентное ядро, нелинейность с затвором, остаток.

    Args:
        cfg: конфиг LRU.
    """

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


class LRUForecaster(nn.Module):
    """Стек линейных рекуррентных блоков над историей и общая по лидам голова.

    Параметры рекуррентности (затухание, фаза, масштаб и проекция входа) обучаются без
    весового затухания, остальные - с базовым затуханием протокола.

    Args:
        cfg: конфиг LRU, словарь с теми же полями или None для значений по умолчанию.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = LRUConfig() if cfg is None else cfg
        if not isinstance(cfg, LRUConfig):
            cfg = LRUConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.embed = nn.Linear(N_RECURRENT_INPUT, cfg.d_model)
        self.blocks = nn.ModuleList([LRUBlock(cfg) for _ in range(cfg.layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = LeadHead(cfg.d_model, cfg.head_hidden, cfg.horizon, self.nq)

    def inputs(self, batch):
        """Вход стека, форма (B, L, N_RECURRENT_INPUT)."""
        return recurrent_inputs(batch)

    def encode(self, batch, scan=None):
        """Выход стека на каждом часе истории, форма (B, L, d_model)."""
        z = self.embed(self.inputs(batch))
        for blk in self.blocks:
            z = blk(z, scan)
        return self.out_norm(z)

    def forward(self, batch, scan=None):
        return self.head(self.encode(batch, scan)[:, -1], batch)

    def optim_groups(self, weight_decay):
        """Группы весового затухания: рекуррентные параметры без затухания, остальные с ним."""
        rec, other = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (rec if name.rsplit(".", 1)[-1] in RECURRENT_PARAMS else other).append(p)
        return [dict(name="recurrent", params=rec, weight_decay=0.0),
                dict(name="other", params=other, weight_decay=weight_decay)]
