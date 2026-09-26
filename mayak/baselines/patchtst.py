"""Трансформер над патчами ряда температуры с обратимой нормализацией окна."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.baselines.neural import median_centered_offsets
from mayak.config import PatchTSTConfig

REVIN_EPS = 1e-5
LOG_SIG_CLAMP = (-5.0, 3.0)


def masked_instance_stats(x, m, min_valid=2, eps=REVIN_EPS):
    """Среднее и разброс окна по валидным часам для нормализации экземпляра.

    Разброс - корень из смещённой дисперсии с малой добавкой. Если валидных часов меньше
    ``min_valid``, среднее равно нулю, а разброс единице: статистика окна не определена.
    Градиент через статистики не идёт.

    Args:
        x: ряд, форма (B, L).
        m: маска валидности, форма (B, L).
        min_valid: наименьшее число валидных часов для статистики.
        eps: добавка к дисперсии.

    Returns:
        Пара тензоров формы (B, 1): среднее и разброс.
    """
    x, m = x.detach(), m.detach().to(x.dtype)
    n = m.sum(-1, keepdim=True)
    ok = n >= min_valid
    mean = (x * m).sum(-1, keepdim=True) / n.clamp_min(1)
    var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / n.clamp_min(1)
    std = torch.sqrt(var + eps)
    return torch.where(ok, mean, torch.zeros_like(mean)), torch.where(ok, std, torch.ones_like(std))


def make_patches(z, patch_len, stride, padding):
    """Разбиение окна на патчи по оси времени.

    В патче сначала идут отсчёты первого канала, затем второго. Дополнение ``end``
    повторяет последний час окна столько раз, каков шаг, и даёт ещё один патч.

    Args:
        z: окно, форма (B, L, C).
        patch_len: длина патча в часах.
        stride: шаг между началами патчей в часах.
        padding: ``end`` или ``none``.

    Returns:
        Тензор формы (B, n_patches, C * patch_len).
    """
    if padding == "end":
        z = torch.cat([z, z[:, -1:].expand(-1, stride, -1)], dim=1)
    p = z.unfold(1, patch_len, stride)
    return p.reshape(p.shape[0], p.shape[1], -1)


class _Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


def _norm(kind, d):
    if kind == "batch":
        return nn.Sequential(_Transpose(), nn.BatchNorm1d(d), _Transpose())
    return nn.LayerNorm(d)


class ResidualAttention(nn.Module):
    """Многоголовое внимание, к логитам которого прибавляются логиты предыдущего слоя.

    Args:
        d_model: ширина представления патча.
        n_heads: число голов.
        attn_dropout: прореживание весов внимания.
        proj_dropout: прореживание выхода.
    """

    def __init__(self, d_model, n_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = self.dk ** -0.5

    def forward(self, x, prev=None):
        B, T, _ = x.shape
        q = self.W_Q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if prev is not None:
            scores = scores + prev
        w = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (w @ v).transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.to_out(out), scores


class TSTEncoderLayer(nn.Module):
    """Слой энкодера: внимание и перцептрон, нормализация после каждой остаточной связи.

    Args:
        cfg: конфиг PatchTST.
    """

    def __init__(self, cfg):
        super().__init__()
        self.res_attention = cfg.res_attention
        self.attn = ResidualAttention(cfg.d_model, cfg.n_heads, cfg.attn_dropout, cfg.dropout)
        self.drop_attn = nn.Dropout(cfg.dropout)
        self.norm_attn = _norm(cfg.norm, cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.d_ff, cfg.d_model))
        self.drop_ffn = nn.Dropout(cfg.dropout)
        self.norm_ffn = _norm(cfg.norm, cfg.d_model)

    def forward(self, x, prev=None):
        a, scores = self.attn(x, prev if self.res_attention else None)
        x = self.norm_attn(x + self.drop_attn(a))
        x = self.norm_ffn(x + self.drop_ffn(self.ff(x)))
        return x, scores


class PatchTST(nn.Module):
    """Трансформер над патчами ряда температуры.

    Окно температуры нормируется своими средним и разбросом по валидным часам и режется
    на патчи. Каждый патч вместе со своей маской линейно вкладывается, получает
    обучаемое позиционное смещение и проходит слои энкодера. Медиана на весь горизонт -
    линейная голова над всеми патчами, развёрнутыми в один вектор. Масштаб интервала -
    отдельная малая линейная голова над средним представлением патчей. Квантили
    строятся в нормированных единицах и возвращаются в градусы обратным преобразованием.
    Давление, влажность, календарь и координаты в модель не входят.

    Args:
        cfg: конфиг PatchTST, словарь с теми же полями или None для значений по
            умолчанию.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = PatchTSTConfig() if cfg is None else cfg
        if not isinstance(cfg, PatchTSTConfig):
            cfg = PatchTSTConfig.from_dict(cfg)
        self.cfg = cfg
        self.nq = cfg.n_quantiles
        self.horizon = cfg.horizon
        self.n_patches = cfg.n_patches
        self.W_P = nn.Linear(2 * cfg.patch_len, cfg.d_model)
        W_pos = torch.empty(self.n_patches, cfg.d_model)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TSTEncoderLayer(cfg) for _ in range(cfg.layers)])
        self.head_mu = nn.Sequential(nn.Flatten(start_dim=-2),
                                     nn.Linear(cfg.d_model * self.n_patches, cfg.horizon),
                                     nn.Dropout(cfg.head_dropout))
        self.head_sigma = nn.Linear(cfg.d_model, cfg.horizon)
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def normalize(self, batch):
        """Нормированная температура окна входа.

        Args:
            batch: батч окон.

        Returns:
            Четвёрка: нормированная температура с нулями на невалидных часах и маска,
            обе формы (B, input_len), затем среднее и разброс окна формы (B, 1).
        """
        L = self.cfg.input_len
        x = batch["x_hist"][:, -L:, 0]
        m = (batch["mask_hist"][:, -L:, 0] > 0).to(x.dtype)
        x = torch.where(m > 0, x, torch.zeros_like(x))
        if self.cfg.revin:
            mean, std = masked_instance_stats(x, m, self.cfg.revin_min_valid)
        else:
            mean, std = torch.zeros_like(x[:, :1]), torch.ones_like(x[:, :1])
        return (x - mean) / std * m, m, mean, std

    def encode(self, batch):
        """Представления патчей формы (B, n_patches, d_model), среднее и разброс окна."""
        z, m, mean, std = self.normalize(batch)
        c = self.cfg
        p = make_patches(torch.stack([z, m], -1), c.patch_len, c.stride, c.padding_patch)
        u = self.W_P(p)
        u = self.drop(u + self.W_pos)
        scores = None
        for layer in self.layers:
            u, scores = layer(u, scores)
        return u, mean, std

    def forward(self, batch):
        u, mean, std = self.encode(batch)
        mu_n = self.head_mu(u.transpose(1, 2))
        sig_n = torch.exp(self.head_sigma(u.mean(dim=1)).clamp(*LOG_SIG_CLAMP))
        offs = median_centered_offsets(self.gaps, self.nq, u.device)
        q_n = mu_n[..., None] + sig_n[..., None] * offs[None]
        q = q_n * std[..., None] + mean[..., None]
        return {"q": q, "mu": mu_n * std + mean, "sigma": sig_n * std}


__all__ = ["PatchTST", "make_patches", "masked_instance_stats"]
