"""PatchTST — трансформер над патчами ряда с независимостью каналов и RevIN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.baselines.cards import BaselineCard, Difference, Source, describe
from mayak.baselines.neural import QUANTILE_HEAD, median_centered_offsets
from mayak.config import PatchTSTConfig

REVIN_EPS = 1e-5
LOG_SIG_CLAMP = (-5.0, 3.0)


def masked_instance_stats(x, m, min_valid=2, eps=REVIN_EPS):
    """RevIN по валидным часам: (B, L) → (μ, σ), каждое (B, 1).

    σ = √(Var + eps) с дисперсией без поправки Бесселя, как в RevIN. Меньше
    ``min_valid`` валидных часов — (0, 1). Статистики отсоединены от графа, как в
    оригинале (``.detach()``).
    """
    x, m = x.detach(), m.detach().to(x.dtype)
    n = m.sum(-1, keepdim=True)
    ok = n >= min_valid
    mean = (x * m).sum(-1, keepdim=True) / n.clamp_min(1)
    var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / n.clamp_min(1)
    std = torch.sqrt(var + eps)
    return torch.where(ok, mean, torch.zeros_like(mean)), torch.where(ok, std, torch.ones_like(std))


def make_patches(z, patch_len, stride, padding):
    """(B, L, C) → (B, n_patches, C·patch_len): в патче сначала P отсчётов канала 0,
    затем канала 1, …; паддинг «end» — повтор последнего шага S раз."""
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
    """Многоголовое внимание с residual attention: к логитам слоя прибавляются логиты
    предыдущего слоя (``prev``), как в _MultiheadAttention оригинала."""

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
    """Post-norm слой энкодера оригинала: MHA → Add & Norm → FFN(GELU) → Add & Norm."""

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


PATCHTST_CARD = BaselineCard(
    key="patchtst", name="PatchTST", kind="neural",
    summary="Трансформер над патчами ряда температуры: разбиение на перекрывающиеся "
            "патчи, независимость каналов, обратимая нормализация экземпляра (RevIN) "
            "и Flatten-голова на весь горизонт.",
    sources=(Source("Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam",
                    "A Time Series is Worth 64 Words: Long-term Forecasting with "
                    "Transformers", "ICLR 2023", "arXiv:2211.14730"),
             Source("T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo",
                    "Reversible Instance Normalization for Accurate Time-Series "
                    "Forecasting against Distribution Shift", "ICLR 2022")),
    code=("https://github.com/yuqinie98/PatchTST (PatchTST_supervised/layers/"
          "PatchTST_backbone.py, layers/RevIN.py)",),
    taken=("разбиение на патчи: длина 16, шаг 8, паддинг «end» повторением последнего "
           "значения S раз (+1 патч)",
           "независимость каналов: каждый канал обрабатывается одной и той же сетью "
           "отдельно, каналы не смешиваются",
           "RevIN: вычитание среднего и деление на σ = √(Var + 1e-5) экземпляра, "
           "статистики без градиента, обратное преобразование на выходе; без аффинных "
           "параметров (affine = 0 в скриптах авторов)",
           "энкодер TSTiEncoder: линейное вложение патча W_P, обучаемое позиционное "
           "кодирование W_pos ~ U(−0.02, 0.02), post-norm слои с BatchNorm, FFN с GELU, "
           "residual attention (логиты предыдущего слоя прибавляются к текущим)",
           "Flatten_Head: flatten (d_model × n_patches) → Linear на горизонт",
           "гиперпараметры авторов для набора Weather: 3 слоя, d_model 128, 16 голов, "
           "d_ff 256, dropout 0.2, head_dropout 0"),
    differences=(
        Difference("вход с маской валидности: статистики RevIN — только по валидным "
                   "часам, невалидные часы после нормализации — ноль, в вложение патча "
                   "подаётся [значения патча, маска патча] (W_P: 2P → d_model)",
                   "в оригинале пропусков нет; маска — общий вход всех моделей "
                   "(блок 1), без неё пропуск неотличим от значения, равного среднему"),
        Difference("одна переменная — температура",
                   "при независимости каналов прогноз температуры не зависит от истории "
                   "давления и влажности; их прогноз в оригинале нам не нужен"),
        Difference("при пустой или почти пустой истории (меньше 2 валидных часов) "
                   "нормировка (μ, σ) = (0, 1)",
                   "статистика экземпляра не определена; в оригинале такого окна нет"),
        QUANTILE_HEAD,
        Difference("голова выдаёт на каждый лид медиану и log σ в нормированных "
                   "единицах; квантили строятся до обратной RevIN",
                   "обратная RevIN — умножение на σ > 0 и сдвиг: монотонность квантилей "
                   "и эквивариантность к масштабу входа сохраняются"),
        Difference("длина входа 672 ч (84 патча) вместо 336 (42 патча)",
                   "контракт данных проекта; статья показывает, что PatchTST выигрывает "
                   "от длинного окна (PatchTST/64 — 512 ч)"),
    ),
    notes=("Параметров около 4 млн, из них ≈ 3.6 млн — Flatten-голова "
           "(d_model·n_patches × 2·H), как и в оригинале при длинном входе.",),
)


@describe(PATCHTST_CARD)
class PatchTST(nn.Module):
    """PatchTST-бейзлайн (конфиг - PatchTSTConfig)."""

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
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(
            cfg.d_model * self.n_patches, 2 * cfg.horizon), nn.Dropout(cfg.head_dropout))
        self.gaps = nn.Parameter(torch.zeros(cfg.horizon, self.nq - 1))

    def normalize(self, batch):
        """Температура и её маска на входном окне → (z·mask, mask, μ, σ)."""
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
        o = self.head(u.transpose(1, 2)).view(-1, 2, self.horizon)
        mu_n = o[:, 0]
        sig_n = torch.exp(o[:, 1].clamp(*LOG_SIG_CLAMP))
        offs = median_centered_offsets(self.gaps, self.nq, u.device)
        q_n = mu_n[..., None] + sig_n[..., None] * offs[None]
        q = q_n * std[..., None] + mean[..., None]
        return {"q": q, "mu": mu_n * std + mean, "sigma": sig_n * std}


__all__ = ["PATCHTST_CARD", "PatchTST", "make_patches", "masked_instance_stats"]
