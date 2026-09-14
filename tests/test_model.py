import math

import torch

from mayak.constants import L_MAX
from mayak.model import MAYAK, M, H, astro_features
from mayak.loss import mayak_loss


def _toy_batch(B=2, L=L_MAX, seed=0):
    g = torch.Generator().manual_seed(seed)

    x_hist = torch.zeros(B, L_MAX, 3)
    mask_hist = torch.zeros(B, L_MAX, 3)
    if L > 0:
        x_hist[:, L_MAX - L:, 0] = 15 + torch.randn(B, L, generator=g)  # T
        x_hist[:, L_MAX - L:, 1] = 1013 + torch.randn(B, L, generator=g)  # P
        x_hist[:, L_MAX - L:, 2] = 60 + 10 * torch.randn(B, L, generator=g)  # RH
        x_hist[:, L_MAX - L:, 2].clamp_(1, 100)
        mask_hist[:, L_MAX - L:] = 1.0

    doy_hist = (torch.rand(B, L_MAX, generator=g) * 365).float()
    hour_hist = (torch.rand(B, L_MAX, generator=g) * 24).float()
    doy_fut = (torch.rand(B, H, generator=g) * 365).float()
    hour_fut = (torch.rand(B, H, generator=g) * 24).float()

    return {
        "lat": torch.rand(B, generator=g) * 120 - 60,
        "lon": torch.rand(B, generator=g) * 360 - 180,
        "elev": torch.rand(B, generator=g) * 500,
        "x_hist": x_hist, "mask_hist": mask_hist,
        "doy_hist": doy_hist, "hour_hist": hour_hist,
        "doy_fut": doy_fut, "hour_fut": hour_fut,
        "y": torch.randn(B, H, generator=g),
    }


def test_quantiles_monotone():
    model = MAYAK()
    out = model(_toy_batch())

    assert (out["q"].diff(dim=-1) >= 0).all(), (
        "квантили должны быть монотонны"
    )


def test_cold_start_is_climatology():
    model = MAYAK()
    out = model(_toy_batch(L=0))

    assert out["o"].abs().mean() < 1e-3


def test_batch_stream_equivalence():
    torch.manual_seed(0)

    model = MAYAK()
    batch = _toy_batch()

    x = batch["x_hist"]
    mask = batch["mask_hist"]

    loc = model.loc(
        batch["lat"],
        batch["lon"],
        batch["elev"],
    )

    astro_h = astro_features(
        batch["doy_hist"],
        batch["hour_hist"],
        batch["lat"][:, None],
        batch["lon"][:, None],
    )

    mu0, sg0, df0 = model.field.evaluate(
        model.field.coefficients(loc),
        astro_h,
    )

    ch, _, vt = model.build_channels(
        x,
        mask,
        astro_h,
        mu0,
        sg0,
        df0,
    )

    feats = model.encoder(ch)

    # Пакетная реализация
    batch_re, batch_im, batch_e = model.readout(
        feats,
        vt,
    )

    # Потоковая реализация
    Bsz = feats.shape[0]

    state = (
        torch.zeros(Bsz, M, dtype=feats.dtype, device=feats.device),
        torch.zeros(Bsz, M, dtype=feats.dtype, device=feats.device),
        torch.zeros(Bsz, M, dtype=feats.dtype, device=feats.device),
    )

    for k in range(feats.shape[1]):
        state = model.readout.step(
            state,
            feats[:, k],
            vt[:, k],
        )

    stream_n_re, stream_n_im, stream_e = state
    _, _, kappa = model.readout.constants()

    den = stream_e + kappa[None, :]

    stream_re = stream_n_re / den
    stream_im = stream_n_im / den

    max_delta_re = (batch_re - stream_re).abs().max()
    max_delta_im = (batch_im - stream_im).abs().max()
    max_delta_e = ((batch_e.double() - stream_e.double()).abs() / stream_e.double().abs().clamp(min=1.0)).max()

    assert max_delta_re < 1e-4, (
        f"batch/stream mismatch for a_re: "
        f"max|Δ|={max_delta_re.item():.6g}"
    )

    assert max_delta_im < 1e-4, (
        f"batch/stream mismatch for a_im: "
        f"max|Δ|={max_delta_im.item():.6g}"
    )

    assert max_delta_e < 1e-4, (
        f"batch/stream mismatch for e: "
        f"max|Δ|={max_delta_e.item():.6g}"
    )


def test_backward():
    """
    Smoke-тест полного forward/backward прохода.
    """
    torch.manual_seed(0)

    model = MAYAK()
    batch = _toy_batch()

    out = model(batch)

    hf = torch.arange(1, H + 1, dtype=torch.float32)

    y = (12.0 + 6.0 * torch.sin(2.0 * math.pi * (L_MAX + hf) / 24.0) + torch.randn(2, H))

    loss = mayak_loss(out, y)

    assert torch.isfinite(loss), "loss должен быть конечным"

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in grads), (
        "ни один обучаемый параметр не получил gradient"
    )

    assert all(
        torch.isfinite(g).all()
        for g in grads
        if g is not None
    ), "градиенты должны быть конечными"
