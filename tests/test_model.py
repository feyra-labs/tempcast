import math

import torch

from mayak.config import ModelConfig
from mayak.constants import L_MAX, H, QUANTILES
from mayak.model import MAYAK, astro_features

M = ModelConfig().n_modes
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
        "y_mask": torch.ones(B, H),
        "norm_scale": torch.full((B, H), 3.0),
    }


def test_quantiles_monotone():
    model = MAYAK()
    out = model(_toy_batch())

    assert (out["q"].diff(dim=-1) >= 0).all(), (
        "квантили должны быть монотонны"
    )


def _climate_field(model, batch, z):
    """Климат-поле на часах горизонта - ровно то, из чего issue() строит медиану."""
    loc = model.loc(batch["lat"], batch["lon"], batch["elev"])
    astro_f = astro_features(batch["doy_fut"], batch["hour_fut"],
                             batch["lat"][:, None], batch["lon"][:, None])
    return model.field.evaluate(model.field.coefficients(loc, z), astro_f)


def test_cold_start_is_climatology():
    """При нулевой истории аномалия строго нулевая, а медиана совпадает с климат-полем.

    Нулевой аномалии мало: медиана равна mu_c + sigma_c·(o + r), и если поправка
    голов r не нулевая, прогноз холодного старта уезжает от климатологии, оставаясь
    при этом с o = 0. Проверяются обе величины.
    """
    model = MAYAK().eval()
    batch = _toy_batch(L=0)
    with torch.no_grad():
        out = model(batch)
        mu_c, sigma_c, _ = _climate_field(model, batch, out["z"])

    assert torch.equal(out["o"], torch.zeros_like(out["o"])), (
        "при полностью невалидной истории амплитуды мод обязаны быть нулевыми")
    assert out["o"].abs().mean() < 1e-3
    torch.testing.assert_close(out["sigma_c"], sigma_c, rtol=0, atol=0)
    torch.testing.assert_close(out["mu"], mu_c, rtol=0, atol=1e-5)
    torch.testing.assert_close(out["q"][..., QUANTILES.index(0.5)], out["mu"],
                               rtol=0, atol=0)


def test_cold_start_median_stays_in_the_field_band():
    """Даже с ненулевыми головами холодный старт не уходит от поля дальше 0.6·sigma_c.

    Границу задаёт конструкция голов (r = 0.6·tanh), и она должна держаться при
    любых весах, а не только при нулевой инициализации.
    """
    torch.manual_seed(7)
    model = MAYAK().eval()
    with torch.no_grad():
        for p in model.heads.parameters():
            p.add_(0.5 * torch.randn_like(p))
        batch = _toy_batch(L=0)
        out = model(batch)
        mu_c, sigma_c, _ = _climate_field(model, batch, out["z"])

    dev = (out["mu"] - mu_c).abs()
    assert (dev > 1e-6).any(), "тест вырожден: головы не сдвинули медиану вовсе"
    assert (dev <= 0.6 * sigma_c + 1e-5).all(), (
        f"медиана холодного старта вышла за полосу поля: "
        f"max(|mu − mu_c| / sigma_c) = {(dev / sigma_c).max().item():.4f}")


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
    max_delta_e = ((batch_e.double() - stream_e.double()).abs()
                   / stream_e.double().abs().clamp(min=1.0)).max()

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


def test_batch_stream_equivalence_full_forecast(record_property):
    """Эквивалентность пакета и потока на полном выпуске прогноза по всем лидам.

    Совпадения состояния мод недостаточно: между состоянием и квантилями лежат
    пропагатор, головы и климат-поле, и накопленная разница видна только на
    выходе. Величина расхождения записывается числом в отчёт прогона.
    """
    torch.manual_seed(0)

    model = MAYAK().eval()
    batch = _toy_batch()

    with torch.no_grad():
        loc = model.loc(batch["lat"], batch["lon"], batch["elev"])
        astro_h = astro_features(batch["doy_hist"], batch["hour_hist"],
                                 batch["lat"][:, None], batch["lon"][:, None])
        astro_f = astro_features(batch["doy_fut"], batch["hour_fut"],
                                 batch["lat"][:, None], batch["lon"][:, None])

        mu0, sg0, df0 = model.field.evaluate(model.field.coefficients(loc), astro_h)
        ch, aT, vt = model.build_channels(batch["x_hist"], batch["mask_hist"],
                                          astro_h, mu0, sg0, df0)
        summ, day_mask = model.daily_summaries(
            aT, model.channel(ch, "dP24"), vt,
            model.lag_valid(batch["mask_hist"][..., 1], 24))
        z, _ = model.passport(loc, summ, day_mask, sample=False)
        feats = model.encoder(ch)

        a_re_b, a_im_b, e_b = model.readout(feats, vt)

        Bsz = feats.shape[0]
        state = tuple(torch.zeros(Bsz, M, dtype=feats.dtype) for _ in range(3))
        for k in range(feats.shape[1]):
            state = model.readout.step(state, feats[:, k], vt[:, k])
        n_re, n_im, e_s = state
        a_re_s, a_im_s = model.readout.normalize(n_re, n_im, e_s)

        out_b = model.issue(loc, z, a_re_b, a_im_b, e_b, astro_f)
        out_s = model.issue(loc, z, a_re_s, a_im_s, e_s, astro_f)

    d_q = (out_b["q"] - out_s["q"]).abs()
    per_lead = d_q.amax(dim=(0, 2))
    record_property("batch_stream_max_abs_q", float(d_q.max()))

    assert out_b["q"].shape == (2, H, len(QUANTILES))
    assert per_lead.shape == (H,)
    assert torch.isfinite(d_q).all()
    assert d_q.max() < 1e-3, (
        f"пакет ≠ поток на полном выпуске: max|Δq| = {d_q.max().item():.3g} °C, "
        f"худший лид {int(per_lead.argmax()) + 1} ч")
    assert (out_s["q"].diff(dim=-1) >= 0).all(), "поток обязан давать монотонные квантили"


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

    loss = mayak_loss(out, dict(batch, y=y, y_mask=torch.ones_like(y)))

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
