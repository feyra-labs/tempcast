import math

import pytest
import torch

from mayak.config import ModelConfig
from mayak.constants import L_MAX, H, QUANTILES
from mayak.model import MAYAK, astro_features
from mayak.modules.heads import R_MAX

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


def _shaken_model(seed=7, scale=0.5):
    """Модель, у которой встряхнуты все слои с нулевой или особой инициализацией.

    Головы, подстройка мод паспортом, модуляция поля паспортом и сам паспорт получают
    заметный шум. На свежей модели выходной слой голов нулевой, и поправка равна нулю
    тривиально; здесь она нулевой быть не обязана.
    """
    torch.manual_seed(seed)
    model = MAYAK().eval()
    with torch.no_grad():
        for mod in (model.heads, model.propagator, model.field.film, model.passport):
            for p in mod.parameters():
                p.add_(scale * torch.randn_like(p))
    return model


def test_cold_start_median_equals_field_with_shaken_heads():
    """Без истории медиана равна климат-полю побитно при любых весах голов."""
    model = _shaken_model()
    batch = _toy_batch(L=0)
    with torch.no_grad():
        out = model(batch)
        mu_c, _, _ = _climate_field(model, batch, out["z"])
        raw = model.heads.fc2(torch.nn.functional.gelu(model.heads.fc1(torch.zeros(
            1, model.heads.in_dim))))
    assert raw[0, 0].abs() > 1e-3, "тест вырожден: выход поправки голов нулевой и без веса"
    assert torch.equal(out["r"], torch.zeros_like(out["r"]))
    assert torch.equal(out["mu"], mu_c)
    assert torch.equal(out["q"][..., QUANTILES.index(0.5)], mu_c)


@pytest.mark.parametrize("L", [6, 24, 672])
def test_correction_is_nonzero_with_history(L):
    """С историей поправка включается, её модуль ограничен."""
    model = _shaken_model()
    with torch.no_grad():
        out = model(_toy_batch(L=L))
    assert out["r"].abs().max() > 1e-4, f"поправка при L={L} нулевая"
    assert out["r"].abs().max() <= R_MAX


def test_evidence_gate_grows_with_evidence():
    """Вес поправки: ноль без свидетельств, монотонный рост, меньше единицы."""
    heads = MAYAK().heads
    e = torch.tensor([0.0, 1e-6, 1.0, 6.0, 24.0, 100.0, 1e4])[:, None].expand(-1, M)
    with torch.no_grad():
        g = heads.evidence_gate(e)
    assert g[0] == 0.0
    assert (g.diff() > 0).all() and (g < 1.0).all()
    assert g[1] < 1e-5, "исчезающе малая масса не должна включать поправку"


def test_stage_without_history_does_not_train_correction():
    """При L=0 функция потерь не даёт градиента ни выходу поправки, ни её порогу.

    Масштаб интервала и зазоры квантилей при этом обучаются.
    """
    model = _shaken_model().train()
    loss = mayak_loss(model(_toy_batch(L=0)), _toy_batch(L=0))
    loss.backward()
    h = model.heads
    assert torch.equal(h.fc2.weight.grad[0], torch.zeros_like(h.fc2.weight.grad[0]))
    assert h.fc2.bias.grad[0] == 0.0
    assert h.r_kappa.grad is None or h.r_kappa.grad == 0.0
    assert h.fc2.weight.grad[1:].abs().sum() > 0, "масштаб интервала обязан учиться"


def test_site_tau_stays_within_bounds():
    """Постоянные времени после подстройки паспортом не выходят за границы мод."""
    model = MAYAK()
    lo, hi = model.cfg.tau_bounds
    with torch.no_grad():
        model.readout.raw_tau.copy_(torch.linspace(-12.0, 12.0, M))
        model.propagator.site.weight.normal_(0.0, 5.0)
        model.propagator.site.bias.normal_(0.0, 5.0)
        tau, omega, _ = model.readout.constants()
        z = 3.0 * torch.randn(256, model.cfg.passport_dim)
        tau_s, _ = model.propagator.site_constants(z, tau, omega)
    assert tau_s.min() >= lo and tau_s.max() <= hi
    assert tau_s.max() == hi, "тест вырожден: ни одна мода не упёрлась в верхнюю границу"
    slowest = torch.exp(-168.0 / tau_s).max()
    assert slowest <= math.exp(-168.0 / hi) + 1e-6


def test_correction_threshold_is_not_decayed():
    model = MAYAK()
    groups = {g["name"]: {id(p) for p in g["params"]} for g in model.optim_groups(1e-2)}
    assert id(model.heads.r_kappa) in groups["no_decay"]


def test_checkpoint_without_correction_threshold_is_rejected():
    model = MAYAK()
    sd = {k: v for k, v in model.state_dict().items() if k != "heads.r_kappa"}
    with pytest.raises(RuntimeError, match="r_kappa"):
        MAYAK().load_state_dict(sd)


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
