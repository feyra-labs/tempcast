"""МАЯК как четыре графа без состояния - контракт компилируемого рантайма.

Монолитный граф (scripts/export_onnx.py) на каждый выпуск пересчитывает всё окно
истории. Рантайм на устройстве вызывает модель в четырёх разных ритмах, поэтому
модель режется на четыре графа, и всё состояние ходит через их входы и выходы:

* ``init``     - раз при старте: координаты → признаки точки ``loc``, коэффициенты
  климат-поля без паспорта и паспорт холодного старта ``z0``;
* ``step``     - раз в час: хвост сырого окна (CHANNEL_MAX_LAG + 1 ч), календарь
  часа, буфер энкодера, накопленное состояние мод → новый буфер, новое состояние мод
  и строка суточного накопителя (aT, dP24, vt, vp24) этого часа;
* ``passport`` - раз в сутки: 24 строки накопителя → суточная сводка, сдвиг ряда
  сводок, новый паспорт ``z``;
* ``issue``    - по запросу: состояние мод, паспорт, календарь горизонта → квантили
  модели (до калибровки).

Каждый граф - тонкая обёртка над методами MAYAK, которыми пользуются пакетный путь и
эталонный потоковый рантайм (``StreamingMayak``): build_channels, encoder.step_shift,
readout.step, daily_summaries, passport, issue. Своей арифметики в обёртках нет.

Хост-код (Rust: runtime-rs/; его исполняемая спецификация на Python - ``GraphRuntime``)
держит кольцо сырого окна, буфер энкодера, суточный накопитель, календарь, QC точки,
калибровку и сериализацию. Буфер энкодера в графе хранится в хронологическом порядке
(старший час первым) и сдвигается внутри графа - индекс кольца не нужен.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from mayak.astro import astro_features
from mayak.config import CHANNEL_MAX_LAG, N_DAILY_SUMMARY

GRAPH_FORMAT = 1
GRAPH_NAMES = ("init", "step", "passport", "issue")
GRAPH_IO = {
    "init": (("lat", "lon", "elev"),
             ("loc", "c_mu", "c_sig", "c_def", "z0")),
    "step": (("x_ctx", "m_ctx", "doy", "hour", "lat", "lon", "c_mu", "c_sig", "c_def",
              "enc_buf", "n_re", "n_im", "e"),
             ("enc_buf_out", "n_re_out", "n_im_out", "e_out", "day_row")),
    "passport": (("loc", "day_acc", "day_summ", "day_mask"),
                 ("day_summ_out", "day_mask_out", "z")),
    "issue": (("loc", "lat", "lon", "z", "n_re", "n_im", "e", "doy_fut", "hour_fut"),
              ("q",)),
}
DAY_ROW = ("aT", "dP24", "vt", "vp24")
CTX = CHANNEL_MAX_LAG + 1


class _Graph(nn.Module):
    """Обёртка в режиме eval. torch.onnx.export восстанавливает режим экспортируемого
    модуля рекурсивно: обёртка в режиме train после экспорта переводит в train и общую
    модель - паспорт начинает сэмплировать z, и пакетный путь перестаёт совпадать с
    потоковым. Поэтому eval выставляется у самой обёртки."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.eval()


class InitGraph(_Graph):
    def forward(self, lat, lon, elev):
        m, D = self.m, self.m.cfg.history_days
        loc = m.loc(lat[:, 0], lon[:, 0], elev[:, 0])
        c_mu, c_sig, c_def = m.field.coefficients(loc)
        z0, _ = m.passport(loc, loc.new_zeros(1, D, N_DAILY_SUMMARY), loc.new_zeros(1, D),
                           sample=False)
        return loc, c_mu, c_sig, c_def, z0


class StepGraph(_Graph):
    """Один час. Календарь хвоста окна не нужен: выход - последний момент хвоста, а
    лаговые каналы зависят только от P и маски. doy/hour часа растягиваются на хвост."""

    def forward(self, x_ctx, m_ctx, doy, hour, lat, lon, c_mu, c_sig, c_def,
                enc_buf, n_re, n_im, e):
        m = self.m
        n = x_ctx.shape[1]
        astro_h = astro_features(doy.expand(-1, n), hour.expand(-1, n), lat, lon)
        mu0, sg0, df0 = m.field.evaluate((c_mu, c_sig, c_def), astro_h)
        ch, aT, vt = m.build_channels(x_ctx, m_ctx, astro_h, mu0, sg0, df0)
        dp24, vp24 = m.channel(ch, "dP24"), m.lag_valid(m_ctx[..., 1], 24)
        feat, enc_buf = m.encoder.step_shift(ch[..., -1], enc_buf)
        n_re, n_im, e = m.readout.step((n_re, n_im, e), feat, vt[:, -1])
        day_row = torch.stack([aT[:, -1], dp24[:, -1], vt[:, -1], vp24[:, -1]], dim=-1)
        return enc_buf, n_re, n_im, e, day_row


class PassportGraph(_Graph):
    def forward(self, loc, day_acc, day_summ, day_mask):
        m = self.m
        summ, has = m.daily_summaries(day_acc[:, 0], day_acc[:, 1], day_acc[:, 2],
                                      day_acc[:, 3])
        day_summ = torch.cat([day_summ[:, 1:], summ], dim=1)
        day_mask = torch.cat([day_mask[:, 1:], has], dim=1)
        z, _ = m.passport(loc, day_summ, day_mask, sample=False)
        return day_summ, day_mask, z


class IssueGraph(_Graph):
    def forward(self, loc, lat, lon, z, n_re, n_im, e, doy_fut, hour_fut):
        m = self.m
        a_re, a_im = m.readout.normalize(n_re, n_im, e)
        astro_f = astro_features(doy_fut, hour_fut, lat, lon)
        return m.issue(loc, z, a_re, a_im, e, astro_f)["q"]


GRAPH_MODULES = dict(init=InitGraph, step=StepGraph, passport=PassportGraph, issue=IssueGraph)


def dims(cfg):
    """Размеры входов и выходов графов и состояния хоста для конфига модели."""
    from mayak.modules.loc import LocEncoder
    loc_dim = LocEncoder(cfg.loc_freqs).out_dim
    enc_buf = (cfg.encoder_kernel - 1) * sum(cfg.encoder_dilations)
    return dict(horizon=cfg.horizon, n_quantiles=cfg.n_quantiles, n_modes=cfg.n_modes,
                passport_dim=cfg.passport_dim, history_days=cfg.history_days,
                n_daily_summary=N_DAILY_SUMMARY, day_row=len(DAY_ROW),
                stream_window=cfg.stream_window, ctx=CTX, loc_dim=loc_dim,
                encoder_width=cfg.encoder_width, enc_buf_len=enc_buf)


def example_inputs(model, seed=0):
    """Правдоподобные входы всех графов (для трассировки и сверки экспорта)."""
    cfg = model.cfg
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    lat, lon, elev = torch.tensor([[52.37]]), torch.tensor([[4.9]]), torch.tensor([[12.0]])
    with torch.no_grad():
        loc, c_mu, c_sig, c_def, z0 = (t.detach() for t in InitGraph(model)(lat, lon, elev))
    D, M, W = cfg.history_days, cfg.n_modes, cfg.encoder_width
    Hh = cfg.horizon
    x_ctx = torch.stack([8 + 5 * r(1, CTX), 1005 + 5 * r(1, CTX),
                         (70 + 10 * r(1, CTX)).clamp(1, 100)], dim=-1)
    m_ctx = (torch.rand(1, CTX, 3, generator=g) < 0.85).float()
    hoy = 4000 + torch.arange(1, Hh + 1, dtype=torch.float32)[None]
    enc_buf = 0.5 * r(1, W, sum(model.encoder.buffer_pads))
    modes = [r(1, M), r(1, M), 5 + torch.rand(1, M, generator=g)]
    return dict(
        init=(lat, lon, elev),
        step=(x_ctx * m_ctx, m_ctx, torch.tensor([[166.5]]), torch.tensor([[12.0]]),
              lat, lon, c_mu, c_sig, c_def, enc_buf, *modes),
        passport=(loc, torch.cat([r(1, 2, 24), (torch.rand(1, 2, 24, generator=g) < 0.8).float()],
                                 dim=1),
                  0.5 * r(1, D, N_DAILY_SUMMARY), (torch.rand(1, D, generator=g) < 0.9).float()),
        issue=(loc, lat, lon, z0 + 0.3 * r(*z0.shape), *modes, hoy / 24.0, hoy % 24),
    )


def export_graphs(model, out_dir, *, conformal=None, aci=None, int8=False, opset=17,
                  check_atol=1e-4, seed=0):
    """Четыре графа ONNX + manifest.json (+ conformal.f32) в out_dir → манифест.

    conformal - сплит-конформная таблица (бины лидов × квантили) или путь к .npy; в
    каталог пишется развёрнутая по лидам таблица (horizon × NQ, float32 LE) - то, что
    рантайм прибавляет к квантилям. aci - ``ACIParams`` или None.
    Каждый граф сверяется с PyTorch на example_inputs (max|Δ| ≤ check_atol).

    Входы, от которых граф не зависит (например, loc в passport или в issue при абляции
    no_anchor), экспорт выбрасывает; манифест перечисляет фактические входы графа, и
    хост подаёт только их.
    """
    import onnx
    import onnxruntime as ort
    from mayak.data.qc import PHYS
    from mayak.metrics import I_MED, conformal_table
    from mayak.provenance import provenance
    from mayak.runtime.streaming import RAW_CHANNELS, STATE_HEADER, STATE_VERSION

    model = model.eval()
    cfg = model.cfg
    os.makedirs(out_dir, exist_ok=True)
    inputs = example_inputs(model, seed)
    graphs, checks = {}, {}
    was_training = model.training
    for name in GRAPH_NAMES:
        mod = GRAPH_MODULES[name](model)
        names_in, names_out = GRAPH_IO[name]
        path = os.path.join(out_dir, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod, inputs[name], path, input_names=list(names_in),
                              output_names=list(names_out), opset_version=opset,
                              dynamo=False, do_constant_folding=True)
            ref = mod(*inputs[name])
        ref = ref if isinstance(ref, tuple) else (ref,)
        proto = onnx.load(path)
        used = [i.name for i in proto.graph.input]
        meta = proto.metadata_props.add()
        meta.key, meta.value = "mayak_graph", name
        onnx.save(proto, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {n: t.detach().numpy() for n, t in zip(names_in, inputs[name])
                              if n in used})
        err = max(float(np.abs(o - r.detach().numpy()).max()) if o.size else 0.0
                  for o, r in zip(out, ref))
        if err > check_atol:
            raise RuntimeError(f"граф {name}: max|ONNX − PyTorch| = {err:.2e} > {check_atol}")
        checks[name] = err
        entry = dict(fp32=f"{name}.onnx", inputs=used, outputs=list(names_out),
                     shapes_in={n: list(t.shape) for n, t in zip(names_in, inputs[name])},
                     shapes_out={n: list(r.shape) for n, r in zip(names_out, ref)})
        if int8:
            entry["int8"] = quantize_graph(path)
        graphs[name] = entry

    model.train(was_training)
    cal = dict(conformal=None, aci=None)
    if conformal is not None:
        shift = np.load(conformal) if isinstance(conformal, str) else np.asarray(conformal)
        table = conformal_table(np.asarray(shift, np.float32), cfg.horizon).astype("<f4")
        table.tofile(os.path.join(out_dir, "conformal.f32"))
        cal["conformal"] = "conformal.f32"
    if aci is not None:
        cal["aci"] = dict(target=aci.target, gamma=aci.gamma, max_factor=aci.max_factor,
                          interval=list(aci.interval))
    from mayak.baselines.statistical import ZQ
    d = dims(cfg)
    d["n_coef"] = [int(inputs["step"][i].shape[-1]) for i in (6, 7, 8)]
    manifest = dict(
        format=GRAPH_FORMAT, model_config=cfg.to_dict(), dims=d,
        quantiles=list(cfg.quantiles), i_med=I_MED, zq=[float(v) for v in ZQ],
        raw_channels=list(RAW_CHANNELS), phys={c: list(PHYS[c]) for c in RAW_CHANNELS},
        state=dict(version=STATE_VERSION, header_bytes=STATE_HEADER.itemsize,
                   nbytes=state_nbytes(cfg)),
        graphs=graphs, calibration=cal, export_check_max_abs=checks, opset=opset,
        provenance=provenance())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest


def quantize_graph(path):
    """Динамическая int8-квантизация весов графа → имя файла рядом с исходным."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out = path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(path, out, weight_type=QuantType.QInt8,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    return os.path.basename(out)


def state_nbytes(cfg):
    from mayak.runtime.streaming import STATE_HEADER
    M, D, W = cfg.n_modes, cfg.history_days, cfg.stream_window
    return (STATE_HEADER.itemsize + 4 * (3 * M + cfg.passport_dim) + 2 * (D * 6 + D)
            + 2 * W * 3 + W * 3)


# Исполняемая спецификация хоста: та же логика, что в runtime-rs/src/runtime.rs.

class TorchBackend:
    def __init__(self, model):
        self.mods = {n: GRAPH_MODULES[n](model) for n in GRAPH_NAMES}

    @torch.no_grad()
    def run(self, name, *args):
        out = self.mods[name](*(torch.from_numpy(np.ascontiguousarray(a)) for a in args))
        out = out if isinstance(out, tuple) else (out,)
        return [o.detach().numpy() for o in out]


class OnnxBackend:
    def __init__(self, model_dir, precision="fp32", threads=1):
        import onnxruntime as ort
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess, self.names = {}, {}
        for n in GRAPH_NAMES:
            g = self.manifest["graphs"][n]
            self.sess[n] = ort.InferenceSession(os.path.join(model_dir, g[precision]), so,
                                                providers=["CPUExecutionProvider"])
            self.names[n] = set(g["inputs"])

    def run(self, name, *args):
        feed = {k: np.ascontiguousarray(a, np.float32) for k, a in zip(GRAPH_IO[name][0], args)
                if k in self.names[name]}
        return self.sess[name].run(None, feed)


class GraphRuntime:
    """Хост поверх четырёх графов: кольцо сырого окна, буфер энкодера, моды, сутки.

    Без калибровки и сериализации (они проверяются эталонными векторами против
    ``StreamingMayak`` и ``mayak.metrics``). Нужен, чтобы отделить ошибку разбиения
    модели на графы от ошибки хоста на Rust: GraphRuntime(TorchBackend) обязан совпасть
    с StreamingMayak, GraphRuntime(OnnxBackend) - с ним же в пределах допуска экспорта.
    """

    def __init__(self, backend, cfg, lat, lon, elev):
        self.b, self.cfg = backend, cfg
        f = lambda v: np.array([[v]], np.float32)
        self.lat, self.lon = f(lat), f(lon)
        self.loc, *self.coefs, self.z0 = backend.run("init", self.lat, self.lon, f(elev))
        self.reset()

    def reset(self):
        c = self.cfg
        W, M, D = c.stream_window, c.n_modes, c.history_days
        self.raw_x = np.zeros((W, 3), np.float32)
        self.raw_m = np.zeros((W, 3), np.float32)
        self.head = 0
        self.enc_buf = np.zeros((1, c.encoder_width, (c.encoder_kernel - 1)
                                 * sum(c.encoder_dilations)), np.float32)
        self.n_re, self.n_im, self.e = (np.zeros((1, M), np.float32) for _ in range(3))
        self.day_summ = np.zeros((1, D, N_DAILY_SUMMARY), np.float32)
        self.day_mask = np.zeros((1, D), np.float32)
        self.day_acc = np.zeros((1, len(DAY_ROW), 24), np.float32)
        self.hours_in_day = 0
        self.z = self.z0.copy()

    def step(self, T, P, RH, doy, hour):
        from mayak.data.qc import point_qc
        x, m = point_qc(T, P, RH)
        W, j = self.cfg.stream_window, self.head
        self.raw_x[j], self.raw_m[j] = np.where(m > 0, x, 0.0), m
        self.head = (j + 1) % W
        idx = (self.head - CTX + np.arange(CTX)) % W
        f = lambda v: np.array([[v]], np.float32)
        self.enc_buf, self.n_re, self.n_im, self.e, row = self.b.run(
            "step", self.raw_x[idx][None], self.raw_m[idx][None], f(doy), f(hour),
            self.lat, self.lon, *self.coefs, self.enc_buf, self.n_re, self.n_im, self.e)
        self.day_acc[0, :, self.hours_in_day] = row[0]
        self.hours_in_day += 1
        if self.hours_in_day == 24:
            self.day_summ, self.day_mask, self.z = self.b.run(
                "passport", self.loc, self.day_acc, self.day_summ, self.day_mask)
            self.day_acc[:] = 0.0
            self.hours_in_day = 0

    def forecast(self, doy_fut, hour_fut):
        f = lambda a: np.asarray(a, np.float32)[None]
        (q,) = self.b.run("issue", self.loc, self.lat, self.lon, self.z, self.n_re, self.n_im,
                          self.e, f(doy_fut), f(hour_fut))
        return q[0]


__all__ = ["CTX", "DAY_ROW", "GRAPH_IO", "GRAPH_NAMES", "GraphRuntime", "OnnxBackend",
           "TorchBackend", "dims", "example_inputs", "export_graphs", "quantize_graph",
           "state_nbytes"]
