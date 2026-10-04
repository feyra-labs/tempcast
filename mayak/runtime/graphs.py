"""МАЯК как пять графов без состояния для рантайма устройства.

Устройство вызывает модель в разных ритмах, поэтому модель режется на графы, а всё
состояние ходит через их входы и выходы:

* ``init``   - при старте: признаки точки, коэффициенты климат-поля без паспорта и
  таблица климатологии точки для отката: среднее и масштаб климат-поля с паспортом
  холодного старта на каждый час високосного года;
* ``step``   - раз в час: хвост сырого окна, календарь часа, буфер энкодера, сумма мод по
  хвосту истории и вклад часа, который из хвоста выходит. На выходе новый буфер, новая
  сумма, вклад этого часа в моды для кольца хоста и строка суточного накопителя;
* ``window`` - при загрузке состояния и холодном старте: пакетный проход по всему окну.
  На выходе буфер энкодера, вклады часов хвоста, строки накопителя и сумма мод по хвосту;
* ``resync`` - раз в сутки: точная сумма мод по кольцу вкладов хвоста;
* ``issue``  - по запросу: паспорт по строкам накопителя за всю историю, вклад края
  истории пакетным проходом по его часам, выпуск квантилей до калибровки.

Каждый граф - тонкая обёртка над методами модели, которыми пользуется пакетный путь.
Своей арифметики в обёртках нет. Хост держит сырое окно, кольца вкладов и строк,
причинный контроль качества, калибровку и сериализацию. Буфер энкодера хранится в
хронологическом порядке и сдвигается внутри графа.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from mayak.astro import astro_features
from mayak.runtime.streaming import (CTX, RAW_CHANNELS, RESYNC_HOURS, STATE_HEADER,
                                     STATE_VERSION, StreamingMayak, state_nbytes)

GRAPH_FORMAT = 6
GRAPH_NAMES = ("init", "step", "window", "resync", "issue")
COEFS = ("c_mu", "c_sig", "c_def")
HOURS_OF_YEAR = 366 * 24
GRAPH_IO = {
    "init": (("lat", "lon", "elev"),
             ("loc", "c_mu", "c_sig", "c_def", "z0", "clim_mu", "clim_sig")),
    "step": (("x_ctx", "m_ctx", "doy", "hour", "lat", "lon", *COEFS,
              "enc_buf", "n_re", "n_im", "e", "u_old", "v_old"),
             ("enc_buf_out", "n_re_out", "n_im_out", "e_out", "u", "v", "row")),
    "window": (("x_win", "m_win", "doy_win", "hour_win", "lat", "lon", *COEFS),
               ("enc_buf", "u_tail", "v_tail", "rows", "n_re", "n_im", "e")),
    "resync": (("u_ring", "v_ring"),
               ("n_re", "n_im", "e")),
    "issue": (("loc", "lat", "lon", *COEFS, "rows", "n_re", "n_im", "e",
               "x_edge", "m_edge", "doy_edge", "hour_edge", "doy_fut", "hour_fut"),
              ("q",)),
}
DAY_ROW = ("aT", "dP24", "vt", "vp24")


class _Graph(nn.Module):
    """Обёртка в режиме eval.

    Экспорт восстанавливает режим экспортируемого модуля рекурсивно: обёртка в режиме
    train после экспорта перевела бы в train и общую модель, паспорт начал бы
    сэмплировать, и пакетный путь перестал бы совпадать с потоковым.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.eval()


def year_calendar():
    """Календарь всех часов високосного года в конвенции проекта.

    Номер часа в таблице климатологии совпадает с номером часа от начала года, поэтому
    таблица подходит и обычному году: его часы просто не доходят до последних суток.

    Returns:
        Пара массивов float32 формы (8784,): день года с долей суток и час UTC.
    """
    sec = np.arange(HOURS_OF_YEAR, dtype=np.int64) * 3600
    return (sec / 86400.0).astype(np.float32), ((sec % 86400) / 3600.0).astype(np.float32)


class InitGraph(_Graph):
    """Старт устройства: всё, что зависит только от точки.

    Таблица климатологии - то, что выдаёт модель при пустой истории без мод и поправки:
    климат-поле с паспортом холодного старта. Хост берёт из неё среднее и масштаб по
    часу года, когда выпуск не удался, и своей арифметики модели не делает.
    """

    def __init__(self, model):
        super().__init__(model)
        doy, hour = year_calendar()
        self.register_buffer("year_doy", torch.from_numpy(doy)[None], persistent=False)
        self.register_buffer("year_hour", torch.from_numpy(hour)[None], persistent=False)

    def forward(self, lat, lon, elev):
        m, D = self.m, self.m.cfg.history_days
        loc = m.loc(lat[:, 0], lon[:, 0], elev[:, 0])
        c_mu, c_sig, c_def = m.field.coefficients(loc)
        z0 = m.passport_from_rows(loc, loc.new_zeros(1, D * 24, len(DAY_ROW)))
        # Календарь года привязан к входу нулевым слагаемым: иначе экспорт развернул бы
        # все его производные в константы и граф вырос бы в несколько раз.
        zero = lat * 0.0
        astro = astro_features(self.year_doy + zero, self.year_hour + zero, lat, lon)
        clim_mu, clim_sig, _ = m.field.evaluate(m.field.coefficients(loc, z0), astro)
        return loc, c_mu, c_sig, c_def, z0, clim_mu, clim_sig


class StepGraph(_Graph):
    """Один час.

    Календарь хвоста окна не нужен: выход считается для последнего часа хвоста, а
    каналы с лагом зависят только от давления и маски. Календарь часа растягивается на
    весь хвост.
    """

    def forward(self, x_ctx, m_ctx, doy, hour, lat, lon, c_mu, c_sig, c_def,
                enc_buf, n_re, n_im, e, u_old, v_old):
        m = self.m
        n = x_ctx.shape[1]
        astro_h = astro_features(doy.expand(-1, n), hour.expand(-1, n), lat, lon)
        mu0, sg0, df0 = m.field.evaluate((c_mu, c_sig, c_def), astro_h)
        ch, aT, vt = m.build_channels(x_ctx, m_ctx, astro_h, mu0, sg0, df0)
        dp24, vp24 = m.channel(ch, "dP24"), m.lag_valid(m_ctx[..., 1], 24)
        feat, enc_buf = m.encoder.step_shift(ch[..., -1], enc_buf)
        u = m.readout.project(feat)
        v = vt[:, -1:]
        n_re, n_im, e = m.readout.step_window((n_re, n_im, e), u, v, u_old, v_old,
                                              m.cfg.stream_tail)
        row = torch.stack([aT[:, -1], dp24[:, -1], vt[:, -1], vp24[:, -1]], dim=-1)
        return enc_buf, n_re, n_im, e, u, v, row


class WindowGraph(_Graph):
    """Пакетный проход по всему сырому окну."""

    def forward(self, x_win, m_win, doy_win, hour_win, lat, lon, c_mu, c_sig, c_def):
        m, cfg = self.m, self.m.cfg
        astro_h = astro_features(doy_win, hour_win, lat, lon)
        out = m.history_pass(x_win, m_win, astro_h, (c_mu, c_sig, c_def))
        W, T, L = x_win.shape[1], cfg.stream_tail, cfg.max_history
        u_tail, v_tail = out["u"][:, W - T:], out["v"][:, W - T:]
        n_re, n_im, e = m.readout.accumulate(u_tail, v_tail)
        return out["enc_buf"], u_tail, v_tail, out["rows"][:, W - L:], n_re, n_im, e


class ResyncGraph(_Graph):
    def forward(self, u_ring, v_ring):
        return self.m.readout.accumulate(u_ring, v_ring)


class IssueGraph(_Graph):
    def forward(self, loc, lat, lon, c_mu, c_sig, c_def, rows, n_re, n_im, e,
                x_edge, m_edge, doy_edge, hour_edge, doy_fut, hour_fut):
        m = self.m
        astro_e = astro_features(doy_edge, hour_edge, lat, lon)
        astro_f = astro_features(doy_fut, hour_fut, lat, lon)
        return m.stream_issue(loc, (c_mu, c_sig, c_def), lat, lon, rows, (n_re, n_im, e),
                              x_edge, m_edge, astro_e, astro_f)["q"]


GRAPH_MODULES = dict(init=InitGraph, step=StepGraph, window=WindowGraph, resync=ResyncGraph,
                     issue=IssueGraph)


def dims(cfg):
    """Размеры входов и выходов графов и состояния хоста для конфига модели.

    Args:
        cfg: конфиг модели.

    Returns:
        Словарь размеров.
    """
    from mayak.config import N_DAILY_SUMMARY
    from mayak.modules.loc import LocEncoder
    loc_dim = LocEncoder(cfg.loc_freqs).out_dim
    enc_buf = (cfg.encoder_kernel - 1) * sum(cfg.encoder_dilations)
    return dict(horizon=cfg.horizon, n_quantiles=cfg.n_quantiles, n_modes=cfg.n_modes,
                passport_dim=cfg.passport_dim, history=cfg.max_history,
                history_days=cfg.history_days, n_daily_summary=N_DAILY_SUMMARY,
                day_row=len(DAY_ROW), stream_window=cfg.stream_window,
                stream_edge=cfg.stream_edge, stream_tail=cfg.stream_tail, ctx=CTX,
                loc_dim=loc_dim, encoder_width=cfg.encoder_width, enc_buf_len=enc_buf,
                hours_of_year=HOURS_OF_YEAR)


def example_inputs(model, seed=0):
    """Правдоподобные входы всех графов для трассировки и сверки экспорта.

    Args:
        model: модель МАЯК.
        seed: сид генератора.

    Returns:
        Словарь: имя графа и кортеж его входов.
    """
    cfg = model.cfg
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    lat, lon, elev = torch.tensor([[52.37]]), torch.tensor([[4.9]]), torch.tensor([[12.0]])
    with torch.no_grad():
        loc, c_mu, c_sig, c_def, *_ = (t.detach() for t in InitGraph(model)(lat, lon, elev))
    coefs = (c_mu, c_sig, c_def)
    M, W, Hh = cfg.n_modes, cfg.encoder_width, cfg.horizon
    Wn, E, T, L = cfg.stream_window, cfg.stream_edge, cfg.stream_tail, cfg.max_history

    def obs(n):
        x = torch.stack([(8 + 5 * r(1, n)).round(), (1005 + 5 * r(1, n)),
                         (70 + 10 * r(1, n)).clamp(1, 100).round()], dim=-1)
        m = (torch.rand(1, n, 3, generator=g) < 0.85).float()
        hoy = 4000 + torch.arange(n, dtype=torch.float32)[None]
        return x * m, m, hoy / 24.0, hoy % 24

    x_ctx, m_ctx, _, _ = obs(CTX)
    x_win, m_win, d_win, h_win = obs(Wn)
    x_edge, m_edge, d_edge, h_edge = obs(E)
    hoy_f = 4000 + Wn + torch.arange(Hh, dtype=torch.float32)[None]
    enc_buf = 0.5 * r(1, W, sum(model.encoder.buffer_pads))
    modes = [r(1, M), r(1, M), 5 + torch.rand(1, M, generator=g)]
    vmask = lambda *s: (torch.rand(*s, generator=g) < 0.9).float()
    rows = torch.cat([r(1, L, 2), vmask(1, L, 2)], dim=-1)
    return dict(
        init=(lat, lon, elev),
        step=(x_ctx, m_ctx, torch.tensor([[166.5]]), torch.tensor([[12.0]]), lat, lon, *coefs,
              enc_buf, *modes, r(1, 2 * M), vmask(1, 1)),
        window=(x_win, m_win, d_win, h_win, lat, lon, *coefs),
        resync=(r(1, T, 2 * M), vmask(1, T)),
        issue=(loc, lat, lon, *coefs, rows, *modes, x_edge, m_edge, d_edge, h_edge,
               hoy_f / 24.0, hoy_f % 24),
    )


def calibration_bins():
    """Бины лидов и длины истории калибровки в том виде, в каком они пишутся в манифест.

    Returns:
        Пара списков пар границ включительно: бины лидов и бины длины истории.
    """
    from mayak.constants import HISTORY_BINS
    from mayak.metrics import LEAD_BINS
    return ([[int(a), int(b)] for a, b in LEAD_BINS],
            [[int(b[0]), int(b[1])] for b in HISTORY_BINS])


def conformal_for_export(conformal, checkpoint=None):
    """Таблица поправок для экспорта.

    Args:
        conformal: путь к таблице с записью о подгонке рядом или сама таблица.
        checkpoint: путь к экспортируемому чекпойнту; если задан, таблица должна быть
            подогнана по нему же.

    Returns:
        Таблица float32 по бинам лидов и длины истории.

    Raises:
        ValueError: таблица подогнана по другому чекпойнту или сдвигает медиану.
    """
    from mayak.leakage import file_digest, load_conformal
    from mayak.metrics import check_conformal_shape
    if not isinstance(conformal, str):
        return check_conformal_shape(conformal)
    shift, rec = load_conformal(conformal)
    want = rec.get("checkpoint_digest")
    if checkpoint is not None and want is not None and file_digest(checkpoint) != want:
        raise ValueError(f"конформная таблица {conformal} подогнана по другому чекпойнту "
                         f"({rec.get('checkpoint')}), а экспортируется {checkpoint}")
    return shift


def export_graphs(model, out_dir, *, conformal=None, aci=None, opset=17, check_atol=1e-4,
                  seed=0, checkpoint=None, runtime=None):
    """Экспорт графов, манифеста и развёрнутой по лидам конформной таблицы.

    Таблица в экспорте - подряд по бинам длины истории, в каждом бине строка на каждый
    лид горизонта, в строке по значению на квантиль. Бины лидов адаптивной калибровки и
    бины длины истории таблицы записаны в раздел калибровки манифеста.

    Каждый граф при экспорте сверяется с PyTorch на правдоподобных входах. Входы, от
    которых граф не зависит, экспорт выбрасывает; манифест перечисляет фактические входы
    графа, и хост подаёт только их.

    Args:
        model: модель.
        out_dir: каталог экспорта.
        conformal: таблица поправок по бинам лидов и длины истории или путь к ней с
            записью о подгонке рядом.
        aci: параметры адаптивной калибровки устройства или None.
        opset: версия набора операций ONNX.
        check_atol: допустимое расхождение графа с PyTorch в единицах масштаба выхода:
            наибольшая разность делится на наибольший модуль выхода, но не меньше единицы.
        seed: сид правдоподобных входов для сверки.
        checkpoint: путь к экспортируемому чекпойнту для сверки с записью о подгонке
            таблицы.
        runtime: параметры хоста устройства с порогами смены точки или None - конфиг
            рантайма по умолчанию.

    Returns:
        Манифест экспорта. В разделе рантайма записаны пороги смены точки.

    Raises:
        ValueError: таблица подогнана по другому чекпойнту или сдвигает медиану.
        RuntimeError: граф расходится с PyTorch больше допуска.
    """
    import onnx
    import onnxruntime as ort
    from mayak.data.qc import DEFAULT_QC, PHYS
    from mayak.metrics import I_MED, conformal_table
    from mayak.provenance import provenance
    from mayak.runtime.site import load_runtime_config

    runtime = load_runtime_config() if runtime is None else runtime
    model = model.eval()
    cfg = model.cfg
    if cfg.stream_tail < 1 or cfg.stream_edge < 1:
        raise ValueError(f"история {cfg.max_history} ч не длиннее рецептивного поля энкодера "
                         f"с лагом каналов ({cfg.stream_edge} ч): хвоста для потоковой суммы "
                         f"нет, графы устройства для такой модели не экспортируются")
    table = None if conformal is None else conformal_for_export(conformal, checkpoint)
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
        err = max(float(np.abs(o - r.detach().numpy()).max()
                        / max(1.0, float(np.abs(r.detach().numpy()).max()))) if o.size else 0.0
                  for o, r in zip(out, ref))
        if err > check_atol:
            raise RuntimeError(f"граф {name}: расхождение ONNX и PyTorch {err:.2e} в единицах "
                               f"масштаба выхода, допуск {check_atol}")
        checks[name] = err
        graphs[name] = dict(file=f"{name}.onnx", inputs=used, outputs=list(names_out),
                            shapes_in={n: list(t.shape) for n, t in zip(names_in, inputs[name])},
                            shapes_out={n: list(r.shape) for n, r in zip(names_out, ref)})

    model.train(was_training)
    lead_bins, history_bins = calibration_bins()
    cal = dict(conformal=None, aci=None, lead_bins=lead_bins, history_bins=history_bins)
    if table is not None:
        conformal_table(table, [lo for lo, _hi in history_bins], cfg.horizon).astype(
            "<f4").tofile(os.path.join(out_dir, "conformal.f32"))
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
        qc=dict(DEFAULT_QC.to_dict(), lookback_hours=DEFAULT_QC.lookback_hours),
        state=dict(version=STATE_VERSION, header_bytes=STATE_HEADER.itemsize,
                   nbytes=state_nbytes(cfg), resync_hours=RESYNC_HOURS),
        graphs=graphs, calibration=cal, runtime=runtime.to_dict(),
        export_check_max_rel=checks, opset=opset,
        provenance=provenance())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest


class TorchBackend:
    def __init__(self, model):
        self.mods = {n: GRAPH_MODULES[n](model) for n in GRAPH_NAMES}

    @torch.no_grad()
    def run(self, name, *args):
        out = self.mods[name](*(torch.from_numpy(np.ascontiguousarray(a)) for a in args))
        out = out if isinstance(out, tuple) else (out,)
        return [o.detach().numpy() for o in out]


class OnnxBackend:
    def __init__(self, model_dir, threads=1):
        import onnxruntime as ort
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess, self.names = {}, {}
        for n in GRAPH_NAMES:
            g = self.manifest["graphs"][n]
            self.sess[n] = ort.InferenceSession(os.path.join(model_dir, g["file"]), so,
                                                providers=["CPUExecutionProvider"])
            self.names[n] = set(g["inputs"])

    def run(self, name, *args):
        feed = {k: np.ascontiguousarray(a, np.float32) for k, a in zip(GRAPH_IO[name][0], args)
                if k in self.names[name]}
        return self.sess[name].run(None, feed)


class GraphRuntime(StreamingMayak):
    """Хост потока поверх исполнителя графов.

    Логика хоста та же, что у потокового рантайма на PyTorch: отличается только
    исполнитель графов. По умолчанию без калибровки интервалов.

    Args:
        backend: исполнитель графов с методом ``run(name, *arrays)``.
        cfg: конфиг модели.
        lat: широта точки.
        lon: долгота точки.
        elev: высота точки, м.
        conformal: таблица поправок или None.
        aci: параметры адаптивной калибровки или None.
        runtime_cfg: параметры хоста с порогами смены точки или None - конфиг рантайма
            по умолчанию.
    """

    def __init__(self, backend, cfg, lat, lon, elev, conformal=None, aci=None,
                 runtime_cfg=None):
        self.m = None
        self._setup(backend, cfg, lat, lon, elev, conformal, aci, runtime_cfg)


def manifest_conformal(manifest, model_dir):
    """Конформная таблица экспорта по бинам лидов и длины истории.

    В экспорте таблица лежит развёрнутой по лидам для каждого бина длины истории; здесь
    она сворачивается обратно в бины, чтобы поправку применяла та же функция, что и
    везде в проекте.

    Args:
        manifest: манифест экспорта.
        model_dir: каталог экспорта.

    Returns:
        Таблица float32 формы (число бинов лидов, число бинов длины истории, число
        квантилей) или None, если в экспорте таблицы нет.

    Raises:
        ValueError: бины манифеста не совпадают с бинами кода; таблица не того размера,
            сдвигает медиану или меняется внутри бина лидов.
    """
    from mayak.metrics import I_MED, LEAD_BINS, conformal_table
    cal = manifest["calibration"]
    check_manifest_bins(manifest)
    if not cal.get("conformal"):
        return None
    d = manifest["dims"]
    los = [lo for lo, _hi in cal["history_bins"]]
    n = len(los) * d["horizon"] * d["n_quantiles"]
    table = np.fromfile(os.path.join(model_dir, cal["conformal"]), "<f4")
    if table.size != n:
        raise ValueError(f"{cal['conformal']}: {4 * table.size} Б, ожидалось {4 * n} "
                         f"(бины длины истории × лиды × квантили)")
    table = table.reshape(len(los), d["horizon"], d["n_quantiles"])
    if np.any(table[..., I_MED] != 0.0):
        raise ValueError(f"{cal['conformal']}: поправка медианы не нулевая - таблица сдвигает "
                         f"точечный прогноз; подгоните таблицу заново")
    shift = np.ascontiguousarray(np.moveaxis(table[:, [lo - 1 for lo, _ in LEAD_BINS]], 0, 1))
    if not np.array_equal(conformal_table(shift, los, d["horizon"]), table):
        raise ValueError(f"{cal['conformal']}: поправка меняется внутри бина лидов - таблица "
                         f"записана не этим экспортом")
    return shift


def check_manifest_bins(manifest):
    """Проверяет, что бины калибровки в манифесте совпадают с бинами кода.

    Args:
        manifest: манифест экспорта.

    Raises:
        ValueError: бины лидов или длины истории другие.
    """
    lead_bins, history_bins = calibration_bins()
    cal = manifest["calibration"]
    if cal.get("lead_bins") != lead_bins or cal.get("history_bins") != history_bins:
        raise ValueError(f"манифест: бины калибровки лидов {cal.get('lead_bins')} и длины "
                         f"истории {cal.get('history_bins')}, в коде {lead_bins} и "
                         f"{history_bins}; экспортируйте графы заново")


def runtime_from_export(model_dir, lat, lon, elev, threads=1, conformal=True, aci=False):
    """Хост потока на графах экспорта - те же входы, что у рантайма устройства.

    Args:
        model_dir: каталог экспорта с графами и манифестом.
        lat: широта точки.
        lon: долгота точки.
        elev: высота точки, м.
        threads: число потоков исполнителя графов.
        conformal: применять конформную таблицу экспорта.
        aci: подстраивать множитель калибровки с параметрами из манифеста.

    Пороги смены точки берутся из манифеста, как у рантайма устройства.

    Returns:
        Хост потока на графах.

    Raises:
        ValueError: манифест другого формата или калибровка включена, а её параметров в
            манифесте нет.
        RuntimeError: граф старта не дал годной таблицы климатологии.
    """
    from mayak.config import ModelConfig, RuntimeConfig
    from mayak.metrics import ACIParams
    backend = OnnxBackend(model_dir, threads)
    man = backend.manifest
    if man.get("format") != GRAPH_FORMAT:
        raise ValueError(f"формат манифеста {man.get('format')}, рантайм читает {GRAPH_FORMAT}; "
                         f"экспортируйте графы заново: python scripts/export_runtime.py")
    check_manifest_bins(man)
    table = manifest_conformal(man, model_dir) if conformal else None
    params = None
    if aci:
        a = man["calibration"].get("aci")
        if a is None:
            raise ValueError("ACI включена, но параметров ACI в манифесте нет")
        params = ACIParams(target=a["target"], gamma=a["gamma"], max_factor=a["max_factor"])
        if list(params.interval) != list(a["interval"]):
            raise ValueError(f"манифест: интервал ACI {a['interval']} не соответствует цели "
                             f"{a['target']}")
    return GraphRuntime(backend, ModelConfig.from_dict(man["model_config"]), lat, lon, elev,
                        conformal=table, aci=params,
                        runtime_cfg=RuntimeConfig.from_dict(man["runtime"]))


__all__ = ["DAY_ROW", "GRAPH_IO", "GRAPH_NAMES", "HOURS_OF_YEAR", "GraphRuntime", "OnnxBackend",
           "TorchBackend", "calibration_bins", "check_manifest_bins", "conformal_for_export",
           "dims", "example_inputs", "export_graphs", "manifest_conformal",
           "runtime_from_export", "state_nbytes", "year_calendar"]
