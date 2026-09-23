"""Замеры рантайма на устройстве - один воспроизводимый прогон.

    cargo build --release --manifest-path runtime-rs/Cargo.toml
    python scripts/bench_device.py --ckpt runs/mayak/stageB/best.ckpt \\
        --manifest data/manifest.csv --out-dir runs/bench_device

Что меряется (всё на одном устройстве, на одних и тех же входах, 1 поток):
* задержка потактового шага и выпуска прогноза: медиана, p95, p99 на длинном прогоне -
  для Rust fp32, Rust int8, эталонного Python (PyTorch, StreamingMayak) и Python +
  ONNX Runtime (те же графы, что у Rust; отделяет выигрыш от языка хоста и от движка);
* пиковая резидентная память процесса (каждая реализация - в отдельном процессе);
* размер бинарника, размер модели fp32 и int8, размер состояния - точным числом байт;
* расхождение выходов Rust fp32 / int8 с эталоном на тех же входах;
* расхождение пакетного и потокового путей на --windows окнах (mayak.runtime.equivalence);
* метрики fp32 против int8 на тестовой выборке (если есть --manifest с кэшем данных).

Итог: out-dir/results.json и out-dir/results.md. Без --ckpt замер идёт на модели эталона
(случайные веса): задержки и память честные, метрики - нет, и отчёт это помечает.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

START_UTC = "2025-01-01T00"
PCTL = (0.50, 0.95, 0.99)


def peak_rss_bytes():
    """Пиковый RSS процесса. None там, где нет resource (Windows): замер идёт на
    устройстве, а Windows не целевая платформа, но --help должен работать везде."""
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def percentile(v, p):
    """Ближайший ранг - та же формула, что в mayak-rt bench."""
    s = np.sort(np.asarray(v, np.float64))
    if s.size == 0:
        return float("nan")
    k = min(max(int(np.ceil(p * s.size)), 1), s.size)
    return float(s[k - 1])


def stats_us(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(v.size), p50_us=percentile(v, 0.5), p95_us=percentile(v, 0.95),
                p99_us=percentile(v, 0.99), max_us=float(v.max()) if v.size else None,
                mean_us=float(v.mean()) if v.size else None)


def load_model(ckpt):
    if ckpt:
        from mayak.lit import load_model as _load
        return _load(ckpt).eval()
    from mayak.runtime.golden import golden_model
    return golden_model()


def start_hour():
    from mayak.timeaxis import to_utc_hour
    return int(to_utc_hour(np.datetime64(START_UTC, "s")))


# --------------------------------------------------------------------------- Python-воркер

def worker(args):
    """Отдельный процесс: одна реализация на Python, замер задержек и пиковой памяти."""
    import torch
    torch.set_num_threads(1)
    from mayak.timeaxis import from_utc_hour, doy_hour
    model = load_model(args.ckpt)
    series = np.fromfile(args.series, "<f4").reshape(-1, 3)
    H = model.cfg.horizon
    t0 = time.perf_counter()
    if args.backend == "torch":
        from mayak.runtime.streaming import StreamingMayak
        rt = StreamingMayak(model, args.lat, args.lon, args.elev)
        fc = lambda d, h: rt.forecast(d, h)[0]
    else:
        from mayak.runtime.graphs import GraphRuntime, OnnxBackend
        rt = GraphRuntime(OnnxBackend(args.model_dir, args.precision), model.cfg,
                          args.lat, args.lon, args.elev)
        fc = rt.forecast
    startup_ms = (time.perf_counter() - t0) * 1e3
    start = args.start_unix_hour
    t_step, t_fc, dump = [], [], []
    for k in range(series.shape[0]):
        d, h = doy_hour(from_utc_hour(start + k))
        obs = [None if np.isnan(v) else float(v) for v in series[k]]
        t = time.perf_counter()
        rt.step(*obs, np.float32(d), np.float32(h))
        dt = (time.perf_counter() - t) * 1e6
        if k >= args.warmup:
            t_step.append(dt)
        if args.forecast_every and (k + 1) % args.forecast_every == 0:
            fd, fh = doy_hour(from_utc_hour(start + k + 1 + np.arange(H)))
            t = time.perf_counter()
            q = fc(fd.astype(np.float32), fh.astype(np.float32))
            dt = (time.perf_counter() - t) * 1e6
            if k >= args.warmup:
                t_fc.append(dt)
            dump.append(np.asarray(q, np.float32))
    np.concatenate([q.ravel() for q in dump]).astype("<f4").tofile(args.dump_q)
    rep = dict(runtime=f"python-{args.backend}", precision=args.precision,
               hours=int(series.shape[0]), step=stats_us(t_step), forecast=stats_us(t_fc),
               startup_ms=startup_ms,
               peak_rss_bytes=peak_rss_bytes())
    if args.backend == "torch":
        rep["state_bytes"] = len(rt.serialize())
        t = time.perf_counter()
        rt.load_state(rt.serialize())
        rep["restore_ms"] = (time.perf_counter() - t) * 1e3
    print(json.dumps(rep))


def run_worker(args, backend, precision, dump):
    cmd = [sys.executable, os.path.abspath(__file__), "_worker", "--backend", backend,
           "--precision", precision, "--series", args.series, "--dump-q", dump,
           "--start-unix-hour", str(args.start_unix_hour), "--lat", str(args.lat),
           "--lon", str(args.lon), "--elev", str(args.elev), "--warmup", str(args.warmup),
           "--forecast-every", str(args.forecast_every), "--model-dir", args.model_dir]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    return json.loads(out.stdout.strip().splitlines()[-1])


def run_rust(args, precision, dump):
    cmd = [args.rust_bin, "bench", "--model", args.model_dir, "--lat", str(args.lat),
           "--lon", str(args.lon), "--elev", str(args.elev), "--series", args.series,
           "--start-unix-hour", str(args.start_unix_hour), "--warmup", str(args.warmup),
           "--forecast-every", str(args.forecast_every), "--no-conformal", "--dump-q", dump,
           "--threads", "1"]
    if precision == "int8":
        cmd.append("--int8")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


# --------------------------------------------------------------------------- метрики int8

def int8_metrics(args, model):
    """Метрики fp32 и int8 на тестовой выборке через графы ONNX (путь устройства)."""
    import torch
    from mayak.data.store import get_store
    from mayak.evaluate import EvalSet
    from mayak.metrics import Evaluation
    from mayak.runtime.graphs import GraphRuntime, OnnxBackend
    store = get_store(args.manifest)
    ds = EvalSet(store.clims(), manifest=args.manifest, time_key="test",
                 max_windows=args.max_windows)
    backends = {p: OnnxBackend(args.model_dir, p) for p in ("fp32", "int8")}
    qs = {p: [] for p in backends}
    ys, ws, mucl, st = [], [], [], []
    meta = ds.window_meta()
    for i in range(len(ds)):
        b = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        for p, be in backends.items():
            rt = GraphRuntime(be, model.cfg, float(b["lat"]), float(b["lon"]), float(b["elev"]))
            for k in range(b["x_hist"].shape[0]):
                obs = [float(b["x_hist"][k, j]) if b["mask_hist"][k, j] > 0 else None
                       for j in range(3)]
                rt.step(*obs, b["doy_hist"][k], b["hour_hist"][k])
            qs[p].append(rt.forecast(b["doy_fut"], b["hour_fut"]))
        ys.append(b["y"])
        ws.append(b["y_mask"])
        mucl.append(b["mu_clim_fut"])
        st.append(meta["station"][i])
    out = {}
    from mayak.metrics import I_MED
    for p in qs:
        q = np.stack(qs[p])
        ev = Evaluation(y=np.stack(ys), mu=q[..., I_MED], q=q, mu_clim=np.stack(mucl),
                        w=np.stack(ws), station=np.array(st, object))
        out[p] = ev.pooled()
    q32, q8 = np.stack(qs["fp32"]), np.stack(qs["int8"])
    out["delta"] = {k: out["int8"][k] - out["fp32"][k] for k in out["fp32"]}
    out["max_abs_dq"] = float(np.abs(q32 - q8).max())
    out["n_windows"] = len(ds)
    return out


# --------------------------------------------------------------------------- отчёт

def device_info():
    cpu = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("Model", "model name")):
                    cpu = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return dict(machine=platform.machine(), system=platform.platform(), cpu=cpu,
                python=platform.python_version())


def fmt(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{nd}f}"


def markdown(res):
    L = [f"# Замеры рантайма МАЯК на устройстве", "",
         f"Устройство: {res['device']['cpu'] or res['device']['machine']} "
         f"({res['device']['system']}); модель: {res['model']}; "
         f"{res['hours']} ч, выпуск каждые {res['forecast_every']} ч, 1 поток.", "",
         "| Реализация | шаг p50, мкс | шаг p95 | шаг p99 | выпуск p50, мкс | выпуск p99 | "
         "пик RSS, МБ | старт, мс | max\\|Δq\\| к эталону, °C |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["runs"].items():
        L.append(f"| {name} | {fmt(r['step']['p50_us'])} | {fmt(r['step']['p95_us'])} | "
                 f"{fmt(r['step']['p99_us'])} | {fmt(r['forecast']['p50_us'])} | "
                 f"{fmt(r['forecast']['p99_us'])} | {fmt(r['peak_rss_bytes'] / 2 ** 20)} | "
                 f"{fmt(r.get('startup_ms'))} | {r.get('max_abs_dq_vs_python', '—')} |")
    s = res["sizes"]
    L += ["", "| Размер | байт |", "|---|---|"]
    for k in ("binary_bytes", "model_fp32_bytes", "model_int8_bytes", "state_bytes",
              "encoder_buffer_bytes"):
        L.append(f"| {k} | {s.get(k)} |")
    eq = res["batch_stream"]
    L += ["", f"Пакет ↔ поток: {eq['n_windows']} окон, max|Δq| = "
              f"{eq['batch_stream_max_abs']:.2e} °C по всем лидам; после перезапуска "
              f"{eq['restart_max_abs']:.2e} °C."]
    if res.get("int8_metrics"):
        m = res["int8_metrics"]
        L += ["", f"fp32 против int8 на тесте ({m['n_windows']} окон, max|Δq| "
                  f"{m['max_abs_dq']:.3f} °C):", "", "| метрика | fp32 | int8 | Δ |",
              "|---|---|---|---|"]
        for k in ("MAE", "RMSE", "CRPS", "PICP90", "Skill"):
            L.append(f"| {k} | {m['fp32'][k]:.4f} | {m['int8'][k]:.4f} | {m['delta'][k]:+.4f} |")
    if not res["trained"]:
        L += ["", "**Модель не обучена (веса эталона): задержки, память и размеры честные, "
                  "метрики и расхождение int8 - нет.**"]
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        ap = argparse.ArgumentParser()
        for k in ("--backend", "--precision", "--series", "--dump-q", "--model-dir", "--ckpt"):
            ap.add_argument(k, default=None)
        for k in ("--start-unix-hour", "--warmup", "--forecast-every"):
            ap.add_argument(k, type=int)
        for k in ("--lat", "--lon", "--elev"):
            ap.add_argument(k, type=float)
        return worker(ap.parse_args(sys.argv[2:]))

    ap = argparse.ArgumentParser(description="замеры рантайма на устройстве (блок 14)")
    ap.add_argument("--ckpt", default=None, help="чекпойнт; без него - модель эталона")
    ap.add_argument("--conformal", default=None)
    ap.add_argument("--out-dir", default="runs/bench_device")
    ap.add_argument("--rust-bin", default="runtime-rs/target/release/mayak-rt")
    ap.add_argument("--hours", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=48)
    ap.add_argument("--forecast-every", type=int, default=24)
    ap.add_argument("--windows", type=int, default=300, help="окон для пакет ↔ поток")
    ap.add_argument("--manifest", default=None, help="data/manifest.csv для метрик int8")
    ap.add_argument("--max-windows", type=int, default=200)
    ap.add_argument("--lat", type=float, default=52.37)
    ap.add_argument("--lon", type=float, default=4.9)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from mayak.runtime.equivalence import divergence, synthetic_series
    from mayak.runtime.graphs import export_graphs
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.ckpt)
    args.model_dir = os.path.join(args.out_dir, "model")
    man = export_graphs(model, args.model_dir, conformal=args.conformal, int8=True)

    s = synthetic_series(args.hours, seed=args.seed)
    x = np.where(s["m"] > 0, s["x"], np.nan).astype("<f4")
    args.series = os.path.join(args.out_dir, "series.f32")
    x.tofile(args.series)
    args.start_unix_hour = start_hour()

    runs, dumps = {}, {}
    for name, fn in (("Python (PyTorch, эталон)", lambda d: run_worker(args, "torch", "fp32", d)),
                     ("Python + ONNX Runtime fp32", lambda d: run_worker(args, "onnx", "fp32", d)),
                     ("Rust fp32", lambda d: run_rust(args, "fp32", d)),
                     ("Rust int8", lambda d: run_rust(args, "int8", d))):
        dumps[name] = os.path.join(args.out_dir, f"q_{len(dumps)}.f32")
        print("…", name, flush=True)
        runs[name] = fn(dumps[name])
    ref = np.fromfile(dumps["Python (PyTorch, эталон)"], "<f4")
    for name, path in dumps.items():
        runs[name]["max_abs_dq_vs_python"] = f"{float(np.abs(np.fromfile(path, '<f4') - ref).max()):.2e}"

    size = lambda p: os.path.getsize(os.path.join(args.model_dir, p))
    g = man["graphs"]
    rust32 = runs["Rust fp32"]
    sizes = dict(binary_bytes=rust32["binary_bytes"],
                 model_fp32_bytes=sum(size(v["fp32"]) for v in g.values()),
                 model_int8_bytes=sum(size(v["int8"]) for v in g.values()),
                 state_bytes=rust32["state_bytes"],
                 encoder_buffer_bytes=rust32["encoder_buffer_bytes"])
    print("… пакет ↔ поток", flush=True)
    res = dict(device=device_info(), model=args.ckpt or "эталон (случайные веса)",
               trained=bool(args.ckpt), hours=args.hours, forecast_every=args.forecast_every,
               runs=runs, sizes=sizes,
               batch_stream=divergence(model, args.windows, seed=args.seed))
    if args.manifest and os.path.exists(args.manifest):
        print("… метрики fp32 / int8", flush=True)
        res["int8_metrics"] = int8_metrics(args, model)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    md = markdown(res)
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
