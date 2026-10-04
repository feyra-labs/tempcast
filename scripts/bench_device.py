r"""Замеры рантайма на устройстве - один воспроизводимый прогон.

    python scripts/bench_device.py --ckpt runs/mayak/stageB/best.ckpt \
        --lat 52.37 --lon 4.90 --elev -2 --out-dir runs/bench_device

Что меряется (всё на одном устройстве, на одних и тех же входах, 1 поток):
* задержка потактового шага и выпуска прогноза: медиана, p95, p99 на длинном прогоне -
  для эталонного потокового рантайма на PyTorch и для Python с ONNX Runtime на графах
  экспорта;
* пиковая резидентная память процесса (каждая реализация - в отдельном процессе);
* размер графов экспорта, размер состояния и буфера энкодера - точным числом байт;
* расхождение выходов ONNX Runtime с эталоном на тех же входах;
* расхождение пакетного и потокового путей при выпусках в случайные часы длинного ряда
  и его изменение во времени.

Итог: out-dir/results.json и out-dir/results.md. Без --ckpt замер идёт на модели со
случайными весами: задержки, память и размеры честные, и отчёт это помечает.
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
UNTRAINED_PERTURB = 0.05


def peak_rss_bytes():
    """Пиковый RSS процесса.

    None там, где нет resource (Windows): замер идёт на устройстве, а Windows не целевая платформа,
    но --help должен работать везде.
    """
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def percentile(v, p):
    """Процентиль методом ближайшего ранга.

    Args:
        v: значения.
        p: процентиль от 0 до 100.

    Returns:
        Значение процентиля; NaN для пустого набора.
    """
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


def load_model(ckpt, seed):
    """Модель замера: из чекпойнта или со случайными весами.

    Случайные веса - инициализация с сидом и шум на всех параметрах, чтобы нулевые
    инициализации голов не делали выход тривиальным.

    Args:
        ckpt: путь к чекпойнту или None.
        seed: сид случайных весов.

    Returns:
        Модель в режиме вывода.
    """
    import torch
    if ckpt:
        from mayak.lit import load_model as _load
        return _load(ckpt).eval()
    from mayak.model import MAYAK
    torch.manual_seed(seed)
    m = MAYAK().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(UNTRAINED_PERTURB * torch.randn_like(p))
    return m


def start_hour():
    from mayak.timeaxis import to_utc_hour
    return int(to_utc_hour(np.datetime64(START_UTC, "s")))


# --------------------------------------------------------------------------- Python-воркер

def worker(args):
    """Отдельный процесс: одна реализация на Python, замер задержек и пиковой памяти.

    Args:
        args: аргументы командной строки замера.
    """
    import torch
    torch.set_num_threads(1)
    model = load_model(args.ckpt, args.seed)
    series = np.fromfile(args.series, "<f4").reshape(-1, 3)
    t0 = time.perf_counter()
    if args.backend == "torch":
        from mayak.runtime.streaming import StreamingMayak
        rt = StreamingMayak(model, args.lat, args.lon, args.elev)
    else:
        from mayak.runtime.graphs import GraphRuntime, OnnxBackend
        rt = GraphRuntime(OnnxBackend(args.model_dir), model.cfg, args.lat, args.lon, args.elev)
    startup_ms = (time.perf_counter() - t0) * 1e3
    start = args.start_unix_hour
    t_step, t_fc, dump = [], [], []
    for k in range(series.shape[0]):
        obs = [None if np.isnan(v) else float(v) for v in series[k]]
        t = time.perf_counter()
        rt.step(*obs, start + k)
        dt = (time.perf_counter() - t) * 1e6
        if k >= args.warmup:
            t_step.append(dt)
        if args.forecast_every and (k + 1) % args.forecast_every == 0:
            t = time.perf_counter()
            q = rt.forecast()[0]
            dt = (time.perf_counter() - t) * 1e6
            if k >= args.warmup:
                t_fc.append(dt)
            dump.append(np.asarray(q, np.float32))
    np.concatenate([q.ravel() for q in dump]).astype("<f4").tofile(args.dump_q)
    rep = dict(runtime=f"python-{args.backend}", hours=int(series.shape[0]),
               step=stats_us(t_step), forecast=stats_us(t_fc), startup_ms=startup_ms,
               peak_rss_bytes=peak_rss_bytes())
    rep["state_bytes"] = len(rt.serialize())
    t = time.perf_counter()
    rt.load_state(rt.serialize())
    rep["restore_ms"] = (time.perf_counter() - t) * 1e3
    print(json.dumps(rep))


def run_worker(args, backend, dump):
    cmd = [sys.executable, os.path.abspath(__file__), "_worker", "--backend", backend,
           "--series", args.series, "--dump-q", dump,
           "--start-unix-hour", str(args.start_unix_hour), "--lat", str(args.lat),
           "--lon", str(args.lon), "--elev", str(args.elev), "--warmup", str(args.warmup),
           "--forecast-every", str(args.forecast_every), "--model-dir", args.model_dir,
           "--seed", str(args.seed)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    return json.loads(out.stdout.strip().splitlines()[-1])


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
    L = ["# Замеры рантайма МАЯК на устройстве", "",
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
    for k in ("model_bytes", "state_bytes", "encoder_buffer_bytes"):
        L.append(f"| {k} | {s.get(k)} |")
    eq = res["batch_stream"]
    L += ["", f"Пакет и поток: {eq['n_issues']} выпусков в случайные часы ряда длиной "
              f"{eq['hours']} ч, max|Δq| = {eq['batch_stream_max_abs']:.2e} °C по всем лидам; "
              f"после перезапуска {eq['restart_max_abs']:.2e} °C."]
    L += ["", "| Выпуск, час ряда | max\\|Δq\\|, °C |", "|---|---|"]
    L += [f"| {end} | {err:.2e} |" for end, err in eq["batch_stream_by_issue"]]
    if not res["trained"]:
        L += ["", "**Модель не обучена (случайные веса): задержки, память и размеры честные.**"]
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        ap = argparse.ArgumentParser()
        for k in ("--backend", "--series", "--dump-q", "--model-dir", "--ckpt"):
            ap.add_argument(k, default=None)
        for k in ("--start-unix-hour", "--warmup", "--forecast-every", "--seed"):
            ap.add_argument(k, type=int)
        for k in ("--lat", "--lon", "--elev"):
            ap.add_argument(k, type=float)
        return worker(ap.parse_args(sys.argv[2:]))

    ap = argparse.ArgumentParser(description="замеры рантайма на устройстве")
    ap.add_argument("--ckpt", default=None, help="чекпойнт; без него - случайные веса")
    ap.add_argument("--conformal", default=None)
    ap.add_argument("--out-dir", default="runs/bench_device")
    ap.add_argument("--hours", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=48)
    ap.add_argument("--forecast-every", type=int, default=24)
    ap.add_argument("--issues", type=int, default=40,
                    help="выпусков в случайные часы для сравнения пакета и потока")
    ap.add_argument("--equivalence-hours", type=int, default=10_000,
                    help="длина ряда для сравнения пакета и потока, часы")
    ap.add_argument("--lat", type=float, default=52.37)
    ap.add_argument("--lon", type=float, default=4.9)
    ap.add_argument("--elev", type=float, required=True,
                    help="высота точки над уровнем моря, м")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from mayak.runtime.equivalence import divergence, synthetic_series
    from mayak.runtime.graphs import export_graphs
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.ckpt, args.seed)
    args.model_dir = os.path.join(args.out_dir, "model")
    man = export_graphs(model, args.model_dir, conformal=args.conformal)

    s = synthetic_series(args.hours, seed=args.seed)
    x = np.where(s["m"] > 0, s["x"], np.nan).astype("<f4")
    args.series = os.path.join(args.out_dir, "series.f32")
    x.tofile(args.series)
    args.start_unix_hour = start_hour()

    runs, dumps = {}, {}
    for name, backend in (("Python (PyTorch, эталон)", "torch"),
                          ("Python + ONNX Runtime", "onnx")):
        dumps[name] = os.path.join(args.out_dir, f"q_{len(dumps)}.f32")
        print("…", name, flush=True)
        runs[name] = run_worker(args, backend, dumps[name])
    ref = np.fromfile(dumps["Python (PyTorch, эталон)"], "<f4")
    for name, path in dumps.items():
        dq = float(np.abs(np.fromfile(path, '<f4') - ref).max())
        runs[name]["max_abs_dq_vs_python"] = f"{dq:.2e}"

    size = lambda p: os.path.getsize(os.path.join(args.model_dir, p))
    d = man["dims"]
    sizes = dict(model_bytes=sum(size(v["file"]) for v in man["graphs"].values()),
                 state_bytes=runs["Python + ONNX Runtime"]["state_bytes"],
                 encoder_buffer_bytes=4 * d["encoder_width"] * d["enc_buf_len"])
    print("… пакет ↔ поток", flush=True)
    res = dict(device=device_info(), model=args.ckpt or "случайные веса",
               trained=bool(args.ckpt), hours=args.hours, forecast_every=args.forecast_every,
               runs=runs, sizes=sizes,
               batch_stream=divergence(model, args.issues, hours=args.equivalence_hours,
                                       seed=args.seed, lat=args.lat, lon=args.lon,
                                       elev=args.elev))
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    md = markdown(res)
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
