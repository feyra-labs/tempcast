//! mayak-rt - потоковый рантайм МАЯК для устройства.
//!
//!   mayak-rt run   --model DIR --lat 52.37 --lon 4.9 [--elev -2] [--state-dir runtime]
//!                  [--aci] [--no-conformal] [--int8] [--threads 1]
//!   mayak-rt bench --model DIR --lat .. --lon .. --series FILE --start-unix-hour H
//!                  [--forecast-every 24] [--warmup 48] [--int8] [--out bench.json]
//!                  [--dump-q q.f32]
//!   mayak-rt info  --model DIR
//!
//! `run` читает stdin построчно:
//!   obs <unix_seconds> <T> <P> <RH>   значения: число, "-" (нет данных) или "nan";
//!   forecast [<unix_seconds>]         прогноз после последнего часа, строка JSON;
//!   status                            состояние рантайма, строка JSON.
//! Час наблюдения должен лежать на целом часе UTC и быть позже последнего шага.
//! Пропущенные часы заполняются пустыми шагами, в том числе простой между перезапусками:
//! абсолютный час последнего шага хранится в заголовке состояния. Простой не короче окна
//! означает холодный старт. После каждого obs состояние атомарно пишется в --state-dir,
//! в два чередующихся файла; при старте свежий файл выбирается по содержимому.
//! До первого наблюдения прогноз выпускается после текущего часа по часам устройства;
//! секунды в команде forecast заменяют часы устройства. Если состояние записано для
//! другой точки, сдвиг в пределах порогов манифеста считается уточнением координат, а
//! больше порога - переносом прибора с холодным стартом; это видно в логе и в status. Любой сбой выпуска заменяется
//! климатологией точки. Если модель не поднялась, в том числе не удался расчёт
//! климатологии точки при старте, процесс завершается с кодом 1.
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use mayak_rt::memory::{peak_rss_bytes, rss_bytes};
use mayak_rt::store::StateStore;
use mayak_rt::{Error, Host, Precision, Result, Runtime, RuntimeOptions};
use serde_json::json;

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_default();
        let (mut kv, mut flags) = (HashMap::new(), Vec::new());
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let a = rest[i]
                .strip_prefix("--")
                .ok_or_else(|| Error::new(format!("ожидался ключ --…, получено {}", rest[i])))?;
            if ["aci", "no-conformal", "int8"].contains(&a) {
                flags.push(a.to_string());
                i += 1;
            } else {
                let v = rest
                    .get(i + 1)
                    .ok_or_else(|| Error::new(format!("--{a}: нет значения")))?;
                kv.insert(a.to_string(), v.clone());
                i += 2;
            }
        }
        Ok(Args { cmd, kv, flags })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn req(&self, k: &str) -> Result<&str> {
        self.get(k).ok_or_else(|| Error::new(format!("нужен --{k}")))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: Option<T>) -> Result<T> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| Error::new(format!("--{k}: не число: {v}"))),
            None => default.ok_or_else(|| Error::new(format!("нужен --{k}"))),
        }
    }
    fn flag(&self, f: &str) -> bool {
        self.flags.iter().any(|x| x == f)
    }
    fn options(&self) -> Result<RuntimeOptions> {
        Ok(RuntimeOptions {
            precision: if self.flag("int8") {
                Precision::Int8
            } else {
                Precision::Fp32
            },
            threads: self.num("threads", Some(1))?,
            conformal: !self.flag("no-conformal"),
            aci: self.flag("aci"),
        })
    }
    fn runtime(&self) -> Result<Runtime> {
        Runtime::new(
            self.req("model")?,
            self.num("lat", None)?,
            self.num("lon", None)?,
            self.num("elev", Some(0.0))?,
            &self.options()?,
        )
    }
}

fn cmd_run(a: &Args) -> Result<()> {
    let rt = a.runtime()?;
    let store = StateStore::new(a.get("state-dir").unwrap_or("runtime"))?;
    let mut host = Host::new(rt, Some(store));
    host.restore();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        if let Some(v) = host.handle(&line?) {
            writeln!(out, "{v}")?;
            out.flush()?;
        }
    }
    Ok(())
}

/// Процентиль по ближайшему рангу (как numpy method="inverted_cdf").
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let k = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[k - 1]
}

fn stats_us(mut v: Vec<f64>) -> serde_json::Value {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({"n": v.len(), "p50_us": percentile(&v, 0.5), "p95_us": percentile(&v, 0.95),
           "p99_us": percentile(&v, 0.99), "max_us": v.last().copied(), "mean_us": mean})
}

fn file_size(p: &Path) -> Option<u64> {
    std::fs::metadata(p).ok().map(|m| m.len())
}

fn cmd_bench(a: &Args) -> Result<()> {
    let rss0 = rss_bytes();
    let t0 = Instant::now();
    let mut rt = a.runtime()?;
    let startup_ms = t0.elapsed().as_secs_f64() * 1e3;
    let raw = std::fs::read(a.req("series")?)?;
    let series: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let n = series.len() / 3;
    let start: i64 = a.num("start-unix-hour", None)?;
    let every: usize = a.num("forecast-every", Some(24))?;
    let warmup: usize = a.num("warmup", Some(48))?;
    let (h, nq) = (rt.horizon(), rt.n_quantiles());
    let (mut t_step, mut t_fc) = (Vec::with_capacity(n), Vec::new());
    let mut dump: Vec<f32> = Vec::new();
    let obs = |v: f32| if v.is_nan() { None } else { Some(v as f64) };
    for k in 0..n {
        let row = &series[k * 3..k * 3 + 3];
        let t = Instant::now();
        rt.step([obs(row[0]), obs(row[1]), obs(row[2])], start + k as i64)?;
        let dt = t.elapsed().as_secs_f64() * 1e6;
        if k >= warmup {
            t_step.push(dt);
        }
        if every > 0 && (k + 1) % every == 0 {
            let t = Instant::now();
            let f = rt.forecast(None)?;
            let dt = t.elapsed().as_secs_f64() * 1e6;
            if k >= warmup {
                t_fc.push(dt);
            }
            dump.extend_from_slice(&f.q);
        }
    }
    let mut state = Vec::new();
    let t = Instant::now();
    rt.serialize(&mut state);
    let ser_us = t.elapsed().as_secs_f64() * 1e6;
    let t = Instant::now();
    rt.load_state(&state)?;
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;
    if let Some(p) = a.get("dump-q") {
        let bytes: Vec<u8> = dump.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
    }
    let m = &rt.manifest;
    let prec = if a.flag("int8") { "int8" } else { "fp32" };
    let model_bytes: u64 = m
        .graphs
        .values()
        .filter_map(|g| {
            file_size(&m.dir.join(if prec == "int8" {
                g.int8.as_deref().unwrap_or(&g.fp32)
            } else {
                &g.fp32
            }))
        })
        .sum();
    let rep = json!({
        "runtime": "rust", "precision": prec, "threads": a.num::<usize>("threads", Some(1))?,
        "hours": n, "warmup": warmup, "forecasts": dump.len() / (h * nq),
        "step": stats_us(t_step), "forecast": stats_us(t_fc),
        "startup_ms": startup_ms, "serialize_us": ser_us, "restore_ms": restore_ms,
        "state_bytes": state.len(), "encoder_buffer_bytes": rt.encoder_buffer_bytes(),
        "memory_bytes": rt.memory_bytes(),
        "rss_before_bytes": rss0, "peak_rss_bytes": peak_rss_bytes(),
        "binary_bytes": std::env::current_exe().ok().and_then(|p| file_size(&p)),
        "model_bytes": model_bytes, "fallbacks": rt.fallbacks,
    });
    match a.get("out") {
        Some(p) => std::fs::write(p, serde_json::to_string_pretty(&rep)?)?,
        None => println!("{}", serde_json::to_string_pretty(&rep)?),
    }
    Ok(())
}

fn cmd_info(a: &Args) -> Result<()> {
    let m = mayak_rt::Manifest::load(a.req("model")?)?;
    println!(
        "{}",
        json!({"dims": {"horizon": m.dims.horizon, "n_quantiles": m.dims.n_quantiles,
        "n_modes": m.dims.n_modes, "stream_window": m.dims.stream_window,
        "stream_edge": m.dims.stream_edge, "stream_tail": m.dims.stream_tail,
        "enc_buf_len": m.dims.enc_buf_len},
        "state_bytes": m.state.nbytes, "conformal": m.calibration.conformal.is_some(),
        "aci": m.calibration.aci.is_some(),
        "site_max_dlat_deg": m.runtime.site_max_dlat_deg,
        "site_max_dlon_deg": m.runtime.site_max_dlon_deg,
        "site_max_delev_m": m.runtime.site_max_delev_m,
        "int8": m.graphs.values().all(|g| g.int8.is_some())})
    );
    Ok(())
}

fn main() {
    let res = Args::parse().and_then(|a| match a.cmd.as_str() {
        "run" => cmd_run(&a),
        "bench" => cmd_bench(&a),
        "info" => cmd_info(&a),
        _ => Err(Error::new("команда: run | bench | info (см. заголовок src/main.rs)")),
    });
    if let Err(e) = res {
        eprintln!("mayak-rt: {e}");
        std::process::exit(1);
    }
}
