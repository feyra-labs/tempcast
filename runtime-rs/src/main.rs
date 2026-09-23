//! mayak-rt - потоковый рантайм МАЯК для устройства.
//!
//!   mayak-rt run   --model DIR --lat 52.37 --lon 4.9 [--elev -2] [--state-dir runtime]
//!                  [--aci] [--no-conformal] [--int8] [--threads 1]
//!                  [--clim-fallback 10] [--sigma-fallback 4]
//!   mayak-rt bench --model DIR --lat .. --lon .. --series FILE --start-unix-hour H
//!                  [--forecast-every 24] [--warmup 48] [--int8] [--out bench.json]
//!                  [--dump-q q.f32]
//!   mayak-rt info  --model DIR
//!
//! `run` читает stdin построчно:
//!   obs <unix_seconds> <T> <P> <RH>   значения: число, "-" (нет данных) или "nan";
//!   forecast                          прогноз после последнего часа → строка JSON;
//!   status                            состояние рантайма → строка JSON.
//! Час наблюдения должен лежать на целом часе UTC. Пропущенные часы заполняются пустыми
//! шагами (пропуск датчика - пустой шаг, а не пропуск шага), в том числе простой между
//! перезапусками: момент последнего шага восстанавливается по часу года из состояния.
//! После каждого obs
//! состояние атомарно пишется в --state-dir (два чередующихся файла).
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mayak_rt::calendar::{doy_hour, future_calendar, hour_of_year, last_hour_with_hoy};
use mayak_rt::memory::{peak_rss_bytes, rss_bytes};
use mayak_rt::state::Snapshot;
use mayak_rt::store::{newer_first, StateStore};
use mayak_rt::{Error, Precision, Result, Runtime, RuntimeOptions};
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
            clim_fallback: self.num("clim-fallback", Some(10.0))?,
            sigma_fallback: self.num("sigma-fallback", Some(4.0))?,
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

fn parse_obs(s: &str) -> Result<Option<f64>> {
    match s {
        "-" | "" => Ok(None),
        "nan" | "NaN" => Ok(Some(f64::NAN)),
        v => v
            .parse()
            .map(Some)
            .map_err(|_| Error::new(format!("значение наблюдения не число: {v}"))),
    }
}

fn forecast_json(rt: &mut Runtime, last_hour: i64) -> serde_json::Value {
    let (h, nq) = (rt.horizon(), rt.n_quantiles());
    let (mut doy, mut hour) = (vec![0.0; h], vec![0.0; h]);
    future_calendar(last_hour, h, &mut doy, &mut hour);
    let theta = rt.theta();
    let f = rt.safe_forecast(&doy, &hour);
    let q: Vec<&[f32]> = f.q.chunks(nq).collect();
    json!({"after_unix_hour": last_hour, "fallback": f.fallback, "theta": theta, "mu": f.mu, "q": q})
}

fn status_json(rt: &Runtime, last_hour: Option<i64>) -> serde_json::Value {
    json!({"filled": rt.filled(), "hours_in_day": rt.hours_in_day(), "theta": rt.theta(),
           "aci_updates": rt.aci_updates(), "aci_misses": rt.aci_misses(),
           "calendar_breaks": rt.calendar_breaks(), "fallbacks": rt.fallbacks,
           "state_bytes": rt.state_nbytes(), "last_unix_hour": last_hour,
           "rss_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})
}

/// Файлы состояния, свежий первым: сначала по содержимому (Snapshot::parse), затем по
/// времени изменения для тех, что не разбираются.
fn ordered_states(store: &StateStore, dims: mayak_rt::manifest::Dims) -> Vec<(PathBuf, Vec<u8>)> {
    let mut v: Vec<(PathBuf, Vec<u8>)> = store
        .candidates()
        .into_iter()
        .filter_map(|f| std::fs::read(&f).ok().map(|b| (f, b)))
        .collect();
    v.sort_by(
        |a, b| match (Snapshot::parse(&a.1, &dims), Snapshot::parse(&b.1, &dims)) {
            (Ok(x), Ok(y)) => newer_first(&x, &y),
            (Ok(_), Err(_)) => std::cmp::Ordering::Less,
            (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        },
    );
    v
}

fn cmd_run(a: &Args) -> Result<()> {
    let mut rt = a.runtime()?;
    let mut store = StateStore::new(a.get("state-dir").unwrap_or("runtime"))?;
    let mut buf = Vec::new();
    let mut restored = false;
    for (f, raw) in ordered_states(&store, rt.manifest.dims.clone()) {
        match rt.load_state(&raw) {
            Ok(()) => {
                eprintln!("mayak-rt: состояние восстановлено из {} ({} Б)", f.display(), raw.len());
                restored = true;
                break;
            }
            Err(e) => eprintln!("mayak-rt: состояние {} не принято: {e}", f.display()),
        }
    }
    if !restored {
        eprintln!("mayak-rt: чистый старт (история пуста)");
    }
    // Час последнего наблюдения: после перезапуска неизвестен до первой строки obs.
    let mut last: Option<i64> = None;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let reply = match parts.as_slice() {
            ["obs", ts, t, p, rh] => (|| -> Result<serde_json::Value> {
                let sec: i64 = ts.parse().map_err(|_| Error::new(format!("время не число: {ts}")))?;
                if sec.rem_euclid(3600) != 0 {
                    return Err(Error::new(format!("момент {sec} не на целом часе UTC")));
                }
                let hour = sec.div_euclid(3600);
                if last.is_none() && rt.filled() > 0 {
                    last = store
                        .load_last_hour()
                        .filter(|l| *l < hour && Some(hour_of_year(doy_hour(*l).0)) == rt.last_hoy());
                }
                if last.is_none() && rt.filled() > 0 {
                    last = last_hour_with_hoy(rt.last_hoy().unwrap_or(0), hour - 1)
                        .filter(|l| hour - l < rt.stream_window() as i64);
                    if last.is_none() {
                        eprintln!(
                            "mayak-rt: простой длиннее окна ({} ч) - чистый старт",
                            rt.stream_window()
                        );
                        rt.reset();
                    }
                }
                if let Some(l) = last {
                    if hour <= l {
                        return Err(Error::new(format!("час {hour} не позже последнего {l}")));
                    }
                    for gap in (l + 1)..hour {
                        let (d, h) = doy_hour(gap);
                        rt.step([None, None, None], d, h)?;
                    }
                }
                let (d, h) = doy_hour(hour);
                rt.step([parse_obs(t)?, parse_obs(p)?, parse_obs(rh)?], d, h)?;
                last = Some(hour);
                rt.serialize(&mut buf);
                store.save(&buf)?;
                store.save_last_hour(hour)?;
                Ok(json!({"ok": true}))
            })(),
            ["forecast"] => match last {
                Some(l) => Ok(forecast_json(&mut rt, l)),
                None => Err(Error::new(
                    "нет ни одного наблюдения в этом запуске: календарь горизонта не определён",
                )),
            },
            ["status"] => Ok(status_json(&rt, last)),
            [] => continue,
            _ => Err(Error::new(format!("неизвестная команда: {line}"))),
        };
        let v = reply.unwrap_or_else(|e| json!({"error": e.to_string()}));
        writeln!(out, "{v}")?;
        out.flush()?;
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
    let (mut doy, mut hour) = (vec![0.0; h], vec![0.0; h]);
    let (mut t_step, mut t_fc) = (Vec::with_capacity(n), Vec::new());
    let mut dump: Vec<f32> = Vec::new();
    let obs = |v: f32| if v.is_nan() { None } else { Some(v as f64) };
    for k in 0..n {
        let (d, hr) = doy_hour(start + k as i64);
        let row = &series[k * 3..k * 3 + 3];
        let t = Instant::now();
        rt.step([obs(row[0]), obs(row[1]), obs(row[2])], d, hr)?;
        let dt = t.elapsed().as_secs_f64() * 1e6;
        if k >= warmup {
            t_step.push(dt);
        }
        if every > 0 && (k + 1) % every == 0 {
            future_calendar(start + k as i64, h, &mut doy, &mut hour);
            let t = Instant::now();
            let f = rt.forecast(&doy, &hour)?;
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
        "n_modes": m.dims.n_modes, "stream_window": m.dims.stream_window, "enc_buf_len": m.dims.enc_buf_len},
        "state_bytes": m.state.nbytes, "conformal": m.calibration.conformal.is_some(),
        "aci": m.calibration.aci.is_some(),
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
