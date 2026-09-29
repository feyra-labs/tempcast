//! Тест: хост на Rust проходит эталонные сценарии, записанные хостом на Python.
//!
//! Каждый сценарий - строки протокола с ожидаемыми ответами, перезапуски хоста с тем же
//! каталогом состояния и сверки состояния на диске. Сверяются коды контроля качества,
//! моменты выпуска, признаки отката, множитель калибровки, сводка, выбранный при
//! перезапуске файл и квантили в пределах допуска своей точности.
//!
//! Эталон лежит среди данных тестов репозитория и пересоздаётся генератором эталона
//! хоста только при изменении поведения.
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use mayak_rt::calendar::{doy_hour, hour_of_year};
use mayak_rt::calib::{aci_score, apply_adaptive, apply_conformal, AciParams};
use mayak_rt::state::HEADER;
use mayak_rt::store::StateStore;
use mayak_rt::{Host, Manifest, Precision, Runtime, RuntimeOptions};
use serde_json::Value;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/data/runtime_golden")
}

struct Golden {
    doc: Value,
    blob: Vec<f32>,
}

fn load() -> Golden {
    let dir = golden_dir();
    let doc: Value = serde_json::from_str(&fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    let raw = fs::read(dir.join("golden.f32")).unwrap();
    let blob = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Golden { doc, blob }
}

impl Golden {
    fn take(&self, r: &Value) -> &[f32] {
        let (o, n) = (
            r["offset"].as_u64().unwrap() as usize,
            r["len"].as_u64().unwrap() as usize,
        );
        &self.blob[o..o + n]
    }

    fn scenario(&self, name: &str) -> &Value {
        self.doc["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == name)
            .unwrap_or_else(|| panic!("в эталоне нет сценария {name}"))
    }

    fn tol(&self, key: &str) -> f32 {
        self.doc["tolerance"][key].as_f64().unwrap() as f32
    }
}

fn f(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

fn score(v: &Value) -> f64 {
    match v.as_str() {
        Some("nan") => f64::NAN,
        Some("inf") => f64::INFINITY,
        _ => v.as_f64().unwrap(),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn runtime(sc: &Value, site: [f64; 3]) -> Runtime {
    let opts = RuntimeOptions {
        precision: if sc["precision"] == "int8" {
            Precision::Int8
        } else {
            Precision::Fp32
        },
        conformal: sc["conformal"].as_bool().unwrap(),
        aci: sc["aci"].as_bool().unwrap(),
        ..Default::default()
    };
    Runtime::new(
        golden_dir().join(sc["model"].as_str().unwrap()),
        site[0],
        site[1],
        site[2],
        &opts,
    )
    .unwrap()
}

/// Пустой каталог состояния сценария с его начальными файлами и порядком их времени
/// изменения.
fn state_dir(g: &Golden, sc: &Value) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("golden_state")
        .join(sc["name"].as_str().unwrap());
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let base = g.doc["mtime_base"].as_u64().unwrap();
    for init in sc["init_files"].as_array().unwrap() {
        let dst = dir.join(init["as"].as_str().unwrap());
        fs::copy(golden_dir().join(init["file"].as_str().unwrap()), &dst).unwrap();
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(base + init["mtime"].as_u64().unwrap());
        File::options().write(true).open(&dst).unwrap().set_modified(t).unwrap();
    }
    dir
}

fn host(sc: &Value, site: [f64; 3], dir: &Path) -> Host {
    Host::new(runtime(sc, site), Some(StateStore::new(dir).unwrap()))
        .with_clock(|| panic!("в эталоне часы устройства не используются"))
}

/// Состояние хоста против файла эталона: заголовок до множителя калибровки, координаты
/// и окно до байта, множитель с допуском.
fn assert_state(rt: &Runtime, file: &Path, theta_tol: f32, what: &str) {
    let mut got = Vec::new();
    rt.serialize(&mut got);
    let want = fs::read(file).unwrap();
    assert_eq!(got.len(), want.len(), "{what}: размер состояния");
    assert_eq!(got[..16], want[..16], "{what}: заголовок состояния");
    let th = |b: &[u8]| f32::from_le_bytes([b[16], b[17], b[18], b[19]]);
    assert!((th(&got) - th(&want)).abs() <= theta_tol, "{what}: множитель калибровки");
    assert_eq!(got[20..HEADER], want[20..HEADER], "{what}: координаты");
    assert_eq!(got[HEADER..], want[HEADER..], "{what}: сырое окно и маски");
}

/// Прогон сценария: наибольшее расхождение квантилей с эталоном.
fn replay(g: &Golden, name: &str) -> f32 {
    let sc = g.scenario(name);
    let int8 = sc["precision"] == "int8";
    let tol = g.tol(if int8 { "q_abs_int8" } else { "q_abs" });
    let theta_tol = g.tol("theta_abs");
    let dir = state_dir(g, sc);
    let mut site = [
        sc["lat"].as_f64().unwrap(),
        sc["lon"].as_f64().unwrap(),
        sc["elev"].as_f64().unwrap(),
    ];
    let mut h: Option<Host> = None;
    let mut worst = 0.0f32;
    for (i, ev) in sc["events"].as_array().unwrap().iter().enumerate() {
        let at = format!("{name}: событие {i}");
        match ev["op"].as_str().unwrap() {
            "restart" => {
                if let Some(s) = ev["site"].as_array() {
                    site = [0, 1, 2].map(|k| s[k].as_f64().unwrap());
                }
                drop(h.take());
                let mut next = host(sc, site, &dir);
                let got = next
                    .restore()
                    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned());
                assert_eq!(got.as_deref(), ev["expect"]["restored"].as_str(), "{at}: восстановлен не тот файл");
                h = Some(next);
            }
            "state" => {
                let file = golden_dir().join(ev["file"].as_str().unwrap());
                assert_state(&h.as_ref().unwrap().rt, &file, theta_tol, &at);
            }
            _ => {
                let host = h.as_mut().unwrap();
                let line = ev["line"].as_str().unwrap();
                let reply = host.handle(line).unwrap();
                let exp = &ev["expect"];
                if exp.get("error").is_some() {
                    assert!(reply.get("error").is_some(), "{at}: {line}: ожидалась ошибка, ответ {reply}");
                    continue;
                }
                assert!(reply.get("error").is_none(), "{at}: {line}: {reply}");
                match line.split_whitespace().next().unwrap() {
                    "obs" => assert_eq!(reply["codes"], exp["codes"], "{at}: коды контроля качества"),
                    "forecast" => {
                        assert_eq!(reply["after_unix_hour"], exp["after_unix_hour"], "{at}: момент выпуска");
                        assert_eq!(reply["fallback"], exp["fallback"], "{at}: признак отката");
                        assert!((f(&reply["theta"]) - f(&exp["theta"])).abs() <= theta_tol, "{at}: множитель");
                        let rows = reply["q"].as_array().unwrap();
                        let nq = rows[0].as_array().unwrap().len();
                        let q: Vec<f32> = rows.iter().flat_map(|r| r.as_array().unwrap().iter().map(f)).collect();
                        let mu: Vec<f32> = reply["mu"].as_array().unwrap().iter().map(f).collect();
                        let i_med = host.rt.manifest.i_med;
                        for (k, row) in q.chunks(nq).enumerate() {
                            assert!(row.windows(2).all(|w| w[0] <= w[1]), "{at}: квантили не монотонны");
                            assert_eq!(row[i_med], mu[k], "{at}: медиана не средний квантиль");
                        }
                        if !exp["q"].is_null() {
                            let err = max_abs(&q, g.take(&exp["q"]));
                            assert!(err <= tol, "{at}: max|Δq| = {err:.2e} > {tol:.0e}");
                            worst = worst.max(err);
                        }
                    }
                    _ => {
                        for (k, v) in exp.as_object().unwrap() {
                            if k == "theta" {
                                assert!((f(&reply[k]) - f(v)).abs() <= theta_tol, "{at}: сводка {k}");
                            } else {
                                assert_eq!(&reply[k], v, "{at}: сводка {k}");
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{name}: max|Δq| Rust и эталона = {worst:.3e} °C");
    worst
}

#[test]
fn every_scenario_has_a_test() {
    let g = load();
    let names: Vec<&str> = g.doc["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "cold_aci",
            "restart",
            "extremes",
            "long",
            "qc",
            "rounding",
            "sparse",
            "int8",
            "fallback",
            "no_obs",
            "site_shift",
            "store_order"
        ]
    );
}

#[test]
fn cold_start_with_conformal_aci_and_bad_commands() {
    replay(&load(), "cold_aci");
}

#[test]
fn restart_from_python_state_forecasts_before_obs() {
    replay(&load(), "restart");
}

#[test]
fn extremes_edges_idle_and_long_idle() {
    replay(&load(), "extremes");
}

#[test]
fn long_run_with_idle_and_restarts() {
    replay(&load(), "long");
}

#[test]
fn quality_control_codes() {
    replay(&load(), "qc");
}

#[test]
fn rounding_on_arrival() {
    replay(&load(), "rounding");
}

#[test]
fn every_second_and_third_hour() {
    replay(&load(), "sparse");
}

#[test]
fn int8_graphs_with_own_tolerance() {
    replay(&load(), "int8");
}

#[test]
fn fallback_is_point_climatology() {
    replay(&load(), "fallback");
}

#[test]
fn forecast_before_first_obs_uses_device_clock() {
    replay(&load(), "no_obs");
}

#[test]
fn restart_with_refined_site() {
    replay(&load(), "site_shift");
}

#[test]
fn freshest_state_is_chosen_by_content() {
    replay(&load(), "store_order");
}

#[test]
fn python_state_roundtrips_byte_exact() {
    let g = load();
    let sc = g.scenario("restart");
    let raw = fs::read(golden_dir().join(sc["init_files"][0]["file"].as_str().unwrap())).unwrap();
    let site = [
        sc["lat"].as_f64().unwrap(),
        sc["lon"].as_f64().unwrap(),
        sc["elev"].as_f64().unwrap(),
    ];
    let mut rt = runtime(sc, site);
    rt.load_state(&raw).unwrap();
    assert_eq!(rt.filled(), rt.stream_window(), "эталон должен начинаться с полного окна");
    let mut back = Vec::new();
    rt.serialize(&mut back);
    assert_eq!(back, raw, "состояние после загрузки и записи изменилось");
}

#[test]
fn state_size_is_pinned() {
    let m = Manifest::load(golden_dir().join("model")).unwrap();
    assert_eq!(mayak_rt::state::nbytes(&m.dims), 3224);
    assert_eq!(m.state.nbytes, 3224);
}

#[test]
fn corrupted_state_is_rejected_and_leaves_cold_start() {
    let g = load();
    let sc = g.scenario("restart");
    let raw = fs::read(golden_dir().join(sc["init_files"][0]["file"].as_str().unwrap())).unwrap();
    let site = [
        sc["lat"].as_f64().unwrap(),
        sc["lon"].as_f64().unwrap(),
        sc["elev"].as_f64().unwrap(),
    ];
    let mut rt = runtime(sc, site);
    for bad in [&raw[..raw.len() - 1], &[b'X'; 3224][..]] {
        assert!(rt.load_state(bad).is_err());
    }
    let mut v = raw.clone();
    v[3] = 3; // прежняя версия
    assert!(rt.load_state(&v).is_err());
    let mut v = raw.clone();
    v[4..6].copy_from_slice(&5000u16.to_le_bytes()); // filled длиннее окна
    assert!(rt.load_state(&v).is_err());
    assert_eq!(rt.filled(), 0);
    assert_eq!(rt.last_hour(), None);
}

#[test]
fn calendar_matches_timeaxis() {
    let g = load();
    let c = &g.doc["calendar"];
    let hours = c["unix_hours"].as_array().unwrap();
    for (i, h) in hours.iter().enumerate() {
        let (d, hr) = doy_hour(h.as_i64().unwrap());
        assert_eq!(d.to_bits(), f(&c["doy"][i]).to_bits(), "doy часа {h}");
        assert_eq!(hr.to_bits(), f(&c["hour"][i]).to_bits(), "hour часа {h}");
        let how = c["hour_of_year"][i].as_u64().unwrap() as usize;
        assert_eq!(hour_of_year(h.as_i64().unwrap()), how, "час года часа {h}");
    }
}

#[test]
fn calibration_matches_metrics() {
    let g = load();
    let m = Manifest::load(golden_dir().join("model")).unwrap();
    let table = m.conformal_table().unwrap().unwrap();
    let (nq, im) = (m.dims.n_quantiles, m.i_med);
    let cal = &g.doc["calibration"];
    for case in cal["cases"].as_array().unwrap() {
        let mut q = g.take(&case["q"]).to_vec();
        if case["conformal"].as_bool().unwrap() {
            apply_conformal(&mut q, &table, nq, im);
        }
        apply_adaptive(&mut q, f(&case["theta"]), nq, im);
        let want = g.take(&case["expect"]);
        for (x, y) in q.iter().zip(want) {
            assert_eq!(x.to_bits(), y.to_bits(), "калибровка расходится до бита");
        }
    }
    let a = m.calibration.aci.as_ref().unwrap();
    let p = AciParams {
        target: a.target,
        gamma: a.gamma,
        max_factor: a.max_factor,
        interval: a.interval,
    };
    let aci = &cal["aci"];
    let mut theta = p.clip(aci["theta0"].as_f64().unwrap());
    for (k, s) in aci["scores"].as_array().unwrap().iter().enumerate() {
        assert_eq!(theta, f(&aci["theta_before"][k]), "θ перед наблюдением {k}");
        let s = score(s);
        if s.is_nan() {
            continue;
        }
        let (t, miss) = p.step(theta, s);
        assert_eq!(miss, aci["miss"][k].as_bool().unwrap());
        theta = t;
    }
    assert_eq!(theta, f(&aci["theta_end"]));
    let sc = &cal["score"];
    let qs = g.take(&sc["q"]);
    for (k, y) in sc["y"].as_array().unwrap().iter().enumerate() {
        let got = aci_score(f(y) as f64, &qs[k * nq..(k + 1) * nq], p.interval, im);
        let want = score(&sc["expect"][k]);
        assert!(
            got == want || (got.is_nan() && want.is_nan()),
            "aci_score {k}: {got} против {want}"
        );
    }
}
