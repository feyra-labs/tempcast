//! Тест: компилируемый рантайм совпадает с эталонными
//! векторами, сгенерированными Python-реализацией (mayak/runtime/golden.py).
//!
//! Эталон лежит в tests/data/runtime_golden репозитория. Пересоздание:
//!     python scripts/make_runtime_golden.py
use std::path::PathBuf;

use mayak_rt::calendar::doy_hour;
use mayak_rt::calib::{aci_score, apply_adaptive, apply_conformal, AciParams};
use mayak_rt::state::{Snapshot, HEADER};
use mayak_rt::{Manifest, Runtime, RuntimeOptions};
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
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    let raw = std::fs::read(dir.join("golden.f32")).unwrap();
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
}

fn f(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

fn obs(v: &Value) -> Option<f64> {
    match v {
        Value::Null => None,
        Value::String(s) if s == "nan" => Some(f64::NAN),
        v => v.as_f64(),
    }
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

fn runtime(sc: &Value) -> Runtime {
    let opts = RuntimeOptions {
        conformal: sc["conformal"].as_bool().unwrap(),
        aci: sc["aci"].as_bool().unwrap(),
        ..Default::default()
    };
    Runtime::new(
        golden_dir().join("model"),
        sc["lat"].as_f64().unwrap(),
        sc["lon"].as_f64().unwrap(),
        sc["elev"].as_f64().unwrap(),
        &opts,
    )
    .unwrap()
}

fn scenario(g: &Golden, name: &str) -> &'static Value {
    let sc = g.doc["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == name)
        .unwrap()
        .clone();
    Box::leak(Box::new(sc))
}

/// Прогон сценария: наибольшее расхождение квантилей с эталоном по записанным выпускам.
fn replay(g: &Golden, sc: &Value, rt: &mut Runtime) -> f32 {
    let tol = g.doc["tolerance"]["q_abs"].as_f64().unwrap() as f32;
    let nq = rt.n_quantiles();
    let h = rt.horizon();
    let mut worst = 0.0f32;
    for (i, ev) in sc["events"].as_array().unwrap().iter().enumerate() {
        if ev["op"] == "step" {
            let o = ev["obs"].as_array().unwrap();
            rt.step([obs(&o[0]), obs(&o[1]), obs(&o[2])], ev["hour"].as_i64().unwrap())
                .unwrap();
            continue;
        }
        let fc = rt.forecast().unwrap();
        assert!(!fc.fallback);
        if !ev["q"].is_null() {
            let refq = g.take(&ev["q"]);
            assert_eq!(refq.len(), h * nq);
            let err = max_abs(&fc.q, refq);
            assert!(
                err <= tol,
                "{}: событие {i}: max|Δq| = {err:.2e} > {tol:.0e}",
                sc["name"]
            );
            for row in fc.q.chunks(nq) {
                assert!(row.windows(2).all(|w| w[0] <= w[1]), "квантили не монотонны");
            }
            worst = worst.max(err);
        }
        let dt = (rt.theta() - f(&ev["theta"])).abs();
        assert!(
            dt <= 1e-6,
            "{}: событие {i}: θ = {} против эталона {}",
            sc["name"],
            rt.theta(),
            ev["theta"]
        );
    }
    let fin = &sc["final"];
    assert!((rt.theta() - f(&fin["theta"])).abs() <= 1e-6);
    assert_eq!(
        rt.aci_updates(),
        fin["aci_updates"].as_u64().unwrap(),
        "{}: число обратных связей ACI",
        sc["name"]
    );
    assert_eq!(
        rt.aci_misses(),
        fin["aci_misses"].as_u64().unwrap(),
        "{}: число промахов ACI",
        sc["name"]
    );
    assert_eq!(rt.filled() as u64, fin["filled"].as_u64().unwrap());
    assert_eq!(rt.last_hour(), fin["last_hour"].as_i64());
    assert_eq!(rt.idle_hours(), fin["idle_hours"].as_u64().unwrap());
    println!("{}: max|Δq| Rust и эталона = {worst:.3e} °C", sc["name"]);
    worst
}

#[test]
fn cold_start_with_conformal_and_aci() {
    let g = load();
    let sc = scenario(&g, "cold_aci");
    replay(&g, sc, &mut runtime(sc));
}

#[test]
fn extremes_edges_idle_and_long_idle() {
    let g = load();
    let sc = scenario(&g, "extremes");
    replay(&g, sc, &mut runtime(sc));
}

#[test]
fn restart_from_python_state_with_full_window() {
    let g = load();
    let sc = scenario(&g, "restart");
    let raw = std::fs::read(golden_dir().join(sc["init_state"].as_str().unwrap())).unwrap();
    let mut rt = runtime(sc);
    rt.load_state(&raw).unwrap();
    assert_eq!(
        rt.filled(),
        rt.stream_window(),
        "эталон должен начинаться с полного окна"
    );
    // Состояние Python, затем Rust, затем байты: совпадение до байта.
    let mut back = Vec::new();
    rt.serialize(&mut back);
    assert_eq!(back, raw, "serialize(load(S)) ≠ S");
    replay(&g, sc, &mut rt);

    // Конечное состояние: заголовок до множителя калибровки и всё окно - точно.
    let end = std::fs::read(golden_dir().join(sc["final"]["state"].as_str().unwrap())).unwrap();
    rt.serialize(&mut back);
    assert_eq!(back.len(), end.len());
    assert_eq!(back[..16], end[..16], "заголовок состояния");
    assert_eq!(back[HEADER..], end[HEADER..], "сырое окно и маски");
    let bounds = rt.manifest.phys_bounds();
    let d = &rt.manifest.dims;
    let (a, b) = (
        Snapshot::parse(&back, d, &bounds).unwrap(),
        Snapshot::parse(&end, d, &bounds).unwrap(),
    );
    assert!((a.theta - b.theta).abs() <= 1e-6);
    assert_eq!(a.site, b.site);
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
    let sc = scenario(&g, "restart");
    let raw = std::fs::read(golden_dir().join(sc["init_state"].as_str().unwrap())).unwrap();
    let mut rt = runtime(sc);
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
fn restart_at_any_hour_continues_run() {
    let g = load();
    let sc = scenario(&g, "cold_aci");
    let events = sc["events"].as_array().unwrap();
    let mut live = runtime(sc);
    let mut buf = Vec::new();
    let mut back: Option<Runtime> = None;
    for (i, ev) in events.iter().enumerate() {
        if ev["op"] != "step" {
            continue;
        }
        let o = ev["obs"].as_array().unwrap();
        let (x, hour) = ([obs(&o[0]), obs(&o[1]), obs(&o[2])], ev["hour"].as_i64().unwrap());
        live.step(x, hour).unwrap();
        if let Some(b) = back.as_mut() {
            b.step(x, hour).unwrap();
        }
        if i == 150 {
            live.serialize(&mut buf);
            let mut b = runtime(sc);
            b.load_state(&buf).unwrap();
            back = Some(b);
        }
    }
    let mut b = back.unwrap();
    let q1 = live.forecast().unwrap().q.clone();
    let q2 = b.forecast().unwrap().q.clone();
    assert!(max_abs(&q1, &q2) <= 5e-4);
    let (mut s1, mut s2) = (Vec::new(), Vec::new());
    live.serialize(&mut s1);
    b.serialize(&mut s2);
    assert_eq!(s1[HEADER..], s2[HEADER..], "окно после перезапуска расходится");
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
