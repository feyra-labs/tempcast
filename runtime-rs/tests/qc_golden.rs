//! Тест: причинный QC хоста совпадает с эталонными векторами кодов.
//!
//! Эталон посчитан Python-реализацией и лежит среди данных тестов репозитория. Он
//! пересоздаётся генератором эталона кодов только при изменении правил или порогов QC.
use std::path::PathBuf;

use mayak_rt::qc::{causal_codes, CausalQc, QcConfig};
use mayak_rt::Manifest;
use serde_json::Value;

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn golden() -> Value {
    let text = std::fs::read_to_string(repo_path("tests/data/qc_causal/golden.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn config(doc: &Value) -> QcConfig {
    serde_json::from_value(doc["config"].clone()).unwrap()
}

fn bounds(doc: &Value) -> [[f64; 2]; 3] {
    let mut out = [[0.0; 2]; 3];
    for (c, name) in ["T", "P", "RH"].into_iter().enumerate() {
        let lo = doc["phys"][name][0].as_f64().unwrap();
        let hi = doc["phys"][name][1].as_f64().unwrap();
        out[c] = [lo, hi];
    }
    out
}

fn code(v: &Value) -> u8 {
    v.as_u64().unwrap() as u8
}

struct Case {
    name: String,
    elev: Option<f64>,
    obs: Vec<[Option<f64>; 3]>,
    codes: Vec<[u8; 3]>,
}

fn cases(doc: &Value) -> Vec<Case> {
    let mut out = Vec::new();
    for c in doc["cases"].as_array().unwrap() {
        let mut obs = Vec::new();
        for row in c["x"].as_array().unwrap() {
            obs.push([row[0].as_f64(), row[1].as_f64(), row[2].as_f64()]);
        }
        let mut codes = Vec::new();
        for row in c["codes"].as_array().unwrap() {
            codes.push([code(&row[0]), code(&row[1]), code(&row[2])]);
        }
        out.push(Case {
            name: c["name"].as_str().unwrap().to_string(),
            elev: c["elev"].as_f64(),
            obs,
            codes,
        });
    }
    out
}

#[test]
fn batch_codes_match_reference() {
    let doc = golden();
    let (cfg, b) = (config(&doc), bounds(&doc));
    let all = cases(&doc);
    assert!(all.len() >= 5);
    for case in all {
        let mut x = Vec::with_capacity(case.obs.len());
        let mut present = Vec::with_capacity(case.obs.len());
        for o in &case.obs {
            x.push(o.map(|v| v.unwrap_or(0.0) as f32));
            present.push(o.map(|v| v.is_some()));
        }
        let got = causal_codes(&x, &present, case.elev, &b, &cfg);
        assert_eq!(got.len(), case.codes.len());
        for (i, (g, want)) in got.iter().zip(&case.codes).enumerate() {
            assert_eq!(g, want, "{}: час {i}", case.name);
        }
    }
}

#[test]
fn stream_codes_match_reference() {
    let doc = golden();
    let (cfg, b) = (config(&doc), bounds(&doc));
    for case in cases(&doc) {
        let mut qc = CausalQc::new(cfg.clone(), b, case.elev);
        assert_eq!(qc.size(), cfg.lookback_hours + 1);
        for (i, o) in case.obs.iter().enumerate() {
            let (v, got) = qc.push(*o);
            assert_eq!(got, case.codes[i], "{}: час {i}", case.name);
            for c in 0..3 {
                if got[c] != 0 {
                    assert_eq!(v[c], 0.0, "{}: час {i}", case.name);
                }
            }
        }
    }
}

#[test]
fn manifest_carries_the_reference_config() {
    let doc = golden();
    let m = Manifest::load(repo_path("tests/data/runtime_golden/model")).unwrap();
    let want = config(&doc);
    assert_eq!(m.qc.lookback_hours, want.lookback_hours);
    assert_eq!(m.qc.scale_floor, want.scale_floor);
    assert_eq!(m.qc.spike_thresh, want.spike_thresh);
    assert_eq!(m.qc.stuck_t_alone_hours, want.stuck_t_alone_hours);
}
