//! Тест: запись значений прибором совпадает с эталонной реализацией.
//!
//! Эталонные случаи - значения ровно посередине между соседними значениями сетки, в
//! том числе отрицательные, и значения рядом с серединой. Файл лежит среди данных тестов
//! репозитория, по нему же проверяется реализация на Python.
use std::path::PathBuf;

use mayak_rt::qc::{CausalQc, QcConfig};
use mayak_rt::record::{record_channel, record_values};
use serde_json::Value;

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn cases() -> Vec<(usize, f32, f32)> {
    let text = std::fs::read_to_string(repo_path("tests/data/recording/cases.json")).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let mut out = Vec::new();
    for c in doc["cases"].as_array().unwrap() {
        let ch = match c["channel"].as_str().unwrap() {
            "T" => 0,
            "P" => 1,
            "RH" => 2,
            other => panic!("неизвестный канал {other}"),
        };
        let x = c["x"].as_f64().unwrap() as f32;
        let want = c["recorded"].as_f64().unwrap() as f32;
        out.push((ch, x, want));
    }
    out
}

#[test]
fn record_matches_reference_cases() {
    let all = cases();
    assert!(all.len() >= 40);
    for (ch, x, want) in all {
        let got = record_channel(x, ch);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "канал {ch}, значение {x}: {got} вместо {want}"
        );
    }
}

#[test]
fn stream_sees_recorded_values() {
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(repo_path("tests/data/qc_causal/golden.json")).unwrap()).unwrap();
    let cfg: QcConfig = serde_json::from_value(doc["config"].clone()).unwrap();
    let bounds = [[-90.0, 60.0], [300.0, 1100.0], [0.0, 100.0]];
    let mut qc = CausalQc::new(cfg, bounds, Some(200.0));
    let (v, codes) = qc.push([Some(12.5), Some(1001.25), Some(60.5)]);
    assert_eq!(codes, [0, 0, 0]);
    assert_eq!(v, record_values([12.5, 1001.25, 60.5]));
    assert_eq!(v, [12.0, 1001.2, 60.0]);
    let (_, codes) = qc.push([Some(60.4), Some(1100.04), Some(100.4)]);
    assert_eq!(codes, [0, 0, 0], "после записи значения попадают в диапазоны");
    let (_, codes) = qc.push([Some(60.6), Some(1100.06), Some(100.6)]);
    assert!(codes.iter().all(|c| c & mayak_rt::qc::RANGE != 0), "{codes:?}");
}
