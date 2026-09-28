//! Тест: конформная таблица применяется только к графам той точности, на которой она
//! подогнана, и никогда не сдвигает медиану.
//!
//! Модель - копия эталонной, в манифесте которой те же файлы графов объявлены ещё и как
//! int8. Так проверяется решение хоста о таблице, а не качество квантизации.
use std::path::{Path, PathBuf};

use mayak_rt::calib::{apply_adaptive, apply_conformal, order_around_median};
use mayak_rt::{Manifest, Precision, Runtime, RuntimeOptions};
use serde_json::Value;

const LAT: f64 = 52.37;
const LON: f64 = 4.9;

fn golden_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/data/runtime_golden/model")
}

/// Копия эталонной модели с заданной точностью таблицы в манифесте.
fn model_copy(name: &str, precision: Option<&str>) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for e in std::fs::read_dir(golden_model()).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    let path = dir.join("manifest.json");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for g in doc["graphs"].as_object_mut().unwrap().values_mut() {
        g["int8"] = g["fp32"].clone();
    }
    doc["calibration"]["precision"] = precision.map(Value::from).unwrap_or(Value::Null);
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    dir
}

fn runtime(dir: &Path, precision: Precision, conformal: bool) -> Runtime {
    let opts = RuntimeOptions {
        precision,
        conformal,
        ..Default::default()
    };
    Runtime::new(dir, LAT, LON, 0.0, &opts).unwrap()
}

/// Выпуск после суток наблюдений: квантили всех лидов подряд.
fn issue(rt: &mut Runtime) -> Vec<f32> {
    for k in 0..30 {
        let obs = [Some(8.0 + 0.3 * k as f64), Some(1011.0), Some(72.0)];
        rt.step(obs, 100.0 + (k / 24) as f32, (k % 24) as f32).unwrap();
    }
    let h = rt.horizon();
    let doy: Vec<f32> = (0..h).map(|k| ((101 * 24 + 6 + k) as f32) / 24.0).collect();
    let hour: Vec<f32> = (0..h).map(|k| ((6 + k) % 24) as f32).collect();
    rt.forecast(&doy, &hour).unwrap().q.clone()
}

#[test]
fn fp32_table_is_rejected_by_int8_graphs() {
    let dir = model_copy("fp32_table", Some("fp32"));
    let m = Manifest::load(&dir).unwrap();
    let (table, why) = m.conformal_for(Precision::Int8).unwrap();
    assert!(table.is_none());
    assert!(why.unwrap().contains("fp32"));
    assert!(!runtime(&dir, Precision::Int8, true).conformal_applied());
    assert!(runtime(&dir, Precision::Fp32, true).conformal_applied());
}

#[test]
fn int8_table_is_rejected_by_fp32_graphs() {
    let dir = model_copy("int8_table", Some("int8"));
    assert!(!runtime(&dir, Precision::Fp32, true).conformal_applied());
    assert!(runtime(&dir, Precision::Int8, true).conformal_applied());
}

#[test]
fn table_without_precision_is_never_applied() {
    let dir = model_copy("untagged_table", None);
    for p in [Precision::Fp32, Precision::Int8] {
        let (table, why) = Manifest::load(&dir).unwrap().conformal_for(p).unwrap();
        assert!(table.is_none() && why.is_some());
        assert!(!runtime(&dir, p, true).conformal_applied());
    }
}

#[test]
fn rejected_table_leaves_raw_forecast() {
    let dir = model_copy("raw_forecast", Some("fp32"));
    let rejected = issue(&mut runtime(&dir, Precision::Int8, true));
    let raw = issue(&mut runtime(&dir, Precision::Int8, false));
    assert_eq!(rejected, raw);
}

#[test]
fn table_changes_width_not_median() {
    let dir = model_copy("median", Some("fp32"));
    let raw = issue(&mut runtime(&dir, Precision::Fp32, false));
    let cal = issue(&mut runtime(&dir, Precision::Fp32, true));
    let m = Manifest::load(&dir).unwrap();
    let (nq, im) = (m.dims.n_quantiles, m.i_med);
    assert_ne!(raw, cal);
    for (a, b) in raw.chunks(nq).zip(cal.chunks(nq)) {
        assert_eq!(a[im].to_bits(), b[im].to_bits());
        assert!(b.windows(2).all(|w| w[0] <= w[1]));
    }
}

#[test]
fn table_with_median_shift_is_refused() {
    let dir = model_copy("median_shift", Some("fp32"));
    let m = Manifest::load(&dir).unwrap();
    let path = dir.join("conformal.f32");
    let mut raw = std::fs::read(&path).unwrap();
    let k = 4 * (5 * m.dims.n_quantiles + m.i_med);
    raw[k..k + 4].copy_from_slice(&0.1f32.to_le_bytes());
    std::fs::write(&path, raw).unwrap();
    let err = m.conformal_table().unwrap_err().to_string();
    assert!(err.contains("медиан"), "{err}");
    let opts = RuntimeOptions::default();
    assert!(Runtime::new(&dir, LAT, LON, 0.0, &opts).is_err());
}

#[test]
fn ordering_keeps_the_median() {
    let mut row = [3.0, -1.0, 2.5, 1.0, 0.5, 4.0, 3.5];
    order_around_median(&mut row, 3);
    assert_eq!(row, [-1.0, -1.0, 1.0, 1.0, 1.0, 4.0, 4.0]);
    let mut nan = [0.0, f32::NAN, 0.5, 1.0, 2.0, f32::NAN, 3.0];
    order_around_median(&mut nan, 3);
    assert!(nan[0].is_nan() && nan[1].is_nan() && nan[5].is_nan() && nan[6].is_nan());
    assert_eq!(&nan[2..5], &[0.5, 1.0, 2.0]);

    let table = [0.4, -0.3, 2.0, 0.0, -2.0, 0.1, 0.2];
    let mut q = [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0];
    apply_conformal(&mut q, &table, 7, 3);
    assert_eq!(q[3], 0.0);
    assert!(q.windows(2).all(|w| w[0] <= w[1]));
    apply_adaptive(&mut q, 0.7, 7, 3);
    assert_eq!(q[3], 0.0);
}
