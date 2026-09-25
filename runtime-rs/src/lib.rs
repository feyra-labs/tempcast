//! Потоковый рантайм МАЯК на Rust.
//!
//! Модель исполняется четырьмя графами ONNX без внутреннего состояния (`init`, `step`,
//! `passport`, `issue`; экспорт - `mayak.runtime.graphs`). Этот крейт - только хост:
//! кольцо сырого окна, буфер энкодера, суточный накопитель, календарь, причинный QC,
//! калибровка интервалов, формат состояния v3 (тот же, что у Python), атомарная запись
//! в два чередующихся файла и откат к климатологии при сбое. Арифметики модели здесь нет.
//!
//! Эталон - Python (`mayak.runtime.streaming.StreamingMayak`); совпадение проверяется
//! эталонными векторами `tests/data/runtime_golden` (тест `tests/golden.rs`).

pub mod calendar;
pub mod calib;
pub mod error;
pub mod graphs;
pub mod manifest;
pub mod memory;
pub mod qc;
pub mod runtime;
pub mod state;
pub mod store;

pub use error::{Error, Result};
pub use graphs::Precision;
pub use manifest::Manifest;
pub use runtime::{Forecast, Runtime, RuntimeOptions};
