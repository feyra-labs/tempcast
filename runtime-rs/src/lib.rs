//! Потоковый рантайм МАЯК на Rust.
//!
//! Модель исполняется графами ONNX без внутреннего состояния: `init`, `step`, `window`,
//! `resync`, `issue`. Этот крейт - только хост: сырое окно наблюдений, кольца вкладов в
//! моды и строк суточного накопителя, буфер энкодера, причинный контроль качества,
//! калибровка интервалов, формат состояния, атомарная запись в два чередующихся файла и
//! откат к климатологии при сбое. Арифметики модели здесь нет.
//!
//! Эталон - реализация на Python; совпадение проверяется эталонными векторами.

pub mod calendar;
pub mod calib;
pub mod error;
pub mod graphs;
pub mod manifest;
pub mod memory;
pub mod qc;
pub mod record;
pub mod runtime;
pub mod state;
pub mod store;

pub use error::{Error, Result};
pub use graphs::Precision;
pub use manifest::Manifest;
pub use runtime::{Forecast, Runtime, RuntimeOptions};
