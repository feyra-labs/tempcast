//! Потоковый рантайм МАЯК на Rust.
//!
//! Модель исполняется графами ONNX без внутреннего состояния: `init`, `step`, `window`,
//! `resync`, `issue`. Этот крейт - только хост: сырое окно наблюдений, кольца вкладов в
//! моды и строк суточного накопителя, буфер энкодера, причинный контроль качества,
//! калибровка интервалов, формат состояния, атомарная запись в два чередующихся файла,
//! построчный протокол, откат к климатологии точки при сбое и правило смены координат
//! прибора между перезапусками. Таблицу климатологии
//! считает граф старта, хост только ищет в ней час года. Арифметики модели здесь нет.
//!
//! Поведение закреплено эталонными векторами и сценариями.

pub mod calendar;
pub mod calib;
pub mod error;
pub mod graphs;
pub mod host;
pub mod manifest;
pub mod memory;
pub mod qc;
pub mod record;
pub mod runtime;
pub mod site;
pub mod state;
pub mod store;

pub use error::{Error, Result};
pub use graphs::Precision;
pub use host::Host;
pub use manifest::Manifest;
pub use runtime::{Forecast, Runtime, RuntimeOptions};
