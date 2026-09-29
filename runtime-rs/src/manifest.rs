//! manifest.json, который пишет `mayak.runtime.graphs.export_graphs`. Все размеры
//! хоста берутся отсюда, а не из констант: крейт работает с любой конфигурацией модели.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::graphs::Precision;
use crate::qc::QcConfig;
use crate::{Error, Result};

pub const FORMAT: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct Dims {
    pub horizon: usize,
    pub n_quantiles: usize,
    pub n_modes: usize,
    pub passport_dim: usize,
    pub history: usize,
    pub day_row: usize,
    pub stream_window: usize,
    pub stream_edge: usize,
    pub stream_tail: usize,
    pub ctx: usize,
    pub loc_dim: usize,
    pub encoder_width: usize,
    pub enc_buf_len: usize,
    pub n_coef: [usize; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphEntry {
    pub fp32: String,
    #[serde(default)]
    pub int8: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AciJson {
    pub target: f64,
    pub gamma: f64,
    pub max_factor: f64,
    pub interval: [usize; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    pub conformal: Option<String>,
    /// Точность графов, на выходах которых подогнана конформная таблица: fp32 или int8.
    #[serde(default)]
    pub precision: Option<String>,
    pub aci: Option<AciJson>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StateInfo {
    pub version: u8,
    pub header_bytes: usize,
    pub nbytes: usize,
    pub resync_hours: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub dims: Dims,
    pub quantiles: Vec<f64>,
    pub i_med: usize,
    pub zq: Vec<f32>,
    pub raw_channels: Vec<String>,
    pub phys: HashMap<String, [f64; 2]>,
    pub qc: QcConfig,
    pub state: StateInfo,
    pub graphs: HashMap<String, GraphEntry>,
    pub calibration: Calibration,
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Manifest {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let text = std::fs::read_to_string(dir.join("manifest.json"))
            .map_err(|e| Error::new(format!("{}: {e}", dir.join("manifest.json").display())))?;
        let mut m: Manifest = serde_json::from_str(&text)?;
        if m.format != FORMAT {
            return Err(Error::new(format!(
                "формат манифеста {}, рантайм читает {FORMAT}",
                m.format
            )));
        }
        m.dir = dir.to_path_buf();
        m.validate()?;
        Ok(m)
    }

    fn validate(&self) -> Result<()> {
        let d = &self.dims;
        if self.quantiles.len() != d.n_quantiles || self.zq.len() != d.n_quantiles || self.i_med >= d.n_quantiles {
            return Err(Error::new("манифест: набор квантилей несогласован"));
        }
        if self.raw_channels != ["T", "P", "RH"] {
            return Err(Error::new(format!(
                "манифест: каналы {:?}, ожидались T, P, RH",
                self.raw_channels
            )));
        }
        if d.ctx > d.stream_window
            || d.day_row != 4
            || d.history > d.stream_window
            || !d.history.is_multiple_of(24)
            || d.stream_edge + d.stream_tail != d.history
            || d.stream_tail == 0
            || d.stream_edge == 0
        {
            return Err(Error::new("манифест: размеры окна несогласованы"));
        }
        if self.state.resync_hours != crate::runtime::RESYNC_HOURS {
            return Err(Error::new(format!(
                "манифест: пересчёт мод каждые {} ч, рантайм пересчитывает каждые {} ч",
                self.state.resync_hours,
                crate::runtime::RESYNC_HOURS
            )));
        }
        for g in ["init", "step", "window", "resync", "issue"] {
            if !self.graphs.contains_key(g) {
                return Err(Error::new(format!("манифест: нет графа {g}")));
            }
        }
        if self.state.version != crate::state::VERSION
            || self.state.nbytes != crate::state::nbytes(d)
            || self.state.header_bytes != crate::state::HEADER
        {
            return Err(Error::new(format!(
                "манифест: формат состояния v{} на {} Б, рантайм пишет v{} на {} Б",
                self.state.version,
                self.state.nbytes,
                crate::state::VERSION,
                crate::state::nbytes(d)
            )));
        }
        Ok(())
    }

    /// Пределы физических диапазонов каналов T, P, RH из манифеста модели.
    pub fn phys_bounds(&self) -> [[f64; 2]; 3] {
        ["T", "P", "RH"].map(|c| self.phys[c])
    }

    /// Развёрнутая конформная таблица (horizon × NQ), если она экспортирована.
    pub fn conformal_table(&self) -> Result<Option<Vec<f32>>> {
        let Some(name) = &self.calibration.conformal else {
            return Ok(None);
        };
        let raw = std::fs::read(self.dir.join(name))?;
        let n = self.dims.horizon * self.dims.n_quantiles;
        if raw.len() != 4 * n {
            return Err(Error::new(format!("{name}: {} Б, ожидалось {}", raw.len(), 4 * n)));
        }
        let table: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let nq = self.dims.n_quantiles;
        if table.chunks_exact(nq).any(|row| row[self.i_med] != 0.0) {
            return Err(Error::new(format!(
                "{name}: поправка медианы не нулевая - таблица сдвигает точечный прогноз; \
                 подгоните таблицу заново"
            )));
        }
        Ok(Some(table))
    }

    /// Конформная таблица, которую можно применять к графам заданной точности.
    ///
    /// Таблица, подогнанная на другой точности или без записанной точности, не
    /// применяется: вместо неё возвращается причина для лога.
    pub fn conformal_for(&self, precision: Precision) -> Result<(Option<Vec<f32>>, Option<String>)> {
        let Some(table) = self.conformal_table()? else {
            return Ok((None, None));
        };
        match self.calibration.precision.as_deref() {
            Some(p) if p == precision.as_str() => Ok((Some(table), None)),
            Some(p) => Ok((
                None,
                Some(format!(
                    "конформная таблица подогнана на {p}, графы считают в {}: таблица не применяется",
                    precision.as_str()
                )),
            )),
            None => Ok((
                None,
                Some(format!(
                    "точность конформной таблицы не записана в манифесте, графы считают в {}: \
                     таблица не применяется",
                    precision.as_str()
                )),
            )),
        }
    }
}
