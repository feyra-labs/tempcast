//! manifest.json, который пишет `mayak.runtime.graphs.export_graphs`. Все размеры
//! хоста берутся отсюда, а не из констант: крейт работает с любой конфигурацией модели.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Error, Result};

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Dims {
    pub horizon: usize,
    pub n_quantiles: usize,
    pub n_modes: usize,
    pub passport_dim: usize,
    pub history_days: usize,
    pub n_daily_summary: usize,
    pub day_row: usize,
    pub stream_window: usize,
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
    pub aci: Option<AciJson>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StateInfo {
    pub version: u8,
    pub header_bytes: usize,
    pub nbytes: usize,
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
        if d.ctx > d.stream_window || d.day_row != 4 || d.stream_window == 0 {
            return Err(Error::new("манифест: размеры окна несогласованы"));
        }
        for g in ["init", "step", "passport", "issue"] {
            if !self.graphs.contains_key(g) {
                return Err(Error::new(format!("манифест: нет графа {g}")));
            }
        }
        if self.state.version != crate::state::VERSION || self.state.nbytes != crate::state::nbytes(d) {
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

    /// Пределы физических диапазонов [T, P, RH] - те же, что mayak.data.qc.PHYS.
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
        Ok(Some(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ))
    }
}
