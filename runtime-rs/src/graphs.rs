//! Четыре графа ONNX (init, step, passport, issue) поверх ONNX Runtime.
//!
//! Входы подаются по именам и без копирования (TensorRef на срезы хоста); графу
//! подаются только те входы, которые в нём остались после экспорта (манифест
//! перечисляет фактические). Выходы копируются в заранее выделенные буферы хоста,
//! поэтому после старта рантайм не растит свою память.
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;

use crate::manifest::{GraphEntry, Manifest};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Int8,
}

/// Вход графа: имя, форма, данные.
pub type Feed<'a> = (&'a str, &'a [usize], &'a [f32]);

pub struct Graph {
    name: &'static str,
    session: Session,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Graph {
    fn open(m: &Manifest, name: &'static str, precision: Precision, threads: usize) -> Result<Self> {
        let GraphEntry {
            fp32,
            int8,
            inputs,
            outputs,
        } = &m.graphs[name];
        let file = match precision {
            Precision::Fp32 => fp32,
            Precision::Int8 => int8
                .as_ref()
                .ok_or_else(|| Error::new(format!("граф {name}: int8 не экспортирован")))?,
        };
        let path = m.dir.join(file);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .commit_from_file(&path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        Ok(Graph {
            name,
            session,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        })
    }

    /// Прогон графа: `feeds` - все известные хосту входы (лишние пропускаются),
    /// `outs` - буферы под выходы в порядке манифеста.
    pub fn run(&mut self, feeds: &[Feed<'_>], outs: &mut [&mut [f32]]) -> Result<()> {
        let mut values: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(self.inputs.len());
        for want in &self.inputs {
            let (_, shape, data) = feeds
                .iter()
                .find(|(n, _, _)| n == want)
                .ok_or_else(|| Error::new(format!("граф {}: хост не знает вход {want}", self.name)))?;
            let t = TensorRef::from_array_view((shape.to_vec(), *data))?;
            values.push((want.clone(), t.into()));
        }
        let res = self.session.run(values)?;
        if outs.len() != self.outputs.len() {
            return Err(Error::new(format!(
                "граф {}: {} выходов, хост ждёт {}",
                self.name,
                self.outputs.len(),
                outs.len()
            )));
        }
        for (name, dst) in self.outputs.iter().zip(outs.iter_mut()) {
            let (_, src) = res[name.as_str()].try_extract_tensor::<f32>()?;
            if src.len() != dst.len() {
                return Err(Error::new(format!(
                    "граф {}: выход {name} длины {}, ожидалось {}",
                    self.name,
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        Ok(())
    }
}

pub struct Graphs {
    pub init: Graph,
    pub step: Graph,
    pub passport: Graph,
    pub issue: Graph,
}

impl Graphs {
    pub fn open(m: &Manifest, precision: Precision, threads: usize) -> Result<Self> {
        Ok(Graphs {
            init: Graph::open(m, "init", precision, threads)?,
            step: Graph::open(m, "step", precision, threads)?,
            passport: Graph::open(m, "passport", precision, threads)?,
            issue: Graph::open(m, "issue", precision, threads)?,
        })
    }
}
