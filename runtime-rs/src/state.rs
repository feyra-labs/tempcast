//! Персистентное состояние: заголовок и сырое окно наблюдений.
//!
//! Формат общий с эталонной реализацией на Python, байт в байт.
//!
//! | поле               | тип   | число                               |
//! |--------------------|-------|-------------------------------------|
//! | magic              | "MYK" | 3 Б                                 |
//! | version            | u8    | 4                                   |
//! | filled             | u16   | часов окна после холодного старта   |
//! | reserved           | u16   |                                     |
//! | last_hour          | i64   | абсолютный час последнего шага      |
//! | aci_theta          | f32   | множитель калибровки в логарифме    |
//! | lat, lon, elev     | f32   | точка, для которой записано окно    |
//! | температура        | i8    | W, целые градусы                    |
//! | давление           | u16   | W, десятые гектопаскаля             |
//! | влажность          | i8    | W, целые проценты                   |
//! | маска наличия      | биты  | три бита на час, младший первым     |
//! | маска годности     | биты  | три бита на час, младший первым     |
//!
//! Всё little-endian, окно от старых часов к новым. Если шагов не было, в last_hour
//! записано наименьшее i64. Всё остальное состояние рантайма восстанавливается из окна.
use crate::manifest::Dims;
use crate::{Error, Result};

pub const MAGIC: &[u8; 3] = b"MYK";
pub const VERSION: u8 = 4;
pub const HEADER: usize = 32;
pub const NO_HOUR: i64 = i64::MIN;
/// Шаг сетки хранения по каналам: доли единицы, в которых записан канал.
pub const STORE_SCALE: [f64; 3] = [1.0, 10.0, 1.0];
const STORE_LO: [f64; 3] = [-128.0, 0.0, -128.0];
const STORE_HI: [f64; 3] = [127.0, 65535.0, 127.0];

/// Размер одной битовой маски окна, байт.
pub fn mask_bytes(w: usize) -> usize {
    (3 * w).div_ceil(8)
}

/// Размер состояния для размеров модели, байт.
pub fn nbytes(d: &Dims) -> usize {
    let w = d.stream_window;
    HEADER + 4 * w + 2 * mask_bytes(w)
}

/// Значение канала на сетке хранения: сетка записи прибора, обрезанная до диапазона
/// типа хранения. Значение вне физического диапазона после обрезки остаётся вне него.
pub fn to_store(v: f32, c: usize) -> f32 {
    let q = (v as f64 * STORE_SCALE[c])
        .round_ties_even()
        .clamp(STORE_LO[c], STORE_HI[c]);
    (q / STORE_SCALE[c]) as f32
}

/// Разобранное состояние.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub filled: usize,
    pub last_hour: Option<i64>,
    pub theta: f32,
    pub site: [f32; 3],
    /// Окно от старых часов к новым: значения на сетке хранения, наличие и годность.
    pub raw: Vec<[f32; 3]>,
    pub present: Vec<[bool; 3]>,
    pub valid: Vec<[bool; 3]>,
}

fn f32_at(raw: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
}

fn unpack(bits: &[u8], w: usize) -> Vec<[bool; 3]> {
    (0..w)
        .map(|h| std::array::from_fn(|c| bits[(h * 3 + c) / 8] >> ((h * 3 + c) % 8) & 1 == 1))
        .collect()
}

fn pack(m: &[[bool; 3]], out: &mut Vec<u8>) {
    let mut bits = vec![0u8; mask_bytes(m.len())];
    for (h, row) in m.iter().enumerate() {
        for (c, &b) in row.iter().enumerate() {
            if b {
                bits[(h * 3 + c) / 8] |= 1 << ((h * 3 + c) % 8);
            }
        }
    }
    out.extend_from_slice(&bits);
}

impl Snapshot {
    /// Разбор с проверками целостности.
    ///
    /// `bounds` - физические диапазоны каналов: годное значение вне них означает
    /// повреждённое состояние.
    pub fn parse(raw: &[u8], d: &Dims, bounds: &[[f64; 2]; 3]) -> Result<Self> {
        if raw.len() < 4 || &raw[..3] != MAGIC {
            return Err(Error::new("не состояние МАЯК: нет заголовка"));
        }
        if raw[3] != VERSION {
            return Err(Error::new(format!(
                "версия состояния {}, рантайм читает {VERSION}; прежние версии не хранят сырое \
                 окно целиком, нужен холодный старт",
                raw[3]
            )));
        }
        if raw.len() != nbytes(d) {
            return Err(Error::new(format!(
                "состояние {} Б не соответствует конфигу модели (ожидалось {} Б)",
                raw.len(),
                nbytes(d)
            )));
        }
        let w = d.stream_window;
        let filled = u16::from_le_bytes([raw[4], raw[5]]) as usize;
        let last = i64::from_le_bytes(raw[8..16].try_into().expect("8 байт"));
        let theta = f32_at(raw, 16);
        let site = [f32_at(raw, 20), f32_at(raw, 24), f32_at(raw, 28)];
        if !theta.is_finite() {
            return Err(Error::new(format!(
                "повреждённый множитель калибровки в состоянии: {theta}"
            )));
        }
        if filled > w || (last == NO_HOUR && filled > 0) {
            return Err(Error::new(format!(
                "повреждённый заголовок состояния: filled={filled}, окно {w}, последний час {last}"
            )));
        }
        let mut off = HEADER;
        let t: Vec<f32> = raw[off..off + w].iter().map(|b| *b as i8 as f32).collect();
        off += w;
        let p: Vec<f32> = raw[off..off + 2 * w]
            .as_chunks::<2>().0.iter()
            .map(|c| (u16::from_le_bytes([c[0], c[1]]) as f64 / STORE_SCALE[1]) as f32)
            .collect();
        off += 2 * w;
        let rh: Vec<f32> = raw[off..off + w].iter().map(|b| *b as i8 as f32).collect();
        off += w;
        let present = unpack(&raw[off..off + mask_bytes(w)], w);
        off += mask_bytes(w);
        let valid = unpack(&raw[off..off + mask_bytes(w)], w);
        let raw_v: Vec<[f32; 3]> = (0..w).map(|h| [t[h], p[h], rh[h]]).collect();
        for h in 0..w {
            for c in 0..3 {
                if valid[h][c] && !present[h][c] {
                    return Err(Error::new("повреждённое окно: годный час без значения"));
                }
                let v = raw_v[h][c] as f64;
                if valid[h][c] && (v < bounds[c][0] || v > bounds[c][1]) {
                    return Err(Error::new(
                        "повреждённое окно: годное значение вне физического диапазона",
                    ));
                }
            }
        }
        Ok(Snapshot {
            filled,
            last_hour: if last == NO_HOUR { None } else { Some(last) },
            theta,
            site,
            raw: raw_v,
            present,
            valid,
        })
    }

    /// Запись в байты состояния.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&(self.filled as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&self.last_hour.unwrap_or(NO_HOUR).to_le_bytes());
        out.extend_from_slice(&self.theta.to_le_bytes());
        for v in self.site {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let q = |h: usize, c: usize| {
            if self.present[h][c] {
                (self.raw[h][c] as f64 * STORE_SCALE[c]).round_ties_even()
            } else {
                0.0
            }
        };
        let w = self.raw.len();
        out.extend((0..w).map(|h| q(h, 0) as i8 as u8));
        for h in 0..w {
            out.extend_from_slice(&(q(h, 1) as u16).to_le_bytes());
        }
        out.extend((0..w).map(|h| q(h, 2) as i8 as u8));
        pack(&self.present, out);
        pack(&self.valid, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_grid_keeps_out_of_range_out() {
        assert_eq!(to_store(-200.0, 0), -128.0);
        assert_eq!(to_store(7000.0, 1), 6553.5);
        assert_eq!(to_store(300.0, 2), 127.0);
        assert_eq!(to_store(-5.0, 2), -5.0);
        assert_eq!(to_store(1013.2, 1), 1013.2);
    }

    #[test]
    fn bits_roundtrip() {
        let m: Vec<[bool; 3]> = (0..11).map(|h| [h % 2 == 0, h % 3 == 0, h == 7]).collect();
        let mut out = Vec::new();
        pack(&m, &mut out);
        assert_eq!(out.len(), mask_bytes(11));
        assert_eq!(unpack(&out, 11), m);
    }
}
