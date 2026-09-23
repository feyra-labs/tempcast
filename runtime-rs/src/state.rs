//! Персистентное состояние, формат v3 - байт в байт тот же, что пишет и читает
//! mayak.runtime.streaming (Python и Rust взаимозаменяемы на одном приборе).
//!
//! | поле          | тип          | число                         |
//! |---------------|--------------|-------------------------------|
//! | magic         | "MYK"        | 3 Б                           |
//! | version       | u8           | 3 (читается и 2 - без θ)      |
//! | filled        | u16          | заполнено часов окна          |
//! | hours_in_day  | u8           | часов в незавершённых сутках  |
//! | reserved      | u8           |                               |
//! | hoy_first     | u16          | час года первой позиции окна  |
//! | hoy_last      | u16          | час года последней позиции    |
//! | aci_theta     | f32          | только v3                     |
//! | n_re, n_im, e | f32          | 3·M                           |
//! | z             | f32          | dz                            |
//! | day_summ      | f16          | D·6                           |
//! | day_mask      | f16          | D                             |
//! | окно, значения| u16 (фикс. точка) | W·3                      |
//! | окно, маска   | u8           | W·3                           |
//!
//! Всё little-endian. Кольца энкодера и накопители текущих суток не пишутся: они
//! восстанавливаются одним проходом по окну (Runtime::load_state).
use half::f16;

use crate::calendar::YEAR_HOURS;
use crate::manifest::Dims;
use crate::{Error, Result};

pub const MAGIC: &[u8; 3] = b"MYK";
pub const VERSION: u8 = 3;
pub const HEADER_V2: usize = 12;
pub const HEADER_V3: usize = 16;
const QMAX: f64 = 65534.0;

pub fn nbytes(d: &Dims) -> usize {
    let (m, dd, w) = (d.n_modes, d.history_days, d.stream_window);
    HEADER_V3 + 4 * (3 * m + d.passport_dim) + 2 * (dd * d.n_daily_summary + dd) + 2 * w * 3 + w * 3
}

/// Фиксированная точка uint16 на канал (encode_raw): rint к чётному, как numpy.
pub fn encode_raw(x: f32, valid: bool, lo: f64, hi: f64) -> u16 {
    if !valid {
        return 0;
    }
    let step = (hi - lo) / QMAX;
    (((x as f64).clamp(lo, hi) - lo) / step).round_ties_even() as u16
}

pub fn decode_raw(q: u16, valid: bool, lo: f64, hi: f64) -> f32 {
    let step = (hi - lo) / QMAX;
    ((lo + q as f64 * step) * if valid { 1.0 } else { 0.0 }) as f32
}

/// Разобранное состояние (без восстановления эфемерной части).
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub version: u8,
    pub filled: usize,
    pub hours_in_day: usize,
    pub hoy_first: i64,
    pub hoy_last: i64,
    pub theta: f32,
    pub n_re: Vec<f32>,
    pub n_im: Vec<f32>,
    pub e: Vec<f32>,
    pub z: Vec<f32>,
    pub day_summ: Vec<f32>,
    pub day_mask: Vec<f32>,
    /// Окно в хронологическом порядке (старший час первым): значения и маска, W·3.
    pub raw_q: Vec<u16>,
    pub raw_m: Vec<u8>,
}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        s
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.bytes(4 * n)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn f16s(&mut self, n: usize) -> Vec<f32> {
        self.bytes(2 * n)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }
    fn u16s(&mut self, n: usize) -> Vec<u16> {
        self.bytes(2 * n)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

impl Snapshot {
    /// Разбор с теми же проверками, что StreamingMayak.load_state.
    pub fn parse(raw: &[u8], d: &Dims) -> Result<Self> {
        if raw.len() < 4 || &raw[..3] != MAGIC {
            return Err(Error::new("не состояние МАЯК формата v2+ (нет заголовка)"));
        }
        let version = raw[3];
        let hdr = match version {
            2 => HEADER_V2,
            VERSION => HEADER_V3,
            v => return Err(Error::new(format!("версия состояния {v}, рантайм читает [2, 3]"))),
        };
        let expected = nbytes(d) - HEADER_V3 + hdr;
        if raw.len() != expected {
            return Err(Error::new(format!(
                "состояние {} Б не соответствует конфигу модели (ожидалось {expected} Б для версии {version})",
                raw.len()
            )));
        }
        let u16at = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]);
        let filled = u16at(4) as usize;
        let hours_in_day = raw[6] as usize;
        let (hoy_first, hoy_last) = (u16at(8) as i64, u16at(10) as i64);
        let theta = if version == VERSION {
            f32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]])
        } else {
            0.0
        };
        if !theta.is_finite() {
            return Err(Error::new(format!("повреждённое θ калибровки в состоянии: {theta}")));
        }
        if filled > d.stream_window || hours_in_day > 23 || hours_in_day > filled {
            return Err(Error::new(format!(
                "повреждённый курсор состояния: filled={filled}, часов в сутках {hours_in_day}, окно {}",
                d.stream_window
            )));
        }
        window_hoys(filled, hoy_first, hoy_last)?;
        let (m, dd, w) = (d.n_modes, d.history_days, d.stream_window);
        let mut r = Reader { b: raw, off: hdr };
        Ok(Snapshot {
            version,
            filled,
            hours_in_day,
            hoy_first,
            hoy_last,
            theta,
            n_re: r.f32s(m),
            n_im: r.f32s(m),
            e: r.f32s(m),
            z: r.f32s(d.passport_dim),
            day_summ: r.f16s(dd * d.n_daily_summary),
            day_mask: r.f16s(dd),
            raw_q: r.u16s(w * 3),
            raw_m: r.bytes(w * 3).to_vec(),
        })
    }

    /// Запись в формате v3.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&(self.filled as u16).to_le_bytes());
        out.push(self.hours_in_day as u8);
        out.push(0);
        out.extend_from_slice(&(self.hoy_first.rem_euclid(65536) as u16).to_le_bytes());
        out.extend_from_slice(&(self.hoy_last.rem_euclid(65536) as u16).to_le_bytes());
        out.extend_from_slice(&self.theta.to_le_bytes());
        for v in self.n_re.iter().chain(&self.n_im).chain(&self.e).chain(&self.z) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.day_summ.iter().chain(&self.day_mask) {
            out.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        for v in &self.raw_q {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.raw_m);
    }
}

/// Часы года n последних позиций окна (StreamingMayak._window_calendar).
pub fn window_hoys(n: usize, hoy_first: i64, hoy_last: i64) -> Result<Vec<i64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let span = hoy_first + n as i64 - 1 - hoy_last;
    if span != 0 && !YEAR_HOURS.contains(&span) {
        return Err(Error::new(format!(
            "календарь окна несогласован: час года {hoy_first} … {hoy_last} на {n} ч"
        )));
    }
    Ok((0..n as i64)
        .map(|k| {
            let h = hoy_first + k;
            if span != 0 && h >= span {
                h - span
            } else {
                h
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_roundtrip_on_grid() {
        let (lo, hi) = (-90.0, 60.0);
        for x in [-90.0f32, -12.34, 0.0, 17.5, 60.0] {
            let q = encode_raw(x, true, lo, hi);
            assert!((decode_raw(q, true, lo, hi) - x).abs() <= ((hi - lo) / QMAX) as f32);
        }
        assert_eq!(encode_raw(99.0, false, lo, hi), 0);
        assert_eq!(decode_raw(123, false, lo, hi), 0.0);
    }

    #[test]
    fn window_calendar_wraps_new_year() {
        let h = window_hoys(4, 8758, 1).unwrap();
        assert_eq!(h, vec![8758, 8759, 0, 1]);
        assert!(window_hoys(4, 100, 50).is_err());
    }
}
