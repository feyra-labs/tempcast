//! Причинный контроль качества часа на устройстве.
//!
//! Коды часа решаются только по этому часу и по прошлым часам, не дальше глубины
//! `lookback_hours` из манифеста. Решение о часе принимается один раз, в момент его
//! прихода. Пороги приходят из манифеста модели, поэтому совпадают с обучением.
//! Совпадение правил с эталоном закрепляет тест на эталонных векторах кодов.

use serde::Deserialize;

pub const MISSING: u8 = 1;
pub const RANGE: u8 = 2;
pub const SPIKE: u8 = 4;
pub const JUMP: u8 = 16;
pub const STUCK: u8 = 32;
pub const UNITS: u8 = 64;

const MAD_TO_SD: f64 = 1.4826;
const MAD_EPS: f64 = 1e-6;
const P_SEA_LEVEL: f64 = 1013.25;

/// Пороги причинного QC. Каналы по порядку: температура, давление, влажность.
#[derive(Debug, Clone, Deserialize)]
pub struct QcConfig {
    pub spike_half: usize,
    pub spike_thresh: f64,
    pub spike_min_valid: usize,
    pub scale_floor: [f64; 3],
    pub jump_half: usize,
    pub jump_thresh: f64,
    pub jump_min_valid: usize,
    pub jump_floor: [f64; 3],
    pub jump_max_gap: usize,
    pub excursion_max_hours: usize,
    pub stuck_hours: [usize; 3],
    #[serde(rename = "stuck_T_alone_hours")]
    pub stuck_t_alone_hours: usize,
    pub stuck_max_gap: usize,
    pub stuck_min_count: usize,
    pub rh_sat: f64,
    pub rh_sat_hours: usize,
    pub units_half: usize,
    pub units_min_valid: usize,
    pub units_day_min_valid: usize,
    pub units_past_days: usize,
    pub units_min_days: usize,
    pub units_ref_k: f64,
    pub units_spread_floor: f64,
    pub units_min_excess: f64,
    pub units_min_conv: f64,
    pub slp_half: usize,
    pub slp_min_sep: f64,
    pub lookback_hours: usize,
}

/// Медиана непустого отсортированного среза.
fn median_sorted(s: &[f64]) -> f64 {
    let n = s.len();
    (s[(n - 1) / 2] + s[n / 2]) / 2.0
}

/// Медиана непустого набора; набор сортируется на месте.
fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.total_cmp(b));
    median_sorted(vals)
}

/// Медиана и медиана абсолютных отклонений непустого набора.
fn median_mad(vals: &mut [f64], dev: &mut Vec<f64>) -> (f64, f64) {
    let med = median(vals);
    dev.clear();
    dev.extend(vals.iter().map(|v| (v - med).abs()));
    dev.sort_by(|a, b| a.total_cmp(b));
    (med, median_sorted(dev))
}

/// Первый час окна, которое кончается часом `i` и берёт `before` часов до него.
fn window_start(i: usize, before: usize) -> usize {
    i.saturating_sub(before)
}

/// Выбросы одного канала относительно медианы прошлого окна.
fn spike_flags(x: &[f64], base: &[bool], cfg: &QcConfig, floor: f64) -> Vec<bool> {
    let n = x.len();
    let before = 2 * cfg.spike_half;
    let mut out = vec![false; n];
    let (mut vals, mut dev) = (Vec::new(), Vec::new());
    for i in 0..n {
        if !base[i] {
            continue;
        }
        vals.clear();
        for s in window_start(i, before)..=i {
            if base[s] {
                vals.push(x[s]);
            }
        }
        if vals.len() < cfg.spike_min_valid {
            continue;
        }
        let (med, mad) = median_mad(&mut vals, &mut dev);
        let scale = f64::max(MAD_TO_SD * (mad + MAD_EPS), floor);
        out[i] = (x[i] - med).abs() > cfg.spike_thresh * scale;
    }
    out
}

/// Второе изменение уровня возвращает ряд к уровню до первого.
fn cancels(da: f64, db: f64) -> bool {
    da * db < 0.0 && (da + db).abs() <= 0.5 * f64::min(da.abs(), db.abs())
}

/// Аномальные скачки между соседними отчётами одного канала.
///
/// Приращение делится на прошедшее время. Обратный скачок, возвращающий ряд к уровню
/// до недавнего скачка, не помечается.
fn jump_flags(x: &[f64], ok: &[bool], ch: usize, cfg: &QcConfig) -> Vec<bool> {
    let n = x.len();
    let mut rate = vec![0.0; n];
    let mut level = vec![0.0; n];
    let mut defined = vec![false; n];
    let mut prev: Option<usize> = None;
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        if let Some(p) = prev {
            let gap = i - p;
            if gap <= cfg.jump_max_gap {
                level[i] = x[i] - x[p];
                rate[i] = level[i] / gap as f64;
                defined[i] = true;
            }
        }
        prev = Some(i);
    }
    let before = 2 * cfg.jump_half;
    let mut cands = Vec::new();
    let (mut vals, mut dev) = (Vec::new(), Vec::new());
    for i in 0..n {
        if !defined[i] || rate[i].abs() <= cfg.jump_floor[ch] {
            continue;
        }
        vals.clear();
        for s in window_start(i, before)..=i {
            if defined[s] {
                vals.push(rate[s]);
            }
        }
        if vals.len() < cfg.jump_min_valid {
            continue;
        }
        let (med, mad) = median_mad(&mut vals, &mut dev);
        let scale = f64::max(MAD_TO_SD * (mad + MAD_EPS), cfg.scale_floor[ch]);
        if (rate[i] - med).abs() > cfg.jump_thresh * scale {
            cands.push(i);
        }
    }
    let mut out = vec![false; n];
    for (k, &b) in cands.iter().enumerate() {
        let mut keep = true;
        for &a in cands[..k].iter().rev() {
            if b - a > cfg.excursion_max_hours {
                break;
            }
            if cancels(level[a], level[b]) {
                keep = false;
                break;
            }
        }
        out[b] = keep;
    }
    out
}

/// Серия соседних отчётов от её начала до каждого отчёта: охват в часах и число отчётов.
fn run_lengths<F>(x: &[f64], ok: &[bool], max_gap: usize, link: F) -> (Vec<usize>, Vec<usize>)
where
    F: Fn(f64, f64) -> bool,
{
    let n = x.len();
    let mut span = vec![0; n];
    let mut count = vec![0; n];
    let mut prev: Option<usize> = None;
    let (mut first, mut cnt) = (0, 0);
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        let joined = prev.is_some_and(|p| i - p <= max_gap && link(x[p], x[i]));
        if joined {
            cnt += 1;
        } else {
            first = i;
            cnt = 1;
        }
        span[i] = i - first + 1;
        count[i] = cnt;
        prev = Some(i);
    }
    (span, count)
}

/// Опорный уровень и разброс по медианам прошлых суточных блоков.
fn past_reference(x: &[f64], ok: &[bool], cfg: &QcConfig) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut day = vec![f64::NAN; n];
    let mut day_ok = vec![false; n];
    let mut vals = Vec::new();
    for i in 0..n {
        vals.clear();
        for s in window_start(i, 23)..=i {
            if ok[s] {
                vals.push(x[s]);
            }
        }
        if !vals.is_empty() {
            day[i] = median(&mut vals);
        }
        day_ok[i] = vals.len() >= cfg.units_day_min_valid;
    }
    let mut reference = vec![f64::NAN; n];
    let mut spread = vec![cfg.units_spread_floor; n];
    let mut dev = Vec::new();
    for i in 0..n {
        vals.clear();
        for k in 1..=cfg.units_past_days {
            if i >= 24 * k && day_ok[i - 24 * k] {
                vals.push(day[i - 24 * k]);
            }
        }
        if vals.is_empty() {
            continue;
        }
        let count = vals.len();
        let (med, mad) = median_mad(&mut vals, &mut dev);
        spread[i] = f64::max(MAD_TO_SD * mad, cfg.units_spread_floor);
        if count >= cfg.units_min_days {
            reference[i] = med;
        }
    }
    (reference, spread)
}

/// Участки температуры в градусах Фаренгейта.
fn fahrenheit_flags(t: &[f64], src: &[bool], base: &[bool], cfg: &QcConfig) -> Vec<bool> {
    let n = t.len();
    let (reference, spread) = past_reference(t, base, cfg);
    let before = 2 * cfg.units_half;
    let mut out = vec![false; n];
    let mut vals = Vec::new();
    for i in 0..n {
        if !src[i] || !reference[i].is_finite() {
            continue;
        }
        vals.clear();
        for s in window_start(i, before)..=i {
            if src[s] {
                vals.push(t[s]);
            }
        }
        if vals.len() < cfg.units_min_valid {
            continue;
        }
        let r = median(&mut vals);
        let thr = f64::max(cfg.units_ref_k * spread[i], cfg.units_min_excess);
        let conv = (r - 32.0) / 1.8;
        let high = r - reference[i] > thr;
        let fits = (conv - reference[i]).abs() <= cfg.units_ref_k * spread[i] && conv >= cfg.units_min_conv;
        out[i] = high && fits;
    }
    out
}

/// Давление, приведённое к уровню моря, вместо станционного.
fn sea_level_flags(p: &[f64], src: &[bool], elev: Option<f64>, cfg: &QcConfig) -> Vec<bool> {
    let n = p.len();
    let mut out = vec![false; n];
    let Some(elev) = elev.filter(|e| e.is_finite()) else {
        return out;
    };
    let p_exp = P_SEA_LEVEL * (1.0 - elev / 44330.0).powf(5.255);
    let sep = P_SEA_LEVEL - p_exp;
    if sep < cfg.slp_min_sep {
        return out;
    }
    let thr = p_exp + sep / 2.0;
    let before = 2 * cfg.slp_half;
    let mut vals = Vec::new();
    for i in 0..n {
        if !src[i] {
            continue;
        }
        vals.clear();
        let mut above = 0;
        for s in window_start(i, before)..=i {
            if src[s] {
                vals.push(p[s]);
                if p[s] > thr {
                    above += 1;
                }
            }
        }
        if vals.len() < cfg.units_min_valid || 2 * above < vals.len() {
            continue;
        }
        out[i] = median(&mut vals) > thr;
    }
    out
}

/// Причинные коды всех часов ряда.
///
/// `x` - сырые значения от старых часов к новым, `present` - маска наличия, `elev` -
/// высота станции для проверки давления, `bounds` - физические диапазоны каналов.
/// Коды часа зависят только от этого часа и от `lookback_hours` часов перед ним.
pub fn causal_codes(
    x: &[[f32; 3]],
    present: &[[bool; 3]],
    elev: Option<f64>,
    bounds: &[[f64; 2]; 3],
    cfg: &QcConfig,
) -> Vec<[u8; 3]> {
    let n = x.len();
    let mut codes = vec![[0u8; 3]; n];
    let mut xs: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut src: [Vec<bool>; 3] = [vec![false; n], vec![false; n], vec![false; n]];
    let mut base: [Vec<bool>; 3] = [vec![false; n], vec![false; n], vec![false; n]];
    for i in 0..n {
        for c in 0..3 {
            let v = x[i][c];
            let has = present[i][c] && v.is_finite();
            let phys = (bounds[c][0] as f32..=bounds[c][1] as f32).contains(&v);
            xs[c][i] = v as f64;
            src[c][i] = has;
            base[c][i] = has && phys;
            if !has {
                codes[i][c] |= MISSING;
            } else if !phys {
                codes[i][c] |= RANGE;
            }
        }
    }
    for c in 0..3 {
        let spikes = spike_flags(&xs[c], &base[c], cfg, cfg.scale_floor[c]);
        let ok: Vec<bool> = (0..n).map(|i| base[c][i] && !spikes[i]).collect();
        let jumps = jump_flags(&xs[c], &ok, c, cfg);
        for i in 0..n {
            if spikes[i] {
                codes[i][c] |= SPIKE;
            }
            if jumps[i] {
                codes[i][c] |= JUMP;
            }
        }
    }
    let g = cfg.stuck_max_gap;
    let sat = cfg.rh_sat;
    let (span_t, cnt_t) = run_lengths(&xs[0], &base[0], g, |a, b| a == b);
    let (span_p, cnt_p) = run_lengths(&xs[1], &base[1], g, |a, b| a == b);
    let (span_r, cnt_r) = run_lengths(&xs[2], &base[2], g, |a, b| a == b && a < sat && b < sat);
    let (span_s, cnt_s) = run_lengths(&xs[2], &base[2], g, |a, b| a >= sat && b >= sat);
    let long = |span: usize, count: usize, hours: usize| span >= hours && count >= cfg.stuck_min_count;
    for i in 0..n {
        let rh_long = long(span_r[i], cnt_r[i], cfg.stuck_hours[0]);
        let t_alone = long(span_t[i], cnt_t[i], cfg.stuck_t_alone_hours);
        let t_joint = long(span_t[i], cnt_t[i], cfg.stuck_hours[0]) && rh_long;
        if t_alone || t_joint {
            codes[i][0] |= STUCK;
        }
        if long(span_p[i], cnt_p[i], cfg.stuck_hours[1]) {
            codes[i][1] |= STUCK;
        }
        let rh_stuck = long(span_r[i], cnt_r[i], cfg.stuck_hours[2]);
        if rh_stuck || long(span_s[i], cnt_s[i], cfg.rh_sat_hours) {
            codes[i][2] |= STUCK;
        }
    }
    let fahrenheit = fahrenheit_flags(&xs[0], &src[0], &base[0], cfg);
    let sea_level = sea_level_flags(&xs[1], &src[1], elev, cfg);
    for i in 0..n {
        if fahrenheit[i] {
            codes[i][0] |= UNITS;
        }
        if sea_level[i] {
            codes[i][1] |= UNITS;
        }
    }
    codes
}

/// Причинный QC потока: кольцо последних сырых часов и коды каждого нового часа.
#[derive(Debug, Clone)]
pub struct CausalQc {
    cfg: QcConfig,
    bounds: [[f64; 2]; 3],
    elev: Option<f64>,
    size: usize,
    x: Vec<[f32; 3]>,
    present: Vec<[bool; 3]>,
}

impl CausalQc {
    pub fn new(cfg: QcConfig, bounds: [[f64; 2]; 3], elev: Option<f64>) -> Self {
        let size = cfg.lookback_hours + 1;
        CausalQc {
            cfg,
            bounds,
            elev,
            size,
            x: Vec::with_capacity(size),
            present: Vec::with_capacity(size),
        }
    }

    /// Длина кольца, ч.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Кольцо пусто: прибор ничего не знает о прошлом.
    pub fn reset(&mut self) {
        self.x.clear();
        self.present.clear();
    }

    /// Заполняет кольцо прошлыми часами без расчёта кодов, от старых к новым.
    pub fn seed(&mut self, x: &[[f32; 3]], present: &[[bool; 3]]) {
        self.reset();
        let from = x.len().saturating_sub(self.size);
        for i in from..x.len() {
            let mut v = [0.0f32; 3];
            for c in 0..3 {
                if present[i][c] {
                    v[c] = x[i][c];
                }
            }
            self.x.push(v);
            self.present.push(present[i]);
        }
    }

    /// Новый час: значения с нулями на месте отбракованных и коды часа.
    pub fn push(&mut self, obs: [Option<f64>; 3]) -> ([f32; 3], [u8; 3]) {
        if self.x.len() == self.size {
            self.x.remove(0);
            self.present.remove(0);
        }
        let mut v = [0.0f32; 3];
        let mut p = [false; 3];
        for c in 0..3 {
            if let Some(o) = obs[c] {
                let f = o as f32;
                if f.is_finite() {
                    v[c] = f;
                    p[c] = true;
                }
            }
        }
        self.x.push(v);
        self.present.push(p);
        let codes = causal_codes(&self.x, &self.present, self.elev, &self.bounds, &self.cfg);
        let last = *codes.last().expect("в кольце есть новый час");
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            if last[c] == 0 {
                out[c] = v[c];
            }
        }
        (out, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_even_and_odd_sets() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn runs_grow_from_their_first_report() {
        let x = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let ok = [true, true, true, false, true, true];
        let (span, count) = run_lengths(&x, &ok, 6, |a, b| a == b);
        assert_eq!(span, vec![1, 2, 1, 0, 3, 4]);
        assert_eq!(count, vec![1, 2, 1, 0, 2, 3]);
    }

    #[test]
    fn return_to_previous_level_cancels() {
        assert!(cancels(20.0, -19.0));
        assert!(!cancels(20.0, -5.0));
        assert!(!cancels(20.0, 20.0));
    }
}
