//! Хост рантайма: то же поведение, что mayak.runtime.streaming.StreamingMayak, но
//! модель считается четырьмя графами ONNX, а всё состояние живёт в буферах этой
//! структуры, выделенных один раз при создании.
//!
//! Ответственность хоста): кольцо сырого окна и календарь, буфер энкодера
//! (двойной буфер - вход и выход графа step меняются местами), суточный накопитель,
//! причинный QC часа, калибровка интервалов, сериализация, откат к климатологии при сбое.
use std::path::Path;

use crate::calendar::{hour_of_year, YEAR_HOURS};
use crate::calib::{aci_score, apply_adaptive, apply_conformal, AciParams};
use crate::graphs::{Graphs, Precision};
use crate::manifest::{Dims, Manifest};
use crate::qc::CausalQc;
use crate::state::{decode_raw, encode_raw, window_hoys, Snapshot};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub precision: Precision,
    pub threads: usize,
    /// Применять конформную таблицу из манифеста.
    pub conformal: bool,
    /// Онлайн-подстройка θ (ACI) с параметрами из манифеста.
    pub aci: bool,
    /// Откат к климатологии: средняя T и σ (как --clim-fallback / --sigma-fallback).
    pub clim_fallback: f32,
    pub sigma_fallback: f32,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions {
            precision: Precision::Fp32,
            threads: 1,
            conformal: true,
            aci: false,
            clim_fallback: 10.0,
            sigma_fallback: 4.0,
        }
    }
}

/// Последний выпуск: квантили (horizon × NQ), медиана и признак отката.
#[derive(Debug, Clone)]
pub struct Forecast {
    pub q: Vec<f32>,
    pub mu: Vec<f32>,
    pub fallback: bool,
}

pub struct Runtime {
    pub manifest: Manifest,
    d: Dims,
    graphs: Graphs,
    bounds: [[f64; 2]; 3],
    qc: CausalQc,
    lat: [f32; 1],
    lon: [f32; 1],
    // признаки точки (граф init)
    loc: Vec<f32>,
    coefs: [Vec<f32>; 3],
    z0: Vec<f32>,
    // сырое окно (кольцо)
    raw_x: Vec<f32>,
    raw_m: Vec<f32>,
    hoy_ring: Vec<i64>,
    head: usize,
    filled: usize,
    // энкодер и моды: текущие буферы и буферы под выход графа
    enc: Vec<f32>,
    enc_next: Vec<f32>,
    modes: [Vec<f32>; 3],
    modes_next: [Vec<f32>; 3],
    // сутки и паспорт
    day_acc: Vec<f32>,
    hours_in_day: usize,
    day_summ: Vec<f32>,
    day_mask: Vec<f32>,
    z: Vec<f32>,
    passport_next: [Vec<f32>; 3],
    // рабочие буферы
    ctx_x: Vec<f32>,
    ctx_m: Vec<f32>,
    day_row: [f32; 4],
    doy_fut: Vec<f32>,
    hour_fut: Vec<f32>,
    out: Forecast,
    // календарь
    last_hoy: Option<i64>,
    calendar_breaks: u64,
    // калибровка
    conformal: Option<Vec<f32>>,
    aci: Option<AciParams>,
    theta: f32,
    aci_updates: u64,
    aci_misses: u64,
    pending_hoy: Vec<i64>,
    pending_q: Vec<f32>,
    pending_last: i64,
    pending: bool,
    // откат
    clim: (f32, f32),
    pub fallbacks: u64,
}

impl Runtime {
    pub fn new(model_dir: impl AsRef<Path>, lat: f64, lon: f64, elev: f64, opts: &RuntimeOptions) -> Result<Self> {
        let manifest = Manifest::load(model_dir)?;
        let d = manifest.dims.clone();
        let mut graphs = Graphs::open(&manifest, opts.precision, opts.threads.max(1))?;
        let conformal = if opts.conformal {
            manifest.conformal_table()?
        } else {
            None
        };
        let aci = match (&manifest.calibration.aci, opts.aci) {
            (Some(a), true) => Some(AciParams {
                target: a.target,
                gamma: a.gamma,
                max_factor: a.max_factor,
                interval: a.interval,
            }),
            (None, true) => return Err(Error::new("ACI включена, но параметров ACI в манифесте нет")),
            _ => None,
        };
        let qc = CausalQc::new(manifest.qc.clone(), manifest.phys_bounds(), Some(elev));
        let (lat, lon, elev) = ([lat as f32], [lon as f32], [elev as f32]);
        let mut loc = vec![0.0; d.loc_dim];
        let mut coefs = d.n_coef.map(|n| vec![0.0f32; n]);
        let mut z0 = vec![0.0; d.passport_dim];
        {
            let s11: &[usize] = &[1, 1];
            let [c0, c1, c2] = &mut coefs;
            graphs.init.run(
                &[("lat", s11, &lat), ("lon", s11, &lon), ("elev", s11, &elev)],
                &mut [&mut loc, c0, c1, c2, &mut z0],
            )?;
        }
        let (w, m, nq, h) = (d.stream_window, d.n_modes, d.n_quantiles, d.horizon);
        let buf = d.encoder_width * d.enc_buf_len;
        let mut rt = Runtime {
            bounds: manifest.phys_bounds(),
            qc,
            manifest,
            graphs,
            lat,
            lon,
            loc,
            coefs,
            z: z0.clone(),
            z0,
            raw_x: vec![0.0; w * 3],
            raw_m: vec![0.0; w * 3],
            hoy_ring: vec![0; w],
            head: 0,
            filled: 0,
            enc: vec![0.0; buf],
            enc_next: vec![0.0; buf],
            modes: [vec![0.0; m], vec![0.0; m], vec![0.0; m]],
            modes_next: [vec![0.0; m], vec![0.0; m], vec![0.0; m]],
            day_acc: vec![0.0; 4 * 24],
            hours_in_day: 0,
            day_summ: vec![0.0; d.history_days * d.n_daily_summary],
            day_mask: vec![0.0; d.history_days],
            passport_next: [
                vec![0.0; d.history_days * d.n_daily_summary],
                vec![0.0; d.history_days],
                vec![0.0; d.passport_dim],
            ],
            ctx_x: vec![0.0; d.ctx * 3],
            ctx_m: vec![0.0; d.ctx * 3],
            day_row: [0.0; 4],
            doy_fut: vec![0.0; h],
            hour_fut: vec![0.0; h],
            out: Forecast {
                q: vec![0.0; h * nq],
                mu: vec![0.0; h],
                fallback: false,
            },
            last_hoy: None,
            calendar_breaks: 0,
            conformal,
            aci,
            theta: 0.0,
            aci_updates: 0,
            aci_misses: 0,
            pending_hoy: vec![0; h],
            pending_q: vec![0.0; h * nq],
            pending_last: -1,
            pending: false,
            clim: (opts.clim_fallback, opts.sigma_fallback),
            fallbacks: 0,
            d,
        };
        rt.reset_calibration(0.0);
        rt.reset();
        Ok(rt)
    }

    /// Холодный старт: история пуста. θ калибровки сохраняется (он относится к прибору).
    pub fn reset(&mut self) {
        self.pending = false;
        self.qc.reset();
        for v in [
            &mut self.raw_x,
            &mut self.raw_m,
            &mut self.enc,
            &mut self.day_acc,
            &mut self.day_summ,
            &mut self.day_mask,
        ] {
            v.fill(0.0);
        }
        self.modes.iter_mut().for_each(|v| v.fill(0.0));
        self.hoy_ring.fill(0);
        self.head = 0;
        self.filled = 0;
        self.hours_in_day = 0;
        self.last_hoy = None;
        self.calendar_breaks = 0;
        self.z.copy_from_slice(&self.z0);
    }

    /// Сброс адаптивной калибровки (θ и счётчики обратной связи).
    pub fn reset_calibration(&mut self, theta: f32) {
        self.theta = match &self.aci {
            Some(a) => a.clip(theta as f64),
            None => theta,
        };
        self.aci_updates = 0;
        self.aci_misses = 0;
        self.pending = false;
    }

    pub fn theta(&self) -> f32 {
        self.theta
    }
    pub fn aci_updates(&self) -> u64 {
        self.aci_updates
    }
    pub fn aci_misses(&self) -> u64 {
        self.aci_misses
    }
    pub fn calendar_breaks(&self) -> u64 {
        self.calendar_breaks
    }
    pub fn filled(&self) -> usize {
        self.filled
    }
    /// Час года последнего шага (из потока или из загруженного состояния).
    pub fn last_hoy(&self) -> Option<i64> {
        self.last_hoy
    }
    pub fn hours_in_day(&self) -> usize {
        self.hours_in_day
    }
    pub fn stream_window(&self) -> usize {
        self.d.stream_window
    }
    pub fn horizon(&self) -> usize {
        self.d.horizon
    }
    pub fn n_quantiles(&self) -> usize {
        self.d.n_quantiles
    }
    pub fn state_nbytes(&self) -> usize {
        crate::state::nbytes(&self.d)
    }
    /// Буфер энкодера в памяти (на диск не пишется), байт.
    pub fn encoder_buffer_bytes(&self) -> usize {
        4 * self.enc.len()
    }

    /// Новый час наблюдений. None или NaN - значения нет. Час проходит причинный QC по
    /// кольцу прошлых сырых часов; отбракованное значение становится пропуском.
    pub fn step(&mut self, obs: [Option<f64>; 3], doy: f32, hour: f32) -> Result<()> {
        let (x, codes) = self.qc.push(obs);
        let m = codes.map(|c| if c == 0 { 1.0 } else { 0.0 });
        if self.aci.is_some() && m[0] > 0.0 {
            self.aci_feedback(x[0] as f64, doy);
        }
        self.ingest(x, m, doy, hour)
    }

    fn check_calendar(&mut self, hoy: i64) {
        if let Some(prev) = self.last_hoy {
            if hoy != prev + 1 && !(hoy == 0 && YEAR_HOURS.contains(&(prev + 1))) {
                self.calendar_breaks += 1;
                eprintln!("mayak-rt: календарь потока не непрерывен: час года {hoy} после {prev}");
            }
        }
        self.last_hoy = Some(hoy);
    }

    fn ingest(&mut self, x: [f32; 3], m: [f32; 3], doy: f32, hour: f32) -> Result<()> {
        let hoy = hour_of_year(doy);
        self.check_calendar(hoy);
        let (j, w) = (self.head, self.d.stream_window);
        for c in 0..3 {
            self.raw_m[j * 3 + c] = m[c];
            self.raw_x[j * 3 + c] = if m[c] > 0.0 { x[c] } else { 0.0 };
        }
        self.hoy_ring[j] = hoy;
        self.head = (j + 1) % w;
        self.filled = (self.filled + 1).min(w);
        self.run_step(j, None, doy, hour)?;
        std::mem::swap(&mut self.enc, &mut self.enc_next);
        std::mem::swap(&mut self.modes, &mut self.modes_next);
        self.accumulate_day()
    }

    /// Граф step для часа в позиции `end` кольца. `lo` - нижняя граница окна при
    /// восстановлении: позиции левее неё читаются как пустые, а не по кольцу.
    fn run_step(&mut self, end: usize, lo: Option<usize>, doy: f32, hour: f32) -> Result<()> {
        let (w, ctx) = (self.d.stream_window as i64, self.d.ctx);
        for i in 0..ctx {
            let p = end as i64 - (ctx as i64 - 1) + i as i64;
            let pos = match lo {
                Some(lo) if p < lo as i64 => None,
                _ => Some(p.rem_euclid(w) as usize),
            };
            for c in 0..3 {
                let (xv, mv) = pos.map_or((0.0, 0.0), |p| (self.raw_x[p * 3 + c], self.raw_m[p * 3 + c]));
                self.ctx_x[i * 3 + c] = xv;
                self.ctx_m[i * 3 + c] = mv;
            }
        }
        let d = &self.d;
        let s_ctx = [1, ctx, 3];
        let s11 = [1, 1];
        let s_c = d.n_coef.map(|n| [1, n]);
        let s_buf = [1, d.encoder_width, d.enc_buf_len];
        let s_m = [1, d.n_modes];
        let (doy, hour) = ([doy], [hour]);
        let [n_re, n_im, e] = &mut self.modes_next;
        self.graphs.step.run(
            &[
                ("x_ctx", &s_ctx, &self.ctx_x),
                ("m_ctx", &s_ctx, &self.ctx_m),
                ("doy", &s11, &doy),
                ("hour", &s11, &hour),
                ("lat", &s11, &self.lat),
                ("lon", &s11, &self.lon),
                ("c_mu", &s_c[0], &self.coefs[0]),
                ("c_sig", &s_c[1], &self.coefs[1]),
                ("c_def", &s_c[2], &self.coefs[2]),
                ("enc_buf", &s_buf, &self.enc),
                ("n_re", &s_m, &self.modes[0]),
                ("n_im", &s_m, &self.modes[1]),
                ("e", &s_m, &self.modes[2]),
            ],
            &mut [&mut self.enc_next, n_re, n_im, e, &mut self.day_row],
        )
    }

    fn accumulate_day(&mut self) -> Result<()> {
        for r in 0..4 {
            self.day_acc[r * 24 + self.hours_in_day] = self.day_row[r];
        }
        self.hours_in_day += 1;
        if self.hours_in_day < 24 {
            return Ok(());
        }
        let d = &self.d;
        let s_loc = [1, d.loc_dim];
        let s_acc = [1, 4, 24];
        let s_summ = [1, d.history_days, d.n_daily_summary];
        let s_mask = [1, d.history_days];
        let [ds, dm, z] = &mut self.passport_next;
        self.graphs.passport.run(
            &[
                ("loc", &s_loc, &self.loc),
                ("day_acc", &s_acc, &self.day_acc),
                ("day_summ", &s_summ, &self.day_summ),
                ("day_mask", &s_mask, &self.day_mask),
            ],
            &mut [ds, dm, z],
        )?;
        std::mem::swap(&mut self.day_summ, &mut self.passport_next[0]);
        std::mem::swap(&mut self.day_mask, &mut self.passport_next[1]);
        std::mem::swap(&mut self.z, &mut self.passport_next[2]);
        self.day_acc.fill(0.0);
        self.hours_in_day = 0;
        Ok(())
    }

    fn aci_feedback(&mut self, y: f64, doy: f32) {
        let Some(aci) = self.aci else { return };
        if !self.pending {
            return;
        }
        let hoy = hour_of_year(doy);
        let Some(k) = self.pending_hoy.iter().position(|&h| h == hoy) else {
            return;
        };
        if (k as i64) <= self.pending_last {
            return;
        }
        self.pending_last = k as i64;
        let nq = self.d.n_quantiles;
        let score = aci_score(
            y,
            &self.pending_q[k * nq..(k + 1) * nq],
            aci.interval,
            self.manifest.i_med,
        );
        let (theta, miss) = aci.step(self.theta, score);
        self.theta = theta;
        self.aci_updates += 1;
        self.aci_misses += miss as u64;
    }

    /// Выпуск прогноза на календарь горизонта. Ошибка графа или нечисловой выход -
    /// Err; откат к климатологии делает `safe_forecast`.
    pub fn forecast(&mut self, doy_fut: &[f32], hour_fut: &[f32]) -> Result<&Forecast> {
        let d = &self.d;
        let (h, nq) = (d.horizon, d.n_quantiles);
        if doy_fut.len() != h || hour_fut.len() != h {
            return Err(Error::new(format!(
                "календарь горизонта длины {}, нужно {h}",
                doy_fut.len()
            )));
        }
        self.doy_fut.copy_from_slice(doy_fut);
        self.hour_fut.copy_from_slice(hour_fut);
        let (s_loc, s11, s_z, s_m, s_h) = ([1, d.loc_dim], [1, 1], [1, d.passport_dim], [1, d.n_modes], [1, h]);
        self.graphs.issue.run(
            &[
                ("loc", &s_loc, &self.loc),
                ("lat", &s11, &self.lat),
                ("lon", &s11, &self.lon),
                ("z", &s_z, &self.z),
                ("n_re", &s_m, &self.modes[0]),
                ("n_im", &s_m, &self.modes[1]),
                ("e", &s_m, &self.modes[2]),
                ("doy_fut", &s_h, &self.doy_fut),
                ("hour_fut", &s_h, &self.hour_fut),
            ],
            &mut [&mut self.out.q],
        )?;
        if self.out.q.iter().any(|v| !v.is_finite()) {
            return Err(Error::new("выход графа issue не конечен"));
        }
        if let Some(t) = &self.conformal {
            apply_conformal(&mut self.out.q, t, nq);
        }
        if self.aci.is_some() {
            for (k, dv) in self.doy_fut.iter().enumerate() {
                self.pending_hoy[k] = (*dv as f64 * 24.0).round_ties_even() as i64;
            }
            self.pending_q.copy_from_slice(&self.out.q);
            self.pending_last = -1;
            self.pending = true;
        }
        apply_adaptive(&mut self.out.q, self.theta, nq, self.manifest.i_med);
        for k in 0..h {
            self.out.mu[k] = self.out.q[k * nq + self.manifest.i_med];
        }
        self.out.fallback = false;
        Ok(&self.out)
    }

    /// Выпуск со сторожевым откатом (safe_forecast): при любой ошибке - климатология
    /// clim ± zq·σ; сбой пишется в лог и считается в `fallbacks`.
    pub fn safe_forecast(&mut self, doy_fut: &[f32], hour_fut: &[f32]) -> &Forecast {
        if let Err(e) = self.forecast(doy_fut, hour_fut) {
            eprintln!("mayak-rt: откат к климатологии: {e}");
            self.fallbacks += 1;
            let (mu, sig) = self.clim;
            let nq = self.d.n_quantiles;
            for k in 0..self.d.horizon {
                self.out.mu[k] = mu;
                for (i, z) in self.manifest.zq.iter().enumerate() {
                    self.out.q[k * nq + i] = mu + z * sig;
                }
            }
            self.out.fallback = true;
        }
        &self.out
    }

    /// Состояние в формате v3.
    pub fn serialize(&self, out: &mut Vec<u8>) {
        let (w, n) = (self.d.stream_window, self.filled);
        let order = (0..w).map(|k| (self.head + k) % w);
        let mut raw_q = Vec::with_capacity(w * 3);
        let mut raw_m = Vec::with_capacity(w * 3);
        for p in order {
            for c in 0..3 {
                let valid = self.raw_m[p * 3 + c] > 0.0;
                let [lo, hi] = self.bounds[c];
                raw_q.push(encode_raw(self.raw_x[p * 3 + c], valid, lo, hi));
                raw_m.push(valid as u8);
            }
        }
        let at = |back: usize| self.hoy_ring[(self.head + w - back) % w];
        let snap = Snapshot {
            version: crate::state::VERSION,
            filled: n,
            hours_in_day: self.hours_in_day,
            hoy_first: if n > 0 { at(n) } else { 0 },
            hoy_last: if n > 0 { at(1) } else { 0 },
            theta: self.theta,
            n_re: self.modes[0].clone(),
            n_im: self.modes[1].clone(),
            e: self.modes[2].clone(),
            z: self.z.clone(),
            day_summ: self.day_summ.clone(),
            day_mask: self.day_mask.clone(),
            raw_q,
            raw_m,
        };
        snap.write(out);
    }

    /// Загрузка состояния и восстановление эфемерной части: буфер энкодера и
    /// незавершённые сутки - прогоном графа step по сохранённому окну (моды не трогаются).
    /// При ошибке рантайм остаётся в состоянии холодного старта.
    pub fn load_state(&mut self, raw: &[u8]) -> Result<()> {
        let s = Snapshot::parse(raw, &self.d)?;
        let hoys = window_hoys(s.filled, s.hoy_first, s.hoy_last)?;
        self.reset();
        let (w, n) = (self.d.stream_window, s.filled);
        self.modes[0].copy_from_slice(&s.n_re);
        self.modes[1].copy_from_slice(&s.n_im);
        self.modes[2].copy_from_slice(&s.e);
        self.z.copy_from_slice(&s.z);
        self.day_summ.copy_from_slice(&s.day_summ);
        self.day_mask.copy_from_slice(&s.day_mask);
        for p in 0..w {
            for c in 0..3 {
                let valid = s.raw_m[p * 3 + c] > 0;
                let [lo, hi] = self.bounds[c];
                self.raw_m[p * 3 + c] = valid as u8 as f32;
                self.raw_x[p * 3 + c] = decode_raw(s.raw_q[p * 3 + c], valid, lo, hi);
            }
        }
        for (k, h) in hoys.iter().enumerate() {
            self.hoy_ring[w - n + k] = *h;
        }
        self.head = 0;
        self.filled = n;
        self.seed_qc();
        self.hours_in_day = s.hours_in_day;
        self.last_hoy = if n > 0 { Some(s.hoy_last) } else { None };
        self.reset_calibration(s.theta);
        if let Err(e) = self.rebuild_from_window() {
            self.reset();
            return Err(e);
        }
        Ok(())
    }

    /// Кольцо QC после загрузки: значения окна, уже прошедшие QC, от старых к новым.
    /// Отбракованные часы в нём становятся пропусками.
    fn seed_qc(&mut self) {
        let (w, n) = (self.d.stream_window, self.filled);
        let mut x = Vec::with_capacity(n);
        let mut present = Vec::with_capacity(n);
        for p in (w - n)..w {
            let mut v = [0.0f32; 3];
            let mut m = [false; 3];
            for c in 0..3 {
                v[c] = self.raw_x[p * 3 + c];
                m[c] = self.raw_m[p * 3 + c] > 0.0;
            }
            x.push(v);
            present.push(m);
        }
        self.qc.seed(&x, &present);
    }

    fn rebuild_from_window(&mut self) -> Result<()> {
        let (w, n, hid) = (self.d.stream_window, self.filled, self.hours_in_day);
        for k in (w - n)..w {
            let hoy = self.hoy_ring[k];
            let doy = (hoy as f64 / 24.0) as f32;
            let hour = hoy.rem_euclid(24) as f32;
            self.run_step(k, Some(w - n), doy, hour)?;
            std::mem::swap(&mut self.enc, &mut self.enc_next);
            if k >= w - hid {
                let col = k - (w - hid);
                for r in 0..4 {
                    self.day_acc[r * 24 + col] = self.day_row[r];
                }
            }
        }
        Ok(())
    }
}
