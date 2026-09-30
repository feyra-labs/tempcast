//! Хост рантайма: модель считается графами ONNX, всё состояние живёт в буферах этой
//! структуры, выделенных один раз при создании.
//!
//! Хост держит сырое окно наблюдений, кольцо вкладов часов хвоста истории в моды, кольцо
//! строк суточного накопителя, буфер энкодера, причинный контроль качества, калибровку
//! интервалов и сериализацию. Место часа в каждом кольце определяется абсолютным часом,
//! поэтому указатель кольца хранить не нужно.
//!
//! Шаг часа - граф шага: сумма мод по хвосту истории сдвигается на час, вклад часа,
//! вышедшего из хвоста, вычитается. Раз в сутки сумма пересчитывается точно по кольцу.
//! Выпуск - граф выпуска: паспорт по строкам накопителя и вклад края истории, признаки
//! которого зависят от момента выпуска. Загрузка состояния и холодный старт - граф
//! полного окна.
//!
//! Момент выпуска - последний шаг. До первого шага его задаёт вызывающий по часам
//! устройства, и прогноз строится по пустому окну, которое этим часом кончается; окно
//! и счётчики при этом не меняются.
//!
//! Смена точки. Состояние помнит координаты и высоту, для которых оно записано. Сдвиг
//! в пределах порогов манифеста - уточнение метаданных: окно сохраняется и
//! пересчитывается для новой точки, множитель калибровки сохраняется. Больше порога -
//! прибор перенесён: окно пустое, множитель нулевой, момент последнего шага сохранён.
//! Контроль качества новых часов берёт текущую высоту.
//!
//! Откат. Граф старта возвращает таблицу климатологии точки: среднее и масштаб
//! климат-поля с паспортом холодного старта на каждый час года. Если выпуск не удался
//! или дал нечисловые квантили, прогноз - квантили нормального распределения с этими
//! средним и масштабом на часах горизонта.
use std::path::Path;

use crate::calendar::{doy_hour, hour_of_year, HOURS_OF_YEAR};
use crate::calib::{aci_score, apply_adaptive, apply_conformal, AciParams};
use crate::graphs::{Graphs, Precision};
use crate::manifest::{Dims, Manifest};
use crate::qc::CausalQc;
use crate::site::{describe_gap, site_change, SiteChange};
use crate::state::{to_store, Snapshot};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub precision: Precision,
    pub threads: usize,
    /// Применять конформную таблицу из манифеста.
    pub conformal: bool,
    /// Онлайн-подстройка множителя калибровки с параметрами из манифеста.
    pub aci: bool,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions {
            precision: Precision::Fp32,
            threads: 1,
            conformal: true,
            aci: false,
        }
    }
}

/// Последний выпуск: квантили по лидам подряд, медиана, признак отката и час, после
/// которого начинается горизонт.
#[derive(Debug, Clone)]
pub struct Forecast {
    pub q: Vec<f32>,
    pub mu: Vec<f32>,
    pub fallback: bool,
    pub after_hour: i64,
}

/// Сколько часов между точными пересчётами суммы мод.
pub const RESYNC_HOURS: i64 = 24;

pub struct Runtime {
    pub manifest: Manifest,
    d: Dims,
    graphs: Graphs,
    bounds: [[f64; 2]; 3],
    qc: CausalQc,
    lat: [f32; 1],
    lon: [f32; 1],
    site: [f32; 3],
    loc: Vec<f32>,
    coefs: [Vec<f32>; 3],
    // таблица климатологии точки для отката: по значению на каждый час года
    clim_mu: Vec<f32>,
    clim_sig: Vec<f32>,
    // сырое окно: место часа - абсолютный час по модулю длины окна
    raw: Vec<f32>,
    present: Vec<bool>,
    valid: Vec<bool>,
    filled: usize,
    last_hour: Option<i64>,
    // энкодер и сумма мод по хвосту: текущие буферы и буферы под выход графа
    enc: Vec<f32>,
    enc_next: Vec<f32>,
    modes: [Vec<f32>; 3],
    modes_next: [Vec<f32>; 3],
    // кольца вкладов хвоста и строк накопителя
    u_ring: Vec<f32>,
    v_ring: Vec<f32>,
    rows: Vec<f32>,
    // рабочие буферы
    ctx_x: Vec<f32>,
    ctx_m: Vec<f32>,
    win_x: Vec<f32>,
    win_m: Vec<f32>,
    win_doy: Vec<f32>,
    win_hour: Vec<f32>,
    edge_x: Vec<f32>,
    edge_m: Vec<f32>,
    edge_doy: Vec<f32>,
    edge_hour: Vec<f32>,
    u_tail: Vec<f32>,
    v_tail: Vec<f32>,
    rows_ord: Vec<f32>,
    u_old: Vec<f32>,
    v_old: [f32; 1],
    u_new: Vec<f32>,
    v_new: [f32; 1],
    row: [f32; 4],
    doy_fut: Vec<f32>,
    hour_fut: Vec<f32>,
    out: Forecast,
    // калибровка
    conformal: Option<Vec<f32>>,
    aci: Option<AciParams>,
    theta: f32,
    aci_updates: u64,
    aci_misses: u64,
    pending_first: i64,
    pending_q: Vec<f32>,
    pending_last: i64,
    pending: bool,
    // учёт
    idle_hours: u64,
    loaded_site: Option<[f32; 3]>,
    site_change: Option<SiteChange>,
    pub fallbacks: u64,
}

impl Runtime {
    pub fn new(model_dir: impl AsRef<Path>, lat: f64, lon: f64, elev: f64, opts: &RuntimeOptions) -> Result<Self> {
        let manifest = Manifest::load(model_dir)?;
        let d = manifest.dims.clone();
        let mut graphs = Graphs::open(&manifest, opts.precision, opts.threads.max(1))?;
        let conformal = if opts.conformal {
            let (table, why) = manifest.conformal_for(opts.precision)?;
            if let Some(why) = why {
                eprintln!("mayak-rt: {why}");
            }
            table
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
        let site = [lat as f32, lon as f32, elev as f32];
        let (lat, lon, elev) = ([lat as f32], [lon as f32], [elev as f32]);
        let mut loc = vec![0.0; d.loc_dim];
        let mut coefs = d.n_coef.map(|n| vec![0.0f32; n]);
        let mut z0 = vec![0.0; d.passport_dim];
        let mut clim_mu = vec![0.0; HOURS_OF_YEAR];
        let mut clim_sig = vec![0.0; HOURS_OF_YEAR];
        {
            let s11: &[usize] = &[1, 1];
            let [c0, c1, c2] = &mut coefs;
            graphs.init.run(
                &[("lat", s11, &lat), ("lon", s11, &lon), ("elev", s11, &elev)],
                &mut [&mut loc, c0, c1, c2, &mut z0, &mut clim_mu, &mut clim_sig],
            )?;
        }
        let good = clim_mu.iter().all(|v| v.is_finite()) && clim_sig.iter().all(|v| v.is_finite() && *v > 0.0);
        if !good {
            return Err(Error::new(
                "граф старта: таблица климатологии точки негодна (нечисловые значения или \
                 неположительный масштаб); это ошибка конфигурации модели или координат",
            ));
        }
        let (w, m, nq, h) = (d.stream_window, d.n_modes, d.n_quantiles, d.horizon);
        let (e, t, l) = (d.stream_edge, d.stream_tail, d.history);
        let buf = d.encoder_width * d.enc_buf_len;
        let mut rt = Runtime {
            bounds: manifest.phys_bounds(),
            qc,
            manifest,
            graphs,
            lat,
            lon,
            site,
            loc,
            coefs,
            clim_mu,
            clim_sig,
            raw: vec![0.0; w * 3],
            present: vec![false; w * 3],
            valid: vec![false; w * 3],
            filled: 0,
            last_hour: None,
            enc: vec![0.0; buf],
            enc_next: vec![0.0; buf],
            modes: [vec![0.0; m], vec![0.0; m], vec![0.0; m]],
            modes_next: [vec![0.0; m], vec![0.0; m], vec![0.0; m]],
            u_ring: vec![0.0; t * 2 * m],
            v_ring: vec![0.0; t],
            rows: vec![0.0; l * 4],
            ctx_x: vec![0.0; d.ctx * 3],
            ctx_m: vec![0.0; d.ctx * 3],
            win_x: vec![0.0; w * 3],
            win_m: vec![0.0; w * 3],
            win_doy: vec![0.0; w],
            win_hour: vec![0.0; w],
            edge_x: vec![0.0; e * 3],
            edge_m: vec![0.0; e * 3],
            edge_doy: vec![0.0; e],
            edge_hour: vec![0.0; e],
            u_tail: vec![0.0; t * 2 * m],
            v_tail: vec![0.0; t],
            rows_ord: vec![0.0; l * 4],
            u_old: vec![0.0; 2 * m],
            v_old: [0.0],
            u_new: vec![0.0; 2 * m],
            v_new: [0.0],
            row: [0.0; 4],
            doy_fut: vec![0.0; h],
            hour_fut: vec![0.0; h],
            out: Forecast {
                q: vec![0.0; h * nq],
                mu: vec![0.0; h],
                fallback: false,
                after_hour: 0,
            },
            conformal,
            aci,
            theta: 0.0,
            aci_updates: 0,
            aci_misses: 0,
            pending_first: 0,
            pending_q: vec![0.0; h * nq],
            pending_last: -1,
            pending: false,
            idle_hours: 0,
            loaded_site: None,
            site_change: None,
            fallbacks: 0,
            d,
        };
        rt.reset_calibration(0.0);
        rt.reset(None)?;
        Ok(rt)
    }

    /// Холодный старт: окно состоит из пустых часов. Множитель калибровки сохраняется.
    ///
    /// `last_hour` - час, которым кончается пустое окно; None - момент ещё не известен,
    /// окно строится при первом шаге.
    pub fn reset(&mut self, last_hour: Option<i64>) -> Result<()> {
        self.pending = false;
        self.qc.reset();
        self.raw.fill(0.0);
        self.present.fill(false);
        self.valid.fill(false);
        for v in [&mut self.enc, &mut self.u_ring, &mut self.v_ring, &mut self.rows] {
            v.fill(0.0);
        }
        self.modes.iter_mut().for_each(|v| v.fill(0.0));
        self.filled = 0;
        self.last_hour = last_hour;
        if let Some(last) = last_hour {
            self.rebuild_at(last, false)?;
        }
        Ok(())
    }

    /// Сброс адаптивной калибровки: множитель и счётчики обратной связи.
    pub fn reset_calibration(&mut self, theta: f32) {
        self.theta = match &self.aci {
            Some(a) => a.clip(theta as f64),
            None => theta,
        };
        self.aci_updates = 0;
        self.aci_misses = 0;
        self.pending = false;
    }

    /// Применяется ли конформная таблица к выпускам.
    pub fn conformal_applied(&self) -> bool {
        self.conformal.is_some()
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
    /// Сколько часов окна прошло после холодного старта, не больше длины окна.
    pub fn filled(&self) -> usize {
        self.filled
    }
    /// Абсолютный час последнего шага, в том числе из загруженного состояния.
    pub fn last_hour(&self) -> Option<i64> {
        self.last_hour
    }
    /// Сколько пустых часов подставлено за простой в этом процессе.
    pub fn idle_hours(&self) -> u64 {
        self.idle_hours
    }
    /// Координаты и высота из загруженного состояния.
    pub fn loaded_site(&self) -> Option<[f32; 3]> {
        self.loaded_site
    }
    /// Текущие координаты и высота так, как они пишутся в состояние.
    pub fn site(&self) -> [f32; 3] {
        self.site
    }
    /// Исход сравнения точки загруженного состояния с текущей; None - состояние не
    /// загружалось.
    pub fn site_change(&self) -> Option<SiteChange> {
        self.site_change
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
    /// Буфер энкодера в памяти, байт.
    pub fn encoder_buffer_bytes(&self) -> usize {
        4 * self.enc.len()
    }
    /// Кольца, буфер энкодера и таблица климатологии в памяти, байт. На диск они не
    /// пишутся: всё восстанавливается из окна и графа старта.
    pub fn memory_bytes(&self) -> usize {
        4 * (self.enc.len()
            + self.u_ring.len()
            + self.v_ring.len()
            + self.rows.len()
            + self.clim_mu.len()
            + self.clim_sig.len())
    }

    /// Момент выпуска: последний шаг, а до первого шага - текущий час устройства.
    pub fn issue_hour(&self, now_hour: Option<i64>) -> Option<i64> {
        self.last_hour.or(now_hour)
    }

    fn slot(&self, hour: i64) -> usize {
        hour.rem_euclid(self.d.stream_window as i64) as usize
    }

    /// Значения, маски годности и календарь часов окна от `first` подряд в буферы.
    fn window_into(&self, first: i64, x: &mut [f32], m: &mut [f32], doy: &mut [f32], hour: &mut [f32]) {
        for (k, (dd, hh)) in doy.iter_mut().zip(hour.iter_mut()).enumerate() {
            let h = first + k as i64;
            let p = self.slot(h);
            for c in 0..3 {
                let ok = self.valid[p * 3 + c];
                m[k * 3 + c] = ok as u8 as f32;
                x[k * 3 + c] = if ok { self.raw[p * 3 + c] } else { 0.0 };
            }
            (*dd, *hh) = doy_hour(h);
        }
    }

    /// Всё модельное состояние из сырого окна, которое кончается часом `last`, одним
    /// пакетным проходом.
    fn rebuild_at(&mut self, last: i64, seed_qc: bool) -> Result<()> {
        let (w, l, t, m) = (self.d.stream_window, self.d.history, self.d.stream_tail, self.d.n_modes);
        let first = last - w as i64 + 1;
        let (mut x, mut mk, mut dy, mut hr) = (
            std::mem::take(&mut self.win_x),
            std::mem::take(&mut self.win_m),
            std::mem::take(&mut self.win_doy),
            std::mem::take(&mut self.win_hour),
        );
        self.window_into(first, &mut x, &mut mk, &mut dy, &mut hr);
        (self.win_x, self.win_m, self.win_doy, self.win_hour) = (x, mk, dy, hr);
        let d = &self.d;
        let (s_w3, s_w, s11) = ([1, w, 3], [1, w], [1, 1]);
        let s_c = d.n_coef.map(|n| [1, n]);
        let [n_re, n_im, e] = &mut self.modes;
        self.graphs.window.run(
            &[
                ("x_win", &s_w3, &self.win_x),
                ("m_win", &s_w3, &self.win_m),
                ("doy_win", &s_w, &self.win_doy),
                ("hour_win", &s_w, &self.win_hour),
                ("lat", &s11, &self.lat),
                ("lon", &s11, &self.lon),
                ("c_mu", &s_c[0], &self.coefs[0]),
                ("c_sig", &s_c[1], &self.coefs[1]),
                ("c_def", &s_c[2], &self.coefs[2]),
            ],
            &mut [
                &mut self.enc,
                &mut self.u_tail,
                &mut self.v_tail,
                &mut self.rows_ord,
                n_re,
                n_im,
                e,
            ],
        )?;
        for k in 0..t {
            let s = (last - t as i64 + 1 + k as i64).rem_euclid(t as i64) as usize;
            self.u_ring[s * 2 * m..(s + 1) * 2 * m].copy_from_slice(&self.u_tail[k * 2 * m..(k + 1) * 2 * m]);
            self.v_ring[s] = self.v_tail[k];
        }
        for k in 0..l {
            let s = (last - l as i64 + 1 + k as i64).rem_euclid(l as i64) as usize;
            self.rows[s * 4..s * 4 + 4].copy_from_slice(&self.rows_ord[k * 4..k * 4 + 4]);
        }
        if seed_qc {
            let n = self.qc.size().min(w);
            let mut xs = Vec::with_capacity(n);
            let mut ps = Vec::with_capacity(n);
            for k in 0..n {
                let p = self.slot(last - n as i64 + 1 + k as i64);
                xs.push(std::array::from_fn(|c| self.raw[p * 3 + c]));
                ps.push(std::array::from_fn(|c| self.present[p * 3 + c]));
            }
            self.qc.seed(&xs, &ps);
        }
        Ok(())
    }

    /// Новый час наблюдений. None или NaN - значения нет. Пропущенные часы между прошлым
    /// шагом и этим заполняются пустыми, простой не короче окна - холодный старт.
    /// Возвращает коды причинного контроля качества этого часа.
    pub fn step(&mut self, obs: [Option<f64>; 3], hour: i64) -> Result<[u8; 3]> {
        match self.last_hour {
            None => self.reset(Some(hour - 1))?,
            Some(last) if hour <= last => {
                return Err(Error::new(format!("час {hour} не позже последнего шага {last}")));
            }
            Some(last) => {
                let gap = hour - last - 1;
                if gap >= self.d.stream_window as i64 {
                    eprintln!(
                        "mayak-rt: простой {gap} ч не короче окна {} ч: холодный старт",
                        self.d.stream_window
                    );
                    self.reset(Some(hour - 1))?;
                } else {
                    for h in (last + 1)..hour {
                        self.push([None, None, None], h)?;
                    }
                }
                self.idle_hours += gap as u64;
            }
        }
        self.push(obs, hour)
    }

    fn push(&mut self, obs: [Option<f64>; 3], hour: i64) -> Result<[u8; 3]> {
        let (x, codes) = self.qc.push(obs);
        let (raw, present) = self.qc.latest();
        if self.aci.is_some() && codes[0] == 0 {
            self.aci_feedback(x[0] as f64, hour);
        }
        let p = self.slot(hour);
        for c in 0..3 {
            self.raw[p * 3 + c] = if present[c] { to_store(raw[c], c) } else { 0.0 };
            self.present[p * 3 + c] = present[c];
            self.valid[p * 3 + c] = codes[c] == 0;
        }
        self.ingest(hour)?;
        Ok(codes)
    }

    /// Один час через граф шага: буфер энкодера, сумма мод, кольца.
    fn ingest(&mut self, hour: i64) -> Result<()> {
        self.last_hour = Some(hour);
        self.filled = (self.filled + 1).min(self.d.stream_window);
        let ctx = self.d.ctx;
        let (mut x, mut mk) = (std::mem::take(&mut self.ctx_x), std::mem::take(&mut self.ctx_m));
        let (mut dy, mut hr) = (vec![0.0; ctx], vec![0.0; ctx]);
        self.window_into(hour - ctx as i64 + 1, &mut x, &mut mk, &mut dy, &mut hr);
        (self.ctx_x, self.ctx_m) = (x, mk);
        let (t, m) = (self.d.stream_tail, self.d.n_modes);
        let slot = hour.rem_euclid(t.max(1) as i64) as usize;
        if t > 0 {
            self.u_old
                .copy_from_slice(&self.u_ring[slot * 2 * m..(slot + 1) * 2 * m]);
            self.v_old[0] = self.v_ring[slot];
        }
        let d = &self.d;
        let (doy, hr) = doy_hour(hour);
        let (doy, hr) = ([doy], [hr]);
        let s_ctx = [1, ctx, 3];
        let s11 = [1, 1];
        let s_c = d.n_coef.map(|n| [1, n]);
        let s_buf = [1, d.encoder_width, d.enc_buf_len];
        let s_m = [1, m];
        let s_u = [1, 2 * m];
        let [n_re, n_im, e] = &mut self.modes_next;
        self.graphs.step.run(
            &[
                ("x_ctx", &s_ctx, &self.ctx_x),
                ("m_ctx", &s_ctx, &self.ctx_m),
                ("doy", &s11, &doy),
                ("hour", &s11, &hr),
                ("lat", &s11, &self.lat),
                ("lon", &s11, &self.lon),
                ("c_mu", &s_c[0], &self.coefs[0]),
                ("c_sig", &s_c[1], &self.coefs[1]),
                ("c_def", &s_c[2], &self.coefs[2]),
                ("enc_buf", &s_buf, &self.enc),
                ("n_re", &s_m, &self.modes[0]),
                ("n_im", &s_m, &self.modes[1]),
                ("e", &s_m, &self.modes[2]),
                ("u_old", &s_u, &self.u_old),
                ("v_old", &s11, &self.v_old),
            ],
            &mut [
                &mut self.enc_next,
                n_re,
                n_im,
                e,
                &mut self.u_new,
                &mut self.v_new,
                &mut self.row,
            ],
        )?;
        std::mem::swap(&mut self.enc, &mut self.enc_next);
        std::mem::swap(&mut self.modes, &mut self.modes_next);
        if t > 0 {
            self.u_ring[slot * 2 * m..(slot + 1) * 2 * m].copy_from_slice(&self.u_new);
            self.v_ring[slot] = self.v_new[0];
        }
        let r = hour.rem_euclid(self.d.history as i64) as usize;
        self.rows[r * 4..r * 4 + 4].copy_from_slice(&self.row);
        if t > 0 && (hour + 1).rem_euclid(RESYNC_HOURS) == 0 {
            self.resync()?;
        }
        Ok(())
    }

    /// Точная сумма мод по кольцу вкладов хвоста вместо скользящей.
    pub fn resync(&mut self) -> Result<()> {
        let (t, m) = (self.d.stream_tail, self.d.n_modes);
        let Some(last) = self.last_hour else { return Ok(()) };
        if t == 0 {
            return Ok(());
        }
        for k in 0..t {
            let s = (last - t as i64 + 1 + k as i64).rem_euclid(t as i64) as usize;
            self.u_tail[k * 2 * m..(k + 1) * 2 * m].copy_from_slice(&self.u_ring[s * 2 * m..(s + 1) * 2 * m]);
            self.v_tail[k] = self.v_ring[s];
        }
        let (s_u, s_v) = ([1, t, 2 * m], [1, t]);
        let [n_re, n_im, e] = &mut self.modes;
        self.graphs.resync.run(
            &[("u_ring", &s_u, &self.u_tail), ("v_ring", &s_v, &self.v_tail)],
            &mut [n_re, n_im, e],
        )
    }

    fn aci_feedback(&mut self, y: f64, hour: i64) {
        let Some(aci) = self.aci else { return };
        if !self.pending {
            return;
        }
        let k = hour - self.pending_first;
        if k < 0 || k >= self.d.horizon as i64 || k <= self.pending_last {
            return;
        }
        self.pending_last = k;
        let (nq, k) = (self.d.n_quantiles, k as usize);
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

    /// Выпуск на часы после момента выпуска. `now_hour` - текущий час по часам
    /// устройства, нужен только до первого шага. Ошибка графа или нечисловой выход -
    /// Err; откат к климатологии делает выпуск с откатом.
    pub fn forecast(&mut self, now_hour: Option<i64>) -> Result<&Forecast> {
        let Some(last) = self.issue_hour(now_hour) else {
            return Err(Error::new(
                "нет ни одного шага и не задан текущий час: момент выпуска не определён",
            ));
        };
        if self.last_hour.is_none() {
            // окно пусто: модельное состояние пустого окна, которое кончается часом
            // выпуска; первый шаг всё равно начнёт с холодного старта
            self.rebuild_at(last, false)?;
        }
        let (l, e, h, nq) = (self.d.history, self.d.stream_edge, self.d.horizon, self.d.n_quantiles);
        for k in 0..l {
            let s = (last - l as i64 + 1 + k as i64).rem_euclid(l as i64) as usize;
            self.rows_ord[k * 4..k * 4 + 4].copy_from_slice(&self.rows[s * 4..s * 4 + 4]);
        }
        let (mut x, mut mk, mut dy, mut hr) = (
            std::mem::take(&mut self.edge_x),
            std::mem::take(&mut self.edge_m),
            std::mem::take(&mut self.edge_doy),
            std::mem::take(&mut self.edge_hour),
        );
        self.window_into(last - l as i64 + 1, &mut x, &mut mk, &mut dy, &mut hr);
        (self.edge_x, self.edge_m, self.edge_doy, self.edge_hour) = (x, mk, dy, hr);
        for k in 0..h {
            (self.doy_fut[k], self.hour_fut[k]) = doy_hour(last + 1 + k as i64);
        }
        let d = &self.d;
        let s11 = [1, 1];
        let s_loc = [1, d.loc_dim];
        let s_c = d.n_coef.map(|n| [1, n]);
        let s_rows = [1, l, 4];
        let s_m = [1, d.n_modes];
        let (s_e3, s_e) = ([1, e, 3], [1, e]);
        let s_h = [1, h];
        self.graphs.issue.run(
            &[
                ("loc", &s_loc, &self.loc),
                ("lat", &s11, &self.lat),
                ("lon", &s11, &self.lon),
                ("c_mu", &s_c[0], &self.coefs[0]),
                ("c_sig", &s_c[1], &self.coefs[1]),
                ("c_def", &s_c[2], &self.coefs[2]),
                ("rows", &s_rows, &self.rows_ord),
                ("n_re", &s_m, &self.modes[0]),
                ("n_im", &s_m, &self.modes[1]),
                ("e", &s_m, &self.modes[2]),
                ("x_edge", &s_e3, &self.edge_x),
                ("m_edge", &s_e3, &self.edge_m),
                ("doy_edge", &s_e, &self.edge_doy),
                ("hour_edge", &s_e, &self.edge_hour),
                ("doy_fut", &s_h, &self.doy_fut),
                ("hour_fut", &s_h, &self.hour_fut),
            ],
            &mut [&mut self.out.q],
        )?;
        if self.out.q.iter().any(|v| !v.is_finite()) {
            return Err(Error::new("выход графа issue не конечен"));
        }
        if let Some(t) = &self.conformal {
            apply_conformal(&mut self.out.q, t, nq, self.manifest.i_med);
        }
        if self.aci.is_some() {
            self.pending_first = last + 1;
            self.pending_q.copy_from_slice(&self.out.q);
            self.pending_last = -1;
            self.pending = true;
        }
        apply_adaptive(&mut self.out.q, self.theta, nq, self.manifest.i_med);
        for k in 0..h {
            self.out.mu[k] = self.out.q[k * nq + self.manifest.i_med];
        }
        self.out.fallback = false;
        self.out.after_hour = last;
        Ok(&self.out)
    }

    /// Откат: квантили нормального распределения по таблице климатологии точки на часах
    /// после `last`.
    pub fn climatology(&mut self, last: i64) -> &Forecast {
        let nq = self.d.n_quantiles;
        for k in 0..self.d.horizon {
            let i = hour_of_year(last + 1 + k as i64);
            let (mu, sig) = (self.clim_mu[i], self.clim_sig[i]);
            self.out.mu[k] = mu;
            for (j, z) in self.manifest.zq.iter().enumerate() {
                self.out.q[k * nq + j] = mu + z * sig;
            }
        }
        self.out.fallback = true;
        self.out.after_hour = last;
        &self.out
    }

    /// Выпуск с откатом: при любой ошибке выпуска - климатология точки; сбой пишется в
    /// лог и считается в `fallbacks`. Err только если момент выпуска не определён.
    pub fn safe_forecast(&mut self, now_hour: Option<i64>) -> Result<&Forecast> {
        let err = self.forecast(now_hour).err();
        if let Some(e) = err {
            let Some(last) = self.issue_hour(now_hour) else {
                return Err(e);
            };
            eprintln!("mayak-rt: откат к климатологии: {e}");
            self.fallbacks += 1;
            self.climatology(last);
        }
        Ok(&self.out)
    }

    /// Состояние в байтах: заголовок и сырое окно от старых часов к новым.
    pub fn serialize(&self, out: &mut Vec<u8>) {
        let w = self.d.stream_window;
        let first = self.last_hour.map(|l| l - w as i64 + 1);
        let pos = |k: usize| first.map_or(k, |f| self.slot(f + k as i64));
        let snap = Snapshot {
            filled: self.filled,
            last_hour: self.last_hour,
            theta: self.theta,
            site: self.site,
            raw: (0..w)
                .map(|k| std::array::from_fn(|c| self.raw[pos(k) * 3 + c]))
                .collect(),
            present: (0..w)
                .map(|k| std::array::from_fn(|c| self.present[pos(k) * 3 + c]))
                .collect(),
            valid: (0..w)
                .map(|k| std::array::from_fn(|c| self.valid[pos(k) * 3 + c]))
                .collect(),
        };
        snap.write(out);
    }

    /// Загрузка состояния и восстановление всего остального одним проходом по окну.
    /// Состояние другой точки: сдвиг в пределах порогов - окно пересчитывается для новой
    /// точки, больше порога - окно пустое, множитель калибровки нулевой, момент
    /// последнего шага сохраняется. При ошибке рантайм остаётся в холодном старте.
    pub fn load_state(&mut self, raw: &[u8]) -> Result<()> {
        let s = Snapshot::parse(raw, &self.d, &self.bounds)?;
        let lim = self.manifest.runtime;
        let (change, gap) = site_change(s.site, self.site, &lim);
        self.reset(None)?;
        self.loaded_site = Some(s.site);
        self.site_change = Some(change);
        if change == SiteChange::Moved {
            eprintln!(
                "mayak-rt: прибор перенесён: {}; холодный старт, множитель калибровки сброшен",
                describe_gap(gap, &lim)
            );
            self.reset_calibration(0.0);
            if let Err(e) = self.reset(s.last_hour) {
                self.reset(None)?;
                return Err(e);
            }
            return Ok(());
        }
        if change == SiteChange::Refined {
            eprintln!(
                "mayak-rt: координаты уточнены: {}; окно пересчитано для новой точки, множитель \
                 калибровки сохранён",
                describe_gap(gap, &lim)
            );
        }
        self.reset_calibration(s.theta);
        let Some(last) = s.last_hour else { return Ok(()) };
        let w = self.d.stream_window;
        self.last_hour = Some(last);
        for k in 0..w {
            let p = self.slot(last - w as i64 + 1 + k as i64);
            for c in 0..3 {
                self.raw[p * 3 + c] = s.raw[k][c];
                self.present[p * 3 + c] = s.present[k][c];
                self.valid[p * 3 + c] = s.valid[k][c];
            }
        }
        self.filled = s.filled;
        if let Err(e) = self.rebuild_at(last, true) {
            self.reset(None)?;
            return Err(e);
        }
        Ok(())
    }
}
