//! Калибровка интервалов - зеркало mayak.metrics: сплит-конформная таблица,
//! адаптивный множитель ширины интервала и его онлайн-подстройка по промахам.
//!
//! Единственная реализация в проекте - Python (`calibrate_forecast`, `ACIParams`,
//! `aci_score`). Здесь - её перенос, закреплённый эталонными векторами: расхождение
//! двух реализаций ловится тестом `golden.rs::calibration`, а не ревью.
//! Разворачивание таблицы по бинам лидов остаётся в Python (conformal.f32 уже по лидам).

#[derive(Debug, Clone, Copy)]
pub struct AciParams {
    pub target: f64,
    pub gamma: f64,
    pub max_factor: f64,
    pub interval: [usize; 2],
}

impl AciParams {
    pub fn theta_min(&self) -> f32 {
        (-self.max_factor.ln()) as f32
    }

    pub fn theta_max(&self) -> f32 {
        self.max_factor.ln() as f32
    }

    /// Округление до f32 и упор в границы (ACIParams.clip).
    pub fn clip(&self, theta: f64) -> f32 {
        (theta as f32).max(self.theta_min()).min(self.theta_max())
    }

    /// θ + γ·(err − α) с упором в границы (ACIParams.update).
    pub fn update(&self, theta: f32, miss: bool) -> f32 {
        self.clip(theta as f64 + self.gamma * (miss as u8 as f64 - self.target))
    }

    /// Одна обратная связь по оценке aci_score → (новое θ, промах).
    pub fn step(&self, theta: f32, score: f64) -> (f32, bool) {
        let miss = score > (theta as f64).exp();
        (self.update(theta, miss), miss)
    }
}

/// Сдвиг квантилей таблицей (horizon × NQ) и восстановление монотонности.
pub fn apply_conformal(q: &mut [f32], table: &[f32], nq: usize) {
    for (row, t) in q.chunks_exact_mut(nq).zip(table.chunks_exact(nq)) {
        for (v, s) in row.iter_mut().zip(t) {
            *v += *s;
        }
        cummax(row);
    }
}

/// Растяжение квантилей вокруг медианы в e^θ раз; при θ = 0 - без арифметики.
pub fn apply_adaptive(q: &mut [f32], theta: f32, nq: usize, i_med: usize) {
    if theta == 0.0 {
        return;
    }
    let k = (theta as f64).exp() as f32;
    for row in q.chunks_exact_mut(nq) {
        let med = row[i_med];
        for v in row.iter_mut() {
            *v = med + k * (*v - med);
        }
        cummax(row);
    }
}

fn cummax(row: &mut [f32]) {
    // np.maximum.accumulate: NaN распространяется вправо.
    for i in 1..row.len() {
        let (a, b) = (row[i - 1], row[i]);
        row[i] = if a.is_nan() || b.is_nan() {
            f32::NAN
        } else if b < a {
            a
        } else {
            b
        };
    }
}

/// Нормированный выход факта за интервал (mayak.metrics.aci_score) для одной строки q.
pub fn aci_score(y: f64, q: &[f32], interval: [usize; 2], i_med: usize) -> f64 {
    let med = q[i_med] as f64;
    if !y.is_finite() || !med.is_finite() {
        return f64::NAN;
    }
    let u = y - med;
    if u == 0.0 {
        return 0.0;
    }
    let d = if u >= 0.0 {
        q[interval[1]] as f64 - med
    } else {
        med - q[interval[0]] as f64
    };
    if d > 0.0 {
        u.abs() / d
    } else {
        f64::INFINITY
    }
}
