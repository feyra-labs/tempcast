//! Калибровка интервалов на устройстве: сплит-конформная таблица, адаптивный множитель
//! ширины интервала и его онлайн-подстройка по промахам.
//!
//! Эталонная реализация калибровки написана на Python; здесь её перенос. Совпадение
//! двух реализаций проверяют эталонные векторы, а не ревью. Таблица приходит уже
//! развёрнутой по лидам, поэтому бинов лидов здесь нет.

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

    /// Округление логарифма множителя до f32 и упор в границы.
    pub fn clip(&self, theta: f64) -> f32 {
        (theta as f32).max(self.theta_min()).min(self.theta_max())
    }

    /// Шаг подстройки после одной обратной связи: промах расширяет интервал, попадание
    /// сужает, размер шага задаёт скорость подстройки. Результат упирается в границы.
    pub fn update(&self, theta: f32, miss: bool) -> f32 {
        self.clip(theta as f64 + self.gamma * (miss as u8 as f64 - self.target))
    }

    /// Одна обратная связь по нормированному выходу факта за интервал. Возвращает новый
    /// логарифм множителя и признак промаха.
    pub fn step(&self, theta: f32, score: f64) -> (f32, bool) {
        let miss = score > (theta as f64).exp();
        (self.update(theta, miss), miss)
    }
}

/// Конформная поправка квантилей таблицей, развёрнутой по лидам: меняется ширина
/// интервалов, медиана остаётся той, что выдала модель.
pub fn apply_conformal(q: &mut [f32], table: &[f32], nq: usize, i_med: usize) {
    for (row, t) in q.chunks_exact_mut(nq).zip(table.chunks_exact(nq)) {
        for (v, s) in row.iter_mut().zip(t) {
            *v += *s;
        }
        order_around_median(row, i_med);
    }
}

/// Растяжение квантилей вокруг медианы в адаптивный множитель раз. Нулевой логарифм
/// множителя оставляет квантили нетронутыми, без арифметики.
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
        order_around_median(row, i_med);
    }
}

/// Порядок квантилей без сдвига медианы: выше медианы каждый квантиль не меньше соседа
/// слева, ниже медианы - не больше соседа справа. Пропуск значения распространяется от
/// медианы наружу.
pub fn order_around_median(row: &mut [f32], i_med: usize) {
    for i in (0..i_med).rev() {
        let (a, b) = (row[i + 1], row[i]);
        row[i] = if a.is_nan() || b.is_nan() {
            f32::NAN
        } else if b > a {
            a
        } else {
            b
        };
    }
    for i in i_med + 1..row.len() {
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

/// Нормированный выход факта за интервал для одной строки квантилей.
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
