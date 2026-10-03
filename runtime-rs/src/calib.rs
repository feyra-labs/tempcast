//! Калибровка интервалов на устройстве: сплит-конформная таблица, адаптивные множители
//! ширины интервала по бинам лидов и их онлайн-подстройка по промахам.
//!
//! Эталонная реализация калибровки написана на Python; здесь её перенос. Совпадение
//! двух реализаций проверяют эталонные векторы, а не ревью. Таблица приходит уже
//! развёрнутой по лидам для каждого бина длины истории; рантайм выбирает её строку по
//! длине истории выпуска.
//!
//! Адаптивная калибровка держит множитель на каждый бин лидов и кольцо по
//! часам-мишеням: для каждого из следующих часов горизонта и каждого бина - медиана и
//! границы интервала одной записи. Выпуск пишет бин, только когда сумма момента выпуска
//! и первого лида бина делится на ширину бина, и тогда пишет все лиды бина: каждый час
//! получает ровно одну запись на бин, а лиды записей идут по кругу через весь бин.
//! Валидный час обновляет каждый бин не больше одного раза. Кольцо на диск не пишется.

/// Число бинов лидов адаптивной калибровки: столько множителей хранит состояние.
pub const N_LEAD_BINS: usize = 4;

/// Час кольца калибровки без записи.
const NO_HOUR: i64 = i64::MIN;

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

/// Номер бина лидов для каждого лида горизонта; лиды за последним бином относятся к
/// последнему.
pub fn lead_bin_of(lead_bins: &[[usize; 2]], horizon: usize) -> Vec<usize> {
    (1..=horizon)
        .map(|h| {
            lead_bins
                .iter()
                .position(|b| b[0] <= h && h <= b[1])
                .unwrap_or(lead_bins.len() - 1)
        })
        .collect()
}

/// Первый лид, считая с единицы, и число лидов каждого бина в пределах горизонта.
/// Лиды бина идут подряд: это гарантирует проверка бинов в манифесте. У бина без лидов
/// на горизонте первый лид ноль и ширина единица; записей у него всё равно нет.
fn lead_bin_spans(lead_bin: &[usize]) -> ([i64; N_LEAD_BINS], [i64; N_LEAD_BINS]) {
    let mut lo = [0i64; N_LEAD_BINS];
    let mut width = [1i64; N_LEAD_BINS];
    for (k, &b) in lead_bin.iter().enumerate() {
        let lead = k as i64 + 1;
        if lo[b] == 0 {
            lo[b] = lead;
        }
        width[b] = lead - lo[b] + 1;
    }
    (lo, width)
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

/// Растяжение квантилей вокруг медианы: у каждого лида множитель своего бина. Лиды с
/// нулевым логарифмом множителя остаются нетронутыми.
pub fn apply_adaptive_bins(q: &mut [f32], theta: &[f32], lead_bin: &[usize], nq: usize, i_med: usize) {
    for (row, &b) in q.chunks_exact_mut(nq).zip(lead_bin) {
        apply_adaptive(row, theta[b], nq, i_med);
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

/// Нормированный выход факта за интервал по его границам и медиане.
pub fn aci_score_bounds(y: f64, lo: f32, med: f32, hi: f32) -> f64 {
    let med = med as f64;
    if !y.is_finite() || !med.is_finite() {
        return f64::NAN;
    }
    let u = y - med;
    if u == 0.0 {
        return 0.0;
    }
    let d = if u >= 0.0 { hi as f64 - med } else { med - lo as f64 };
    if d > 0.0 {
        u.abs() / d
    } else {
        f64::INFINITY
    }
}

/// Нормированный выход факта за интервал для одной строки квантилей.
pub fn aci_score(y: f64, q: &[f32], interval: [usize; 2], i_med: usize) -> f64 {
    aci_score_bounds(y, q[interval[0]], q[i_med], q[interval[1]])
}

/// Адаптивная калибровка прибора по бинам лидов: множители, счётчики обратной связи и
/// кольцо по часам-мишеням. Без параметров множители только хранятся, кольца нет.
#[derive(Debug, Clone)]
pub struct Adaptive {
    pub params: Option<AciParams>,
    horizon: usize,
    /// Номер бина для каждого лида горизонта.
    pub lead_bin: Vec<usize>,
    // первый лид и ширина каждого бина в пределах горизонта: по ним выбираются выпуски,
    // которые пишут бин в кольцо
    bin_lo: [i64; N_LEAD_BINS],
    bin_width: [i64; N_LEAD_BINS],
    pub theta: [f32; N_LEAD_BINS],
    pub updates: [u64; N_LEAD_BINS],
    pub misses: [u64; N_LEAD_BINS],
    // кольцо: место часа - абсолютный час по модулю горизонта, внутри - бин лидов
    ring_hour: Vec<i64>,
    ring: Vec<[f32; 3]>,
}

impl Adaptive {
    pub fn new(params: Option<AciParams>, lead_bins: &[[usize; 2]], horizon: usize) -> Self {
        let n = if params.is_some() { horizon * N_LEAD_BINS } else { 0 };
        let lead_bin = lead_bin_of(lead_bins, horizon);
        let (bin_lo, bin_width) = lead_bin_spans(&lead_bin);
        Adaptive {
            params,
            horizon,
            lead_bin,
            bin_lo,
            bin_width,
            theta: [0.0; N_LEAD_BINS],
            updates: [0; N_LEAD_BINS],
            misses: [0; N_LEAD_BINS],
            ring_hour: vec![NO_HOUR; n],
            ring: vec![[0.0; 3]; n],
        }
    }

    /// Новые множители, нулевые счётчики и пустое кольцо.
    pub fn reset(&mut self, theta: [f32; N_LEAD_BINS]) {
        let p = self.params;
        self.theta = theta.map(|t| match &p {
            Some(a) => a.clip(t as f64),
            None => t,
        });
        self.updates = [0; N_LEAD_BINS];
        self.misses = [0; N_LEAD_BINS];
        self.clear();
    }

    /// Пустое кольцо; множители и счётчики остаются.
    pub fn clear(&mut self) {
        self.ring_hour.fill(NO_HOUR);
    }

    /// Размер кольца в памяти, байт.
    pub fn ring_bytes(&self) -> usize {
        8 * self.ring_hour.len() + 12 * self.ring.len()
    }

    /// Запись выпуска в кольцо: квантили после конформной таблицы и до множителя, лиды
    /// подряд, горизонт начинается после часа `after`. Пишутся только бины, для которых
    /// сумма `after` и первого лида бина делится на ширину бина; у такого бина - все лиды.
    pub fn record(&mut self, after: i64, q: &[f32], nq: usize, i_med: usize) {
        let Some(p) = self.params else { return };
        let [i, j] = p.interval;
        for k in 0..self.horizon {
            let b = self.lead_bin[k];
            if (after + self.bin_lo[b]).rem_euclid(self.bin_width[b]) != 0 {
                continue;
            }
            let h = after + 1 + k as i64;
            let s = h.rem_euclid(self.horizon as i64) as usize * N_LEAD_BINS + b;
            let row = &q[k * nq..(k + 1) * nq];
            self.ring_hour[s] = h;
            self.ring[s] = [row[i], row[i_med], row[j]];
        }
    }

    /// Обратная связь валидного факта часа: каждый бин с записью на этот час - один раз.
    pub fn feedback(&mut self, y: f64, hour: i64) {
        let Some(p) = self.params else { return };
        let s0 = hour.rem_euclid(self.horizon as i64) as usize * N_LEAD_BINS;
        for b in 0..N_LEAD_BINS {
            if self.ring_hour[s0 + b] != hour {
                continue;
            }
            self.ring_hour[s0 + b] = NO_HOUR;
            let [lo, med, hi] = self.ring[s0 + b];
            let score = aci_score_bounds(y, lo, med, hi);
            if score.is_nan() {
                continue;
            }
            let (t, miss) = p.step(self.theta[b], score);
            self.theta[b] = t;
            self.updates[b] += 1;
            self.misses[b] += miss as u64;
        }
    }
}
