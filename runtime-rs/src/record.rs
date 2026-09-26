//! Запись значений так, как их пишет прибор.
//!
//! Температура записывается целыми градусами Цельсия, влажность целыми процентами,
//! давление десятыми долями гектопаскаля. Округление к ближайшему значению сетки, ровно
//! половина округляется к чётному. Совпадение с эталонной реализацией закреплено общим
//! файлом эталонных случаев в данных тестов репозитория.

/// Сколько делений сетки записи приходится на единицу измерения по каналам T, P, RH.
pub const RECORD_SCALE: [f64; 3] = [1.0, 10.0, 1.0];

/// Округление к ближайшему целому, ровно половина к чётному. Отрицательный ноль
/// становится обычным нулём, нечисловое значение остаётся нечисловым.
pub fn round_half_even(v: f64) -> f64 {
    v.round_ties_even() + 0.0
}

/// Значение канала `ch` (0 - T, 1 - P, 2 - RH) на сетке записи прибора.
pub fn record_channel(v: f32, ch: usize) -> f32 {
    let s = RECORD_SCALE[ch];
    (round_half_even(v as f64 * s) / s) as f32
}

/// Значения T, P, RH на сетке записи прибора.
pub fn record_values(v: [f32; 3]) -> [f32; 3] {
    [
        record_channel(v[0], 0),
        record_channel(v[1], 1),
        record_channel(v[2], 2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_go_to_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(-0.5).to_bits(), 0.0f64.to_bits());
        assert_eq!(round_half_even(-2.5), -2.0);
    }

    #[test]
    fn record_is_idempotent() {
        for i in 0..2000 {
            let v = [i as f32 * 0.37 - 300.0, 900.0 + i as f32 * 0.123, i as f32 * 0.051];
            let once = record_values(v);
            assert_eq!(record_values(once), once);
        }
    }
}
