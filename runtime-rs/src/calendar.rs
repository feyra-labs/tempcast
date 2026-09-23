//! Календарная конвенция проекта (mayak/timeaxis.py).
//!
//! * момент - целое число часов от эпохи Unix в UTC;
//! * `doy` - день года с нуля (1 января = 0) с дробной частью, равной доле суток;
//! * `hour` - час UTC в [0, 24).
//!
//! Арифметика повторяет Python до бита: секунды от начала года делятся на 86400 в f64,
//! затем результат округляется до f32 (как `window_calendar`).

/// Длины года в часах: календарь потока непрерывен, если час года растёт на 1 или
/// переходит в 0 после 8759 / 8783.
pub const YEAR_HOURS: [i64; 2] = [365 * 24, 366 * 24];

/// Дни от 1970-01-01 до даты (y, m, d) по пролептическому григорианскому календарю.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Год по числу дней от эпохи.
pub fn year_of_days(z: i64) -> i64 {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    yoe + era * 400 + if m <= 2 { 1 } else { 0 }
}

/// Час от эпохи → (doy, hour) по конвенции проекта.
pub fn doy_hour(unix_hour: i64) -> (f32, f32) {
    let days = unix_hour.div_euclid(24);
    let year = year_of_days(days);
    let start = days_from_civil(year, 1, 1) * 24;
    let sec = ((unix_hour - start) * 3600) as f64;
    ((sec / 86400.0) as f32, ((sec % 86400.0) / 3600.0) as f32)
}

/// Календарь горизонта: часы last_obs + 1 … last_obs + horizon (timeaxis.future_calendar).
pub fn future_calendar(last_obs_hour: i64, horizon: usize, doy: &mut [f32], hour: &mut [f32]) {
    for h in 0..horizon {
        let (d, hr) = doy_hour(last_obs_hour + 1 + h as i64);
        doy[h] = d;
        hour[h] = hr;
    }
}

/// Последний час от эпохи не позже `before`, у которого час года равен `hoy`
/// (состояние хранит только час года; после перезапуска так восстанавливается момент
/// последнего шага, чтобы заполнить простой пустыми шагами). None - такого часа нет
/// в текущем и предыдущем году.
pub fn last_hour_with_hoy(hoy: i64, before: i64) -> Option<i64> {
    let year = year_of_days(before.div_euclid(24));
    for y in [year, year - 1] {
        let start = days_from_civil(y, 1, 1) * 24;
        let len = (days_from_civil(y + 1, 1, 1) - days_from_civil(y, 1, 1)) * 24;
        let u = start + hoy;
        if hoy < len && u <= before {
            return Some(u);
        }
    }
    None
}

/// doy → час года (mayak.runtime.streaming.hour_of_year): round половин к чётному.
pub fn hour_of_year(doy: f32) -> i64 {
    (doy as f64 * 24.0).round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_leap_years() {
        assert_eq!(doy_hour(0), (0.0, 0.0));
        let dec31_2024 = days_from_civil(2024, 12, 31) * 24 + 23;
        assert_eq!(doy_hour(dec31_2024), ((365.0f64 + 23.0 / 24.0) as f32, 23.0));
        let dec31_2100 = days_from_civil(2100, 12, 31) * 24;
        assert_eq!(doy_hour(dec31_2100).0, 364.0);
        assert_eq!(year_of_days(days_from_civil(2000, 2, 29)), 2000);
    }

    #[test]
    fn last_hour_recovery_across_new_year() {
        let t = days_from_civil(2025, 1, 1) * 24 + 5;
        assert_eq!(last_hour_with_hoy(3, t), Some(t - 2));
        assert_eq!(last_hour_with_hoy(8783, t), Some(t - 6)); // 2024 - високосный
    }
}
