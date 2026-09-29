//! Календарная конвенция проекта (mayak/timeaxis.py).
//!
//! * момент - целое число часов от эпохи Unix в UTC;
//! * `doy` - день года с нуля (1 января = 0) с дробной частью, равной доле суток;
//! * `hour` - час UTC в [0, 24).
//!
//! Арифметика повторяет Python до бита: секунды от начала года делятся на 86400 в f64,
//! затем результат округляется до f32 (как `window_calendar`).

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

/// Число часов високосного года: длина таблицы климатологии точки.
pub const HOURS_OF_YEAR: usize = 366 * 24;

/// Номер часа от начала года UTC: от 0 до 8783 в високосном году, до 8759 в обычном.
pub fn hour_of_year(unix_hour: i64) -> usize {
    let days = unix_hour.div_euclid(24);
    let start = days_from_civil(year_of_days(days), 1, 1) * 24;
    (unix_hour - start) as usize
}

/// Календарь горизонта: часы после последнего наблюдения, по одному на лид.
pub fn future_calendar(last_obs_hour: i64, horizon: usize, doy: &mut [f32], hour: &mut [f32]) {
    for h in 0..horizon {
        let (d, hr) = doy_hour(last_obs_hour + 1 + h as i64);
        doy[h] = d;
        hour[h] = hr;
    }
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
    fn hour_of_year_covers_leap_years() {
        assert_eq!(hour_of_year(0), 0);
        assert_eq!(hour_of_year(days_from_civil(2024, 12, 31) * 24 + 23), HOURS_OF_YEAR - 1);
        assert_eq!(hour_of_year(days_from_civil(2025, 12, 31) * 24 + 23), 8759);
        assert_eq!(hour_of_year(days_from_civil(2025, 1, 1) * 24), 0);
        assert_eq!(hour_of_year(days_from_civil(1969, 12, 31) * 24 + 5), 8759 - 18);
    }
}
