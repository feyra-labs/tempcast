//! Смена координат и высоты прибора между записью состояния и текущим запуском.
//!
//! Исходов три. Та же точка - всё как обычно. Уточнение: широта, долгота и высота
//! отличаются не больше порогов из манифеста; окно сохраняется и пересчитывается для
//! новой точки, множитель калибровки сохраняется. Перенос: хотя бы одна разница больше
//! порога; окно опустошается, множитель калибровки обнуляется, момент последнего шага
//! сохраняется.
//!
//! Точки сравниваются так, как они лежат в состоянии, в одинарной точности. Разница
//! долгот берётся по кратчайшей дуге. Пороги включительные. Точка с нечисловыми
//! значениями считается перенесённой.
use crate::manifest::RuntimeJson;

/// Исход сравнения точки загруженного состояния с текущей.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteChange {
    Same,
    Refined,
    Moved,
}

impl SiteChange {
    /// Имя исхода в сводке рантайма.
    pub fn as_str(self) -> &'static str {
        match self {
            SiteChange::Same => "same",
            SiteChange::Refined => "refined",
            SiteChange::Moved => "moved",
        }
    }
}

/// Разница долгот по кратчайшей дуге, градусы: от 0 до 180.
pub fn lon_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    if d.is_nan() {
        d
    } else {
        d.min(360.0 - d)
    }
}

/// Разницы широты, долготы и высоты двух точек: градусы, градусы, метры.
pub fn site_gap(old: [f32; 3], new: [f32; 3]) -> [f64; 3] {
    let (o, n) = (old.map(f64::from), new.map(f64::from));
    [(n[0] - o[0]).abs(), lon_gap(n[1], o[1]), (n[2] - o[2]).abs()]
}

/// Исход сравнения и разницы точки состояния `old` с точкой запуска `new`.
pub fn site_change(old: [f32; 3], new: [f32; 3], lim: &RuntimeJson) -> (SiteChange, [f64; 3]) {
    let gap = site_gap(old, new);
    let limits = [lim.site_max_dlat_deg, lim.site_max_dlon_deg, lim.site_max_delev_m];
    let kind = if gap.iter().all(|g| *g == 0.0) {
        SiteChange::Same
    } else if gap.iter().zip(limits).all(|(g, l)| *g <= l) {
        SiteChange::Refined
    } else {
        SiteChange::Moved
    };
    (kind, gap)
}

/// Сдвиг точки и пороги одной строкой для лога.
pub fn describe_gap(gap: [f64; 3], lim: &RuntimeJson) -> String {
    format!(
        "широта на {:.4}° (порог {}°), долгота на {:.4}° (порог {}°), высота на {:.1} м (порог {} м)",
        gap[0],
        lim.site_max_dlat_deg,
        gap[1],
        lim.site_max_dlon_deg,
        gap[2],
        lim.site_max_delev_m
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIM: RuntimeJson = RuntimeJson {
        site_max_dlat_deg: 0.5,
        site_max_dlon_deg: 0.5,
        site_max_delev_m: 100.0,
    };

    #[test]
    fn longitude_wraps_around_the_date_line() {
        assert!((lon_gap(179.9, -179.9) - 0.2).abs() < 1e-9);
        assert_eq!(lon_gap(180.0, -180.0), 0.0);
        assert_eq!(lon_gap(10.0, 370.0), 0.0);
        assert!(lon_gap(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn thresholds_are_inclusive_and_nan_means_moved() {
        let base = [52.0, 4.9, 10.0];
        assert_eq!(site_change(base, base, &LIM).0, SiteChange::Same);
        assert_eq!(site_change(base, [52.5, 4.9, 110.0], &LIM).0, SiteChange::Refined);
        assert_eq!(site_change(base, [52.0, 4.9, 110.5], &LIM).0, SiteChange::Moved);
        assert_eq!(site_change([f32::NAN, 4.9, 10.0], base, &LIM).0, SiteChange::Moved);
    }
}
