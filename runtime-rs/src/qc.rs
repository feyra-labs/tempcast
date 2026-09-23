//! Поточечный контроль качества часа - mayak.data.qc.point_qc: значение принимается,
//! если оно конечно и лежит в физическом диапазоне канала (границы включены).
//! Пределы берутся из манифеста, то есть из того же PHYS, что в обучении.

pub fn point_qc(obs: [Option<f64>; 3], bounds: &[[f64; 2]; 3]) -> ([f32; 3], [f32; 3]) {
    let mut x = [0.0f32; 3];
    let mut m = [0.0f32; 3];
    for j in 0..3 {
        if let Some(v) = obs[j] {
            if v.is_finite() && bounds[j][0] <= v && v <= bounds[j][1] {
                x[j] = v as f32;
                m[j] = 1.0;
            }
        }
    }
    (x, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    const B: [[f64; 2]; 3] = [[-90.0, 60.0], [300.0, 1100.0], [0.0, 100.0]];

    #[test]
    fn channel_isolation_and_edges() {
        let (x, m) = point_qc([Some(75.0), Some(1100.0), Some(f64::NAN)], &B);
        assert_eq!(m, [0.0, 1.0, 0.0]);
        assert_eq!(x, [0.0, 1100.0, 0.0]);
        let (_, m) = point_qc([Some(-90.0), None, Some(0.0)], &B);
        assert_eq!(m, [1.0, 0.0, 1.0]);
    }
}
