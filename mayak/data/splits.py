"""Границы временных сплитов в индексах часа"""

def time_bounds(n_hours, hours_per_year=8766, calib_days=60):
    """Возвращает словарь с диапазонами [начало, конец) в часах.

    train  — всё, кроме последнего года и калибровочного хвоста перед ним;
    calib  — последние calib_days суток предпоследнего года (time-val);
    test   — последний год.
    """
    test_start = max(0, n_hours - hours_per_year)
    calib_start = max(0, test_start - calib_days * 24)
    return {
        "train": (0, calib_start),
        "calib": (calib_start, test_start),
        "test":  (test_start, n_hours),
    }