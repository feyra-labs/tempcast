"""Пример сквозного инференса МАЯК на устройстве (потоковый путь A):
загрузка модели и конформной таблицы → восстановление состояния после ребута →
почасовые шаги по данным датчиков → выпуск прогноза → атомарное сохранение
состояния (<4 КБ). QC точки и watchdog-фолбэк — внутри StreamingMayak/safe_forecast.

Запуск:
    python -m mayak.runtime.run_inference --ckpt runs/stageB/best.ckpt \
        --conformal runs/conformal.npy --lat 52.37 --lon 4.90 --elev -2
"""
import argparse, os
from datetime import datetime, timezone, timedelta
import numpy as np

from mayak.constants import H
from mayak.runtime.streaming import StreamingMayak, safe_forecast
from mayak.timeaxis import future_calendar, utc_to_doy_hour

STATE_FILES = ["runtime/state_a.bin", "runtime/state_b.bin"]


def latest_state():
    cand = [(f, os.path.getmtime(f)) for f in STATE_FILES if os.path.exists(f)]
    return max(cand, key=lambda x: x[1])[0] if cand else None


def save_state(stream, toggle):
    """Атомарная запись в чередуемый файл: пишем во временный, затем os.replace."""
    f = STATE_FILES[toggle % 2]; tmp = f + ".tmp"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(tmp, "wb") as fh:
        fh.write(stream.serialize())
    os.replace(tmp, f)


def read_sensors(t):
    """ЗАГЛУШКА. Здесь читаете свои датчики и возвращаете (T °C, P гПа, RH %).
    При отказе любого датчика верните None для него — QC опустит маску, прогноз
    деградирует плавно к климатологии. Тут — синтетика для примера."""
    import math
    T = 8 + 5 * math.sin(2 * math.pi * t.hour / 24) + np.random.randn()
    return float(T), 1013.0 + np.random.randn(), 80.0 + 3 * np.random.randn()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conformal", default=None)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--clim-fallback", type=float, default=10.0,
                    help="климат-средняя T (°C) для watchdog-фолбэка")
    ap.add_argument("--sigma-fallback", type=float, default=4.0)
    args = ap.parse_args()
    from mayak.lit import LitMayak                     # тяжёлый импорт — после разбора аргументов

    model = LitMayak.load_from_checkpoint(args.ckpt, map_location="cpu").model.eval()
    conf = args.conformal if (args.conformal and os.path.exists(args.conformal)) else None
    stream = StreamingMayak(model, args.lat, args.lon, args.elev, conformal=conf)
    print("Конформная калибровка:", "включена" if conf else "ОТКЛЮЧЕНА (таблица не передана)")

    st = latest_state()
    if st:
        with open(st, "rb") as fh:
            stream.load_state(fh.read())
        print("Состояние восстановлено из", st, f"({os.path.getsize(st)} Б)")
    else:
        print("Чистый старт (история пуста, L=0). Первый прогноз = климат-поле + паспорт.")
        # Если есть сохранённая история первого включения — можно прогреться:
        # stream.warm_start(x_hist, mask_hist, doy_hist, hour_hist)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for k in range(48):
        ts = now - timedelta(hours=48 - k)
        doy, hour = utc_to_doy_hour(ts)
        T, P, RH = read_sensors(ts)
        stream.step(T, P, RH, doy, hour)
        save_state(stream, k)

    # лид 1 — час, следующий за последним наблюдением (как в датасете: история до t−1, цель с t)
    last_obs = now - timedelta(hours=1)
    doy_f, hour_f = future_calendar(last_obs, H)
    mu_clim_fb = np.full(H, args.clim_fallback, np.float32)
    q, mu = safe_forecast(stream, doy_f, hour_f, mu_clim_fb, args.sigma_fallback)

    for h in (1, 24, 72, 168):
        j = h - 1
        print(f"  +{h:>3} ч:  T̂ = {mu[j]:5.1f} °C   "
              f"90%-интервал [{q[j,0]:5.1f}, {q[j,6]:5.1f}]")


if __name__ == "__main__":
    main()
