"""Замер: пакет ↔ поток на полном выпуске и стоимость часа.

Пишет JSON с числами, которые README и описание рантайма заявляют словами:
max|Δq| пакет ↔ поток по всем лидам, расхождение после перезапуска из сохранённого
состояния, FLOP и время энкодера на час (инкрементальный шаг против пересчёта по
буферу рецептивного поля), размер персистентного состояния и буферов энкодера.

    python scripts/stream_equivalence.py --ckpt runs/mayak/stageB/best.ckpt
    python scripts/stream_equivalence.py --windows 4 --out runs/stream_equivalence.json
Без --ckpt замер идёт на случайно инициализированной модели (сид --seed).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="эквивалентность пакет/поток и стоимость часа")
    ap.add_argument("--ckpt", default=None, help="чекпойнт; по умолчанию - случайная модель")
    ap.add_argument("--windows", type=int, default=16, help="число окон для расхождения")
    ap.add_argument("--reps", type=int, default=300, help="повторов для замера времени")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/stream_equivalence.json")
    args = ap.parse_args()

    import torch

    from mayak.runtime.equivalence import divergence, step_cost, stream_hour_ms
    if args.ckpt:
        from mayak.lit import load_model
        model = load_model(args.ckpt).eval()
    else:
        from mayak.model import MAYAK
        torch.manual_seed(args.seed)
        model = MAYAK().eval()
    torch.set_num_threads(1)

    rep = dict(ckpt=args.ckpt, seed=args.seed, threads=1)
    rep.update(divergence(model, args.windows, seed=args.seed))
    rep.update(step_cost(model, args.reps, seed=args.seed))
    ms, stream = stream_hour_ms(model, hours=args.reps, seed=args.seed)
    rep.update(stream_step_ms=ms, state_bytes=len(stream.serialize()),
               stream_window=model.cfg.stream_window)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(f"пакет ↔ поток, max|Δq|:        {rep['batch_stream_max_abs']:.2e} °C "
          f"({rep['n_windows']} окон, все лиды)")
    print(f"после перезапуска, max|Δq|:    {rep['restart_max_abs']:.2e} °C")
    print(f"FLOP энкодера на час:          {rep['encoder_flops_step']} против "
          f"{rep['encoder_flops_full_window']} (×{rep['flops_ratio']:.0f})")
    print(f"время энкодера на час, мс:     {rep['encoder_ms_step']:.3f} против "
          f"{rep['encoder_ms_full_window']:.3f} (×{rep['time_ratio']:.1f})")
    print(f"полный step рантайма, мс:      {ms:.3f}")
    print(f"состояние на диске:            {rep['state_bytes']} Б; буферы энкодера в "
          f"памяти {rep['encoder_buffer_bytes']} Б")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
