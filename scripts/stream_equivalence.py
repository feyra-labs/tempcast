"""Замер: поток против пакета на полном выпуске, стоимость часа и выпуска.

Пишет JSON с числами, которые README и описание рантайма заявляют словами:
наибольшее расхождение квантилей потока и пакета по всем лидам при выпусках в случайные
часы длинного ряда и его изменение во времени, расхождение после перезапуска из
сохранённого состояния, FLOP и время шага часа, выпуска и пакетного прохода, размер
состояния на диске и колец в памяти.

    python scripts/stream_equivalence.py --ckpt runs/mayak/stageB/best.ckpt
    python scripts/stream_equivalence.py --issues 40 --hours 10000

Без --ckpt замер идёт на случайно инициализированной модели с сидом --seed.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="эквивалентность потока и пакета, стоимость часа")
    ap.add_argument("--ckpt", default=None, help="чекпойнт; по умолчанию - случайная модель")
    ap.add_argument("--issues", type=int, default=40, help="выпусков в случайные часы")
    ap.add_argument("--hours", type=int, default=10_000, help="длина ряда, часы")
    ap.add_argument("--reps", type=int, default=200, help="повторов для замера времени")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/stream_equivalence.json")
    args = ap.parse_args()

    import torch

    from mayak.runtime.equivalence import divergence, runtime_cost, stream_hour_ms
    if args.ckpt:
        from mayak.lit import load_model
        model = load_model(args.ckpt).eval()
    else:
        from mayak.model import MAYAK
        torch.manual_seed(args.seed)
        model = MAYAK().eval()
    torch.set_num_threads(1)

    rep = dict(ckpt=args.ckpt, seed=args.seed, threads=1)
    rep.update(divergence(model, args.issues, hours=args.hours, seed=args.seed))
    rep.update(runtime_cost(model, args.reps, seed=args.seed))
    ms, stream = stream_hour_ms(model, hours=args.reps, seed=args.seed)
    rep.update(stream_step_ms=ms, state_bytes=len(stream.serialize()),
               memory_bytes=stream.memory_nbytes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    errs = [e for _, e in rep["batch_stream_by_issue"]]
    third = max(1, len(errs) // 3)
    print(f"поток и пакет, max|Δq|:         {rep['batch_stream_max_abs']:.2e} °C "
          f"({rep['n_issues']} выпусков на {rep['hours']} ч, все лиды)")
    print(f"первая и последняя треть ряда:  {max(errs[:third]):.2e} и "
          f"{max(errs[-third:]):.2e} °C")
    print(f"после перезапуска, max|Δq|:     {rep['restart_max_abs']:.2e} °C")
    print(f"FLOP шага / выпуска / пакета:   {rep['step_flops']} / {rep['issue_flops']} / "
          f"{rep['pack_flops']}")
    print(f"время шага / выпуска / пакета:  {rep['step_ms']:.3f} / {rep['issue_ms']:.3f} / "
          f"{rep['pack_ms']:.3f} мс")
    print(f"полный шаг рантайма, мс:        {ms:.3f}")
    print(f"состояние на диске:             {rep['state_bytes']} Б; кольца и буфер в памяти "
          f"{rep['memory_bytes']} Б")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
