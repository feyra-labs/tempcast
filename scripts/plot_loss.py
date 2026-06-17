import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_csv(tag):
    cands = sorted(glob.glob(f"runs/{tag}/version_*/metrics.csv"), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"не найден metrics.csv для runs/{tag}/ — включён ли CSVLogger?")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stageB", help="stageA / stageB / baseline_gru / ...")
    ap.add_argument("--csv", default=None, help="явный путь к metrics.csv (перекрывает --tag)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv_path = args.csv or find_csv(args.tag)
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for col, label in [("train/loss", "train"), ("val/loss", "val")]:
        if col in df.columns:
            sub = df[["step", col]].dropna()
            ax.plot(sub["step"], sub[col], marker="." if label == "val" else None,
                    ms=6, lw=1.5, label=label)
    ax.set_xlabel("шаг обучения"); ax.set_ylabel("loss (pinball, аномальная шкала)")
    ax.set_title(f"Кривые loss — {args.tag}")
    ax.legend(); ax.grid(alpha=0.3)
    out = args.out or f"runs/plots/loss_{args.tag}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("Сохранено:", out)


if __name__ == "__main__":
    main()