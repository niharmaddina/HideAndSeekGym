"""Plot episode return against timesteps for one or more training runs.

    python scripts/plot_curve.py runs/seed0 runs/seed1 --out assets/learning_curve.png

Each run directory must hold the monitor.csv written by train.py. Returns are smoothed with a
moving average over --window episodes; the raw curve is drawn faintly underneath.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run: Path) -> pd.DataFrame:
    csv = next(iter(sorted(run.glob("*monitor.csv"))))        # Monitor writes <name>.monitor.csv
    df = pd.read_csv(csv, skiprows=1)                        # first line is the Monitor header comment
    df["timesteps"] = df["l"].cumsum()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="assets/learning_curve.png")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--title", default="PPO on HideAndSeekEnv")
    ap.add_argument("--labels", nargs="+", help="legend labels, one per run (default: the run directory name)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))
    if args.labels and len(args.labels) != len(args.runs):
        ap.error("--labels needs one label per run")
    for i, run in enumerate(args.runs):
        df = load(Path(run))
        smooth = df["r"].rolling(args.window, min_periods=1).mean()
        line, = ax.plot(df["timesteps"], smooth, label=args.labels[i] if args.labels else Path(run).name)
        ax.plot(df["timesteps"], df["r"], alpha=0.2, color=line.get_color())
        first, last = df["r"].iloc[: args.window].mean(), df["r"].iloc[-args.window:].mean()
        print(f"{run}: {len(df)} episodes, first {args.window} mean return {first:.1f}, last {args.window} mean return {last:.1f}")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"episode return (moving average, {args.window} episodes)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
