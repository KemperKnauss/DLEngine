from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def pareto_mask(frame: pd.DataFrame, x_col: str, y_col: str, maximize_y: bool) -> pd.Series:
    mask = []
    for _, row in frame.iterrows():
        cheaper_or_equal = frame[x_col] <= row[x_col]
        if maximize_y:
            better_or_equal = frame[y_col] >= row[y_col]
            strictly_better = (frame[x_col] < row[x_col]) | (frame[y_col] > row[y_col])
        else:
            better_or_equal = frame[y_col] <= row[y_col]
            strictly_better = (frame[x_col] < row[x_col]) | (frame[y_col] < row[y_col])
        dominated = (cheaper_or_equal & better_or_equal & strictly_better).any()
        mask.append(not dominated)
    return pd.Series(mask, index=frame.index)


def plot_frontier(frame: pd.DataFrame, x_col: str, y_col: str, maximize_y: bool, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.dropna(subset=[x_col, y_col]).copy()
    clean[x_col] = pd.to_numeric(clean[x_col])
    clean[y_col] = pd.to_numeric(clean[y_col])
    clean = clean.sort_values(x_col)
    frontier = clean[pareto_mask(clean, x_col, y_col, maximize_y)].sort_values(x_col)

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(clean[x_col], clean[y_col], s=72, alpha=0.72, label="models")
    plt.plot(frontier[x_col], frontier[y_col], linewidth=2.4, marker="o", label="Pareto frontier")
    for _, row in clean.iterrows():
        plt.annotate(str(row["model"]), (row[x_col], row[y_col]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))
    plt.title(f"Pareto frontier: {y_col.replace('_', ' ')} vs {x_col.replace('_', ' ')}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pareto frontier graphs from evaluation metrics.")
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics)
    plot_frontier(frame, "latency_ms", "top1", True, args.out_dir / "pareto_top1_vs_latency.png")
    plot_frontier(frame, "params", "top1", True, args.out_dir / "pareto_top1_vs_params.png")
    plot_frontier(frame, "latency_ms", "value_rmse", False, args.out_dir / "pareto_value_rmse_vs_latency.png")
    print(f"Wrote Pareto plots to {args.out_dir}")


if __name__ == "__main__":
    main()

