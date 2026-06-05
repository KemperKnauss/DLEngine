from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_scatter(frame: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.dropna(subset=[x_col, y_col]).copy()
    clean[x_col] = pd.to_numeric(clean[x_col])
    clean[y_col] = pd.to_numeric(clean[y_col])

    plt.figure(figsize=(8.0, 5.2))
    for compression, group in clean.groupby("compression"):
        plt.scatter(group[x_col], group[y_col], s=70, alpha=0.75, label=compression)
        for _, row in group.iterrows():
            plt.annotate(str(row["model"]), (row[x_col], row[y_col]), xytext=(5, 4), textcoords="offset points", fontsize=7)
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pruning(frame: pd.DataFrame, output_path: Path) -> None:
    pruning = frame[frame["compression"] == "unstructured_pruning"].copy()
    if pruning.empty:
        return
    pruning["sparsity_target"] = pd.to_numeric(pruning["sparsity_target"])
    pruning["top1"] = pd.to_numeric(pruning["top1"])

    plt.figure(figsize=(7.5, 5.0))
    for base_model, group in pruning.groupby(pruning["model"].str.replace(r"_pruned_\d+$", "", regex=True)):
        group = group.sort_values("sparsity_target")
        plt.plot(group["sparsity_target"], group["top1"], marker="o", linewidth=2.0, label=base_model)
    plt.xlabel("sparsity target")
    plt.ylabel("top1")
    plt.title("Pruning: top1 vs sparsity")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compression experiment metrics.")
    parser.add_argument("--final-metrics", type=Path, required=True)
    parser.add_argument("--compression-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/overnight/compression"))
    args = parser.parse_args()

    final = pd.read_csv(args.final_metrics)
    final["compression"] = "original"
    final["sparsity_target"] = ""
    final["nonzero_params"] = final["params"]
    final["actual_sparsity"] = 0.0

    compression = pd.read_csv(args.compression_metrics)
    combined = pd.concat([final, compression], ignore_index=True, sort=False)

    save_scatter(combined, "latency_ms", "top1", args.out_dir / "compression_top1_vs_latency.png", "Compression: top1 vs latency")
    save_scatter(combined, "model_mb", "top1", args.out_dir / "compression_top1_vs_model_mb.png", "Compression: top1 vs model size")
    save_scatter(combined, "latency_ms", "value_rmse", args.out_dir / "compression_value_rmse_vs_latency.png", "Compression: value RMSE vs latency")
    plot_pruning(combined, args.out_dir / "pruning_top1_vs_sparsity.png")
    print(f"Wrote compression plots to {args.out_dir}")


if __name__ == "__main__":
    main()
