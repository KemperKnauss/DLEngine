from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def base_model_name(model: str) -> str:
    return re.sub(r"_(dynamic_quantized|finetune_control|pruned_\d+)$", "", str(model))


def display_model_name(model: str, compression: str) -> str:
    match = re.search(r"(small|depthwise)_cnn_(\d+)x(\d+)", str(model))
    base = str(model)
    if match:
        family = "SCNN" if match.group(1) == "small" else "DW-CNN"
        base = f"{family}-{match.group(2)}x{match.group(3)}"
    suffixes = {
        "original": "Original",
        "finetune_control": "Control",
        "dynamic_quantization": "INT8",
        "unstructured_pruning": "Pruned",
    }
    suffix = suffixes.get(str(compression), str(compression))
    sparsity = re.search(r"_pruned_(\d+)$", str(model))
    if sparsity:
        suffix = f"Pruned {sparsity.group(1)}%"
    return f"{base} {suffix}"


def save_scatter(frame: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.dropna(subset=[x_col, y_col]).copy()
    clean[x_col] = pd.to_numeric(clean[x_col])
    clean[y_col] = pd.to_numeric(clean[y_col])

    colors = {
        "original": "#4c78a8",
        "finetune_control": "#72b7b2",
        "dynamic_quantization": "#f58518",
        "unstructured_pruning": "#e45756",
    }
    plt.figure(figsize=(8.4, 5.5))
    for compression, group in clean.groupby("compression"):
        plt.scatter(
            group[x_col],
            group[y_col],
            s=74,
            alpha=0.82,
            color=colors.get(compression),
            label=compression.replace("_", " ").title(),
        )
        for _, row in group.iterrows():
            plt.annotate(
                display_model_name(row["model"], row["compression"]),
                (row[x_col], row[y_col]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
    labels = {
        "latency_ms": "CPU latency (ms / position)",
        "top1": "Top-1 move accuracy",
        "model_mb": "Checkpoint size (MB)",
        "value_rmse": "Value RMSE",
    }
    plt.xlabel(labels.get(x_col, x_col.replace("_", " ")))
    plt.ylabel(labels.get(y_col, y_col.replace("_", " ")))
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

    controls = frame[frame["compression"] == "finetune_control"].copy()
    plt.figure(figsize=(7.8, 5.2))
    for base_model, group in pruning.groupby(pruning["model"].map(base_model_name)):
        group = group.sort_values("sparsity_target")
        control = controls[controls["model"].map(base_model_name) == base_model]
        if not control.empty:
            control_row = control.iloc[0]
            group = pd.concat(
                [
                    pd.DataFrame([{"sparsity_target": 0.0, "top1": float(control_row["top1"])}]),
                    group[["sparsity_target", "top1"]],
                ],
                ignore_index=True,
            )
        plt.plot(
            group["sparsity_target"],
            group["top1"],
            marker="o",
            linewidth=2.0,
            label=display_model_name(base_model, "original").replace(" Original", ""),
        )
    plt.xlabel("Target sparsity")
    plt.ylabel("Top-1 move accuracy")
    plt.title("Pruning accuracy after equal fine-tuning")
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
    compressed_bases = set(compression["model"].map(base_model_name))
    final = final[final["model"].isin(compressed_bases)].copy()
    combined = pd.concat([final, compression], ignore_index=True, sort=False)

    save_scatter(combined, "latency_ms", "top1", args.out_dir / "compression_top1_vs_latency.png", "Compression: top1 vs latency")
    save_scatter(combined, "model_mb", "top1", args.out_dir / "compression_top1_vs_model_mb.png", "Compression: top1 vs model size")
    save_scatter(combined, "latency_ms", "value_rmse", args.out_dir / "compression_value_rmse_vs_latency.png", "Compression: value RMSE vs latency")
    plot_pruning(combined, args.out_dir / "pruning_top1_vs_sparsity.png")
    print(f"Wrote compression plots to {args.out_dir}")


if __name__ == "__main__":
    main()
