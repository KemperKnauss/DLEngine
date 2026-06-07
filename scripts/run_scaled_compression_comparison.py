from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from chess_student.comparison import (
    StockfishComparisonDataset,
    build_comparison_model,
    count_parameters,
    estimated_sparse_payload_mb,
    parameter_sparsity,
    topk_move_predictions,
)
from scripts.run_fair_compression_comparison import (
    ROOT,
    TrainResult,
    apply_global_pruning,
    attach_gpu_latency,
    dynamic_quantize,
    evaluate_model,
    fine_tune_one_epoch,
    init_wandb,
    load_checkpoint,
    log,
    make_loader,
    measure_latency_ms,
    remove_pruning,
    save_checkpoint,
    save_figure,
    serialized_size_mb,
    train_model,
    verify_split_isolation,
)


DEFAULT_SPLIT_ROOT = ROOT / "data" / "overnight" / "final_v2" / "splits"
DEFAULT_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77)
DEFAULT_ALPHAS = (0.25, 0.50, 0.75)
DEFAULT_TEMPERATURES = (1.0, 2.0, 4.0)


def parse_numbers(value: str, kind: type[int] | type[float]) -> tuple[int | float, ...]:
    return tuple(kind(part.strip()) for part in value.split(",") if part.strip())


def checkpoint_result(path: Path, device: torch.device) -> TrainResult:
    model, metadata = load_checkpoint(path, device)
    return TrainResult(
        model=model,
        history=[],
        seconds=float(metadata.get("train_seconds", 0.0)),
        peak_gpu_mb=float(metadata.get("peak_gpu_mb", 0.0)),
        epochs_completed=int(metadata.get("epochs_completed", 0)),
        best_val_loss=float(metadata.get("best_val_loss", float("nan"))),
        best_epoch=int(metadata.get("best_epoch", 0)),
        wandb_url=str(metadata.get("wandb_url", "")),
    )


def training_metadata(result: TrainResult, **extra: Any) -> dict[str, Any]:
    return {
        "train_seconds": result.seconds,
        "peak_gpu_mb": result.peak_gpu_mb,
        "epochs_completed": result.epochs_completed,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "wandb_url": result.wandb_url,
        **extra,
    }


def validation_quality(
    model: nn.Module,
    dataset: StockfishComparisonDataset,
    batch_size: int,
) -> dict[str, float]:
    device = torch.device("cpu")
    model = model.cpu().eval()
    loader = make_loader(dataset, batch_size, False, device, 0)
    total = 0
    top1 = 0
    top3 = 0
    policy_sum = 0.0
    from chess_student.comparison import sparse_policy_cross_entropy

    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["board"])
            policy = sparse_policy_cross_entropy(
                logits,
                batch["legal_indices"],
                batch["target_actions"],
                batch["target_probs"],
            )
            predictions = topk_move_predictions(logits, batch["legal_indices"], 3)
            best = batch["best_move"]
            size = batch["board"].shape[0]
            total += size
            top1 += int((predictions[:, 0] == best).sum().item())
            top3 += int((predictions == best.unsqueeze(1)).any(dim=1).sum().item())
            policy_sum += float(policy.item()) * size
    return {
        "val_top1": top1 / total,
        "val_top3": top3 / total,
        "val_policy_ce": policy_sum / total,
    }


def latency_distribution(
    model: nn.Module,
    sample: torch.Tensor,
    repeats: int,
    trials: int,
    cpu_threads: int,
) -> dict[str, float]:
    values = [
        measure_latency_ms(
            model,
            sample,
            torch.device("cpu"),
            repeats,
            cpu_threads,
        )
        for _ in range(trials)
    ]
    return {
        "cpu_latency_ms": statistics.median(values),
        "cpu_latency_q1_ms": float(np.percentile(values, 25)),
        "cpu_latency_q3_ms": float(np.percentile(values, 75)),
        "cpu_latency_trials": json.dumps(values),
    }


def evaluate_condition(
    model: nn.Module,
    val_data: StockfishComparisonDataset,
    test_data: StockfishComparisonDataset,
    sample: torch.Tensor,
    seed: int,
    condition: str,
    family: str,
    method: str,
    checkpoint: Path,
    actual_mb: float,
    sparse_mb: float | None,
    batch_size: int,
    latency_repeats: int,
    latency_trials: int,
    cpu_threads: int,
    base_params: int | None = None,
    base_prunable_params: int | None = None,
    base_nonzero_prunable_params: int | None = None,
    train_seconds: float = 0.0,
    peak_gpu_mb: float = 0.0,
    epochs_completed: int = 0,
    sparsity_target: float | None = None,
    gpu_model: nn.Module | None = None,
    gpu_device: torch.device | None = None,
) -> dict[str, Any]:
    row = evaluate_model(
        model,
        test_data,
        condition,
        family,
        method,
        checkpoint,
        actual_mb,
        sparse_mb,
        batch_size,
        1,
        cpu_threads,
        base_params=base_params,
        base_prunable_params=base_prunable_params,
        base_nonzero_prunable_params=base_nonzero_prunable_params,
        train_seconds=train_seconds,
        peak_gpu_mb=peak_gpu_mb,
        epochs_completed=epochs_completed,
        sparsity_target=sparsity_target,
    )
    row.update(validation_quality(model, val_data, batch_size))
    row.update(
        latency_distribution(
            model,
            sample,
            latency_repeats,
            latency_trials,
            cpu_threads,
        )
    )
    row["seed"] = seed
    if gpu_model is not None and gpu_device is not None:
        attach_gpu_latency(row, gpu_model, sample, gpu_device, latency_repeats)
    return row


def aggregate_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    numeric = (
        "top1",
        "top3",
        "policy_ce",
        "value_rmse",
        "value_pearson",
        "actual_model_mb",
        "actual_sparsity",
        "sparse_estimated_mb",
        "val_top1",
        "val_policy_ce",
        "gpu_latency_ms",
        "peak_gpu_mb",
        "train_seconds",
    )
    for condition, group in rows.groupby("condition", sort=False):
        record: dict[str, Any] = {
            "condition": condition,
            "family": group.iloc[0]["family"],
            "method": group.iloc[0]["method"],
            "seeds": int(group["seed"].nunique()),
            "params": float(group["params"].mean()),
            "prunable_params": float(group["prunable_params"].mean()),
            "nonzero_prunable_params": float(group["nonzero_prunable_params"].mean()),
            "sparsity_target": group.iloc[0]["sparsity_target"],
            "cpu_latency_ms": float(group["cpu_latency_ms"].median()),
            "cpu_latency_q1_ms": float(group["cpu_latency_ms"].quantile(0.25)),
            "cpu_latency_q3_ms": float(group["cpu_latency_ms"].quantile(0.75)),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                record[f"{column}_mean"] = float("nan")
                record[f"{column}_std"] = float("nan")
                record[f"{column}_ci95"] = float("nan")
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{column}_mean"] = mean
            record[f"{column}_std"] = std
            record[f"{column}_ci95"] = 1.96 * std / math.sqrt(len(values))
        records.append(record)
    return pd.DataFrame(records)


def pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, row in frame.iterrows():
        dominated = (
            (frame["cpu_latency_ms"] <= row["cpu_latency_ms"])
            & (frame["top1_mean"] >= row["top1_mean"])
            & (
                (frame["cpu_latency_ms"] < row["cpu_latency_ms"])
                | (frame["top1_mean"] > row["top1_mean"])
            )
        ).any()
        keep.append(not dominated)
    return frame[pd.Series(keep, index=frame.index)].sort_values("cpu_latency_ms")


def label_for(condition: str) -> str:
    labels = {
        "teacher_fp32": "Teacher FP32",
        "teacher_int8": "Teacher INT8",
        "teacher_control": "Teacher control",
        "student_direct": "Direct student",
        "student_distilled": "Distilled student",
        "student_distilled_int8": "Distilled + INT8",
        "student_distilled_control": "Student control",
    }
    if condition in labels:
        return labels[condition]
    return condition.replace("teacher_pruned_", "Teacher prune ").replace(
        "student_distilled_pruned_", "Student prune "
    ) + ("%" if condition.endswith(("25", "50", "75")) else "")


def validation_selected_condition(
    aggregate: pd.DataFrame,
    prefix: str,
) -> str:
    candidates = aggregate[aggregate["condition"].str.startswith(prefix)].copy()
    candidates = candidates.sort_values(
        ["val_top1_mean", "actual_model_mb_mean"],
        ascending=[False, True],
    )
    return str(candidates.iloc[0]["condition"])


def poster_assets(aggregate: pd.DataFrame, poster_dir: Path) -> dict[str, str]:
    best_combined = validation_selected_condition(
        aggregate,
        "student_distilled_",
    )
    if best_combined == "student_distilled_control":
        alternatives = aggregate[
            aggregate["condition"].isin(
                [
                    "student_distilled_int8",
                    "student_distilled_pruned_25",
                    "student_distilled_pruned_50",
                    "student_distilled_pruned_75",
                ]
            )
        ].sort_values(["val_top1_mean", "actual_model_mb_mean"], ascending=[False, True])
        best_combined = str(alternatives.iloc[0]["condition"])

    comparison_conditions = [
        "teacher_fp32",
        "teacher_int8",
        "student_direct",
        "student_distilled",
        best_combined,
    ]
    selected = aggregate.set_index("condition").loc[comparison_conditions].reset_index()
    labels = [label_for(value) for value in selected["condition"]]
    colors = ["#4c78a8", "#f58518", "#72b7b2", "#54a24b", "#b279a2"]
    y = np.arange(len(selected))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    panels = (
        ("top1_mean", "top1_ci95", "Top-1 agreement", lambda value: f"{100 * value:.1f}%"),
        ("actual_model_mb_mean", None, "Serialized size (MB)", lambda value: f"{value:.1f}"),
        ("cpu_latency_ms", None, "CPU latency (ms)", lambda value: f"{value:.2f}"),
    )
    for axis, (column, error_column, title, formatter) in zip(axes, panels):
        errors = selected[error_column] if error_column else None
        axis.barh(y, selected[column], xerr=errors, color=colors, alpha=0.9, capsize=3)
        axis.set_title(title)
        axis.grid(True, axis="x", alpha=0.25)
        maximum = float(selected[column].max())
        for index, value in enumerate(selected[column]):
            axis.text(
                float(value) + maximum * 0.025,
                index,
                formatter(float(value)),
                va="center",
                fontsize=9,
            )
        axis.set_xlim(0, maximum * 1.22)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Compression methods: accuracy, storage, and speed", fontsize=16)
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_1_compression_method_comparison")

    plot = aggregate.copy()
    frontier = pareto_frontier(plot)
    method_colors = {
        "baseline": "#4c78a8",
        "quantization": "#f58518",
        "pruning": "#e45756",
        "direct_training": "#72b7b2",
        "distillation": "#54a24b",
        "combined": "#b279a2",
        "control": "#9d9da1",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for method, group in plot.groupby("method"):
        xerr = np.vstack(
            [
                group["cpu_latency_ms"] - group["cpu_latency_q1_ms"],
                group["cpu_latency_q3_ms"] - group["cpu_latency_ms"],
            ]
        )
        ax.errorbar(
            group["cpu_latency_ms"],
            group["top1_mean"],
            xerr=xerr,
            yerr=group["top1_ci95"],
            fmt="o",
            capsize=3,
            markersize=7,
            color=method_colors.get(method, "#666666"),
            label=method.replace("_", " ").title(),
        )
    ax.plot(
        frontier["cpu_latency_ms"],
        frontier["top1_mean"],
        linestyle="--",
        linewidth=1.8,
        color="#222222",
        label="Pareto frontier",
    )
    annotations = {
        "teacher_fp32",
        "teacher_int8",
        "student_direct",
        "student_distilled",
        best_combined,
        *frontier["condition"].tolist(),
    }
    for _, row in plot.iterrows():
        if row["condition"] in annotations:
            ax.annotate(
                label_for(str(row["condition"])),
                (row["cpu_latency_ms"], row["top1_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Median CPU latency (ms / position, one thread)")
    ax.set_ylabel("Mean top-1 Stockfish move agreement")
    ax.set_title("Accuracy vs practical CPU inference cost")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_2_accuracy_vs_cpu_latency")

    best_teacher_pruned = validation_selected_condition(aggregate, "teacher_pruned_")
    table_conditions = [
        "teacher_fp32",
        "teacher_int8",
        best_teacher_pruned,
        "student_direct",
        "student_distilled",
        best_combined,
    ]
    table = aggregate.set_index("condition").loc[table_conditions].reset_index()
    teacher_size = float(
        aggregate[aggregate["condition"] == "teacher_fp32"].iloc[0][
            "actual_model_mb_mean"
        ]
    )
    display = pd.DataFrame(
        {
            "Model": table["condition"].map(label_for),
            "Top-1 (95% CI)": [
                f"{100 * mean:.2f}% +/- {100 * ci:.2f}%"
                for mean, ci in zip(table["top1_mean"], table["top1_ci95"])
            ],
            "Value RMSE": table["value_rmse_mean"].map(lambda value: f"{value:.3f}"),
            "Size (MB)": table["actual_model_mb_mean"].map(lambda value: f"{value:.2f}"),
            "Size reduction": table["actual_model_mb_mean"].map(
                lambda value: f"{100 * (teacher_size - value) / teacher_size:.1f}%"
            ),
            "CPU ms": table["cpu_latency_ms"].map(lambda value: f"{value:.2f}"),
            "Sparsity": table["actual_sparsity_mean"].map(
                lambda value: f"{100 * value:.1f}%"
            ),
        }
    )
    poster_dir.mkdir(parents=True, exist_ok=True)
    display.to_csv(poster_dir / "poster_compact_results_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.axis("off")
    rendered = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1, 1.5)
    for column in range(len(display.columns)):
        rendered[(0, column)].set_facecolor("#dce8ed")
        rendered[(0, column)].set_text_props(weight="bold")
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_3_compact_results_table")
    return {
        "best_combined": best_combined,
        "best_teacher_pruned": best_teacher_pruned,
    }


def diagnostic_assets(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    sweep: pd.DataFrame,
    diagnostic_dir: Path,
) -> None:
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for family, prefix, color in (
        ("Teacher", "teacher_", "#4c78a8"),
        ("Distilled student", "student_distilled_", "#f58518"),
    ):
        conditions = [
            "teacher_control" if family == "Teacher" else "student_distilled_control",
            f"{prefix}pruned_25",
            f"{prefix}pruned_50",
            f"{prefix}pruned_75",
        ]
        subset = aggregate.set_index("condition").loc[conditions]
        ax.errorbar(
            [0, 0.25, 0.50, 0.75],
            subset["top1_mean"],
            yerr=subset["top1_ci95"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=family,
            color=color,
        )
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Mean top-1 agreement")
    ax.set_title("Pruning tolerance after matched fine-tuning")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, diagnostic_dir / "accuracy_vs_pruning_sparsity", 180)

    pivot = sweep.pivot_table(
        index="alpha",
        columns="temperature",
        values="val_policy_ce",
        aggfunc="mean",
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    image = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Teacher-loss weight (alpha)")
    ax.set_title("Validation policy loss for distillation sweep")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            ax.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if pivot.iloc[row, column] > pivot.values.mean() else "black",
            )
    fig.colorbar(image, ax=ax, label="Mean validation policy CE")
    fig.tight_layout()
    save_figure(fig, diagnostic_dir / "distillation_validation_heatmap", 180)

    key_conditions = [
        "teacher_fp32",
        "teacher_int8",
        "student_direct",
        "student_distilled",
        "student_distilled_int8",
    ]
    groups = [
        per_seed[per_seed["condition"] == condition]["top1"].to_numpy()
        for condition in key_conditions
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=[label_for(value) for value in key_conditions])
    ax.set_ylabel("Held-out top-1 agreement")
    ax.set_title("Variation across random seeds")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, diagnostic_dir / "top1_distribution_across_seeds", 180)


def train_seed_models(
    seed: int,
    alphas: tuple[float, ...],
    temperatures: tuple[float, ...],
    train_data: StockfishComparisonDataset,
    val_data: StockfishComparisonDataset,
    device: torch.device,
    checkpoint_root: Path,
    history_root: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    wandb_enabled: bool,
    artifact_name: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    seed_root = checkpoint_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, Any]] = []
    wandb_urls: dict[str, str] = {}

    torch.manual_seed(seed)
    teacher_path = seed_root / "teacher_fp32.pt"
    if teacher_path.exists():
        teacher_result = checkpoint_result(teacher_path, device)
    else:
        teacher_result = train_model(
            build_comparison_model("teacher"),
            train_data,
            val_data,
            device,
            batch_size,
            epochs,
            patience,
            learning_rate,
            1.0e-4,
            seed,
            "stockfish",
            None,
            f"{artifact_name}_seed{seed}_teacher",
            wandb_enabled,
            wandb_group="comparison_v4",
        )
        save_checkpoint(
            teacher_result.model,
            "teacher",
            teacher_path,
            training_metadata(teacher_result),
        )
        pd.DataFrame(teacher_result.history).to_csv(
            history_root / f"seed_{seed}_teacher.csv",
            index=False,
        )
    teacher = teacher_result.model.to(device).eval()
    wandb_urls[f"seed_{seed}_teacher"] = teacher_result.wandb_url

    torch.manual_seed(seed + 1000)
    initial_student = deepcopy(build_comparison_model("student").state_dict())
    direct_path = seed_root / "student_direct.pt"
    if direct_path.exists():
        direct_result = checkpoint_result(direct_path, device)
    else:
        direct_model = build_comparison_model("student")
        direct_model.load_state_dict(initial_student)
        direct_result = train_model(
            direct_model,
            train_data,
            val_data,
            device,
            batch_size,
            epochs,
            patience,
            learning_rate,
            1.0e-4,
            seed,
            "stockfish",
            None,
            f"{artifact_name}_seed{seed}_direct",
            wandb_enabled,
            wandb_group="comparison_v4",
        )
        save_checkpoint(
            direct_result.model,
            "student",
            direct_path,
            training_metadata(direct_result),
        )
        pd.DataFrame(direct_result.history).to_csv(
            history_root / f"seed_{seed}_direct.csv",
            index=False,
        )
    wandb_urls[f"seed_{seed}_direct"] = direct_result.wandb_url

    for alpha in alphas:
        for temperature in temperatures:
            tag = f"a{alpha:.2f}_t{temperature:.1f}".replace(".", "p")
            candidate_path = seed_root / f"student_distilled_{tag}.pt"
            if candidate_path.exists():
                result = checkpoint_result(candidate_path, device)
                _, metadata = load_checkpoint(candidate_path, torch.device("cpu"))
                quality = {
                    "val_top1": float(metadata.get("val_top1", float("nan"))),
                    "val_top3": float(metadata.get("val_top3", float("nan"))),
                    "val_policy_ce": float(metadata.get("val_policy_ce", float("nan"))),
                }
                if not math.isfinite(quality["val_policy_ce"]):
                    quality = validation_quality(result.model, val_data, batch_size)
            else:
                model = build_comparison_model("student")
                model.load_state_dict(initial_student)
                result = train_model(
                    model,
                    train_data,
                    val_data,
                    device,
                    batch_size,
                    epochs,
                    patience,
                    learning_rate,
                    1.0e-4,
                    seed,
                    "distill",
                    teacher,
                    f"{artifact_name}_seed{seed}_{tag}",
                    wandb_enabled,
                    distill_alpha=alpha,
                    distill_temperature=temperature,
                    wandb_group="comparison_v4",
                )
                quality = validation_quality(result.model, val_data, batch_size)
                save_checkpoint(
                    result.model,
                    "student",
                    candidate_path,
                    training_metadata(
                        result,
                        alpha=alpha,
                        temperature=temperature,
                        **quality,
                    ),
                )
                pd.DataFrame(result.history).to_csv(
                    history_root / f"seed_{seed}_{tag}.csv",
                    index=False,
                )
            wandb_urls[f"seed_{seed}_{tag}"] = result.wandb_url
            sweep_rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "temperature": temperature,
                    **quality,
                    "best_val_objective": result.best_val_loss,
                    "best_epoch": result.best_epoch,
                    "epochs_completed": result.epochs_completed,
                    "train_seconds": result.seconds,
                    "checkpoint": str(candidate_path),
                }
            )
    teacher.cpu()
    return sweep_rows, wandb_urls


def condition_rows_for_seed(
    seed: int,
    alpha: float,
    temperature: float,
    val_data: StockfishComparisonDataset,
    test_data: StockfishComparisonDataset,
    train_data: StockfishComparisonDataset,
    device: torch.device,
    checkpoint_root: Path,
    batch_size: int,
    prune_learning_rate: float,
    latency_repeats: int,
    latency_trials: int,
    cpu_threads: int,
) -> list[dict[str, Any]]:
    seed_root = checkpoint_root / f"seed_{seed}"
    teacher_path = seed_root / "teacher_fp32.pt"
    direct_path = seed_root / "student_direct.pt"
    tag = f"a{alpha:.2f}_t{temperature:.1f}".replace(".", "p")
    distilled_path = seed_root / f"student_distilled_{tag}.pt"
    teacher_result = checkpoint_result(teacher_path, device)
    direct_result = checkpoint_result(direct_path, device)
    distilled_result = checkpoint_result(distilled_path, device)
    teacher = teacher_result.model
    direct = direct_result.model
    distilled = distilled_result.model
    sample = test_data[0]["board"].unsqueeze(0)
    rows: list[dict[str, Any]] = []

    for condition, family, method, model, path, result in (
        ("teacher_fp32", "teacher", "baseline", teacher, teacher_path, teacher_result),
        ("student_direct", "student", "direct_training", direct, direct_path, direct_result),
        (
            "student_distilled",
            "student",
            "distillation",
            distilled,
            distilled_path,
            distilled_result,
        ),
    ):
        rows.append(
            evaluate_condition(
                model,
                val_data,
                test_data,
                sample,
                seed,
                condition,
                family,
                method,
                path,
                path.stat().st_size / (1024 * 1024),
                None,
                batch_size,
                latency_repeats,
                latency_trials,
                cpu_threads,
                train_seconds=result.seconds,
                peak_gpu_mb=result.peak_gpu_mb,
                epochs_completed=result.epochs_completed,
                gpu_model=model,
                gpu_device=device,
            )
        )

    for base_name, family, base_model, base_path, objective in (
        ("teacher", "teacher", teacher, teacher_path, "stockfish"),
        ("student_distilled", "student", distilled, distilled_path, "distill"),
    ):
        base_model = base_model.cpu().eval()
        base_nonzero, base_prunable, _ = parameter_sparsity(base_model)
        quantized = dynamic_quantize(base_model)
        quantized_path = seed_root / f"{base_name}_int8.pt"
        quantized_mb = serialized_size_mb(quantized, quantized_path, object_save=True)
        quantized = torch.load(quantized_path, map_location="cpu", weights_only=False)
        rows.append(
            evaluate_condition(
                quantized,
                val_data,
                test_data,
                sample,
                seed,
                f"{base_name}_int8",
                family,
                "quantization" if family == "teacher" else "combined",
                quantized_path,
                quantized_mb,
                None,
                batch_size,
                latency_repeats,
                latency_trials,
                cpu_threads,
                base_params=count_parameters(base_model),
                base_prunable_params=base_prunable,
                base_nonzero_prunable_params=base_nonzero,
            )
        )

        teacher_for_distill = teacher.to(device) if objective == "distill" else None
        control = deepcopy(base_model).to(device)
        control_seconds = fine_tune_one_epoch(
            control,
            train_data,
            device,
            batch_size,
            prune_learning_rate,
            seed + 1,
            objective,
            teacher_for_distill,
            alpha,
            temperature,
        )
        control.cpu()
        control_path = seed_root / f"{base_name}_control.pt"
        control_mb = serialized_size_mb(control, control_path)
        control_condition = (
            "teacher_control" if family == "teacher" else "student_distilled_control"
        )
        rows.append(
            evaluate_condition(
                control,
                val_data,
                test_data,
                sample,
                seed,
                control_condition,
                family,
                "control",
                control_path,
                control_mb,
                None,
                batch_size,
                latency_repeats,
                latency_trials,
                cpu_threads,
                train_seconds=control_seconds,
                epochs_completed=1,
            )
        )

        for sparsity in (0.25, 0.50, 0.75):
            pruned = deepcopy(base_model).to(device)
            apply_global_pruning(pruned, sparsity)
            prune_seconds = fine_tune_one_epoch(
                pruned,
                train_data,
                device,
                batch_size,
                prune_learning_rate,
                seed + int(100 * sparsity),
                objective,
                teacher_for_distill,
                alpha,
                temperature,
            )
            remove_pruning(pruned)
            pruned.cpu()
            percentage = int(100 * sparsity)
            pruned_path = seed_root / f"{base_name}_pruned_{percentage}.pt"
            pruned_mb = serialized_size_mb(pruned, pruned_path)
            condition = (
                f"teacher_pruned_{percentage}"
                if family == "teacher"
                else f"student_distilled_pruned_{percentage}"
            )
            rows.append(
                evaluate_condition(
                    pruned,
                    val_data,
                    test_data,
                    sample,
                    seed,
                    condition,
                    family,
                    "pruning" if family == "teacher" else "combined",
                    pruned_path,
                    pruned_mb,
                    estimated_sparse_payload_mb(pruned),
                    batch_size,
                    latency_repeats,
                    latency_trials,
                    cpu_threads,
                    train_seconds=prune_seconds,
                    epochs_completed=1,
                    sparsity_target=sparsity,
                )
            )
        if teacher_for_distill is not None:
            teacher_for_distill.cpu()
    return rows


def log_summary_wandb(
    enabled: bool,
    aggregate: pd.DataFrame,
    sweep: pd.DataFrame,
    poster_dir: Path,
    selected_alpha: float,
    selected_temperature: float,
) -> str:
    run = init_wandb(
        enabled,
        "comparison_v4_summary",
        "summary",
        {
            "selected_alpha": selected_alpha,
            "selected_temperature": selected_temperature,
            "seeds": int(aggregate["seeds"].max()),
        },
        group="comparison_v4",
    )
    if run is None:
        return ""
    import wandb

    run.log(
        {
            "aggregate_metrics": wandb.Table(dataframe=aggregate),
            "distillation_sweep": wandb.Table(dataframe=sweep),
        }
    )
    for path in sorted(poster_dir.glob("*.png")):
        run.log({path.stem: wandb.Image(str(path))})
    url = run.url or ""
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scaled multi-seed compression comparison.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--artifact-name", default="comparison_v4")
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    parser.add_argument(
        "--temperatures",
        default=",".join(str(value) for value in DEFAULT_TEMPERATURES),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--prune-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--latency-repeats", type=int, default=300)
    parser.add_argument("--latency-trials", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-runtime-minutes", type=int, default=210)
    parser.add_argument("--minimum-seeds", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    deadline = started + args.max_runtime_minutes * 60
    seeds = tuple(int(value) for value in parse_numbers(args.seeds, int))
    alphas = tuple(float(value) for value in parse_numbers(args.alphas, float))
    temperatures = tuple(float(value) for value in parse_numbers(args.temperatures, float))
    if args.smoke:
        seeds = seeds[:2]
        alphas = (alphas[0], alphas[-1])
        temperatures = (temperatures[0], temperatures[-1])
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.latency_repeats = min(args.latency_repeats, 20)
        args.latency_trials = min(args.latency_trials, 2)
        args.minimum_seeds = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the scaled experiment.")
    output_root = ROOT / "outputs" / args.artifact_name
    checkpoint_root = ROOT / "checkpoints" / args.artifact_name
    figure_root = ROOT / "figures" / args.artifact_name
    history_root = output_root / "histories"
    poster_dir = figure_root / "poster"
    diagnostic_dir = figure_root / "diagnostics"
    for path in (output_root, checkpoint_root, poster_dir, diagnostic_dir):
        path.mkdir(parents=True, exist_ok=True)

    limits = {"train": 400, "val": 50, "test": 50} if args.smoke else {}
    log(f"Loading datasets for scaled experiment on {device}")
    train_data = StockfishComparisonDataset(
        args.split_root / "train.jsonl",
        limit=limits.get("train"),
    )
    val_data = StockfishComparisonDataset(
        args.split_root / "val.jsonl",
        limit=limits.get("val"),
    )
    test_data = StockfishComparisonDataset(
        args.split_root / "test.jsonl",
        limit=limits.get("test"),
    )
    overlaps = verify_split_isolation(train_data, val_data, test_data)
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "train_positions": len(train_data),
        "val_positions": len(val_data),
        "test_positions": len(test_data),
        "split_overlaps": overlaps,
        "seeds_requested": seeds,
        "alphas": alphas,
        "temperatures": temperatures,
        "arguments": vars(args),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    all_sweep_rows: list[dict[str, Any]] = []
    wandb_urls: dict[str, str] = {}
    completed_seeds: list[int] = []
    for seed in seeds:
        if (
            time.perf_counter() >= deadline
            and len(completed_seeds) >= args.minimum_seeds
        ):
            log("Runtime launch deadline reached; aggregating completed seeds.")
            break
        log(f"Training seed {seed}")
        sweep_rows, seed_urls = train_seed_models(
            seed,
            alphas,
            temperatures,
            train_data,
            val_data,
            device,
            checkpoint_root,
            history_root,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            not args.no_wandb,
            args.artifact_name,
        )
        all_sweep_rows.extend(sweep_rows)
        wandb_urls.update(seed_urls)
        completed_seeds.append(seed)
        pd.DataFrame(all_sweep_rows).to_csv(
            output_root / "distillation_sweep.csv",
            index=False,
        )

    if len(completed_seeds) < args.minimum_seeds:
        raise RuntimeError(
            f"Only {len(completed_seeds)} seeds completed; minimum is {args.minimum_seeds}."
        )
    sweep = pd.DataFrame(all_sweep_rows)
    selection = (
        sweep.groupby(["alpha", "temperature"], as_index=False)["val_policy_ce"]
        .mean()
        .sort_values(["val_policy_ce", "alpha", "temperature"])
        .iloc[0]
    )
    selected_alpha = float(selection["alpha"])
    selected_temperature = float(selection["temperature"])
    log(
        "Selected distillation settings on validation: "
        f"alpha={selected_alpha:.2f}, temperature={selected_temperature:.1f}"
    )
    (output_root / "selected_distillation.json").write_text(
        json.dumps(
            {
                "alpha": selected_alpha,
                "temperature": selected_temperature,
                "selection_metric": "mean validation Stockfish policy cross-entropy",
                "test_metrics_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for seed in completed_seeds:
        log(f"Evaluating and compressing seed {seed}")
        rows.extend(
            condition_rows_for_seed(
                seed,
                selected_alpha,
                selected_temperature,
                val_data,
                test_data,
                train_data,
                device,
                checkpoint_root,
                args.batch_size,
                args.prune_learning_rate,
                args.latency_repeats,
                args.latency_trials,
                args.cpu_threads,
            )
        )
        pd.DataFrame(rows).to_csv(output_root / "per_seed_metrics.csv", index=False)

    per_seed = pd.DataFrame(rows)
    aggregate = aggregate_metrics(per_seed)
    aggregate.to_csv(output_root / "aggregate_metrics.csv", index=False)
    selections = poster_assets(aggregate, poster_dir)
    diagnostic_assets(per_seed, aggregate, sweep, diagnostic_dir)
    summary_url = log_summary_wandb(
        not args.no_wandb,
        aggregate,
        sweep,
        poster_dir,
        selected_alpha,
        selected_temperature,
    )
    wandb_urls["comparison_v4_summary"] = summary_url
    (output_root / "wandb_runs.json").write_text(
        json.dumps(wandb_urls, indent=2),
        encoding="utf-8",
    )
    summary = {
        "completed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_seeds": completed_seeds,
        "conditions_per_seed": int(per_seed["condition"].nunique()),
        "selected_alpha": selected_alpha,
        "selected_temperature": selected_temperature,
        **selections,
        "per_seed_metrics": str(output_root / "per_seed_metrics.csv"),
        "aggregate_metrics": str(output_root / "aggregate_metrics.csv"),
        "poster_dir": str(poster_dir),
        "diagnostic_dir": str(diagnostic_dir),
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    log(
        f"Completed scaled experiment with {len(completed_seeds)} seeds in "
        f"{summary['elapsed_seconds'] / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
