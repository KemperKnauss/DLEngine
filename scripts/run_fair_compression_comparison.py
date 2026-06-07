from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import nn
from torch.utils.data import DataLoader

from chess_student.comparison import (
    StockfishComparisonDataset,
    build_comparison_model,
    count_parameters,
    distillation_loss,
    estimated_sparse_payload_mb,
    parameter_sparsity,
    pearson,
    prunable_modules,
    sparse_policy_cross_entropy,
    stockfish_loss,
    topk_move_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT / "data" / "overnight" / "final_v2" / "splits"


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    seconds: float
    peak_gpu_mb: float
    epochs_completed: int
    best_val_loss: float
    best_epoch: int
    wandb_url: str


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def model_checkpoint(model: nn.Module, kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }


def save_checkpoint(model: nn.Module, kind: str, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_checkpoint(model, kind, metadata), path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model = build_comparison_model(checkpoint["kind"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint.get("metadata", {})


def make_loader(
    dataset: StockfishComparisonDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def init_wandb(
    enabled: bool,
    run_name: str,
    job_type: str,
    config: dict[str, Any],
):
    if not enabled:
        return None
    import wandb

    return wandb.init(
        project="dlengine-chess-compression",
        group="comparison_v3",
        name=run_name,
        job_type=job_type,
        config=config,
        reinit=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    objective: str,
    teacher: nn.Module | None,
    distill_alpha: float,
    distill_temperature: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()

    totals = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "teacher_policy": 0.0,
        "teacher_value": 0.0,
    }
    examples = 0
    amp_enabled = training and device.type == "cuda"

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, value_pred = model(batch["board"])
            if objective == "distill":
                if teacher is None:
                    raise ValueError("Distillation objective requires a teacher.")
                with torch.no_grad():
                    teacher_logits, teacher_value = teacher(batch["board"])
                loss, parts = distillation_loss(
                    logits,
                    value_pred,
                    teacher_logits,
                    teacher_value,
                    batch,
                    alpha=distill_alpha,
                    temperature=distill_temperature,
                )
            else:
                loss, parts = stockfish_loss(logits, value_pred, batch)

        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        size = batch["board"].shape[0]
        examples += size
        totals["loss"] += float(loss.detach().item()) * size
        for key in ("policy", "value", "teacher_policy", "teacher_value"):
            if key in parts:
                totals[key] += float(parts[key].detach().item()) * size

    return {key: value / max(1, examples) for key, value in totals.items()}


def train_model(
    model: nn.Module,
    train_data: StockfishComparisonDataset,
    val_data: StockfishComparisonDataset,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    objective: str,
    teacher: nn.Module | None,
    run_name: str,
    wandb_enabled: bool,
    distill_alpha: float = 0.7,
    distill_temperature: float = 2.0,
) -> TrainResult:
    torch.manual_seed(seed)
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device)
    train_loader = make_loader(train_data, batch_size, True, device, seed)
    val_loader = make_loader(val_data, batch_size, False, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    config = {
        "objective": objective,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "distill_alpha": distill_alpha,
        "distill_temperature": distill_temperature,
        "params": count_parameters(model),
        "train_positions": len(train_data),
        "val_positions": len(val_data),
    }
    wandb_run = init_wandb(wandb_enabled, run_name, "train", config)
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            objective,
            teacher,
            distill_alpha,
            distill_temperature,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                None,
                objective,
                teacher,
                distill_alpha,
                distill_temperature,
            )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        log(
            f"{run_name} epoch={epoch} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(row)

        if val_metrics["loss"] < best_val - 1.0e-5:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    wandb_url = ""
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "train_seconds": seconds,
                "peak_gpu_mb": peak_gpu_mb,
            }
        )
        wandb_url = wandb_run.url or ""
        wandb_run.finish()
    return TrainResult(
        model=model,
        history=history,
        seconds=seconds,
        peak_gpu_mb=peak_gpu_mb,
        epochs_completed=len(history),
        best_val_loss=best_val,
        best_epoch=best_epoch,
        wandb_url=wandb_url,
    )


def fine_tune_one_epoch(
    model: nn.Module,
    train_data: StockfishComparisonDataset,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    seed: int,
    objective: str,
    teacher: nn.Module | None,
) -> float:
    loader = make_loader(train_data, batch_size, True, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    metrics = run_epoch(
        model,
        loader,
        device,
        optimizer,
        scaler,
        objective,
        teacher,
        0.7,
        2.0,
    )
    log(f"fine_tune objective={objective} loss={metrics['loss']:.4f}")
    return time.perf_counter() - started


def apply_global_pruning(model: nn.Module, amount: float) -> None:
    prune.global_unstructured(
        prunable_modules(model),
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def remove_pruning(model: nn.Module) -> None:
    for module, name in prunable_modules(model):
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)


def dynamic_quantize(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        return torch.ao.quantization.quantize_dynamic(
            deepcopy(model).cpu().eval(),
            {nn.Linear},
            dtype=torch.qint8,
        )


def serialized_size_mb(model: nn.Module, path: Path, object_save: bool = False) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    if object_save:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def measure_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    repeats: int,
    cpu_threads: int,
) -> float:
    previous_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    model = model.to(device).eval()
    sample = sample[:1].to(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    if device.type == "cpu":
        torch.set_num_threads(previous_threads)
    return (time.perf_counter() - started) * 1000 / repeats


def evaluate_model(
    model: nn.Module,
    dataset: StockfishComparisonDataset,
    condition: str,
    family: str,
    method: str,
    checkpoint: Path,
    actual_mb: float,
    sparse_estimated_mb: float | None,
    batch_size: int,
    latency_repeats: int,
    cpu_threads: int,
    base_params: int | None = None,
    base_prunable_params: int | None = None,
    base_nonzero_prunable_params: int | None = None,
    train_seconds: float = 0.0,
    peak_gpu_mb: float = 0.0,
    epochs_completed: int = 0,
    sparsity_target: float | None = None,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = model.cpu().eval()
    loader = make_loader(dataset, batch_size, False, device, 7)
    total = 0
    top1 = 0
    top3 = 0
    policy_sum = 0.0
    value_predictions = []
    value_targets = []
    sample = None

    with torch.no_grad():
        for batch in loader:
            if sample is None:
                sample = batch["board"][:1]
            logits, value_pred = model(batch["board"])
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
            value_predictions.append(value_pred.float())
            value_targets.append(batch["value"].float())

    predicted = torch.cat(value_predictions)
    target = torch.cat(value_targets)
    mse = float(F.mse_loss(predicted, target).item())
    nonzero, prunable_total, actual_sparsity = parameter_sparsity(model)
    if base_prunable_params is not None:
        prunable_total = base_prunable_params
        nonzero = (
            base_nonzero_prunable_params
            if base_nonzero_prunable_params is not None
            else base_prunable_params
        )
        actual_sparsity = 1.0 - (nonzero / max(1, prunable_total))
    params = base_params if base_params is not None else count_parameters(model)
    cpu_latency = measure_latency_ms(
        model,
        sample,
        torch.device("cpu"),
        latency_repeats,
        cpu_threads,
    )
    return {
        "condition": condition,
        "family": family,
        "method": method,
        "checkpoint": str(checkpoint),
        "positions": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "policy_ce": policy_sum / total,
        "value_mse": mse,
        "value_rmse": math.sqrt(mse),
        "value_pearson": pearson(predicted, target),
        "params": params,
        "prunable_params": prunable_total,
        "nonzero_prunable_params": nonzero,
        "actual_sparsity": actual_sparsity,
        "sparsity_target": "" if sparsity_target is None else sparsity_target,
        "actual_model_mb": actual_mb,
        "sparse_estimated_mb": "" if sparse_estimated_mb is None else sparse_estimated_mb,
        "cpu_latency_ms": cpu_latency,
        "gpu_latency_ms": "",
        "peak_gpu_mb": peak_gpu_mb,
        "train_seconds": train_seconds,
        "epochs_completed": epochs_completed,
    }


def attach_gpu_latency(
    row: dict[str, Any],
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> None:
    if device.type != "cuda":
        return
    row["gpu_latency_ms"] = measure_latency_ms(model, sample, device, repeats, 1)
    model.cpu()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def log_evaluation_wandb(enabled: bool, row: dict[str, Any]) -> str:
    run = init_wandb(enabled, row["condition"], "evaluation", row)
    if run is None:
        return ""
    run.log({key: value for key, value in row.items() if isinstance(value, (int, float))})
    url = run.url or ""
    run.finish()
    return url


def short_label(condition: str) -> str:
    replacements = {
        "teacher_fp32": "Teacher FP32",
        "teacher_int8": "Teacher INT8",
        "teacher_control": "Teacher control",
        "student_direct": "Direct student",
        "student_distilled": "Distilled student",
        "student_distilled_int8": "Distilled + INT8",
        "student_distilled_control": "Student control",
    }
    if condition in replacements:
        return replacements[condition]
    return condition.replace("teacher_pruned_", "Teacher prune ").replace(
        "student_distilled_pruned_", "Student prune "
    ) + ("%" if condition.endswith(("25", "50", "75")) else "")


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    mask = []
    for _, row in frame.iterrows():
        dominated = (
            (frame["cpu_latency_ms"] <= row["cpu_latency_ms"])
            & (frame["top1"] >= row["top1"])
            & (
                (frame["cpu_latency_ms"] < row["cpu_latency_ms"])
                | (frame["top1"] > row["top1"])
            )
        ).any()
        mask.append(not dominated)
    return pd.Series(mask, index=frame.index)


def save_figure(fig: plt.Figure, base_path: Path, dpi: int = 300) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def poster_plots(frame: pd.DataFrame, poster_dir: Path) -> None:
    teacher = frame[frame["condition"] == "teacher_fp32"].iloc[0]
    teacher_mb = float(teacher["actual_model_mb"])
    teacher_top1 = float(teacher["top1"])

    teacher_pruned = frame[
        frame["condition"].str.startswith("teacher_pruned_")
    ].sort_values(["top1", "actual_model_mb"], ascending=[False, True])
    student_pruned = frame[
        frame["condition"].str.startswith("student_distilled_pruned_")
    ].sort_values(["top1", "actual_model_mb"], ascending=[False, True])
    graph_one_conditions = [
        "teacher_int8",
        str(teacher_pruned.iloc[0]["condition"]),
        "student_direct",
        "student_distilled",
        "student_distilled_int8",
        str(student_pruned.iloc[0]["condition"]),
    ]
    plot_frame = frame[
        frame["condition"].isin(graph_one_conditions)
    ].copy()
    plot_frame["size_reduction_pct"] = (
        100.0 * (teacher_mb - plot_frame["actual_model_mb"]) / teacher_mb
    )
    plot_frame["top1_delta_pp"] = 100.0 * (plot_frame["top1"] - teacher_top1)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = {
        "baseline": "#4c78a8",
        "quantization": "#f58518",
        "pruning": "#e45756",
        "direct_training": "#72b7b2",
        "distillation": "#54a24b",
        "combined": "#b279a2",
        "control": "#9d9da1",
    }
    label_offsets = {
        "teacher_int8": (6, 5),
        "student_direct": (6, 18),
        "student_distilled": (6, -18),
        "student_distilled_int8": (6, 6),
        str(teacher_pruned.iloc[0]["condition"]): (6, 5),
        str(student_pruned.iloc[0]["condition"]): (6, 0),
    }
    for _, row in plot_frame.iterrows():
        ax.scatter(
            row["size_reduction_pct"],
            row["top1_delta_pp"],
            s=70,
            color=colors.get(row["method"], "#666666"),
        )
        ax.annotate(
            short_label(row["condition"]),
            (row["size_reduction_pct"], row["top1_delta_pp"]),
            xytext=label_offsets.get(row["condition"], (6, 5)),
            textcoords="offset points",
            fontsize=7,
        )
        if row["method"] == "pruning" and pd.notna(row["sparse_estimated_mb"]):
            sparse_reduction = 100.0 * (
                teacher_mb - float(row["sparse_estimated_mb"])
            ) / teacher_mb
            ax.plot(
                [row["size_reduction_pct"], sparse_reduction],
                [row["top1_delta_pp"], row["top1_delta_pp"]],
                linestyle=":",
                color=colors["pruning"],
                alpha=0.6,
            )
            ax.scatter(
                sparse_reduction,
                row["top1_delta_pp"],
                s=55,
                facecolors="none",
                edgecolors=colors["pruning"],
                marker="D",
            )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Actual serialized size reduction vs FP32 teacher (%)")
    ax.set_ylabel("Top-1 agreement change vs FP32 teacher (percentage points)")
    ax.set_title("Compression gained vs accuracy retained")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_1_accuracy_loss_vs_size_reduction")

    latency = frame.sort_values("cpu_latency_ms").copy()
    frontier = latency[pareto_mask(latency)].sort_values("cpu_latency_ms")
    annotation_conditions = {
        "teacher_fp32",
        "teacher_int8",
        "student_direct",
        "student_distilled",
        "student_distilled_int8",
        str(teacher_pruned.iloc[0]["condition"]),
        str(student_pruned.iloc[0]["condition"]),
        *frontier["condition"].tolist(),
    }
    latency_label_offsets = {
        "teacher_fp32": (5, 5),
        "teacher_int8": (5, 5),
        "student_direct": (5, 5),
        "student_distilled": (5, 18),
        "student_distilled_int8": (5, 5),
        str(teacher_pruned.iloc[0]["condition"]): (5, 5),
        str(student_pruned.iloc[0]["condition"]): (5, -14),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for method, group in latency.groupby("method"):
        ax.scatter(
            group["cpu_latency_ms"],
            group["top1"],
            s=70,
            label=method.replace("_", " ").title(),
            color=colors.get(method),
        )
        for _, row in group.iterrows():
            if row["condition"] not in annotation_conditions:
                continue
            ax.annotate(
                short_label(row["condition"]),
                (row["cpu_latency_ms"], row["top1"]),
                xytext=latency_label_offsets.get(row["condition"], (5, 4)),
                textcoords="offset points",
                fontsize=7,
            )
    ax.plot(
        frontier["cpu_latency_ms"],
        frontier["top1"],
        color="#222222",
        linewidth=1.8,
        linestyle="--",
        label="Pareto frontier",
    )
    ax.set_xlabel("CPU latency (ms / position, one thread)")
    ax.set_ylabel("Top-1 Stockfish move agreement")
    ax.set_title("Accuracy vs practical CPU inference cost")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_2_accuracy_vs_cpu_latency")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for family, prefix, color in (
        ("Teacher", "teacher_", "#4c78a8"),
        ("Distilled student", "student_distilled_", "#f58518"),
    ):
        control_name = (
            "teacher_control" if family == "Teacher" else "student_distilled_control"
        )
        points = [(0.0, float(frame[frame["condition"] == control_name].iloc[0]["top1"]))]
        for sparsity in (25, 50, 75):
            condition = f"{prefix}pruned_{sparsity}"
            row = frame[frame["condition"] == condition]
            if not row.empty:
                points.append((sparsity / 100.0, float(row.iloc[0]["top1"])))
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=family, color=color)
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Top-1 Stockfish move agreement")
    ax.set_title("Pruning tolerance after matched fine-tuning")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_3_accuracy_vs_pruning_sparsity")


def compact_table(frame: pd.DataFrame, poster_dir: Path) -> None:
    teacher_pruned = frame[
        frame["condition"].str.startswith("teacher_pruned_")
    ].sort_values(["top1", "actual_model_mb"], ascending=[False, True])
    combined = frame[
        frame["condition"].str.startswith("student_distilled_")
        & ~frame["condition"].isin(["student_distilled", "student_distilled_control"])
    ].sort_values(["top1", "actual_model_mb"], ascending=[False, True])
    conditions = [
        "teacher_fp32",
        "teacher_int8",
        teacher_pruned.iloc[0]["condition"],
        "student_direct",
        "student_distilled",
        combined.iloc[0]["condition"],
    ]
    table = frame.set_index("condition").loc[conditions].reset_index()
    teacher_mb = float(frame[frame["condition"] == "teacher_fp32"].iloc[0]["actual_model_mb"])
    table["model"] = table["condition"].map(short_label)
    table["size_reduction_pct"] = 100.0 * (
        teacher_mb - table["actual_model_mb"]
    ) / teacher_mb
    output = table[
        [
            "condition",
            "model",
            "top1",
            "value_rmse",
            "actual_model_mb",
            "size_reduction_pct",
            "cpu_latency_ms",
            "actual_sparsity",
        ]
    ]
    poster_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(poster_dir / "poster_compact_results_table.csv", index=False)

    display = output.copy()
    display["top1"] = display["top1"].map(lambda value: f"{100 * value:.2f}%")
    display["value_rmse"] = display["value_rmse"].map(lambda value: f"{value:.3f}")
    display["actual_model_mb"] = display["actual_model_mb"].map(lambda value: f"{value:.2f}")
    display["size_reduction_pct"] = display["size_reduction_pct"].map(
        lambda value: f"{value:.1f}%"
    )
    display["cpu_latency_ms"] = display["cpu_latency_ms"].map(lambda value: f"{value:.2f}")
    display["actual_sparsity"] = display["actual_sparsity"].map(
        lambda value: f"{100 * value:.1f}%"
    )
    display = display[
        [
            "model",
            "top1",
            "value_rmse",
            "actual_model_mb",
            "size_reduction_pct",
            "cpu_latency_ms",
            "actual_sparsity",
        ]
    ]
    display.columns = [
        "Model",
        "Top-1",
        "Value RMSE",
        "Size (MB)",
        "Size reduction",
        "CPU ms",
        "Sparsity",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 2.7))
    ax.axis("off")
    rendered = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.45)
    for column in range(len(display.columns)):
        rendered[(0, column)].set_facecolor("#dce8ed")
        rendered[(0, column)].set_text_props(weight="bold")
    fig.tight_layout()
    save_figure(fig, poster_dir / "poster_compact_results_table")


def diagnostic_plots(
    frame: pd.DataFrame,
    histories: dict[str, list[dict[str, Any]]],
    diagnostic_dir: Path,
) -> None:
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_history = False
    for name, history in histories.items():
        if not history:
            continue
        history_frame = pd.DataFrame(history)
        ax.plot(history_frame["epoch"], history_frame["train_loss"], label=f"{name} train")
        ax.plot(
            history_frame["epoch"],
            history_frame["val_loss"],
            linestyle="--",
            label=f"{name} val",
        )
        plotted_history = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Method-specific objective loss")
    ax.set_title("Training and validation objectives")
    ax.grid(True, alpha=0.25)
    if plotted_history:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "Training histories unavailable for resumed checkpoints",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    save_figure(fig, diagnostic_dir / "training_validation_loss", 180)

    for x_column, filename, xlabel in (
        ("cpu_latency_ms", "value_rmse_vs_cpu_latency", "CPU latency (ms / position)"),
        ("actual_model_mb", "value_rmse_vs_model_size", "Serialized size (MB)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(frame[x_column], frame["value_rmse"], s=65)
        for _, row in frame.iterrows():
            ax.annotate(
                short_label(row["condition"]),
                (row[x_column], row["value_rmse"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value RMSE")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_figure(fig, diagnostic_dir / filename, 180)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = frame.sort_values("params")
    ax.bar(
        [short_label(value) for value in ordered["condition"]],
        ordered["nonzero_prunable_params"],
    )
    ax.set_ylabel("Nonzero prunable parameters")
    ax.tick_params(axis="x", rotation=55, labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, diagnostic_dir / "nonzero_parameter_comparison", 180)

    student_histories = {
        name: pd.DataFrame(history)
        for name, history in histories.items()
        if name in {"direct_student", "distilled_student"} and history
    }
    if student_histories:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, history_frame in student_histories.items():
            ax.plot(
                history_frame["epoch"],
                history_frame["val_policy"],
                marker="o",
                label=name.replace("_", " ").title(),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Stockfish policy cross-entropy")
        ax.set_title("Direct vs distilled student learning")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, diagnostic_dir / "direct_vs_distilled_learning", 180)

    gpu_frame = frame[
        frame["gpu_latency_ms"].notna() & (frame["gpu_latency_ms"] > 0)
    ].copy()
    if not gpu_frame.empty:
        gpu_frame = gpu_frame.sort_values("gpu_latency_ms")
        labels = [short_label(value) for value in gpu_frame["condition"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar(labels, gpu_frame["gpu_latency_ms"], color="#4c78a8")
        axes[0].set_ylabel("GPU latency (ms / position)")
        axes[0].set_title("Batch-one GPU latency")
        axes[1].bar(labels, gpu_frame["peak_gpu_mb"], color="#f58518")
        axes[1].set_ylabel("Peak training memory (MiB)")
        axes[1].set_title("Peak CUDA memory")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, diagnostic_dir / "gpu_latency_and_peak_memory", 180)


def verify_split_isolation(
    train_data: StockfishComparisonDataset,
    val_data: StockfishComparisonDataset,
    test_data: StockfishComparisonDataset,
) -> dict[str, int]:
    sets = {
        "train": set(train_data.game_ids),
        "val": set(val_data.game_ids),
        "test": set(test_data.game_ids),
    }
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Game-level split leakage detected: {overlaps}")
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fair Stockfish surrogate compression comparison.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--artifact-name", default="comparison_v3")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--prune-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-runtime-minutes", type=int, default=210)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    deadline = started + args.max_runtime_minutes * 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this experiment.")

    output_root = ROOT / "outputs" / args.artifact_name
    checkpoint_root = ROOT / "checkpoints" / args.artifact_name
    figure_root = ROOT / "figures" / args.artifact_name
    poster_dir = figure_root / "poster"
    diagnostic_dir = figure_root / "diagnostics"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    limits = {"train": 400, "val": 50, "test": 50} if args.smoke else {}
    log(f"Loading compact datasets on {device}")
    train_data = StockfishComparisonDataset(
        args.split_root / "train.jsonl", limit=limits.get("train")
    )
    val_data = StockfishComparisonDataset(
        args.split_root / "val.jsonl", limit=limits.get("val")
    )
    test_data = StockfishComparisonDataset(
        args.split_root / "test.jsonl", limit=limits.get("test")
    )
    overlaps = verify_split_isolation(train_data, val_data, test_data)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "train_positions": len(train_data),
        "val_positions": len(val_data),
        "test_positions": len(test_data),
        "split_overlaps": overlaps,
        "arguments": vars(args),
    }
    with (output_root / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, default=str)

    wandb_enabled = not args.no_wandb
    histories: dict[str, list[dict[str, Any]]] = {}
    wandb_urls: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    metrics_path = output_root / "final_metrics.csv"
    sample = test_data[0]["board"].unsqueeze(0)
    effective_epochs = 1 if args.smoke else args.epochs

    teacher_path = checkpoint_root / "teacher_fp32.pt"
    if teacher_path.exists():
        teacher, teacher_meta = load_checkpoint(teacher_path, device)
        teacher_result = TrainResult(
            teacher,
            [],
            float(teacher_meta.get("train_seconds", 0.0)),
            float(teacher_meta.get("peak_gpu_mb", 0.0)),
            int(teacher_meta.get("epochs_completed", 0)),
            float(teacher_meta.get("best_val_loss", 0.0)),
            int(teacher_meta.get("best_epoch", 0)),
            str(teacher_meta.get("wandb_url", "")),
        )
    else:
        teacher_result = train_model(
            build_comparison_model("teacher"),
            train_data,
            val_data,
            device,
            args.batch_size,
            effective_epochs,
            args.patience,
            args.learning_rate,
            1.0e-4,
            args.seed,
            "stockfish",
            None,
            f"{args.artifact_name}_teacher_fp32",
            wandb_enabled,
        )
        metadata = {
            "train_seconds": teacher_result.seconds,
            "peak_gpu_mb": teacher_result.peak_gpu_mb,
            "epochs_completed": teacher_result.epochs_completed,
            "best_val_loss": teacher_result.best_val_loss,
            "best_epoch": teacher_result.best_epoch,
            "wandb_url": teacher_result.wandb_url,
        }
        save_checkpoint(teacher_result.model, "teacher", teacher_path, metadata)
    teacher = teacher_result.model
    histories["teacher"] = teacher_result.history
    wandb_urls["teacher_fp32"] = teacher_result.wandb_url
    teacher_mb = teacher_path.stat().st_size / (1024 * 1024)
    teacher_row = evaluate_model(
        teacher,
        test_data,
        "teacher_fp32",
        "teacher",
        "baseline",
        teacher_path,
        teacher_mb,
        None,
        args.batch_size,
        args.latency_repeats,
        args.cpu_threads,
        train_seconds=teacher_result.seconds,
        peak_gpu_mb=teacher_result.peak_gpu_mb,
        epochs_completed=teacher_result.epochs_completed,
    )
    attach_gpu_latency(teacher_row, teacher, sample, device, args.latency_repeats)
    rows.append(teacher_row)
    write_rows(metrics_path, rows)

    student_initial = build_comparison_model("student").state_dict()
    direct_path = checkpoint_root / "student_direct.pt"
    if direct_path.exists():
        direct, direct_meta = load_checkpoint(direct_path, device)
        direct_result = TrainResult(
            direct, [], float(direct_meta.get("train_seconds", 0.0)),
            float(direct_meta.get("peak_gpu_mb", 0.0)),
            int(direct_meta.get("epochs_completed", 0)), 0.0, 0,
            str(direct_meta.get("wandb_url", "")),
        )
    else:
        direct_model = build_comparison_model("student")
        direct_model.load_state_dict(student_initial)
        direct_result = train_model(
            direct_model,
            train_data,
            val_data,
            device,
            args.batch_size,
            effective_epochs,
            args.patience,
            args.learning_rate,
            1.0e-4,
            args.seed,
            "stockfish",
            None,
            f"{args.artifact_name}_student_direct",
            wandb_enabled,
        )
        save_checkpoint(
            direct_result.model,
            "student",
            direct_path,
            {
                "train_seconds": direct_result.seconds,
                "peak_gpu_mb": direct_result.peak_gpu_mb,
                "epochs_completed": direct_result.epochs_completed,
                "wandb_url": direct_result.wandb_url,
            },
        )
    histories["direct_student"] = direct_result.history
    wandb_urls["student_direct"] = direct_result.wandb_url
    direct_mb = direct_path.stat().st_size / (1024 * 1024)
    direct_row = evaluate_model(
        direct_result.model,
        test_data,
        "student_direct",
        "student",
        "direct_training",
        direct_path,
        direct_mb,
        None,
        args.batch_size,
        args.latency_repeats,
        args.cpu_threads,
        train_seconds=direct_result.seconds,
        peak_gpu_mb=direct_result.peak_gpu_mb,
        epochs_completed=direct_result.epochs_completed,
    )
    attach_gpu_latency(direct_row, direct_result.model, sample, device, args.latency_repeats)
    rows.append(direct_row)
    write_rows(metrics_path, rows)

    distilled_path = checkpoint_root / "student_distilled.pt"
    teacher = teacher.to(device).eval()
    if distilled_path.exists():
        distilled, distilled_meta = load_checkpoint(distilled_path, device)
        distilled_result = TrainResult(
            distilled, [], float(distilled_meta.get("train_seconds", 0.0)),
            float(distilled_meta.get("peak_gpu_mb", 0.0)),
            int(distilled_meta.get("epochs_completed", 0)), 0.0, 0,
            str(distilled_meta.get("wandb_url", "")),
        )
    else:
        distilled_model = build_comparison_model("student")
        distilled_model.load_state_dict(student_initial)
        distilled_result = train_model(
            distilled_model,
            train_data,
            val_data,
            device,
            args.batch_size,
            effective_epochs,
            args.patience,
            args.learning_rate,
            1.0e-4,
            args.seed,
            "distill",
            teacher,
            f"{args.artifact_name}_student_distilled",
            wandb_enabled,
        )
        save_checkpoint(
            distilled_result.model,
            "student",
            distilled_path,
            {
                "train_seconds": distilled_result.seconds,
                "peak_gpu_mb": distilled_result.peak_gpu_mb,
                "epochs_completed": distilled_result.epochs_completed,
                "wandb_url": distilled_result.wandb_url,
            },
        )
    distilled = distilled_result.model
    histories["distilled_student"] = distilled_result.history
    wandb_urls["student_distilled"] = distilled_result.wandb_url
    distilled_mb = distilled_path.stat().st_size / (1024 * 1024)
    distilled_row = evaluate_model(
        distilled,
        test_data,
        "student_distilled",
        "student",
        "distillation",
        distilled_path,
        distilled_mb,
        None,
        args.batch_size,
        args.latency_repeats,
        args.cpu_threads,
        train_seconds=distilled_result.seconds,
        peak_gpu_mb=distilled_result.peak_gpu_mb,
        epochs_completed=distilled_result.epochs_completed,
    )
    attach_gpu_latency(distilled_row, distilled, sample, device, args.latency_repeats)
    rows.append(distilled_row)
    write_rows(metrics_path, rows)

    if time.perf_counter() > deadline:
        raise TimeoutError("Runtime budget reached after core model training.")

    for base_name, family, base_model, base_path, base_mb, base_objective in (
        ("teacher", "teacher", teacher.cpu(), teacher_path, teacher_mb, "stockfish"),
        (
            "student_distilled",
            "student",
            distilled.cpu(),
            distilled_path,
            distilled_mb,
            "distill",
        ),
    ):
        quantized = dynamic_quantize(base_model)
        quantized_path = checkpoint_root / f"{base_name}_int8.pt"
        quantized_mb = serialized_size_mb(quantized, quantized_path, object_save=True)
        quantized = torch.load(quantized_path, map_location="cpu", weights_only=False)
        base_nonzero, base_prunable, _ = parameter_sparsity(base_model)
        quantized_row = evaluate_model(
            quantized,
            test_data,
            f"{base_name}_int8",
            family,
            "quantization" if family == "teacher" else "combined",
            quantized_path,
            quantized_mb,
            None,
            args.batch_size,
            args.latency_repeats,
            args.cpu_threads,
            base_params=count_parameters(base_model),
            base_prunable_params=base_prunable,
            base_nonzero_prunable_params=base_nonzero,
        )
        rows.append(quantized_row)
        write_rows(metrics_path, rows)

        control = deepcopy(base_model).to(device)
        teacher_for_objective = teacher.to(device) if base_objective == "distill" else None
        control_seconds = fine_tune_one_epoch(
            control,
            train_data,
            device,
            args.batch_size,
            args.prune_learning_rate,
            args.seed + 1,
            base_objective,
            teacher_for_objective,
        )
        control.cpu()
        control_path = checkpoint_root / f"{base_name}_control.pt"
        control_mb = serialized_size_mb(control, control_path)
        control_condition = (
            "teacher_control"
            if base_name == "teacher"
            else "student_distilled_control"
        )
        control_row = evaluate_model(
            control,
            test_data,
            control_condition,
            family,
            "control",
            control_path,
            control_mb,
            None,
            args.batch_size,
            args.latency_repeats,
            args.cpu_threads,
            train_seconds=control_seconds,
            epochs_completed=1,
        )
        rows.append(control_row)
        write_rows(metrics_path, rows)

        for sparsity in (0.25, 0.50, 0.75):
            pruned = deepcopy(base_model).to(device)
            apply_global_pruning(pruned, sparsity)
            prune_seconds = fine_tune_one_epoch(
                pruned,
                train_data,
                device,
                args.batch_size,
                args.prune_learning_rate,
                args.seed + int(sparsity * 100),
                base_objective,
                teacher_for_objective,
            )
            remove_pruning(pruned)
            pruned.cpu()
            percentage = int(sparsity * 100)
            pruned_path = checkpoint_root / f"{base_name}_pruned_{percentage}.pt"
            pruned_mb = serialized_size_mb(pruned, pruned_path)
            sparse_mb = estimated_sparse_payload_mb(pruned)
            condition = (
                f"teacher_pruned_{percentage}"
                if base_name == "teacher"
                else f"student_distilled_pruned_{percentage}"
            )
            pruned_row = evaluate_model(
                pruned,
                test_data,
                condition,
                family,
                "pruning" if family == "teacher" else "combined",
                pruned_path,
                pruned_mb,
                sparse_mb,
                args.batch_size,
                args.latency_repeats,
                args.cpu_threads,
                train_seconds=prune_seconds,
                epochs_completed=1,
                sparsity_target=sparsity,
            )
            rows.append(pruned_row)
            write_rows(metrics_path, rows)

    for row in rows:
        wandb_urls[row["condition"]] = log_evaluation_wandb(wandb_enabled, row)

    frame = pd.DataFrame(rows)
    poster_plots(frame, poster_dir)
    compact_table(frame, poster_dir)
    diagnostic_plots(frame, histories, diagnostic_dir)
    for name, history in histories.items():
        if history:
            write_history(output_root / f"{name}_history.csv", history)

    with (output_root / "wandb_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(wandb_urls, handle, indent=2)
    summary = {
        "completed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "conditions": len(rows),
        "metrics": str(metrics_path),
        "poster_dir": str(poster_dir),
        "diagnostic_dir": str(diagnostic_dir),
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Completed {len(rows)} conditions in {summary['elapsed_seconds'] / 60:.1f} minutes")


if __name__ == "__main__":
    main()
