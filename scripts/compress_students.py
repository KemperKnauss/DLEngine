from __future__ import annotations

import argparse
import csv
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess_student.data import StockfishJsonlDataset, mask_illegal_logits
from chess_student.models import build_model, count_parameters
from scripts.train_student import soft_cross_entropy


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() < 2:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum() * (vy * vy).sum())
    return float(((vx * vy).sum() / denom).item()) if denom > 0 else float("nan")


def nonzero_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for parameter in model.parameters():
        total += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter.detach()).item())
    return nonzero, total


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def measure_latency_ms(model: nn.Module, batch: torch.Tensor, repeats: int, device: torch.device) -> float:
    model.eval()
    batch = batch[:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000 / repeats


def load_student(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        channels=int(checkpoint["channels"]),
        depth=int(checkpoint["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_model(
    model: nn.Module,
    labels_path: Path,
    batch_size: int,
    latency_repeats: int,
    device: torch.device,
    model_label: str,
    checkpoint_path: Path,
    compression: str,
    sparsity_target: float | None = None,
) -> dict[str, float | int | str]:
    dataset = StockfishJsonlDataset(labels_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0
    top1 = 0
    top3 = 0
    value_preds = []
    value_targets = []
    policy_losses = []
    first_batch = None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {model_label}"):
            boards = batch["board"].to(device)
            if first_batch is None:
                first_batch = boards.detach().cpu()
            target_policy = batch["policy"].to(device)
            best_move = batch["best_move"].to(device)
            target_value = batch["value"].to(device)
            policy_logits, value_pred = model(boards)
            policy_logits = mask_illegal_logits(policy_logits, batch["legal_indices"])
            policy_losses.append((-(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1)).cpu())
            predictions = torch.topk(policy_logits, k=3, dim=1).indices
            top1 += int((predictions[:, 0] == best_move).sum().item())
            top3 += int((predictions == best_move.unsqueeze(1)).any(dim=1).sum().item())
            total += boards.size(0)
            value_preds.append(value_pred.cpu())
            value_targets.append(target_value.cpu())

    pred_values = torch.cat(value_preds)
    target_values = torch.cat(value_targets)
    value_mse = float(F.mse_loss(pred_values, target_values).item())
    nonzero, param_total = nonzero_parameters(model)
    latency_ms = measure_latency_ms(model, first_batch, latency_repeats, device)
    return {
        "model": model_label,
        "checkpoint": str(checkpoint_path),
        "compression": compression,
        "sparsity_target": "" if sparsity_target is None else sparsity_target,
        "positions": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "policy_ce": float(torch.cat(policy_losses).mean().item()),
        "value_mse": value_mse,
        "value_rmse": math.sqrt(value_mse),
        "value_pearson": pearson(pred_values, target_values),
        "params": count_parameters(model),
        "nonzero_params": nonzero,
        "actual_sparsity": 1.0 - (nonzero / max(1, param_total)),
        "model_mb": checkpoint_size_mb(checkpoint_path),
        "latency_ms": latency_ms,
    }


def append_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(metrics)


def maybe_log_wandb(args: argparse.Namespace, metrics: dict[str, float | int | str], group: str) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed; skipping compression wandb log.")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=group,
        name=str(metrics["model"]),
        config=metrics,
        job_type=group,
    )
    run.log(metrics)
    run.finish()


def quantize_model(model: nn.Module) -> nn.Module:
    cpu_model = deepcopy(model).cpu().eval()
    return torch.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)


def prune_targets(model: nn.Module) -> list[tuple[nn.Module, str]]:
    targets: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            targets.append((module, "weight"))
    return targets


def apply_pruning(model: nn.Module, amount: float) -> None:
    for module, parameter_name in prune_targets(model):
        prune.l1_unstructured(module, name=parameter_name, amount=amount)


def remove_pruning(model: nn.Module) -> None:
    for module, parameter_name in prune_targets(model):
        if hasattr(module, f"{parameter_name}_orig"):
            prune.remove(module, parameter_name)


def finetune_model(
    model: nn.Module,
    labels_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    value_loss_weight: float,
) -> None:
    dataset = StockfishJsonlDataset(labels_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        for batch in tqdm(loader, desc=f"finetune {epoch}"):
            boards = batch["board"].to(device)
            policy = batch["policy"].to(device)
            value = batch["value"].to(device)
            policy_logits, value_pred = model(boards)
            policy_logits = mask_illegal_logits(policy_logits, batch["legal_indices"])
            policy_loss = soft_cross_entropy(policy_logits, policy)
            value_loss = F.mse_loss(value_pred, value)
            loss = policy_loss + value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += boards.size(0)
            total_loss += loss.item() * boards.size(0)
        print(f"finetune_epoch={epoch} loss={total_loss / max(1, total):.4f}")


def save_model_object(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and prune trained chess students.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/overnight/compression"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/overnight/compression_metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--prune-sparsities", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--prune-finetune-epochs", type=int, default=2)
    parser.add_argument("--prune-lr", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="dlengine-chess-compression")
    args = parser.parse_args()

    device = torch.device(args.device)
    for checkpoint_path in args.checkpoints:
        base_label = checkpoint_path.parent.name
        model, checkpoint = load_student(checkpoint_path, device)

        control = deepcopy(model).to(device)
        finetune_model(
            control,
            args.train_labels,
            device,
            args.prune_finetune_epochs,
            args.batch_size,
            args.prune_lr,
            args.value_loss_weight,
        )
        control_path = args.out_dir / f"{base_label}_finetune_control.pt"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": checkpoint["model_name"],
                "channels": checkpoint["channels"],
                "depth": checkpoint["depth"],
                "state_dict": control.state_dict(),
                "params": count_parameters(control),
                "compression": "finetune_control",
            },
            control_path,
        )
        control_metrics = evaluate_model(
            control,
            args.test_labels,
            args.batch_size,
            args.latency_repeats,
            device,
            f"{base_label}_finetune_control",
            control_path,
            "finetune_control",
        )
        append_metrics(args.metrics_out, control_metrics)
        maybe_log_wandb(args, control_metrics, "pruning_control")

        quantized = quantize_model(model)
        quantized_path = args.out_dir / f"{base_label}_dynamic_quantized.pt"
        save_model_object(quantized, quantized_path)
        quant_metrics = evaluate_model(
            quantized,
            args.test_labels,
            args.batch_size,
            args.latency_repeats,
            torch.device("cpu"),
            f"{base_label}_dynamic_quantized",
            quantized_path,
            "dynamic_quantization",
        )
        append_metrics(args.metrics_out, quant_metrics)
        maybe_log_wandb(args, quant_metrics, "quantization")

        for sparsity in args.prune_sparsities:
            pruned = deepcopy(model).to(device)
            apply_pruning(pruned, sparsity)
            finetune_model(
                pruned,
                args.train_labels,
                device,
                args.prune_finetune_epochs,
                args.batch_size,
                args.prune_lr,
                args.value_loss_weight,
            )
            remove_pruning(pruned)
            pruned_path = args.out_dir / f"{base_label}_pruned_{int(sparsity * 100)}.pt"
            torch.save(
                {
                    "model_name": checkpoint["model_name"],
                    "channels": checkpoint["channels"],
                    "depth": checkpoint["depth"],
                    "state_dict": pruned.state_dict(),
                    "params": count_parameters(pruned),
                    "compression": "unstructured_pruning",
                    "sparsity_target": sparsity,
                },
                pruned_path,
            )
            prune_metrics = evaluate_model(
                pruned,
                args.test_labels,
                args.batch_size,
                args.latency_repeats,
                device,
                f"{base_label}_pruned_{int(sparsity * 100)}",
                pruned_path,
                "unstructured_pruning",
                sparsity,
            )
            append_metrics(args.metrics_out, prune_metrics)
            maybe_log_wandb(args, prune_metrics, "pruning")


if __name__ == "__main__":
    main()
