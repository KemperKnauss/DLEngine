from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chess_student.data import StockfishJsonlDataset, mask_illegal_logits
from chess_student.models import build_model, count_parameters


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() < 2:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum() * (vy * vy).sum())
    return float(((vx * vy).sum() / denom).item()) if denom > 0 else float("nan")


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def measure_latency_ms(model: torch.nn.Module, batch: torch.Tensor, repeats: int, device: torch.device) -> float:
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


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        channels=int(checkpoint["channels"]),
        depth=int(checkpoint["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = StockfishJsonlDataset(args.labels, args.temperature_cp, args.value_scale_cp)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total = 0
    top1 = 0
    top3 = 0
    value_preds = []
    value_targets = []
    policy_losses = []

    with torch.no_grad():
        first_batch = None
        for batch in loader:
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
    value_rmse = math.sqrt(value_mse)
    latency_ms = measure_latency_ms(model, first_batch, args.latency_repeats, device)

    metrics = {
        "model": args.model_label or checkpoint["model_name"],
        "checkpoint": str(args.checkpoint),
        "positions": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "policy_ce": float(torch.cat(policy_losses).mean().item()),
        "value_mse": value_mse,
        "value_rmse": value_rmse,
        "value_pearson": pearson(pred_values, target_values),
        "params": count_parameters(model),
        "model_mb": checkpoint_size_mb(args.checkpoint),
        "latency_ms": latency_ms,
    }
    return metrics


def append_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained chess student against Stockfish labels.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/stockfish_labels.jsonl"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature-cp", type=float, default=80.0)
    parser.add_argument("--value-scale-cp", type=float, default=1000.0)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metrics = evaluate(args)
    append_metrics(args.metrics_out, metrics)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

