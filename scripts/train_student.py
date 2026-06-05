from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from chess_student.data import StockfishJsonlDataset
from chess_student.models import build_model, count_parameters


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed; continuing without wandb logging.")
        return None
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name,
        config=config,
        job_type="train",
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    examples = 0

    for batch in tqdm(loader, desc=desc):
        boards = batch["board"].to(device)
        policy = batch["policy"].to(device)
        value = batch["value"].to(device)
        policy_logits, value_pred = model(boards)
        policy_loss = soft_cross_entropy(policy_logits, policy)
        value_loss = F.mse_loss(value_pred, value)
        loss = policy_loss + value_loss_weight * value_loss

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = boards.size(0)
        examples += batch_size
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size

    return {
        "loss": total_loss / max(1, examples),
        "policy_loss": total_policy_loss / max(1, examples),
        "value_loss": total_value_loss / max(1, examples),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = StockfishJsonlDataset(args.labels, args.temperature_cp, args.value_scale_cp)
    if args.val_labels:
        train_data = dataset
        val_data = StockfishJsonlDataset(args.val_labels, args.temperature_cp, args.value_scale_cp)
        split_metadata = {
            "mode": "explicit",
            "train_labels": str(args.labels),
            "val_labels": str(args.val_labels),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "seed": args.seed,
        }
    else:
        val_size = max(1, int(len(dataset) * args.val_fraction))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
        split_metadata = {
            "mode": "random_split",
            "labels": str(args.labels),
            "val_fraction": args.val_fraction,
            "train_size": train_size,
            "val_size": val_size,
            "seed": args.seed,
            "train_indices": list(train_data.indices),
            "val_indices": list(val_data.indices),
        }

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, channels=args.channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2)

    wandb_run = maybe_init_wandb(
        args,
        {
            **vars(args),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "params": count_parameters(model),
        },
    )

    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, args.value_loss_weight, optimizer, f"epoch {epoch} train")
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, args.value_loss_weight, None, f"epoch {epoch} val")

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/policy_loss": train_metrics["policy_loss"],
                    "train/value_loss": train_metrics["value_loss"],
                    "val/loss": val_metrics["loss"],
                    "val/policy_loss": val_metrics["policy_loss"],
                    "val/value_loss": val_metrics["value_loss"],
                }
            )

        checkpoint = {
            "model_name": args.model,
            "channels": args.channels,
            "depth": args.depth,
            "state_dict": model.state_dict(),
            "params": count_parameters(model),
        }
        torch.save(checkpoint, args.out_dir / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(checkpoint, args.out_dir / "best.pt")

    with (args.out_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"best_val_loss": best_val, "best_epoch": best_epoch}, handle, indent=2, default=str)
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact neural student on Stockfish labels.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels/stockfish_labels.jsonl"))
    parser.add_argument("--val-labels", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/small_cnn"))
    parser.add_argument("--model", choices=["small_cnn", "depthwise_cnn"], default="small_cnn")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--temperature-cp", type=float, default=80.0)
    parser.add_argument("--value-scale-cp", type=float, default=1000.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="dlengine-chess-compression")
    parser.add_argument("--wandb-group", default="distillation")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

