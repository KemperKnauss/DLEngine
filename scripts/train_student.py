from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from chess_student.data import StockfishJsonlDataset
from chess_student.models import build_model, count_parameters


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = StockfishJsonlDataset(args.labels, args.temperature_cp, args.value_scale_cp)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, channels=args.channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            boards = batch["board"].to(device)
            policy = batch["policy"].to(device)
            value = batch["value"].to(device)
            policy_logits, value_pred = model(boards)
            policy_loss = soft_cross_entropy(policy_logits, policy)
            value_loss = F.mse_loss(value_pred, value)
            loss = policy_loss + args.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * boards.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                boards = batch["board"].to(device)
                policy = batch["policy"].to(device)
                value = batch["value"].to(device)
                policy_logits, value_pred = model(boards)
                loss = soft_cross_entropy(policy_logits, policy) + args.value_loss_weight * F.mse_loss(value_pred, value)
                val_loss += loss.item() * boards.size(0)

        train_loss /= max(1, train_size)
        val_loss /= max(1, val_size)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        checkpoint = {
            "model_name": args.model,
            "channels": args.channels,
            "depth": args.depth,
            "state_dict": model.state_dict(),
            "params": count_parameters(model),
        }
        torch.save(checkpoint, args.out_dir / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, args.out_dir / "best.pt")

    with (args.out_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"best_val_loss": best_val}, handle, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact neural student on Stockfish labels.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels/stockfish_labels.jsonl"))
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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

