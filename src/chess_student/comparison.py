from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .data import MAX_LEGAL_MOVES, centipawns_to_value, legal_action_indices
from .encoding import ACTION_SIZE, BOARD_CHANNELS, board_to_tensor, move_to_index


MAX_TEACHER_MOVES = 5


class StockfishComparisonDataset(Dataset):
    """Eager compact tensors for repeated teacher/student training passes."""

    def __init__(
        self,
        labels_path: str | Path,
        temperature_cp: float = 80.0,
        value_scale_cp: float = 1000.0,
        limit: int | None = None,
    ) -> None:
        rows: list[dict[str, Any]] = []
        with Path(labels_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break

        boards = []
        legal = []
        target_actions = []
        target_probs = []
        best_moves = []
        values = []
        game_ids = []

        for row in rows:
            board = chess.Board(row["fen"])
            labels = row["labels"][:MAX_TEACHER_MOVES]
            perspective = 1.0 if board.turn == chess.WHITE else -1.0
            scores = torch.tensor(
                [perspective * float(label["eval_cp"]) for label in labels],
                dtype=torch.float32,
            )
            probabilities = torch.softmax(scores / temperature_cp, dim=0)

            actions = torch.full((MAX_TEACHER_MOVES,), -1, dtype=torch.int32)
            probs = torch.zeros(MAX_TEACHER_MOVES, dtype=torch.float32)
            for index, (label, probability) in enumerate(zip(labels, probabilities)):
                actions[index] = move_to_index(chess.Move.from_uci(label["move"]))
                probs[index] = probability

            boards.append(board_to_tensor(board).to(torch.uint8))
            legal.append(legal_action_indices(board).to(torch.int32))
            target_actions.append(actions)
            target_probs.append(probs)
            best_moves.append(actions[0].to(torch.long))
            best_eval = float(labels[0]["eval_cp"]) if labels else 0.0
            values.append(torch.tensor([centipawns_to_value(best_eval, value_scale_cp)]))
            game_ids.append(str(row.get("game_id", "")))

        self.boards = torch.stack(boards)
        self.legal_indices = torch.stack(legal)
        self.target_actions = torch.stack(target_actions)
        self.target_probs = torch.stack(target_probs)
        self.best_moves = torch.stack(best_moves)
        self.values = torch.stack(values)
        self.game_ids = game_ids

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "board": self.boards[index].float(),
            "legal_indices": self.legal_indices[index].long(),
            "target_actions": self.target_actions[index].long(),
            "target_probs": self.target_probs[index],
            "best_move": self.best_moves[index],
            "value": self.values[index],
            "game_id": self.game_ids[index],
        }


class StockfishSurrogate(nn.Module):
    def __init__(self, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        input_dim = BOARD_CHANNELS * 8 * 8
        first, second = hidden_dims
        self.hidden_dims = hidden_dims
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, first),
            nn.ReLU(inplace=True),
            nn.Linear(first, second),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(second, ACTION_SIZE)
        self.value_head = nn.Sequential(nn.Linear(second, 1), nn.Tanh())

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(board)
        return self.policy_head(features), self.value_head(features)


def build_comparison_model(kind: str) -> StockfishSurrogate:
    if kind == "teacher":
        return StockfishSurrogate((1024, 512))
    if kind == "student":
        return StockfishSurrogate((256, 128))
    raise ValueError(f"Unknown comparison model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def legal_logits(logits: torch.Tensor, legal_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = legal_indices.to(logits.device)
    valid = indices >= 0
    gathered = logits.gather(1, indices.clamp_min(0)).float()
    return gathered.masked_fill(~valid, -1.0e9), valid


def sparse_policy_cross_entropy(
    logits: torch.Tensor,
    legal_indices: torch.Tensor,
    target_actions: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    gathered, _ = legal_logits(logits, legal_indices)
    log_probs = F.log_softmax(gathered, dim=1)
    actions = target_actions.to(logits.device)
    targets = target_probs.to(logits.device)
    legal = legal_indices.to(logits.device)
    positions = (actions.unsqueeze(2) == legal.unsqueeze(1)).to(torch.int64).argmax(dim=2)
    valid_targets = actions >= 0
    selected = log_probs.gather(1, positions)
    return -(selected * targets * valid_targets).sum(dim=1).mean()


def stockfish_loss(
    logits: torch.Tensor,
    value_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    value_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy = sparse_policy_cross_entropy(
        logits,
        batch["legal_indices"],
        batch["target_actions"],
        batch["target_probs"],
    )
    value = F.mse_loss(value_pred.float(), batch["value"].to(value_pred.device).float())
    return policy + value_loss_weight * value, {"policy": policy, "value": value}


def distillation_loss(
    student_logits: torch.Tensor,
    student_value: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_value: torch.Tensor,
    batch: dict[str, torch.Tensor],
    alpha: float = 0.7,
    temperature: float = 2.0,
    value_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stockfish_total, stockfish_parts = stockfish_loss(
        student_logits,
        student_value,
        batch,
        value_loss_weight,
    )
    student_legal, valid = legal_logits(student_logits, batch["legal_indices"])
    teacher_legal, _ = legal_logits(teacher_logits, batch["legal_indices"])
    student_log_probs = F.log_softmax(student_legal / temperature, dim=1)
    teacher_probs = F.softmax(teacher_legal.detach() / temperature, dim=1)
    teacher_probs = teacher_probs.masked_fill(~valid.to(teacher_probs.device), 0.0)
    policy_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    policy_kl = policy_kl * (temperature**2)
    value_mse = F.mse_loss(student_value.float(), teacher_value.detach().float())
    teacher_total = policy_kl + value_loss_weight * value_mse
    total = (1.0 - alpha) * stockfish_total + alpha * teacher_total
    return total, {
        "policy": stockfish_parts["policy"],
        "value": stockfish_parts["value"],
        "teacher_policy": policy_kl,
        "teacher_value": value_mse,
    }


def topk_move_predictions(
    logits: torch.Tensor,
    legal_indices: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    gathered, _ = legal_logits(logits, legal_indices)
    positions = torch.topk(gathered, k=k, dim=1).indices
    return legal_indices.to(logits.device).gather(1, positions)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() < 2:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denominator = torch.sqrt((vx * vx).sum() * (vy * vy).sum())
    return float(((vx * vy).sum() / denominator).item()) if denominator > 0 else float("nan")


def prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(module, "weight") for module in model.modules() if isinstance(module, nn.Linear)]


def parameter_sparsity(model: nn.Module) -> tuple[int, int, float]:
    nonzero = 0
    total = 0
    for module, name in prunable_modules(model):
        parameter = getattr(module, name)
        total += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter.detach()).item())
    return nonzero, total, 1.0 - (nonzero / max(1, total))


def estimated_sparse_payload_mb(model: nn.Module) -> float:
    """Approximate COO-like storage: float32 value + int32 flat index per nonzero."""
    bytes_total = 0
    prunable_ids = {id(getattr(module, name)) for module, name in prunable_modules(model)}
    for parameter in model.parameters():
        if id(parameter) in prunable_ids:
            nonzero = int(torch.count_nonzero(parameter.detach()).item())
            bytes_total += nonzero * (4 + 4)
        else:
            bytes_total += parameter.numel() * parameter.element_size()
    return bytes_total / (1024 * 1024)


def value_rmse(mse: float) -> float:
    return math.sqrt(mse)


__all__ = [
    "ACTION_SIZE",
    "MAX_LEGAL_MOVES",
    "StockfishComparisonDataset",
    "StockfishSurrogate",
    "build_comparison_model",
    "count_parameters",
    "distillation_loss",
    "estimated_sparse_payload_mb",
    "legal_logits",
    "parameter_sparsity",
    "pearson",
    "prunable_modules",
    "sparse_policy_cross_entropy",
    "stockfish_loss",
    "topk_move_predictions",
    "value_rmse",
]
