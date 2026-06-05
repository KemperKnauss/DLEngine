from __future__ import annotations

import json
import math
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset

from .encoding import ACTION_SIZE, board_to_tensor, move_to_index


def centipawns_to_value(eval_cp: float, scale: float = 1000.0) -> float:
    """Map centipawns to a stable regression target in roughly [-1, 1]."""
    return math.tanh(float(eval_cp) / scale)


def multipv_policy(labels: list[dict], temperature_cp: float, turn: chess.Color) -> torch.Tensor:
    policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    if not labels:
        return policy

    perspective = 1.0 if turn == chess.WHITE else -1.0
    scores = torch.tensor([perspective * float(row["eval_cp"]) for row in labels], dtype=torch.float32)
    probs = torch.softmax(scores / temperature_cp, dim=0)
    for row, prob in zip(labels, probs):
        move = chess.Move.from_uci(row["move"])
        policy[move_to_index(move)] = prob
    return policy


class StockfishJsonlDataset(Dataset):
    def __init__(
        self,
        labels_path: str | Path,
        temperature_cp: float = 80.0,
        value_scale_cp: float = 1000.0,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.temperature_cp = temperature_cp
        self.value_scale_cp = value_scale_cp
        with self.labels_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        board = chess.Board(row["fen"])
        labels = row["labels"]
        best_eval = labels[0]["eval_cp"] if labels else 0.0
        return {
            "fen": row["fen"],
            "board": board_to_tensor(board),
            "policy": multipv_policy(labels, self.temperature_cp, board.turn),
            "best_move": torch.tensor(move_to_index(chess.Move.from_uci(labels[0]["move"])), dtype=torch.long),
            "value": torch.tensor([centipawns_to_value(best_eval, self.value_scale_cp)], dtype=torch.float32),
        }

