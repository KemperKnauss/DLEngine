from __future__ import annotations

import sys
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from chess_student.data import mask_illegal_logits, multipv_policy
from chess_student.encoding import move_to_index


class PolicyTargetTest(unittest.TestCase):
    def test_policy_prefers_higher_white_perspective_score_for_white_to_move(self) -> None:
        labels = [
            {"move": "e2e4", "eval_cp": 100},
            {"move": "d2d4", "eval_cp": 50},
        ]

        policy = multipv_policy(labels, temperature_cp=80.0, turn=chess.WHITE)

        self.assertGreater(
            policy[move_to_index(chess.Move.from_uci("e2e4"))],
            policy[move_to_index(chess.Move.from_uci("d2d4"))],
        )

    def test_policy_prefers_lower_white_perspective_score_for_black_to_move(self) -> None:
        labels = [
            {"move": "e7e5", "eval_cp": -100},
            {"move": "c7c5", "eval_cp": -50},
        ]

        policy = multipv_policy(labels, temperature_cp=80.0, turn=chess.BLACK)

        self.assertGreater(
            policy[move_to_index(chess.Move.from_uci("e7e5"))],
            policy[move_to_index(chess.Move.from_uci("c7c5"))],
        )

    def test_illegal_moves_are_masked_before_ranking(self) -> None:
        board = chess.Board()
        logits = torch.zeros((1, 64 * 64 * 5))
        illegal_move = chess.Move.from_uci("e2e5")
        legal_move = chess.Move.from_uci("e2e4")
        logits[0, move_to_index(illegal_move)] = 100.0
        logits[0, move_to_index(legal_move)] = 1.0
        legal_indices = torch.full((1, 256), -1, dtype=torch.long)
        moves = [move_to_index(move) for move in board.legal_moves]
        legal_indices[0, : len(moves)] = torch.tensor(moves)

        masked = mask_illegal_logits(logits, legal_indices)

        self.assertEqual(int(masked.argmax(dim=1).item()), move_to_index(legal_move))
