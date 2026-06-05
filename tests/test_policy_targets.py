from __future__ import annotations

import sys
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_student.data import multipv_policy
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
