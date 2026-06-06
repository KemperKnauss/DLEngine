from __future__ import annotations

import unittest
import json

from scripts.split_labels import grouped_split_indices, split_indices


class SplitLabelsTest(unittest.TestCase):
    def test_split_indices_are_disjoint_and_complete(self) -> None:
        splits = split_indices(total=100, val_fraction=0.1, test_fraction=0.1, seed=7)
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(train | val | test, set(range(100)))

    def test_grouped_split_keeps_games_together(self) -> None:
        rows = [
            json.dumps({"game_id": str(game_id), "fen": f"fen-{game_id}-{position}"}) + "\n"
            for game_id in range(20)
            for position in range(5)
        ]
        splits = grouped_split_indices(rows, "game_id", 0.1, 0.1, seed=7)
        split_for_index = {
            index: split_name
            for split_name, indices in splits.items()
            for index in indices
        }
        for game_id in range(20):
            game_splits = {split_for_index[index] for index in range(game_id * 5, game_id * 5 + 5)}
            self.assertEqual(len(game_splits), 1)
