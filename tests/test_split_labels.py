from __future__ import annotations

import unittest

from scripts.split_labels import split_indices


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
