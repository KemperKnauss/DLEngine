from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_scaled_compression_comparison import (
    aggregate_metrics,
    validation_selected_condition,
)


class ScaledComparisonTest(unittest.TestCase):
    def sample_rows(self) -> pd.DataFrame:
        rows = []
        for seed, top1, latency in ((7, 0.20, 0.5), (17, 0.22, 0.7)):
            rows.append(
                {
                    "condition": "teacher_pruned_25",
                    "family": "teacher",
                    "method": "pruning",
                    "seed": seed,
                    "top1": top1,
                    "top3": 0.4,
                    "policy_ce": 2.9,
                    "value_rmse": 0.2,
                    "value_pearson": 0.5,
                    "actual_model_mb": 40.0,
                    "actual_sparsity": 0.25,
                    "sparse_estimated_mb": 50.0,
                    "val_top1": 0.21,
                    "val_policy_ce": 2.8,
                    "gpu_latency_ms": 0.4,
                    "peak_gpu_mb": 100.0,
                    "train_seconds": 10.0,
                    "params": 1000,
                    "prunable_params": 900,
                    "nonzero_prunable_params": 675,
                    "sparsity_target": 0.25,
                    "cpu_latency_ms": latency,
                }
            )
        return pd.DataFrame(rows)

    def test_aggregation_calculates_seed_uncertainty_and_median_latency(self) -> None:
        aggregate = aggregate_metrics(self.sample_rows())
        row = aggregate.iloc[0]
        self.assertEqual(row["seeds"], 2)
        self.assertAlmostEqual(row["top1_mean"], 0.21)
        self.assertGreater(row["top1_ci95"], 0)
        self.assertAlmostEqual(row["cpu_latency_ms"], 0.6)

    def test_validation_selection_does_not_use_test_top1(self) -> None:
        aggregate = pd.DataFrame(
            [
                {
                    "condition": "teacher_pruned_25",
                    "val_top1_mean": 0.30,
                    "top1_mean": 0.10,
                    "actual_model_mb_mean": 40.0,
                },
                {
                    "condition": "teacher_pruned_50",
                    "val_top1_mean": 0.20,
                    "top1_mean": 0.90,
                    "actual_model_mb_mean": 20.0,
                },
            ]
        )
        selected = validation_selected_condition(aggregate, "teacher_pruned_")
        self.assertEqual(selected, "teacher_pruned_25")


if __name__ == "__main__":
    unittest.main()
