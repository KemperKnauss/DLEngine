from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

from chess_student.comparison import (
    ACTION_SIZE,
    build_comparison_model,
    distillation_loss,
    parameter_sparsity,
    prunable_modules,
)


class ComparisonExperimentTest(unittest.TestCase):
    def sample_batch(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        legal = torch.full((batch_size, 256), -1, dtype=torch.long)
        legal[:, :3] = torch.tensor([0, 1, 2])
        target_actions = torch.full((batch_size, 5), -1, dtype=torch.long)
        target_actions[:, :2] = torch.tensor([0, 1])
        target_probs = torch.zeros(batch_size, 5)
        target_probs[:, :2] = torch.tensor([0.75, 0.25])
        return {
            "board": torch.zeros(batch_size, 18, 8, 8),
            "legal_indices": legal,
            "target_actions": target_actions,
            "target_probs": target_probs,
            "best_move": torch.zeros(batch_size, dtype=torch.long),
            "value": torch.zeros(batch_size, 1),
        }

    def test_model_shapes_and_distillation_loss(self) -> None:
        batch = self.sample_batch()
        teacher = build_comparison_model("teacher")
        student = build_comparison_model("student")
        teacher_logits, teacher_value = teacher(batch["board"])
        student_logits, student_value = student(batch["board"])
        self.assertEqual(teacher_logits.shape, (2, ACTION_SIZE))
        self.assertEqual(student_logits.shape, (2, ACTION_SIZE))
        self.assertEqual(student_value.shape, (2, 1))
        loss, parts = distillation_loss(
            student_logits,
            student_value,
            teacher_logits,
            teacher_value,
            batch,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["teacher_policy"]))

    def test_global_pruning_reaches_target(self) -> None:
        model = build_comparison_model("student")
        prune.global_unstructured(
            prunable_modules(model),
            pruning_method=prune.L1Unstructured,
            amount=0.5,
        )
        for module, name in prunable_modules(model):
            prune.remove(module, name)
        _, _, sparsity = parameter_sparsity(model)
        self.assertAlmostEqual(sparsity, 0.5, places=3)

    def test_quantized_model_serializes_and_reloads(self) -> None:
        model = build_comparison_model("student").eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantized.pt"
            torch.save(quantized, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            logits, value = loaded(torch.zeros(1, 18, 8, 8))
        self.assertEqual(logits.shape, (1, ACTION_SIZE))
        self.assertEqual(value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
