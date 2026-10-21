"""Leakage, partial supervision, balanced coverage and binary VLM scoring."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from anomaly_eval.training.data import TASKS
from anomaly_eval.training.smolvlm import (
    epoch_indices,
    evaluate_scores,
    macro_ap,
    messages,
    response_loss,
    task_ap,
)


class SmolVLMTrainingTests(unittest.TestCase):
    def test_prediction_requires_real_temporal_context(self):
        from scripts.predict_smolvlm import prepare_views

        self.assertEqual(prepare_views("smoking", ["first", "last"]), (["last"], None))
        images = list(range(4))
        self.assertIsNone(prepare_views("loitering", images)[0])
        self.assertEqual(
            prepare_views("loitering", images, [0, 7, 13, 20]), (images, None)
        )
        self.assertIsNone(prepare_views("unattended_items", images, [0, 0, 0, 20])[0])
        self.assertIsNone(prepare_views("littering", images)[0])
        self.assertEqual(prepare_views("littering", list(range(8)))[0], list(range(8)))
        with self.assertRaisesRegex(ValueError, "finite, chronological"):
            prepare_views("loitering", images, [20, 13, 7, 0])

    def test_training_has_full_coverage_reproducible_balance_and_no_dev(self):
        rows = [
            {
                "split": "train",
                "task": task,
                "target": y,
                "source": "fixture",
                "group_id": str(i),
            }
            for task in TASKS
            for y in (0, 1)
            for i in range(2)
        ]
        indices = epoch_indices(rows, 42, 1)
        self.assertEqual(len(indices), 2 * len(rows))
        self.assertEqual(set(indices), set(range(len(rows))))
        self.assertEqual(indices, epoch_indices(rows, 42, 1))
        self.assertNotEqual(indices, epoch_indices(rows, 42, 2))
        with self.assertRaisesRegex(ValueError, "only known training"):
            epoch_indices([{**rows[0], "split": "test"}], 42, 1)

    def test_response_loss_has_one_known_binary_objective(self):
        logits = torch.zeros(2, 10, requires_grad=True)
        loss = response_loss(logits, torch.tensor([1, 0]), [3, 7])
        loss.backward()
        self.assertLess(logits.grad[0, 7], 0)
        self.assertLess(logits.grad[1, 3], 0)
        self.assertGreater(logits.grad[0, 3], 0)
        with self.assertRaisesRegex(ValueError, "Unknown objectives"):
            response_loss(logits, torch.tensor([-1, 0]), [3, 7])

    def test_prompt_contains_only_queried_definition_and_all_views(self):
        content = messages("unattended_items", 4)[0]["content"]
        self.assertEqual(sum(x["type"] == "image" for x in content), 4)
        self.assertIn(
            "Ownership and duration require evidence across time", content[-1]["text"]
        )
        self.assertNotIn("housing", content[-1]["text"])
        self.assertNotIn("cigarette", content[-1]["text"])

    def test_selection_weights_sources_then_tasks_and_uses_frozen_thresholds(self):
        rows = [
            {"task": task, "target": y, "source": "fixture", "group_id": str(y)}
            for task in TASKS
            for y in (0, 1)
        ]
        dev = np.array([0.1, 0.9] * 5)
        metrics = evaluate_scores(rows, dev)
        self.assertEqual(macro_ap(rows, dev), 1)
        self.assertEqual(task_ap(rows, dev, "smoking"), 1)
        thresholds = {t: value["thresholds"] for t, value in metrics.items()}
        shifted = evaluate_scores(rows, dev + 10, thresholds)
        for task in TASKS:
            self.assertEqual(shifted[task]["thresholds"], thresholds[task])
            self.assertEqual(
                shifted[task]["operating_points"]["dev_fpr_budget"]["fpr"], 1
            )
        with self.assertRaisesRegex(ValueError, "finite binary margin"):
            evaluate_scores(rows, np.full(len(rows), np.nan))


if __name__ == "__main__":
    unittest.main()
