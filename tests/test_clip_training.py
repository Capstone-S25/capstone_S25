"""Training regressions for masks, source holdouts and dev-only operating points."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from anomaly_eval.training.clip import (
    binary_metrics,
    frozen_extract,
    masked_binary_loss,
    pool_views,
    select_points,
)
from anomaly_eval.training.data import group_class_weights, validate_records
from anomaly_eval.training.predict import temporal_support


def example(split, target, group):
    return {
        "id": f"{split}-{target}-{group}",
        "task": "smoking",
        "target": target,
        "split": split,
        "group_id": group,
        "views": ["unused.png"],
        "evidence": "source_annotation",
        "pixel_hashes": [f"{split}-{target}-{group}"],
    }


class ClipTrainingTests(unittest.TestCase):
    def test_duration_targets_require_timed_history(self):
        self.assertFalse(temporal_support("loitering", None, 4))
        self.assertFalse(temporal_support("unattended_items", [0.0, 3.0], 2))
        self.assertTrue(temporal_support("loitering", [0.0, 6.0, 13.0, 20.0], 4))
        self.assertTrue(temporal_support("smoking", None, 1))
        self.assertFalse(temporal_support("littering", None, 4))
        self.assertTrue(temporal_support("littering", None, 8))
        self.assertFalse(temporal_support("loitering", [0.0, 0.0, 0.0, 20.0], 4))
        self.assertFalse(
            temporal_support(
                "loitering", [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0], 8
            )
        )

    def test_frozen_cache_checks_pixels_and_rejects_trainable_encoders(self):
        encoder = SimpleNamespace(
            backbone=torch.nn.Linear(2, 2).requires_grad_(False),
            name="fixture",
            revision="pinned",
        )
        rows = [{"pixel_hashes": ["first"]}]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("anomaly_eval.training.clip.ROOT", Path(directory)),
            patch(
                "anomaly_eval.training.clip.extract",
                return_value=np.ones((1, 8), np.float32),
            ) as extract,
        ):
            frozen_extract(rows, encoder, 1, 0)
            frozen_extract(rows, encoder, 1, 0)
            self.assertEqual(extract.call_count, 1)
            frozen_extract([{"pixel_hashes": ["changed"]}], encoder, 1, 0)
            self.assertEqual(extract.call_count, 2)
            encoder.backbone.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "immutable frozen base"):
                frozen_extract(rows, encoder, 1, 0)

    def test_sampling_balances_dataset_domains_before_recording_groups(self):
        rows = [{**example("train", 1, "camera"), "source": "cctv"} for _ in range(12)]
        rows += [
            {**example("train", 1, "photo" + str(i)), "source": "photos"}
            for i in range(30)
        ]
        weights = group_class_weights(rows)
        self.assertAlmostEqual(sum(weights[:12]), sum(weights[12:]))

    def test_unknown_targets_have_no_gradient(self):
        values = torch.zeros(2, 5, requires_grad=True)
        loss = masked_binary_loss(
            values, torch.tensor([0, 3]), torch.tensor([1.0, 0.0])
        )
        loss.backward()
        self.assertEqual(int((values.grad != 0).sum()), 2)
        self.assertLess(values.grad[0, 0], 0)
        self.assertGreater(values.grad[1, 3], 0)

    def test_rejects_source_and_pixel_leakage(self):
        rows = [
            example(s, y, s + str(y)) for s in ["train", "dev", "test"] for y in [0, 1]
        ]
        validate_records(rows, Path("/tmp"), check_files=False)
        rows[2]["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "Source group crosses"):
            validate_records(rows, Path("/tmp"), check_files=False)
        rows[2]["group_id"] = "independent"
        rows[2]["pixel_hashes"] = rows[0]["pixel_hashes"]
        with self.assertRaisesRegex(ValueError, "pixels cross"):
            validate_records(rows, Path("/tmp"), check_files=False)

    def test_requires_each_supported_task_to_have_both_holdout_classes(self):
        rows = [
            example(s, y, s + str(y)) for s in ["train", "dev", "test"] for y in [0, 1]
        ]
        with self.assertRaisesRegex(ValueError, "both classes"):
            validate_records(rows[:-1], Path("/tmp"), check_files=False)

    def test_sampling_mass_equal_across_classes_and_groups(self):
        rows = (
            [example("train", 0, "a") for _ in range(8)]
            + [example("train", 0, "b")]
            + [example("train", 1, "c") for _ in range(3)]
        )
        weights = group_class_weights(rows)
        self.assertAlmostEqual(sum(weights[:8]), weights[8])
        self.assertAlmostEqual(sum(weights[:9]), sum(weights[9:]))

    def test_temporal_pool_preserves_order(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        a = pool_views(x, [2])
        b = pool_views(x.flip(0), [2])
        self.assertEqual(a.shape, (1, 8))
        self.assertFalse(torch.allclose(a, b))
        self.assertAlmostEqual(float(a.norm()), 1.0, places=6)

    def test_fpr_budget_with_ties_and_all_negative_fallback(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.5, 0.5, 0.5, 0.5])
        points = select_points(y, score)
        m = binary_metrics(y, score, points["dev_fpr_budget"])
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(binary_metrics(y, score, points["dev_max_f1"])["recall"], 1.0)
        with self.assertRaises(ValueError):
            select_points([0, 1], [float("nan"), 1])


if __name__ == "__main__":
    unittest.main()
