"""Meaningful representation and source-selection regressions."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from anomaly_eval.training.strategies import geometry, overlap, represent, source_ap
from scripts import fetch_strategy_sources
from scripts.extract_strategy_features import contextual_box


class TrainingStrategyTests(unittest.TestCase):
    def test_cached_download_keeps_actual_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "source.jpg"
            target.write_bytes(b"image")
            receipt = {
                "url": "https://actual.example/image.jpg",
                "path": "source.jpg",
                "bytes": 5,
                "sha256": hashlib.sha256(b"image").hexdigest(),
            }
            target.with_name(target.name + ".receipt.json").write_text(
                json.dumps(receipt)
            )
            with (
                patch.object(fetch_strategy_sources, "ROOT", root),
                patch.object(fetch_strategy_sources.requests, "get") as network,
            ):
                self.assertEqual(
                    fetch_strategy_sources.fetch(
                        "https://another.example/copy.jpg", target
                    ),
                    receipt,
                )
                network.assert_not_called()
                target.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    fetch_strategy_sources.fetch(receipt["url"], target)

    def test_ordered_representation_preserves_intermediate_order(self):
        views = np.eye(4, dtype=np.float32)
        shuffled = views[[0, 2, 1, 3]]
        np.testing.assert_allclose(represent(views), represent(shuffled))
        self.assertFalse(
            np.allclose(
                represent(views, ordered=True), represent(shuffled, ordered=True)
            )
        )

    def test_missing_regions_are_finite_and_not_fake_ownership(self):
        result = geometry([[], [], [], []])
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape, (24,))
        person = {"box": [0.1, 0.1, 0.3, 0.7], "label": 1}
        self.assertFalse(np.allclose(result, geometry([[person]] * 4)))

    def test_region_padding_is_contextual_and_bounded(self):
        box = contextual_box([0.05, 0.1, 0.95, 0.8])
        self.assertEqual(box[0], 0)
        self.assertEqual(box[2], 1)
        self.assertLess(box[1], 0.1)
        self.assertGreater(box[3], 0.8)

    def test_overlap_empty_and_identical(self):
        self.assertEqual(overlap([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(overlap([0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1]), 0)

    def test_source_selection_does_not_weight_the_largest_domain(self):
        rows = [{"source": "large", "target": v} for v in [0, 1] * 10] + [
            {"source": "small", "target": v} for v in [0, 1]
        ]
        scores = np.array([0, 1] * 10 + [1, 0])
        metric = source_ap(rows, scores)
        self.assertAlmostEqual(metric["mean_source_ap"], 0.75)
        self.assertAlmostEqual(metric["worst_source_ap"], 0.5)


if __name__ == "__main__":
    unittest.main()
