"""Reject incompatible source runs before writing a combined quality report."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.expanded.combine_accuracy_runs import combine


class AssemblyTests(unittest.TestCase):
    def fixture(self, root, name, model):
        source = root / name
        cohort = source / "smoking_frame" / "dev"
        directory = cohort / model
        directory.mkdir(parents=True)
        state = {
            "status": "complete_for_available_tasks",
            "dataset_fingerprint": "same",
            "benchmarks": [
                {
                    "cohort": "smoking_frame",
                    "split": "dev",
                    "output": "smoking_frame/dev",
                }
            ],
        }
        run = {
            "definitions": {"normal": "normal", "smoking": "smoking"},
            "sample_identity": [{"sha256": "a"}],
            "samples": [{"path": "a.png"}],
            "code_sha256": {"model.py": "same"},
            "metric_version": 2,
            "evaluation_split": "dev",
            "options": {"device": "cuda"},
            "models": [model],
        }
        (source / "evaluation.json").write_text(json.dumps(state))
        (cohort / "run.json").write_text(json.dumps(run))
        (directory / "summary.json").write_text(
            json.dumps({"status": "ok", "model": model})
        )
        return source, cohort, run

    def test_rejects_different_pixels_even_with_same_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, _, _ = self.fixture(root, "a", "mobileclip2-s2")
            b, cohort, original = self.fixture(root, "b", "siglip2-b16-224")
            changed = copy.deepcopy(original)
            changed["sample_identity"][0]["sha256"] = "changed pixels"
            (cohort / "run.json").write_text(json.dumps(changed))
            output = root / "combined"
            with self.assertRaisesRegex(ValueError, "sample_identity"):
                combine([a, b], output)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads((cohort / "run.json").read_text()), changed)

    def test_rejects_different_device_and_overlapping_models(self):
        for mismatch in ("device", "overlap"):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                a, _, _ = self.fixture(root, "a", "mobileclip2-s2")
                b, cohort, run = self.fixture(
                    root,
                    "b",
                    "mobileclip2-s2" if mismatch == "overlap" else "siglip2-b16-224",
                )
                if mismatch == "device":
                    run["options"]["device"] = "cpu"
                    (cohort / "run.json").write_text(json.dumps(run))
                with self.assertRaisesRegex(
                    ValueError, "device" if mismatch == "device" else "overlapping"
                ):
                    combine([a, b], root / "combined")
                self.assertFalse((root / "combined").exists())


if __name__ == "__main__":
    unittest.main()
