"""Regression coverage for VadCLIP source identity, chronology and validation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from anomaly_eval.vadclip.data import (
    SOURCE_IDENTITY_POLICY,
    SmokingSourceError,
    load_smoking_sessions,
    make_three_way_split,
)
from anomaly_eval.vadclip.extract_features import main as extract_main


def frame(index, clock="03-00-00-000_PM", camera="c4", export="a"):
    suffix = f"_{clock}" if clock is not None else ""
    return f"frame_{camera}_{index:06d}_2025-10-28{suffix}_jpg.rf.{export}.jpg"


def corpus(root, rows, split="train"):
    folder = Path(root) / "smoking" / "smoking-video.v1i.coco" / split
    folder.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for image_id, (name, label) in enumerate(rows, start=1):
        (folder / name).write_bytes(name.encode())
        images.append({"id": image_id, "file_name": name})
        labels = label if isinstance(label, tuple) else (label,)
        annotations.extend(
            {"image_id": image_id, "category_id": 1 if y else 2}
            for y in labels
            if y is not None
        )
    source = folder / "_annotations.coco.json"
    source.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "Smoking"},
                    {"id": 2, "name": "Not Smoking"},
                ],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return folder


class SmokingSourceTests(unittest.TestCase):
    def test_restarted_index_at_distinct_times_keeps_both_labels(self):
        with tempfile.TemporaryDirectory() as root:
            later = frame(284, "03-00-00-000_PM")
            earlier = frame(284, "10-00-00-000_AM", export="b")
            corpus(root, [(later, 1)])
            corpus(root, [(earlier, 0)], "valid")
            sessions = load_smoking_sessions(root)
            self.assertEqual(list(sessions), ["c4:2025-10-28"])
            session = sessions["c4:2025-10-28"]
            self.assertEqual([Path(p).name for p in session.frames], [earlier, later])
            self.assertEqual(session.labels, [0, 1])
            self.assertEqual(session.frame_indices, [284, 284])
            self.assertEqual(len(set(session.source_ids)), 2)

    def test_clock_orders_restarted_indices_and_handles_noon_midnight_milliseconds(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            rows = [
                (frame(1, "12-00-00-002_PM"), 1),
                (frame(100, "12-00-00-001_AM"), 0),
                (frame(50, "11-59-59-999_AM"), 1),
                (frame(2, "12-00-00-001_PM"), 0),
            ]
            corpus(root, rows)
            session = load_smoking_sessions(root)["c4:2025-10-28"]
            self.assertEqual(session.frame_indices, [100, 50, 2, 1])
            self.assertEqual(session.labels, [0, 1, 0, 1])
            self.assertEqual(
                session.timestamps,
                [
                    "2025-10-28T00:00:00.001",
                    "2025-10-28T11:59:59.999",
                    "2025-10-28T12:00:00.001",
                    "2025-10-28T12:00:00.002",
                ],
            )
            self.assertEqual(session.ordering, "capture_timestamp")

    def test_same_clock_with_distinct_indices_retains_distinct_frames(self):
        with tempfile.TemporaryDirectory() as root:
            corpus(root, [(frame(1), 0), (frame(2), 1)])
            session = load_smoking_sessions(root)["c4:2025-10-28"]
            self.assertEqual(session.frame_indices, [1, 2])

    def test_matching_duplicate_exports_count_once_across_original_splits(self):
        with tempfile.TemporaryDirectory() as root:
            corpus(root, [(frame(284, export="b"), 1)])
            corpus(root, [(frame(284, export="a"), 1)], "valid")
            report = {}
            session = load_smoking_sessions(root, report=report)["c4:2025-10-28"]
            self.assertEqual(len(session.frames), 1)
            self.assertEqual(session.labels, [1])
            self.assertEqual(report["duplicate_exports"], 1)
            self.assertEqual(report["images_listed"], 2)

    def test_all_real_conflicts_are_reported_without_relabeling(self):
        with tempfile.TemporaryDirectory() as root:
            positive = [(frame(i, export="positive"), 1) for i in (284, 285)]
            negative = [(frame(i, export="negative"), 0) for i in (284, 285)]
            train = corpus(root, positive)
            valid = corpus(root, negative, "valid")
            before = {
                p: p.read_bytes()
                for p in (
                    train / "_annotations.coco.json",
                    valid / "_annotations.coco.json",
                )
            }
            report = {}
            with self.assertRaises(SmokingSourceError) as caught:
                load_smoking_sessions(root, report=report)
            self.assertIs(caught.exception.report, report)
            self.assertEqual(len(report["issues"]), 2)
            self.assertEqual(report["status"], "error")
            for issue in report["issues"]:
                self.assertEqual(issue["kind"], "conflicting_annotation")
                self.assertEqual(
                    {r["label"] for r in issue["records"]}, {"Smoking", "Not Smoking"}
                )
                self.assertTrue(all("image_id" in r for r in issue["records"]))
                self.assertTrue(
                    all(Path(r["annotations"]).is_file() for r in issue["records"])
                )
            self.assertIn(positive[0][0], str(caught.exception))
            self.assertIn(negative[0][0], str(caught.exception))
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_legacy_indices_remain_ordered_and_conflicts_still_fail(self):
        with tempfile.TemporaryDirectory() as root:
            corpus(root, [(frame(2, None), 1), (frame(1, None), 0)])
            session = load_smoking_sessions(root)["c4:2025-10-28"]
            self.assertEqual(session.frame_indices, [1, 2])
            self.assertEqual(session.timestamps, [None, None])
            self.assertEqual(session.ordering, "frame_index")
            corpus(root, [(frame(1, None, export="copy"), 1)], "valid")
            with self.assertRaisesRegex(SmokingSourceError, "Conflicting annotation"):
                load_smoking_sessions(root)

    def test_mixed_timestamp_and_legacy_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            corpus(root, [(frame(1), 1), (frame(2, None), 0)])
            with self.assertRaises(SmokingSourceError) as caught:
                load_smoking_sessions(root)
            self.assertEqual(
                caught.exception.report["issues"][0]["kind"], "ambiguous_order"
            )

    def test_invalid_or_incomplete_clock_is_rejected(self):
        for clock in ("13-00-00-000_PM", "10-60-00-000_AM", "10-00-00-000"):
            with self.subTest(clock=clock), tempfile.TemporaryDirectory() as root:
                corpus(root, [(frame(1, clock), 1)])
                with self.assertRaises(SmokingSourceError) as caught:
                    load_smoking_sessions(root)
                self.assertEqual(
                    caught.exception.report["issues"][0]["kind"], "invalid_timestamp"
                )

    def test_missing_labeled_image_fails_and_unknown_labels_are_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            folder = corpus(root, [(frame(1), 1), (frame(2), None)])
            report = {}
            session = load_smoking_sessions(root, report=report)["c4:2025-10-28"]
            self.assertEqual(session.labels, [1])
            self.assertEqual(report["unlabeled_images"], 1)
            (folder / frame(1)).unlink()
            with self.assertRaises(SmokingSourceError) as caught:
                load_smoking_sessions(root)
            self.assertEqual(caught.exception.report["missing_images"], 1)

    def test_smoking_and_nonsmoking_people_in_one_image_is_positive(self):
        with tempfile.TemporaryDirectory() as root:
            corpus(root, [(frame(1), (0, 1))])
            session = load_smoking_sessions(root)["c4:2025-10-28"]
            self.assertEqual(session.labels, [1])


class SmokingValidationCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            (frame(284, "03-00-00-000_PM"), 1),
            (frame(284, "10-00-00-000_AM", export="b"), 0),
            (frame(1, camera="c1"), 0),
            (frame(1, camera="c2"), 1),
        ]
        corpus(self.root, self.rows)

    def test_validation_only_needs_no_weights_and_preserves_group_split(self):
        report_path = self.root / "review" / "sources.json"
        corpus(self.root / "raw" / "datasets", self.rows)
        # Default roots remain dynamic; no explicit --root is necessary.
        with (
            patch.dict(os.environ, {"CAPSTONE_DATA_DIR": str(self.root)}),
            patch("anomaly_eval.vadclip.extract_features.load_clip_b16") as encoder,
            patch(
                "anomaly_eval.vadclip.extract_features.torch.set_num_threads"
            ) as threads,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = extract_main(
                ["--validate-only", "--validation-report", str(report_path)]
            )
        self.assertEqual(status, 0)
        encoder.assert_not_called()
        threads.assert_not_called()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["retained_frames"], 4)
        self.assertEqual(report["session_count"], 3)
        self.assertEqual(report["identity_policy"], SOURCE_IDENTITY_POLICY)
        self.assertEqual(
            report["splits"], make_three_way_split(list(report["sessions"]), 0.25, 0.2)
        )
        self.assertEqual(
            len({k for values in report["splits"].values() for k in values}), 3
        )

    def test_failed_validation_saves_all_records_and_does_not_create_feature_cache(
        self,
    ):
        corpus(self.root, [(frame(284, export="conflict"), 0)], "valid")
        report_path = self.root / "failed_sources.json"
        cache = self.root / "features"
        with (
            patch("anomaly_eval.vadclip.extract_features.load_clip_b16") as encoder,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = extract_main(
                [
                    "--root",
                    str(self.root),
                    "--weights",
                    "missing.pt",
                    "--cache-dir",
                    str(cache),
                    "--validation-report",
                    str(report_path),
                ]
            )
        self.assertEqual(status, 1)
        self.assertFalse(cache.exists())
        encoder.assert_not_called()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["issues"][0]["kind"], "conflicting_annotation")
        self.assertEqual(len(report["issues"][0]["records"]), 2)

    def test_empty_or_insufficient_corpus_is_not_a_valid_training_split(self):
        corpus(self.root, [(frame(1), 1)])
        report_path = self.root / "split_error.json"
        with contextlib.redirect_stderr(io.StringIO()):
            status = extract_main(
                [
                    "--root",
                    str(self.root),
                    "--validate-only",
                    "--validation-report",
                    str(report_path),
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(report_path.read_text())["issues"][0]["kind"], "invalid_split"
        )

    def test_extraction_rebuilds_legacy_cache_and_reuses_new_ordered_features(self):
        cache = self.root / "features"
        cache.mkdir()
        (cache / "manifest.json").write_text(
            json.dumps({"schema": 2, "fingerprint": "legacy"})
        )
        weights = self.root / "clip.pt"
        weights.write_bytes(b"test encoder weights")
        arguments = [
            "--root",
            str(self.root),
            "--weights",
            str(weights),
            "--cache-dir",
            str(cache),
        ]

        def features(model, preprocess, frames, batch_size, device):
            return np.arange(len(frames) * 512, dtype=np.float32).reshape(-1, 512)

        with (
            patch("anomaly_eval.vadclip.extract_features.load_clip_b16") as encoder,
            patch("anomaly_eval.vadclip.extract_features.make_preprocess"),
            patch(
                "anomaly_eval.vadclip.extract_features.extract_session",
                side_effect=features,
            ) as extract,
            patch("anomaly_eval.vadclip.extract_features.torch.set_num_threads"),
            patch(
                "anomaly_eval.vadclip.extract_features.torch.set_num_interop_threads"
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            extract_main(arguments)
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(extract.call_count, 3)
            manifest = json.loads((cache / "manifest.json").read_text())
            self.assertNotEqual(manifest["fingerprint"], "legacy")
            self.assertEqual(manifest["identity_policy"], SOURCE_IDENTITY_POLICY)
            session = manifest["sessions"]["c4:2025-10-28"]
            self.assertEqual(session["frame_indices"], [284, 284])
            self.assertEqual(session["ordering"], "capture_timestamp")
            self.assertEqual(
                session["timestamps"],
                ["2025-10-28T10:00:00.000", "2025-10-28T15:00:00.000"],
            )
            with np.load(cache / session["file"]) as values:
                self.assertEqual(values["labels"].tolist(), [0, 1])
            encoder.reset_mock()
            extract.reset_mock()
            extract_main(arguments)
            encoder.assert_not_called()
            extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
