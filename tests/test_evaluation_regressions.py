"""Regression checks for evaluation correctness, leakage, cache integrity and padding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import f1_score

from anomaly_eval.analysis import analyze, calibrate
from anomaly_eval.data import file_sha256, read_manifest, resolve_original
from anomaly_eval.datasets.fetch import extract_safe
from anomaly_eval.datasets.prepare import assign_splits
from anomaly_eval.metrics import detection_metrics, frame_metrics, select_threshold
from anomaly_eval.vadclip.data import (
    build_clip,
    expand_scores,
    load_smoking_sessions,
    make_three_way_split,
)
from anomaly_eval.vadclip.model import CLIPVAD
from anomaly_eval.vadclip.train import alignment_loss, load_clips, snapshot_state


def row(path, label, pred, status="ok", score=None, group="g"):
    return {
        "path": str(path),
        "label": label,
        "prediction": pred,
        "status": status,
        "anomaly_score": score,
        "repeat": 0,
        "group_id": group,
    }


class MetricRegressionTests(unittest.TestCase):
    def test_abstention_is_a_false_negative_and_does_not_hide_support(self):
        m = detection_metrics(
            [
                row("a", "smoking", "smoking"),
                row("b", "smoking", None, "abstain"),
                row("c", "normal", "normal"),
            ],
            ["normal", "smoking", "unattended_items"],
        )
        self.assertEqual(m["binary"]["f1"], 2 / 3)
        self.assertEqual(m["binary"]["recall"], 0.5)
        self.assertEqual(m["per_class"]["smoking"]["support"], 2)
        self.assertEqual(m["scored_labels"], ["normal", "smoking"])
        self.assertEqual(m["unsupported_labels"], ["unattended_items"])

    def test_all_abstain_retains_all_ground_truth(self):
        m = detection_metrics(
            [row("a", "normal", None, "abstain"), row("b", "smoking", None, "error")],
            ["normal", "smoking"],
        )
        self.assertEqual(m["macro_f1"], 0)
        self.assertEqual(m["binary"]["fn"], 1)
        self.assertEqual(m["binary"]["tn"], 0)
        self.assertEqual(m["binary"]["unanswered_normal"], 1)

    def test_macro_classes_never_depend_on_model_predictions(self):
        labels = ["normal", "smoking", "unattended_items"]
        a = detection_metrics([row("a", "smoking", "smoking")], labels)
        b = detection_metrics([row("a", "smoking", "unattended_items")], labels)
        self.assertEqual(a["scored_labels"], b["scored_labels"])

    def test_threshold_matches_reported_precision_and_f1_including_ties(self):
        for scores in ([0.1, 0.9], [0.5, 0.5]):
            y = np.array([0, 1])
            s = np.array(scores)
            m = frame_metrics(y, s)
            self.assertAlmostEqual(m["f1"], f1_score(y, s >= m["threshold"]))
            json.dumps(m, allow_nan=False)
        self.assertEqual(select_threshold([0, 1], [0.1, 0.9]), 0.9)

    def test_evaluation_rejects_fractional_training_targets(self):
        with self.assertRaises(ValueError):
            frame_metrics([0.5, 0.5], [0.1, 0.9])

    def test_original_truth_and_positive_count_survive_all_representative_lengths(self):
        y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        features = np.arange(32, dtype=np.float32).reshape(8, 4)
        for length in (2, 3, 8, 16):
            _, soft, real = build_clip(features, y, length)
            score = expand_scores(soft[:real], len(y))
            m = frame_metrics(y, score, 0.5)
            self.assertEqual(m["n_pos"], 4)
            self.assertEqual(m["n"], 8)

    def test_scaled_alignment_loss_remains_stable_without_probability_clamping(self):
        logits = torch.tensor(
            [[[-1000.0, 1000.0], [1000.0, -1000.0]]], requires_grad=True
        )
        labels = torch.tensor([[0.5, 1.0]])
        loss = alignment_loss(logits, labels)
        torch.testing.assert_close(loss, torch.tensor([[1000.0, 2000.0]]))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_best_cpu_snapshot_survives_optimizer_updates(self):
        model = torch.nn.Linear(1, 1, bias=False)
        before = model.weight.detach().clone()
        saved = snapshot_state(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        torch.testing.assert_close(saved["weight"], before)
        self.assertFalse(torch.equal(model.weight, before))


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 8)

    def encode_token(self, t):
        return self.embedding(t)

    def encode_text(self, e, t):
        return e.sum(1)


def tiny_tokenize(text):
    t = torch.zeros(len(text), 77, dtype=torch.long)
    t[:, 0] = 1
    t[:, 1] = torch.arange(3, 3 + len(text))
    t[:, 2] = 99
    return t


class TemporalPaddingTests(unittest.TestCase):
    def test_padded_content_cannot_change_real_outputs_and_gradients_are_finite(self):
        torch.set_num_threads(2)
        with patch("anomaly_eval.vadclip.model.tokenize", side_effect=tiny_tokenize):
            for graph in (False, True):
                torch.manual_seed(7)
                model = CLIPVAD(
                    2,
                    embed_dim=8,
                    visual_width=8,
                    visual_length=8,
                    visual_layers=1,
                    attn_window=3,
                    clip_model=TinyClip(),
                    use_graph=graph,
                )
                model.train()
                x = torch.randn(2, 8, 8)
                lens = torch.tensor([2, 5])
                changed = x.clone()
                for i, n in enumerate(lens):
                    changed[i, n:] = 100 * torch.randn_like(changed[i, n:])
                a = model(x, None, ["normal", "smoking"], lens)
                b = model(changed, None, ["normal", "smoking"], lens)
                self.assertFalse(
                    torch.isneginf(model.build_attention_mask(3)).all(1).any()
                )
                for i, n in enumerate(lens):
                    torch.testing.assert_close(a[1][i, :n], b[1][i, :n])
                    torch.testing.assert_close(a[3][i, :n], b[3][i, :n])
                (a[1].sum() + a[3].sum()).backward()
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )


class SelectionAndCacheTests(unittest.TestCase):
    def test_task_splits_keep_protocols_and_historical_groups_together(self):
        rows = [
            {"group_id": g, "label": label, "protocol": p}
            for g in ["new1", "new2", "old"]
            for label in ["normal", "unattended_items"]
            for p in ["frame", "window"]
        ]
        assign_splits(rows, 42, ["old"])
        for g in ["new1", "new2", "old"]:
            self.assertEqual(len({r["split"] for r in rows if r["group_id"] == g}), 1)
        self.assertEqual({r["split"] for r in rows if r["group_id"] == "old"}, {"dev"})
        self.assertEqual({r["split"] for r in rows}, {"dev", "test"})

    def test_source_split_is_independent_of_input_listing_order(self):
        rows = [
            {"group_id": g, "label": l}
            for g in ("positive1", "positive2")
            for l in ("normal", "unattended_items")
        ]
        rows += [{"group_id": "walk" + str(i), "label": "normal"} for i in range(3)]
        original = assign_splits([dict(r) for r in rows], 42)
        reversed_rows = assign_splits([dict(r) for r in reversed(rows)], 42)
        self.assertEqual(
            {r["group_id"]: r["split"] for r in original},
            {r["group_id"]: r["split"] for r in reversed_rows},
        )

    def test_three_way_split_is_disjoint_and_preserves_all_groups(self):
        keys = [str(i) for i in range(16)]
        s = make_three_way_split(keys)
        self.assertEqual(
            sorted([k for group in s.values() for k in group]), sorted(keys)
        )
        self.assertEqual(len({k for group in s.values() for k in group}), 16)
        self.assertTrue(all(s.values()))

    def test_missing_smoking_annotation_is_unknown_not_normal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "smoking" / "smoking-video.v1i.coco" / "train"
            folder.mkdir(parents=True)
            (folder / "a.jpg").write_bytes(b"a")
            (folder / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "categories": [{"id": 1, "name": "person"}],
                        "images": [{"id": 1, "file_name": "a.jpg"}],
                        "annotations": [{"image_id": 1, "category_id": 1}],
                    }
                )
            )
            self.assertEqual(load_smoking_sessions(root), {})

    def test_drive_image_and_label_ids_resolve_via_original_filename(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "datasets" / "images" / "sample__drive_img.jpg"
            label = root / "datasets" / "labels" / "sample__drive_label.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_text("0 .5 .5 .1 .1")
            entries = [
                {
                    "original_path": "datasets/images/sample.jpg",
                    "path": "datasets/images/" + image.name,
                },
                {
                    "original_path": "datasets/labels/sample.txt",
                    "path": "datasets/labels/" + label.name,
                },
            ]
            (root / "download_receipt.json").write_text(
                json.dumps({"renamed_files": entries})
            )
            self.assertEqual(
                resolve_original(root / "datasets/images/sample.jpg"), image
            )
            self.assertEqual(
                resolve_original(root / "datasets/labels/sample.txt"), label
            )
            image2 = image.with_name("sample__other.jpg")
            image2.write_bytes(b"different")
            entries.append(
                {
                    "original_path": "datasets/images/sample.jpg",
                    "path": "datasets/images/" + image2.name,
                }
            )
            (root / "download_receipt.json").write_text(
                json.dumps({"renamed_files": entries})
            )
            with self.assertRaises(ValueError):
                resolve_original(root / "datasets/images/sample.jpg")

    def test_identical_image_bytes_cannot_cross_dev_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.jpg").write_bytes(b"same")
            (root / "b.jpg").write_bytes(b"same")
            manifest = root / "m.csv"
            manifest.write_text(
                "path,label,split,group_id\na.jpg,normal,dev,g1\nb.jpg,smoking,test,g2\n"
            )
            with self.assertRaisesRegex(ValueError, "Content leakage"):
                read_manifest(manifest, {"normal": "n", "smoking": "s"})

    def test_feature_cache_rejects_old_schema_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                load_clips(root)
            f = root / "f.npz"
            np.savez(f, feat=np.ones((2, 4), dtype=np.float32), labels=np.array([0, 1]))
            manifest = {
                "schema": 2,
                "train": ["train"],
                "dev": ["dev"],
                "test": ["test"],
                "sessions": {
                    "train": {
                        "file": "f.npz",
                        "sha256": file_sha256(f),
                        "split": "train",
                    }
                },
            }
            manifest["sessions"]["dev"] = {
                "file": "f.npz",
                "sha256": file_sha256(f),
                "split": "dev",
            }
            manifest["sessions"]["test"] = {
                "file": "f.npz",
                "sha256": file_sha256(f),
                "split": "test",
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            f.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_clips(root)

    def test_tar_traversal_and_links_rejected(self):
        import tarfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "bad.tar"
            with tarfile.open(p, "w") as t:
                member = tarfile.TarInfo("../outside")
                member.type = tarfile.DIRTYPE
                t.addfile(member)
            with self.assertRaises(ValueError):
                extract_safe(p, root / "extract")


class StatisticalPipelineTests(unittest.TestCase):
    def make_run(self, root, split, models, rows):
        root.mkdir()
        defs = {"normal": "n", "smoking": "s"}
        samples = [
            {
                "path": r["path"],
                "label": r["label"],
                "group_id": r["group_id"],
                "split": "test",
            }
            for r in rows[models[0]]
        ]
        run = {
            "definitions": defs,
            "metric_version": 2,
            "experiment_fingerprint": "fixed",
            "evaluation_split": split,
            "models": models,
            "samples": samples,
            "options": {"repeats": 1},
        }
        (root / "run.json").write_text(json.dumps(run))
        for m in models:
            (root / m).mkdir()
            (root / m / "predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows[m])
            )
            (root / m / "summary.json").write_text(
                json.dumps({"metrics": detection_metrics(rows[m], list(defs))})
            )

    def test_cluster_bootstrap_is_paired_and_rejects_missing_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            models = ["a", "b"]
            a = [
                row("n", "normal", "normal", score=0.1, group="g1"),
                row("p", "smoking", "smoking", score=0.9, group="g2"),
            ]
            b = [{**r, "prediction": "normal"} for r in a]
            self.make_run(root, "test", models, {"a": a, "b": b})
            result = analyze(root, 50, 42)
            self.assertEqual(result["unit"], "source_group")
            self.assertEqual(result["n_groups"], 2)
            self.assertEqual(
                result["paired_differences"]["a minus b"][
                    "accuracy_including_failures"
                ]["point"],
                0.5,
            )
            (root / "a" / "predictions.jsonl").write_text(json.dumps(a[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                analyze(root, 10, 42)

    def test_test_labels_cannot_tune_operating_point(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dev = root / "dev"
            test = root / "test"
            self.make_run(
                dev,
                "dev",
                ["a"],
                {
                    "a": [
                        row("dn", "normal", "normal", score=0.2, group="dev1"),
                        row("dp", "smoking", "smoking", score=0.8, group="dev2"),
                    ]
                },
            )
            self.make_run(
                test,
                "test",
                ["a"],
                {
                    "a": [
                        row("tn", "normal", "normal", score=0.1, group="test1"),
                        row("tp", "smoking", "smoking", score=0.7, group="test2"),
                    ]
                },
            )
            r = calibrate(dev, test)
            m = r["models"]["a"]
            self.assertEqual(m["thresholds"]["dev_max_f1"], 0.8)
            self.assertEqual(
                m["operating_points"]["dev_max_f1"]["all_samples"]["binary"]["fn"], 1
            )
            self.assertEqual(
                m["operating_points"]["dev_max_f1"]["score_metrics"][
                    "threshold_source"
                ],
                "development",
            )


if __name__ == "__main__":
    unittest.main()
