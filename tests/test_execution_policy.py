"""Placement, native batch equivalence and uncapped preparation regressions."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from anomaly_eval.data import _receipt_indices, original_name, resolve_original
from anomaly_eval.datasets.labels import LabeledImage
from anomaly_eval.datasets.prepare import prepare, read_reviews
from anomaly_eval.metrics import average_precision
from anomaly_eval.models import MODELS, SigLIPAdapter
from anomaly_eval.resources import CPUSettings
from anomaly_eval.runtime import move_inputs, resolve_device, resolve_dtype


class ExecutionTests(unittest.TestCase):
    def test_auto_placement_and_explicit_unavailable_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda")
        with self.assertRaises(ValueError):
            resolve_device("other")

    def test_cpu_controls_preserve_visible_gpu_selection(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            CPUSettings(threads=8).configure_environment()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_dtype_policy_and_integer_tokens(self):
        self.assertEqual(resolve_dtype("auto", "cpu", True), torch.float32)
        self.assertEqual(resolve_dtype("auto", "cuda", False), torch.float32)
        with self.assertRaises(ValueError):
            resolve_dtype("float16", "cpu", True)
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.ones(1, 3, 4, 4),
        }
        moved = move_inputs(inputs, "cpu", torch.bfloat16)
        self.assertEqual(moved["input_ids"].dtype, torch.int64)
        self.assertEqual(moved["pixel_values"].dtype, torch.bfloat16)

    def test_average_precision_matches_sklearn_with_ties(self):
        from sklearn.metrics import average_precision_score

        y, scores = [0, 1, 1, 0, 1, 0], [0.8, 0.8, 0.4, 0.4, 0.1, 0.0]
        self.assertAlmostEqual(
            average_precision(list(zip(y, scores))), average_precision_score(y, scores)
        )

    def test_siglip_batch_matches_individual_native_scores(self):
        from PIL import Image
        from transformers import SiglipConfig, SiglipModel

        torch.manual_seed(77)
        torch.set_num_threads(2)
        native = SiglipModel(
            SiglipConfig(
                text_config={
                    "vocab_size": 10,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_position_embeddings": 64,
                },
                vision_config={
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 16,
                    "patch_size": 4,
                },
            )
        ).eval()
        tokens = torch.tensor([[1] * 63 + [2], [1] * 63 + [3]])
        pixels = torch.rand(2, 3, 16, 16)
        images = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16), "white")]

        class Processor:
            def __call__(self, **kwargs):
                if "text" in kwargs:
                    return {"input_ids": tokens}
                batch = kwargs["images"]
                return {
                    "pixel_values": torch.stack(
                        [pixels[images.index(im)] for im in batch]
                    )
                }

        with (
            patch(
                "transformers.AutoProcessor.from_pretrained", return_value=Processor()
            ),
            patch("transformers.AutoModel.from_pretrained", return_value=native),
        ):
            adapter = SigLIPAdapter(
                MODELS["siglip2-b16-224"],
                {"normal": "normal", "smoking": "smoking"},
                {},
            )
        batched = adapter.predict_batch(images)
        for im, actual in zip(images, batched):
            single = adapter.predict(im)
            self.assertEqual(actual.label, single.label)
            self.assertAlmostEqual(actual.anomaly_score, single.anomaly_score, places=5)
            torch.testing.assert_close(
                torch.tensor(list(actual.scores.values())),
                torch.tensor(list(single.scores.values())),
            )


class PreparationTests(unittest.TestCase):
    def test_receipt_index_reuses_parsing_and_invalidates_after_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "renamed.png"
            actual.write_bytes(b"first")
            receipt = root / "download_receipt.json"
            entry = {"original_path": "original.png", "path": "renamed.png"}
            receipt.write_text(json.dumps({"renamed_files": [entry]}))
            _receipt_indices.cache_clear()
            self.assertEqual(resolve_original(root / "original.png"), actual)
            self.assertEqual(original_name(actual), "original.png")
            self.assertEqual(_receipt_indices.cache_info().misses, 1)
            second = root / "conflict.png"
            second.write_bytes(b"second")
            receipt.write_text(
                json.dumps(
                    {
                        "renamed_files": [
                            entry,
                            {"original_path": "original.png", "path": "conflict.png"},
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous original"):
                resolve_original(root / "original.png")
            self.assertEqual(_receipt_indices.cache_info().misses, 2)

    def test_zero_preparation_cap_retains_all_unique_frames(self):
        from PIL import Image

        from anomaly_eval.paths import Workspace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for i in range(24):
                path = root / f"{i}.png"
                Image.new("RGB", (8, 8), (i, 20, 40)).save(path)
                samples.append(
                    LabeledImage(path, "normal" if i < 12 else "smoking", "recording")
                )
            workspace = Workspace.current()
            from dataclasses import replace

            with (
                patch(
                    "anomaly_eval.datasets.prepare.collect_smoking",
                    return_value=(samples[12:], samples[:12]),
                ),
                patch(
                    "anomaly_eval.datasets.prepare.Workspace.current",
                    return_value=replace(workspace, bundle_dir=root / "no_bundle"),
                ),
            ):
                full = prepare(
                    root / "raw", root / "additional", root / "full", per_group=0
                )
                capped = prepare(
                    root / "raw", root / "additional", root / "capped", per_group=8
                )
            self.assertEqual(full["confirmed_rows"], 24)
            self.assertEqual(capped["confirmed_rows"], 16)
            self.assertEqual(full["sampling"]["available_unique_rows"], 24)

    def test_review_import_rejects_unknown_protocol(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            csv = root / "review.csv"
            csv.write_text(
                "path,label,task,group_id,source,evidence,review_note,protocol\na.png,smoking,smoking,g1,local,human_review,Reviewed smoking,video\n"
            )
            with self.assertRaisesRegex(ValueError, "protocol"):
                read_reviews(csv)


if __name__ == "__main__":
    unittest.main()
