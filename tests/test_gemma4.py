"""Protect development isolation, source balance and resumable Gemma scores."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import (
    BASELINE,
    POLICIES,
    ROOT,
    TASKS,
    choose_routes,
    development_screen,
    question,
    view_indices,
)
from anomaly_eval.gemma4.study import Scores, calibrate, every_source_cutoff, run


def examples():
    return [
        {
            "id": f"{task}/{source}/{label}/{group}",
            "task": task,
            "source": source,
            "target": label,
            "group_id": f"{source}/{label}/{group}",
            "split": "dev",
            "cohort": "main",
            "views": ["a.jpg", "b.jpg", "c.jpg"],
        }
        for task in TASKS
        for source in ["large", "small"]
        for label in [0, 1]
        for group in range(2)
    ]


class GemmaProtocolTests(unittest.TestCase):
    def test_paired_uncertainty_keeps_shared_group_sampling_and_refuses_tiny_sources(
        self,
    ):
        from anomaly_eval.gemma4.uncertainty import interval

        rows = [
            {"id": f"{group}/{label}", "group_id": str(group), "target": label}
            for group in range(4)
            for label in [0, 1]
        ]
        scores = {r["id"]: float(r["target"]) for r in rows}
        result = interval(rows, scores, scores, repetitions=100)
        self.assertEqual(result["paired_delta_ci95"], [0.0, 0.0])
        self.assertEqual(result["optimized_ci95"], [1.0, 1.0])
        self.assertEqual(result["valid_repetitions"], 100)
        self.assertEqual(interval(rows[:4], scores, scores)["status"], "too_few_groups")

    def test_ensemble_fits_primary_development_and_handles_constant_model(self):
        from anomaly_eval.gemma4.ensemble import fit_development, transform

        rows = examples()
        predictions = {
            "useful": {r["id"]: float(r["target"]) for r in rows},
            "constant": {r["id"]: 7.0 for r in rows},
        }
        fitted = fit_development(rows, predictions)
        self.assertEqual(set(fitted["routes"].values()), {"useful"})
        values = transform(
            rows, predictions, fitted, "equal_weight_mean_of_development_z_scores"
        )
        self.assertTrue(
            all(values[r["id"]] == (0.5 if r["target"] else -0.5) for r in rows)
        )
        for field, value in [("split", "test"), ("cohort", "xrf55")]:
            with self.assertRaises(ValueError):
                fit_development([{**r, field: value} for r in rows], predictions)

    def test_gguf_margin_requires_untruncated_post_sampling_probabilities(self):
        import math

        from anomaly_eval.gemma4.llama_backend import LlamaServerBackend

        backend = object.__new__(LlamaServerBackend)
        backend.requests = Mock()
        backend.url = "http://local.invalid"
        backend.model_name = "fixture"
        backend._tokens = [10784, 3771]
        response = backend.requests.post.return_value
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "top_probs": [
                                    {"token": "Yes", "prob": 0.8},
                                    {"token": "No", "prob": 0.2},
                                ]
                            }
                        ]
                    },
                    "message": {"content": "Yes"},
                }
            ]
        }
        with patch.object(backend, "images_for", return_value=[]):
            result = backend._one(examples()[0], BASELINE)
        self.assertAlmostEqual(result["score"], math.log(4))
        self.assertIsNone(result["answer_token_mass"])
        body = backend.requests.post.call_args.kwargs["json"]
        self.assertTrue(body["post_sampling_probs"])
        self.assertEqual(body["samplers"], ["temperature"])
        self.assertEqual(body["temperature"], 1)
        self.assertEqual(body["repeat_penalty"], 1)
        self.assertEqual(body["logit_bias"], {"10784": 100, "3771": 100})

    def test_full_flow_freezes_both_calibrations_before_test_forward(self):
        rows = examples()
        heldout = [{**r, "id": "test/" + r["id"], "split": "test"} for r in rows]
        xrf = [
            {**r, "id": "xrf/" + r["id"], "source": "xrf", "cohort": "xrf55"}
            for r in rows
            if r["task"] == "smoking"
        ]
        all_rows = rows + heldout + xrf
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = output / "samples.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
            atomic_json(
                output / "protocol.json",
                {
                    "manifest_sha256": digest(manifest),
                    "definitions_sha256": digest(
                        ROOT / "anomaly_eval/config/anomalies.json"
                    ),
                    "code_sha256": digest(ROOT / "anomaly_eval/gemma4/protocol.py"),
                    "development_screen_ids": [r["id"] for r in rows],
                },
            )

            class Backend:
                batch_size = 8

                def __init__(self):
                    self.metadata = {"fixture": True}

                def verify_batch(self, rows, policy):
                    return {"fixture": True}

                def score(self, batch, policy):
                    if any(r["split"] == "test" for r in batch):
                        directory = output / "gemma4-e2b"
                        self_case.assertTrue((directory / "selection.json").exists())
                        saved = json.loads((directory / "calibration.json").read_text())
                        self_case.assertEqual(set(saved), {"baseline", "optimized"})
                        for method in saved:
                            self_case.assertEqual(
                                set(saved[method]), set(TASKS) | {"xrf55_source_dev"}
                            )
                    return [{"score": float(r["target"])} for r in batch]

            self_case = self
            with patch("builtins.print"):
                result = run("gemma4-e2b", output, Backend(), {})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["heldout_examples"], len(heldout))

    def test_prompts_cannot_disclose_labels_source_or_review_notes(self):
        row = examples()[0]
        edited = {
            **row,
            "target": 1,
            "source": "secret_dataset_name",
            "split": "test",
            "notes": "positive confirmed by reviewer",
            "group_id": "secret_group",
        }
        for policy in POLICIES:
            self.assertEqual(question(row, policy), question(edited, policy))
            self.assertIn(
                "Exact elapsed duration is not supplied", question(row, policy)
            )
        row["timestamps"] = [0, 15, 30]
        self.assertIn("0, 15, 30", question(row, BASELINE))
        self.assertEqual(
            view_indices({"views": list(range(16))}), [0, 2, 4, 6, 9, 11, 13, 15]
        )

    def test_screen_is_reproducible_and_balances_source_class_groups(self):
        rows = examples()
        selected = development_screen(rows, per_task=4)
        self.assertEqual(selected, development_screen(list(reversed(rows)), per_task=4))
        for task in TASKS:
            members = [r for r in selected if r["task"] == task]
            self.assertEqual(
                {(r["source"], r["target"]) for r in members},
                {("large", 0), ("large", 1), ("small", 0), ("small", 1)},
            )
        for split in ["train", "test", "external"]:
            with self.assertRaises(ValueError):
                development_screen([{**rows[0], "split": split}])

    def test_routes_use_development_only_and_ties_keep_baseline(self):
        rows = examples()
        predictions = {
            p.name: {r["id"]: float(r["target"]) for r in rows} for p in POLICIES
        }
        chosen = choose_routes(rows, predictions)
        self.assertEqual(set(chosen["routes"].values()), {BASELINE.name})
        for field, value in [
            ("split", "test"),
            ("split", "external"),
            ("cohort", "xrf55"),
        ]:
            invalid = copy.deepcopy(rows)
            invalid[0][field] = value
            with self.assertRaises(ValueError):
                choose_routes(invalid, predictions)

    def test_threshold_satisfies_each_source_and_rejects_heldout(self):
        rows = [r for r in examples() if r["task"] == "smoking"]
        scores = {
            r["id"]: r["target"] * 2 + (r["source"] == "large") * 10 for r in rows
        }
        cutoff = every_source_cutoff(rows, scores)
        for source in ["large", "small"]:
            negative = [r for r in rows if r["source"] == source and not r["target"]]
            self.assertLessEqual(
                sum(scores[r["id"]] >= cutoff for r in negative) / len(negative), 0.05
            )
        with self.assertRaises(ValueError):
            calibrate([{**r, "split": "test"} for r in rows], scores)

    def test_cache_resumes_without_rescoring_and_rejects_identity_changes(self):
        class Backend:
            batch_size = 2
            calls = 0

            def score(self, rows, policy):
                self.calls += len(rows)
                return [{"score": float(r["target"])} for r in rows]

        rows = examples()[:3]
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "scores"
            cache = Scores(directory, "manifest", {"revision": "pinned"})
            cache.score(rows[:2], BASELINE, backend, "fixture")
            resumed = Scores(directory, "manifest", {"revision": "pinned"})
            self.assertEqual(
                resumed.score(rows, BASELINE, backend, "fixture"),
                {r["id"]: float(r["target"]) for r in rows},
            )
            self.assertEqual(backend.calls, 3)
            with self.assertRaises(ValueError):
                Scores(directory, "changed", {"revision": "pinned"})
            with self.assertRaises(ValueError):
                resumed.score([{**rows[0], "target": 1}], BASELINE, backend, "fixture")
            record = json.loads(
                (directory / (BASELINE.name + ".jsonl")).read_text().splitlines()[0]
            )
            self.assertEqual(record["split"], "dev")


if __name__ == "__main__":
    unittest.main()
