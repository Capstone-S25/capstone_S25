"""Check stage-two label isolation, source safeguards and evaluation ordering."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from anomaly_eval.gemma4 import optimization as opt
from anomaly_eval.gemma4.protocol import BASELINE, POLICIES, TASKS


class OptimizationTests(unittest.TestCase):
    def test_source_safeguard_rejects_mean_gain_that_harms_a_source(self):
        rows = []
        baseline = {}
        candidate = {}
        for task in TASKS:
            for source, labels, old, new in [
                ("a", [1, 0], [2, 0], [0, 2]),
                ("b", [1, 0, 0, 0], [0, 1, 2, 3], [3, 2, 1, 0]),
            ]:
                for i, (y, b, c) in enumerate(zip(labels, old, new)):
                    ident = f"{task}/{source}/{i}"
                    rows.append(
                        {
                            "id": ident,
                            "task": task,
                            "source": source,
                            "target": y,
                            "split": "dev",
                            "cohort": "main",
                            "group_id": ident,
                        }
                    )
                    baseline[ident] = b
                    candidate[ident] = c
        name = opt.NEW_POLICIES[0].name
        result = opt.fit(
            rows,
            {BASELINE.name: baseline, name: candidate},
            {t: [BASELINE.name, name] for t in TASKS},
        )
        self.assertEqual(set(result["routes"]["full_dev_selected"].values()), {name})
        self.assertEqual(
            set(result["routes"]["source_safeguarded"].values()), {BASELINE.name}
        )
        with self.assertRaisesRegex(ValueError, "primary development"):
            opt.fit([{**r, "split": "test"} for r in rows], {}, {})

    def test_prompt_and_crops_ignore_labels_sources_and_review_metadata(self):
        row = {
            "task": "smoking",
            "views": ["fixture.png"],
            "target": 1,
            "source": "positive",
            "review": "yes",
        }
        inputs = opt.Inputs(
            {
                "fixture.png": [
                    {"label": 1, "confidence": 0.9, "box": [0.4, 0.1, 0.6, 0.9]}
                ]
            }
        )
        other = {
            **row,
            "target": 0,
            "source": "negative",
            "review": "no",
            "split": "external",
        }
        for policy in opt.NEW_POLICIES:
            self.assertEqual(inputs.prompt(row, policy), inputs.prompt(other, policy))
        with patch.object(opt, "_image", return_value=Image.new("RGB", (100, 100))):
            images = inputs.images(row, opt.NEW_POLICIES[2])
            self.assertEqual(len(images), 2)
            self.assertEqual(images[0].size, (100, 100))
            self.assertLess(images[1].width, 100)
            self.assertEqual(
                [i.size for i in images],
                [i.size for i in inputs.images(other, opt.NEW_POLICIES[2])],
            )
        self.assertFalse(opt.eligible(POLICIES[-1], "smoking", [row]))

    def test_new_test_forwards_follow_all_calibrations_and_replay_detects_tampering(
        self,
    ):
        rows = [
            {
                "id": f"{t}/{split}/{y}",
                "task": t,
                "source": "fixture",
                "split": split,
                "cohort": "main",
                "target": y,
                "group_id": f"{t}/{split}/{y}",
                "views": ["fixture.png"],
            }
            for t in TASKS
            for split in ["dev", "test"]
            for y in [0, 1]
        ]
        rows += [
            {
                "id": f"xrf/{split}/{y}",
                "task": "smoking",
                "source": "proxy",
                "split": split,
                "cohort": "xrf55",
                "target": y,
                "group_id": f"xrf/{split}/{y}",
                "views": ["fixture.png"],
            }
            for split in ["dev", "test"]
            for y in [0, 1]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "previous"
            output = root / "new"
            previous.mkdir()
            output.mkdir()
            name = "gemma4-e2b"
            old = previous / name
            (old / "scores").mkdir(parents=True)
            (previous / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            (previous / "protocol.json").write_text(
                json.dumps(
                    {
                        "manifest_sha256": opt.digest(previous / "samples.jsonl"),
                        "development_screen_ids": [
                            r["id"]
                            for r in rows
                            if r["cohort"] == "main" and r["split"] == "dev"
                        ],
                    }
                )
            )
            for policy in POLICIES:
                (old / "scores" / (policy.name + ".jsonl")).write_text(
                    "".join(
                        json.dumps(
                            {
                                **{k: v for k, v in r.items() if k != "views"},
                                "score": 0.0,
                                "answer_token_mass": 1.0,
                                "policy": policy.name,
                            }
                        )
                        + "\n"
                        for r in rows
                    )
                )
            (old / "selection.json").write_text(
                json.dumps({"routes": {t: BASELINE.name for t in TASKS}})
            )
            (old / "results.json").write_text("{}")
            (old / "checkpoint.json").write_text(
                json.dumps({"path": str(old), "backend": "fixture"})
            )
            with (
                patch.object(opt, "MODELS", {name: "fixture"}),
                patch.object(opt, "DETECTIONS", []),
            ):
                protocol = opt.prepare(previous, output)

            class Backend:
                batch_size = 8
                heldout = 0

                def __init__(self):
                    self.metadata = {"backend": "fixture"}

                def score(self, members, policy):
                    for r in members:
                        if r["split"] == "test":
                            self.heldout += 1
                            cutoffs = json.loads(
                                (output / name / "calibration.json").read_text()
                            )
                            self_test.assertEqual(
                                set(cutoffs),
                                {
                                    "full_dev_selected",
                                    "source_safeguarded",
                                    "baseline_blend",
                                },
                            )
                    return [
                        {
                            "score": 0.0
                            if policy in POLICIES
                            else float(2 * r["target"] - 1),
                            "answer_token_mass": 1.0,
                        }
                        for r in members
                    ]

                def verify_batch(self, members, policy):
                    return {"status": "passed"}

                def close(self):
                    pass

            self_test = self
            backend = Backend()
            with patch.object(opt, "make_backend", return_value=backend):
                result = opt.run_model(
                    previous, output, name, protocol, rows, opt.Inputs({})
                )
            self.assertEqual(result["heldout_examples"], 12)
            self.assertGreater(backend.heldout, 0)
            self.assertEqual(
                opt.verify_model(previous, output, name)["status"], "passed"
            )
            path = output / name / "full_dev_selected_predictions.jsonl"
            records = opt.read_rows(path)
            records[0]["score"] += 1
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "Audit numeric mismatch"):
                opt.verify_model(previous, output, name)


if __name__ == "__main__":
    unittest.main()
