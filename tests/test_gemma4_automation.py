"""Exercise the five-model combination and completed-run reporting paths."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anomaly_eval.gemma4 import suite as run_gemma4_suite
from anomaly_eval.gemma4.ensemble import evaluate
from anomaly_eval.gemma4.protocol import BASELINE, MODELS, ROOT, TASKS


class GemmaAutomationTests(unittest.TestCase):
    def test_all_five_cached_models_produce_calibrated_combinations(self):
        rows = [
            {
                "id": f"{task}/{split}/{y}",
                "task": task,
                "split": split,
                "source": "fixture",
                "group_id": f"{task}/{split}/{y}",
                "cohort": "main",
                "target": y,
            }
            for task in TASKS
            for split in ["dev", "test"]
            for y in [0, 1]
        ]
        rows += [
            {
                "id": f"xrf/{split}/{y}",
                "task": "smoking",
                "split": split,
                "source": "proxy",
                "group_id": f"xrf/{split}/{y}",
                "cohort": "xrf55",
                "target": y,
            }
            for split in ["dev", "test"]
            for y in [0, 1]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            (output / "protocol.json").write_text('{"fixture": true}')
            for filename, source in [
                ("ensemble_protocol.json", "gemma4_ensembles.json"),
                ("safeguard_protocol.json", "gemma4_safeguard.json"),
            ]:
                (output / filename).write_bytes(
                    (ROOT / "anomaly_eval/config" / source).read_bytes()
                )
            for i, name in enumerate(MODELS):
                directory = output / name
                (directory / "scores").mkdir(parents=True)
                (directory / "results.json").write_text("{}")
                (directory / "selection.json").write_text(
                    json.dumps({"routes": {task: BASELINE.name for task in TASKS}})
                )
                (directory / "scores" / (BASELINE.name + ".jsonl")).write_text(
                    "".join(
                        json.dumps({**r, "score": (2 * r["target"] - 1) * (i + 1)})
                        + "\n"
                        for r in rows
                    )
                )
            result = evaluate(output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["heldout_examples"], 12)
            self.assertEqual(len(result["methods"]), 3)
            for tasks in result["methods"].values():
                for task in TASKS:
                    value = tasks[task]["source_metrics"]["fixture"][
                        "dev_every_source_fpr_budget"
                    ]
                    self.assertEqual(
                        (value["ap"], value["fpr"], value["recall"]), (1, 0, 1)
                    )
            fitted = json.loads((output / "ensembles/selection.json").read_text())
            self.assertEqual(
                set(fitted["baseline_safeguard"]["routes"].values()),
                {"gemma4-e2b/baseline"},
            )

    def test_complete_suite_always_regenerates_report_and_honors_audit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in MODELS:
                directory = output / name
                directory.mkdir()
                (directory / "checkpoint.json").write_text('{"path":"unused"}')
                (directory / "results.json").write_text("{}")
            with (
                patch("sys.argv", ["suite", "--output", tmp]),
                patch.object(
                    run_gemma4_suite, "update_report", return_value=0
                ) as report,
            ):
                self.assertEqual(run_gemma4_suite.main(), 0)
                report.assert_called_once_with(output)
            with (
                patch("sys.argv", ["suite", "--output", tmp]),
                patch.object(run_gemma4_suite, "update_report", return_value=1),
            ):
                self.assertEqual(run_gemma4_suite.main(), 1)
            self.assertEqual(
                json.loads((output / "suite_status.json").read_text())["phase"],
                "finished_with_failures",
            )

    def test_resume_preserves_checkpoint_spelling_in_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            directory = output / "gemma4-e2b"
            directory.mkdir()
            (directory / "checkpoint.json").write_text('{"path":"/absolute/model"}')
            (directory / "runtime.json").write_text('{"checkpoint":"relative/model"}')

            def launch(command, **_kwargs):
                (directory / "results.json").write_text("{}")
                self.assertEqual(
                    command[command.index("--checkpoint") + 1], "relative/model"
                )
                return SimpleNamespace(pid=123, wait=lambda: 0)

            with (
                patch("sys.argv", ["suite", "--output", tmp, "--models", "gemma4-e2b"]),
                patch.object(
                    run_gemma4_suite.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                patch.object(run_gemma4_suite.subprocess, "Popen", side_effect=launch),
                patch.object(run_gemma4_suite, "update_report", return_value=0),
            ):
                self.assertEqual(run_gemma4_suite.main(), 0)


if __name__ == "__main__":
    unittest.main()
