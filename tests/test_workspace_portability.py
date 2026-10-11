"""Server relocation, installed CLI dispatch and location-independent artifacts."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anomaly_eval.cli import main as cli_main
from anomaly_eval.data import (
    experiment_options,
    fingerprint,
    read_manifest,
    sample_identity,
)
from anomaly_eval.datasets.prepare import prepare
from anomaly_eval.paths import (
    PACKAGE_ROOT,
    configure,
    portable_path,
    resolve_path,
)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_server_config_relative_paths_and_flag_precedence(self):
        config = self.root / "settings" / "server.json"
        config.parent.mkdir()
        config.write_text(
            json.dumps(
                {
                    "data_dir": "../data",
                    "results_dir": "../runs",
                    "cache_dir": "../cache",
                }
            )
        )
        os.environ["CAPSTONE_DATA_DIR"] = str(self.root / "environment")
        workspace = configure(config, {"data_dir": self.root / "flag"})
        self.assertEqual(workspace.data_dir, self.root / "flag")
        self.assertEqual(workspace.results_dir, self.root / "runs")
        self.assertEqual(workspace.cache_dir, self.root / "cache")
        self.assertEqual(os.environ["HF_HUB_CACHE"], str(self.root / "cache" / "hub"))

    def test_each_runtime_root_is_independent_and_exportable(self):
        kwargs = {
            field: self.root / field
            for field in (
                "data_dir",
                "results_dir",
                "cache_dir",
                "bundle_dir",
                "config_dir",
            )
        }
        workspace = configure(overrides=kwargs)
        for field, path in kwargs.items():
            self.assertEqual(getattr(workspace, field), path)
        self.assertEqual(
            resolve_path("${DATA_ROOT}/raw/image.jpg"),
            self.root / "data_dir" / "raw" / "image.jpg",
        )
        self.assertEqual(
            portable_path(self.root / "data_dir" / "raw" / "image.jpg"),
            "${DATA_ROOT}/raw/image.jpg",
        )

    def test_relocating_a_manifest_preserves_image_identity_and_fingerprint(self):
        from PIL import Image

        for server in ("server_a", "server_b"):
            data = self.root / server / "data"
            (data / "raw").mkdir(parents=True)
            Image.new("RGB", (10, 10), color="red").save(data / "raw" / "positive.png")
            (data / "prepared").mkdir()
            (data / "prepared" / "manifest.csv").write_text(
                "path,label,split,group_id\n${DATA_ROOT}/raw/positive.png,smoking,test,recording-1\n"
            )
        identities = []
        for server in ("server_a", "server_b"):
            data = self.root / server / "data"
            configure(overrides={"data_dir": data})
            samples = read_manifest(
                data / "prepared" / "manifest.csv",
                {"normal": "normal", "smoking": "smoking"},
            )
            identities.append(sample_identity(samples))
            self.assertEqual(samples[0].path, data / "raw" / "positive.png")
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(fingerprint(identities[0]), fingerprint(identities[1]))

    def test_location_flags_do_not_change_experiment_identity(self):
        a = {
            "output": "/old/run",
            "manifest": "/old/data.csv",
            "cache_dir": "/old/cache",
            "data_dir": "/old/data",
            "threads": 4,
            "seed": 42,
        }
        b = {
            **a,
            "output": "/new/run",
            "manifest": "/new/data.csv",
            "cache_dir": "/new/cache",
            "data_dir": "/new/data",
        }
        self.assertEqual(
            fingerprint(experiment_options(a)), fingerprint(experiment_options(b))
        )
        self.assertNotEqual(
            fingerprint(experiment_options(a)),
            fingerprint(experiment_options({**a, "threads": 2})),
        )

    def test_explicit_maps_rebase_legacy_linux_and_windows_paths(self):
        configure(
            path_maps=[
                f"/old/server/data={self.root}/data",
                f"C:\\capstone\\data={self.root}/data",
            ]
        )
        self.assertEqual(
            resolve_path("/old/server/data/raw/x.jpg"),
            self.root / "data" / "raw" / "x.jpg",
        )
        self.assertEqual(
            resolve_path("C:\\capstone\\data\\raw\\x.jpg"),
            self.root / "data" / "raw" / "x.jpg",
        )
        with self.assertRaises(ValueError):
            resolve_path("D:\\other\\x.jpg")

    def test_relative_config_paths_do_not_depend_on_launch_directory(self):
        config = self.root / "server.json"
        config.write_text('{"data_dir":"data"}')
        with patch("pathlib.Path.cwd", return_value=Path("/")):
            self.assertEqual(configure(config).data_dir, self.root / "data")

    def test_bad_config_fails_before_computation(self):
        config = self.root / "server.json"
        for value in ({"unknown": "value"}, {"data_dir": None}, {"data_dir": ""}):
            config.write_text(json.dumps(value))
            with self.assertRaises((ValueError, TypeError)):
                configure(config)

    def test_cli_does_not_confuse_a_directory_named_run_with_command(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = cli_main(["--data-root", str(self.root / "run"), "paths"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(out.getvalue())["data_dir"], str(self.root / "run"))

    def test_cli_permission_error_identifies_path_and_returns_failure(self):
        denied = self.root / "restricted" / "runs"
        stderr = io.StringIO()
        with (
            patch(
                "anomaly_eval.runner.main",
                side_effect=PermissionError(13, "Permission denied", str(denied)),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = cli_main(["run", "--prepare-only"])
        self.assertEqual(result, 1)
        self.assertIn(str(denied), stderr.getvalue())
        self.assertIn("--results-root", stderr.getvalue())
        self.assertIn("--prepared-dir", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_checkout_launcher_works_from_an_unrelated_directory(self):
        command = [
            sys.executable,
            str(PACKAGE_ROOT.parent / "scripts" / "evaluate.py"),
            "--data-root",
            str(self.root / "data"),
            "paths",
        ]
        result = subprocess.run(
            command, cwd=self.root, capture_output=True, text=True, check=True
        )
        self.assertEqual(json.loads(result.stdout)["data_dir"], str(self.root / "data"))

    def test_prepared_manifests_are_relative_and_relocate_with_bundle(self):
        # Use the curated bundle, no source-video downloads or model weights.
        bundle = PACKAGE_ROOT.parent / "test_data"
        configure(overrides={"bundle_dir": bundle, "data_dir": self.root / "data"})
        dataset = prepare(
            self.root / "missing_raw",
            self.root / "missing_additional",
            self.root / "data" / "evaluation",
        )
        self.assertTrue(
            all(
                not Path(value).is_absolute() for value in dataset["manifests"].values()
            )
        )
        original = dataset["fingerprint"]
        moved = self.root / "other_server" / "test_data"
        shutil.copytree(bundle, moved)
        configure(
            overrides={
                "bundle_dir": moved,
                "data_dir": self.root / "other_server" / "data",
            }
        )
        other = prepare(
            self.root / "missing_raw",
            self.root / "missing_additional",
            self.root / "other_server" / "data" / "evaluation",
        )
        self.assertEqual(original, other["fingerprint"])
        p = (
            self.root
            / "other_server"
            / "data"
            / "evaluation"
            / other["manifests"]["unattended_items_frame"]
        )
        images = read_manifest(
            p, {"normal": "normal", "unattended_items": "unattended"}
        )
        self.assertTrue(all(s.path.is_relative_to(moved) for s in images))

    def test_copied_feature_cache_keeps_content_fingerprint(self):
        import numpy as np

        from anomaly_eval.data import file_sha256
        from anomaly_eval.vadclip.train import load_clips

        cache = self.root / "cache"
        cache.mkdir()
        metadata = {}
        for group in ("train", "dev", "test"):
            f = cache / (group + ".npz")
            np.savez(f, feat=np.ones((2, 4), dtype=np.float32), labels=np.array([0, 1]))
            metadata[group] = {"file": f.name, "sha256": file_sha256(f), "split": group}
        (cache / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "train": ["train"],
                    "dev": ["dev"],
                    "test": ["test"],
                    "sessions": metadata,
                    "fingerprint": "stable-content",
                }
            )
        )
        moved = self.root / "new_server" / "cache"
        shutil.copytree(cache, moved)
        first, a = load_clips(cache)
        second, b = load_clips(moved)
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        for group in metadata:
            np.testing.assert_array_equal(first[group]["feat"], second[group]["feat"])


class RunnerModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def run_mode(self, *arguments):
        # Prepare real bundle cohorts; capture child commands without model weights.
        with (
            patch(
                "anomaly_eval.runner.run_stage", return_value={"status": "ok"}
            ) as stage,
            patch("anomaly_eval.runner.analyze"),
            patch("anomaly_eval.analysis.generate"),
            patch("anomaly_eval.runner.calibrate"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = cli_main(
                [
                    "--data-root",
                    str(self.root / "data"),
                    "--results-root",
                    str(self.root / "results"),
                    "--cache-root",
                    str(self.root / "cache"),
                    "run",
                    "--models",
                    "mobileclip2-s2",
                    *arguments,
                ]
            )
        run = next((self.root / "results").iterdir())
        return (
            code,
            json.loads((run / "evaluation.json").read_text()),
            [call.args[0] for call in stage.call_args_list],
        )

    def test_full_evaluation_never_passes_a_rate_cap_and_uses_all_inputs(self):
        code, state, commands = self.run_mode()
        self.assertEqual(code, 0)
        self.assertEqual(state["options"]["protocols"], ["frame", "window"])
        self.assertEqual(state["options"]["bootstrap"], 2000)
        self.assertTrue(commands)
        for command in commands:
            self.assertNotIn("--max-samples-per-second", command)
            self.assertEqual(command[command.index("--duty-cycle") + 1], "1.0")
            self.assertEqual(command[command.index("--limit") + 1], "0")
        for record in state["benchmarks"]:
            selection = record["selection"]
            self.assertEqual(
                selection["selected_samples"], selection["available_samples"]
            )

    def test_quick_mode_preserves_both_classes_with_a_shared_small_cohort(self):
        code, state, commands = self.run_mode("--mode", "quick")
        self.assertEqual(code, 0)
        self.assertEqual(state["options"]["protocols"], ["frame"])
        self.assertEqual(state["options"]["bootstrap"], 200)
        self.assertEqual(len(commands), 3)  # smoking dev, unattended dev/test
        for command in commands:
            self.assertNotIn("--max-samples-per-second", command)
            self.assertEqual(command[command.index("--limit") + 1], "8")
        for record in state["benchmarks"]:
            selection = record["selection"]
            self.assertEqual(selection["selected_samples"], 8)
            self.assertEqual(len(selection["selected_label_counts"]), 2)
        self.assertFalse(state["full_task_coverage"])

    def test_deployment_explicitly_passes_rate_and_duty_on_a_capped_cohort(self):
        code, state, commands = self.run_mode(
            "--mode", "deployment", "--camera-fps", "2", "--duty-cycle", "0.5"
        )
        self.assertEqual(code, 0)
        self.assertEqual(state["options"]["limit"], 16)
        for command in commands:
            self.assertEqual(
                command[command.index("--max-samples-per-second") + 1], "2.0"
            )
            self.assertEqual(command[command.index("--duty-cycle") + 1], "0.5")

    def test_explicit_options_override_the_quick_preset(self):
        code, state, commands = self.run_mode(
            "--mode",
            "quick",
            "--limit",
            "4",
            "--protocols",
            "window",
            "--bootstrap",
            "40",
        )
        self.assertEqual(code, 0)
        self.assertEqual(state["options"]["bootstrap"], 40)
        self.assertEqual(len(commands), 2)
        self.assertTrue(
            all(r["selection"]["selected_samples"] == 4 for r in state["benchmarks"])
        )

    def test_missing_five_task_coverage_stops_before_any_model_is_started(self):
        code, state, commands = self.run_mode("--require-all-tasks")
        self.assertEqual(code, 2)
        self.assertEqual(state["status"], "incomplete_data")
        self.assertFalse(commands)
        self.assertEqual(len(state["requested_task_coverage"]), 5)
        self.assertEqual(
            state["requested_task_coverage"]["rough_sleeping"]["available_protocols"],
            [],
        )

    def test_coverage_only_counts_requested_input_protocols(self):
        code, state, commands = self.run_mode(
            "--tasks", "smoking", "--protocols", "window", "--require-all-tasks"
        )
        self.assertEqual(code, 2)
        self.assertFalse(state["full_task_coverage"])
        self.assertFalse(commands)


if __name__ == "__main__":
    unittest.main()
