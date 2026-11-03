"""Portable CPU bundle integrity, isolation and launch contract."""

import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.deployment import cpu


class CpuBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.artifacts = self.root / "artifacts"
        self.cache = self.root / "cache"
        for name in cpu.SOURCE_FILES:
            self.write(self.source / name, b"runtime-fixture")
        for name in cpu.ARTIFACTS:
            self.write(self.artifacts / name, b"model-fixture")
        self.write(
            self.artifacts / cpu.ARTIFACTS[3],
            json.dumps(
                {
                    "definitions_path": "anomaly_eval/config/anomalies.json",
                    "definitions_sha256": cpu.sha256(
                        self.source / "anomaly_eval/config/anomalies.json"
                    ),
                }
            ).encode(),
        )
        self.write(self.artifacts / cpu.ARTIFACTS[-1], b'{"status":"complete"}')
        for name in cpu.BASE_FILES:
            # Mirror HF's symlinked cache: the portable bundle must dereference it.
            blob = self.cache / "blobs" / name
            self.write(blob, b"base-fixture")
            path = self.cache / cpu.CACHE_MODEL / "snapshots" / cpu.REVISION / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(blob)
        self.write(self.artifacts / ".env", b"secret-do-not-copy")
        self.write(self.cache / "token", b"secret-do-not-copy")
        self.output = self.root / "transfer.tar.gz"

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def build(self):
        with (
            patch.object(cpu, "source_files", return_value=sorted(cpu.SOURCE_FILES)),
            patch.object(
                cpu.subprocess, "check_output", return_value=b"commit-fixture\n"
            ),
        ):
            return cpu.build(self.artifacts, self.cache, self.output, self.source)

    def extract(self):
        self.build()
        with tarfile.open(self.output) as archive:
            self.assertTrue(all(p.isfile() or p.isdir() for p in archive.getmembers()))
            archive.extractall(self.root / "relocated", filter="data")
        return self.root / "relocated/capstone-mobileclip2-s2-cpu"

    def test_archive_relocates_and_excludes_unrelated_files(self):
        root = self.extract()
        manifest = cpu.verify(root)
        self.assertEqual(manifest["source_commit"], "commit-fixture")
        self.assertFalse((root / ".env").exists())
        self.assertFalse((root / "token").exists())
        self.assertFalse((root / cpu.SNAPSHOT / cpu.BASE_FILES[0]).is_symlink())
        self.assertEqual(
            self.output.with_name(self.output.name + ".sha256").read_text().split()[0],
            cpu.sha256(self.output),
        )
        self.assertTrue((root / "SHA256SUMS").is_file())

    def test_tampered_and_missing_weights_fail_verification(self):
        root = self.extract()
        weight = root / cpu.ARTIFACTS[0]
        weight.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            cpu.verify(root)
        weight.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            cpu.verify(root)

    def test_missing_input_does_not_publish_an_archive(self):
        (self.artifacts / cpu.ARTIFACTS[0]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing inputs"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_archive_is_preserved(self):
        self.output.write_bytes(b"keep-me")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"keep-me")

    def test_manifest_cannot_escape_root_or_substitute_symlinks(self):
        for name in ["../secret", "/etc/passwd", "safe/../../secret", "..\\secret"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cpu.contained(self.root, name)
        link = self.root / "linked"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            cpu.contained(self.root, "linked/docs/cpu_deployment.md")

    def test_missing_manifest_entries_and_wrong_revision_are_rejected(self):
        root = self.extract()
        path = root / "bundle_manifest.json"
        original = json.loads(path.read_text())
        missing = {
            **original,
            "files": [p for p in original["files"] if p["path"] != cpu.ARTIFACTS[0]],
        }
        path.write_text(json.dumps(missing))
        with self.assertRaisesRegex(ValueError, "Incomplete manifest"):
            cpu.verify(root)
        path.write_text(json.dumps({**original, "base_model": {"revision": "wrong"}}))
        with self.assertRaisesRegex(ValueError, "Unexpected model"):
            cpu.verify(root)

    def test_training_definition_mismatch_fails_build(self):
        (self.source / "anomaly_eval/config/anomalies.json").write_bytes(
            b"new-definitions"
        )
        with self.assertRaisesRegex(ValueError, "training provenance"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_source_allowlist_excludes_datasets_secrets_and_notebook_experiments(self):
        for name in [
            ".env",
            "data/train.csv",
            "notebooks/exp/newcam/train.py",
            "reports/report.md",
            "data-pipeline/samples/videos/loitering.avi",
        ]:
            with self.subTest(name=name):
                self.assertFalse(cpu.runtime_source(name))
        for name in [
            "anomaly_eval/reasoning/backends.py",
            "scripts/strategy/run_strategy_screen.py",
            "data-pipeline/data_pipeline/detection/stage1.py",
        ]:
            self.assertTrue(cpu.runtime_source(name))

    def test_environment_and_launcher_force_cpu_with_local_cache(self):
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_CACHE": "/original-machine"},
        ):
            env = cpu.environment(self.root)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertTrue(env["HF_HUB_CACHE"].startswith(str(self.root)))
        video = self.root / "video.mp4"
        video.touch()
        args = SimpleNamespace(
            action="video",
            video=video,
            out=self.root / "out",
            target_fps=5,
            safety_scan_seconds=10,
        )
        cmd = cpu.command(args, self.root)
        self.assertEqual(cmd[cmd.index("--vlm-device") + 1], "cpu")
        self.assertEqual(cmd[cmd.index("--stage1-device") + 1], "cpu")
        self.assertEqual(cmd[cmd.index("--clip-python") + 1], cpu.sys.executable)
        self.assertNotIn("gemma", " ".join(cmd))


if __name__ == "__main__":
    unittest.main()
