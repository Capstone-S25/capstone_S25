"""Real HTTP failure/recovery checks for the catalog downloader; no public network."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import tempfile
import threading
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anomaly_eval.datasets.catalog import (
    DEFAULT_CATALOG,
    load_catalog,
    main,
    run_assets,
    run_external,
)
from anomaly_eval.datasets.transfer import (
    Asset,
    Options,
    TransferError,
    Transfers,
    public_url,
)


def checksum(value):
    return hashlib.sha256(value).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        state = self.server.state
        with state["lock"]:
            state["requests"].append((self.path, self.headers.get("Range")))
            state["active"] += 1
            state["peak"] = max(state["active"], state["peak"])
            count = sum(path == self.path for path, _ in state["requests"])
        try:
            if self.path.startswith("/slow"):
                time.sleep(0.04)
            if self.path == "/missing":
                self.send_response(404)
                self.end_headers()
                return
            if self.path == "/defer":
                self.send_response(429)
                self.send_header("Retry-After", "3600")
                self.end_headers()
                return
            if self.path == "/retry" and count == 1:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            value = state["zip"] if self.path.startswith("/zip") else state["value"]
            requested = self.headers.get("Range")
            start, end = 0, len(value) - 1
            ranged = requested and self.path not in {"/ignore", "/zip-ignore"}
            if ranged:
                first, last = requested.removeprefix("bytes=").split("-")
                start = int(first)
                end = int(last) if last else end
                self.send_response(206)
                self.send_header(
                    "Content-Range",
                    f"bytes {start + 1 if self.path == '/bad-range' else start}-{end}/{len(value)}",
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("ETag", '"fixture-v1"')
            self.end_headers()
            if self.path == "/interrupt" and count == 1:
                self.wfile.write(value[: 2 * 1024**2])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.wfile.write(value[start : end + 1])
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with state["lock"]:
                state["active"] -= 1


class DatasetCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as target:
            target.writestr("large-unused.bin", b"x" * (2 * 1024**2))
            target.writestr("wanted.txt", b"selected member")
        self.server.state = {
            "lock": threading.Lock(),
            "requests": [],
            "active": 0,
            "peak": 0,
            "value": b"fixture content",
            "zip": archive.getvalue(),
        }
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def asset(self, path="result.bin", endpoint="/file", value=None, **kwargs):
        value = self.server.state["value"] if value is None else value
        return Asset(
            "fixture",
            path,
            (self.url + endpoint,),
            checksum(value),
            len(value),
            **kwargs,
        )

    def client(self, **kwargs):
        client = Transfers(self.root, Options(min_free_bytes=0, retries=0, **kwargs))
        self.addCleanup(client.close)
        return client

    def seed_partial(self, asset, value):
        target = self.root / asset.path
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.client().state_paths(asset)
        partial = state["partial"]
        partial.write_bytes(value)
        state["metadata"].parent.mkdir(parents=True, exist_ok=True)
        state["metadata"].write_text(
            json.dumps({"fingerprint": asset.fingerprint, "etag": '"fixture-v1"'})
        )

    def test_cached_rerun_does_not_request_network_and_hashes_when_changed(self):
        client = self.client()
        asset = self.asset()
        self.assertEqual(client.one(asset)["status"], "downloaded")
        with patch(
            "anomaly_eval.datasets.transfer.digest",
            side_effect=AssertionError("Unnecessary hash"),
        ):
            self.assertEqual(
                client.one(asset)["verification"], "unchanged_receipt_stat"
            )
        self.assertEqual(len(self.server.state["requests"]), 1)
        (self.root / asset.path).write_bytes(b"corrupt content")
        self.assertEqual(client.one(asset)["status"], "downloaded")
        self.assertEqual(
            (self.root / asset.path).read_bytes(), self.server.state["value"]
        )

    def test_invalid_receipt_is_repaired_using_pinned_file_hash(self):
        asset = self.asset()
        (self.root / asset.path).write_bytes(self.server.state["value"])
        receipt = self.client().state_paths(asset)["receipt"]
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{broken")
        self.assertEqual(self.client().one(asset)["status"], "cached")
        self.assertEqual(self.server.state["requests"], [])

    def test_cached_only_creates_no_asset_directories_or_receipts(self):
        asset = self.asset("nested/missing.bin")
        self.assertEqual(
            self.client(cached_only=True).one(asset)["status"], "missing_or_invalid"
        )
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(self.server.state["requests"], [])

    def test_verified_audit_detects_same_stat_corruption(self):
        asset = self.asset()
        client = self.client()
        client.one(asset)
        target = self.root / asset.path
        original = target.stat()
        target.write_bytes(b"X" * asset.size)
        import os

        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(
            self.client(cached_only=True, verify=True).one(asset)["status"],
            "missing_or_invalid",
        )

    def test_valid_resume(self):
        asset = self.asset()
        self.seed_partial(asset, self.server.state["value"][:5])
        self.assertEqual(self.client().one(asset)["status"], "downloaded")
        self.assertEqual(self.server.state["requests"][0][1], "bytes=5-")
        self.assertEqual(
            (self.root / asset.path).read_bytes(), self.server.state["value"]
        )

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        asset = self.asset(endpoint="/ignore")
        self.seed_partial(asset, b"fixtu")
        self.client().one(asset)
        self.assertEqual(
            (self.root / asset.path).read_bytes(), self.server.state["value"]
        )

    def test_malformed_range_does_not_publish(self):
        asset = self.asset(endpoint="/bad-range")
        self.seed_partial(asset, b"fixtu")
        with self.assertRaises(TransferError) as caught:
            self.client().one(asset)
        self.assertEqual(caught.exception.status, "integrity_error")
        self.assertFalse((self.root / asset.path).exists())

    def test_dropped_connection_retains_and_resumes_written_bytes(self):
        self.server.state["value"] = b"R" * (3 * 1024**2)
        asset = self.asset(endpoint="/interrupt")
        client = Transfers(
            self.root, Options(retries=1, max_retry_delay=0, min_free_bytes=0)
        )
        self.addCleanup(client.close)
        self.assertEqual(client.one(asset)["status"], "downloaded")
        self.assertEqual(self.server.state["requests"][1][1], "bytes=2097152-")

    def test_transient_http_is_retried(self):
        client = Transfers(
            self.root, Options(retries=1, max_retry_delay=0, min_free_bytes=0)
        )
        self.addCleanup(client.close)
        client.one(self.asset(endpoint="/retry"))
        self.assertEqual(len(self.server.state["requests"]), 2)

    def test_not_found_is_not_retried_but_mirror_fallback_works(self):
        asset = replace(self.asset(), urls=(self.url + "/missing", self.url + "/file"))
        self.assertEqual(self.client().one(asset)["status"], "downloaded")
        self.assertEqual(len(self.server.state["requests"]), 2)

    def test_long_retry_after_defers_host_without_sleeping(self):
        client = self.client()
        started = time.monotonic()
        with self.assertRaises(TransferError) as caught:
            client.one(self.asset(endpoint="/defer"))
        self.assertEqual(caught.exception.status, "deferred")
        with self.assertRaises(TransferError):
            client.one(self.asset("next.bin"))
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(len(self.server.state["requests"]), 1)

    def test_checksum_failure_preserves_previous_valid_filename(self):
        asset = replace(self.asset(), sha256="0" * 64)
        target = self.root / asset.path
        target.write_bytes(b"old data")
        with self.assertRaises(TransferError):
            self.client().one(asset)
        self.assertEqual(target.read_bytes(), b"old data")
        self.assertFalse(self.client().state_paths(asset)["partial"].exists())

    def test_download_metadata_cannot_be_mistaken_for_training_annotations(self):
        asset = self.asset("labels/vid0001.json")
        client = self.client()
        client.one(asset)
        labels = self.root / "labels"
        self.assertEqual(list(labels.glob("vid*.json")), [labels / "vid0001.json"])
        self.assertEqual(list(labels.rglob("*.json")), [labels / "vid0001.json"])
        self.assertTrue(
            client.state_paths(asset)["receipt"].is_relative_to(
                self.root / ".fetch_state"
            )
        )

    def test_workers_overlap_but_obey_host_bound(self):
        options = Options(workers=6, per_host=2, retries=0, min_free_bytes=0)
        assets = [self.asset(f"{i}.bin", endpoint=f"/slow{i}") for i in range(8)]
        result = run_assets(assets, self.root, options)
        self.assertTrue(result["complete"])
        self.assertEqual(result["counts"], {"downloaded": 8})
        self.assertEqual(self.server.state["peak"], 2)

    def test_failure_isolated_and_journal_is_recoverable(self):
        assets = [self.asset("missing.bin", "/missing"), self.asset("good.bin")]
        result = run_assets(assets, self.root, Options(retries=0, min_free_bytes=0))
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"], {"not_found": 1, "downloaded": 1})
        self.assertEqual(len(Path(result["journal"]).read_text().splitlines()), 2)
        self.assertTrue((self.root / "good.bin").exists())

    def test_two_fetchers_coordinate_same_target(self):
        asset = self.asset(endpoint="/slow")
        clients = [self.client(), self.client()]
        with ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(lambda c: c.one(asset), clients))
        self.assertEqual({row["status"] for row in rows}, {"downloaded", "cached"})
        self.assertEqual(len(self.server.state["requests"]), 1)

    def test_cancelled_transfer_makes_no_network_request(self):
        client = self.client()
        client.cancel.set()
        with self.assertRaises(TransferError) as caught:
            client.one(self.asset())
        self.assertEqual(caught.exception.status, "cancelled")
        self.assertEqual(self.server.state["requests"], [])

    def test_payload_budget_and_disk_guard(self):
        with self.assertRaises(TransferError) as caught:
            self.client(max_download_bytes=1).one(self.asset())
        self.assertEqual(caught.exception.status, "deferred_budget")
        with (
            patch(
                "anomaly_eval.datasets.transfer.shutil.disk_usage",
                return_value=type("Disk", (), {"free": 0})(),
            ),
            self.assertRaises(TransferError) as caught,
        ):
            self.client().one(self.asset())
        self.assertEqual(caught.exception.status, "disk_full")
        self.assertEqual(self.server.state["requests"], [])

    def test_selected_zip_member_avoids_whole_archive(self):
        value = b"selected member"
        asset = self.asset(
            "chosen.txt",
            "/zip",
            value,
            kind="zip_member",
            member="wanted.txt",
            archive_size=len(self.server.state["zip"]),
        )
        client = self.client()
        client.one(asset)
        self.assertEqual((self.root / asset.path).read_bytes(), value)
        self.assertLess(client.bytes_received, len(self.server.state["zip"]) // 2)
        self.assertTrue(all(ranged for _, ranged in self.server.state["requests"]))

    def test_zip_host_ignoring_range_never_downloads_full_archive(self):
        asset = self.asset(
            "chosen.txt",
            "/zip-ignore",
            b"selected member",
            kind="zip_member",
            member="wanted.txt",
            archive_size=len(self.server.state["zip"]),
        )
        with self.assertRaises(TransferError):
            self.client().one(asset)
        self.assertFalse((self.root / asset.path).exists())

    def test_zip_crc_corruption_is_rejected(self):
        original = self.server.state["zip"]
        offset = original.index(b"selected member")
        self.server.state["zip"] = original[:offset] + b"X" + original[offset + 1 :]
        asset = self.asset(
            "chosen.txt",
            "/zip",
            b"selected member",
            kind="zip_member",
            member="wanted.txt",
            archive_size=len(original),
        )
        with self.assertRaises(TransferError):
            self.client().one(asset)
        self.assertFalse((self.root / asset.path).exists())

    def test_path_escape_and_symlink_rejected(self):
        client = self.client()
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        for path in [
            "../escape.bin",
            "/absolute.bin",
            "escape/file.bin",
            "back\\slash",
        ]:
            with self.subTest(path=path), self.assertRaises(TransferError):
                client.one(self.asset(path))
        self.assertEqual(self.server.state["requests"], [])

    def custom_catalog(self, rows):
        manifest = self.root / "assets.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        catalog = self.root / "catalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "profiles": {"research": ["fixture"]},
                    "sources": [
                        {
                            "id": "fixture",
                            "manifest": "assets.jsonl",
                            "manifest_sha256": checksum(manifest.read_bytes()),
                            "license": {"id": "fixture"},
                        }
                    ],
                }
            )
        )
        return catalog

    def test_new_source_is_config_only_and_plan_does_not_write_output(self):
        asset = self.asset()
        row = {
            "path": asset.path,
            "urls": list(asset.urls),
            "sha256": asset.sha256,
            "size": asset.size,
        }
        path = self.custom_catalog([row])
        destination = self.root / "untouched"
        with redirect_stdout(io.StringIO()):
            code = main(
                ["--catalog", str(path), "--output", str(destination), "--plan"]
            )
        self.assertEqual(code, 0)
        self.assertFalse(destination.exists())
        self.assertEqual(self.server.state["requests"], [])

    def test_manifest_tamper_and_destination_collisions_rejected(self):
        asset = {"path": "same.bin", "urls": [self.url + "/file"]}
        path = self.custom_catalog([asset, asset])
        with self.assertRaises(ValueError):
            load_catalog(path)
        path = self.custom_catalog([asset])
        (self.root / "assets.jsonl").write_text("{}\n")
        with self.assertRaises(ValueError):
            load_catalog(path)

    def test_sidecar_and_parent_file_collisions_rejected(self):
        for first, second in [
            ("file.bin", ".file.bin.fetch.part"),
            ("nested", "nested/file.bin"),
        ]:
            with self.subTest(first=first):
                path = self.custom_catalog(
                    [
                        {"path": name, "urls": [self.url + "/file"]}
                        for name in [first, second]
                    ]
                )
                with self.assertRaises(ValueError):
                    load_catalog(path)

    def test_interruption_drains_workers_and_retains_partial_receipt(self):
        assets = [self.asset(f"{i}.bin", "/slow") for i in range(10)]
        with patch("anomaly_eval.datasets.catalog.wait", side_effect=KeyboardInterrupt):
            result = run_assets(
                assets, self.root, Options(workers=1, retries=0, min_free_bytes=0)
            )
        self.assertTrue(result["interrupted"])
        self.assertFalse(result["complete"])
        self.assertLess(result["processed_files"], result["planned_files"])
        self.assertEqual(
            json.loads((self.root / "fetch_all_receipt.json").read_text())[
                "interrupted"
            ],
            True,
        )

    def test_unpinned_partial_from_different_mirror_is_not_appended(self):
        asset = replace(self.asset(), sha256=None)
        self.seed_partial(asset, b"wrong old bytes")
        # No saved URL identity: a validator from another mirror is insufficient.
        self.client().one(asset)
        self.assertIsNone(self.server.state["requests"][0][1])
        self.assertEqual(
            (self.root / asset.path).read_bytes(), self.server.state["value"]
        )

    def test_unpinned_url_version_change_does_not_adopt_old_cache(self):
        asset = replace(self.asset(), sha256=None)
        client = self.client()
        client.one(asset)
        self.server.state["value"] = b"changed content"
        newer = replace(asset, urls=(self.url + "/version2",))
        self.assertEqual(client.one(newer)["status"], "downloaded")
        self.assertEqual((self.root / asset.path).read_bytes(), b"changed content")
        self.assertEqual(len(self.server.state["requests"]), 2)

    def test_live_providers_are_deferred_for_bounded_audits(self):
        args = SimpleNamespace(
            cached_only=False, limit=1, retry_failed=None, max_download_gb=None
        )
        with patch(
            "anomaly_eval.datasets.drive.main",
            side_effect=AssertionError("Unexpected provider network"),
        ):
            result = run_external(
                {"id": "drive_original", "provider": "drive"}, args, {}
            )
        self.assertEqual(result["status"], "deferred")

    def test_drive_provider_uses_selected_data_root_and_limits(self):
        args = SimpleNamespace(
            cached_only=False,
            limit=None,
            retry_failed=None,
            max_download_gb=None,
            output=self.root,
            workers=8,
            per_host=2,
            retries=1,
        )
        source = {
            "id": "drive_original",
            "provider": "drive",
            "homepage": "https://drive.google.com/drive/folders/fixture",
        }
        with patch("anomaly_eval.datasets.drive.main", return_value=0) as provider:
            result = run_external(source, args, {})
        argv = provider.call_args.args[0]
        self.assertEqual(argv[argv.index("--output") + 1], str(self.root / "raw"))
        self.assertEqual(argv[argv.index("--workers") + 1], "2")
        self.assertEqual(result["status"], "ok")

    def test_retry_failed_selects_only_previous_failures(self):
        path = self.custom_catalog(
            [
                {"path": "a.bin", "urls": [self.url + "/file"]},
                {"path": "b.bin", "urls": [self.url + "/file"]},
            ]
        )
        receipt = self.root / "previous.json"
        receipt.write_text(
            json.dumps(
                {
                    "files": [
                        {"source": "fixture", "path": "a.bin", "status": "cached"},
                        {"source": "fixture", "path": "b.bin", "status": "failed"},
                    ]
                }
            )
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                ["--catalog", str(path), "--plan", "--retry-failed", str(receipt)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["files"], 1)

    def test_complete_packaged_catalog_is_portable_and_has_attribution(self):
        _, sources, assets, _ = load_catalog(DEFAULT_CATALOG, ["all"])
        self.assertGreaterEqual(len(sources), 17)
        self.assertGreater(len(assets), 4000)
        for asset in assets:
            self.assertFalse(asset.path.startswith("data/"))
            self.assertIsNotNone(asset.sha256)
            self.assertIsNotNone(asset.size)
            if asset.source == "commons_photos":
                self.assertTrue(asset.attribution.get("source_page"))
                self.assertTrue(asset.attribution.get("LicenseShortName"))

    def test_public_receipts_omit_url_secrets(self):
        result = public_url(
            "https://name:secret@example.com/a?api_key=secret&token=secret&datasetVersionNumber=6&download=1"
        )
        self.assertEqual(
            result, "https://example.com/a?datasetVersionNumber=6&download=1"
        )


if __name__ == "__main__":
    unittest.main()
