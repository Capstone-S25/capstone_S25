"""Protect model range assembly from ignored, truncated or oversized responses."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from anomaly_eval.gemma4.download import part_digest, range_into


def response(code=206, content_range="bytes 3-5/8", blocks=(b"d", b"ef")):
    value = Mock()
    value.__enter__ = Mock(return_value=value)
    value.__exit__ = Mock(return_value=False)
    value.status_code = code
    value.headers = {"Content-Range": content_range}
    value.iter_content.return_value = iter(blocks)
    return value


class GemmaDownloadTests(unittest.TestCase):
    def test_exact_range_preserves_neighboring_bytes_and_authenticates_part(self):
        client = Mock()
        client.request.return_value = response()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.part"
            path.write_bytes(b"00000000")
            sha = range_into(
                client, "https://public.invalid/model", path, 3, 5, 8, attempts=1
            )
            self.assertEqual(path.read_bytes(), b"000def00")
            self.assertEqual(sha, hashlib.sha256(b"def").hexdigest())
            self.assertEqual(part_digest(path, 3, 3), sha)
            self.assertEqual(
                client.request.call_args.kwargs["headers"]["Range"], "bytes=3-5"
            )

    def test_wrong_range_ignored_range_or_oversize_cannot_overwrite_neighbors(self):
        for bad in [
            response(code=200),
            response(content_range="bytes 0-2/8"),
            response(blocks=(b"defg",)),
        ]:
            client = Mock()
            client.request.return_value = bad
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.part"
                path.write_bytes(b"00000000")
                with self.assertRaises(RuntimeError):
                    range_into(
                        client,
                        "https://public.invalid/model",
                        path,
                        3,
                        5,
                        8,
                        attempts=1,
                    )
                self.assertEqual(path.read_bytes(), b"00000000")

    def test_retry_replaces_truncated_part_without_corrupting_adjacent_parts(self):
        client = Mock()
        client.request.side_effect = [response(blocks=(b"X",)), response()]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("anomaly_eval.gemma4.download.time.sleep"),
        ):
            path = Path(tmp) / "model.part"
            path.write_bytes(b"00000000")
            range_into(
                client, "https://public.invalid/model", path, 3, 5, 8, attempts=2
            )
            self.assertEqual(path.read_bytes(), b"000def00")


if __name__ == "__main__":
    unittest.main()
