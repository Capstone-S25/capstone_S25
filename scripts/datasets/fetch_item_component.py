"""Acquire only pinned public data files for the small-item component study."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/event_improvements_20261017"
REVISION = "966439fcac1103b392c49d4135f128860a39d1f8"
EXPECTED_SHA256 = "a08b52d24dea014fd508f5901cab9da0f145bdfe0b5c965b9746bad9edec29ec"
URL = (
    "https://codeload.github.com/hangsuuuu/video-of-abandoned-object-detection/zip/"
    + REVISION
)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "author_source.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Source archive hash does not match the pinned receipt")
        partial.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Cached source archive hash mismatch")
    destination = DATA / "author_source"
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe source archive path")
            if member.is_dir():
                continue
            if path.suffix not in {".mp4", ".json", ".md"} and path.name not in {
                "annotation",
                "Annotation",
                "video3",
                "video5",
                "video6",
            }:
                raise ValueError("Unexpected non-data source file: " + str(path))
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            value = source.read(member)
            if target.exists():
                if target.read_bytes() != value:
                    raise ValueError("Existing source file differs: " + str(target))
            else:
                target.write_bytes(value)
    receipt = {
        "url": URL,
        "sha256": EXPECTED_SHA256,
        "bytes": archive.stat().st_size,
        "repository": "https://github.com/hangsuuuu/video-of-abandoned-object-detection",
        "revision": REVISION,
    }
    path = DATA / "receipt.json"
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("VERIFIED SOURCE", archive.stat().st_size, EXPECTED_SHA256)


if __name__ == "__main__":
    main()
