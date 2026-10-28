"""Fetch and extract only AVSS data/annotations from the authors' public bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

URL = (
    "http://www-vpu.eps.uam.es/publications/AODsurvey/AbandonedObjectDetection-CODE.zip"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    metadata = data / "source_metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    archive = metadata / "aodsurvey.zip"
    if not archive.exists():
        with requests.get(URL, stream=True, timeout=(15, 40)) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", 0))
            if size and shutil.disk_usage(data).free < size * 2:
                raise OSError("Insufficient space for archive and selected video data")
            partial = archive.with_suffix(".part")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1048576):
                    stream.write(chunk)
            if size and partial.stat().st_size != size:
                raise ValueError("Incomplete archive")
            partial.replace(archive)
    source = data / "avss_source"
    source.mkdir(exist_ok=True)
    files = []
    wanted = {
        f"AVSSS07_{difficulty}.{extension}"
        for difficulty in ["EASY", "MEDIUM", "HARD"]
        for extension in ["txt", "mpg"]
    }
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = Path(info.filename).name
            use = ("/datasets/AVSS2007/" in info.filename and name in wanted) or (
                info.filename
                in {
                    "AbandonedObjectDetection-main/datasets/annotation_guidelines.txt",
                    "AbandonedObjectDetection-main/datasets/readme.md",
                }
            )
            if not use:
                continue
            destination = source / name
            if not destination.exists():
                with bundle.open(info) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            with bundle.open(info) as src:
                expected = hashlib.file_digest(src, "sha256").hexdigest()
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError("Cached extracted data differ from archive: " + name)
            files.append(
                {
                    "path": str(destination),
                    "archive_path": info.filename,
                    "sha256": actual,
                    "bytes": destination.stat().st_size,
                }
            )
    if len(files) != 8:
        raise ValueError(
            "Expected exactly six video/annotation files and two data documents"
        )
    receipt = {
        "source": "http://www-vpu.eps.uam.es/publications/AODsurvey/",
        "archive_url": URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "files": files,
        "downloaded_code_executed": False,
        "integrity_note": "Local content hashes; no author-published checksum available",
    }
    (metadata / "avss_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("AVSS SOURCE VERIFIED", len(files), "files", flush=True)


if __name__ == "__main__":
    main()
