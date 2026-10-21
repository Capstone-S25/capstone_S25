"""Install a pinned portable LaTeX compiler without changing system packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.17.0"
ASSET = f"tectonic-{VERSION}-x86_64-unknown-linux-musl.tar.gz"
URL = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/{ASSET}"
ARCHIVE_SHA = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/tools/tectonic",
    )
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit(
            "This pinned bootstrap is Linux x86_64; install Tectonic or XeLaTeX for your platform."
        )
    args.prefix.mkdir(parents=True, exist_ok=True)
    archive = args.prefix / ASSET
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != ARCHIVE_SHA:
            raise ValueError(
                "Downloaded compiler archive does not match the official pinned release digest"
            )
        partial.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Cached compiler archive hash mismatch")
    with tarfile.open(archive) as source:
        members = [
            m
            for m in source.getmembers()
            if m.isfile() and Path(m.name).name == "tectonic"
        ]
        if len(members) != 1:
            raise ValueError("Unexpected compiler archive contents")
        binary = args.prefix / "tectonic"
        binary.write_bytes(source.extractfile(members[0]).read())
        binary.chmod(0o755)
    receipt = {
        "release": f"tectonic@{VERSION}",
        "url": URL,
        "archive_sha256": ARCHIVE_SHA,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "installation": "Workspace-local portable binary; no system package changes",
    }
    (args.prefix / "install_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("Compiler ready:", binary)


if __name__ == "__main__":
    main()
