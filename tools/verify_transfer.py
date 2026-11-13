"""Verify every file of a transfer bundle (Python 3.12 standard library only).

Copied to the bundle root by tools/build_deployment.py. Run it there:

    python3.12 verify_transfer.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
    seen, total = set(), 0
    for item in manifest["files"]:
        name = item["path"]
        path = ROOT / name
        if name in seen or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Invalid or duplicate package path: " + name)
        seen.add(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing file or unexpected link: " + name)
        if path.stat().st_size != item["bytes"]:
            raise ValueError("File size mismatch: " + name)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError("Checksum mismatch: " + name)
        total += item["bytes"]
    print(f"Verified {len(seen)} packaged files ({total / 1024**3:.2f} GiB).")
    for name, value in manifest["components"].items():
        print(f"  {name}: {value}")
    print("Next: python3.12 deploy.py setup   (see START_HERE.md)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print("Transfer verification failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
