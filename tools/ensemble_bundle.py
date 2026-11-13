"""Build or verify an ensemble classifier folder (default models/ensemble-20261113).

    python tools/ensemble_bundle.py build EXPORT_DIR --hf-cache ~/.cache/huggingface/hub
    python tools/ensemble_bundle.py verify [models/ensemble-20261113]

`build` copies the exported heads and the two pinned image encoders into one
folder with regular files and writes manifest.json (sizes and SHA-256).
`verify` checks that folder with the standard library only.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "models/ensemble-20261113"
ENCODERS = {
    "siglip2-b": (
        "models--google--siglip2-base-patch16-224",
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),
    "mobileclip2-b": (
        "models--timm--MobileCLIP2-B-OpenCLIP",
        ["open_clip_config.json", "open_clip_model.safetensors"],
    ),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(export, hf_cache, output, link=False):
    """link=True symlinks the large encoder files (a development checkout only)."""
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ["ensemble.json", "heads.npz", "smoking_detector.pt"]:
        files[name] = Path(export) / name
    for folder, (repo, names) in ENCODERS.items():
        snapshots = sorted((Path(hf_cache).expanduser() / repo / "snapshots").iterdir())
        if len(snapshots) != 1:
            raise ValueError(f"Expected one cached snapshot of {repo}")
        for name in names:
            files[f"{folder}/{name}"] = snapshots[0] / name
    entries = []
    for relative, source in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if link and relative not in {
            "ensemble.json",
            "heads.npz",
            "smoking_detector.pt",
        }:
            target.symlink_to(source.resolve(strict=True))
        else:
            shutil.copyfile(source.resolve(strict=True), target)
        entries.append(
            {"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target)}
        )
    (output / "manifest.json").write_text(
        json.dumps({"files": entries}, indent=1) + "\n"
    )
    return verify(output)


def verify(directory=DEFAULT):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    for item in manifest["files"]:
        path = (directory / item["path"]).resolve(strict=True)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Ensemble file mismatch: " + item["path"])
    bundle = json.loads((directory / "ensemble.json").read_text())
    if sha256(directory / "heads.npz") != bundle["heads_sha256"]:
        raise ValueError("Ensemble heads do not match ensemble.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("export", type=Path)
    b.add_argument("--hf-cache", type=Path, default=Path("~/.cache/huggingface/hub"))
    b.add_argument("--output", type=Path, default=DEFAULT)
    b.add_argument(
        "--link", action="store_true", help="Symlink encoder weights instead of copying"
    )
    v = sub.add_parser("verify")
    v.add_argument("directory", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    manifest = (
        build(args.export, args.hf_cache, args.output, args.link)
        if args.command == "build"
        else verify(args.directory)
    )
    print(f"Ensemble bundle verified: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
