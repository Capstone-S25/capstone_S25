"""Fetch pinned Gemma 4 evaluation checkpoints and verify publisher file hashes."""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import DEFAULT_OUTPUT, ROOT


def main(argv=None):
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/models/gemma4_hub")
    parser.add_argument(
        "--ram-cache", type=Path, default=Path("/dev/shm/capstone_gemma4_20261027")
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    specs = json.loads((ROOT / "anomaly_eval/config/gemma4_models.json").read_text())[
        "models"
    ]
    if args.models != ["all"]:
        if set(args.models) - {s["name"] for s in specs}:
            parser.error("Unknown model name")
        specs = [s for s in specs if s["name"] in args.models]
    if args.workers < 1:
        parser.error("workers must be positive")

    def one(spec):
        cache = args.cache if spec["cache"] == "workspace" else args.ram_cache
        print("FETCH", spec["name"], spec["repo"], flush=True)
        location = Path(
            snapshot_download(
                spec["repo"],
                revision=spec["revision"],
                cache_dir=cache,
                max_workers=4,
                token=False,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.gguf",
                    "*.jinja",
                    "tokenizer.model",
                    "*.txt",
                    "README.md",
                ],
            )
        )
        files = []
        for expected in spec["files"]:
            path = location / expected["path"]
            if (
                path.stat().st_size != expected["bytes"]
                or digest(path) != expected["sha256"]
            ):
                raise ValueError("Checkpoint size/hash mismatch: " + spec["name"])
            files.append(expected)
        result = {
            **spec,
            "path": str(location),
            "verified_files": files,
            "catalog_sha256": digest(ROOT / "anomaly_eval/config/gemma4_models.json"),
        }
        atomic_json(args.output / spec["name"] / "checkpoint.json", result)
        print("VERIFIED CHECKPOINT", spec["name"], location, flush=True)
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(one, specs):
            pass


def runtime_main(argv=None):
    """Install the pinned official llama.cpp CUDA runtime."""
    parser = argparse.ArgumentParser(description=runtime_main.__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config = ROOT / "anomaly_eval/config/gemma4_runtime.json"
    spec = json.loads(config.read_text())
    destination = ROOT / spec["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / spec["archive"]
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    spec["url"], headers={"User-Agent": "capstone-gemma-evaluation"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    partial.open("wb") as stream,
                ):
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
                if (
                    partial.stat().st_size != spec["bytes"]
                    or digest(partial) != spec["sha256"]
                ):
                    raise ValueError("Runtime archive identity mismatch")
                partial.replace(archive)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    if archive.stat().st_size != spec["bytes"] or digest(archive) != spec["sha256"]:
        raise ValueError("Pinned runtime archive differs")
    if not (destination / spec["binary"]).exists():
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
    atomic_json(
        args.output / "llama_runtime_verification.json",
        {
            **spec,
            "config_sha256": digest(config),
            "binary_sha256": digest(destination / spec["binary"]),
        },
    )
    print(destination / spec["binary"])


if __name__ == "__main__":
    main()
