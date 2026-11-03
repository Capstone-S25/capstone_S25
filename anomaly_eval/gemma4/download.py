"""Bounded parallel HTTP ranges for a pinned Gemma checkpoint stalled in Xet."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from anomaly_eval.datasets.transfer import (
    Options,
    TransferError,
    Transfers,
    atomic_json,
    digest,
)
from anomaly_eval.gemma4.protocol import DEFAULT_OUTPUT, ROOT


def range_into(client, url, path, start, end, total, attempts=4):
    expected = end - start + 1
    for attempt in range(attempts):
        try:
            with client.request(
                url,
                stream=True,
                headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"},
            ) as response:
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{total}"
                ):
                    raise ValueError(
                        "Server did not honor the exact requested byte range"
                    )
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected range compression")
                received = 0
                sha = hashlib.sha256()
                with path.open("r+b", buffering=0) as stream:
                    stream.seek(start)
                    for block in response.iter_content(1024 * 1024):
                        if received + len(block) > expected:
                            raise ValueError(
                                "Range response exceeded the declared part"
                            )
                        if stream.write(block) != len(block):
                            raise OSError("Incomplete range write")
                        sha.update(block)
                        received += len(block)
                if received != expected:
                    raise ValueError("Range response ended early")
                return sha.hexdigest()
        except (requests.RequestException, OSError, TransferError, ValueError) as error:
            if (
                attempt + 1 == attempts
                or isinstance(error, TransferError)
                and not error.retryable
            ):
                raise RuntimeError(
                    f"Part {start}-{end} failed: {type(error).__name__}"
                ) from None
            time.sleep(min(8, 2**attempt))
    raise AssertionError("Unreachable transfer state")


def part_digest(path, start, length):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        while length:
            block = stream.read(min(length, 4 * 1024 * 1024))
            if not block:
                raise ValueError("Truncated cached part")
            sha.update(block)
            length -= len(block)
    return sha.hexdigest()


def main(argv=None):
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/models/gemma4_hub")
    parser.add_argument(
        "--ram-cache", type=Path, default=Path("/dev/shm/capstone_gemma4_20261027")
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or args.chunk_mib < 1:
        parser.error("Use 1–16 workers and positive chunk size")
    specs = json.loads((ROOT / "anomaly_eval/config/gemma4_models.json").read_text())[
        "models"
    ]
    spec = next((s for s in specs if s["name"] == args.model), None)
    if spec is None or spec["backend"] != "transformers" or len(spec["files"]) != 1:
        parser.error(
            "This fallback supports the catalog's single-file native checkpoints"
        )
    cache = args.cache if spec["cache"] == "workspace" else args.ram_cache
    snapshot = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            cache_dir=cache,
            token=False,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "tokenizer.model",
                "*.txt",
                "README.md",
            ],
        )
    )
    expected = spec["files"][0]
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir(exist_ok=True)
    target = blobs / expected["sha256"]
    partial = blobs / (expected["sha256"] + ".parallel.part")
    state_path = blobs / (expected["sha256"] + ".parallel.json")
    size = expected["bytes"]
    chunk = args.chunk_mib * 1024**2
    parts = [
        (start, min(size - 1, start + chunk - 1)) for start in range(0, size, chunk)
    ]
    identity = {
        "sha256": expected["sha256"],
        "bytes": size,
        "chunk": chunk,
        "revision": spec["revision"],
    }
    client = Transfers(cache, Options(workers=args.workers, per_host=args.workers))
    with (blobs / (expected["sha256"] + ".parallel.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not target.exists():
            completed = {}
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["identity"] != identity:
                    raise ValueError("Cached parallel-transfer identity differs")
                completed = state["completed"]
            if not partial.exists():
                # Xet uses unique temp names. Reuse the largest existing prefix;
                # final publisher SHA verification authenticates every byte.
                previous = sorted(
                    blobs.glob(expected["sha256"] + "*.incomplete"),
                    key=lambda p: p.stat().st_size,
                    reverse=True,
                )
                prefix = previous[0].stat().st_size if previous else 0
                if previous and 0 < prefix <= size:
                    shutil.copyfile(previous[0], partial)
                    for index, (start, end) in enumerate(parts):
                        if end < prefix:
                            completed[str(index)] = part_digest(
                                partial, start, end - start + 1
                            )
                else:
                    partial.touch()
                with partial.open("r+b") as stream:
                    stream.truncate(size)
            for index, (start, end) in enumerate(parts):
                key = str(index)
                if (
                    key in completed
                    and part_digest(partial, start, end - start + 1) != completed[key]
                ):
                    completed.pop(key)
            atomic_json(state_path, {"identity": identity, "completed": completed})
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{expected['path']}"
            try:
                with ThreadPoolExecutor(args.workers) as pool:
                    futures = {
                        pool.submit(
                            range_into, client, url, partial, start, end, size
                        ): index
                        for index, (start, end) in enumerate(parts)
                        if str(index) not in completed
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        completed[str(index)] = future.result()
                        atomic_json(
                            state_path, {"identity": identity, "completed": completed}
                        )
                        print(
                            spec["name"],
                            f"{len(completed)}/{len(parts)} parts",
                            flush=True,
                        )
            finally:
                client.close()
            if partial.stat().st_size != size or digest(partial) != expected["sha256"]:
                raise ValueError(
                    "Final publisher weight hash mismatch; no checkpoint published"
                )
            partial.replace(target)
        if target.stat().st_size != size or digest(target) != expected["sha256"]:
            raise ValueError("Existing publisher weight hash mismatch")
        link = snapshot / expected["path"]
        if not link.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(os.path.relpath(target, snapshot))
        atomic_json(
            args.output / spec["name"] / "parallel_transfer.json",
            {
                "identity": identity,
                "verified": True,
                "workers": args.workers,
                "snapshot": str(snapshot),
            },
        )
    command = [
        sys.executable,
        "-m",
        "anomaly_eval.gemma4",
        "fetch",
        "--models",
        spec["name"],
        "--workers",
        "1",
        "--output",
        str(args.output),
        "--cache",
        str(args.cache),
        "--ram-cache",
        str(args.ram_cache),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
