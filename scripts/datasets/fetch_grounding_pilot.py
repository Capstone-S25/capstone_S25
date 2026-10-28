"""Acquire the exact public detector revision used by the video pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHTS_SHA = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/models/video_grounding")
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            "IDEA-Research/grounding-dino-tiny",
            revision=PIN,
            local_dir=out,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
            max_workers=6,
        )
    actual = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    if actual != WEIGHTS_SHA:
        raise ValueError("Detector weights differ from pinned pilot revision")
    metadata = out / "repository_metadata.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["sha"] != PIN:
            raise ValueError("Repository revision mismatch")
    else:
        metadata.write_text(
            json.dumps(
                {"id": "IDEA-Research/grounding-dino-tiny", "sha": PIN}, indent=2
            )
            + "\n"
        )
    print("PINNED GROUNDING WEIGHTS VERIFIED", actual, flush=True)


if __name__ == "__main__":
    main()
