"""Dataset inventory, bounded review candidates and portable bundle export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from anomaly_eval.data import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    file_sha256,
    read_definitions,
)
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    resolve_arguments,
    resolve_path,
)


def suggested_label(path: Path, definitions: dict[str, str]) -> str:
    parts = {re.sub(r"[\s-]+", "_", part.lower()) for part in path.parts[:-1]}
    candidates = [label for label in definitions if label in parts]
    return candidates[0] if len(candidates) == 1 else ""


def sample_video(
    video: Path, output: Path, fps: float, max_frames: int, max_side: int, threads: int
) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "ffmpeg is required for --sample-videos; install it with your OS package manager"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("frame_*.jpg"))
    if existing:
        raise ValueError(
            f"Frames already exist in {output}; use a new --frames-dir for another sampling run"
        )
    scale = f"scale=w=min({max_side}\\,iw):h=min({max_side}\\,ih):force_original_aspect_ratio=decrease"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-threads",
        str(threads),
        "-i",
        str(video),
        "-an",
        "-sn",
        "-vf",
        f"fps={fps},{scale}",
        "-filter_threads",
        str(threads),
        "-threads",
        str(threads),
        "-frames:v",
        str(max_frames),
        str(output / "frame_%06d.jpg"),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode:
        raise ValueError(f"ffmpeg failed for {video}: {result.stderr.strip()}")
    return sorted(output.glob("frame_*.jpg"))


def inventory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Workspace.current().data_dir / "raw"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Workspace.current().data_dir / "manifest.csv"
    )
    parser.add_argument(
        "--definitions", type=Path, default=Workspace.current().definitions
    )
    parser.add_argument(
        "--sample-videos",
        action="store_true",
        help="Extract bounded frame samples with ffmpeg",
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=Workspace.current().data_dir / "frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.2,
        help="Sample rate; 0.2 means one frame every 5 seconds",
    )
    parser.add_argument("--max-frames-per-video", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    args = resolve_arguments(parser.parse_args(argv))
    if (
        not math.isfinite(args.fps)
        or args.fps <= 0
        or min(args.max_frames_per_video, args.max_side, args.threads) < 1
    ):
        parser.error(
            "fps must be positive and finite; frame, size, and thread limits must be at least 1"
        )
    if args.manifest.exists():
        parser.error(
            "Manifest already exists; choose another --manifest to preserve reviewed labels"
        )
    if not args.data_dir.is_dir():
        parser.error("data-dir does not exist; run anomaly-eval fetch-drive first")
    try:
        definitions = read_definitions(args.definitions)
        root = args.data_dir.resolve()
        frames_root = args.frames_dir.resolve()
        if args.sample_videos and frames_root.is_relative_to(root):
            raise ValueError(
                "frames-dir must be outside data-dir to prevent indexing frames twice"
            )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        rows = []
        videos = 0
        for path in files:
            relative = path.relative_to(root)
            suggestion = suggested_label(relative, definitions)
            group = relative.as_posix()
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, group, suggestion))
            elif path.suffix.lower() in VIDEO_EXTENSIONS:
                videos += 1
                if args.sample_videos:
                    digest = hashlib.sha256(group.encode()).hexdigest()[:12]
                    output = frames_root / f"{path.stem}_{digest}"
                    frames = sample_video(
                        path,
                        output,
                        args.fps,
                        args.max_frames_per_video,
                        args.max_side,
                        args.threads,
                    )
                    rows.extend((frame, group, suggestion) for frame in frames)
                    print(f"Sampled {len(frames)} frames: {relative}")
        if not rows:
            raise ValueError(
                "No images found. Extract archives or use --sample-videos for video data"
            )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "split", "group_id", "suggested_label"])
            for path, group, suggestion in rows:
                writer.writerow(
                    [
                        os.path.relpath(path, args.manifest.parent.resolve()),
                        "",
                        "test",
                        group,
                        suggestion,
                    ]
                )
        print(f"Wrote {len(rows)} images to {args.manifest}; found {videos} videos.")
        print(
            "Review label and split columns before evaluating accuracy. Folder suggestions are not ground truth."
        )
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1


def generate_candidates(root: Path, cap=24):
    rows = []
    for video in sorted((root / "aboda").glob("*.avi")):
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )
        duration = float(probe["format"]["duration"])
        for i in range(4):
            t = duration * (i + 1) / 5
            out = root / "review" / "aboda" / (video.stem + f"_{i}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(video),
                    "-ss",
                    str(t),
                    "-frames:v",
                    "1",
                    str(out),
                ],
                check=True,
            )
            rows.append(
                {
                    "path": portable_path(out),
                    "source": "aboda",
                    "group_id": "aboda:" + video.stem,
                    "timestamp_seconds": t,
                    "source_annotation": "video only",
                    "label": "",
                    "task": "",
                    "evidence": "pending_review",
                    "review_note": "",
                }
            )
    for archive in sorted((root / "d2_cdnet").glob("*.tar")):
        counts = {}
        source = []
        with tarfile.open(archive, mode="r|") as tar:
            for member in tar:
                # Only real RGB input and paired GT candidates; never synthetic diversity modalities.
                match = re.search(
                    r"(.+)/(input|groundtruth)/(?:in|gt)(\d+)\.(?:jpg|png)$",
                    member.name,
                )
                if not match or not member.isfile():
                    continue
                group, kind, index = match.groups()
                index = int(index)
                if index % 60:
                    continue
                key = (group, kind)
                if counts.get(key, 0) >= cap:
                    continue
                counts[key] = counts.get(key, 0) + 1
                # Flatten safe output names; source archive member remains provenance.
                relative = (
                    Path(group).parts[-1]
                    + "_"
                    + kind
                    + "_"
                    + str(index)
                    + Path(member.name).suffix
                )
                out = root / "review" / "d2_cdnet" / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, out.open("wb") as dst:
                    import shutil

                    shutil.copyfileobj(src, dst)
                if kind == "input":
                    source.append(
                        {
                            "path": portable_path(out),
                            "source": "d2_cdnet:" + archive.stem,
                            "group_id": "d2_cdnet:" + group,
                            "timestamp_seconds": "",
                            "source_annotation": member.name
                            + " (foreground masks are not event labels)",
                            "label": "",
                            "task": "",
                            "evidence": "pending_review",
                            "review_note": "",
                        }
                    )
        rows.extend(source)
    output = root / "review" / "candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} candidates require sequence/event review: {output}")
    return rows


def review_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Workspace.current().data_dir / "additional"
    )
    p.add_argument("--per-video", type=int, default=24)
    a = resolve_arguments(p.parse_args(argv))
    if a.per_video < 1:
        p.error("per-video must be positive")
    generate_candidates(a.root.resolve(), a.per_video)
    return 0


def find_samples(run: dict) -> list[dict]:
    for value in [run.get("samples")]:
        if (
            isinstance(value, list)
            and len(value) > 0
            and all(isinstance(e, dict) for e in value)
            and "path" in value[0]
            and "label" in value[0]
        ):
            return value
    raise SystemExit("no sample list with path/label entries found in run.json")


def export_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = resolve_arguments(parser.parse_args(argv))

    run = json.loads(args.run_json.read_text())
    samples = find_samples(run)
    rows: list[dict] = []
    for sample in samples:
        src = resolve_path(sample["path"], args.run_json.parent)
        if not src.is_file():
            raise SystemExit(f"missing source frame: {src}")
        dest_dir = args.out_dir / str(sample["label"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        digest_src = file_sha256(src)
        if dest.exists():
            # Never silently overwrite a different file.
            if file_sha256(dest) != digest_src:
                raise SystemExit(f"refusing to overwrite differing file: {dest}")
        else:
            shutil.copy2(src, dest)
        rows.append(
            {
                "file": f"{sample['label']}/{src.name}",
                "label": sample["label"],
                "group_id": sample.get("group_id", ""),
                "split": sample.get("split", ""),
                "source_path": portable_path(src),
                "sha256": digest_src,
            }
        )

    if not rows:
        raise SystemExit("No recorded samples to export")
    manifest = args.out_dir / "MANIFEST.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_label: dict[str, int] = {}
    for row in rows:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
    total = sum(
        p.stat().st_size
        for p in args.out_dir.rglob("*")
        if p.is_file() and p.name != "MANIFEST.csv"
    )
    print(f"copied {len(rows)} frames -> {args.out_dir}")
    for label in sorted(by_label):
        print(f"  {label}: {by_label[label]}")
    print(f"total image bytes: {total} ({total / 1e6:.1f} MB)")
