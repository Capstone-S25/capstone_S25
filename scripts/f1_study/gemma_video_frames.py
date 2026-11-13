"""Dense frames for the Gemma "continuous video or selected frames" comparison.

For held-out windows whose source video is on disk, decode 32 evenly spaced
frames across the same time span that the window's selected views cover.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from common import OUT, ROOT, load_rows

DENSE = 32
TARGET = OUT / "gemma_video"


def video_for(row):
    folder = Path(row["views"][0]).parent
    source = row["source"]
    if source == "hmdb51_action_transfer":
        return (
            ROOT
            / "data/strategy_sources_20261016/hmdb51/videos"
            / folder.parent.name
            / f"{folder.name}.avi"
        )
    if source == "mivia_iwdd":
        return (
            ROOT
            / "data/training_sources_20261015/mivia_iwdd/test/videos"
            / f"{folder.name}.mp4"
        )
    if source == "unattended_reviewed":
        return ROOT / "data/additional/aboda" / f"{folder.name.split('_')[1]}.avi"
    if source == "meva_abandon_action_transfer":
        return ROOT / "data/strategy_sources_20261016/meva" / f"{folder.name}.avi"
    return None


def index(path):
    return int(re.findall(r"\d+", Path(path).stem)[-1])


def choose(rows):
    random.seed(0)
    by = defaultdict(list)
    for row in rows:
        if row["split"] == "test" and video_for(row) is not None:
            if index(row["views"][-1]) > index(row["views"][0]):
                by[(row["source"], row["target"])].append(row)
    limits = {
        ("hmdb51_action_transfer", 0): 291,
        ("unattended_reviewed", 0): 150,
        ("unattended_reviewed", 1): 150,
    }
    chosen = []
    for key, group in sorted(by.items()):
        random.shuffle(group)
        chosen += group[: limits.get(key, len(group))]
    return chosen


def main():
    rows = choose(load_rows())
    by_video = defaultdict(list)
    for row in rows:
        by_video[video_for(row)].append(row)
    manifest = []
    for video, group in sorted(by_video.items()):
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        wanted = {}
        for row in group:
            first, last = index(row["views"][0]), index(row["views"][-1])
            row["_dense"] = np.rint(np.linspace(first, last, DENSE)).astype(int)
            for i in row["_dense"]:
                wanted[int(i)] = None
        folder = TARGET / "frames" / re.sub(r"[^A-Za-z0-9_.-]", "_", video.stem)
        folder.mkdir(parents=True, exist_ok=True)
        position, last_frame = 0, None
        for target in sorted(wanted):
            path = folder / f"{target:07d}.jpg"
            wanted[target] = path
            if path.exists():
                continue
            while position <= target:
                ok, frame = capture.read()
                if not ok:
                    break
                last_frame, position = frame, position + 1
            if last_frame is None:
                raise RuntimeError(f"Could not decode {video}")
            height, width = last_frame.shape[:2]
            scale = min(1.0, 1536 / max(height, width))
            image = (
                last_frame
                if scale == 1.0
                else cv2.resize(
                    last_frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            )
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        capture.release()
        for row in group:
            first = index(row["views"][0])
            manifest.append(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "target": row["target"],
                    "source": row["source"],
                    "group_id": row["group_id"],
                    "fps": fps,
                    "selected": row["views"],
                    "selected_times": [
                        round((index(v) - first) / fps, 2) for v in row["views"]
                    ],
                    "dense": [
                        str(wanted[int(i)].relative_to(ROOT)) for i in row["_dense"]
                    ],
                    "dense_times": [
                        round((int(i) - first) / fps, 2) for i in row["_dense"]
                    ],
                }
            )
        print(video.name, len(group), "rows", len(wanted), "frames", flush=True)
    (TARGET / "manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest)
    )
    print(len(manifest), "rows")


if __name__ == "__main__":
    main()
