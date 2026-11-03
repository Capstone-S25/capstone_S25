"""Freeze new-source video transfer cohorts and the historical diagnostic cohort."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from anomaly_eval.data import image_pixel_sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/strategy_sources_20261016"
DATA = ROOT / "data/strategy_study_20261016"


def frame_images(video, indices, destination):
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    paths = []
    try:
        for index in indices:
            path = destination / f"{int(index):07d}.jpg"
            if not path.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                okay, frame = cap.read()
                if not okay:
                    raise ValueError(f"Cannot decode {video.name} frame {index}")
                h, w = frame.shape[:2]
                if max(h, w) > 1280:
                    frame = cv2.resize(
                        frame,
                        (round(w * 1280 / max(h, w)), round(h * 1280 / max(h, w))),
                    )
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(path.relative_to(ROOT)))
    finally:
        cap.release()
    return paths


def meva():
    rows, rejected = [], []
    index = json.loads((SOURCE / "meva/index.json").read_text())
    for row in index:
        if row["status"] != "downloaded":
            rejected.append({"name": row["name"], "reason": row["status"]})
            continue
        cap = cv2.VideoCapture(str(SOURCE / "meva" / (row["name"] + ".avi")))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if abs(fps - 30) > 0.01:
            raise ValueError(
                "Annotation frame alignment requires the original 30 FPS timeline"
            )
        intervals = []
        for start, end in sorted(row["intervals_original_frames"]):
            if intervals and start <= intervals[-1][1] + 30:
                intervals[-1][1] = max(end, intervals[-1][1])
            else:
                intervals.append([start, end])
        endpoints = [(1, end) for _, end in intervals if end >= 600 and end < count]
        before = intervals[0][0] - 300
        if before >= 600:
            endpoints.append((0, before))
        for target, end in endpoints:
            frames = np.rint(np.linspace(end - 600, end, 4)).astype(int)
            paths = frame_images(
                SOURCE / "meva" / (row["name"] + ".avi"),
                frames,
                DATA / "meva_frames" / row["name"],
            )
            rows.append(
                {
                    "id": f"meva:{row['name']}:{end}",
                    "task": "unattended_items",
                    "target": target,
                    "source": "meva_abandon_action_transfer",
                    "group_id": "meva_time:" + row["name"].rsplit(".", 2)[0],
                    "camera": row["name"].split(".")[-1],
                    "views": paths,
                    "split": "test",
                    "pixel_hashes": [image_pixel_sha256(ROOT / p) for p in paths],
                    "evidence": "published_abandon_package_activity_intervals",
                    "qualification": "New-source offline action-window transfer, not verified persistent unattended-item status; positive window ends at annotated abandonment, control precedes first annotation by 10 seconds; contributed CMU and Kitware-training labels have different audit levels",
                }
            )
    return rows, rejected


def hmdb():
    videos = sorted((SOURCE / "hmdb51/videos").rglob("*.avi"))

    def one(path):
        action = path.parent.name
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count < 8:
            return None, {"name": str(path), "reason": "fewer_than_8_frames"}
        try:
            # HMDB AVI indexed seeking near EOF is unreliable. Sample the actual
            # sequentially decoded timeline rather than dropping good videos.
            cap = cv2.VideoCapture(str(path))
            decoded = []
            while True:
                okay, frame = cap.read()
                if not okay:
                    break
                decoded.append(frame)
            cap.release()
            if len(decoded) < 8:
                raise ValueError("Fewer than eight sequentially decoded frames")
            indices = np.rint(np.linspace(0, len(decoded) - 1, 8)).astype(int)
            directory = DATA / "hmdb_frames" / action / path.stem
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for i in indices:
                target = directory / f"{i:07d}.jpg"
                cv2.imwrite(str(target), decoded[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(str(target.relative_to(ROOT)))
            title = re.split(r"_(?:smoke|drink|eat|chew|talk|sit|stand)_", path.stem)[0]
            return {
                "id": "hmdb:" + action + ":" + path.stem,
                "task": "smoking",
                "target": int(action == "smoke"),
                "source": "hmdb51_action_transfer",
                "group_id": "hmdb_title:" + title,
                "views": paths,
                "split": "test",
                "pixel_hashes": [image_pixel_sha256(ROOT / p) for p in paths],
                "evidence": "author_primary_action_class",
                "qualification": "New-source offline action classification on eight uniform clip views; movie/web domain; other primary actions are non-exhaustive smoking controls; no HMDB images used for training or model selection",
            }, None
        except (ValueError, OSError, cv2.error) as exc:
            return None, {"name": str(path), "reason": str(exc)}

    with ThreadPoolExecutor(12) as pool:
        values = list(pool.map(one, videos))
    return [r for r, _ in values if r], [e for _, e in values if e]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["meva", "hmdb", "historical"])
    parser.add_argument("--output-tag")
    args = parser.parse_args()
    tag = args.output_tag or args.source
    path = DATA / (tag + ".jsonl")
    if path.exists():
        raise ValueError("Transfer cohort is frozen; do not overwrite")
    if args.source == "historical":
        old = ROOT / "data/clip_training_20261015/samples.jsonl"
        rows = [
            r
            for r in map(json.loads, old.read_text().splitlines())
            if r["split"] in {"test", "external"}
        ]
        rejected = []
    else:
        rows, rejected = meva() if args.source == "meva" else hmdb()
        # Decoded identities must not overlap any strategy training/development image.
        training = [
            json.loads(line)
            for line in (DATA / "samples.jsonl").read_text().splitlines()
        ]
        pixels = {h for r in training for h in r["pixel_hashes"]}
        retained = []
        for row in rows:
            if pixels.intersection(row["pixel_hashes"]):
                rejected.append(
                    {"name": row["id"], "reason": "pixels_overlap_strategy_manifest"}
                )
            else:
                retained.append(row)
        rows = retained
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (DATA / (tag + "_audit.json")).write_text(
        json.dumps(
            {
                "rows": len(rows),
                "groups": len({r["group_id"] for r in rows}),
                "positive": sum(r["target"] for r in rows),
                "rejected": rejected,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "TRANSFER COHORT PREPARED",
        args.source,
        len(rows),
        "rejected",
        len(rejected),
        flush=True,
    )


if __name__ == "__main__":
    main()
