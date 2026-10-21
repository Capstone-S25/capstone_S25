"""Prepare grouped causal video windows without changing previous studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/strategy_study_20261016"
SOURCE = ROOT / "data/strategy_sources_20261016"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    final = args.output / "windows.jsonl"
    if final.exists():
        raise ValueError("Completed pilot manifest is immutable; use a new output")
    old = [json.loads(s) for s in (OLD / "samples.jsonl").read_text().splitlines()]
    historical = [
        json.loads(s) for s in (OLD / "historical.jsonl").read_text().splitlines()
    ]
    rows, recordings, jobs = [], [], []
    # Existing camera assignments retained. Photos and CCTV are static auxiliary
    # examples, explicitly not manufactured video events.
    for row in old + historical:
        if row["task"] == "smoking":
            rows.append(
                {
                    **row,
                    "recording_id": row["id"],
                    "timestamps": [0.0],
                    "evaluation_kind": "retrospective_static",
                    "previously_inspected": True,
                }
            )
    hmdb = [
        json.loads(s) for s in (OLD / "hmdb_screened.jsonl").read_text().splitlines()
    ]
    # Use a single deterministic title split, independent of model predictions.
    groups = sorted({re.sub(r"\W", "", r["group_id"].lower()) for r in hmdb})
    random.Random(42).shuffle(groups)
    n = max(1, round(len(groups) * 0.2))
    fold = {
        g: "test" if i < n else "dev" if i < 2 * n else "train"
        for i, g in enumerate(groups)
    }
    for row in hmdb:
        action, stem = row["id"].split(":", 2)[1:]
        path = SOURCE / "hmdb51/videos" / action / (stem + ".avi")
        group = re.sub(r"\W", "", row["group_id"].lower())
        jobs.append(
            {
                "id": row["id"],
                "path": path,
                "mode": "clip",
                "row": {
                    **row,
                    "split": fold[group],
                    "group_id": "hmdb_title:" + group,
                    "recording_id": row["id"],
                    "evaluation_kind": "retrospective_action_clip",
                    "qualification": "Previously inspected HMDB reused for grouped adaptation; primary-action labels are non-exhaustive; not a fresh test",
                    "previously_inspected": True,
                },
            }
        )
    review = json.loads(
        (ROOT / "data/expanded_20261013/review_decisions.json").read_text()
    )
    group_by_tag = {}
    for row in old + historical:
        if row["task"] == "unattended_items":
            tag = Path(row["views"][0]).parent.name
            group_by_tag[tag] = (row["group_id"], row["split"])
    for tag, intervals in review["intervals"].items():
        if tag not in group_by_tag:
            continue
        group, split = group_by_tag[tag]
        if tag.startswith("caviar_"):
            name = tag.removeprefix("caviar_")
            path = ROOT / "data/additional/caviar" / name / (name + ".mpg")
        else:
            path = (
                ROOT / "data/additional/aboda" / (tag.removeprefix("aboda_") + ".avi")
            )
        jobs.append(
            {
                "id": tag,
                "path": path,
                "mode": "continuous",
                "intervals": intervals,
                "group_id": group,
                "split": split,
            }
        )
    meva = [
        json.loads(s) for s in (OLD / "meva_screened.jsonl").read_text().splitlines()
    ]
    by_video = defaultdict(list)
    for row in meva:
        by_video[row["id"].split(":")[1]].append(row)
    for name, windows in by_video.items():
        jobs.append(
            {
                "id": "meva:" + name,
                "path": SOURCE / "meva" / (name + ".avi"),
                "mode": "transfer",
                "windows": windows,
            }
        )

    def decode(job):
        video = job["path"]
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if job["mode"] == "clip":
            # Previous manifest records the true last sequentially decoded frame;
            # HMDB AVI headers overstate frame counts and seeking is unreliable.
            count = int(Path(job["row"]["views"][-1]).stem) + 1
        if fps <= 0 or count < 2:
            raise ValueError(f"Invalid video: {video}")
        if job["mode"] == "clip":
            wanted = np.unique(
                np.rint(np.linspace(0, count - 1, min(16, count))).astype(int)
            ).tolist()
        elif job["mode"] == "transfer":
            wanted = sorted(
                {
                    round(t)
                    for r in job["windows"]
                    for t in np.arange(
                        int(r["id"].rsplit(":", 1)[1]) - 600,
                        int(r["id"].rsplit(":", 1)[1]) + 1,
                        15,
                    )
                }
            )
        else:
            wanted = sorted(
                {min(count - 1, round(t * fps)) for t in np.arange(0, count / fps, 0.5)}
            )
        directory = (
            args.output / "frames" / hashlib.sha256(job["id"].encode()).hexdigest()[:20]
        )
        directory.mkdir(parents=True, exist_ok=True)
        frames, desired = [], set(wanted)
        index = 0
        while cap.grab():
            if index in desired:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Decode failed {video}:{index}")
                h, w = frame.shape[:2]
                if max(h, w) > 960:
                    frame = cv2.resize(
                        frame, (round(w * 960 / max(h, w)), round(h * 960 / max(h, w)))
                    )
                destination = directory / f"{index:07d}.jpg"
                if not destination.exists() and not cv2.imwrite(
                    str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
                ):
                    raise OSError(destination)
                frames.append(
                    {
                        "path": str(destination.relative_to(ROOT)),
                        "index": index,
                        "time": index / fps,
                    }
                )
            index += 1
        cap.release()
        actual = {f["index"] for f in frames}
        if any(i < index and i not in actual for i in wanted) or (
            job["mode"] == "clip" and len(frames) != len(wanted)
        ):
            raise ValueError(
                f"Sequential decode length mismatch: {video}: {len(frames)}/{len(wanted)}"
            )
        record = {
            "id": job["id"],
            "video": str(video.relative_to(ROOT)),
            "fps": fps,
            "decoded_frames": index,
            "duration": index / fps,
            "frames": frames,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "mode": job["mode"],
        }
        windows = []
        if job["mode"] == "clip":
            windows = [
                {
                    **job["row"],
                    "views": [f["path"] for f in frames],
                    "timestamps": [f["time"] for f in frames],
                }
            ]
        elif job["mode"] == "transfer":
            for row in job["windows"]:
                end = int(row["id"].rsplit(":", 1)[1]) / fps
                selected = [
                    f for f in frames if end - 20 - 1e-5 <= f["time"] <= end + 1e-5
                ]
                windows.append(
                    {
                        **row,
                        "recording_id": job["id"],
                        "views": [f["path"] for f in selected],
                        "timestamps": [f["time"] for f in selected],
                        "split": "transfer",
                        "evaluation_kind": "retrospective_abandonment_action_proxy",
                        "previously_inspected": True,
                    }
                )
        else:
            record.update(
                intervals=job["intervals"],
                group_id=job["group_id"],
                split=job["split"],
                label_quality="provisional assistant timeline review; not independent human ground truth",
            )
            for i, f in enumerate(frames):
                t = f["time"]
                labels = [
                    int(label == "unattended_items")
                    for a, b, label in job["intervals"]
                    if a <= t <= b
                ]
                if t < 20:
                    continue
                if len(set(labels)) > 1:
                    raise ValueError("Contradictory intervals")
                selected = [v for v in frames[: i + 1] if v["time"] >= t - 20 - 1e-5]
                windows.append(
                    {
                        "id": f"{job['id']}:{f['index']}",
                        "task": "unattended_items",
                        "target": labels[0] if labels else -1,
                        "source": "unattended_reviewed",
                        "group_id": job["group_id"],
                        "split": job["split"],
                        "recording_id": job["id"],
                        "views": [v["path"] for v in selected],
                        "timestamps": [v["time"] for v in selected],
                        "evaluation_kind": "retrospective_continuous_provisional",
                        "evidence": "existing_provisional_review_intervals",
                        "previously_inspected": True,
                    }
                )
        return record, windows

    with ThreadPoolExecutor(12) as pool:
        for i, (record, windows) in enumerate(pool.map(decode, jobs)):
            recordings.append(record)
            rows.extend(windows)
            if i % 100 == 0:
                print("decoded", i + 1, "/", len(jobs), flush=True)
    folds = defaultdict(set)
    for row in rows:
        folds[row["group_id"]].add(row["split"])
    if any(len(v) > 1 for v in folds.values()):
        raise ValueError("Group leakage")
    write_jsonl(args.output / "recordings.jsonl", recordings)
    write_jsonl(final, rows)
    metadata = {
        "windows": len(rows),
        "recordings": len(recordings),
        "unique_images": len({p for r in rows for p in r["views"]}),
        "counts": dict(
            Counter(
                f"{r['task']}/{r['source']}/{r['split']}/{r['target']}" for r in rows
            )
        ),
        "manifest_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
        "policy": "camera-held old folds; deterministic normalized HMDB title folds; no new final-test claim",
        "sampling": "16 uniform HMDB views; 2 Hz continuous videos; causal 20-second windows",
    }
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("VIDEO PILOT PREPARED", len(rows), len(recordings), flush=True)


if __name__ == "__main__":
    main()
