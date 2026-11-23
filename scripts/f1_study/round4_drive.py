"""Round 4: score the released ensemble on the Drive's new labelled video sets.

Windows follow the runtime: a 2 fps buffer; smoking and sleeping read the frame
at t, littering eight frames over the last 3.5 s, loitering and unattended
items four frames over the last 20 s plus the video's first frame. Each window
is labelled from the set's own JSON (positive, negative, or ignored when the
label does not decide it) and scored with the runtime's EnsembleStrategy.

Run with the stage-1 environment (CUDA, ultralytics, cv2):
    PYTHONPATH=../../.local/worktrees/vision-app-integration \
    VISION_MODELS_DIR=../../dist/capstone-cpu-deployment-20261114/models \
    ../../.venv-stage1/bin/python round4_drive.py [--sets ...] [--step 2]
Writes results/f1_study_20261111/round4/windows.jsonl (appends; resumable).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/datasets"
OUT = ROOT / "results/f1_study_20261111/round4"
FRAMES = OUT / "frames"
BUFFER_FPS = 2.0

SETS = {
    "icvl_smoking": ("smoking", "smoking/ICVL (smoking subset)"),
    "icvl_littering": ("littering", "littering/ICVL (littering subset)"),
    "icvl_loitering": ("loitering", "loitering/ICVL (loitering subset)"),
    "iitb_loitering": ("loitering", "loitering/IITB Corridor (loitering subset)"),
    "aboda": ("unattended_items", "unattended items/ABODA"),
    "d2cdnet": ("unattended_items", "unattended items/D2-CDNet (Subset)"),
    "chad": ("rough_sleeping", "rough sleeping/CHAD (relevant clips)"),
}


def label_path(video_rel):
    parts = list(Path(video_rel).parts)
    parts[parts.index("videos")] = "labels"
    return RAW / Path(*parts).with_suffix(".json")


def extract(video, folder, fps):
    """Frames on the runtime's 2 fps grid, at the frame index nearest each tick."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "frames.json"
    if marker.exists():
        return json.loads(marker.read_text())
    capture = cv2.VideoCapture(str(video))
    step = fps / BUFFER_FPS
    out, index, tick = [], 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index == round(tick * step):
            path = folder / f"{tick:05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out.append([index, index / fps, str(path.relative_to(ROOT))])
            tick += 1
            while round(tick * step) <= index:  # fps below 2: skip ticks
                tick += 1
        index += 1
    capture.release()
    marker.write_text(json.dumps(out))
    return out


def within(t, spans, pad=0.0):
    return any(a - pad <= t <= b + pad for a, b in spans)


def label(task, meta, t, t0):
    """1 positive, 0 negative, None ignored, for a window ending at t that
    reads frames from t0 to t."""
    fps, events = meta["fps"], meta.get("events", [])
    if task == "smoking":
        marked = [
            (a / fps, b / fps)
            for e in events
            for a, b in _stretches(e["smoking_frames"])
        ]
        if within(t, marked, 0.25):
            return 1
        return (
            None if within(t, [(e["t_start_s"], e["t_end_s"]) for e in events]) else 0
        )
    if task == "littering":
        acts = [tuple(e["t_littering_s"]) for e in events]
        if any(a <= t and b >= t0 for a, b in acts):
            return 1
        present = [(e["t_start_s"], e["t_end_s"]) for e in events]
        return None if any(a <= t and b >= t0 for a, b in present) else 0
    if task == "loitering":
        if within(t, [(e["t_loitering_from_s"], e["t_end_s"]) for e in events]):
            return 1
        return (
            None
            if within(t, [(e["t_start_s"], e["t_end_s"] + 20) for e in events])
            else 0
        )
    if task == "unattended_items":
        end = meta["duration_s"]
        active = [
            (
                e["t_unattended_s"],
                e["t_removed_s"] if e["t_removed_s"] is not None else end,
            )
            for e in events
        ]
        if within(t, active):
            return 1
        busy = [
            (
                e["t_put_down_s"],
                (e["t_removed_s"] if e["t_removed_s"] is not None else end) + 20,
            )
            for e in events
        ]
        return None if within(t, busy) else 0
    if task == "rough_sleeping":
        lying = [(e["t_start_s"], e["t_end_s"]) for e in events]
        if within(t, lying):
            return 1
        return None if within(t, lying, 5.0) else 0
    raise ValueError(task)


def _stretches(value):
    """smoking_frames is a list of [first, last] pairs (or one pair)."""
    if value and isinstance(value[0], (int, float)):
        return [value]
    return value


def windows(task, frames, step):
    times = np.array([f[1] for f in frames])
    paths = [str(ROOT / f[2]) for f in frames]
    every = max(1, round(step * BUFFER_FPS))
    for end in range(0, len(frames), every):
        t = times[end]
        if task in {"smoking", "rough_sleeping"}:
            yield t, t, [paths[end]], None
        elif task == "littering":
            if end < 7:
                continue
            idx = list(range(end - 7, end + 1))
            yield t, times[idx[0]], [paths[i] for i in idx], None
        else:
            if t - times[0] < 19.5:
                continue
            start = int(np.searchsorted(times, t - 20.0 - 0.5 / BUFFER_FPS))
            span = times[start : end + 1]
            targets = np.linspace(max(span[0], span[-1] - 20), span[-1], 4)
            idx = [start + int(np.argmin(abs(span - v))) for v in targets]
            yield t, times[idx[0]], [paths[i] for i in idx], paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument(
        "--step", type=float, default=2.0, help="seconds between window ends"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="videos per set (for a trial)")
    a = ap.parse_args()

    from vision_runtime.ensemble import EnsembleStrategy

    es = EnsembleStrategy(device=a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "windows.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["set"], r["video"]))
    manifest = json.loads((RAW / "drive_20261115_videos.json").read_text())
    with out_path.open("a") as out:
        for name in a.sets:
            task, prefix = SETS[name]
            videos = sorted(
                (m for m in manifest if m["drive_path"].startswith(prefix)),
                key=lambda m: m["drive_path"],
            )
            for n, m in enumerate(videos[: a.limit]):
                video = Path(m["drive_path"]).stem
                if (name, video) in done:
                    continue
                meta = json.loads(label_path(m["drive_path"]).read_text())
                started = time.time()
                frames = extract(
                    RAW / m["local_path"], FRAMES / name / video, m["encoded"]["fps"]
                )
                rows = []
                for t, t0, views, reference in windows(task, frames, a.step):
                    y = label(task, meta, t, t0)
                    if y is None:
                        continue
                    score, channel = es.score_detail(task, views, reference)
                    rows.append(
                        {
                            "set": name,
                            "task": task,
                            "video": video,
                            "camera": meta.get("camera") or video,
                            "t": round(float(t), 3),
                            "target": y,
                            "score": score,
                            "channel": channel,
                        }
                    )
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                out.flush()
                pos = sum(r["target"] for r in rows)
                print(
                    f"{name} {n + 1}/{len(videos)} {video}: {len(rows)} windows ({pos} positive) "
                    f"in {time.time() - started:.0f} s",
                    flush=True,
                )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
