"""Acquire a pinned, scene/subject-separated public smoking-action pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/generalization_pilot_20261020/xrf55"
META = ROOT / "data/generalization_pilot_20261020/source_metadata"
BASE = "https://www.kaggle.com/api/v1/datasets/download/airslab2020/xrf55-rgb-depth-ir/"
ACTIONS = {"03": "using_phone", "13": "drinking", "14": "eating", "15": "smoking"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plan():
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "acquisition_protocol.json"
    if target.exists():
        raise ValueError("Acquisition plan is already frozen")
    readme = (META / "xrf55_public_release_readme.bin").read_text()
    required = ["CC BY-NC 4.0", "75 frames", "15 fps", "01-20", "scene4"]
    # The release uses a backtick 01-20 repetition example through the filename
    # convention, but published text calls its range `01`-`20`.
    assert all(v in readme for v in required if v != "01-20")
    view = json.loads((META / "xrf55_view.json").read_text())
    assert view["currentVersionNumber"] == 6 and not view["isPrivate"]
    public = {
        "scene1": [
            "01",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "13",
            "14",
            "19",
            "21",
            "22",
            "24",
            "26",
            "27",
            "28",
        ],
        "scene2": ["05", "24", "31"],
        "scene3": ["06", "07"],
        "scene4": ["03", "04", "13"],
    }
    dev_subjects = set(public["scene3"])
    test_subjects = set(public["scene4"])
    tasks = []
    for scene, subjects in public.items():
        for subject in subjects:
            if scene in {"scene1", "scene2"}:
                if subject in dev_subjects | test_subjects:
                    continue
                split = "train"
                reps = [1, 4, 7, 10, 13]
            elif scene == "scene3":
                split = "dev"
                reps = list(range(1, 21))
            else:
                split = "test"
                reps = list(range(1, 21))
            for action, name in ACTIONS.items():
                for repetition in reps:
                    filename = f"rgb_public__{scene}__{subject}/rgb_public/{scene}/{subject}/{subject}_{action}/{subject}_{action}_{repetition:02d}.mkv"
                    tasks.append(
                        {
                            "id": f"xrf55:{scene}:{subject}:{action}:{repetition:02d}",
                            "scene": scene,
                            "subject": subject,
                            "action": name,
                            "action_id": action,
                            "repetition": repetition,
                            "split": split,
                            "target": int(action == "15"),
                            "source": "xrf55_rgb_public",
                            "filename": filename,
                            "url": BASE
                            + quote(filename, safe="")
                            + "?datasetVersionNumber=6",
                            "video": str(
                                (
                                    DATA
                                    / "videos"
                                    / scene
                                    / subject
                                    / f"{subject}_{action}_{repetition:02d}.mkv"
                                ).relative_to(ROOT)
                            ),
                        }
                    )
    assert len(tasks) == 740
    save(
        target,
        {
            "status": "frozen_before_download_or_model_predictions",
            "date": "2026-10-20",
            "dataset": "airslab2020/xrf55-rgb-depth-ir",
            "version": 6,
            "license": "CC BY-NC 4.0 (publisher release README)",
            "publisher": "https://github.com/airslab2020/XRF55-repo",
            "readme_sha256": digest(META / "xrf55_public_release_readme.bin"),
            "catalog_sha256": digest(META / "xrf55_view.json"),
            "action_labels": ACTIONS,
            "split_rule": "Train scenes1/2; dev scene3; test scene4. Remove train numeric subject IDs appearing in dev/test conservatively, even if scene-local numbering might denote different people.",
            "official_split_difference": "Official repetition split is superseded by this stricter scene/subject pilot; train uses repetitions1,4,7,10,13; dev/test retain all20.",
            "counts": {
                s: sum(t["split"] == s for t in tasks) for s in ["train", "dev", "test"]
            },
            "qualification": "Scripted laboratory action clips, not CCTV event streams. Repetitions and scenes are correlated. Source primary-action labels do not supply exhaustive negatives for other objectives, continuous smoking onsets or independent human review.",
            "tasks": tasks,
            "no_source_code_execution": True,
        },
    )
    print(
        "XRF55 PLAN",
        len(tasks),
        {s: sum(t["split"] == s for t in tasks) for s in ["train", "dev", "test"]},
        flush=True,
    )


def fetch_one(task):
    destination = ROOT / task["video"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + quote(task["filename"], safe="") + "?datasetVersionNumber=6"
    if not destination.exists():
        partial = destination.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        # EBML header, not a login page, archive or generated preview.
        with partial.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x1aE\xdf\xa3":
            raise ValueError("Expected MKV file: " + task["id"])
        partial.replace(destination)
    return {
        **task,
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
    }


def fetch():
    protocol = json.loads((DATA / "acquisition_protocol.json").read_text())
    tasks = protocol["tasks"]
    if shutil.disk_usage(DATA).free < 3 * 1024**3:
        raise OSError("Need 3 GiB free for selected source clips and preparation")
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch_one, t): t for t in tasks}
        try:
            for future in as_completed(pending):
                completed.append(future.result())
                if len(completed) % 40 == 0:
                    print("XRF55 FETCH", len(completed), "/", len(tasks), flush=True)
        except Exception:
            for future in pending:
                future.cancel()
            raise
    completed.sort(key=lambda r: r["id"])
    save(
        DATA / "acquisition_receipt.json",
        {
            "status": "complete",
            "version": 6,
            "files": completed,
            "total_bytes": sum(r["bytes"] for r in completed),
            "protocol_sha256": digest(DATA / "acquisition_protocol.json"),
        },
    )
    print(
        "XRF55 ACQUIRED",
        len(completed),
        "videos",
        sum(r["bytes"] for r in completed),
        "bytes",
        flush=True,
    )


def prepare():
    import cv2
    import numpy as np
    from PIL import Image

    from anomaly_eval.data import image_pixel_sha256

    receipt = json.loads((DATA / "acquisition_receipt.json").read_text())
    if (DATA / "samples.jsonl").exists():
        raise ValueError("Prepared source manifest is already frozen")
    rows, identities = [], []
    for i, task in enumerate(receipt["files"]):
        path = ROOT / task["video"]
        assert digest(path) == task["sha256"]
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(fps - 15) > 0.1 or count != 75:
            raise ValueError(
                f"Unexpected publisher clip metadata: {task['id']} {fps} {count}"
            )
        wanted = set(np.linspace(0, count - 1, 16).round().astype(int).tolist())
        frames = []
        index = 0
        while cap.grab():
            if index in wanted:
                okay, bgr = cap.retrieve()
                if not okay:
                    raise ValueError("Failed sequential decode")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                dest = (
                    DATA
                    / "frames"
                    / task["scene"]
                    / task["subject"]
                    / path.stem
                    / f"{index:03d}.jpg"
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                image.save(dest, quality=92)
                pixel = image_pixel_sha256(dest)
                frames.append(
                    {
                        "path": str(dest.relative_to(ROOT)),
                        "index": index,
                        "time": index / fps,
                    }
                )
                identities.append(
                    {
                        "path": str(dest.relative_to(ROOT)),
                        "split": task["split"],
                        "rgb_sha256": pixel,
                        "sha256": digest(dest),
                    }
                )
            index += 1
        cap.release()
        assert index == 75 and len(frames) == 16
        rows.append(
            {
                **{k: v for k, v in task.items() if k not in {"url", "filename"}},
                "task": "smoking",
                "views": [f["path"] for f in frames],
                "timestamps": [f["time"] for f in frames],
                "frames": frames,
                "group_id": "xrf55:subject:" + task["subject"],
                "recording_group": f"xrf55:{task['scene']}:{task['subject']}:{task['action_id']}",
                "duration": count / fps,
                "fps": fps,
                "decoded_frames": index,
                "label_evidence": "publisher primary action and filename; no independent review",
                "evidence": "publisher primary action and filename; no independent review",
                "pixel_hashes": [r["rgb_sha256"] for r in identities[-16:]],
                "evaluation_kind": "new_source_scene_and_subject_separated_offline_action",
            }
        )
        if (i + 1) % 100 == 0:
            print("XRF55 PREPARE", i + 1, "/", len(receipt["files"]), flush=True)
    seen = {}
    for r in identities:
        if r["rgb_sha256"] in seen and seen[r["rgb_sha256"]] != r["split"]:
            raise ValueError("Cross-split exact pixels")
        seen[r["rgb_sha256"]] = r["split"]
    split_subjects = {
        s: {r["subject"] for r in rows if r["split"] == s}
        for s in ["train", "dev", "test"]
    }
    assert not (
        split_subjects["train"] & split_subjects["dev"]
        or split_subjects["train"] & split_subjects["test"]
        or split_subjects["dev"] & split_subjects["test"]
    )
    (DATA / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (DATA / "input_identities.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in identities)
    )
    save(
        DATA / "preparation.json",
        {
            "status": "complete",
            "examples": len(rows),
            "views": len(identities),
            "scene_and_numeric_subject_disjoint": True,
            "cross_split_exact_pixels": 0,
            "split_subjects": {s: sorted(v) for s, v in split_subjects.items()},
            "manifest_sha256": digest(DATA / "samples.jsonl"),
            "receipt_sha256": digest(DATA / "acquisition_receipt.json"),
            "no_model_predictions_used": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "fetch", "prepare"])
    args = parser.parse_args()
    {"plan": plan, "fetch": fetch, "prepare": prepare}[args.stage]()


if __name__ == "__main__":
    main()
