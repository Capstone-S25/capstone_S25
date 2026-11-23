"""New-site evidence for round 3: footage the released heads never saw.

Writes `results/f1_study_20261111/round3/rows.jsonl`. Every row has a `site`
split: "fit" rows may be used for fitting in round-3 variants, "shift" rows
are only ever scored. Whole cameras (or subjects, or clips) stay on one side.

| Source | Task | Content |
|---|---|---|
| smoking_site2 | smoking | Alert snapshots of a second CCTV estate, every image has a smoker |
| xrf55 | smoking | Indoor clips of phoning, drinking, eating: hand-to-face negatives |
| littering_campus | littering | 30-frame clips of staged littering on a campus |
| virat, avenue | all five | Windows of ordinary car-park / walkway footage, as negatives |
| meva | smoking, sleeping, littering | The same, from the MEVA cameras |

Windows of ordinary footage follow the runtime: the frame at t for smoking and
sleeping, eight frames over the last 7 s for littering, four frames over the
last 20 s plus the camera's first frame for loitering and unattended items.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re

import cv2
import numpy as np
from common import OUT, ROOT, TASKS
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

RAW = ROOT / "data/raw/datasets"
WORK = OUT / "round3"
FRAMES = WORK / "frames"


def key(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:24]


def relative(path):
    return str(path.relative_to(ROOT))


def thumb(path, size=(24, 14)):
    with Image.open(path) as image:
        a = np.asarray(image.convert("L").resize(size), np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a / np.linalg.norm(a)


def side(name, share=0.5):
    """Deterministic fit/shift side for a camera or subject."""
    return "fit" if int(key("side", name), 16) % 1000 < share * 1000 else "shift"


def smoking_site2():
    paths = sorted((RAW / "smoking/smoking_cctv_image.v1i.coco").glob("*/*.jpg"))
    x = np.stack([thumb(p) for p in paths])
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.5, linkage="average", metric="cosine"
    ).fit_predict(x)
    rows = []
    for path, label in zip(paths, labels):
        group = f"smoking_site2:{label}"
        rows.append(
            {
                "id": key("site2", path.name),
                "task": "smoking",
                "target": 1,
                "source": "smoking_site2",
                "group_id": group,
                "site": side(group),
                "views": [relative(path)],
            }
        )
    return rows


def xrf55():
    rows = []
    path = ROOT / "data/generalization_pilot_20261020/xrf55/samples.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["target"] == 1:
            # The "smoking" clips are mimed: a hand goes to the face, but no
            # cigarette is in view, so they are not smoking evidence.
            continue
        views = row["views"]
        idx = np.rint(np.linspace(0, len(views) - 1, 8)).astype(int)
        rows.append(
            {
                "id": key("xrf55", row["id"]),
                "task": "smoking",
                "target": row["target"],
                "source": "xrf55",
                "group_id": row["group_id"],
                "site": "shift" if row["split"] == "test" else "fit",
                "views": [views[i] for i in idx],
                "action": row["action"],
            }
        )
    return rows


def littering_campus():
    """Rebuild the 30-frame clips: files are named video_<n>_<frame>, many
    clips share a name, and train/test hold random frames of each clip."""
    root = RAW / "littering/Littering"
    by_video = collections.defaultdict(list)
    for split in ["train", "test"]:
        for path in sorted((root / split / "images").iterdir()):
            m = re.match(r"video_(\d+)_(\d+)_jpg", path.name)
            by_video[int(m.group(1))].append((int(m.group(2)), path))
    rows = []
    for video, items in sorted(by_video.items()):
        x = np.stack([thumb(p) for _, p in items])
        views = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.25, linkage="average", metric="cosine"
        ).fit_predict(x)
        for view in sorted(set(views)):
            members = [i for i in range(len(items)) if views[i] == view]
            by_frame = collections.defaultdict(list)
            for i in members:
                by_frame[items[i][0]].append(i)
            count = max(len(v) for v in by_frame.values())
            # Chain frames into `count` clips by appearance, frame to frame.
            chains = [[] for _ in range(count)]
            last = [None] * count
            for frame in sorted(by_frame):
                here = by_frame[frame]
                if any(v is None for v in last):
                    order = list(range(len(here)))
                    slots = [c for c in range(count) if last[c] is None][: len(here)]
                    pairs = list(zip(slots, order))
                    free = [c for c in range(count) if last[c] is not None]
                    rest = order[len(slots) :]
                    if rest and free:
                        cost = np.array(
                            [[1 - x[last[c]] @ x[here[j]] for j in rest] for c in free]
                        )
                        r, c = linear_sum_assignment(cost)
                        pairs += [(free[a], rest[b]) for a, b in zip(r, c)]
                else:
                    cost = np.array(
                        [[1 - x[last[c]] @ x[j] for j in here] for c in range(count)]
                    )
                    r, c = linear_sum_assignment(cost)
                    pairs = list(zip(r, c))
                for chain, j in pairs:
                    chains[chain].append(here[j])
                    last[chain] = here[j]
            group = f"littering_campus:video{video}:view{view}"
            for n, chain in enumerate(chains):
                if len(chain) < 16:
                    continue
                idx = np.rint(np.linspace(0, len(chain) - 1, 8)).astype(int)
                rows.append(
                    {
                        "id": key("campus", video, view, n),
                        "task": "littering",
                        "target": 1,
                        "source": "littering_campus",
                        "group_id": group,
                        "site": side(f"littering_campus:video{video}"),
                        "views": [relative(items[chain[i]][1]) for i in idx],
                        "frames": len(chain),
                    }
                )
    return rows


def extract(video, folder, limit=1280):
    """One frame per second as JPEG; returns the seconds written."""
    folder.mkdir(parents=True, exist_ok=True)
    done = sorted(folder.glob("*.jpg"))
    if (folder / "done").exists():
        return [int(p.stem) for p in done]
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    seconds, index, wanted = [], 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= round(wanted * fps):
            h, w = frame.shape[:2]
            scale = min(1.0, limit / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            cv2.imwrite(
                str(folder / f"{wanted:05d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            seconds.append(wanted)
            wanted += 1
        index += 1
    capture.release()
    (folder / "done").write_text("")
    return seconds


def normal_windows(source, videos, group_of, site_of, uncertain=(), tasks=TASKS):
    rows = []
    for video in videos:
        folder = FRAMES / source / video.stem
        seconds = extract(video, folder)
        frame = lambda s, folder=folder: relative(folder / f"{s:05d}.jpg")
        for t in range(20, (seconds[-1] if seconds else 0) + 1, 10):
            views = {
                "smoking": [frame(t)],
                "rough_sleeping": [frame(t)],
                "littering": [frame(s) for s in range(t - 7, t + 1)],
                "loitering": [frame(s) for s in (t - 20, t - 13, t - 7, t)],
                "unattended_items": [frame(s) for s in (t - 20, t - 13, t - 7, t)],
            }
            for task in tasks:
                row = {
                    "id": key(source, video.stem, t, task),
                    "task": task,
                    "target": 0,
                    "source": source,
                    "group_id": group_of(video),
                    "site": site_of(video),
                    "views": views[task],
                    "window_end_s": t,
                    "video": video.stem,
                }
                if task in {"loitering", "unattended_items"}:
                    row["reference"] = frame(0)
                if task in uncertain:
                    row["label_uncertain"] = True
                rows.append(row)
    return rows


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rows = smoking_site2() + xrf55() + littering_campus()
    virat = sorted((RAW / "loitering/VIRAT Video Dataset").glob("*.mp4"))
    scene = lambda v: v.stem.split("_")[2][:4]
    rows += normal_windows(
        "virat",
        virat,
        lambda v: f"virat:{scene(v)}",
        lambda v: "fit" if scene(v) == "0100" else "shift",
        # VIRAT was collected by the team as loitering material and has no
        # labels here: people do stand around in it, and unload cars. The
        # windows the released unattended-item head flagged were looked at:
        # groups standing, traffic, a patio; no left item in any of them.
        uncertain=("loitering", "littering"),
    )
    avenue = sorted((RAW / "loitering/Avenue Dataset/training_videos").glob("*.avi"))
    rows += normal_windows(
        "avenue", avenue, lambda v: "avenue:camera", lambda v: "shift"
    )
    # MEVA has labelled abandonment and people waiting, and the benchmark
    # already scores its unattended-item windows, so only three heads see it.
    meva = sorted((ROOT / "data/strategy_sources_20261016/meva").glob("*.avi"))
    rows += normal_windows(
        "meva",
        meva,
        lambda v: "meva:" + v.stem.split(".")[-1],
        lambda v: side("meva:" + v.stem.split(".")[-1]),
        uncertain=("littering",),
        tasks=("smoking", "rough_sleeping", "littering"),
    )
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids")
    with open(WORK / "rows.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    count = collections.Counter(
        (r["source"], r["task"], r["site"], r["target"]) for r in rows
    )
    for k in sorted(count):
        print(k, count[k])
    groups = collections.defaultdict(set)
    for r in rows:
        groups[r["source"], r["site"]].add(r["group_id"])
    print({k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
