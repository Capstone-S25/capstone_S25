"""Review and build expanded accuracy cohorts from local source recordings.

OpenCV is used only for source video decoding, never model inference. Recordings
are decoded at native resolution and sampled using actual frame indices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from anomaly_eval.data import (
    file_sha256,
    original_name,
    read_manifest,
    resolve_original,
)
from anomaly_eval.datasets.labels import collect_littering
from anomaly_eval.datasets.prepare import (
    assign_splits,
    deduplicate_examples,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/expanded_20261013"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def board(items, path, columns=4, width=352, height=264):
    font = ImageFont.truetype(FONT, 15)
    image = Image.new(
        "RGB",
        (width * columns, (height + 28) * ((len(items) + columns - 1) // columns)),
        "white",
    )
    draw = ImageDraw.Draw(image)
    for i, (source, label) in enumerate(items):
        with Image.open(source) as im:
            im = im.convert("RGB")
            im.thumbnail((width, height))
            x, y = i % columns * width, i // columns * (height + 28)
            image.paste(im, (x, y))
        draw.text((x + 5, y + height + 3), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def decode_video(pair):
    import cv2

    tag, video = pair
    cv2.setNumThreads(1)
    destination = DATA / "video_frames" / tag
    metadata = destination / "frames.json"
    digest = file_sha256(video)
    if metadata.exists():
        existing = json.loads(metadata.read_text())
        if existing["source_sha256"] != digest:
            raise ValueError("Changed source recording")
        return tag, existing
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"No frame rate: {video}")
    frame_index = 0
    count = 0
    rows = []
    while capture.grab():
        if frame_index == round(count * fps / 2):
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"Failed frame decode: {video}/{frame_index}")
            path = destination / f"frame_{frame_index:06d}.png"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps,
                    "path": str(path.relative_to(ROOT)),
                }
            )
            count += 1
        frame_index += 1
    capture.release()
    result = {
        "source": str(video.relative_to(ROOT)),
        "source_sha256": digest,
        "fps": fps,
        "decoded_frames": frame_index,
        "samples": rows,
    }
    save_json(metadata, result)
    indices = sorted({round((len(rows) - 1) * i / 23) for i in range(24)})
    board(
        [
            (ROOT / rows[i]["path"], f"{tag}  t={rows[i]['timestamp']:.2f}s")
            for i in indices
        ],
        DATA / "review" / f"{tag}.jpg",
    )
    return tag, result


def review():
    pairs = [
        ("aboda_" + p.stem, p)
        for p in sorted((ROOT / "data/additional/aboda").glob("*.avi"))
    ]
    pairs += [
        ("caviar_" + p.parent.name, p)
        for p in sorted((ROOT / "data/additional/caviar").glob("*/*.mpg"))
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        videos = dict(executor.map(decode_video, pairs))
    save_json(DATA / "videos.json", videos)
    groups = defaultdict(list)
    for item in collect_littering(ROOT / "data/raw/datasets"):
        source = original_name(item.path)
        stem = source.split(".rf.")[0]
        groups[item.group_id].append((stem, item.path))
    canonical = {}
    for group, examples in sorted(groups.items()):
        by_moment = {}
        for stem, path in sorted(examples, key=lambda item: str(item[1])):
            by_moment.setdefault(stem, path)
        pool = sorted(
            by_moment.items(),
            key=lambda item: [
                int(x) if x.isdigit() else x for x in re.split(r"(\d+)", item[0])
            ],
        )
        canonical[group] = [
            {"moment": stem, "path": str(path.relative_to(ROOT))} for stem, path in pool
        ]
        indices = sorted({round((len(pool) - 1) * i / 15) for i in range(16)})
        items = []
        for i in indices:
            stem, path = pool[i]
            label_path = resolve_original(
                path.parent.parent
                / "labels"
                / (Path(original_name(path)).stem + ".txt")
            )
            classes = (
                sorted(
                    {
                        line.split()[0]
                        for line in label_path.read_text().splitlines()
                        if line.strip()
                    }
                )
                if label_path
                else []
            )
            items.append((path, f"{stem}  class={','.join(classes)}"))
        board(items, DATA / "review" / (group.replace(":", "_") + ".jpg"))
    save_json(DATA / "littering_canonical.json", canonical)
    print(
        "Review videos",
        len(videos),
        "littering groups",
        len(canonical),
        "canonical moments",
        sum(map(len, canonical.values())),
        flush=True,
    )


def make_row(
    path, label, task, group, source, evidence, note, origin=None, protocol="frame"
):
    return {
        "path": str(path.resolve()),
        "label": label,
        "task": task,
        "group_id": group,
        "source": source,
        "evidence": evidence,
        "review_note": note,
        "origin": origin,
        "protocol": protocol,
        "sha256": file_sha256(path),
    }


def write_cohort(name, rows, definitions, qualification, split=True):
    directory = DATA / "cohorts"
    unique, _, dedup = deduplicate_examples(rows)
    if split:
        assign_splits(unique)
    else:
        for row in unique:
            row["split"] = (
                "test"
                if row["group_id"] in {"aboda:classroom", "aboda:walkway"}
                else "dev"
            )
    path = directory / (name + ".csv")
    write_manifest(path, unique)
    save_json(directory / (name + "_definitions.json"), definitions)
    # Exercise the production leakage/hash validation before scheduling inference.
    read_manifest(path, definitions)
    info = {
        "name": name,
        "manifest": str(path.relative_to(ROOT)),
        "definitions": str(
            (directory / (name + "_definitions.json")).relative_to(ROOT)
        ),
        "qualification": qualification,
        "samples": len(unique),
        "counts": {
            s: dict(Counter(r["label"] for r in unique if r["split"] == s))
            for s in ["dev", "test"]
        },
        "groups": {
            s: sorted({r["group_id"] for r in unique if r["split"] == s})
            for s in ["dev", "test"]
        },
        "deduplication": dedup,
    }
    save_json(directory / (name + "_dataset.json"), info)
    print(name, info["samples"], info["counts"], flush=True)
    return info


def build():
    videos = json.loads((DATA / "videos.json").read_text())
    definitions = json.loads((ROOT / "anomaly_eval/config/anomalies.json").read_text())
    # Reviewed before model inference. Camera families are deliberately held together.
    intervals = {
        "caviar_LeftBag": [(8, 36, "normal"), (45, 52, "unattended_items")],
        "caviar_LeftBag_AtChair": [(5, 18, "normal"), (28, 31, "unattended_items")],
        "caviar_LeftBox": [(5, 21, "normal"), (26, 34, "unattended_items")],
        "caviar_LeftBag_PickedUp": [(5, 18, "normal"), (44, 53, "normal")],
        "caviar_LeftBag_BehindChair": [(5, 12, "normal")],
        "caviar_Walk1": [(5, 24, "normal")],
        "caviar_Walk2": [(5, 30, "normal")],
        "caviar_Walk3": [(5, 40, "normal")],
        "aboda_video1": [
            (0, 32, "normal"),
            (37, 43, "normal"),
            (54, 72, "unattended_items"),
        ],
        "aboda_video2": [(0, 28, "normal")],
        "aboda_video3": [(0, 45, "normal"), (72, 80, "unattended_items")],
        "aboda_video4": [(0, 9, "normal")],
        "aboda_video5": [(0, 30, "normal"), (70, 90, "unattended_items")],
        "aboda_video6": [(0, 38, "normal"), (125, 195, "unattended_items")],
        "aboda_video7": [(0, 105, "normal"), (134, 159, "unattended_items")],
        "aboda_video8": [(0, 105, "normal"), (130, 159, "unattended_items")],
        "aboda_video9": [(0, 29, "normal"), (44, 81, "unattended_items")],
        "aboda_video10": [(0, 38, "normal"), (52, 74, "unattended_items")],
    }
    groups = {
        "aboda_video1": "aboda:lobby",
        **{f"aboda_video{i}": "aboda:walkway" for i in [2, 3, 4]},
        **{f"aboda_video{i}": "aboda:entrance" for i in [5, 6]},
        **{f"aboda_video{i}": "aboda:classroom" for i in [7, 8, 9, 10]},
    }
    rows = {"frame": [], "window": []}
    review_info = {
        "label_review": "assistant visual timeline review; CAVIAR positive intervals also checked against XML leaving-object roles",
        "independent_human_verification": False,
        "window_seconds": 20,
        "sampling_fps": 2,
        "intervals": intervals,
        "excluded_positive_candidates": {
            "caviar_LeftBag_BehindChair": "Item hidden by furniture and no object role in XML",
            "caviar_LeftBag_PickedUp": "Carrier remains in scene; proximity interpretation ambiguous",
            "aboda_video2": "Placement/left item not clearly visible in reviewed timeline",
            "aboda_video4": "Left item occluded behind street furniture",
            "aboda_video11": "Item already present; ownership/deposition cannot be established",
        },
    }
    import xml.etree.ElementTree as ET

    for tag, ranges in intervals.items():
        video = videos[tag]
        group = groups.get(tag, "caviar:inria_camera")
        frames = video["samples"]
        xml_frames = None
        if tag.startswith("caviar_"):
            sequence = tag.removeprefix("caviar_")
            tree = ET.parse(
                next((ROOT / "data/additional/caviar" / sequence).glob("*.xml"))
            ).getroot()
            xml_frames = {int(f.attrib["number"]): f for f in tree.findall("frame")}
        for sample in frames:
            t = sample["timestamp"]
            labels = {label for start, end, label in ranges if start <= t <= end}
            if not labels:
                continue
            if len(labels) != 1:
                raise ValueError("Conflicting interval labels")
            label = labels.pop()
            if label == "unattended_items" and xml_frames is not None:
                f = xml_frames.get(sample["frame_index"])
                if f is None or not any(
                    r.text == "leaving object" for r in f.iter("role")
                ):
                    raise ValueError(f"No object-role support: {tag}/{t}")
            origin = f"expanded-video://{video['source']}#{video['source_sha256']}@{t}"
            note = f"{tag} at {t:.3f}s; source-native frame {sample['frame_index']}; labels frozen before inference; assistant timeline review, not independently human verified"
            rows["frame"].append(
                make_row(
                    ROOT / sample["path"],
                    label,
                    "unattended_items",
                    group,
                    tag,
                    "assistant_visual_review",
                    note,
                    origin,
                )
            )
            # Fixed 20-second past-only context; actual sampled times in metadata.
            selected = [
                min(
                    frames,
                    key=lambda r: abs(r["timestamp"] - max(0, t - 20 + 20 * j / 3)),
                )
                for j in range(4)
            ]
            path = DATA / "windows" / tag / f"frame_{sample['frame_index']:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = False
            if path.exists():
                try:
                    with Image.open(path) as im:
                        im.verify()
                    cached = True
                except (OSError, SyntaxError):
                    pass
            if not cached:
                with Image.open(ROOT / selected[0]["path"]) as im:
                    w, h = im.size
                sheet = Image.new("RGB", (w * 2, h * 2 + 32), "white")
                draw = ImageDraw.Draw(sheet)
                for j, item in enumerate(selected):
                    with Image.open(ROOT / item["path"]) as im:
                        sheet.paste(im.convert("RGB"), (j % 2 * w, 32 + j // 2 * h))
                draw.text(
                    (6, 6),
                    "Past 20 s: top left, top right, bottom left, bottom right",
                    fill="black",
                )
                sheet.save(path, compress_level=1)
            rows["window"].append(
                make_row(
                    path,
                    label,
                    "unattended_items",
                    group,
                    tag,
                    "assistant_visual_review",
                    note + "; past times " + str([x["timestamp"] for x in selected]),
                    origin,
                    "window",
                )
            )
    save_json(DATA / "review_decisions.json", review_info)
    cohorts = []
    for protocol in ["frame", "window"]:
        cohorts.append(
            write_cohort(
                "unattended_expanded_" + protocol,
                rows[protocol],
                {
                    "normal": "An ordinary scene with no evidence of unattended items.",
                    "unattended_items": definitions["unattended_items"],
                },
                "Provisional source-video study; assistant visual review, XML-supported CAVIAR positives; test holds out entire ABODA classroom/walkway camera families. Independent human verification still needed. Windows have 20-second past context, compared with 6 seconds previously.",
                split=False,
            )
        )
    # Do not silently equate source action/pose/person boxes with verified events.
    weak = []
    for item in collect_littering(ROOT / "data/raw/datasets"):
        weak.append(
            make_row(
                item.path,
                "littering",
                "littering",
                "littering:campus_unknown_sources",
                "littering_yolo",
                "weak_action_proxy",
                "Nonempty source action box; target event not independently verified; source IDs/augmentations are not independent recordings",
            )
        )
    for r in weak:
        r["split"] = "dev"
    # All group members remain dev-only; no fabricated normal controls.
    cohorts.append(
        write_cohort(
            "littering_weak",
            weak,
            {
                "normal": "An ordinary scene with no rubbish disposal.",
                "littering": definitions["littering"],
            },
            "Positive-only weak action-label diagnostic. Report candidate identification rate, not verified event recall, precision/FPR or held-out event accuracy. Exact pixel aliases removed.",
            split=False,
        )
    )
    people = []
    for annotation in sorted(
        (ROOT / "data/raw/datasets/loitering").rglob("_annotations.coco.json")
    ):
        content = json.loads(annotation.read_text())
        category = {c["id"]: c["name"] for c in content["categories"]}
        supported = {
            a["image_id"]
            for a in content["annotations"]
            if category.get(a["category_id"]) == "person"
        }
        for im in content["images"]:
            if im["id"] not in supported:
                continue
            path = resolve_original(annotation.parent / im["file_name"])
            if path is None:
                raise ValueError("Missing annotated person frame")
            people.append(
                make_row(
                    path,
                    "loitering",
                    "loitering",
                    "loitering:unknown_recordings",
                    "loitering_person_coco",
                    "weak_person_proxy",
                    "Person box confirms presence only; loitering duration and intention unverified",
                )
            )
    cohorts.append(
        write_cohort(
            "loitering_weak",
            people,
            {
                "normal": "An ordinary scene with no loitering.",
                "loitering": definitions["loitering"],
            },
            "Positive-only dataset-membership diagnostic: model loitering prediction rate on person-labelled candidate frames; not verified loitering recall.",
            split=False,
        )
    )
    # A separately named grounding check on the same images has genuine person labels.
    people_presence = [
        {**r, "label": "person_presence", "task": "person_presence"} for r in people
    ]
    cohorts.append(
        write_cohort(
            "person_presence",
            people_presence,
            {
                "normal": "A scene with no visible person.",
                "person_presence": "A photo containing one or more visible people.",
            },
            "Positive-only person-presence diagnostic; not a loitering test and no false-positive estimate.",
            split=False,
        )
    )
    pose_root = ROOT / "data/raw/datasets/rough sleeping/kaggle/laying_dataset"
    crops = []
    scenes = []
    for image in sorted((pose_root / "images").glob("*.png")):
        label_path = resolve_original(
            pose_root / "labels" / (Path(original_name(image)).stem + ".txt")
        )
        if label_path is None:
            continue
        boxes = [
            line.split()[:5]
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]
        if any(b[0] == "0" for b in boxes):
            scenes.append(
                make_row(
                    image,
                    "rough_sleeping",
                    "rough_sleeping",
                    "rough:synthetic_source",
                    "synthetic_pose",
                    "weak_posture_proxy",
                    "Laying/fallen source pose annotation; bedding/public sleeping not established",
                )
            )
        with Image.open(image) as im:
            im = im.convert("RGB")
            w, h = im.size
            for i, b in enumerate(boxes):
                cls = int(b[0])
                xc, yc, bw, bh = map(float, b[1:])
                if cls not in [0, 1]:
                    raise ValueError("Unknown pose label")
                bounds = (
                    max(0, int((xc - bw / 2) * w)),
                    max(0, int((yc - bh / 2) * h)),
                    min(w, math.ceil((xc + bw / 2) * w)),
                    min(h, math.ceil((yc + bh / 2) * h)),
                )
                path = DATA / "pose_crops" / f"{image.stem}_{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                im.crop(bounds).save(path)
                crops.append(
                    make_row(
                        path,
                        "lying_person" if cls == 0 else "normal",
                        "lying_person",
                        "pose:" + original_name(image).split(".rf.")[0],
                        "synthetic_pose_oracle",
                        "source_bbox_pose_annotation",
                        f"Oracle ground-truth box crop, not deployment detector; source={image.relative_to(ROOT)}; box={bounds}; same-image people share split",
                    )
                )
    cohorts.append(
        write_cohort(
            "rough_sleeping_weak",
            scenes,
            {
                "normal": "An ordinary scene without rough sleeping.",
                "rough_sleeping": definitions["rough_sleeping"],
            },
            "Positive-only posture candidate diagnostic with original rough-sleeping definition; not verified rough-sleeping event recall.",
            split=False,
        )
    )
    cohorts.append(
        write_cohort(
            "lying_pose_oracle",
            crops,
            {
                "normal": "A person standing or walking upright.",
                "lying_person": "A person lying horizontally or fallen on the ground.",
            },
            "Synthetic lying-versus-standing oracle box-crop diagnostic. Ground-truth boxes are free here and unavailable to deployment; not a rough-sleeping event test. Group separation uses source image.",
            split=True,
        )
    )
    save_json(DATA / "cohorts.json", cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["review", "build"])
    args = parser.parse_args()
    if args.stage == "review":
        review()
    else:
        build()


if __name__ == "__main__":
    main()
