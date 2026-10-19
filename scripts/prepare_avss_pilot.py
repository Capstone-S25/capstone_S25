"""Freeze author-annotated AVSS2007 videos as a fresh-source final cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    manifest = data / "fresh_avss.jsonl"
    if manifest.exists():
        raise ValueError("Fresh cohort is already frozen")
    rows, recordings = [], []
    for path in sorted((data / "avss_source").glob("*.mpg")):
        text = path.with_suffix(".txt").read_text()
        # MEDIUM has an unclosed descriptor in the unused schema config. Parse
        # the data block unchanged rather than silently repairing event spans.
        data_block = re.search(r"<data>.*?</data>", text, re.DOTALL).group()
        tree = ET.fromstring(
            '<viper xmlns="http://lamp.cfar.umd.edu/viper#" xmlns:data="http://lamp.cfar.umd.edu/viperdata#">'
            + data_block
            + "</viper>"
        )
        events = []
        for obj in tree.iter():
            if (
                obj.tag.endswith("}object")
                and obj.attrib.get("name") == "AbandonedObject"
            ):
                a, b = map(int, obj.attrib["framespan"].split(":"))
                events.append((a, b))
        if not events:
            raise ValueError("Missing unattended event annotations")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = {round(i * fps * 0.5) for i in range(int(count / fps * 2))}
        directory = data / "fresh_frames" / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        frames = []
        index = 0
        while cap.grab():
            if index in wanted:
                okay, frame = cap.retrieve()
                if not okay:
                    raise ValueError("Failed AVSS decode")
                dest = directory / f"{index:07d}.jpg"
                if not dest.exists() and not cv2.imwrite(
                    str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
                ):
                    raise OSError(dest)
                frames.append(
                    {
                        "index": index,
                        "time": index / fps,
                        "path": str(dest.relative_to(ROOT)),
                    }
                )
            index += 1
        cap.release()
        original_events = events.copy()
        if any(a >= index for a, b in events):
            raise ValueError(f"Event onset exceeds decoded video: {path.name}")
        # Some source end spans include NUMFRAMES; clip only the EOF boundary,
        # retain the original interval and expose this alignment qualification.
        events = [(a, min(b, index - 1)) for a, b in events]
        # Frame spans are source timeline boundaries. Actual encoded FPS wins
        # over the annotation's nominal metadata, which is retained for audit.
        normal = []
        last = 0
        for a, b in sorted(events):
            if a > last:
                normal.append([last / fps, a / fps])
            last = b + 1
        if last < index:
            normal.append([last / fps, index / fps])
        intervals = [[a / fps, b / fps, "unattended_items"] for a, b in events] + [
            [a, b, "normal"] for a, b in normal
        ]
        # Endpoints are half-open for negatives at the positive transition.
        rec = {
            "id": "avss:" + path.stem,
            "video": str(path.relative_to(ROOT)),
            "fps": fps,
            "decoded_frames": index,
            "duration": index / fps,
            "frames": frames,
            "intervals": intervals,
            "events_frames": events,
            "original_events_frames": original_events,
            "group_id": "avss2007:camera",
            "split": "fresh_test",
            "mode": "continuous",
            "label_quality": "published author temporal annotations; >=30 seconds stationary plus owner absent",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "annotation_sha256": hashlib.sha256(
                path.with_suffix(".txt").read_bytes()
            ).hexdigest(),
            "right_censored_annotation": any(b >= index for a, b in original_events),
        }
        recordings.append(rec)
        for i, f in enumerate(frames):
            if f["time"] < 20:
                continue
            target = int(any(a <= f["index"] <= b for a, b in events))
            selected = [
                v for v in frames[: i + 1] if v["time"] >= f["time"] - 20 - 1e-5
            ]
            rows.append(
                {
                    "id": rec["id"] + ":" + str(f["index"]),
                    "recording_id": rec["id"],
                    "task": "unattended_items",
                    "target": target,
                    "source": "avss2007_author_temporal",
                    "split": "fresh_test",
                    "group_id": rec["group_id"],
                    "views": [v["path"] for v in selected],
                    "timestamps": [v["time"] for v in selected],
                    "evaluation_kind": "fresh_source_continuous",
                    "evidence": "published_AbandonedObject_framespans",
                    "previously_inspected": False,
                }
            )
        print("FROZEN", path.stem, len(frames), events, "fps", fps, flush=True)
    if len(recordings) != 3:
        raise ValueError("Expected all three AVSS sequences")
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "fresh_recordings.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recordings)
    )
    (data / "fresh_cohort.json").write_text(
        json.dumps(
            {
                "source": "http://www-vpu.eps.uam.es/publications/AODsurvey/",
                "archive_url": "http://www-vpu.eps.uam.es/publications/AODsurvey/AbandonedObjectDetection-CODE.zip",
                "archive_sha256": hashlib.sha256(
                    (data / "source_metadata/aodsurvey.zip").read_bytes()
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "recordings": 3,
                "conservative_source_groups": 1,
                "note": "Physical camera identities unverified; all three sequences conservatively grouped as one source block. HARD event end exceeds decoded EOF and is right-censored, without rescaling its onset. No images inspected before selection.",
                "frame_origin": "Published zero-based frame spans used without offset; actual encoded FPS retained; review alignment after selection",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
