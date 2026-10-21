"""Run frozen full-stream tile/tracker ablations without changing old results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_eval.training.event_memory import poisson_upper_rate
from anomaly_eval.training.stream_tiles import (
    MODES,
    TILES,
    fuse,
    global_box,
    tile_pixels,
)
from anomaly_eval.training.video import (
    CausalTracker,
    event_metrics,
    make_alerts,
    unattended_state,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generalization_pilot_20261020"
OUT = ROOT / "results/generalization_pilot_20261020"
QUERY = "person . bag . backpack . suitcase . cardboard box . package ."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise ValueError("Existing protocol is frozen; use its saved stages")
    previous = ROOT / "data/video_pilot_20261016"
    windows = read_rows(previous / "modeling_windows.jsonl")
    test_ids = {
        r["recording_id"]
        for r in windows
        if r["task"] == "unattended_items" and r["split"] == "test"
    }
    records = []
    for record in read_rows(previous / "recordings.jsonl"):
        if record["id"] in test_ids and record["mode"] == "continuous":
            records.append(
                {
                    **record,
                    "evaluation_group": "historical_provisional",
                    "previously_inspected": True,
                }
            )
    for record in read_rows(previous / "fresh_recordings.jsonl"):
        group = (
            "avss_censored" if record["right_censored_annotation"] else "avss_primary"
        )
        records.append(
            {**record, "evaluation_group": group, "previously_inspected": True}
        )
    assert len(records) == 10
    for record in records:
        times = [f["time"] for f in record["frames"]]
        assert all(a < b for a, b in pairwise(times))
        assert times[0] == 0 and times[-1] >= record["duration"] - 1
    write_rows(DATA / "recordings.jsonl", records)
    paths = sorted({f["path"] for r in records for f in r["frames"]})
    identities = []
    for path in paths:
        with Image.open(ROOT / path) as im:
            im = im.convert("RGB")
            pixel_hash = hashlib.sha256(
                str(im.size).encode() + im.tobytes()
            ).hexdigest()
        identities.append(
            {"path": path, "sha256": digest(ROOT / path), "rgb_sha256": pixel_hash}
        )
    write_rows(DATA / "input_identities.jsonl", identities)
    checkpoint = ROOT / "data/models/video_grounding"
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            previous / "recordings.jsonl",
            previous / "fresh_recordings.jsonl",
            previous / "modeling_windows.jsonl",
            checkpoint / "repository_metadata.json",
            ROOT / "anomaly_eval/training/stream_tiles.py",
            ROOT / "anomaly_eval/training/video.py",
            Path(__file__),
        ]
    }
    protocol = {
        "date": "2026-10-20",
        "status": "frozen_before_new_stream_predictions",
        "scope": "Continuous full-stream detector -> association -> item/owner state -> alarm comparison",
        "modes": MODES,
        "tiles": TILES,
        "query": QUERY,
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "nms_iou": 0.5,
        "maximum_item_area_fraction": 0.2,
        "sample_rate_hz": 2,
        "tracker": {
            "new_threshold": 0.25,
            "low_threshold": 0.2,
            "max_gap": 2,
            "owner_seconds": 2,
        },
        "state": {"stationary_seconds": 30, "owner_absent_seconds": 0.5},
        "alerts": {"persistence": 2, "release": 1, "cooldown": 10, "max_gap": 1.1},
        "weights_sha256": digest(checkpoint / "model.safetensors"),
        "recordings": len(records),
        "unique_input_images": len(paths),
        "gpu_views": len(paths) * 5,
        "input_manifest_sha256": digest(DATA / "recordings.jsonl"),
        "input_identities_sha256": digest(DATA / "input_identities.jsonl"),
        "source_files_sha256": sources,
        "selection": "No search, retraining, threshold fitting or test-based promotion",
        "precision": "float32 CUDA; TF32 disabled; no latency benchmark or pacing",
        "qualification": "Retrospective existing streams. Published AVSS events separate from provisional ABODA intervals. HARD remains right-censored. Camera diversity and independent human review are not supplied by this ablation.",
        "missing_evidence": "No ownership or observations cannot establish normal truth; unreviewed time remains unknown for scoring",
        "future_sources": "New public sources are audited and frozen separately; no synthetic or annotated-output video is silently added as real CCTV.",
    }
    save(OUT / "protocol.json", protocol)
    save(
        OUT / "run.json",
        {
            "status": "prepared",
            "no_latency_testing": True,
            "protocol_sha256": digest(OUT / "protocol.json"),
        },
    )
    print(
        "FROZEN",
        len(records),
        "streams",
        len(paths),
        "images",
        len(paths) * 5,
        "GPU views",
        flush=True,
    )


def check_inputs():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(DATA / "recordings.jsonl") == protocol["input_manifest_sha256"]
    assert (
        digest(DATA / "input_identities.jsonl") == protocol["input_identities_sha256"]
    )
    for path, value in protocol["source_files_sha256"].items():
        assert digest(ROOT / path) == value, path
    assert (
        digest(ROOT / "data/models/video_grounding/model.safetensors")
        == protocol["weights_sha256"]
    )
    return protocol


def detect(batch_size):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    protocol = check_inputs()
    if (OUT / "results.json").exists():
        raise ValueError("Scored run is immutable")
    if not torch.cuda.is_available():
        raise RuntimeError("Accuracy inference requires CUDA; CPU fallback is disabled")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = read_rows(DATA / "recordings.jsonl")
    paths = sorted({f["path"] for r in records for f in r["frames"]})
    output = DATA / "raw_tile_predictions.jsonl"
    cached = read_rows(output) if output.exists() else []
    done = {(r["path"], tuple(r["tile"])) for r in cached}
    assert len(done) == len(cached), "Duplicate cached view"
    tasks = [(p, t) for p in paths for t in TILES if (p, t) not in done]
    checkpoint = ROOT / "data/models/video_grounding"
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.float32,
            disable_custom_kernels=True,
        )
        .eval()
        .cuda()
    )
    save(
        OUT / "gpu_environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "python": platform.python_version(),
            "precision": "float32",
            "tf32": False,
            "no_latency_testing": True,
        },
    )
    start_time = time.monotonic()
    with output.open("a") as stream, torch.inference_mode():
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            images, positions = [], []
            for path, tile in batch:
                with Image.open(ROOT / path) as im:
                    w, h = im.size
                    region = tile_pixels(tile, w, h)
                    images.append(im.convert("RGB").crop(region))
                    positions.append((region, w, h))
            inputs = processor(
                images=images, text=[QUERY] * len(images), return_tensors="pt"
            ).to("cuda")
            prediction = model(**inputs)
            decoded = processor.post_process_grounded_object_detection(
                prediction,
                inputs.input_ids,
                threshold=protocol["box_threshold"],
                text_threshold=protocol["text_threshold"],
                target_sizes=[im.size[::-1] for im in images],
            )
            for (path, tile), (region, w, h), result in zip(batch, positions, decoded):
                found = []
                for box, score, label in zip(
                    result["boxes"].cpu(), result["scores"].cpu(), result["text_labels"]
                ):
                    item = any(
                        v in str(label).lower()
                        for v in ["bag", "backpack", "suitcase", "box", "package"]
                    )
                    if not item and "person" not in str(label).lower():
                        continue
                    box = global_box(box, region, w, h)
                    if box is None or (
                        item and (box[2] - box[0]) * (box[3] - box[1]) > 0.2
                    ):
                        continue
                    found.append(
                        {
                            "box": box,
                            "label": 1001 if item else 1,
                            "confidence": float(score),
                            "text_label": str(label),
                        }
                    )
                stream.write(
                    json.dumps(
                        {"path": path, "tile": tile, "detections": found},
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            for im in images:
                im.close()
            if start % (batch_size * 20) == 0:
                print(
                    "STREAM VIEWS",
                    min(start + batch_size, len(tasks)),
                    "/",
                    len(tasks),
                    "elapsed_s",
                    round(time.monotonic() - start_time),
                    flush=True,
                )
    final = read_rows(output)
    assert len(final) == protocol["gpu_views"]
    save(
        OUT / "detection_complete.json",
        {
            "status": "complete",
            "views": len(final),
            "raw_sha256": digest(output),
            "protocol_sha256": digest(OUT / "protocol.json"),
        },
    )


def evaluate():
    check_inputs()
    if (OUT / "results.json").exists():
        raise ValueError("Scored run is immutable")
    by_path = defaultdict(list)
    raw = DATA / "raw_tile_predictions.jsonl"
    completion = json.loads((OUT / "detection_complete.json").read_text())
    assert completion["raw_sha256"] == digest(raw)
    for row in read_rows(raw):
        by_path[row["path"]].append(row)
    records = read_rows(DATA / "recordings.jsonl")
    rows, traces, alerts = [], [], []
    for mode in MODES:
        for record in records:
            tracker = CausalTracker(new_threshold=0.25, low_threshold=0.2)
            times, scores = [], []
            diagnostics = {
                "frames": 0,
                "item_visible_frames": 0,
                "owned_item_frames": 0,
                "eligible_frames": 0,
                "person_proposals": 0,
                "item_proposals": 0,
                "unique_item_track_ids": set(),
            }
            for frame in record["frames"]:
                found = fuse(by_path[frame["path"]], mode)
                live = tracker.update(frame["time"], found)
                eligible = unattended_state(live)
                times.append(frame["time"])
                scores.append(float(eligible))
                items = [t for t in live if t["kind"] == "item"]
                owned = [t for t in items if t["owner"] is not None]
                diagnostics["frames"] += 1
                diagnostics["item_visible_frames"] += bool(items)
                diagnostics["owned_item_frames"] += bool(owned)
                diagnostics["eligible_frames"] += eligible
                diagnostics["person_proposals"] += sum(d["label"] == 1 for d in found)
                diagnostics["item_proposals"] += sum(d["label"] == 1001 for d in found)
                diagnostics["unique_item_track_ids"].update(t["id"] for t in items)
                traces.append(
                    {
                        "mode": mode,
                        "recording_id": record["id"],
                        "time": frame["time"],
                        "path": frame["path"],
                        "eligible": eligible,
                        "tracks": live,
                    }
                )
            emitted = make_alerts(times, scores, 0.5)
            # Published event intervals take precedence at shared boundaries.
            events = [
                (a, b)
                for a, b, label in record["intervals"]
                if label == "unattended_items"
            ]
            normal = [
                (a, b) for a, b, label in record["intervals"] if label == "normal"
            ]
            metrics = event_metrics(emitted, events, normal)
            # Original pilot starts stream evaluation at t>=20s. Also report
            # that exact exposure for a fair comparison with its state tables.
            aligned_normal = [(max(a, 20), b) for a, b in normal if b > 20]
            aligned = event_metrics(
                [t for t in emitted if t >= 20], events, aligned_normal
            )
            diagnostics["unique_item_track_ids"] = len(
                diagnostics["unique_item_track_ids"]
            )
            rows.append(
                {
                    "mode": mode,
                    "recording_id": record["id"],
                    "evaluation_group": record["evaluation_group"],
                    "metrics": metrics,
                    "aligned_previous_t20_metrics": aligned,
                    "diagnostics": diagnostics,
                    "published_annotation": record["evaluation_group"].startswith(
                        "avss"
                    ),
                    "previously_inspected": True,
                }
            )
            alerts.extend(
                {"mode": mode, "recording_id": record["id"], "time": t} for t in emitted
            )
    aggregate = []
    for mode in MODES:
        for group in sorted({r["evaluation_group"] for r in rows}):
            group_rows = [
                r for r in rows if r["mode"] == mode and r["evaluation_group"] == group
            ]
            result = {
                "mode": mode,
                "evaluation_group": group,
                "recordings": len(group_rows),
            }
            for metric in [
                "events",
                "detected_events",
                "normal_false_alerts",
                "duplicate_alerts",
                "unknown_time_alerts",
                "normal_seconds",
                "reviewed_seconds",
            ]:
                result[metric] = sum(r["metrics"][metric] for r in group_rows)
            result["event_recall"] = (
                result["detected_events"] / result["events"]
                if result["events"]
                else None
            )
            result["false_alerts_per_normal_camera_hour"] = (
                result["normal_false_alerts"] * 3600 / result["normal_seconds"]
                if result["normal_seconds"]
                else None
            )
            result["false_rate_one_sided_95pct_upper_per_hour"] = poisson_upper_rate(
                result["normal_false_alerts"], result["normal_seconds"]
            )
            delays = [d for r in group_rows for d in r["metrics"]["delays_seconds"]]
            result["median_delay_seconds"] = (
                float(np.median(delays)) if delays else None
            )
            aggregate.append(result)
    write_rows(OUT / "traces.jsonl", traces)
    write_rows(OUT / "alerts.jsonl", alerts)
    save(
        OUT / "results.json",
        {
            "rows": rows,
            "aggregate": aggregate,
            "qualification": json.loads((OUT / "protocol.json").read_text())[
                "qualification"
            ],
        },
    )
    save(
        OUT / "run.json",
        {
            "status": "evaluated",
            "no_latency_testing": True,
            "arms": len(MODES),
            "recordings": len(records),
            "trace_rows": len(traces),
            "alerts": len(alerts),
            "protocol_sha256": digest(OUT / "protocol.json"),
            "raw_sha256": digest(raw),
        },
    )
    print(
        "EVALUATED",
        len(traces),
        "trace decisions;",
        len(alerts),
        "emitted alerts",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "detect", "evaluate"])
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "detect": lambda: detect(args.batch_size),
        "evaluate": evaluate,
    }[args.stage]()


if __name__ == "__main__":
    main()
