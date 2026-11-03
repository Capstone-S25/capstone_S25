"""Test fixed open-vocabulary person/package detections and causal tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from anomaly_eval.training.video import CausalTracker, track_features

ROOT = Path(__file__).resolve().parents[2]
QUERY = "person . bag . backpack . suitcase . cardboard box . package ."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scope", choices=["train", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    data = args.data.resolve()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = [
        r
        for n in ["recordings.jsonl", "fresh_recordings.jsonl"]
        for r in map(json.loads, (data / n).read_text().splitlines())
        if r["mode"] in {"continuous", "transfer"}
    ]
    if args.scope == "train":
        records = [r for r in records if r.get("split") == "train"]
    paths = sorted({f["path"] for r in records for f in r["frames"]})
    output = data / "grounding_detections.jsonl"
    cached = (
        {r["path"]: r for r in map(json.loads, output.read_text().splitlines())}
        if output.exists()
        else {}
    )
    pending = [p for p in paths if p not in cached]
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
    with output.open("a") as stream, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            images = [Image.open(ROOT / p).convert("RGB") for p in batch]
            sizes = [im.size[::-1] for im in images]
            inputs = processor(
                images=images, text=[QUERY] * len(images), return_tensors="pt"
            ).to("cuda")
            prediction = model(**inputs)
            decoded = processor.post_process_grounded_object_detection(
                prediction,
                inputs.input_ids,
                threshold=0.25,
                text_threshold=0.20,
                target_sizes=sizes,
            )
            for p, (h, w), result in zip(batch, sizes, decoded):
                labels = result.get("text_labels", result.get("labels"))
                found = []
                for box, score, label in zip(
                    result["boxes"].cpu(), result["scores"].cpu(), labels
                ):
                    label = str(label).lower()
                    item = any(
                        v in label
                        for v in ["bag", "backpack", "suitcase", "box", "package"]
                    )
                    if not item and "person" not in label:
                        continue
                    b = [
                        max(0.0, min(1.0, float(x) / s))
                        for x, s in zip(box, [w, h, w, h])
                    ]
                    if b[2] <= b[0] or b[3] <= b[1]:
                        continue
                    if item and (b[2] - b[0]) * (b[3] - b[1]) > 0.2:
                        continue
                    found.append(
                        {
                            "box": b,
                            "label": 1001 if item else 1,
                            "confidence": float(score),
                            "text_label": label,
                        }
                    )
                if found:
                    keep = batched_nms(
                        torch.tensor([d["box"] for d in found]),
                        torch.tensor([d["confidence"] for d in found]),
                        torch.tensor([d["label"] for d in found]),
                        0.5,
                    ).tolist()
                    found = [found[i] for i in keep]
                row = {"path": p, "detections": found}
                cached[p] = row
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            if start % 160 == 0:
                print(
                    "GROUNDED",
                    args.scope,
                    min(start + args.batch_size, len(pending)),
                    "/",
                    len(pending),
                    flush=True,
                )
    tracks = []
    for record in records:
        tracker = CausalTracker(new_threshold=0.25, low_threshold=0.2)
        for frame in record["frames"]:
            live = tracker.update(frame["time"], cached[frame["path"]]["detections"])
            tracks.append(
                {
                    "path": frame["path"],
                    "tracks": live,
                    "geometry": track_features(live).tolist(),
                }
            )
    track_path = data / ("grounding_tracks_" + args.scope + ".jsonl")
    track_path.write_text("".join(json.dumps(r) + "\n" for r in tracks))
    metadata = {
        "model": "IDEA-Research/grounding-dino-tiny",
        "revision": json.loads((checkpoint / "repository_metadata.json").read_text())[
            "sha"
        ],
        "weights_sha256": hashlib.sha256(
            (checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
        "query": QUERY,
        "box_threshold": 0.25,
        "text_threshold": 0.2,
        "nms_iou": 0.5,
        "maximum_item_area_fraction": 0.2,
        "tracker_new_threshold": 0.25,
        "tracker_low_threshold": 0.2,
        "images": len(paths),
        "inference_batch_size": args.batch_size,
        "reused_cached_images": len(paths) - len(pending),
        "precision": "float32; TF32 disabled",
        "scope": args.scope,
        "geometry_sha256": hashlib.sha256(track_path.read_bytes()).hexdigest(),
        "visual_features_changed": False,
        "anomaly_labels_used_for_detection": False,
    }
    (data / ("grounding_" + args.scope + ".json")).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("GROUNDED TRACKS COMPLETE", args.scope, len(paths), flush=True)


if __name__ == "__main__":
    main()
