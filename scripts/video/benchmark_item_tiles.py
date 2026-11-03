"""Compare fixed full-frame and tiled detections on source-box diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from anomaly_eval.training.strategies import overlap
from scripts.video.detect_grounded_pilot import QUERY

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/event_improvements_20261017"
OUT = ROOT / "results/event_improvements_20261017"


def main():
    if (OUT / "component_results.json").exists():
        raise ValueError("Completed component evaluation is immutable")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    rows = [
        json.loads(s)
        for s in (DATA / "component_frames.jsonl").read_text().splitlines()
    ]
    tasks = [
        (r, tile)
        for r in rows
        for tile in [
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.6, 0.6),
            (0.4, 0.0, 1.0, 0.6),
            (0.0, 0.4, 0.6, 1.0),
            (0.4, 0.4, 1.0, 1.0),
        ]
    ]
    output = DATA / "tiled_detections.jsonl"
    cached = (
        [json.loads(s) for s in output.read_text().splitlines()]
        if output.exists()
        else []
    )
    done = {(r["path"], tuple(r["tile"])) for r in cached}
    tasks = [(r, t) for r, t in tasks if (r["path"], t) not in done]
    with output.open("a") as stream, torch.inference_mode():
        for start in range(0, len(tasks), 8):
            batch = tasks[start : start + 8]
            images, regions, sizes = [], [], []
            for r, t in batch:
                with Image.open(ROOT / r["path"]) as image:
                    w, h = image.size
                    box = (
                        round(t[0] * w),
                        round(t[1] * h),
                        round(t[2] * w),
                        round(t[3] * h),
                    )
                    im = image.convert("RGB").crop(box)
                    images.append(im)
                    sizes.append(im.size[::-1])
                    regions.append((box, w, h))
            inputs = processor(
                images=images, text=[QUERY] * len(images), return_tensors="pt"
            ).to("cuda")
            pred = model(**inputs)
            decoded = processor.post_process_grounded_object_detection(
                pred,
                inputs.input_ids,
                threshold=0.25,
                text_threshold=0.2,
                target_sizes=sizes,
            )
            for (r, t), result, (box, w, h) in zip(batch, decoded, regions):
                found = []
                for b, s, label in zip(
                    result["boxes"].cpu(), result["scores"].cpu(), result["text_labels"]
                ):
                    item = any(
                        v in str(label).lower()
                        for v in ["bag", "backpack", "suitcase", "box", "package"]
                    )
                    if not item and "person" not in str(label).lower():
                        continue
                    b = [
                        max(0.0, min(1.0, (float(v) + offset) / size))
                        for v, offset, size in zip(
                            b, [box[0], box[1], box[0], box[1]], [w, h, w, h]
                        )
                    ]
                    if (
                        b[2] <= b[0]
                        or b[3] <= b[1]
                        or (item and (b[2] - b[0]) * (b[3] - b[1]) > 0.2)
                    ):
                        continue
                    found.append(
                        {
                            "box": b,
                            "label": 1001 if item else 1,
                            "confidence": float(s),
                            "text_label": str(label),
                        }
                    )
                value = {"path": r["path"], "tile": t, "detections": found}
                cached.append(value)
                stream.write(json.dumps(value) + "\n")
            stream.flush()
            if start % 80 == 0:
                print("TILES", min(start + 8, len(tasks)), "/", len(tasks), flush=True)
    by_path = defaultdict(list)
    for r in cached:
        by_path[r["path"]].append(r)
    aggregate = defaultdict(
        lambda: {"matched": 0, "reference_boxes": 0, "predicted_items": 0, "frames": 0}
    )
    predictions = []
    for r in rows:
        assert len(by_path[r["path"]]) == 5
        for mode in ["full", "full_plus_tiles"]:
            ds = [
                d
                for x in by_path[r["path"]]
                if mode != "full" or tuple(x["tile"]) == (0.0, 0.0, 1.0, 1.0)
                for d in x["detections"]
                if d["label"] == 1001
            ]
            if ds:
                keep = batched_nms(
                    torch.tensor([d["box"] for d in ds]),
                    torch.tensor([d["confidence"] for d in ds]),
                    torch.zeros(len(ds), dtype=torch.int64),
                    0.5,
                ).tolist()
                ds = [ds[i] for i in keep]
            predictions.append(
                {
                    "path": r["path"],
                    "source": r["source"],
                    "mode": mode,
                    "detections": ds,
                }
            )
            for cutoff in [0.3, 0.5]:
                matched = 0
                if ds:
                    cost = np.asarray(
                        [[1 - overlap(g, d["box"]) for d in ds] for g in r["boxes"]]
                    )
                    a, b = linear_sum_assignment(cost)
                    matched = sum(cost[i, j] <= 1 - cutoff for i, j in zip(a, b))
                v = aggregate[(r["source"], mode, cutoff)]
                v["matched"] += int(matched)
                v["reference_boxes"] += len(r["boxes"])
                v["predicted_items"] += len(ds)
                v["frames"] += 1
    results = [
        {
            "source": s,
            "mode": m,
            "iou": c,
            **v,
            "recall": v["matched"] / v["reference_boxes"],
        }
        for (s, m, c), v in aggregate.items()
    ]
    (OUT / "component_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions)
    )
    (OUT / "component_results.json").write_text(
        json.dumps(
            {
                "aggregate": results,
                "images": len(rows),
                "encoded_views": len(cached),
                "device": "cuda",
                "precision": "float32; TF32 disabled",
                "query": QUERY,
                "batch_size": 8,
                "model_revision": json.loads(
                    (checkpoint / "repository_metadata.json").read_text()
                )["sha"],
                "model_sha256": hashlib.sha256(
                    (checkpoint / "model.safetensors").read_bytes()
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    (DATA / "component_frames.jsonl").read_bytes()
                ).hexdigest(),
                "qualification": "Partial item references: recall only, no item precision. Fixed geometric tiles, no source boxes used for inference; training CAVIAR component and highway-debris stress test, not fresh unattended event validation.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
