"""Cache shared causal tracks and pretrained CLIP scene/multiple-region features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from anomaly_eval.training.clip import Encoder
from anomaly_eval.training.video import TRACK_FEATURES, CausalTracker, track_features

ROOT = Path(__file__).resolve().parents[2]


def build_tracks(data, detections, rows):
    output = data / "tracks.jsonl"
    if output.exists():
        return {r["path"]: r for r in map(json.loads, output.read_text().splitlines())}
    records = [
        r
        for name in ["recordings.jsonl", "fresh_recordings.jsonl"]
        for r in map(json.loads, (data / name).read_text().splitlines())
    ]
    indexed = {}
    for record in records:
        tracker = CausalTracker()
        for frame in record["frames"]:
            tracks = tracker.update(frame["time"], detections[frame["path"]])
            indexed[frame["path"]] = {
                "path": frame["path"],
                "tracks": tracks,
                "geometry": track_features(tracks).tolist(),
            }
    for row in rows:
        for p in row["views"]:
            if p not in indexed:
                tracks = CausalTracker().update(0.0, detections[p])
                indexed[p] = {
                    "path": p,
                    "tracks": tracks,
                    "geometry": track_features(tracks).tolist(),
                }
    output.write_text("".join(json.dumps(r) + "\n" for r in indexed.values()))
    (data / "tracker.json").write_text(
        json.dumps(
            {
                "type": "Hungarian IoU/constant-velocity baseline; not ByteTrack",
                "max_gap_seconds": 2.0,
                "new_track_threshold": 0.5,
                "association_threshold": 0.2,
                "owner_observation_seconds": 2.0,
                "owner_policy": "proximity hypothesis retained after departure; no cold-start ownership inference",
                "feature_names": TRACK_FEATURES,
                "limitations": [
                    "no appearance re-identification",
                    "no camera-motion compensation",
                    "ownership is heuristic",
                ],
                "code_sha256": hashlib.sha256(
                    (ROOT / "anomaly_eval/training/video.py").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return indexed


def regions(tracks):
    persons = sorted(
        [t for t in tracks if t["kind"] == "person"],
        key=lambda t: t["confidence"],
        reverse=True,
    )[:4]
    items = sorted(
        [t for t in tracks if t["kind"] == "item"],
        key=lambda t: t["confidence"],
        reverse=True,
    )[:2]
    boxes = []
    for person in persons:
        x1, y1, x2, y2 = person["box"]
        boxes.extend([[x1, y1, x2, y2], [x1, y1, x2, y1 + 0.55 * (y2 - y1)]])
    person_by_id = {t["id"]: t for t in tracks if t["kind"] == "person"}
    for item in items:
        b = item["box"]
        owner = person_by_id.get(item["owner"])
        if owner is not None:
            p = owner["box"]
            b = [min(b[0], p[0]), min(b[1], p[1]), max(b[2], p[2]), max(b[3], p[3])]
        boxes.append(b)
    output = []
    for x1, y1, x2, y2 in boxes:
        dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        output.append(
            [max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)]
        )
    return output


class Views(Dataset):
    def __init__(self, tasks, encoder):
        self.tasks, self.encoder = tasks, encoder

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        path, box = self.tasks[index]
        if box is None:
            return self.encoder.image_tensor(ROOT / path)
        with Image.open(ROOT / path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            im = im.crop(tuple(x * s for x, s in zip(box, [w, h, w, h])))
            side = max(im.size)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
            canvas.thumbnail((768, 768))
            if self.encoder.name == "mobileclip2-s2":
                return self.encoder.processor(canvas)
            return self.encoder.processor(images=canvas, return_tensors="pt")[
                "pixel_values"
            ][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    args = parser.parse_args()
    data = args.data.resolve()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [
        r
        for name in ["windows.jsonl", "fresh_avss.jsonl"]
        for r in map(json.loads, (data / name).read_text().splitlines())
    ]
    detections = {
        r["path"]: r["detections"]
        for r in map(json.loads, (data / "detections.jsonl").read_text().splitlines())
    }
    tracks = build_tracks(data, detections, rows)
    paths = sorted(tracks)
    output = data / "features" / args.model
    output.mkdir(parents=True, exist_ok=True)
    final = output / "features.npz"
    if final.exists():
        raise ValueError("Completed features are immutable")
    encoder = Encoder(args.model, "cuda").eval()
    tasks = []
    indices = defaultdict(list)
    for p in paths:
        indices[p, "full"].append(len(tasks))
        tasks.append((p, None))
        boxes = regions(tracks[p]["tracks"])
        for box in boxes:
            indices[p, "crop"].append(len(tasks))
            tasks.append((p, box))
    loader = DataLoader(
        Views(tasks, encoder),
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    values = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = batch.cuda(non_blocking=True)
            f = (
                encoder.backbone.encode_image(batch, normalize=True)
                if args.model == "mobileclip2-s2"
                else F.normalize(
                    encoder.backbone.get_image_features(pixel_values=batch), dim=-1
                )
            )
            values.append(f.cpu().numpy())
            if i % 100 == 0:
                print(
                    args.model,
                    "ENCODED",
                    min((i + 1) * 64, len(tasks)),
                    "/",
                    len(tasks),
                    flush=True,
                )
    encoded = np.concatenate(values)
    full = np.stack([encoded[indices[p, "full"][0]] for p in paths])
    crop_mean = []
    crop_max = []
    for i, p in enumerate(paths):
        local = encoded[indices[p, "crop"]] if indices[p, "crop"] else full[i : i + 1]
        crop_mean.append(local.mean(0))
        crop_max.append(local.max(0))
    geometry = np.asarray([tracks[p]["geometry"] for p in paths], dtype=np.float32)
    roi_indices, roi_images, roi_boxes, roi_person = [], [], [], []
    for image_index, p in enumerate(paths):
        person_count = 2 * min(
            4, sum(t["kind"] == "person" for t in tracks[p]["tracks"])
        )
        for j, task_index in enumerate(indices[p, "crop"]):
            roi_indices.append(task_index)
            roi_images.append(image_index)
            roi_boxes.append(tasks[task_index][1])
            roi_person.append(j < person_count)
    np.savez_compressed(
        final,
        paths=np.asarray(paths),
        full=full,
        crop_mean=np.stack(crop_mean),
        crop_max=np.stack(crop_max),
        geometry=geometry,
        roi_features=encoded[roi_indices],
        roi_image_indices=np.asarray(roi_images, dtype=np.int64),
        roi_boxes=np.asarray(roi_boxes, dtype=np.float32),
        roi_is_person=np.asarray(roi_person, dtype=bool),
    )
    (output / "identity.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_revision": encoder.revision,
                "windows_sha256": hashlib.sha256(
                    (data / "windows.jsonl").read_bytes()
                ).hexdigest(),
                "fresh_manifest_sha256": hashlib.sha256(
                    (data / "fresh_avss.jsonl").read_bytes()
                ).hexdigest(),
                "features_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                "images": len(paths),
                "encoded_views": len(tasks),
                "precision": "float32; TF32 disabled",
                "region_policy": "up to four persons, each full and upper-55%-body, plus two item/observed-owner contextual crops; fallback scene",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print("FEATURES COMPLETE", args.model, len(paths), flush=True)


if __name__ == "__main__":
    main()
