"""Extract and cache frozen ViT-B/16 per-frame features for the smoking corpus.

Run from the project root:

    python -m anomaly_eval.vadclip.extract_features \
        --root data/raw/datasets \
        --weights data/models/openai_clip_vitb16.pt \
        --cache-dir results/vadclip/features

For every session it loads each frame through the official CLIP transform, runs the frozen
vision tower (``encode_image`` -> raw [512] features), and writes one ``.npz`` per session:
    feat   : float32 [T, 512]   per-frame features in temporal order
    labels : int8    [T]        dense per-frame smoke label (0/1)
It also writes ``manifest.json`` with the session metadata and the deterministic
session-level train/valid split. Features are cached so training/ablations never re-run the
backbone. No image content is displayed -- only tensors flow through the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.data import code_hashes, file_sha256, fingerprint
from anomaly_eval.paths import Workspace, portable_path, resolve_arguments, resolve_path
from anomaly_eval.runtime import add_device_argument, resolve_device

from .clip_b16 import load_clip_b16, make_preprocess
from .data import (
    SOURCE_IDENTITY_POLICY,
    SmokingSourceError,
    load_smoking_sessions,
    make_three_way_split,
)


def extract_session(
    model, preprocess, frames: list[str], batch_size: int = 32, device="cpu"
) -> np.ndarray:
    """Batched frozen-vision features for an ordered frame list -> [T, 512] float32."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            chunk = frames[i : i + batch_size]
            imgs = torch.stack([preprocess(_open(p)) for p in chunk]).to(device)
            f = model.encode_image(imgs).cpu().numpy()  # [b, 512] raw (unnormalised)
            feats.append(f.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _open(path: str):
    from PIL import Image

    with Image.open(path) as im:
        from PIL import ImageOps

        return ImageOps.exif_transpose(im).convert(
            "RGB"
        )  # decouple from the file handle before batch processing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=Workspace.current().data_dir / "raw" / "datasets",
        help="datasets root (contains smoking/); defaults to configured data root/raw/datasets",
    )
    ap.add_argument("--weights", help="path to openai_clip_vitb16.pt")
    ap.add_argument("--cache-dir")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate annotations, source identity, chronology and splits without loading a model",
    )
    ap.add_argument(
        "--validation-report",
        type=Path,
        help="Save source counts, split membership and all validation issues as JSON",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--valid-frac", type=float, default=0.25)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=234)
    add_device_argument(ap)
    args = resolve_arguments(ap.parse_args(argv))

    if min(args.batch_size, args.threads) < 1:
        ap.error("batch-size and threads must be positive")
    if not args.validate_only and (not args.weights or not args.cache_dir):
        ap.error("weights and cache-dir are required unless --validate-only is set")
    args.root = str(resolve_path(args.root))
    report = {}
    try:
        sessions = load_smoking_sessions(args.root, report=report)
        keys = sorted(sessions)
        try:
            splits = make_three_way_split(
                keys, args.valid_frac, args.test_frac, args.seed
            )
        except ValueError as exc:
            report["status"] = "error"
            report["issues"].append({"kind": "invalid_split", "message": str(exc)})
            raise SmokingSourceError(report) from exc
        report.update(seed=args.seed, splits=splits)
    except SmokingSourceError as exc:
        if args.validation_report:
            _write_report(args.validation_report, report)
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    if args.validation_report:
        _write_report(args.validation_report, report)
    print(
        f"[validate] {report['retained_frames']} source frames in "
        f"{report['session_count']} camera/day groups; "
        f"{report['duplicate_exports']} duplicate exports removed; "
        f"{report['unlabeled_images']} unlabeled images excluded",
        flush=True,
    )
    if args.validate_only:
        return 0

    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.weights = str(resolve_path(args.weights))
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(resolve_device(args.device))
    print(f"[extract] loading ViT-B/16 from {args.weights}", flush=True)
    model = None
    preprocess = None

    weights_sha = file_sha256(Path(args.weights))
    source = {
        k: [
            {
                "sha256": file_sha256(Path(p)),
                "label": y,
                "path": portable_path(Path(p), Path(args.root)),
                "source_id": source_id,
                "timestamp": timestamp,
            }
            for p, y, source_id, timestamp in zip(
                sessions[k].frames,
                sessions[k].labels,
                sessions[k].source_ids,
                sessions[k].timestamps,
            )
        ]
        for k in keys
    }
    identity = fingerprint(
        {
            "source": source,
            "identity_policy": SOURCE_IDENTITY_POLICY,
            "weights": weights_sha,
            "preprocess": "official_clip224_exif_v2",
            "device": str(device),
            "dtype": "float32",
            "splits": splits,
            "code": code_hashes(),
        }
    )
    manifest = {
        "schema": 2,
        "seed": args.seed,
        **splits,
        "sessions": {},
        "fingerprint": identity,
        "weights_sha256": weights_sha,
        "identity_policy": SOURCE_IDENTITY_POLICY,
        "source_validation": report,
        "evaluation_context": "offline_ordered_sparse_camera_date_frames",
        "note": "Frame indices need not be continuous. This is not a streaming latency evaluation.",
        "source_root": portable_path(Path(args.root)),
        "source": source,
        "device": str(device),
        "dtype": "float32",
    }
    old = (
        json.loads((cache_dir / "manifest.json").read_text())
        if (cache_dir / "manifest.json").is_file()
        else {}
    )
    reuse = old.get("fingerprint") == identity
    t0 = time.time()
    for n, key in enumerate(keys):
        s = sessions[key]
        out = cache_dir / (fingerprint(key)[:20] + ".npz")
        if (
            reuse
            and out.is_file()
            and file_sha256(out) == old["sessions"][key]["sha256"]
        ):
            data = np.load(out)
            feat, labels = data["feat"], data["labels"]
        else:
            if model is None:
                model = load_clip_b16(args.weights, device).eval()
                preprocess = make_preprocess(224)
            feat = extract_session(model, preprocess, s.frames, args.batch_size, device)
            labels = np.asarray(s.labels, dtype=np.int8)
            assert feat.shape[0] == len(labels), f"{key}: {feat.shape} vs {len(labels)}"
            np.savez_compressed(out, feat=feat, labels=labels)
        manifest["sessions"][key] = {
            "n_frames": int(feat.shape[0]),
            "n_positive": int(labels.sum()),
            "split": next(split for split, group in splits.items() if key in group),
            "file": out.name,
            "sha256": file_sha256(out),
            "paths": [portable_path(Path(p), Path(args.root)) for p in s.frames],
            "frame_indices": s.frame_indices,
            "timestamps": s.timestamps,
            "source_ids": s.source_ids,
            "ordering": s.ordering,
            "sparse_indices": any(
                b - a != 1 for a, b in zip(s.frame_indices, s.frame_indices[1:])
            ),
        }
        print(
            f"[extract] ({n + 1}/{len(keys)}) {key:<24} T={feat.shape[0]:<5} pos={int(labels.sum()):<4}",
            flush=True,
        )

    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[extract] done in {time.time() - t0:.1f}s -> {cache_dir}", flush=True)


def _write_report(path: Path, report: dict):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
