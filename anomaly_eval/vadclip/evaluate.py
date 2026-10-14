"""Evaluate a frozen supervised VadCLIP smoking adapter on original held-out binary frames.

This is offline inference with future context. Adapter-only latency excludes the
image encoder and must not be compared with the end-to-end image benchmark FPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.data import file_sha256
from anomaly_eval.metrics import frame_metrics
from anomaly_eval.paths import resolve_arguments, resolve_path
from anomaly_eval.runtime import add_device_argument, resolve_device, synchronize

from .clip_b16 import load_clip_b16
from .data import expand_scores
from .model import CLIPVAD
from .train import frame_scores, load_clips


def score_items(model, items, device="cpu"):
    truth, vis, align, groups = [], [], [], []
    start = time.perf_counter()
    model.eval()
    synchronize(device)
    with torch.inference_mode():
        for item in items:
            pv, pa = frame_scores(
                model,
                torch.from_numpy(item["feat"])[None],
                torch.tensor([item["length"]]),
                device,
            )
            n = len(item["raw_label"])
            truth.append(item["raw_label"])
            vis.append(expand_scores(pv[0, : item["length"]].numpy(), n))
            align.append(expand_scores(pa[0, : item["length"]].numpy(), n))
            groups.extend([item["key"]] * n)
    if not truth:
        raise ValueError("No held-out clips")
    y, v, a = np.concatenate(truth), np.concatenate(vis), np.concatenate(align)
    synchronize(device)
    return (
        y,
        {
            "visual": v,
            "align": a,
            "dual_mean": 0.5 * (v + a),
            "dual_max": np.maximum(v, a),
        },
        groups,
        time.perf_counter() - start,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    add_device_argument(p)
    a = resolve_arguments(p.parse_args(argv))
    for field in ("cache_dir", "weights", "run_dir"):
        setattr(a, field, resolve_path(getattr(a, field)))
    if a.threads < 1:
        p.error("threads must be positive")
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=a.threads)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(resolve_device(a.device))
    ckpt = torch.load(
        a.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    cfg = ckpt["config"]
    items, manifest = load_clips(a.cache_dir, cfg["visual_length"])
    if ckpt.get("cache_fingerprint") != manifest["fingerprint"] or ckpt.get(
        "weights_sha256"
    ) != file_sha256(a.weights):
        raise ValueError(
            "Checkpoint, weights and feature-cache provenance do not match"
        )
    if "thresholds" not in ckpt:
        raise ValueError("Re-train checkpoint with development calibration")
    model = CLIPVAD(
        num_class=2,
        clip_model=load_clip_b16(str(a.weights), device),
        device=device,
        visual_length=cfg["visual_length"],
        attn_window=cfg["attn_window"],
        visual_layers=cfg["temporal_layers"],
        use_graph=not cfg.get("no_graph", False),
    ).to(device)
    model.load_state_dict(ckpt["state"])
    y, scores, groups, seconds = score_items(
        model, [items[k] for k in manifest["test"]], device
    )
    result = {
        "protocol": "offline_supervised_smoking",
        "split": "test",
        "device": str(device),
        "cache_fingerprint": manifest["fingerprint"],
        "n_test_frames": len(y),
        "selected_branch": ckpt["selected_branch"],
        "metrics": {
            name: frame_metrics(y, score, ckpt["thresholds"][name])
            for name, score in scores.items()
        },
        "adapter_only_seconds": seconds,
        "encoder_included_in_latency": False,
    }
    (a.run_dir / "eval.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        a.run_dir / "scores.npz",
        y_true=y,
        group_id=np.asarray(groups),
        **{"score_" + k: v for k, v in scores.items()},
    )
    print(json.dumps(result, indent=2))
    return 0
