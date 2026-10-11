"""Frozen CLIP and supervised linear-probe baselines on the SAME original test frames.

Thresholds are fitted on dev only. Fixed C=1 is not tuned on test. Frozen encoder
features are shared with VadCLIP, but supervision differs from zero-shot VLMs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from anomaly_eval.data import file_sha256
from anomaly_eval.metrics import frame_metrics, select_threshold
from anomaly_eval.paths import resolve_arguments, resolve_path

from .clip_b16 import load_clip_b16, tokenize
from .data import PROMPTS
from .train import load_clips


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "weights", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = resolve_arguments(p.parse_args(argv))
    for field in ("cache_dir", "weights", "out_dir"):
        setattr(a, field, resolve_path(getattr(a, field)))
    if a.threads < 1:
        p.error("threads must be positive")
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=a.threads)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    items, manifest = load_clips(a.cache_dir)
    if manifest["weights_sha256"] != file_sha256(a.weights):
        raise ValueError("Encoder weights differ from cache")
    encoder = load_clip_b16(str(a.weights), torch.device("cpu"))
    with torch.inference_mode():
        tokens = tokenize(PROMPTS)
        texts = torch.nn.functional.normalize(
            encoder.encode_text(encoder.encode_token(tokens), tokens), dim=-1
        ).numpy()
        scale = float(encoder.logit_scale.exp())
    arrays = {
        split: (
            np.concatenate([items[k]["raw_feat"] for k in manifest[split]]),
            np.concatenate([items[k]["raw_label"] for k in manifest[split]]),
        )
        for split in ("train", "dev", "test")
    }
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=manifest["seed"])
    clf.fit(*arrays["train"])

    def scores(x):
        norm = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
        logits = scale * norm @ texts.T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return {
            "zeroshot": (e / e.sum(axis=1, keepdims=True))[:, 1],
            "linearprobe": clf.predict_proba(x)[:, 1],
        }

    dev, test = scores(arrays["dev"][0]), scores(arrays["test"][0])
    thresholds = {
        name: select_threshold(arrays["dev"][1], s) for name, s in dev.items()
    }
    result = {
        "split": "test",
        "cache_fingerprint": manifest["fingerprint"],
        "metrics": {
            name: frame_metrics(arrays["test"][1], s, thresholds[name])
            for name, s in test.items()
        },
        "thresholds": thresholds,
        "encoder_included_in_latency": False,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "baselines.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        a.out_dir / "scores.npz",
        y_true=arrays["test"][1],
        **{"score_" + k: v for k, v in test.items()},
    )
    print(json.dumps(result, indent=2))
    return 0
