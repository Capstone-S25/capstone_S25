"""Check new CLIP feature caches against native CUDA image forwards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from anomaly_eval.training.clip import Encoder, save_json
from scripts.generalization.train_source_generalization import MODELS, context, digest
from scripts.strategy.evaluate_strategy_transfer import matrix
from scripts.strategy.extract_strategy_features import Views
from scripts.strategy.run_strategy_screen import predict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/generalization_pilot_20261020/xrf55"
OUT = ROOT / "results/generalization_pilot_20261020"


def main():
    assert torch.cuda.is_available(), "Native validation requires CUDA"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [json.loads(s) for s in (DATA / "samples.jsonl").read_text().splitlines()]
    selected = [
        next(
            r
            for r in rows
            if r["split"] == split and r["target"] == int(split != "train")
        )
        for split in ("train", "dev", "test")
    ]
    paths = sorted({p for r in selected for p in r["views"]})
    detections = {
        r["path"]: r["detections"]
        for r in map(json.loads, (DATA / "detections.jsonl").read_text().splitlines())
    }
    results = []
    for name in MODELS:
        encoder = Encoder(name, "cuda").eval()
        _, _, cached = context(name)
        live = {}
        errors = {}
        for stage in ("full", "crop"):
            dataset = Views(paths, encoder, detections if stage == "crop" else None)
            values = []
            with torch.inference_mode():
                for start in range(0, len(paths), 16):
                    batch = torch.stack(
                        [dataset[i] for i in range(start, min(start + 16, len(paths)))]
                    ).cuda()
                    features = (
                        encoder.backbone.encode_image(batch, normalize=True)
                        if name == "mobileclip2-s2"
                        else F.normalize(
                            encoder.backbone.get_image_features(pixel_values=batch),
                            dim=-1,
                        )
                    )
                    values.extend(features.cpu().numpy())
            live[stage, 0] = dict(zip(paths, values))
            expected = np.stack([cached[stage, 0][p] for p in paths])
            actual = np.stack(values)
            error = float(np.max(np.abs(actual - expected)))
            assert error < 0.0003, (name, stage, error)
            errors[stage] = error
        directory = OUT / "smoking" / name
        selection = json.loads((directory / "selection.json").read_text())
        heads = torch.load(
            directory / "heads.pt", weights_only=True, map_location="cpu"
        )["heads"]
        score_errors = {}
        for family, candidate in selection["selected"].items():
            recipe = selection["candidates"][candidate]["recipe"]
            scores = predict(heads[candidate], matrix(selected, recipe, live, {}))
            expected = predict(heads[candidate], matrix(selected, recipe, cached, {}))
            error = float(np.max(np.abs(scores - expected)))
            assert error < 0.003, (name, family, error)
            score_errors[family] = error
        results.append(
            {
                "model": name,
                "revision": encoder.revision,
                "cases": [r["id"] for r in selected],
                "frames_per_stage": len(paths),
                "stages": 2,
                "max_feature_abs_errors": errors,
                "max_selected_margin_abs_errors": score_errors,
            }
        )
        del encoder
        torch.cuda.empty_cache()
    save_json(
        OUT / "native_feature_validation.json",
        {
            "status": "passed",
            "device": torch.cuda.get_device_name(),
            "precision": "float32; TF32 disabled",
            "models": results,
            "code_sha256": digest(Path(__file__)),
            "manifest_sha256": digest(DATA / "samples.jsonl"),
            "qualification": "Numerical cache validation only; no route, label or threshold changes.",
        },
    )
    print("Native CUDA feature and classifier validation passed", flush=True)


if __name__ == "__main__":
    main()
