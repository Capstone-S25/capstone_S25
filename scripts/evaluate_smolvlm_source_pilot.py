"""Frozen original/LoRA transfer on the new scripted-action proxy source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import binary_metrics, save_json, select_points
from anomaly_eval.training.smolvlm import (
    TokenBatches,
    digest,
    predict,
    processor_and_model,
    write_predictions,
)
from scripts.run_smolvlm_finetuning import features

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/generalization_pilot_20261020/xrf55"
OUT = ROOT / "results/generalization_pilot_20261020/smolvlm_transfer"
PRIOR = ROOT / "results/smolvlm_finetuning_20261019"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise ValueError("SmolVLM source protocol already frozen")
    prepared = json.loads((SOURCE / "preparation.json").read_text())
    assert prepared["manifest_sha256"] == digest(SOURCE / "samples.jsonl")
    examples = [
        r for r in rows(SOURCE / "samples.jsonl") if r["split"] in {"dev", "test"}
    ]
    assert len(examples) == 400
    # The validated native prompt accepts up to eight images. Subsample the
    # already frozen sixteen-view manifest identically for both model routes.
    indices = np.linspace(0, 15, 8).round().astype(int).tolist()
    examples = [
        {
            **r,
            **{
                key: [r[key][i] for i in indices]
                for key in ("views", "pixel_hashes", "timestamps", "frames")
                if key in r
            },
        }
        for r in examples
    ]
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
    old = json.loads((PRIOR / "protocol.json").read_text())
    selection = json.loads((PRIOR / "seed_42/selection.json").read_text())
    route = selection["task_routes"]["smoking"]
    save_json(
        OUT / "protocol.json",
        {
            "status": "frozen_before_new_source_scores",
            "manifest_sha256": digest(OUT / "samples.jsonl"),
            "checkpoint_sha256": old["checkpoint_sha256"],
            "model": old["model"],
            "revision": old["revision"],
            "source_manifest_sha256": digest(SOURCE / "samples.jsonl"),
            "routes": ["original", route],
            "adapter_sha256": selection["adapters_sha256"][route],
            "prior_selection_sha256": digest(PRIOR / "seed_42/selection.json"),
            "views": "Eight uniformly spaced views selected from the sixteen frozen source views, identical original/adapted; source smoking training was static in19Octoberstudy",
            "source_view_indices": indices,
            "code_sha256": digest(Path(__file__)),
            "scoring": "Float32 first-answer Yes-minus-No margin; no explanation or latency experiment",
            "calibration": "For both frozen routes, score dev and freeze new-source maximumF1/5%FPR cutoffs before loading any test scores; retain original19Octobercutoffs separately",
            "qualification": "XRF55 scripted smoking-gesture proxy with masked training actors; no verified cigarette/continuous smoking onset truth. New scene/subject source transfer, not fresh CCTV accuracy. No new SmolVLM training.",
        },
    )
    print("SMOLVLM SOURCE CONTROL FROZEN", len(examples), "examples", flush=True)


def run():
    from peft import PeftModel

    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "samples.jsonl") == protocol["manifest_sha256"]
    assert digest(SOURCE / "samples.jsonl") == protocol["source_manifest_sha256"]
    assert digest(Path(__file__)) == protocol["code_sha256"]
    examples = rows(OUT / "samples.jsonl")
    dev = [r for r in examples if r["split"] == "dev"]
    test = [r for r in examples if r["split"] == "test"]
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features(OUT, 64, 6, audited_rows=examples)
    prior = json.loads((PRIOR / "seed_42/selection.json").read_text())
    routes = protocol["routes"]
    # Freeze both route calibrations before the first test forward pass.
    for route in routes:
        name = "original" if route == "original" else "lora"
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        if (directory / "calibration.json").exists():
            continue
        processor, model = processor_and_model()
        if route != "original":
            adapter = PRIOR / route / "adapter"
            assert (
                digest(adapter / "adapter_model.safetensors")
                == protocol["adapter_sha256"]
            )
            model = PeftModel.from_pretrained(model, adapter).float().eval()
        scores, masses = predict(
            model,
            TokenBatches(dev, processor, OUT / "visual_cache"),
            dev,
            8,
            progress=name + " dev",
        )
        write_predictions(directory / "dev_predictions.jsonl", dev, scores, masses)
        save_json(
            directory / "calibration.json",
            {
                "new_source_thresholds": select_points(
                    [r["target"] for r in dev], scores
                ),
                "original19oct_thresholds": prior["thresholds"][route]["smoking"],
                "route": route,
                "uses_test": False,
                "scores_sha256": digest(directory / "dev_predictions.jsonl"),
            },
        )
        del model
        torch.cuda.empty_cache()
    for route in routes:
        name = "original" if route == "original" else "lora"
        directory = OUT / name
        if (directory / "results.json").exists():
            continue
        calibration = json.loads((directory / "calibration.json").read_text())
        processor, model = processor_and_model()
        if route != "original":
            model = (
                PeftModel.from_pretrained(model, PRIOR / route / "adapter")
                .float()
                .eval()
            )
        batches = TokenBatches(test, processor, OUT / "visual_cache")
        scores, masses = predict(model, batches, test, 8, progress=name + " test")
        write_predictions(directory / "test_predictions.jsonl", test, scores, masses)
        # One native-image case per route verifies the new eight-view cache path.
        from anomaly_eval.models import load_image
        from anomaly_eval.training.smolvlm import messages, response_ids

        images = [load_image(ROOT / p, 512, "letterbox") for p in test[0]["views"]]
        native = processor.apply_chat_template(
            messages("smoking", len(images)), add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[native], images=[images], return_tensors="pt").to(
            "cuda"
        )
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            logits = (
                model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            )
        no, yes = response_ids(processor)
        live = float((logits[yes] - logits[no]).item())
        error = abs(live - float(scores[0]))
        assert error < 0.002, (live, scores[0])
        result = {
            "route": route,
            "examples": len(test),
            "positive": sum(r["target"] for r in test),
            "new_source_calibrated": {
                point: binary_metrics([r["target"] for r in test], scores, cutoff)
                for point, cutoff in calibration["new_source_thresholds"].items()
            },
            "original19oct_cutoffs": {
                point: binary_metrics([r["target"] for r in test], scores, cutoff)
                for point, cutoff in calibration["original19oct_thresholds"].items()
            },
            "yes_no_token_mass_min": float(masses.min()),
            "native_case": test[0]["id"],
            "native_cached_score_abs_error": error,
            "qualification": protocol["qualification"],
        }
        save_json(directory / "results.json", result)
        del model
        torch.cuda.empty_cache()
    save_json(
        OUT / "run.json",
        {
            "status": "complete",
            "routes": len(routes),
            "development_examples": len(dev),
            "heldout_examples": len(test),
            "no_latency_testing": True,
            "no_training": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.stage]()


if __name__ == "__main__":
    main()
