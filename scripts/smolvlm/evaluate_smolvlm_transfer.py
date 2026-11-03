"""Evaluate frozen SmolVLM selections on additional, untrained source domains."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.data import image_pixel_sha256
from anomaly_eval.training.clip import save_json
from anomaly_eval.training.smolvlm import (
    ROOT,
    TokenBatches,
    counts,
    digest,
    evaluate_scores,
    predict,
    processor_and_model,
    write_predictions,
)
from scripts.smolvlm.run_smolvlm_finetuning import DEFAULT, features


def prepare(study):
    destination = study / "additional_transfer"
    destination.mkdir(exist_ok=True)
    if (destination / "protocol.json").exists():
        return [
            json.loads(s)
            for s in (destination / "samples.jsonl").read_text().splitlines()
        ]
    main = [json.loads(s) for s in (study / "samples.jsonl").read_text().splitlines()]
    groups = {r["group_id"] for r in main}
    pixels = {p for r in main for p in r["pixel_hashes"]}
    inputs = [
        ROOT / f"data/strategy_study_20261016/{kind}_screened.jsonl"
        for kind in ("hmdb", "rough", "meva")
    ]
    proposed = [json.loads(s) for p in inputs for s in p.read_text().splitlines()]
    paths = sorted({v for r in proposed for v in r["views"]})
    with ThreadPoolExecutor(12) as pool:
        actual = dict(
            zip(paths, pool.map(lambda p: image_pixel_sha256(ROOT / p), paths))
        )
    rows, excluded, repaired = [], [], []
    for row in proposed:
        hashes = [actual[p] for p in row["views"]]
        if row["group_id"] in groups or any(p in pixels for p in hashes):
            excluded.append({"id": row["id"], "reason": "group_or_pixels_overlap_main"})
            continue
        if hashes != row.get("pixel_hashes"):
            repaired.append(row["id"])
        rows.append({**row, "pixel_hashes": hashes, "split": "external"})
    rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate transfer example identity")
    manifest = destination / "samples.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main_protocol = json.loads((study / "protocol.json").read_text())
    save_json(
        destination / "protocol.json",
        {
            "manifest_sha256": digest(manifest),
            "checkpoint_sha256": main_protocol["checkpoint_sha256"],
            "main_manifest_sha256": main_protocol["manifest_sha256"],
            "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "counts": counts(rows),
            "excluded": excluded,
            "repaired_pixel_metadata_ids": repaired,
            "labels_views_splits_used_for_training": False,
            "selection_uses_these_scores": False,
            "qualification": "Previously inspected source-transfer cohorts; HMDB primary-action controls are non-exhaustive negatives and eight-view smoking differs from static training. MEVA action windows do not prove persistent unattended-item status. Public-sleep labels provisional.",
        },
    )
    print(
        "Additional transfer frozen",
        len(rows),
        "examples; pixel metadata repaired",
        len(repaired),
        flush=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows = prepare(args.study)
    if args.prepare_only:
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.study / "additional_transfer"
    features(directory, 64, 6, audited_rows=rows)
    from peft import PeftModel

    selection = json.loads(
        (args.study / f"seed_{args.seed}/selection.json").read_text()
    )
    required = set(selection["task_routes"].values()) | {
        "original",
        selection["joint_adapter"],
    }
    for route in sorted(required):
        out = directory / "evaluation" / route.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "results.json").exists():
            continue
        processor, model = processor_and_model()
        if route != "original":
            adapter = args.study / route / "adapter"
            if (
                digest(adapter / "adapter_model.safetensors")
                != selection["adapters_sha256"][route]
            ):
                raise ValueError("Frozen adapter changed before transfer evaluation")
            model = PeftModel.from_pretrained(model, adapter).float().eval()
        batches = TokenBatches(rows, processor, directory / "visual_cache")
        scores, masses = predict(model, batches, rows, 8, progress="Transfer " + route)
        write_predictions(out / "external_predictions.jsonl", rows, scores, masses)
        save_json(
            out / "results.json",
            evaluate_scores(rows, scores, selection["thresholds"][route]),
        )
        del model
        torch.cuda.empty_cache()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite transfer scores")


if __name__ == "__main__":
    main()
