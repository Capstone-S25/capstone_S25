"""Replay SmolVLM selection/metrics and check selected adapters on native inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.models import load_image
from anomaly_eval.training.clip import save_json
from anomaly_eval.training.data import TASKS, read_records
from anomaly_eval.training.smolvlm import (
    ROOT,
    SPEC,
    TokenBatches,
    digest,
    evaluate_scores,
    local_checkpoint,
    macro_ap,
    messages,
    processor_and_model,
    response_ids,
    task_ap,
)


def read(path):
    return json.loads(path.read_text())


def predictions(path, rows):
    values = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(values) == len(rows)
    for value, row in zip(values, rows):
        assert all(
            value[k] == row[k]
            for k in ("id", "task", "target", "source", "group_id", "split")
        )
    scores = np.array([r["score"] for r in values])
    assert np.isfinite(scores).all()
    assert all(0 <= r["yes_no_token_mass"] <= 1.00001 for r in values)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/smolvlm_finetuning_20261019"
    )
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()
    study = args.study
    protocol = read(study / "protocol.json")
    assert digest(study / "samples.jsonl") == protocol["manifest_sha256"]
    assert (
        digest(Path(local_checkpoint()) / "model.safetensors")
        == protocol["checkpoint_sha256"]
    )
    rows = read_records(study / "samples.jsonl", ROOT, check_files=True)
    dev = [r for r in rows if r["split"] == "dev"]
    index = read(study / "visual_cache/index.json")
    assert index["manifest_sha256"] == protocol["manifest_sha256"]
    assert index["tokens_sha256"] == digest(study / "visual_cache/tokens.npy")
    tokens = np.load(study / "visual_cache/tokens.npy", mmap_mode="r")
    assert list(tokens.shape) == index["shape"]
    for start in range(0, len(tokens), 128):
        assert np.isfinite(tokens[start : start + 128]).all()
    audit = {
        "status": "running",
        "manifest_sha256": protocol["manifest_sha256"],
        "examples": len(rows),
        "decoded_images_checked": len(index["pixel_to_index"]),
        "development_predictions_replayed": 0,
        "heldout_predictions_replayed": 0,
        "adapter_checks": [],
        "live_checks": [],
        "seeds": [],
    }
    originals = predictions(study / "baseline/dev_predictions.jsonl", dev)
    assert evaluate_scores(dev, originals) == read(
        study / "baseline/development_results.json"
    )
    audit["development_predictions_replayed"] += len(dev)
    all_dev_scores = {"original": originals}
    evaluation_paths = set()
    selections = sorted(study.glob("seed_*/selection.json"))
    assert any(p.parent.name == "seed_42" for p in selections)
    for path in selections:
        selection = read(path)
        assert selection["manifest_sha256"] == protocol["manifest_sha256"]
        assert selection["selection_used_test"] is False
        audit["seeds"].append(selection["seed"])
        scores = {"original": originals}
        for route, expected in selection["adapters_sha256"].items():
            candidate = study / route
            assert digest(candidate / "adapter/adapter_model.safetensors") == expected
            config = read(candidate / "adapter/adapter_config.json")
            assert config["base_model_name_or_path"] == SPEC.checkpoint
            assert config["revision"] == SPEC.revision
            state = read(candidate / "state.json")
            history = read(candidate / "history.json")
            assert state["status"] == "complete" and state["device"] == "cuda"
            best = max(history, key=lambda h: h["dev_macro_ap"])
            assert state["best_epoch"] == best["epoch"]
            assert state["best_dev_macro_ap"] == best["dev_macro_ap"]
            assert all(
                h["unique_training_examples"] == state["training_rows"] for h in history
            )
            scores[route] = predictions(candidate / "dev_predictions.jsonl", dev)
            all_dev_scores[route] = scores[route]
            assert macro_ap(dev, scores[route]) == state["best_dev_macro_ap"]
            assert evaluate_scores(dev, scores[route]) == read(
                candidate / "development_results.json"
            )
            from safetensors.torch import load_file

            weights = load_file(candidate / "adapter/adapter_model.safetensors")
            assert all("lora_" in key and "text_model.layers" in key for key in weights)
            assert all(torch.isfinite(t).all() for t in weights.values())
            nonzero = sum(
                int((t != 0).sum()) for k, t in weights.items() if "lora_B" in k
            )
            assert (
                nonzero > 0
            )  # B starts at zero: this establishes actual weight updates.
            audit["adapter_checks"].append(
                {
                    "route": route,
                    "parameters": sum(t.numel() for t in weights.values()),
                    "nonzero_B_entries": nonzero,
                    "sha256": expected,
                }
            )
            audit["development_predictions_replayed"] += len(dev)
        trained = [r for r in scores if r != "original"]
        assert selection["joint_adapter"] == max(
            trained, key=lambda route: macro_ap(dev, scores[route])
        )
        for task in TASKS:
            route = max(scores, key=lambda route: task_ap(dev, scores[route], task))
            assert selection["task_routes"][task] == route
        for route, values in scores.items():
            assert selection["thresholds"][route] == {
                t: v["thresholds"] for t, v in evaluate_scores(dev, values).items()
            }
        required = set(selection["task_routes"].values()) | {
            selection["joint_adapter"],
            "original",
        }
        for route in required:
            directory = study / "evaluation" / route.replace("/", "__")
            if directory in evaluation_paths:
                continue
            evaluation_paths.add(directory)
            results = read(directory / "results.json")
            for split in ("test", "external"):
                selected = [r for r in rows if r["split"] == split]
                values = predictions(directory / f"{split}_predictions.jsonl", selected)
                assert (
                    evaluate_scores(selected, values, selection["thresholds"][route])
                    == results[split]
                )
                assert (
                    path.stat().st_mtime_ns
                    < (directory / f"{split}_predictions.jsonl").stat().st_mtime_ns
                    or route == "original"
                )
                audit["heldout_predictions_replayed"] += len(selected)
    repeated_path = study / "repeated_gain_routes.json"
    if repeated_path.exists():
        repeated = read(repeated_path)
        primary = read(study / "seed_42/selection.json")
        secondary = read(study / "seed_43/selection.json")
        assert repeated["primary_selection_sha256"] == digest(
            study / "seed_42/selection.json"
        )
        assert repeated["secondary_selection_sha256"] == digest(
            study / "seed_43/selection.json"
        )
        assert repeated["used_heldout_predictions"] is False
        for task in TASKS:
            original = task_ap(dev, all_dev_scores["original"], task)
            route = primary["task_routes"][task]
            first = task_ap(dev, all_dev_scores[route], task) - original
            second = (
                task_ap(dev, all_dev_scores[secondary["joint_adapter"]], task)
                - original
            )
            assert repeated["development_gains"][task] == {
                "primary_development_ap_gain": first,
                "secondary_development_ap_gain": second,
            }
            assert repeated["task_routes"][task] == (
                route if first > 0 and second > 0 else "original"
            )
        audit["optional_repeated_gain_routes"] = repeated["task_routes"]
    transfer = study / "additional_transfer"
    if (transfer / "protocol.json").exists():
        transfer_protocol = read(transfer / "protocol.json")
        assert (
            digest(transfer / "samples.jsonl") == transfer_protocol["manifest_sha256"]
        )
        transfer_rows = [
            json.loads(s) for s in (transfer / "samples.jsonl").read_text().splitlines()
        ]
        assert not (
            {r["group_id"] for r in rows} & {r["group_id"] for r in transfer_rows}
        )
        transfer_index = read(transfer / "visual_cache/index.json")
        assert transfer_index["manifest_sha256"] == transfer_protocol["manifest_sha256"]
        assert transfer_index["tokens_sha256"] == digest(
            transfer / "visual_cache/tokens.npy"
        )
        assert not (
            {p for r in rows for p in r["pixel_hashes"]}
            & {p for r in transfer_rows for p in r["pixel_hashes"]}
        )
        primary = read(study / "seed_42/selection.json")
        required = set(primary["task_routes"].values()) | {
            "original",
            primary["joint_adapter"],
        }
        for route in required:
            assert (
                transfer / "evaluation" / route.replace("/", "__") / "results.json"
            ).is_file()
        from concurrent.futures import ThreadPoolExecutor

        from anomaly_eval.data import image_pixel_sha256

        expected_pixels = {
            path: pixel
            for row in transfer_rows
            for path, pixel in zip(row["views"], row["pixel_hashes"])
        }
        paths = list(expected_pixels)
        with ThreadPoolExecutor(12) as pool:
            for path, actual in zip(
                paths, pool.map(lambda p: image_pixel_sha256(ROOT / p), paths)
            ):
                assert actual == expected_pixels[path]
        audit["transfer_decoded_images_checked"] = len(set(expected_pixels.values()))
        for directory in sorted((transfer / "evaluation").glob("*")):
            route = directory.name.replace("__", "/")
            selection = next(
                read(p)
                for p in selections
                if route == "original" or route in read(p)["thresholds"]
            )
            values = predictions(
                directory / "external_predictions.jsonl", transfer_rows
            )
            assert evaluate_scores(
                transfer_rows, values, selection["thresholds"][route]
            ) == read(directory / "results.json")
            audit["heldout_predictions_replayed"] += len(transfer_rows)
    if not args.skip_live:
        from peft import PeftModel

        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        live_routes = {"original"}
        for path in selections:
            selection = read(path)
            live_routes.update(selection["task_routes"].values())
            live_routes.add(selection["joint_adapter"])
        for route in sorted(live_routes):
            processor, model = processor_and_model()
            if route != "original":
                model = (
                    PeftModel.from_pretrained(model, study / route / "adapter")
                    .float()
                    .eval()
                )
            test = [r for r in rows if r["split"] == "test"]
            saved = predictions(
                study
                / "evaluation"
                / route.replace("/", "__")
                / "test_predictions.jsonl",
                test,
            )
            batches = TokenBatches(test, processor, study / "visual_cache")
            for task in TASKS:
                position = next(
                    i
                    for i, r in enumerate(test)
                    if r["task"] == task and r["target"] == 1
                )
                row = test[position]
                text = processor.apply_chat_template(
                    messages(task, len(row["views"])), add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=[
                        [load_image(ROOT / p, 512, "letterbox") for p in row["views"]]
                    ],
                    return_tensors="pt",
                ).to("cuda")
                inputs["position_ids"] = (
                    inputs["attention_mask"].cumsum(-1) - 1
                ).clamp_min(0)
                with torch.inference_mode():
                    native = (
                        model(**inputs, logits_to_keep=1, use_cache=False)
                        .logits[0, -1]
                        .float()
                    )
                    cached_inputs, _ = batches.batch([position], torch.float32)
                    cached = (
                        model(**cached_inputs, logits_to_keep=1, use_cache=False)
                        .logits[0, -1]
                        .float()
                    )
                ids = response_ids(processor)
                score = float(native[ids[1]] - native[ids[0]])
                cached_score = float(cached[ids[1]] - cached[ids[0]])
                error = abs(score - saved[position])
                assert error < 0.002, (route, task, score, saved[position])
                assert abs(cached_score - score) < 0.002
                audit["live_checks"].append(
                    {
                        "route": route,
                        "task": task,
                        "id": row["id"],
                        "saved_score": float(saved[position]),
                        "native_score": score,
                        "native_saved_abs_error": error,
                        "cache_native_abs_error": abs(cached_score - score),
                    }
                )
            del model
            torch.cuda.empty_cache()
    audit["status"] = "passed"
    save_json(study / "verification.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
