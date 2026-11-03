"""Audit all completed training runs and load real checkpoints for inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import Encoder, evaluate, mean_ap
from anomaly_eval.training.data import TASKS, read_records
from anomaly_eval.training.predict import main as predict

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/clip_finetuning_20261015"


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(8)
    state = read(RUN / "run.json")
    assert state["status"] == "complete" and state["no_latency_testing"]
    manifest = ROOT / "data/clip_training_20261015/samples.jsonl"
    rows = read_records(manifest, ROOT)
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit = {
        "study_status": "complete",
        "finalized_date": "2026-10-16",
        "timezone": "Asia/Jakarta",
        "manifest_sha256": fingerprint,
        "samples": len(rows),
        "predictions_recomputed": 0,
        "initial_amp_predictions_recomputed": 0,
        "development_predictions_recomputed": 0,
        "runs": [],
        "checkpoint_checks": [],
        "inference_checks": [],
    }
    for model in state["models"]:
        for seed in state["seeds"]:
            directory = RUN / f"seed_{seed}" / model
            execution = read(directory / "state.json")
            results = read(directory / "results.json")
            selection = read(directory / "selection.json")
            assert execution["status"] == "complete" and execution["device"] == "cuda"
            assert execution["manifest_sha256"] == fingerprint
            assert set(execution["supported_tasks"]) == set(TASKS)
            methods = ["zero_shot", "linear_probe", "finetuned"]
            expected = max(methods, key=lambda method: mean_ap(results[method]["dev"]))
            assert selection["selected"] == expected
            for task in TASKS:
                route = max(
                    methods,
                    key=lambda method: mean_ap({task: results[method]["dev"][task]}),
                )
                assert route == selection["task_routes"][task]
            history = read(directory / "history.json")
            best = max(history, key=lambda entry: entry["dev_macro_ap"])
            assert (best["learning_rate"], best["epoch"]) == (
                selection["finetuned_lr"],
                selection["finetuned_epoch"],
            )
            if seed != 42:
                primary = read(RUN / "seed_42" / model / "selection.json")
                assert {x["learning_rate"] for x in history} == {
                    primary["finetuned_lr"]
                }
            for method in methods:
                for split in ["test", "external"]:
                    path = directory / f"{method}_{split}_predictions.jsonl"
                    predicted = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    expected_rows = [r for r in rows if r["split"] == split]
                    assert [r["id"] for r in predicted] == [
                        r["id"] for r in expected_rows
                    ]
                    assert all(
                        (p["target"], p["task"], p["group_id"])
                        == (r["target"], r["task"], r["group_id"])
                        for p, r in zip(predicted, expected_rows)
                    )
                    scores = np.array(
                        [[r["scores"][t] for t in TASKS] for r in predicted]
                    )
                    assert np.isfinite(scores).all()
                    thresholds = {
                        t: v["thresholds"] for t, v in results[method]["dev"].items()
                    }
                    calculated = evaluate(predicted, scores, thresholds)
                    assert calculated == results[method][split], (
                        model,
                        seed,
                        method,
                        split,
                    )
                    assert (
                        directory / "selection.json"
                    ).stat().st_mtime_ns <= path.stat().st_mtime_ns
                    audit["initial_amp_predictions_recomputed"] += len(predicted)
            final_dir = directory / "float32_evaluation"
            final_state = read(final_dir / "state.json")
            final = read(final_dir / "results.json")
            assert (
                final_state["status"] == "complete"
                and final_state["precision"] == "float32"
                and not final_state["TF32"]
            )
            assert (
                read(final_dir / "selection.json")["task_routes"]
                == selection["task_routes"]
            )
            for method in methods:
                development = [
                    json.loads(line)
                    for line in (final_dir / f"{method}_dev_predictions.jsonl")
                    .read_text()
                    .splitlines()
                ]
                scores = np.array(
                    [[r["scores"][t] for t in TASKS] for r in development]
                )
                calibrated = evaluate(development, scores)
                assert calibrated == final[method]["dev"]
                audit["development_predictions_recomputed"] += len(development)
                thresholds = {t: r["thresholds"] for t, r in calibrated.items()}
                for split in ["test", "external"]:
                    predicted = [
                        json.loads(line)
                        for line in (final_dir / f"{method}_{split}_predictions.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    assert [r["id"] for r in predicted] == [
                        r["id"] for r in rows if r["split"] == split
                    ]
                    scores = np.array(
                        [[r["scores"][t] for t in TASKS] for r in predicted]
                    )
                    assert (
                        evaluate(predicted, scores, thresholds) == final[method][split]
                    )
                    audit["predictions_recomputed"] += len(predicted)
            for relative, expected_hash in execution["code_sha256"].items():
                source = ROOT / relative
                snapshot = RUN / "code_snapshots" / expected_hash / source.name
                if not snapshot.exists():
                    assert (
                        hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
                    )
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, snapshot)
                assert (
                    hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_hash
                )
            audit["runs"].append(
                {
                    "model": model,
                    "seed": seed,
                    "metrics_match": True,
                    "dev_selection_matches": True,
                    "pinned_input_hash_matches": True,
                    "selection_precedes_heldout_scores": True,
                }
            )
    # A weight comparison establishes that the encoder itself was fine-tuned.
    for model in state["models"]:
        base = Encoder(model, "cpu")
        parameters = dict(base.named_parameters())
        for seed in state["seeds"]:
            checkpoint = torch.load(
                RUN / f"seed_{seed}" / model / "finetuned_best.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert (
                tuple(checkpoint["tasks"]) == TASKS
                and checkpoint["base_revision"] == base.revision
            )
            squared = 0.0
            updated = 0
            count = 0
            for name, value in checkpoint["parameters"].items():
                assert (
                    name in parameters
                    and value.shape == parameters[name].shape
                    and torch.isfinite(value).all()
                )
                if not name.startswith("backbone."):
                    continue
                difference = value - parameters[name].detach()
                squared += float(difference.square().sum())
                updated += int(bool(difference.abs().max() > 0))
                count += value.numel()
                assert ".text" not in name
            assert squared > 1e-8 and updated > 0
            audit["checkpoint_checks"].append(
                {
                    "model": model,
                    "seed": seed,
                    "vision_parameters_saved": count,
                    "updated_vision_tensors": updated,
                    "vision_delta_l2": squared**0.5,
                    "base_revision": base.revision,
                }
            )
        del base, parameters
    # Check both static and timed inputs against the actual saved candidate scores.
    for model in state["models"]:
        for task in ["smoking", "unattended_items"]:
            row = next(
                r
                for r in rows
                if r["task"] == task and r["split"] == "test" and r["target"] == 1
            )
            output = RUN / f"inference_check_{model}_{task}.json"
            arguments = [
                "--model",
                model,
                "--images",
                *row["views"],
                "--device",
                "cuda",
                "--operating-point",
                "raw",
                "--output",
                str(output),
            ]
            if task == "unattended_items":
                arguments += ["--timestamps", "0", "6.666667", "13.333333", "20"]
            predict(arguments)
            value = read(output)["predictions"]
            method = value[task]["method_selected_on_development"]
            directory = RUN / "seed_42" / model / "float32_evaluation"
            expected = next(
                json.loads(line)
                for line in (directory / f"{method}_test_predictions.jsonl")
                .read_text()
                .splitlines()
                if json.loads(line)["id"] == row["id"]
            )
            delta = abs(value[task]["score"] - expected["scores"][task])
            assert delta < 0.001, (
                model,
                task,
                delta,
            )
            if task == "smoking":
                assert all(
                    value[t]["decision"] == "unknown"
                    for t in ["littering", "loitering", "unattended_items"]
                )
            audit["inference_checks"].append(
                {
                    "model": model,
                    "task": task,
                    "loaded_checkpoint": True,
                    "saved_score_absolute_difference": delta,
                    "float32_batch_tolerance": 0.001,
                }
            )
    assert len(audit["runs"]) == 6 and audit["predictions_recomputed"] == 46926
    (RUN / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    provenance = read(RUN / "provenance.json")
    provenance["worker_runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    provenance["finalized_date"] = "2026-10-16"
    (RUN / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "Verified six model/seed runs, 46,926 saved predictions, visual-weight updates and checkpoint-backed inference.",
        flush=True,
    )


if __name__ == "__main__":
    main()
