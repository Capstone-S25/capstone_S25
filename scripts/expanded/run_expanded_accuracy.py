"""Run expanded accuracy cohorts on CUDA; no CPU/latency replay is scheduled."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from anomaly_eval.analysis import analyze, calibrate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/expanded_20261013"
OUTPUT = ROOT / "results/expanded_accuracy_20261013"
MODELS = ["mobileclip2-s2", "siglip2-b16-224", "smolvlm-500m", "minicpm-v4"]


def save(value):
    (OUTPUT / "state.json").write_text(json.dumps(value, indent=2) + "\n")


def main():
    cohorts = [
        json.loads(p.read_text())
        for p in sorted((DATA / "cohorts").glob("*_dataset.json"))
    ]
    # Small grounding study and real event cohorts finish before larger diagnostics.
    order = [
        "unattended_expanded_frame",
        "unattended_expanded_window",
        "lying_pose_oracle",
        "rough_sleeping_weak",
        "person_presence",
        "loitering_weak",
        "littering_weak",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "logs").mkdir()
    state = {
        "status": "running",
        "kind": "expanded_GPU_accuracy_only",
        "cohorts": cohorts,
        "models": MODELS,
        "runs": [],
        "no_latency_experiments": True,
        "previous_smoking_and_vadclip_results": "results/gpu_all_models_20261013",
        "review_decisions": "data/expanded_20261013/review_decisions.json",
    }
    save(state)
    for name in order:
        metadata = DATA / "cohorts" / (name + "_dataset.json")
        deadline = time.monotonic() + 3600
        while not metadata.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f"Preparation did not finish: {name}")
            time.sleep(1)
        cohort = json.loads(metadata.read_text())
        if not any(c["name"] == name for c in state["cohorts"]):
            state["cohorts"].append(cohort)
        save(state)
        for split in ["dev", "test"]:
            if not cohort["counts"][split]:
                continue
            destination = OUTPUT / name / split
            command = [
                str(ROOT / "scripts/evaluate_gpu.sh"),
                "benchmark",
                "--manifest",
                str(ROOT / cohort["manifest"]),
                "--definitions",
                str(ROOT / cohort["definitions"]),
                "--output",
                str(destination),
                "--split",
                split,
                "--limit",
                "0",
                "--models",
                *MODELS,
                "--device",
                "cuda",
                "--dtype",
                "auto",
                "--threads",
                "8",
                "--workers",
                "4",
                "--batch-size",
                "64",
                "--image-policy",
                "letterbox",
                "--max-side",
                "768",
                "--max-new-tokens",
                "64",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--timeout-seconds",
                "14400",
                "--offline",
            ]
            print("Starting", name, split, cohort["counts"][split], flush=True)
            log = OUTPUT / "logs" / f"{name}_{split}.log"
            run = {
                "cohort": name,
                "split": split,
                "output": str(destination.relative_to(ROOT)),
                "command": command,
                "status": "running",
            }
            state["runs"].append(run)
            save(state)
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            run["exit_code"] = result.returncode
            if result.returncode:
                run["status"] = "failed"
                state["status"] = "incomplete"
                save(state)
                raise RuntimeError(f"Failed {name}/{split}; see {log}")
            for model in MODELS:
                summary = json.loads((destination / model / "summary.json").read_text())
                if summary["status"] != "ok" or summary["errors"]:
                    state["status"] = "incomplete"
                    run["status"] = "model_failed"
                    save(state)
                    raise RuntimeError(f"Model failure: {name}/{split}/{model}")
            # Do not print bootstrap intervals from a single unsupported-source proxy group.
            if len(cohort["groups"][split]) > 1:
                analyze(destination, 2000, 42)
            run["status"] = "ok"
            save(state)
            print("Completed", name, split, flush=True)
        if cohort["counts"]["dev"] and cohort["counts"]["test"]:
            calibrate(OUTPUT / name / "dev", OUTPUT / name / "test", 0.05)
    state["status"] = "complete"
    save(state)
    print("Expanded GPU accuracy runs complete:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
