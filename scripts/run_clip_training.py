"""Prepare frozen source holdouts and run two CLIP models with three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/training_sources_20261015"
OUT = ROOT / "results/clip_finetuning_20261015"
MODELS = ["mobileclip2-s2", "siglip2-b16-224"]


def save(value):
    path = OUT / "run.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-recorded-source-gaps", action="store_true")
    parser.add_argument("--resume-preparation", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()
    if args.resume_training:
        state = json.loads((OUT / "run.json").read_text())
        manifest = ROOT / "data/clip_training_20261015/samples.jsonl"
        expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for path in OUT.glob("seed_*/*/state.json"):
            recorded = json.loads(path.read_text())
            if recorded["manifest_sha256"] != expected:
                raise ValueError("Cannot resume after a dataset change")
        state["status"] = "training"
        save(state)
        env = dict(os.environ)
        env["TOKENIZERS_PARALLELISM"] = "false"
        execute_runs(state, env)
        return
    if args.resume_preparation:
        previous = json.loads((OUT / "run.json").read_text())
        if previous["runs"]:
            raise ValueError("Preparation resume cannot overwrite any model run")
    OUT.mkdir(parents=True, exist_ok=args.resume_preparation)
    (OUT / "logs").mkdir(exist_ok=args.resume_preparation)
    state = {
        "status": "waiting_for_source_transfers",
        "models": MODELS,
        "seeds": [42, 43, 44],
        "selection": "Macro task AP, with mean source AP within task, on development data only",
        "test_selection": False,
        "no_latency_testing": True,
        "runs": [],
    }
    save(state)
    deadline = time.monotonic() + 7200
    while True:
        required = []
        for split in ["train", "test"]:
            required.extend(
                Path(x["local_path"])
                for x in json.loads(
                    (SOURCES / "mivia_iwdd" / (split + "_files.json")).read_text()
                )
                if x["path"].endswith(".mp4")
            )
        street = json.loads((SOURCES / "street_scene/samples.json").read_text())
        required.extend(ROOT / p for row in street for p in row["views"])
        missing = {str(p) for p in required if not p.exists()}
        waived = set()
        gaps = SOURCES / "mivia_iwdd/unavailable_files_direct.json"
        if args.allow_recorded_source_gaps and gaps.exists():
            unavailable = {r["path"] for r in json.loads(gaps.read_text())}
            for split in ["train", "test"]:
                waived.update(
                    x["local_path"]
                    for x in json.loads(
                        (SOURCES / "mivia_iwdd" / (split + "_files.json")).read_text()
                    )
                    if x["path"] in unavailable
                )
        state["recorded_source_gaps"] = sorted(missing & waived)
        state["missing_source_files"] = len(missing)
        save(state)
        if not (missing - waived):
            break
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Source transfers remain incomplete; no partial training silently started"
            )
        time.sleep(20)  # Wait for preparation only; never pace training/inference.
    state["status"] = "preparing_manifest"
    save(state)
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        + ":/tmp/capstone_eval_deps"
        + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    with (OUT / "logs/prepare.log").open("w") as stream:
        command = [sys.executable, "scripts/prepare_clip_training.py"]
        if state["recorded_source_gaps"]:
            command.append("--allow-incomplete-downloads")
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            state["status"] = "preparation_failed"
            save(state)
            raise
    dataset = json.loads(
        (ROOT / "data/clip_training_20261015/dataset.json").read_text()
    )
    assert all(dataset["counts"][task] for task in dataset["tasks"])
    state["dataset"] = dataset
    state["status"] = "training"
    save(state)
    execute_runs(state, env)


def execute_runs(state, env):
    for seed in state["seeds"]:
        processes = []
        for model in MODELS:
            destination = OUT / f"seed_{seed}"
            existing = destination / model / "state.json"
            if existing.exists():
                recorded = json.loads(existing.read_text())
                if recorded["status"] == "complete":
                    print("Reusing completed", model, "seed", seed, flush=True)
                    continue
                raise ValueError(
                    f"Preserve/archive the incomplete model attempt before resuming: {existing.parent}"
                )
            command = [
                str(ROOT / "scripts/evaluate_gpu.sh"),
                "train-clip",
                "--model",
                model,
                "--output",
                str(destination),
                "--batch-size",
                "32",
                "--workers",
                "6",
                "--threads",
                "8",
                "--epochs",
                "10",
                "--seed",
                str(seed),
            ]
            if seed != 42:
                choice = json.loads(
                    (OUT / "seed_42" / model / "selection.json").read_text()
                )
                command += ["--learning-rates", str(choice["finetuned_lr"])]
            log = OUT / "logs" / f"{model}_seed_{seed}.log"
            stream = log.open("w")
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            run = {
                "model": model,
                "seed": seed,
                "command": command,
                "log": str(log.relative_to(ROOT)),
                "output": str((destination / model).relative_to(ROOT)),
                "pid": process.pid,
                "status": "running",
            }
            state["runs"].append(run)
            processes.append((process, stream, run))
            save(state)
            print("Started", model, "seed", seed, flush=True)
        failed = []
        for process, stream, run in processes:
            code = process.wait()
            stream.close()
            run["exit_code"] = code
            run["status"] = "ok" if code == 0 else "failed"
            save(state)
            if code:
                failed.append(run["model"])
        if failed:
            state["status"] = "incomplete"
            save(state)
            raise RuntimeError("Training failed: " + ", ".join(failed))
    state["status"] = "complete"
    save(state)
    print("Completed both models and all three seeds", flush=True)


if __name__ == "__main__":
    main()
