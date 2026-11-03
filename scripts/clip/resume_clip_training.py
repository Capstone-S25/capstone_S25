"""Resume seed replications independently, adopting already running model jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/clip_finetuning_20261015"
MODELS = ["mobileclip2-s2", "siglip2-b16-224"]


def read(path):
    return json.loads(path.read_text())


def save(state):
    temporary = OUT / "run.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(OUT / "run.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-controller", type=int)
    parser.add_argument("--max-concurrent", type=int, default=3)
    args = parser.parse_args()
    if args.max_concurrent < 1 or args.max_concurrent > 3:
        parser.error("This GPU study allows 1–3 concurrent model jobs")
    if args.replace_controller:
        process = Path(f"/proc/{args.replace_controller}")
        command = (process / "cmdline").read_bytes().split(b"\0")
        if not any(x.endswith(b"scripts/clip/run_clip_training.py") for x in command):
            raise ValueError("Replacement PID is not the study controller")
        os.kill(args.replace_controller, signal.SIGSTOP)
    state = read(OUT / "run.json")
    state["scheduler"] = "independent development-selected seed replications"
    state["replaced_controller_pid"] = args.replace_controller
    state["max_concurrent_model_jobs"] = args.max_concurrent
    state["backend_note"] = (
        "Initial seed-42 jobs retain their original code snapshots; later jobs disable convolution autotuning for variable frame-batch shapes. No source labels or selection criteria changed."
    )
    expected = hashlib.sha256(
        (ROOT / "data/clip_training_20261015/samples.jsonl").read_bytes()
    ).hexdigest()
    processes = {}
    streams = {}
    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    while True:
        complete = set()
        active = set()
        for run in state["runs"]:
            key = (run["model"], run["seed"])
            directory = ROOT / run["output"]
            path = directory / "state.json"
            model_state = read(path) if path.exists() else None
            if model_state and model_state["manifest_sha256"] != expected:
                raise ValueError("Training manifest changed")
            if model_state and model_state["status"] == "complete":
                run["status"] = "ok"
                run["exit_code"] = 0
                complete.add(key)
                if key in processes:
                    code = processes.pop(key).wait()
                    streams.pop(key).close()
                    if code:
                        raise RuntimeError(
                            f"Model exited unsuccessfully after writing results: {key}"
                        )
            elif run["status"] == "running":
                if key in processes and processes[key].poll() is not None:
                    run["status"] = "failed"
                    run["exit_code"] = processes[key].returncode
                    state["status"] = "incomplete"
                    save(state)
                    raise RuntimeError(f"Model failed: {key}; inspect its log")
                if not Path(f"/proc/{run['pid']}").exists():
                    raise RuntimeError(f"Adopted model process disappeared: {key}")
                active.add(key)
        if len(complete) == 6:
            state["status"] = "complete"
            save(state)
            if args.replace_controller:
                # The old stopped controller has no model work left to supervise.
                os.kill(args.replace_controller, signal.SIGKILL)
            print("Completed both models and all three seeds", flush=True)
            return
        for seed in [42, 43, 44]:
            for model in MODELS:
                key = (model, seed)
                if key in complete or key in active:
                    continue
                if seed != 42 and (model, 42) not in complete:
                    continue
                if len(active) >= args.max_concurrent:
                    break
                directory = OUT / f"seed_{seed}" / model
                if directory.exists():
                    raise ValueError(
                        f"Refusing to overwrite incomplete attempt: {directory}"
                    )
                choice = read(OUT / "seed_42" / model / "selection.json")
                command = [
                    str(ROOT / "scripts/evaluate_gpu.sh"),
                    "train-clip",
                    "--model",
                    model,
                    "--output",
                    str(directory.parent),
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
                    "--learning-rates",
                    str(choice["finetuned_lr"]),
                ]
                log = OUT / "logs" / f"{model}_seed_{seed}.log"
                stream = log.open("w")
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                processes[key] = process
                streams[key] = stream
                state["runs"].append(
                    {
                        "model": model,
                        "seed": seed,
                        "command": command,
                        "log": str(log.relative_to(ROOT)),
                        "output": str(directory.relative_to(ROOT)),
                        "pid": process.pid,
                        "status": "running",
                    }
                )
                active.add(key)
                print("Started", model, "seed", seed, flush=True)
        state["status"] = "training"
        save(state)
        time.sleep(5)


if __name__ == "__main__":
    main()
