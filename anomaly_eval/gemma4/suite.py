"""Resume the complete Gemma study, sequentially owning this run's GPU jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from anomaly_eval.datasets.transfer import atomic_json
from anomaly_eval.gemma4.protocol import DEFAULT_OUTPUT, MODELS, ROOT


def update_report(output):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "anomaly_eval.gemma4",
            "report",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--wait-pid",
        type=int,
        nargs="+",
        help="Wait for an already running evaluation owned by this study",
    )
    parser.add_argument("--download-timeout-hours", type=float, default=6)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = args.output / "suite_status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        state["phase"] = (
            "waiting_for_owned_evaluations" if args.wait_pid else "model_evaluations"
        )
        atomic_json(status_path, state)
        if args.wait_pid:
            state.pop("waiting_for_existing_pid", None)
            state["waiting_for_existing_pids"] = args.wait_pid
            atomic_json(status_path, state)
            published = set()
            while any(Path(f"/proc/{pid}").exists() for pid in args.wait_pid):
                completed = {
                    name
                    for name in MODELS
                    if (args.output / name / "results.json").exists()
                }
                if completed != published:
                    report_code = update_report(args.output)
                    for name in completed:
                        state[name] = {
                            "status": "complete",
                            "result": str(args.output / name / "results.json"),
                        }
                    state["latest_report_exit_code"] = report_code
                    atomic_json(status_path, state)
                    published = completed
                time.sleep(5)
            state.pop("waiting_for_existing_pids", None)
            state["phase"] = "model_evaluations"
        pending = list(args.models)
        deadline = time.monotonic() + args.download_timeout_hours * 3600
        while pending:
            name = next(
                (n for n in pending if (args.output / n / "checkpoint.json").exists()),
                None,
            )
            if name is None:
                if time.monotonic() > deadline:
                    for item in pending:
                        state[item] = {
                            "status": "checkpoint_unavailable",
                            "result": None,
                        }
                    atomic_json(status_path, state)
                    return 1
                time.sleep(5)
                continue
            pending.remove(name)
            directory = args.output / name
            if (directory / "results.json").exists():
                state[name] = {
                    "status": "complete",
                    "result": str(directory / "results.json"),
                }
                atomic_json(status_path, state)
                continue
            checkpoint = json.loads((directory / "checkpoint.json").read_text())["path"]
            # Preserve a prior run's argument spelling so relative/absolute paths
            # to the same pinned checkpoint do not invalidate its strict cache.
            if (directory / "runtime.json").exists():
                previous = json.loads((directory / "runtime.json").read_text())
                checkpoint = previous.get("checkpoint") or checkpoint
            preflight = [
                str(ROOT / ".venv-gemma4/bin/python"),
                "-m",
                "anomaly_eval.gemma4",
                "preflight",
                "--model",
                name,
                "--output",
                str(args.output),
            ]
            command = [
                "bash",
                str(ROOT / "scripts/gemma4_gpu.sh"),
                "evaluate",
                "--model",
                name,
                "--checkpoint",
                checkpoint,
                "--output",
                str(args.output),
                "--batch-size",
                "8",
            ]
            state[name] = {
                "status": "running",
                "command": command,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_json(status_path, state)
            with (directory / "run.log").open("a") as log:
                preflight_code = subprocess.run(
                    preflight,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    check=False,
                ).returncode
                if preflight_code:
                    state[name].update(
                        status="preflight_failed", exit_code=preflight_code
                    )
                    atomic_json(status_path, state)
                    continue
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
                )
                state[name]["pid"] = child.pid
                atomic_json(status_path, state)
                code = child.wait()
            state[name].update(
                status="complete"
                if code == 0 and (directory / "results.json").exists()
                else "failed",
                exit_code=code,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                result=str(directory / "results.json")
                if (directory / "results.json").exists()
                else None,
            )
            atomic_json(status_path, state)
            state["latest_report_exit_code"] = update_report(args.output)
            atomic_json(status_path, state)
        state["latest_report_exit_code"] = update_report(args.output)
        failed = any(
            v.get("status") != "complete" for k, v in state.items() if k in args.models
        ) or state["latest_report_exit_code"] not in {None, 0}
        state["phase"] = "finished_with_failures" if failed else "complete"
        state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(status_path, state)
        return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
