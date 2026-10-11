"""One CPU-only entry point: fetch, prepare, validate, benchmark, calibrate, bootstrap and report.

Examples:
  anomaly-eval run --fetch --prepare-only
  anomaly-eval run --threads 4 --max-rss-mb 24576
  anomaly-eval run --skip-prepare --offline --vadclip

Every run gets a new directory; old predictions are never silently reused/deleted.
Incomplete task coverage is recorded explicitly, and --require-all-tasks enforces it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from anomaly_eval.analysis import analyze, calibrate, write_report
from anomaly_eval.data import (
    Sample,
    code_hashes,
    experiment_options,
    file_sha256,
    fingerprint,
    read_definitions,
    read_manifest,
    select_test_samples,
    selection_summary,
)
from anomaly_eval.datasets.fetch import download, fetch
from anomaly_eval.datasets.prepare import prepare
from anomaly_eval.models import DEFAULT_MODELS
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    resolve_arguments,
    resolve_path,
)
from anomaly_eval.resources import run_stage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--data-dir", type=Path, default=Workspace.current().data_dir)
    p.add_argument("--cache-dir", type=Path, default=Workspace.current().model_cache)
    p.add_argument(
        "--prepared-dir",
        type=Path,
        help="Writable prepared-data directory; default DATA_ROOT/evaluation",
    )
    p.add_argument("--review-csv", type=Path)
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch additional ABODA/CAVIAR plus D2 metadata and configured Roboflow exports",
    )
    p.add_argument(
        "--fetch-drive",
        action="store_true",
        help="Also fetch the original shared Drive corpus",
    )
    p.add_argument(
        "--use-cookies",
        action="store_true",
        help="Use the local gdown cookie cache for Drive downloads",
    )
    p.add_argument("--drive-workers", type=int, default=2)
    p.add_argument("--fetch-timeout", type=float, default=72000)
    p.add_argument("--d2-shards", nargs="*", default=[])
    p.add_argument("--roboflow-version", action="append", default=[])
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument(
        "--mode",
        choices=["evaluation", "quick", "deployment"],
        default="evaluation",
        help="Full unpaced evaluation, small unpaced pilot, or paced deployment simulation",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Inputs per task/protocol/split; 0 means all. Defaults: evaluation=0, quick=8, deployment=16",
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            k
            for k in read_definitions(Workspace.current().definitions)
            if k != "normal"
        ],
    )
    p.add_argument(
        "--protocols",
        nargs="+",
        choices=["frame", "window"],
        help="Defaults to frame/window for evaluation, frame only for quick/deployment",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu-cores", help="Optional CPU affinity list, e.g. 0,1,2,3")
    p.add_argument(
        "--duty-cycle",
        type=float,
        default=1.0,
        help="Active fraction in deployment mode; default 1.0",
    )
    p.add_argument("--max-rss-mb", type=float, default=24576)
    p.add_argument(
        "--camera-fps",
        type=float,
        default=1,
        help="Deployment frame budget; adds pacing only in deployment mode",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600,
        help="Per image model including load",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--image-policy", choices=["native", "letterbox"], default="letterbox"
    )
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--bootstrap",
        type=int,
        help="Group-bootstrap draws; evaluation defaults to 2000, quick/deployment to 200",
    )
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-all-tasks", action="store_true")
    p.add_argument(
        "--vadclip",
        action="store_true",
        help="Also extract/train/calibrate/test offline smoking adapter and feature baselines",
    )
    p.add_argument("--vadclip-epochs", type=int, default=100)
    p.add_argument("--vadclip-timeout", type=float, default=14400)
    a = resolve_arguments(p.parse_args(argv))
    if a.protocols is None:
        a.protocols = ["frame", "window"] if a.mode == "evaluation" else ["frame"]
    if a.limit is None:
        a.limit = {"evaluation": 0, "quick": 8, "deployment": 16}[a.mode]
    if a.bootstrap is None:
        a.bootstrap = 2000 if a.mode == "evaluation" else 200
    if a.limit < 0 or a.limit == 1:
        p.error("limit must be 0 (all) or at least 2 to retain event/normal support")
    if a.mode != "deployment" and a.duty_cycle != 1.0:
        p.error("--duty-cycle below 1 requires --mode deployment")
    if a.mode == "deployment" and a.vadclip:
        p.error(
            "VadCLIP is an offline study; use --vadclip in evaluation or quick mode"
        )
    a.data_dir = resolve_path(a.data_dir)
    a.cache_dir = resolve_path(a.cache_dir)
    if a.output:
        a.output = resolve_path(a.output)
    if a.review_csv:
        a.review_csv = resolve_path(a.review_csv)
    os.environ["CAPSTONE_DATA_DIR"] = str(a.data_dir)
    if (
        min(
            a.threads,
            a.repeats,
            a.max_side,
            a.max_new_tokens,
            a.bootstrap,
            a.vadclip_epochs,
        )
        < 1
        or min(a.camera_fps, a.max_rss_mb, a.timeout_seconds, a.vadclip_timeout) <= 0
        or not 0 < a.duty_cycle <= 1
        or not 0 <= a.target_fpr < 1
    ):
        p.error("Invalid resource, sampling or statistical settings")
    if (
        any(
            not math.isfinite(v)
            for v in (
                a.camera_fps,
                a.max_rss_mb,
                a.timeout_seconds,
                a.vadclip_timeout,
                a.duty_cycle,
                a.target_fpr,
                a.fetch_timeout,
            )
        )
        or a.fetch_timeout <= 0
        or a.drive_workers < 1
    ):
        p.error("Resource settings must be finite and positive")
    from anomaly_eval.resources import CPUSettings

    CPUSettings(a.threads, a.duty_cycle).configure_environment()
    from anomaly_eval.models import MODELS

    if not set(a.models) <= set(MODELS):
        p.error("Unknown model names")
    if not set(a.tasks) <= (
        set(read_definitions(Workspace.current().definitions)) - {"normal"}
    ):
        p.error("Unknown tasks")
    output = (
        a.output
        or Workspace.current().results_dir
        / ("evaluation_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "benchmarks": [],
        "steps": [],
        "options": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        },
        "code_sha256": code_hashes(),
        "workspace": Workspace.current().as_dict(),
    }

    def save():
        (output / "evaluation.json").write_text(
            json.dumps(state, indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        if a.fetch_drive:
            if a.offline:
                raise ValueError("--fetch-drive and --offline cannot be combined")
            command = [
                sys.executable,
                "-m",
                "anomaly_eval",
                "fetch-drive",
                "--output",
                str(a.data_dir / "raw"),
                "--extract-zip",
                "--continue-on-error",
                "--anonymous-listing",
                "--workers",
                str(a.drive_workers),
                "--request-interval",
                "0.5",
                "--name-conflicts",
                "rename",
            ]
            if a.use_cookies:
                command.append("--use-cookies")
            state["steps"].append(
                run_stage(
                    command,
                    output / "logs" / "fetch_drive.log",
                    a.fetch_timeout,
                    a.max_rss_mb,
                )
            )
            save()
        if a.fetch:
            if a.offline:
                raise ValueError("--fetch and --offline cannot be combined")
            versions = {
                k: int(v) for k, v in (x.split("=") for x in a.roboflow_version)
            }
            from anomaly_eval.datasets.fetch import SOURCES

            state["fetch"] = fetch(
                a.data_dir / "additional", SOURCES, a.d2_shards, versions
            )
            save()
        evaluation = (
            resolve_path(a.prepared_dir)
            if a.prepared_dir
            else a.data_dir / "evaluation"
        )
        if a.skip_prepare:
            dataset = json.loads((evaluation / "dataset.json").read_text())
        else:
            print("Preparing reviewed task cohorts...", flush=True)
            dataset = prepare(
                a.data_dir / "raw" / "datasets",
                a.data_dir / "additional",
                evaluation,
                a.review_csv,
                seed=a.seed,
            )
        state["dataset"] = dataset
        state["dataset_fingerprint"] = dataset["fingerprint"]
        state["requested_task_coverage"] = {
            task: {
                "available_protocols": [
                    protocol
                    for protocol in a.protocols
                    if protocol in dataset["coverage"].get(task, {})
                ],
                "test_eligible_protocols": [
                    protocol
                    for protocol in a.protocols
                    if dataset["coverage"]
                    .get(task, {})
                    .get(protocol, {})
                    .get("eligible_for_calibrated_test", False)
                ],
            }
            for task in a.tasks
        }
        state["full_task_coverage"] = all(
            c["test_eligible_protocols"]
            for c in state["requested_task_coverage"].values()
        )
        for task, coverage in state["requested_task_coverage"].items():
            status = (
                "dev/test data available"
                if coverage["test_eligible_protocols"]
                else "development only"
                if coverage["available_protocols"]
                else "no confirmed data for requested protocols"
            )
            print(f"{task}: {status}", flush=True)
        save()
        if a.require_all_tasks and not state["full_task_coverage"]:
            state["status"] = "incomplete_data"
            save()
            write_report(output, state, a)
            print(
                "Requested tasks lack verified dev/test coverage; inference was not started."
            )
            print(f"Report: {output / 'report.md'}")
            return 2
        definitions = read_definitions(Workspace.current().definitions)
        for task in a.tasks:
            for protocol in a.protocols:
                cohort = task + "_" + protocol
                if cohort not in dataset["manifests"]:
                    continue
                manifest = resolve_path(dataset["manifests"][cohort], evaluation)
                taskdefs = {
                    "normal": f"An ordinary scene with no evidence of {task.replace('_', ' ')}.",
                    task: definitions[task],
                }
                samples = read_manifest(manifest, taskdefs)
                cfg = output / (cohort + "_definitions.json")
                cfg.write_text(json.dumps(taskdefs, indent=2) + "\n")
                state.setdefault("manifest_sha256", {})[cohort] = file_sha256(manifest)
                if a.prepare_only:
                    continue
                directories = {}
                for split in ("dev", "test"):
                    if not any(s.split == split for s in samples):
                        continue
                    # Do not score a single-class held-out set as if task evaluation exists.
                    if (
                        split == "test"
                        and not dataset["coverage"][task][protocol][
                            "eligible_for_calibrated_test"
                        ]
                    ):
                        continue
                    runout = output / cohort / split
                    available = [
                        Sample(s.path, s.label, "test", s.group_id)
                        for s in samples
                        if s.split == split
                    ]
                    selected = select_test_samples(available, a.limit or None, a.seed)
                    selection = selection_summary(available, selected)
                    command = [
                        sys.executable,
                        "-m",
                        "anomaly_eval",
                        "benchmark",
                        "--manifest",
                        str(manifest),
                        "--definitions",
                        str(cfg),
                        "--models",
                        *a.models,
                        "--split",
                        split,
                        "--output",
                        str(runout),
                        "--limit",
                        str(a.limit),
                        "--threads",
                        str(a.threads),
                        "--duty-cycle",
                        str(a.duty_cycle),
                        "--max-rss-mb",
                        str(a.max_rss_mb),
                        "--timeout-seconds",
                        str(a.timeout_seconds),
                        "--repeats",
                        str(a.repeats),
                        "--seed",
                        str(a.seed),
                        "--cache-dir",
                        str(a.cache_dir),
                        "--image-policy",
                        a.image_policy,
                        "--max-side",
                        str(a.max_side),
                        "--max-new-tokens",
                        str(a.max_new_tokens),
                    ]
                    if a.cpu_cores:
                        command += ["--cpu-cores", a.cpu_cores]
                    if a.mode == "deployment":
                        command += ["--max-samples-per-second", str(a.camera_fps)]
                    if a.offline:
                        command.append("--offline")
                    print(
                        f"Running {cohort}/{split}: {len(selected)}/{len(available)} inputs, mode={a.mode}, "
                        + (
                            f"paced at {a.camera_fps:g}/s..."
                            if a.mode == "deployment"
                            else "no pacing sleeps..."
                        ),
                        flush=True,
                    )
                    step = run_stage(
                        command,
                        output / "logs" / (cohort + "_" + split + ".log"),
                        a.timeout_seconds * len(a.models) + 60,
                        a.max_rss_mb + 1024,
                    )
                    state["steps"].append(step)
                    state["benchmarks"].append(
                        {
                            "cohort": cohort,
                            "split": split,
                            "output": portable_path(runout, output),
                            "selection": selection,
                        }
                    )
                    directories[split] = runout
                    save()
                    try:
                        analyze(runout, a.bootstrap, a.seed)
                        from anomaly_eval.analysis import generate

                        generate(runout, runout / "figures")
                    except (ValueError, OSError) as exc:
                        state.setdefault("analysis_errors", []).append(str(exc))
                        save()
                if set(directories) == {"dev", "test"}:
                    try:
                        calibrate(directories["dev"], directories["test"], a.target_fpr)
                    except (ValueError, OSError) as exc:
                        state.setdefault("analysis_errors", []).append(str(exc))
                        save()
        if a.vadclip and not a.prepare_only:
            if not (a.data_dir / "raw" / "datasets" / "smoking").is_dir():
                state["vadclip"] = {
                    "status": "missing_data",
                    "note": "Fetch original smoking COCO corpus; bundled 32 development frames are not a training corpus.",
                }
            else:
                weights = (
                    Workspace.current().cache_dir / "vadclip" / "openai_clip_vitb16.pt"
                )
                sha = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
                if not weights.is_file():
                    if a.offline:
                        raise ValueError(
                            "Offline VadCLIP needs official cached ViT-B/16 weights"
                        )
                    download(
                        f"https://openaipublic.azureedge.net/clip/models/{sha}/ViT-B-16.pt",
                        weights,
                        expected_sha=sha,
                    )
                vdir = output / "vadclip"
                cache = Workspace.current().cache_dir / "vadclip" / "features"
                run = vdir / "adapter"
                stages = [
                    (
                        "extract_features",
                        [
                            "--root",
                            str(a.data_dir / "raw" / "datasets"),
                            "--cache-dir",
                            str(cache),
                            "--weights",
                            str(weights),
                            "--seed",
                            str(a.seed),
                            "--batch-size",
                            "8",
                        ],
                    ),
                    (
                        "train",
                        [
                            "--cache-dir",
                            str(cache),
                            "--weights",
                            str(weights),
                            "--out-dir",
                            str(run),
                            "--epochs",
                            str(a.vadclip_epochs),
                        ],
                    ),
                    (
                        "evaluate",
                        [
                            "--cache-dir",
                            str(cache),
                            "--weights",
                            str(weights),
                            "--run-dir",
                            str(run),
                        ],
                    ),
                    (
                        "baselines",
                        [
                            "--cache-dir",
                            str(cache),
                            "--weights",
                            str(weights),
                            "--out-dir",
                            str(vdir / "baselines"),
                        ],
                    ),
                ]
                state["vadclip"] = {"status": "running"}
                for name, opts in stages:
                    if a.offline and name == "extract_features":
                        opts.append("--offline")
                    command = [
                        sys.executable,
                        "-m",
                        "anomaly_eval",
                        "vadclip",
                        "extract" if name == "extract_features" else name,
                        *opts,
                        "--threads",
                        str(a.threads),
                    ]
                    step = run_stage(
                        command,
                        output / "logs" / ("vadclip_" + name + ".log"),
                        a.vadclip_timeout,
                        a.max_rss_mb,
                    )
                    state["steps"].append(step)
                    save()
                    if step["status"] != "ok":
                        state["vadclip"]["status"] = "error"
                        break
                else:
                    state["vadclip"]["status"] = "ok"
        failed = (
            any(s["status"] != "ok" for s in state["steps"])
            or bool(state.get("analysis_errors"))
            or state.get("vadclip", {}).get("status") in {"error", "missing_data"}
        )
        state["status"] = (
            "error"
            if failed
            else "prepared"
            if a.prepare_only
            else "complete_for_available_tasks"
        )
        state["experiment_fingerprint"] = fingerprint(
            {
                "options": experiment_options(state["options"]),
                "code": state["code_sha256"],
                "dataset": dataset["fingerprint"],
            }
        )
        save()
        write_report(output, state, a)
        save()
        print(
            f"Report: {output / 'report.md'}\nFull requested task coverage: {state['full_task_coverage']}"
        )
        return (
            1
            if failed
            else 2
            if a.require_all_tasks and not state["full_task_coverage"]
            else 0
        )
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        save()
        return 130
    except Exception as exc:
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
