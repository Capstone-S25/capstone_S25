"""Benchmark anomaly models in isolated workers with explicit CPU/GPU placement."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from anomaly_eval.data import (
    Sample,
    code_hashes,
    experiment_options,
    file_sha256,
    fingerprint,
    read_definitions,
    read_manifest,
    sample_identity,
    select_test_samples,
    selection_summary,
)
from anomaly_eval.metrics import detection_metrics, percentile
from anomaly_eval.models import DEFAULT_MODELS, MODELS, create_adapter, load_image
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    process_environment,
    resolve_arguments,
    resolve_path,
)
from anomaly_eval.resources import (
    ResourceMonitor,
    Throttler,
    add_cpu_arguments,
    settings_from_args,
)
from anomaly_eval.runtime import (
    add_device_argument,
    available_threads,
    resolve_device,
    synchronize,
)


def cpu_description() -> str:
    description = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                return value.strip()
    return description or platform.machine()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def predict_rows(adapter, tests, args, repeat, batch_size, throttle, monitor):
    """Batch CLIP accuracy work; record batch time and explicit amortized timing."""
    for offset in range(0, len(tests), batch_size):
        samples = tests[offset : offset + batch_size]
        monitor.check()
        rows = [
            {
                "path": portable_path(s.path),
                "group_id": s.group_id,
                "label": s.label,
                "origin": s.origin,
                "pixel_sha256": s.pixel_sha256,
                "protocol": s.protocol,
                "repeat": repeat,
                "prediction": None,
                "anomaly_score": None,
                "scores": {},
                "status": "error",
            }
            for s in samples
        ]
        with throttle.operation():
            synchronize(adapter.device)
            started = time.perf_counter()
            try:
                images = [
                    load_image(s.path, args.max_side, adapter.image_policy)
                    for s in samples
                ]
                predictions = (
                    adapter.predict_batch(images)
                    if batch_size > 1
                    else [adapter.predict(images[0])]
                )
                if len(predictions) != len(samples):
                    raise ValueError(
                        "Adapter returned the wrong number of batch predictions"
                    )
                for row, prediction in zip(rows, predictions):
                    row.update(
                        prediction=prediction.label,
                        anomaly_score=prediction.anomaly_score,
                        scores=prediction.scores,
                        raw_response=prediction.raw_response,
                        generated_tokens=prediction.generated_tokens,
                        response_format=prediction.response_format,
                        abstention_reason=prediction.abstention_reason,
                        status="ok" if prediction.label else "abstain",
                    )
            except (MemoryError, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001 -- preserve failures for every planned input
                for row in rows:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            synchronize(adapter.device)
            seconds = time.perf_counter() - started
        for row in rows:
            row.update(
                latency_seconds=seconds / len(samples),
                batch_seconds=seconds,
                batch_size=len(samples),
                batch_index=offset // batch_size,
                latency_kind="batch_amortized" if batch_size > 1 else "single_input",
            )
            yield row


def worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**config["options"])
    output = resolve_path(config["output"], config_path.parent)
    name = config["model"]
    settings = settings_from_args(args)
    settings.configure_environment()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.cache_dir:
        args.cache_dir = str(resolve_path(args.cache_dir))
    definitions = config["definitions"]
    samples = [
        Sample(
            resolve_path(row["path"], config_path.parent),
            row["label"],
            row["split"],
            row["group_id"],
            row.get("origin"),
            row.get("pixel_sha256"),
            row.get("protocol", "frame"),
        )
        for row in config["samples"]
    ]
    references = [sample for sample in samples if sample.split == "reference"]
    tests = [sample for sample in samples if sample.split == "test"]
    summary = {
        "model": name,
        "checkpoint": MODELS[name].checkpoint,
        "model_revision": MODELS[name].revision,
        "status": "error",
    }
    rows = []
    monitor = None
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        # Reject missing reference classes before downloading any vision weights.
        if MODELS[name].kind in {"vision", "timm"}:
            missing = set(definitions) - {sample.label for sample in references}
            if missing:
                raise ValueError(
                    "Vision baselines need labeled reference images for every class; missing "
                    + ", ".join(sorted(missing))
                )
        with ResourceMonitor(args.max_rss_mb) as monitor:
            settings.configure_worker()
            import torch

            torch.manual_seed(args.seed)
            if getattr(args, "device", "cpu").startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
            throttle = Throttler(settings)
            load_start = time.perf_counter()
            adapter = create_adapter(
                name,
                definitions,
                args.cache_dir,
                args.offline,
                args.max_new_tokens,
                args.max_side,
                device=getattr(args, "device", "cpu"),
                dtype=getattr(args, "dtype", "float32"),
            )
            adapter.image_policy = getattr(args, "image_policy", "native")
            adapter.device = getattr(adapter, "device", "cpu")
            requested_batch = getattr(args, "batch_size", 1)
            batch_size = requested_batch or (
                32 if adapter.device.startswith("cuda") else 1
            )
            if not getattr(adapter, "supports_batch", False):
                batch_size = 1
            summary["batch_size"] = batch_size
            summary["latency_kind"] = (
                "batch_amortized" if batch_size > 1 else "single_input"
            )
            summary["resolved_revision"] = getattr(
                getattr(getattr(adapter, "model", None), "config", None),
                "_commit_hash",
                None,
            )
            summary["resolved_revision"] = summary["resolved_revision"] or getattr(
                adapter, "resolved_revision", MODELS[name].revision
            )
            summary["input_policy"] = {
                "max_side": args.max_side,
                "image_policy": getattr(args, "image_policy", "native"),
                "native_processor": str(
                    getattr(adapter, "preprocess", getattr(adapter, "processor", ""))
                ),
                "max_new_tokens": args.max_new_tokens,
                "dtype": str(getattr(adapter, "dtype", torch.float32)).removeprefix(
                    "torch."
                ),
                "device": adapter.device,
                "batch_size": batch_size,
            }
            if adapter.device.startswith("cuda"):
                summary["gpu_name"] = torch.cuda.get_device_name(adapter.device)
            synchronize(adapter.device)
            summary["load_seconds"] = time.perf_counter() - load_start
            monitor.check()
            preparation_start = time.perf_counter()
            adapter.prepare(references, throttle, monitor.check, args.max_side)
            summary["reference_seconds"] = time.perf_counter() - preparation_start
            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                monitor.check()
                with throttle.operation():
                    adapter.predict(
                        load_image(tests[0].path, args.max_side, adapter.image_policy)
                    )
            synchronize(adapter.device)
            summary["warmup_seconds"] = time.perf_counter() - warmup_start
            sleep_before = throttle.total_sleep_seconds
            measured_start = time.perf_counter()
            measured_cpu_start = time.process_time()
            with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for repeat in range(args.repeats):
                    for row in predict_rows(
                        adapter, tests, args, repeat, batch_size, throttle, monitor
                    ):
                        rows.append(row)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
            measured_seconds = time.perf_counter() - measured_start
            measured_cpu = time.process_time() - measured_cpu_start
            latencies = [
                row["latency_seconds"] for row in rows if row["status"] != "error"
            ]
            completed = len(latencies)
            if adapter.device.startswith("cuda"):
                summary["peak_gpu_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(adapter.device) / 1024**2
                )
                summary["peak_gpu_reserved_mb"] = (
                    torch.cuda.max_memory_reserved(adapter.device) / 1024**2
                )
            responses = [row for row in rows if row.get("raw_response") is not None]
            formats = Counter(
                row.get("response_format") or "unrecorded" for row in responses
            )
            summary.update(
                status="ok"
                if all(row["status"] != "error" for row in rows)
                else "partial",
                samples=len(tests),
                repeats=args.repeats,
                attempts=len(rows),
                completed_inferences=completed,
                errors=sum(row["status"] == "error" for row in rows),
                abstentions=sum(row["status"] == "abstain" for row in rows),
                latency_mean_seconds=statistics.mean(latencies) if latencies else None,
                latency_p50_seconds=percentile(latencies, 0.5),
                latency_p95_seconds=percentile(latencies, 0.95),
                batch_latency_p95_seconds=percentile(
                    list(
                        {
                            (r["repeat"], r["batch_index"]): r["batch_seconds"]
                            for r in rows
                            if r["status"] != "error"
                        }.values()
                    ),
                    0.95,
                ),
                active_samples_per_second=completed / sum(latencies)
                if sum(latencies)
                else None,
                wall_samples_per_second=completed / measured_seconds
                if measured_seconds
                else None,
                measured_wall_seconds=measured_seconds,
                measured_cpu_seconds=measured_cpu,
                throttle_sleep_seconds=throttle.total_sleep_seconds - sleep_before,
                average_cpu_cores=measured_cpu / measured_seconds
                if measured_seconds
                else None,
                peak_rss_mb=monitor.peak_rss_mb,
                metrics=detection_metrics(rows, list(definitions)),
                prediction_counts=dict(
                    Counter(row["prediction"] or "__unanswered__" for row in rows)
                ),
                response_diagnostics={
                    "returned_responses": len(responses),
                    "formats": dict(formats),
                    "abstention_reasons": dict(
                        Counter(
                            row["abstention_reason"]
                            for row in responses
                            if row.get("abstention_reason")
                        )
                    ),
                    "json_label_fraction": formats["json_label"] / len(responses)
                    if responses
                    else None,
                },
                naive_baselines={
                    name: detection_metrics(
                        [
                            {
                                **r,
                                "prediction": label,
                                "status": "ok",
                                "anomaly_score": None,
                            }
                            for r in rows
                        ],
                        list(definitions),
                    )
                    for name, label in (
                        ("always_normal", "normal"),
                        (
                            "always_anomaly",
                            next(k for k in definitions if k != "normal"),
                        ),
                    )
                },
                package_versions={
                    package: importlib.metadata.version(package)
                    for package in (
                        "torch",
                        "transformers",
                        "timm",
                        "open_clip_torch",
                        "Pillow",
                        "psutil",
                    )
                },
            )
    except Exception as exc:  # noqa: BLE001 -- isolate model failures and preserve diagnostics
        summary["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        summary["total_wall_seconds"] = time.perf_counter() - started
        summary["total_cpu_seconds"] = time.process_time() - cpu_started
        if monitor:
            summary["peak_rss_mb"] = monitor.peak_rss_mb
        write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


def stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_model(
    config_path: Path, timeout: float, max_rss_mb: float | None, cancelled=None
) -> dict:
    import psutil

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve_path(config["output"], config_path.parent)
    reason = None
    peak_rss_mb = 0.0
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "anomaly_eval",
                "benchmark",
                "--worker-config",
                str(config_path),
            ],
            env=process_environment(),
            stdout=log,
            stderr=log,
        )
        started = time.perf_counter()
        watched = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    reason = "Benchmark interrupted"
                    break
                try:
                    peak_rss_mb = max(peak_rss_mb, watched.memory_info().rss / 1024**2)
                except psutil.NoSuchProcess:
                    pass
                if max_rss_mb is not None and peak_rss_mb > max_rss_mb:
                    reason = f"Worker exceeded {max_rss_mb:g} MiB RSS"
                    break
                if time.perf_counter() - started > timeout:
                    reason = f"Worker exceeded {timeout:g} second model timeout (including loading and pacing)"
                    break
                time.sleep(0.05)
        finally:
            stop_worker(process)
    summary_path = output / "summary.json"
    if reason or not summary_path.is_file():
        summary = {
            "model": config["model"],
            "checkpoint": MODELS[config["model"]].checkpoint,
            "status": "error",
            "error": reason or f"Worker exited {process.returncode} without a summary",
            "total_wall_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb,
        }
        write_json(summary_path, summary)
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary


def write_comparison(output: Path, summaries: list[dict]) -> None:
    columns = [
        "model",
        "status",
        "samples",
        "latency_mean_seconds",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "active_samples_per_second",
        "wall_samples_per_second",
        "throttle_sleep_seconds",
        "json_label_fraction",
        "explicit_unknowns",
        "invalid_responses",
        "peak_rss_mb",
        "load_seconds",
        "errors",
        "abstentions",
        "prediction_coverage",
        "accuracy_including_failures",
        "macro_f1",
        "binary_f1",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "false_positive_rate",
        "specificity",
        "device",
        "dtype",
        "peak_gpu_allocated_mb",
        "batch_size",
        "latency_kind",
        "batch_latency_p95_seconds",
        "error",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = {column: summary.get(column) for column in columns}
            diagnostics = summary.get("response_diagnostics", {})
            row["json_label_fraction"] = diagnostics.get("json_label_fraction")
            row["explicit_unknowns"] = diagnostics.get("abstention_reasons", {}).get(
                "explicit_unknown", 0
            )
            row["invalid_responses"] = diagnostics.get("formats", {}).get("invalid", 0)
            metrics = summary.get("metrics", {})
            for column in (
                "prediction_coverage",
                "accuracy_including_failures",
                "macro_f1",
            ):
                row[column] = metrics.get(column)
            row["binary_f1"] = (metrics.get("binary") or {}).get("f1")
            row["roc_auc"] = (metrics.get("binary") or {}).get("roc_auc")
            for column in (
                "average_precision",
                "precision",
                "recall",
                "false_positive_rate",
                "specificity",
            ):
                row[column] = (metrics.get("binary") or {}).get(column)
            for column in ("device", "dtype"):
                row[column] = summary.get("input_policy", {}).get(column)
            writer.writerow(row)
    write_json(output / "results.json", {"models": summaries})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV of test/reference image paths and reviewed labels",
    )
    parser.add_argument(
        "--definitions", type=Path, default=Workspace.current().definitions
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names, or all",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="New results directory; defaults to results/<UTC timestamp>",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum inputs from the selected split; retain labels and source groups (0 means all)",
    )
    parser.add_argument(
        "--image-policy",
        choices=["native", "letterbox"],
        default="native",
        help="Pad to square to preserve full field of view through native center crops",
    )
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup inferences excluded from timings"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-side", type=int, default=512, help="Maximum input image edge"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Generative VLM token budget"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900,
        help="Timeout per model, including load and sleeps",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        help="Optional sampled per-worker memory limit in MiB",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Workspace.current().model_cache,
        help="Hugging Face weight cache",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached models; no network access",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent model processes for accuracy; use 1 for latency comparisons",
    )
    add_cpu_arguments(parser)
    parser.set_defaults(threads=available_threads())
    add_device_argument(parser)
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="CLIP batch size; 0 selects GPU=32/CPU=1; VLMs use 1",
    )
    args = resolve_arguments(parser.parse_args(argv))
    if args.worker_config:
        return worker(args.worker_config)
    if args.list_models:
        for name, spec in MODELS.items():
            print(f"{name:22} {spec.kind:6} {spec.checkpoint}")
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless using --list-models")
    names = list(MODELS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    unknown = set(names) - MODELS.keys()
    if unknown:
        parser.error("Unknown models: " + ", ".join(sorted(unknown)))
    if (
        min(args.repeats, args.max_side, args.max_new_tokens, args.workers) < 1
        or min(args.warmup, args.limit, args.batch_size) < 0
    ):
        parser.error(
            "repeats, max-side, and max-new-tokens must be positive; warmup and limit nonnegative"
        )
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive and finite")
    if args.max_rss_mb is not None and (
        not math.isfinite(args.max_rss_mb) or args.max_rss_mb <= 0
    ):
        parser.error("max-rss-mb must be positive and finite")
    try:
        import psutil

        settings = settings_from_args(args)
        settings.configure_environment()
        args.device = resolve_device(args.device)
        definitions = read_definitions(args.definitions)
        samples = read_manifest(args.manifest.resolve(), definitions)
        split_samples = [
            replace(s, split="test") for s in samples if s.split == args.split
        ]
        tests = select_test_samples(split_samples, args.limit or None, args.seed)
        if not tests:
            raise ValueError(f"No {args.split} samples available")
        references = [sample for sample in samples if sample.split == "reference"]
        output = (
            args.output
            or Workspace.current().results_dir
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
        options = {
            key: portable_path(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        serialized_samples = [
            {**asdict(sample), "path": portable_path(sample.path)}
            for sample in references + tests
        ]
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_model": cpu_description(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "logical_cpus": os.cpu_count(),
            "total_ram_mb": psutil.virtual_memory().total / 1024**2,
            "available_cpu_cores": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "options": options,
            "workspace": Workspace.current().as_dict(),
            "definitions": definitions,
            "manifest_sha256": file_sha256(args.manifest),
            "definitions_sha256": file_sha256(args.definitions),
            "samples": serialized_samples,
            "selection": selection_summary(split_samples, tests),
            "models": names,
            "evaluation_split": args.split,
            "sample_identity": sample_identity(references + tests),
            "code_sha256": code_hashes(),
            "metric_version": 2,
        }
        metadata["experiment_fingerprint"] = fingerprint(
            {
                "options": experiment_options(options),
                **{
                    key: metadata[key]
                    for key in (
                        "definitions",
                        "sample_identity",
                        "code_sha256",
                        "models",
                    )
                },
            }
        )
        write_json(output / "run.json", metadata)
        summaries = []
        configurations = []
        for name in names:
            model_output = output / name
            model_output.mkdir()
            config_path = model_output / "worker_config.json"
            write_json(
                config_path,
                {
                    "model": name,
                    "output": portable_path(model_output),
                    "options": options,
                    "definitions": definitions,
                    "samples": serialized_samples,
                },
            )
            print(
                f"Running {name} on {len(tests)} test images ({settings.threads} CPU threads)...",
                flush=True,
            )
            configurations.append(config_path)
        cancelled = threading.Event()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_model, config, args.timeout_seconds, args.max_rss_mb, cancelled
                )
                for config in configurations
            ]
            try:
                for future in as_completed(futures):
                    summary = future.result()
                    summaries.append(summary)
                    summaries.sort(key=lambda item: names.index(item["model"]))
                    # Completed results survive interruption in requested model order.
                    write_comparison(output, summaries)
                    print(
                        f"  {summary['model']} {summary['status']}: {summary.get('error', str(summary.get('completed_inferences', 0)) + ' completed inferences')}",
                        flush=True,
                    )
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        print(f"Results: {output / 'comparison.csv'}")
        return 0 if all(summary["status"] == "ok" for summary in summaries) else 1
    except (ValueError, OSError, ImportError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; completed model results and streamed predictions are preserved.",
            file=sys.stderr,
        )
        return 130
