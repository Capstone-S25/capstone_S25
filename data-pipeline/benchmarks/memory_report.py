"""Build RAM-focused JSON, CSV and Markdown reports from measured suite results."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

SINGAPORE = timezone(timedelta(hours=8))
LABELS = {
    "baseline": "Idle baseline",
    "recording": "Recording only",
    "streaming": "Streaming incl. decoding",
    "full": "Recording + streaming",
}


def _aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        if "_soak" not in run["case"] and not run["consumer_delay_seconds"]:
            groups[(run["mode"], run["cameras"])].append(run)
    rows = []
    for (mode, cameras), repeats in groups.items():
        rows.append(
            {
                "mode": mode,
                "scenario": LABELS[mode],
                "cameras": cameras,
                "repeats": len(repeats),
                "pipeline_mean_mib": median(
                    r["metrics"]["app_rss_mean_mib"] for r in repeats
                ),
                "pipeline_peak_mib": max(
                    r["metrics"]["app_rss_peak_mib"] for r in repeats
                ),
                "python_mean_mib": median(
                    r["metrics"]["python_rss_mean_mib"] for r in repeats
                ),
                "ffmpeg_mean_mib": median(
                    r["metrics"]["ffmpeg_rss_mean_mib"] for r in repeats
                ),
                "host_used_mean_gib": median(
                    r["host"]["host_used_mean_mib"] / 1024 for r in repeats
                ),
                "host_used_peak_gib": max(
                    r["host"]["host_used_peak_mib"] / 1024 for r in repeats
                ),
                "host_available_min_gib": min(
                    r["host"]["host_available_min_mib"] / 1024 for r in repeats
                ),
                "wsl_mean_mib": median(
                    r["host"]["wsl_vm_rss_mean_mib"] for r in repeats
                ),
                "docker_desktop_mean_mib": median(
                    r["host"]["docker_desktop_rss_mean_mib"] for r in repeats
                ),
                "mock_publisher_mean_mib": median(
                    r["host"]["mock_publisher_rss_mean_mib"] for r in repeats
                ),
                "peak_queued_frames": max(
                    r["metrics"]["queued_frames_peak"] for r in repeats
                ),
                "peak_queued_image_mib": max(
                    r["metrics"]["queued_image_bytes_peak_mib"] for r in repeats
                ),
                "produced_frames_total": sum(
                    c["produced_frames"] for r in repeats for c in r["cameras_detail"]
                ),
                "consumed_frames_total": sum(
                    c["consumed_frames"] for r in repeats for c in r["cameras_detail"]
                ),
                "dropped_frames_total": sum(
                    c["dropped_frames"] for r in repeats for c in r["cameras_detail"]
                ),
                "drain_and_cleanup_peak_seconds": max(
                    max(
                        0,
                        r["elapsed_seconds_including_drain"]
                        - r["warmup_seconds"]
                        - r["measurement_seconds"],
                    )
                    for r in repeats
                ),
                "dropped_frames_max": max(
                    sum(c["dropped_frames"] for c in r["cameras_detail"])
                    for r in repeats
                ),
            }
        )
    baseline = next(row for row in rows if row["mode"] == "baseline")
    for row in rows:
        row["host_change_from_idle_mib"] = (
            row["host_used_mean_gib"] - baseline["host_used_mean_gib"]
        ) * 1024
    return rows


def build_report_data(root):
    root = Path(root)
    all_runs = json.loads((root / "results.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    runs = [
        r for r in all_runs if r["status"] == "completed" and not r["monitoring_errors"]
    ]
    topologies = {r.get("consumer_topology", "per_camera") for r in runs}
    if len(topologies) != 1:
        raise ValueError("Do not combine results from different consumer topologies")
    consumer_topology = topologies.pop()
    rows = _aggregate(runs)
    by_key = {(row["mode"], row["cameras"]): row for row in rows}
    required = [
        ("baseline", 0),
        ("recording", 1),
        ("streaming", 1),
        ("full", 1),
        ("full", 2),
        ("full", 4),
    ]
    if any(key not in by_key for key in required):
        raise ValueError("Completed results are missing a required comparison scenario")
    baseline = by_key[("baseline", 0)]
    component_rows = [
        {
            "component": LABELS[mode],
            "additional_mean_mib": by_key[(mode, 1)]["pipeline_mean_mib"]
            - baseline["pipeline_mean_mib"],
        }
        for mode in ("recording", "streaming", "full")
    ]
    four_runs = [
        r
        for r in runs
        if r["mode"] == "full"
        and r["cameras"] == 4
        and "_soak" not in r["case"]
        and not r["consumer_delay_seconds"]
    ]
    images = {
        entry.split("|", 1)[0].strip(): entry.split("|", 1)[1].strip()
        for entry in environment["containers"]
    }
    containers = []
    for name in sorted({name for r in runs for name in r["containers"]}):
        samples = [r["containers"][name] for r in four_runs if name in r["containers"]]
        if not samples:
            raise ValueError(f"No four-camera container measurements for {name}")
        test_only = "mediamtx" in images.get(name, "").lower() or name == "mock-cctv"
        containers.append(
            {
                "name": name,
                "image": images.get(name, "unknown"),
                "test_only": test_only,
                "four_camera_mean_mib": median(s["mean_mib"] for s in samples),
                "peak_mib": max(
                    r["containers"][name]["peak_mib"]
                    for r in runs
                    if name in r["containers"]
                ),
            }
        )
    special = []
    for r in runs:
        if "_soak" not in r["case"] and not r["consumer_delay_seconds"]:
            continue
        special.append(
            {
                "case": r["case"],
                "label": "Slow consumer"
                if r["consumer_delay_seconds"]
                else "Stability",
                "warmup_seconds": r["warmup_seconds"],
                "measurement_seconds": r["measurement_seconds"],
                "consumer_delay_seconds": r["consumer_delay_seconds"],
                "elapsed_seconds": r["elapsed_seconds_including_drain"],
                "drain_and_cleanup_seconds": max(
                    0,
                    r["elapsed_seconds_including_drain"]
                    - r["warmup_seconds"]
                    - r["measurement_seconds"],
                ),
                "pipeline_peak_mib": r["metrics"]["app_rss_peak_mib"],
                "host_used_mean_gib": r["host"]["host_used_mean_mib"] / 1024,
                "host_used_peak_gib": r["host"]["host_used_peak_mib"] / 1024,
                "host_available_min_gib": r["host"]["host_available_min_mib"] / 1024,
                "host_change_from_idle_mib": r["host"]["host_used_mean_mib"]
                - baseline["host_used_mean_gib"] * 1024,
                "peak_queued_frames": r["metrics"]["queued_frames_peak"],
                "peak_queued_image_mib": r["metrics"]["queued_image_bytes_peak_mib"],
                "produced_frames": sum(
                    c["produced_frames"] for c in r["cameras_detail"]
                ),
                "consumed_frames": sum(
                    c["consumed_frames"] for c in r["cameras_detail"]
                ),
                "dropped_frames": sum(c["dropped_frames"] for c in r["cameras_detail"]),
                "late_queue_growth_mib_per_min": r["metrics"].get(
                    "queued_image_late_slope_mib_per_minute", 0
                ),
                "late_pipeline_growth_mib_per_min": r["metrics"].get(
                    "app_rss_late_slope_mib_per_minute", 0
                ),
            }
        )
    started = datetime.fromisoformat(environment["started_at_utc"]).astimezone(
        SINGAPORE
    )
    peak = max(
        r["metrics"]["app_rss_peak_mib"]
        for r in runs
        if r["mode"] == "full" and r["cameras"] == 4 and not r["consumer_delay_seconds"]
    )
    observed_peak = max(
        r["metrics"]["app_rss_peak_mib"]
        for r in runs
        if r["mode"] == "full" and r["cameras"] == 4
    )
    production_containers = sum(
        c["four_camera_mean_mib"] for c in containers if not c["test_only"]
    )
    test_containers = sum(
        c["four_camera_mean_mib"] for c in containers if c["test_only"]
    )
    four = by_key[("full", 4)]
    vm_remainder = four["wsl_mean_mib"] - production_containers - test_containers
    production_reference = (
        observed_peak
        + production_containers
        + four["docker_desktop_mean_mib"]
        + vm_remainder
        if vm_remainder >= 0
        else None
    )
    return {
        "date": started.date().isoformat(),
        "started_at_singapore": started.isoformat(),
        "generated_at_singapore": datetime.now(UTC).astimezone(SINGAPORE).isoformat(),
        "run_directory": root.name,
        "environment": environment,
        "settings": {"consumer_topology": consumer_topology}
        | {
            key: runs[0][key]
            for key in (
                "target_fps",
                "queue_size",
                "queue_policy",
                "warmup_seconds",
                "measurement_seconds",
            )
        },
        "completed_runs": len(runs),
        "excluded_runs": [r["case"] for r in all_runs if r not in runs],
        "scenarios": rows,
        "component_estimates": component_rows,
        "containers": containers,
        "production_container_reference_mib": production_containers,
        "excluded_test_container_reference_mib": test_containers,
        "estimated_wsl_remainder_mib": vm_remainder,
        "estimated_production_ingestion_runtime_mib": production_reference,
        "four_camera_fast_peak_mib": peak,
        "four_camera_observed_peak_mib": observed_peak,
        "host_available_min_mib": min(
            r["host"]["host_available_min_mib"] for r in runs
        ),
        "special_runs": special,
        "run_validation": [
            {
                "case": r["case"],
                "recording_processes_peak": r["metrics"]["ffmpeg_process_count_peak"],
                "cameras": r["cameras_detail"],
            }
            for r in runs
        ],
    }


def _table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        "",
    ]


def write_report(root):
    root = Path(root)
    data = build_report_data(root)
    rows = data["scenarios"]
    by_key = {(r["mode"], r["cameras"]): r for r in rows}
    baseline, four = by_key[("baseline", 0)], by_key[("full", 4)]
    profile, settings = data["environment"]["input"], data["settings"]
    shared_consumer = settings["consumer_topology"] == "shared_round_robin"
    consumer_setup = (
        "one shared round-robin consumer across all camera queues"
        if shared_consumer
        else "independent consumers, one per camera (historical architecture)"
    )
    delay_scope = "across all cameras" if shared_consumer else "per camera"
    one, two = by_key[("full", 1)], by_key[("full", 2)]
    increment = two["pipeline_mean_mib"] - one["pipeline_mean_mib"]
    predicted = one["pipeline_mean_mib"] + 3 * increment
    lines = [
        "# Bounded-queue pipeline RAM benchmark",
        "",
        f"Benchmark date: {data['date']} (Singapore). Completed runs: {data['completed_runs']}.",
        "",
        f"Consumer setup: {consumer_setup}.",
        "",
        "## Main results",
        "",
        f"- Four cameras with fast consumption peaked at **{data['four_camera_fast_peak_mib']:.1f} MiB of application RAM**, including the stability run.",
        f"- Shared Python baseline: **{baseline['pipeline_mean_mib']:.1f} MiB**. Recording adds **{data['component_estimates'][0]['additional_mean_mib']:.1f} MiB** and streaming, including decoding, adds **{data['component_estimates'][1]['additional_mean_mib']:.1f} MiB** in separate one-camera comparisons.",
        f"- Whole-machine RAM averaged **{four['host_used_mean_gib']:.2f} GiB** in the four-camera comparison runs. Minimum available RAM was **{four['host_available_min_gib']:.2f} GiB**. These host figures include the OS, other applications and test infrastructure.",
        f"- Production container reference: **{data['production_container_reference_mib']:.1f} MiB** working memory, excluding **{data['excluded_test_container_reference_mib']:.1f} MiB** from MediaMTX. Docker/OS overhead and future models still require their own allowance.",
        "",
    ]
    lines += _table(
        ["Recording + streaming", "Typical pipeline MiB", "Peak pipeline MiB"],
        [
            [
                f"{r['cameras']} camera(s)",
                f"{r['pipeline_mean_mib']:.1f}",
                f"{r['pipeline_peak_mib']:.1f}",
            ]
            for r in (one, two, four)
        ],
    )
    lines += [
        "These comparison values summarize the three short repeats per camera count. The headline four-camera peak also includes the longer stability run. All application figures include the shared Python runtime once and the recording FFmpeg children.",
        "",
        "## Pipeline RAM and camera scaling",
        "",
        "Application RAM is the sampled resident RAM (RSS) of Python plus its recording FFmpeg children. Shared library pages may be counted more than once when process RSS values are summed. Application figures are supporting measurements, not values to add to host RAM.",
        "",
    ]
    lines += _table(
        [
            "Scenario",
            "Cameras",
            "Typical RAM MiB",
            "Peak RAM MiB",
            "Max dropped frames/run",
        ],
        [
            [
                r["scenario"],
                r["cameras"],
                f"{r['pipeline_mean_mib']:.1f}",
                f"{r['pipeline_peak_mib']:.1f}",
                r["dropped_frames_max"],
            ]
            for r in rows
        ],
    )
    lines += [
        "Typical values are medians of per-run steady-window means; peaks include startup and draining. The idle Python baseline is shared and counted once, not multiplied by camera count.",
        "",
        "## Component estimates for one camera",
        "",
    ]
    lines += _table(
        ["Component", "Additional RAM above Python baseline (MiB)"],
        [
            [r["component"], f"{r['additional_mean_mib']:.1f}"]
            for r in data["component_estimates"]
        ],
    )
    lines += [
        "Recording includes FFmpeg and metadata registration. Streaming includes PyAV decoding, selection, BGR conversion, the bounded queue and a counting consumer. Decoding is not a third production worker and is not added again. Threads share Python memory, so these are comparative estimates rather than exact per-thread ownership; their increments need not add exactly to the combined result.",
        "",
        f"The observed one-to-two-camera increase was {increment:.1f} MiB. That increment predicts {predicted:.1f} MiB at four cameras, versus {four['pipeline_mean_mib']:.1f} MiB measured ({100 * (predicted / four['pipeline_mean_mib'] - 1):+.1f}% prediction error). This interpolation applies only to this workload and topology.",
        "",
        "## Whole-system RAM",
        "",
        "Host used RAM is total physical RAM minus available RAM. It already includes the pipeline, Docker/WSL, PostgreSQL, MediaMTX, publishers, profiling and unrelated applications. Do not sum process or container values into these totals. A host change from idle is an observed difference, not exact application attribution; background activity can make it negative.",
        "",
    ]
    lines += _table(
        [
            "Scenario",
            "Cameras",
            "Typical used GiB",
            "Peak used GiB",
            "Min available GiB",
            "Change from idle MiB",
        ],
        [
            [
                r["scenario"],
                r["cameras"],
                f"{r['host_used_mean_gib']:.2f}",
                f"{r['host_used_peak_gib']:.2f}",
                f"{r['host_available_min_gib']:.2f}",
                f"{r['host_change_from_idle_mib']:+.1f}",
            ]
            for r in rows
        ]
        + [
            [
                r["label"],
                4,
                f"{r['host_used_mean_gib']:.2f}",
                f"{r['host_used_peak_gib']:.2f}",
                f"{r['host_available_min_gib']:.2f}",
                f"{r['host_change_from_idle_mib']:+.1f}",
            ]
            for r in data["special_runs"]
        ],
    )
    lines += [
        "## PostgreSQL, Docker and production accounting",
        "",
        "The container table uses four-camera steady working-memory measurements. Docker CLI statistics on Linux exclude reclaimable cache. Container values overlap Docker/WSL and host measurements.",
        "",
    ]
    lines += _table(
        ["Container", "Typical MiB", "Peak MiB", "Production treatment"],
        [
            [
                c["name"],
                f"{c['four_camera_mean_mib']:.1f}",
                f"{c['peak_mib']:.1f}",
                "Exclude: mock relay" if c["test_only"] else "Include",
            ]
            for c in data["containers"]
        ],
    )
    lines += _table(
        ["Additional layer", "Four-camera typical MiB", "Accounting"],
        [
            [
                "WSL VM",
                f"{four['wsl_mean_mib']:.1f}",
                "Includes container allocations and VM overhead; do not add to host",
            ],
            [
                "Docker Desktop processes",
                f"{four['docker_desktop_mean_mib']:.1f}",
                "Runtime overhead remains while production services use Docker",
            ],
            [
                "Local mock publishers",
                f"{four['mock_publisher_mean_mib']:.1f}",
                "Test-only; real cameras encode and publish their own streams",
            ],
        ],
    )
    lines += [
        f"The estimated known production-container subtotal is {data['production_container_reference_mib']:.1f} MiB, excluding MediaMTX's {data['excluded_test_container_reference_mib']:.1f} MiB. It is not total server RAM and does not include models or Docker/OS overhead. Pipeline RAM already excludes MediaMTX and mock publishers, so those costs are not subtracted from application figures again. Recording FFmpeg processes remain necessary in production.",
        "",
        "Raw host totals remain as measured. Removing a container does not guarantee an identical immediate host-RAM reduction because WSL caching and shared memory behave differently. Exact whole-host RAM without MediaMTX has not been measured; it requires direct-camera input. All visible WSL processes are included, not necessarily Docker alone.",
        "The WSL measurement includes containers, VM infrastructure and retained cache. It cannot isolate Docker-only VM overhead exactly; the full VM allocation is shown instead. Docker Desktop process RAM is measured separately.",
        "",
    ]
    if data["estimated_production_ingestion_runtime_mib"] is not None:
        reference = data["estimated_production_ingestion_runtime_mib"]
        lines += ["## Production ingestion accounting estimate", ""]
        lines += _table(
            ["Four-camera production accounting estimate", "RAM MiB"],
            [
                [
                    "Pipeline, including Python and recording FFmpeg (peak incl. stress)",
                    f"{data['four_camera_observed_peak_mib']:.1f}",
                ],
                [
                    "Retained production containers (typical)",
                    f"{data['production_container_reference_mib']:.1f}",
                ],
                [
                    "Docker Desktop processes (typical)",
                    f"{four['docker_desktop_mean_mib']:.1f}",
                ],
                [
                    "WSL remainder after all measured container working memory",
                    f"{data['estimated_wsl_remainder_mib']:.1f}",
                ],
                ["Illustrative ingestion + runtime subtotal", f"{reference:.1f}"],
            ],
        )
        lines += [
            f"The illustrative subtotal is **{reference / 1024:.2f} GiB before Windows/host OS, unrelated applications and models**. It retains PostgreSQL and runtime overhead while omitting MediaMTX working memory and the publishers. The WSL remainder is VM resident RAM minus all container working-memory references; it includes VM infrastructure, retained cache and potentially other WSL workloads. These different counters do not isolate ownership exactly, and the subtotal mixes a pipeline peak with typical service values. It is a rough accounting estimate, not a measured production peak, an upper bound, or a server specification. It does not modify the measured host totals.",
            "",
        ]
    lines += [
        "## Queue stability and stress",
        "",
    ]
    lines += _table(
        [
            "Normal scenario",
            "Cameras",
            "Peak waiting",
            "Image MiB",
            "Consumed / dropped",
            "Max drain/cleanup s",
        ],
        [
            [
                r["scenario"],
                r["cameras"],
                r["peak_queued_frames"],
                f"{r['peak_queued_image_mib']:.2f}",
                f"{r['consumed_frames_total']:,} / {r['dropped_frames_total']:,}",
                f"{r['drain_and_cleanup_peak_seconds']:.1f}",
            ]
            for r in rows
            if r["mode"] in ("streaming", "full")
        ],
    )
    lines += [
        "Normal-run frame counts sum all repeats, including warm-up. Queue peaks are simultaneous waiting frames across cameras, tracked on every enqueue/dequeue rather than inferred from sampling. Drain/cleanup is time after the planned producer deadline; it also includes recorder finalization and worker teardown, so it is an upper bound on queue-only draining.",
        "",
    ]
    lines += _table(
        [
            "Case",
            "Measured s",
            "Total s",
            "Peak RAM MiB",
            "Peak queued",
            "Dropped",
            "Drain/cleanup s",
        ],
        [
            [
                r["label"],
                f"{r['measurement_seconds']:g}",
                f"{r['elapsed_seconds']:.1f}",
                f"{r['pipeline_peak_mib']:.1f}",
                r["peak_queued_frames"],
                r["dropped_frames"],
                f"{r['drain_and_cleanup_seconds']:.1f}",
            ]
            for r in data["special_runs"]
        ],
    )
    for run in data["special_runs"]:
        lines += [
            f"- {run['label']}: produced {run['produced_frames']:,}, consumed {run['consumed_frames']:,}, dropped {run['dropped_frames']:,}; peak queued image data {run['peak_queued_image_mib']:.1f} MiB. Late queue-image trend: {run['late_queue_growth_mib_per_min']:.2f} MiB/min; late application trend: {run['late_pipeline_growth_mib_per_min']:.2f} MiB/min."
        ]
    payload = profile["width"] * profile["height"] * 3 / 1024**2
    lines += [
        "",
        f"Each selected BGR image uses {payload:.3f} MiB before metadata and decoder buffers. The configured bound is {settings['queue_size']} waiting frames per camera, or {4 * settings['queue_size']} across four cameras. The frame being processed and decoder/conversion buffers are outside that queue limit. On overflow, the oldest waiting frame is dropped; the active frame is not interrupted. Stable queue memory can therefore coexist with missed live frames. Recording preserves video independently, but does not automatically recover skipped detections.",
        "",
        "## Workload, method and limits",
        "",
        f"- Host: {data['environment']['os']}; {data['environment']['host_ram_mib'] / 1024:.2f} GiB physical RAM; Python {data['environment']['python']}.",
        f"- Input: {profile['codec']}, {profile['width']} x {profile['height']}, {profile['source_fps']} source FPS over RTSP/TCP; up to {settings['target_fps']:g} selected FPS per camera. Every incoming frame is still decoded.",
        f"- Queue: {settings['queue_size']} waiting frames per camera, drop oldest on overflow. Cameras share one Python process and database engine, with independent worker pairs and {consumer_setup}.",
        f"- Normal trials: {settings['warmup_seconds']:g}s warm-up + {settings['measurement_seconds']:g}s measurement, {baseline['repeats']} fresh-process repeats per scenario. Stability and stress durations are shown above. The slow consumer sleeps {next((r['consumer_delay_seconds'] for r in data['special_runs'] if r['consumer_delay_seconds']), 0):g}s per frame {delay_scope}.",
        "- Process RAM is sampled approximately every 0.2s, host RAM every 1s, and Docker approximately every 5-8s. Short peaks between samples may be missed. Peak/minimum host figures cover each full observed run, while typical host figures exclude warm-up and draining.",
        "- The benchmark uses the production worker and runner helpers. The sampler retains timestamped rows in memory; small long-run growth may include profiling overhead. The coordinator and sampling commands are also included in host totals.",
        "- A dedicated timestamp-normalized publisher reads the sample video, uses one-second H.264 keyframes and no B-frames, and shares the existing mock server. Existing streams/services remain running. The idle baseline is services-running, not Docker-off or OS-only.",
        "- Simulated cameras read independent connections to one mock stream. Distinct physical camera scenes, source codecs and resolutions may alter memory. The current input is not a 1080p/4K hardware-sizing benchmark.",
        "- No YOLO/VLM weights, inference workspaces or GPU memory are included. A shared consumer needs to sustain the aggregate selected rate, not just the per-camera rate. Model and deployment-container changes require another measurement.",
        "- Windows background applications, available-memory pressure, shared pages and Docker/WSL caching affect RAM. A fully containerized/Linux production deployment must be measured separately. Do not interpret the current host total or component reference subtotal as a final hardware recommendation.",
        f"- This host was already busy: idle-run used RAM peaked at {baseline['host_used_peak_gib']:.2f} GiB, and the lowest available RAM across all runs was {data['host_available_min_mib']:.0f} MiB. Memory pressure and unrelated activity limit causal interpretation of host deltas.",
        f"- Completed runs: {data['completed_runs']}; excluded runs: {len(data['excluded_runs'])}. Successful runs are checked for bounded queues, expected recording processes, complete draining and produced = consumed + dropped. No new test cases were added.",
        "",
        "## Reproduction and artifacts",
        "",
        "Run `npm run benchmark:memory:suite -- --queue-size 8` with PostgreSQL migrated and the mock RTSP server running. The suite owns only its separate publisher and camera workers; it leaves existing services and recordings intact.",
        "",
        f"Raw data: `recordings/benchmarks/{root.name}/`. Each case contains process.csv, host.csv, docker.csv, summary.json and worker.log. Suite files include environment.json, results.json, report_data.json, aggregate.csv and report.md. Benchmark date: {data['date']} Singapore time.",
        "",
    ]
    if data["excluded_runs"]:
        lines += ["Excluded cases: " + ", ".join(data["excluded_runs"]), ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "report_data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    write_report(parser.parse_args().root)
