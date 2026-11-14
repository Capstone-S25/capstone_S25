"""Generate a standalone Markdown report and aggregate CSV from measured runs."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import median


def _aggregate(results):
    grouped = defaultdict(list)
    for run in results:
        if "_soak" not in run["case"] and not run["consumer_delay_seconds"]:
            grouped[(run["mode"], run["cameras"])].append(run)
    rows = []
    for (mode, cameras), runs in grouped.items():
        metrics = [run["metrics"] for run in runs]
        rows.append(
            {
                "mode": mode,
                "cameras": cameras,
                "repeats": len(runs),
                "app_rss_mean_mib": median(m["app_rss_mean_mib"] for m in metrics),
                "app_rss_peak_mib": max(m["app_rss_peak_mib"] for m in metrics),
                "app_uss_mean_mib": median(m["app_uss_mean_mib"] for m in metrics),
                "python_rss_mean_mib": median(
                    m["python_rss_mean_mib"] for m in metrics
                ),
                "ffmpeg_rss_mean_mib": median(
                    m["ffmpeg_rss_mean_mib"] for m in metrics
                ),
                "queue_frames_peak": max(m["queued_frames_peak"] for m in metrics),
                "host_used_mean_gib": median(
                    r["host"]["host_used_mean_mib"] / 1024 for r in runs
                ),
                "host_available_min_gib": min(
                    r["host"]["host_available_min_mib"] / 1024 for r in runs
                ),
                "wsl_rss_mean_mib": median(
                    r["host"]["wsl_vm_rss_mean_mib"] for r in runs
                ),
                "desktop_rss_mean_mib": median(
                    r["host"]["docker_desktop_rss_mean_mib"] for r in runs
                ),
                "mock_publisher_rss_mean_mib": median(
                    r["host"]["mock_publisher_rss_mean_mib"] for r in runs
                ),
            }
        )
    return rows


def _direct_camera_services(results):
    """Exclude mock-only services from a labelled direct-camera budget estimate."""
    runs = [
        run
        for run in results
        if run["mode"] == "full"
        and run["cameras"] == 4
        and "_soak" not in run["case"]
        and not run["consumer_delay_seconds"]
    ]
    postgres = median(
        run["containers"]["capstone-postgres-dev"]["mean_mib"] for run in runs
    )
    mtx = median(run["containers"]["mock-cctv"]["mean_mib"] for run in runs)
    return {
        "postgres_mib": postgres,
        "excluded_mtx_mib": mtx,
        "test_container_total_mib": postgres + mtx,
        "production_known_container_estimate_mib": postgres,
    }


def write_report(root):
    root = Path(root)
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if any(run["status"] != "completed" for run in results):
        raise ValueError("Cannot report failed scenarios as completed measurements")
    rows = _aggregate(results)
    by_key = {(row["mode"], row["cameras"]): row for row in rows}
    baseline = by_key[("baseline", 0)]
    production = _direct_camera_services(results)
    profile = environment["input"]
    first = results[0]
    queue_size = first.get("queue_size", 0)
    queue_description = (
        f"bounded drop-oldest, {queue_size} waiting frames per camera"
        if queue_size
        else "historical unbounded FIFO"
    )
    local_time = datetime.now(UTC).astimezone(timezone(timedelta(hours=8)))
    lines = [
        "# Pipeline memory benchmark report",
        "",
        f"Generated {local_time:%Y-%m-%d %H:%M} Singapore time.",
        "",
        "## Workload and method",
        "",
        f"- Host: {environment['os']}; {environment['host_ram_mib'] / 1024:.2f} GiB physical RAM; Python {environment['python']}.",
        f"- Input: {profile['codec']}, {profile['width']}×{profile['height']}, {profile['source_fps']} FPS, RTSP over TCP.",
        f"- Selection rate: up to {first['target_fps']:g} FPS per camera; decoding still reads the full incoming stream.",
        "- One mock stream supplies independent connections; each simulated camera owns its decoder, recording session, FFmpeg process and queue.",
        "- Cameras share one Python process and database engine. Each camera has its own counting consumer; no YOLO or VLM model is loaded.",
        f"- Normal cases: {first['warmup_seconds']:g}s warm-up then {first['measurement_seconds']:g}s measurement; {rows[0]['repeats']} fresh-process repeats per case.",
        "- Reported means are medians of per-run steady-window means. Peaks are the highest sampled value across complete runs, including startup and draining.",
        "- Process RSS/USS is sampled approximately every 0.2s, host memory every 1s, Docker statistics approximately every 5–8s.",
        "- RSS is resident memory; USS is private resident memory. Summed RSS may count shared pages more than once. Short peaks between samples may be missed.",
        f"- Queue policy: {queue_description}. Instrumentation counts image bytes and drops without changing the configured policy.",
        "- Existing services and the mock publisher remain running. Baseline means no camera workers, not an otherwise empty machine or Docker-off measurement.",
        "- Whole-host measurements also include the benchmark coordinator and its monitoring commands; component application measurements exclude that separate coordinator.",
        "- The in-process sampler retains measurements until the run ends. Small long-run application memory growth can include sample-storage overhead; queue-image growth is reported separately.",
        "- Decoder memory belongs to the streaming worker. The offline recorded-segment decoder is not running and is not counted again.",
        "",
        "## Component and camera measurements",
        "",
        "Application figures include Python plus recording FFmpeg processes; they exclude Docker services.",
        "",
        "| Scenario | Cameras | Python mean MiB | FFmpeg mean MiB | App mean RSS MiB | App peak RSS MiB | App mean USS MiB | Peak queued frames |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if environment.get("dedicated_publisher"):
        lines.insert(
            lines.index("## Component and camera measurements"),
            "The suite owns a separate looping sample-video publisher with normalized timestamps, H.264 at 25 FPS, one-second keyframes and no B-frames. Its memory is test infrastructure and is included in host/publisher totals. Existing streams were not modified.\n",
        )
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['cameras']} | {row['python_rss_mean_mib']:.1f} | {row['ffmpeg_rss_mean_mib']:.1f} | {row['app_rss_mean_mib']:.1f} | {row['app_rss_peak_mib']:.1f} | {row['app_uss_mean_mib']:.1f} | {row['queue_frames_peak']} |"
        )
    lines.extend(
        [
            "",
            "### Additional memory for one camera",
            "",
            "These are controlled-run differences from the same imported Python baseline, not exact ownership of shared thread allocations.",
            "",
            "| Component | Additional steady RSS MiB | Additional steady USS MiB |",
            "| --- | ---: | ---: |",
        ]
    )
    for mode in ("recording", "streaming", "full"):
        row = by_key[(mode, 1)]
        lines.append(
            f"| {mode} | {row['app_rss_mean_mib'] - baseline['app_rss_mean_mib']:.1f} | {row['app_uss_mean_mib'] - baseline['app_uss_mean_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Docker and whole-machine memory",
            "",
            "Whole-machine used memory is total RAM minus available RAM. It already includes the application, Docker/WSL, OS, publisher and other open applications; do not add component numbers to it.",
            "WSL figures include all visible vmmem processes, not necessarily Docker alone. Docker Desktop figures are the sum of its visible Windows processes and may share pages.",
            "",
            "| Scenario | Cameras | Host used mean GiB | Host available minimum GiB | WSL VM RSS mean MiB | Docker Desktop RSS mean MiB | Mock publisher RSS mean MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['cameras']} | {row['host_used_mean_gib']:.2f} | {row['host_available_min_gib']:.2f} | {row['wsl_rss_mean_mib']:.1f} | {row['desktop_rss_mean_mib']:.1f} | {row['mock_publisher_rss_mean_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### Running containers",
            "",
            "Docker CLI statistics exclude reclaimable cache on Linux. These container figures overlap WSL/host totals and must not be added to them.",
            "",
            "| Container | Services-idle mean MiB | Four-camera mean MiB | Highest observed MiB |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    names = sorted({name for run in results for name in run["containers"]})
    for name in names:
        idle = [
            r["containers"][name]["mean_mib"]
            for r in results
            if r["mode"] == "baseline" and name in r["containers"]
        ]
        four = [
            r["containers"][name]["mean_mib"]
            for r in results
            if r["mode"] == "full"
            and r["cameras"] == 4
            and "_soak" not in r["case"]
            and not r["consumer_delay_seconds"]
            and name in r["containers"]
        ]
        peak = max(
            r["containers"][name]["peak_mib"]
            for r in results
            if name in r["containers"]
        )
        idle_text = f"{median(idle):.1f}" if idle else "not sampled"
        four_text = f"{median(four):.1f}" if four else "not sampled"
        lines.append(f"| {name} | {idle_text} | {four_text} | {peak:.1f} |")
    lines.extend(
        [
            "",
            "## Direct-camera production estimate (MediaMTX excluded)",
            "",
            "For the intended production topology, real cameras provide the RTSP streams directly. The recording and streaming workers connect to each camera URL; MediaMTX is not required unless a relay is deliberately introduced.",
            "This section is an accounting estimate using the four-camera measurements, not a new benchmark with MediaMTX stopped. Historical measurements above and raw CSV/JSON data remain unchanged.",
            "",
            "| Component | Observed in the test setup | Direct-camera production budget |",
            "| --- | --- | --- |",
            f"| PostgreSQL container | {production['postgres_mib']:.1f} MiB mean working memory | Retain {production['postgres_mib']:.1f} MiB as the measured reference |",
            f"| MediaMTX mock relay | {production['excluded_mtx_mib']:.1f} MiB mean working memory | Exclude: 0 MiB allocated to this service |",
            f"| Known container working-memory subtotal | {production['test_container_total_mib']:.1f} MiB (PostgreSQL + MediaMTX) | About {production['production_known_container_estimate_mib']:.1f} MiB (PostgreSQL only) |",
            "| Local mock-video publishers | Test-only FFmpeg encoders | Exclude; encoding/stream publishing happens on the cameras |",
            "| Recording FFmpeg processes | Included in application measurements | Keep; the server still needs to record the incoming streams |",
            "| Docker runtime / VM overhead | Included in host observations | Keep an allowance while PostgreSQL or other production services use Docker |",
            "",
            f"Remove approximately {production['excluded_mtx_mib']:.1f} MiB from the known-container working-memory budget. This is not an exact physical host-RAM saving: container working memory, process RSS and WSL allocation are different, overlapping measurements. Do not subtract all of WSL/Docker Desktop or directly rewrite the measured whole-host totals.",
            "The measured 200.0 MiB fast-consumer application peak and provisional 250 MiB application allowance already exclude MediaMTX, so neither is reduced again. A direct-camera deployment run is still needed to measure actual host savings and production-resolution behavior.",
            "",
        ]
    )
    one, two, four = [by_key[("full", n)] for n in (1, 2, 4)]
    additional = two["app_rss_mean_mib"] - one["app_rss_mean_mib"]
    prediction = one["app_rss_mean_mib"] + 3 * additional
    fast_four_peak = max(
        run["metrics"]["app_rss_peak_mib"]
        for run in results
        if run["mode"] == "full"
        and run["cameras"] == 4
        and not run["consumer_delay_seconds"]
    )
    error_percent = (
        100 * (prediction - four["app_rss_mean_mib"]) / four["app_rss_mean_mib"]
    )
    lines.extend(
        [
            "",
            "## Scaling estimate",
            "",
            f"The increase from one to two cameras was {additional:.1f} MiB steady RSS. Extrapolating that increment predicts {prediction:.1f} MiB for four cameras; the actual four-camera result was {four['app_rss_mean_mib']:.1f} MiB ({error_percent:+.1f}% prediction error).",
            f"Across fast-consumer four-camera runs, including any stability run, the highest sampled application RSS was {fast_four_peak:.1f} MiB. A provisional application-only allowance with 25% headroom is {fast_four_peak * 1.25:.1f} MiB for this workload. This excludes extra queue backlog and models and is not a whole-machine RAM recommendation.",
            "Shared Python overhead is counted once. Use measurements at the actual resolution/codec and include separate model memory, database growth, OS/services and queue capacity before sizing deployment hardware.",
            "",
            "## Sustained operation and slower consumer",
            "",
            "| Case | Input cameras | Measurement s | Total including drain s | App peak RSS MiB | Peak queued frames | Peak queued image MiB | Late RSS slope MiB/min | Late queue slope MiB/min | Dropped frames |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    private_increment = two["app_uss_mean_mib"] - one["app_uss_mean_mib"]
    position = lines.index("## Sustained operation and slower consumer")
    lines[position:position] = [
        f"Private resident memory increased by approximately {private_increment:.1f} MiB from one to two cameras. Four cameras used {four['app_uss_mean_mib']:.1f} MiB private resident memory on average, versus {four['app_rss_mean_mib']:.1f} MiB summed RSS. Shared executable/library pages explain part of the difference; neither figure should be added to host used RAM.",
        "",
    ]
    for run in results:
        if "_soak" in run["case"] or run["consumer_delay_seconds"]:
            m = run["metrics"]
            dropped = sum(c.get("dropped_frames", 0) for c in run["cameras_detail"])
            lines.append(
                f"| {run['case']} | {run['cameras']} | {run['measurement_seconds']:g} | {run['elapsed_seconds_including_drain']:.1f} | {m['app_rss_peak_mib']:.1f} | {m['queued_frames_peak']} | {m['queued_image_bytes_peak_mib']:.1f} | {m.get('app_rss_late_slope_mib_per_minute', 0):.2f} | {m.get('queued_image_late_slope_mib_per_minute', 0):.2f} | {dropped} |"
            )
    image_mib = profile["width"] * profile["height"] * 3 / 1024**2
    lines.extend(
        [
            "",
            f"Each queued BGR image in this workload uses {image_mib:.3f} MiB, excluding frame/Python metadata and decoder reference buffers. The slow case sleeps 0.75s per consumed frame per camera, intentionally slower than selected input.",
            (
                "The bounded queue limits waiting frames by dropping the oldest on overflow. Drop counts must be considered alongside memory and consumed frames; lower memory does not imply every selected frame was processed."
                if queue_size
                else "A positive queue-growth rate means memory has no fixed long-term bound. Finishing a finite stress test is not evidence that an unbounded queue is safe indefinitely. Near-zero growth during the soak establishes behavior only for that measured interval."
            ),
            "",
            "## Limits and deployment interpretation",
            "",
            "- All simulated cameras read the same mock stream through independent connections. Distinct scenes and codecs may change memory behavior.",
            "- The current stream is 640×360. These numbers are not a 1080p/4K capacity estimate; frame payload scales with pixel count, but total decoder memory is not reliably extrapolated that way.",
            "- Multiple independent consumers are used. A future single shared YOLO consumer must sustain the aggregate selected FPS; its queue behavior requires a new benchmark.",
            f"- At four cameras and {first['target_fps']:g} selected FPS each, a shared consumer would need to handle up to {4 * first['target_fps']:g} FPS (about {1000 / (4 * first['target_fps']):.0f}ms per frame on average, including processing overhead) to avoid sustained backlog.",
            "- YOLO/VLM model weights, inference workspace and GPU memory are absent. They can dominate the eventual memory budget.",
            "- The Python pipeline runs on the Windows host, while PostgreSQL and MediaMTX run in Docker. A fully containerized/Linux deployment must be measured again.",
            "- The mock CCTV container and local FFmpeg publisher are test infrastructure; real cameras may remove them. Their cost is included in host measurements here.",
            "- Background Windows applications and Docker VM caching can move host totals independently of the pipeline. Host delta is not exact application attribution.",
            "- Services were not stopped, so OS-only and Docker-off baselines were not measured. PostgreSQL contains the existing development data plus real benchmark segment metadata.",
            "",
            "## Artifacts",
            "",
            f"Raw runs: `recordings/benchmarks/{root.name}/`. Each case includes process, host and Docker CSV samples, a JSON summary and worker log. `results.json` contains all run summaries; `aggregate.csv` contains the table aggregates.",
            "",
        ]
    )
    if environment.get("excluded_attempt"):
        attempt = environment["excluded_attempt"]
        position = lines.index("## Artifacts")
        lines[position:position] = [
            "## Input timing finding",
            "",
            f"An earlier four-camera attempt against the pre-existing mock feed stopped after {attempt['elapsed_seconds']:.1f}s with `{attempt['reason']}`. No memory figures from that attempt are included in this report. Evidence is retained in `recordings/benchmarks/{attempt['directory']}/full_4cam_r1/worker.log`.",
            "All reported comparisons use the separate timestamp-normalized feed. The earlier timing failure warrants a separate ingestion-reliability investigation; changing benchmark input is not a production fix.",
            "",
        ]
    monitoring_errors = [error for run in results for error in run["monitoring_errors"]]
    if monitoring_errors:
        lines.extend(
            [
                "### Monitoring limitations",
                "",
                *[f"- {error}" for error in monitoring_errors],
                "",
            ]
        )
    findings = [
        "## Findings",
        "",
        f"- Four cameras with fast counting consumers peaked at **{fast_four_peak:.1f} MiB summed application RSS**, including the stability run. This covers Python and recording FFmpeg processes, not Docker services or model inference.",
        f"- Above the {baseline['app_rss_mean_mib']:.1f} MiB shared Python baseline, one camera added approximately **{by_key[('recording', 1)]['app_rss_mean_mib'] - baseline['app_rss_mean_mib']:.1f} MiB for recording** and **{by_key[('streaming', 1)]['app_rss_mean_mib'] - baseline['app_rss_mean_mib']:.1f} MiB for streaming including decoding**.",
        f"- Whole-host memory averaged **{four['host_used_mean_gib']:.2f} GiB** during the four-camera comparison windows. It includes Docker, OS and unrelated applications and is not the project's standalone RAM requirement.",
        f"- Direct-camera production **excludes MediaMTX and mock publishers**. Removing MediaMTX's {production['excluded_mtx_mib']:.1f} MiB mean working memory reduces the known-container reference budget from {production['test_container_total_mib']:.1f} to **{production['production_known_container_estimate_mib']:.1f} MiB** (PostgreSQL only). This is an estimate; application-only figures are unchanged.",
    ]
    slow_runs = [run for run in results if run["consumer_delay_seconds"]]
    if slow_runs:
        slow = slow_runs[-1]["metrics"]
        dropped = sum(
            c.get("dropped_frames", 0) for c in slow_runs[-1]["cameras_detail"]
        )
        conclusion = (
            f"The bounded queues discarded {dropped} oldest frames on overflow."
            if queue_size
            else "An unbounded queue therefore has no fixed memory ceiling if this imbalance continues."
        )
        findings.append(
            f"- With deliberately slow consumers, queued image data grew by **{slow['queued_image_late_slope_mib_per_minute']:.1f} MiB/min**, reached **{slow['queued_frames_peak']} frames**, and application RSS peaked at **{slow['app_rss_peak_mib']:.1f} MiB**. {conclusion}"
        )
    findings.extend(
        [
            "- These are **640×360, H.264, 2 selected FPS per camera** measurements. YOLO/VLM and production-resolution capacity remain unmeasured.",
            "",
        ]
    )
    position = lines.index("## Workload and method")
    lines[position:position] = findings
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    write_report(parser.parse_args().root)
