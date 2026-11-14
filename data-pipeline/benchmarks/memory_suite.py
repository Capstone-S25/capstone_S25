"""Benchmark components and 1/2/4 cameras with Docker and host memory samples."""

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4

import psutil

from data_pipeline.video_ingestion.streaming.frame_queue import DEFAULT_FRAME_QUEUE_SIZE

PROJECT_DIR = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
DEFAULT_RTSP_URL = "rtsp://localhost:8554/camera_01"


def _docker(*arguments):
    return subprocess.run(
        ["docker", *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    ).stdout


def _parse_memory(value):
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?i?B)\s*", value)
    if not match:
        raise ValueError(f"Unexpected Docker memory value: {value}")
    amount, unit = match.groups()
    power = {"B": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit[0]]
    return float(amount) * (1024 if "i" in unit else 1000) ** power


def _host_sample(started):
    memory = psutil.virtual_memory()
    wsl_rss = desktop_rss = publisher_rss = 0
    missing = 0
    for process in psutil.process_iter(["pid", "name"]):
        name = (process.info["name"] or "").lower()
        try:
            if name.startswith("vmmem"):
                wsl_rss += process.memory_info().rss
            elif "docker desktop" in name or name.startswith("com.docker"):
                desktop_rss += process.memory_info().rss
            elif name.startswith("ffmpeg"):
                command = process.cmdline()
                publishing = any(
                    command[i : i + 2] == ["-f", "rtsp"]
                    for i in range(len(command) - 1)
                )
                if publishing and any(
                    "localhost:8554/" in argument for argument in command
                ):
                    publisher_rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            missing += 1
    return {
        "elapsed_seconds": perf_counter() - started,
        "host_total_bytes": memory.total,
        "host_used_bytes": memory.total - memory.available,
        "host_available_bytes": memory.available,
        "wsl_vm_rss_bytes": wsl_rss,
        "docker_desktop_rss_bytes": desktop_rss,
        "mock_publisher_rss_bytes": publisher_rss,
        "unreadable_processes": missing,
    }


def _host_monitor(started, stop, rows, errors):
    try:
        while True:
            rows.append(_host_sample(started))
            if stop.wait(1):
                break
    except Exception as error:
        errors.append(f"Host monitor: {type(error).__name__}: {error}")


def _docker_monitor(started, stop, rows, errors):
    while True:
        try:
            output = _docker("stats", "--no-stream", "--format", "{{json .}}")
            for line in output.splitlines():
                data = json.loads(line)
                rows.append(
                    {
                        "elapsed_seconds": perf_counter() - started,
                        "name": data["Name"],
                        "memory_bytes": _parse_memory(data["MemUsage"].split("/")[0]),
                    }
                )
        except Exception as error:
            errors.append(f"Docker monitor: {type(error).__name__}: {error}")
        if stop.wait(5):
            return


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _service_summary(host_rows, docker_rows, warmup, duration):
    host = [r for r in host_rows if warmup <= r["elapsed_seconds"] <= warmup + duration]
    if not host:
        raise RuntimeError("No host measurements in the active window")
    summary = {}
    for field in (
        "host_used_bytes",
        "host_available_bytes",
        "wsl_vm_rss_bytes",
        "docker_desktop_rss_bytes",
        "mock_publisher_rss_bytes",
    ):
        name = field.removesuffix("_bytes")
        summary[f"{name}_mean_mib"] = mean(r[field] for r in host) / MIB
        summary[f"{name}_peak_mib"] = max(r[field] for r in host) / MIB
        summary[f"{name}_min_mib"] = min(r[field] for r in host) / MIB
    containers = {}
    for name in sorted({r["name"] for r in docker_rows}):
        samples = [
            r["memory_bytes"]
            for r in docker_rows
            if r["name"] == name and warmup <= r["elapsed_seconds"] <= warmup + duration
        ]
        if samples:
            containers[name] = {
                "mean_mib": mean(samples) / MIB,
                "peak_mib": max(samples) / MIB,
                "samples": len(samples),
            }
    return {"host": summary, "containers": containers}


def _run_case(args, root, mode, cameras, repeat, *, duration=None, delay=0):
    label = f"{mode}_{cameras}cam_r{repeat}" + ("_slow" if delay else "")
    if duration is not None:
        label += "_soak"
    output = root / label
    output.mkdir()
    measure_duration = duration or args.duration
    command = [
        sys.executable,
        "-m",
        "benchmarks.memory_components",
        "--mode",
        mode,
        "--cameras",
        str(cameras),
        "--warmup",
        str(args.warmup),
        "--duration",
        str(measure_duration),
        "--consumer-delay",
        str(delay),
        "--queue-size",
        str(args.queue_size),
        "--output",
        str(output),
    ]
    for url in args.rtsp_url or [DEFAULT_RTSP_URL]:
        command.extend(["--rtsp-url", url])
    host_rows, docker_rows, errors = [], [], []
    stop = Event()
    started = perf_counter()
    monitors = [
        Thread(target=_host_monitor, args=(started, stop, host_rows, errors)),
        Thread(target=_docker_monitor, args=(started, stop, docker_rows, errors)),
    ]
    for monitor in monitors:
        monitor.start()
    print(
        f"START {label}: warmup={args.warmup:g}s, measurement={measure_duration:g}s",
        flush=True,
    )
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                next_progress = 30
                while process.poll() is None:
                    if stop.wait(0.5):
                        break
                    elapsed = perf_counter() - started
                    if elapsed >= next_progress:
                        print(f"RUNNING {label}: {elapsed:.0f}s elapsed", flush=True)
                        next_progress += 30
            except KeyboardInterrupt:
                (output / "STOP").touch()
                process.wait(timeout=180)
                raise
        if process.returncode != 0:
            raise RuntimeError(f"Scenario failed: {output / 'worker.log'}")
    finally:
        stop.set()
        for monitor in monitors:
            monitor.join()
        _write_csv(output / "host.csv", host_rows)
        _write_csv(output / "docker.csv", docker_rows)
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    offset = result["started_perf_counter"] - started
    result.update(
        _service_summary(host_rows, docker_rows, args.warmup + offset, measure_duration)
    )
    result.update(
        case=label, monitoring_errors=errors, host_clock_offset_seconds=offset
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"DONE {label}: app peak={result['metrics']['app_rss_peak_mib']:.1f} MiB, queue peak={result['metrics']['queued_frames_peak']} frames",
        flush=True,
    )
    return result


def _environment(url):
    import av

    with av.open(url, options={"rtsp_transport": "tcp"}, timeout=5) as video:
        stream = video.streams.video[0]
        frame = next(video.decode(stream))
        profile = {
            "width": frame.width,
            "height": frame.height,
            "source_fps": str(stream.average_rate),
            "codec": stream.codec_context.name,
        }
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "host_ram_mib": psutil.virtual_memory().total / MIB,
        "docker": _docker(
            "info",
            "--format",
            "{{.OperatingSystem}} | VM memory limit={{.MemTotal}} bytes",
        ).strip(),
        "containers": _docker("ps", "--format", "{{.Names}} | {{.Image}}").splitlines(),
        "input": profile,
    }


@contextmanager
def _benchmark_publisher(args, root):
    """Own a separate, timestamp-normalized feed without changing existing streams."""
    if args.rtsp_url:
        yield
        return
    url = f"rtsp://localhost:8554/memory_benchmark_{root.name[-6:]}"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-fflags",
        "+genpts",
        "-i",
        str(PROJECT_DIR / "samples/videos/loitering.avi"),
        "-map",
        "0:v:0",
        "-vf",
        "setpts=N/(25*TB)",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-g",
        "25",
        "-bf",
        "0",
        "-an",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]
    with (root / "publisher.log").open("w", encoding="utf-8") as log:
        publisher = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    args.rtsp_url = [url]
    try:
        # Opening once establishes that frames, not just the RTSP server, are ready.
        deadline = perf_counter() + 30
        while True:
            try:
                _environment(url)
                break
            except Exception:
                if publisher.poll() is not None or perf_counter() >= deadline:
                    raise RuntimeError(
                        f"Benchmark publisher failed: {root / 'publisher.log'}"
                    ) from None
                Event().wait(1)
        yield
    finally:
        if publisher.poll() is None:
            try:
                publisher.communicate(input="q\n", timeout=10)
            except subprocess.TimeoutExpired:
                publisher.kill()
                publisher.communicate()
        if publisher.stdin is not None:
            publisher.stdin.close()


def _run_suite(args, root):
    environment = _environment((args.rtsp_url or [DEFAULT_RTSP_URL])[0])
    environment["dedicated_publisher"] = (root / "publisher.log").exists()
    (root / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SUITE OUTPUT {root}", flush=True)
    results = []
    cases = [
        ("baseline", 0),
        ("recording", 1),
        ("streaming", 1),
        ("full", 1),
        ("full", 2),
        ("full", 4),
    ]
    for repeat in range(1, args.repeats + 1):
        for mode, cameras in cases:
            results.append(_run_case(args, root, mode, cameras, repeat))
            (root / "results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
    if args.soak_duration:
        results.append(_run_case(args, root, "full", 4, 1, duration=args.soak_duration))
    if not args.skip_stress:
        results.append(_run_case(args, root, "full", 4, 1, delay=0.75))
    (root / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    from benchmarks.memory_report import write_report

    write_report(root)
    print(f"REPORT {root / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queue-size", type=int, default=DEFAULT_FRAME_QUEUE_SIZE)
    parser.add_argument("--soak-duration", type=float, default=600)
    parser.add_argument("--rtsp-url", action="append", default=[])
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.queue_size <= 0:
        parser.error("--queue-size must be a positive integer")
    if (
        args.warmup < 0
        or args.duration <= 0
        or args.repeats <= 0
        or args.soak_duration < 0
    ):
        parser.error(
            "Durations and repeats must be positive; warmup and soak may be zero"
        )
    root = args.output or PROJECT_DIR / "recordings" / "benchmarks" / (
        datetime.now(UTC).strftime("suite_%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    )
    root.mkdir(parents=True)
    with _benchmark_publisher(args, root):
        _run_suite(args, root)


if __name__ == "__main__":
    main()
