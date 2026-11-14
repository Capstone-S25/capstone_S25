"""Measure resident memory of the pipeline and its FFmpeg child process."""

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import Event, Thread
from time import perf_counter, sleep
from uuid import uuid4

import psutil

from data_pipeline.database.session import create_engine_and_session_factory
from data_pipeline.pipeline.runner import run_pipeline
from data_pipeline.video_ingestion.streaming.frame_queue import DEFAULT_FRAME_QUEUE_SIZE
from data_pipeline.video_ingestion.streaming.worker import StreamFrame
from main import _build_video_source, _load_database_url

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_INTERVAL_SECONDS = 0.1
BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class MemorySample:
    elapsed_seconds: float
    python_rss_bytes: int
    children_rss_bytes: int

    @property
    def total_rss_bytes(self) -> int:
        return self.python_rss_bytes + self.children_rss_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["file", "rtsp"], default="file")
    parser.add_argument(
        "--video", type=Path, default=PROJECT_DIR / "samples/videos/loitering.avi"
    )
    parser.add_argument("--rtsp-url", default="rtsp://localhost:8554/camera_01")
    parser.add_argument("--target-fps", type=float, default=2)
    parser.add_argument("--queue-size", type=int, default=DEFAULT_FRAME_QUEUE_SIZE)
    parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Request shutdown after this many seconds at the next callback (default: 30).",
    )
    parser.add_argument(
        "--consumer-delay",
        type=float,
        default=0,
        help="Seconds spent on each frame to simulate a slower consumer (default: 0).",
    )
    args = parser.parse_args()
    if args.queue_size <= 0:
        parser.error("--queue-size must be a positive integer")
    for name in ("target_fps", "duration", "consumer_delay"):
        value = getattr(args, name)
        if (
            not math.isfinite(value)
            or value < 0
            or (name != "consumer_delay" and value == 0)
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be finite and {'nonnegative' if name == 'consumer_delay' else 'positive'}"
            )
    return args


def _measure_memory(process: psutil.Process, started: float) -> MemorySample:
    children_rss = 0
    for child in process.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            # FFmpeg may finish between listing children and reading its memory.
            continue
    return MemorySample(
        elapsed_seconds=perf_counter() - started,
        python_rss_bytes=process.memory_info().rss,
        children_rss_bytes=children_rss,
    )


def _sample_memory(
    process: psutil.Process,
    started: float,
    stop_event: Event,
    samples: list[MemorySample],
    errors: list[Exception],
) -> None:
    try:
        while not stop_event.wait(SAMPLE_INTERVAL_SECONDS):
            samples.append(_measure_memory(process, started))
    except Exception as error:
        errors.append(error)


def _save_samples(output_dir: Path, samples: list[MemorySample]) -> None:
    with (output_dir / "memory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "elapsed_seconds",
                "python_rss_bytes",
                "children_rss_bytes",
                "total_rss_bytes",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    round(sample.elapsed_seconds, 4),
                    sample.python_rss_bytes,
                    sample.children_rss_bytes,
                    sample.total_rss_bytes,
                ]
            )


def _memory_summary(
    samples: list[MemorySample], first_frame_seconds: float | None
) -> dict[str, float]:
    active_samples = [
        sample
        for sample in samples
        if first_frame_seconds is not None
        and sample.elapsed_seconds >= first_frame_seconds
    ]
    baseline = samples[0].python_rss_bytes
    peak_total = max(sample.total_rss_bytes for sample in samples)
    values = {
        "baseline_python_rss": baseline,
        "peak_python_rss": max(sample.python_rss_bytes for sample in samples),
        "peak_ffmpeg_rss": max(sample.children_rss_bytes for sample in samples),
        "peak_combined_rss": peak_total,
        "peak_combined_increase_over_baseline": peak_total - baseline,
    }
    if active_samples:
        values["mean_combined_rss_after_first_frame"] = mean(
            sample.total_rss_bytes for sample in active_samples
        )
    return {name: round(value / BYTES_PER_MIB, 2) for name, value in values.items()}


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    source = replace(_build_video_source(args), source_id=f"benchmark_{run_id}")
    engine, session_factory = create_engine_and_session_factory(
        _load_database_url(PROJECT_DIR)
    )
    output_dir = PROJECT_DIR / "recordings" / "benchmarks" / run_id
    output_dir.mkdir(parents=True)
    process = psutil.Process()
    started = perf_counter()
    samples = [_measure_memory(process, started)]
    sampler_errors: list[Exception] = []
    sampler_stop = Event()
    sampler = Thread(
        target=_sample_memory,
        args=(process, started, sampler_stop, samples, sampler_errors),
        name="memory-sampler",
    )
    consumed_frames = 0
    first_frame_seconds: float | None = None
    frame_shape: tuple[int, ...] | None = None
    stop_requested = False
    pipeline_error: BaseException | None = None

    def consume(frame: StreamFrame) -> None:
        nonlocal consumed_frames, first_frame_seconds, frame_shape, stop_requested
        elapsed = perf_counter() - started
        if first_frame_seconds is None:
            first_frame_seconds = elapsed
            frame_shape = frame.image.shape
        if not stop_requested and elapsed >= args.duration:
            stop_requested = True
            # The runner catches this and stops input before draining the queue.
            raise KeyboardInterrupt
        if args.consumer_delay:
            sleep(args.consumer_delay)
        consumed_frames += 1

    sampler.start()
    try:
        run_pipeline(
            source,
            PROJECT_DIR / "recordings",
            session_factory=session_factory,
            target_fps=args.target_fps,
            queue_size=args.queue_size,
            frame_handler=consume,
        )
    except BaseException as error:
        pipeline_error = error
    finally:
        sampler_stop.set()
        sampler.join()
        engine.dispose()

    _save_samples(output_dir, samples)
    report = {
        "status": "failed" if pipeline_error or sampler_errors else "completed",
        "source_type": source.source_type.value,
        "source_id": source.source_id,
        "frame_shape": frame_shape,
        "target_fps": args.target_fps,
        "queue_size": args.queue_size,
        "queue_policy": "drop_oldest",
        "requested_duration_seconds": args.duration,
        "elapsed_seconds_including_drain": round(perf_counter() - started, 3),
        "consumer_delay_seconds": args.consumer_delay,
        "frames_consumed": consumed_frames,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "sample_count": len(samples),
        "memory_mib": _memory_summary(samples, first_frame_seconds),
        "notes": [
            "Baseline is Python RSS after imports and engine creation, before workers start.",
            "Measures Python and its FFmpeg child; excludes PostgreSQL, Docker and YOLO.",
            "Combined RSS can count shared memory pages more than once.",
            "The consumer counts frames; an optional delay simulates slower processing.",
            "Requested duration stops input at the next callback; draining can take longer.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Results: {output_dir}")
    if pipeline_error:
        raise pipeline_error
    if sampler_errors:
        raise RuntimeError(
            "Memory sampling failed; results are incomplete"
        ) from sampler_errors[0]


if __name__ == "__main__":
    main()
