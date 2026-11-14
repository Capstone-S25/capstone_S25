"""Run one isolated component/camera memory scenario in a fresh process."""

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from statistics import mean
from threading import Event, Lock, Thread, Timer
from time import perf_counter, sleep

import psutil

from data_pipeline.database.session import create_engine_and_session_factory
from data_pipeline.pipeline import runner
from data_pipeline.video_ingestion.recording.worker import run_recording_worker
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.worker import (
    StreamFrame,
    run_streaming_worker,
)
from main import _load_database_url

PROJECT_DIR = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
SAMPLE_INTERVAL_SECONDS = 0.2


class QueueTotals:
    """Track simultaneous backlog across all camera queues, including brief peaks."""

    def __init__(self):
        self.lock = Lock()
        self.frames = self.image_bytes = self.peak_frames = self.peak_image_bytes = 0

    def update(self, frames, image_bytes):
        with self.lock:
            self.frames += frames
            self.image_bytes += image_bytes
            self.peak_frames = max(self.peak_frames, self.frames)
            self.peak_image_bytes = max(self.peak_image_bytes, self.image_bytes)

    def snapshot(self):
        with self.lock:
            return {
                "queued_frames": self.frames,
                "queued_image_bytes": self.image_bytes,
                "queued_frames_peak": self.peak_frames,
                "queued_image_bytes_peak": self.peak_image_bytes,
            }


class MeasuredQueue(Queue):
    """An ordinary unbounded FIFO queue that also counts queued image bytes."""

    def __init__(self, totals=None):
        super().__init__(maxsize=0)
        self.totals = totals if totals is not None else QueueTotals()
        self.image_bytes = 0
        self.peak_image_bytes = 0
        self.peak_frames = 0
        self.produced = 0

    def _put(self, frame):
        # Queue calls _put/_get while holding its own mutex.
        super()._put(frame)
        self.produced += 1
        self.image_bytes += frame.image.nbytes
        self.peak_image_bytes = max(self.peak_image_bytes, self.image_bytes)
        self.peak_frames = max(self.peak_frames, self._qsize())
        self.totals.update(1, frame.image.nbytes)

    def _get(self):
        frame = super()._get()
        self.image_bytes -= frame.image.nbytes
        self.totals.update(-1, -frame.image.nbytes)
        return frame

    def snapshot(self):
        with self.mutex:
            return {
                "queued_frames": self._qsize(),
                "queued_image_bytes": self.image_bytes,
                "peak_queued_frames": self.peak_frames,
                "peak_queued_image_bytes": self.peak_image_bytes,
                "produced_frames": self.produced,
            }


@dataclass
class CameraMetrics:
    queue: MeasuredQueue
    consumed_frames: int = 0
    frame_shape: tuple[int, ...] | None = None


def _run_camera(args, source, session_factory, stop, metrics):
    try:
        if args.mode == "recording":
            run_recording_worker(
                source,
                PROJECT_DIR / "recordings",
                session_factory=session_factory,
                stop_event=stop,
            )
            return

        def consume(frame: StreamFrame):
            metrics.frame_shape = frame.image.shape
            if args.consumer_delay:
                sleep(args.consumer_delay)
            metrics.consumed_frames += 1

        state = runner._PipelineState(stop_event=stop, frames=metrics.queue)
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                if args.mode == "full":
                    runner._start_workers(
                        executor,
                        state,
                        source,
                        PROJECT_DIR / "recordings",
                        session_factory=session_factory,
                        target_fps=args.target_fps,
                        segment_seconds=5,
                        retention_hours=24,
                    )
                else:
                    state.workers.append(
                        executor.submit(
                            run_streaming_worker,
                            source,
                            metrics.queue,
                            target_fps=args.target_fps,
                            stop_event=stop,
                            producer_done=state.producer_done,
                        )
                    )
                runner._consume_frames(state, consume)
            except BaseException as error:
                runner._remember_error(state, error)
            finally:
                runner._finish_workers(state)
        if state.first_error is not None:
            raise state.first_error
    except BaseException:
        stop.set()
        raise


def _sample(process, started, metrics):
    own = process.memory_full_info()
    rss, uss = own.rss, own.uss
    ffmpeg_rss = ffmpeg_uss = 0
    children = process.children(recursive=True)
    for child in children:
        try:
            memory = child.memory_full_info()
            ffmpeg_rss += memory.rss
            ffmpeg_uss += memory.uss
        except psutil.NoSuchProcess:
            continue
    queues = (
        metrics[0].queue.totals.snapshot()
        if metrics
        else {
            "queued_frames": 0,
            "queued_image_bytes": 0,
            "queued_frames_peak": 0,
            "queued_image_bytes_peak": 0,
        }
    )
    return {
        "elapsed_seconds": perf_counter() - started,
        "python_rss_bytes": rss,
        "python_uss_bytes": uss,
        "ffmpeg_rss_bytes": ffmpeg_rss,
        "ffmpeg_uss_bytes": ffmpeg_uss,
        "app_rss_bytes": rss + ffmpeg_rss,
        "app_uss_bytes": uss + ffmpeg_uss,
        "ffmpeg_process_count": len(children),
        **queues,
        "consumed_frames": sum(camera.consumed_frames for camera in metrics),
    }


def _monitor(started, metrics, stop, rows, errors, input_stop, stop_file):
    process = psutil.Process()
    try:
        while not stop.wait(SAMPLE_INTERVAL_SECONDS):
            if stop_file.exists():
                input_stop.set()
            rows.append(_sample(process, started, metrics))
    except Exception as error:
        errors.append(error)


def _summarize(rows, args):
    active = [
        r
        for r in rows
        if args.warmup <= r["elapsed_seconds"] < args.warmup + args.duration
    ]
    if not active:
        raise RuntimeError("No samples in the measurement window")
    summary = {}
    for field in (
        "python_rss_bytes",
        "python_uss_bytes",
        "ffmpeg_rss_bytes",
        "ffmpeg_uss_bytes",
        "app_rss_bytes",
        "app_uss_bytes",
        "queued_image_bytes",
    ):
        name = field.removesuffix("_bytes")
        summary[f"{name}_mean_mib"] = mean(r[field] for r in active) / MIB
        summary[f"{name}_peak_mib"] = max(r[field] for r in rows) / MIB
    summary["queued_frames_peak"] = max(r["queued_frames_peak"] for r in rows)
    summary["queued_image_bytes_peak_mib"] = (
        max(r["queued_image_bytes_peak"] for r in rows) / MIB
    )
    summary["ffmpeg_process_count_peak"] = max(r["ffmpeg_process_count"] for r in rows)
    # Fit a simple slope to the latter half of the steady measurement window.
    late = [
        r for r in active if r["elapsed_seconds"] >= args.warmup + args.duration / 2
    ]
    if len(late) >= 2:
        x_mean = mean(r["elapsed_seconds"] for r in late)
        denominator = sum((r["elapsed_seconds"] - x_mean) ** 2 for r in late)
        for field in ("app_rss_bytes", "queued_image_bytes"):
            y_mean = mean(r[field] for r in late)
            slope = (
                sum((r["elapsed_seconds"] - x_mean) * (r[field] - y_mean) for r in late)
                / denominator
            )
            summary[field.removesuffix("_bytes") + "_late_slope_mib_per_minute"] = (
                slope * 60 / MIB
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["baseline", "recording", "streaming", "full"], required=True
    )
    parser.add_argument("--cameras", type=int, choices=[0, 1, 2, 4], required=True)
    parser.add_argument("--rtsp-url", action="append", default=[])
    parser.add_argument("--target-fps", type=float, default=2)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--consumer-delay", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.mode == "baseline") != (args.cameras == 0):
        parser.error("Only baseline mode uses zero cameras")
    urls = args.rtsp_url or ["rtsp://localhost:8554/camera_01"]
    if len(urls) != 1 and len(urls) < args.cameras:
        parser.error("Supply one shared URL or a URL for each camera")
    logging.basicConfig(level=logging.WARNING)
    args.output.mkdir(parents=True, exist_ok=True)
    engine, factory = create_engine_and_session_factory(_load_database_url(PROJECT_DIR))
    queue_totals = QueueTotals()
    metrics = [CameraMetrics(MeasuredQueue(queue_totals)) for _ in range(args.cameras)]
    stop, monitor_stop = Event(), Event()
    started = perf_counter()
    rows = [_sample(psutil.Process(), started, metrics)]
    errors = []
    monitor_errors = []
    monitor = Thread(
        target=_monitor,
        args=(
            started,
            metrics,
            monitor_stop,
            rows,
            monitor_errors,
            stop,
            args.output / "STOP",
        ),
    )
    deadline = Timer(args.warmup + args.duration, stop.set)
    monitor.start()
    deadline.start()
    try:
        if args.mode == "baseline":
            stop.wait()
        else:
            with ThreadPoolExecutor(max_workers=args.cameras) as executor:
                futures = []
                for index in range(args.cameras):
                    source = VideoSource(
                        f"bench_{args.output.parent.name}_{args.output.name}_{index}",
                        f"Benchmark camera {index + 1}",
                        SourceType.RTSP,
                        urls[0] if len(urls) == 1 else urls[index],
                    )
                    futures.append(
                        executor.submit(
                            _run_camera, args, source, factory, stop, metrics[index]
                        )
                    )
                for future in futures:
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append(error)
                        stop.set()
    finally:
        stop.set()
        deadline.cancel()
        monitor_stop.set()
        monitor.join()
        rows.append(_sample(psutil.Process(), started, metrics))
        engine.dispose()
    with (args.output / "process.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "failed" if errors or monitor_errors else "completed",
        "started_perf_counter": started,
        "mode": args.mode,
        "cameras": args.cameras,
        "target_fps": args.target_fps,
        "warmup_seconds": args.warmup,
        "measurement_seconds": args.duration,
        "consumer_delay_seconds": args.consumer_delay,
        "elapsed_seconds_including_drain": perf_counter() - started,
        "samples": len(rows),
        "baseline_python_rss_mib": rows[0]["python_rss_bytes"] / MIB,
        "cameras_detail": [
            dict(
                camera.queue.snapshot(),
                consumed_frames=camera.consumed_frames,
                frame_shape=camera.frame_shape,
            )
            for camera in metrics
        ],
        "errors": [type(error).__name__ for error in errors + monitor_errors],
        "metrics": _summarize(rows, args) if not errors and not monitor_errors else {},
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    if errors or monitor_errors:
        raise RuntimeError("Benchmark scenario failed") from (errors + monitor_errors)[
            0
        ]


if __name__ == "__main__":
    main()
