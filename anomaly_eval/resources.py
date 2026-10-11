"""Limit inference threads and pace sequential CPU work without importing torch."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from anomaly_eval.paths import process_environment


@dataclass(frozen=True)
class CPUSettings:
    """Resource limits on the host CPU; these do not emulate another processor."""

    threads: int = 2
    duty_cycle: float = 1.0
    max_samples_per_second: float | None = None
    cpu_cores: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError("threads must be at least 1")
        if not math.isfinite(self.duty_cycle) or not 0 < self.duty_cycle <= 1:
            raise ValueError("duty-cycle must be in (0, 1]")
        rate = self.max_samples_per_second
        if rate is not None and (not math.isfinite(rate) or rate <= 0):
            raise ValueError("max-samples-per-second must be positive")
        if any(core < 0 for core in self.cpu_cores):
            raise ValueError("CPU core IDs cannot be negative")

    def configure_environment(self) -> None:
        """Call before importing numerical libraries or spawning workers."""
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ):
            os.environ[name] = str(self.threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def configure_worker(self) -> None:
        if self.cpu_cores:
            if not hasattr(os, "sched_setaffinity"):
                raise ValueError("--cpu-cores is only supported on Linux")
            allowed = os.sched_getaffinity(0)
            if not set(self.cpu_cores) <= allowed:
                raise ValueError(f"CPU cores must be within {sorted(allowed)}")
            os.sched_setaffinity(0, self.cpu_cores)
        import torch

        torch.set_num_threads(self.threads)
        torch.set_num_interop_threads(1)


class Throttler:
    """Sleep after each operation to limit duty cycle and start-to-start rate.

    This is cooperative pacing, not an OS-enforced CPU utilization quota. An
    individual forward pass can use all configured threads until it completes.
    """

    def __init__(self, settings: CPUSettings):
        self.settings = settings
        self.total_sleep_seconds = 0.0

    @contextmanager
    def operation(self) -> Iterator[None]:
        start = time.perf_counter()
        yield
        active = time.perf_counter() - start
        target = active / self.settings.duty_cycle
        if self.settings.max_samples_per_second is not None:
            target = max(target, 1 / self.settings.max_samples_per_second)
        delay = max(0.0, target - active)
        if delay:
            time.sleep(delay)
            self.total_sleep_seconds += delay


def add_cpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threads", type=int, default=2, help="CPU inference threads (default: 2)"
    )
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=1.0,
        help="Active/sleep ratio in (0, 1]; default 1 adds no pacing sleeps",
    )
    parser.add_argument(
        "--max-samples-per-second", type=float, help="Optional maximum operation rate"
    )
    parser.add_argument(
        "--cpu-cores",
        help="Optional Linux CPU affinity, e.g. 0,1; IDs must be available",
    )


def settings_from_args(args: argparse.Namespace) -> CPUSettings:
    cores = (
        tuple(int(core.strip()) for core in args.cpu_cores.split(","))
        if args.cpu_cores
        else ()
    )
    return CPUSettings(
        args.threads, args.duty_cycle, args.max_samples_per_second, cores
    )


class ResourceMonitor:
    """Sample worker RSS during loading, reference preparation, and inference."""

    def __init__(self, max_rss_mb: float | None):
        import psutil

        self.process = psutil.Process()
        self.max_rss_mb = max_rss_mb
        self.peak_rss_mb = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stopped.is_set():
            self.peak_rss_mb = max(
                self.peak_rss_mb, self.process.memory_info().rss / 1024**2
            )
            self.stopped.wait(0.05)

    def check(self) -> None:
        rss = self.process.memory_info().rss / 1024**2
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        if self.max_rss_mb is not None and self.peak_rss_mb > self.max_rss_mb:
            raise MemoryError(f"Worker exceeded {self.max_rss_mb:g} MiB RSS")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stopped.set()
        self.thread.join()


def run_stage(command, output, timeout, memory):
    """Watch total RSS of a stage and its descendants, preserving logs on failure."""
    import psutil

    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    reason = None
    peak = 0
    env = process_environment()
    env["CUDA_VISIBLE_DEVICES"] = ""
    with output.open("w") as f:
        proc = subprocess.Popen(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=os.name != "nt",
        )
        monitor = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try:
                    rss = (
                        sum(
                            p.memory_info().rss
                            for p in [monitor, *monitor.children(recursive=True)]
                            if p.is_running()
                        )
                        / 1024**2
                    )
                    peak = max(peak, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.monotonic() - start > timeout:
                    reason = "stage timeout"
                if memory and peak > memory:
                    reason = "stage process-tree memory limit"
                if reason:
                    if os.name != "nt":
                        import signal

                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        for child in monitor.children(recursive=True):
                            child.terminate()
                        proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    break
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                if os.name != "nt":
                    import signal

                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    for child in monitor.children(recursive=True):
                        child.terminate()
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
    return {
        "command": command,
        "returncode": proc.wait(),
        "seconds": time.monotonic() - start,
        "peak_process_tree_rss_mb": peak,
        "status": "error" if reason or proc.returncode else "ok",
        "error": reason,
        "log": str(output),
    }
