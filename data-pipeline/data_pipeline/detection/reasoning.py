"""Bounded FIFO candidate confirmation with resident, isolated model workers."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class ResidentClient:
    def __init__(
        self, command, repo_root, log_path, *, timeout=120.0, startup_timeout=180.0
    ):
        if not all(math.isfinite(v) and v > 0 for v in (timeout, startup_timeout)):
            raise ValueError("Worker timeouts must be positive")
        self.command = list(map(str, command))
        self.repo_root = Path(repo_root)
        self.log_path = Path(log_path)
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.process = None
        self.log = None
        self.ready = None
        self.lock = threading.Lock()

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise TimeoutError("Resident inference worker exceeded its deadline")
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Resident inference worker exited; inspect {self.log_path}"
            )
        return json.loads(line)

    def start(self):
        if self.process is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.repo_root,
            env={
                **os.environ,
                "PYTHONPATH": str(self.repo_root),
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        try:
            self.ready = self._read(self.startup_timeout)
            if self.ready.get("type") != "ready":
                raise RuntimeError(
                    self.ready.get("error", "Invalid worker startup response")
                )
        except BaseException:
            self.close()
            raise

    def classify(self, candidate):
        with self.lock:
            self.start()
            ident = uuid4().hex
            try:
                self.process.stdin.write(
                    json.dumps({"id": ident, "candidate": candidate}) + "\n"
                )
                self.process.stdin.flush()
                response = self._read(self.timeout)
                if response.get("id") != ident:
                    raise RuntimeError("Worker response identity mismatch")
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["result"]
            except (BrokenPipeError, OSError, ValueError):
                self.close()
                raise

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            if process.stdout:
                process.stdout.close()
        if self.log is not None:
            self.log.close()
            self.log = None


class CascadeClient:
    """Record classifier scores, then verify every candidate; no negative-score veto."""

    def __init__(self, classifier, reasoner):
        self.classifier, self.reasoner = classifier, reasoner

    def start(self):
        try:
            self.classifier.start()
            self.reasoner.start()
        except BaseException:
            self.close()
            raise

    def classify(self, candidate):
        started = time.perf_counter()
        classifier = None
        error = None
        try:
            classifier = self.classifier.classify(candidate)
        except Exception as exc:
            error = str(exc)
        reasoned = self.reasoner.classify(candidate)
        predictions = {}
        for task, value in reasoned["predictions"].items():
            clip = classifier["predictions"].get(task, {}) if classifier else {}
            review = bool(
                error
                or clip.get("decision") != value["decision"]
                or value["decision"] == "unknown"
            )
            predictions[task] = {
                **value,
                "classifier_decision": clip.get("decision", "error"),
                "review_required": review,
                "alert": bool(value.get("alert") and not review),
            }
        return {
            "predictions": predictions,
            "classifier": classifier,
            "classifier_error": error,
            "reasoner": reasoned,
            "seconds": time.perf_counter() - started,
            "scope": "Uncalibrated generative confirmation; every candidate is verified; only agreement may dispatch an alert.",
        }

    def close(self):
        try:
            self.reasoner.close()
        finally:
            self.classifier.close()


class CandidateService:
    """Durable candidate audit plus one FIFO consumer. Overflow is explicit, never normal."""

    def __init__(self, out_dir, client, *, max_pending=16):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.out_dir = Path(out_dir)
        self.client = client
        self.pending = queue.Queue(maxsize=max_pending)
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.thread = None
        self.closed = False
        self.fatal_error = None
        self.counts = {"submitted": 0, "completed": 0, "failed": 0, "rejected": 0}
        self.timings = []

    def _write(self, record):
        source = record["source_id"]
        if not source or Path(source).name != source or source in {".", ".."}:
            raise ValueError("Invalid candidate source_id")
        directory = self.out_dir / source
        directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, allow_nan=False) + "\n"
        with self.lock:
            with (directory / "candidate_journal.jsonl").open("a") as output:
                output.write(line)
            if record["status"] in {"completed", "error", "rejected"}:
                with (directory / "candidates.jsonl").open("a") as output:
                    output.write(line)

    def start(self):
        with self.start_lock:
            if self.thread:
                return
            self.client.start()
            self.thread = threading.Thread(
                target=self._consume, name="candidate-reasoner", daemon=True
            )
            self.thread.start()

    def submit(self, candidate):
        if self.closed:
            raise RuntimeError("Candidate service is closed")
        if self.fatal_error is not None:
            raise RuntimeError("Candidate journal worker failed") from self.fatal_error
        self.start()
        arrived = time.perf_counter()
        # Preserve an independently inspectable admission record before inference.
        self._write({**candidate, "status": "queued", "queued_at_unix_s": time.time()})
        try:
            self.pending.put_nowait((candidate, arrived))
        except queue.Full:
            self.counts["rejected"] += 1
            self._write(
                {
                    **candidate,
                    "status": "rejected",
                    "reason": "candidate_queue_full",
                    "vlm": {
                        t: {"decision": "not_run", "alert": False}
                        for t in candidate["tasks"]
                    },
                }
            )
            return False
        self.counts["submitted"] += 1
        return True

    def _consume(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                candidate, arrived = item
                started = time.perf_counter()
                record = {**candidate, "queue_seconds": started - arrived}
                try:
                    response = self.client.classify(candidate)
                    predictions = response["predictions"]
                    if set(predictions) != set(candidate["tasks"]):
                        raise ValueError("Model response omitted or added tasks")
                    if any(
                        p.get("decision")
                        not in {"anomaly", "normal", "unknown", "not_run"}
                        for p in predictions.values()
                    ):
                        raise ValueError("Invalid model decision")
                    record.update(
                        status="completed", vlm=predictions, inference=response
                    )
                    self.counts["completed"] += 1
                except Exception as exc:
                    record.update(
                        status="error",
                        error=str(exc),
                        vlm={
                            t: {
                                "decision": "unknown",
                                "reason": "inference_failed",
                                "alert": False,
                            }
                            for t in candidate["tasks"]
                        },
                    )
                    self.counts["failed"] += 1
                record["service_seconds"] = time.perf_counter() - started
                record["response_seconds"] = time.perf_counter() - arrived
                self.timings.append(
                    {
                        k: record[k]
                        for k in [
                            "queue_seconds",
                            "service_seconds",
                            "response_seconds",
                        ]
                    }
                )
                self._write(record)
                logger.info(
                    "Decision | %s | %s | %s | queue=%.3fs service=%.3fs",
                    candidate["source_id"],
                    candidate.get("event_id"),
                    {t: v["decision"] for t, v in record["vlm"].items()},
                    record["queue_seconds"],
                    record["service_seconds"],
                )
            except BaseException as exc:
                self.fatal_error = exc
                return
            finally:
                self.pending.task_done()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.thread:
                while self.pending.unfinished_tasks:
                    if self.fatal_error is not None:
                        raise RuntimeError(
                            "Candidate audit write failed"
                        ) from self.fatal_error
                    threading.Event().wait(0.02)
                if self.fatal_error is not None:
                    raise RuntimeError(
                        "Candidate audit write failed"
                    ) from self.fatal_error
                self.pending.put(None)
                self.thread.join()
        finally:
            self.client.close()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "reasoning_summary.json").write_text(
            json.dumps(
                {
                    "counts": self.counts,
                    "workers": (
                        {
                            "classifier": self.client.classifier.ready,
                            "reasoner": self.client.reasoner.ready,
                        }
                        if isinstance(self.client, CascadeClient)
                        else getattr(self.client, "ready", None)
                    ),
                    "requests": self.timings,
                    "scope": "Processing/queue measurements; no ground-truth event-quality or paced-RTSP guarantee.",
                },
                indent=2,
            )
            + "\n"
        )


class RecordOnly:
    def start(self):
        pass

    def close(self):
        pass

    def classify(self, candidate):
        return {
            "predictions": {
                t: {"decision": "not_run", "alert": False} for t in candidate["tasks"]
            }
        }


def add_arguments(parser, repo_root):
    parser.add_argument(
        "--vlm",
        choices=["none", "predict-clip", "clip", "gemma", "cascade"],
        default="none",
    )
    parser.add_argument(
        "--vlm-python", default=str(repo_root / ".venv-gemma4/bin/python")
    )
    parser.add_argument(
        "--clip-python", default=str(repo_root / ".venv-gpu/bin/python")
    )
    parser.add_argument(
        "--vlm-model",
        choices=[
            "gemma4-e2b",
            "gemma4-e4b",
            "gemma4-12b",
            "gemma4-26b-a4b",
            "gemma4-31b",
        ],
        default="gemma4-26b-a4b",
    )
    parser.add_argument(
        "--clip-model",
        choices=["siglip2-b16-224", "mobileclip2-s2"],
        default="siglip2-b16-224",
    )
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--stage1-device", default="cpu")
    parser.add_argument(
        "--safety-scan-seconds",
        type=float,
        default=10,
        help="Periodic whole-scene review interval; 0 reproduces rule-only gating",
    )
    parser.add_argument(
        "--attention", choices=["sdpa", "math", "eager"], default="sdpa"
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-format", choices=["auto", "prompt"], default="auto")
    parser.add_argument("--queue-capacity", type=int, default=16)
    parser.add_argument("--inference-timeout", type=float, default=120)


def build_service(args, repo_root, out_dir):
    def resident(backend, model, python):
        interpreter = Path(python).expanduser()
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"Model Python not found: {interpreter}; provide --{backend if backend == 'clip' else 'vlm'}-python"
            )
        command = [
            str(interpreter),
            "-u",
            "-m",
            "anomaly_eval.reasoning.worker",
            "--backend",
            backend,
            "--model",
            model,
            "--device",
            args.vlm_device,
            "--attention",
            args.attention,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output-format",
            args.output_format,
            "--runtime-dir",
            str(Path(out_dir) / "worker_logs" / (backend + "_" + model + "_runtime")),
        ]
        return ResidentClient(
            command,
            repo_root,
            Path(out_dir) / "worker_logs" / f"{backend}_{model}.log",
            timeout=args.inference_timeout,
        )

    if args.vlm == "none":
        client = RecordOnly()
    elif args.vlm in {"clip", "predict-clip"}:
        client = resident("clip", args.clip_model, args.clip_python)
    elif args.vlm == "gemma":
        client = resident("gemma", args.vlm_model, args.vlm_python)
    else:
        client = CascadeClient(
            resident("clip", args.clip_model, args.clip_python),
            resident("gemma", args.vlm_model, args.vlm_python),
        )
    return CandidateService(out_dir, client, max_pending=args.queue_capacity)
