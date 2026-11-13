"""Bounded screening and confirmation queues, with separate provisional updates.

PostgreSQL schedules recorded segments. These queues schedule AI event work and
do not replace it. Event journals are durable; in-flight AI jobs are not leases
and are not automatically replayed after a process crash.
"""

import copy
import json
import logging
import queue
import threading
import time
from collections import Counter
from pathlib import Path

from .evidence import unknown

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self, directory, cleanup=None, on_write=None):
        """on_write(candidate, row) runs after each journaled transition.

        It lets an application mirror events elsewhere (for example into the
        incidents database). Its failures are logged and never stop screening.
        """
        self.on_write = on_write
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {}
        self.pending = 0
        self.active = Counter()
        self.released = False
        self.cleanup = cleanup
        self.error = None

    def write(self, candidate, status, **fields):
        ident = candidate["event_id"]
        with self.lock:
            if self.released:
                return
            previous = self.records.get(ident, {})
            row = {
                **previous,
                "event_id": ident,
                "timestamp_s": candidate["timestamp_s"],
                "available_at_s": candidate.get(
                    "available_at_s", candidate["timestamp_s"]
                ),
                "kind": candidate["event_type"],
                "entity": candidate.get("entity_id"),
                "source_id": candidate.get("source_id"),
                "status": status,
                "updated_unix_s": time.time(),
                **fields,
            }
            # Write before exposing the state to readers.
            try:
                with (self.directory / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            except Exception as exc:
                self.error = exc
                raise
            self.records[ident] = row
        if self.on_write is not None:
            try:
                self.on_write(candidate, copy.deepcopy(row))
            except Exception:
                logger.exception("Event listener failed for %s", ident)

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(
                    "The AI event journal could not be written"
                ) from self.error
            return copy.deepcopy(list(self.records.values()))

    def retain(self, event_id=None):
        with self.lock:
            self.pending += 1
            self.active[event_id] += 1

    def done(self, event_id=None):
        with self.lock:
            self.pending -= 1
            self.active[event_id] -= 1
            if not self.active[event_id]:
                del self.active[event_id]
            self._clean()

    def prune(self, keep):
        """Compact a live session's rolling journal, preserving in-flight evidence.

        Upload/camera journals remain append-only unless this method is called.
        Return retired event IDs so their caller can delete event-owned files.
        """
        if keep < 1:
            raise ValueError("Retained event count must be positive")
        with self.lock:
            terminal = {
                "screened",
                "reviewed",
                "not_run",
                "screening_error",
                "confirmation_error",
                "confirmation_not_run",
            }
            removable = [
                ident
                for ident, row in list(self.records.items())[:-keep]
                if not self.active.get(ident) and row["status"] in terminal
            ]
            if not removable:
                return []
            retired = set(removable)
            remaining = {k: v for k, v in self.records.items() if k not in retired}
            path = self.directory / "events.jsonl"
            temporary = path.with_suffix(".jsonl.compacting")
            try:
                with temporary.open("w") as stream:
                    for row in remaining.values():
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                temporary.replace(path)
            except Exception as exc:
                self.error = exc
                raise
            finally:
                temporary.unlink(missing_ok=True)
            self.records = remaining
            return removable

    def release(self):
        with self.lock:
            self.released = True
            self._clean()

    def _clean(self):
        if self.released and self.pending == 0 and self.cleanup:
            cleanup, self.cleanup = self.cleanup, None
            cleanup()


def consensus(provisional, reviewed):
    output = {}
    if set(reviewed) != set(provisional):
        raise ValueError("Confirmation omitted or added tasks")
    for task, value in reviewed.items():
        if value.get("decision") not in {"anomaly", "normal", "unknown"}:
            raise ValueError("Invalid confirmation decision")
        disagreement = (
            value["decision"] == "unknown"
            or value["decision"] != provisional[task]["decision"]
        )
        output[task] = {
            **value,
            "classifier_decision": provisional[task]["decision"],
            "review_required": disagreement,
            "alert": bool(value.get("alert") and not disagreement),
        }
    return output


class ConfirmationQueue:
    def __init__(self, backend, capacity=8, max_age=120):
        if capacity < 1 or max_age <= 0:
            raise ValueError("Queue capacity and age must be positive")
        self.backend, self.max_age = backend, max_age
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = False
        self.thread = threading.Thread(
            target=self._consume, daemon=True, name="gemma-review"
        )
        self.thread.start()

    def submit(self, candidate, provisional, store):
        if self.closed:
            raise RuntimeError("Confirmation queue is closed")
        store.retain(candidate["event_id"])
        try:
            # Publish before the worker can finish and overwrite this transition.
            store.write(
                candidate,
                "confirmation_pending",
                predictions=provisional["predictions"],
            )
            self.queue.put_nowait(
                (copy.deepcopy(candidate), provisional, store, time.monotonic())
            )
        except queue.Full:
            try:
                store.write(
                    candidate, "confirmation_not_run", reason="confirmation_queue_full"
                )
            finally:
                store.done(candidate["event_id"])
            return False
        except BaseException:
            store.done(candidate["event_id"])
            raise
        return True

    def _consume(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                candidate, provisional, store, queued = item
                try:
                    age = time.monotonic() - queued
                    if store.released:
                        continue
                    if age > self.max_age:
                        store.write(
                            candidate,
                            "confirmation_not_run",
                            reason="confirmation_deadline_expired",
                        )
                        continue
                    store.write(candidate, "confirming", confirmation_queue_seconds=age)
                    reviewed = self.backend.classify(candidate)
                    result = consensus(
                        provisional["predictions"], reviewed["predictions"]
                    )
                    store.write(
                        candidate,
                        "reviewed",
                        confirmation=result,
                        confirmation_details=reviewed,
                    )
                except Exception as exc:
                    logger.exception("Gemma review failed")
                    store.write(
                        candidate,
                        "confirmation_error",
                        reason=str(exc),
                        confirmation={
                            t: unknown("Confirmation failed")
                            for t in candidate["tasks"]
                        },
                    )
                finally:
                    store.done(candidate["event_id"])
            except BaseException:
                logger.exception("Could not journal confirmation state")
            finally:
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.join()
            self.queue.put(None)
            self.thread.join()


class ScreeningQueue:
    """Bounded screening work for one event store.

    A full queue records new windows as not run (cameras), replaces the waiting
    window with the newest one (latest_only, live preview), or makes the
    producer wait for space (wait_for_space, uploads with no real-time deadline).
    """

    def __init__(
        self,
        classifier,
        store,
        confirmation=None,
        capacity=16,
        latest_only=False,
        wait_for_space=False,
    ):
        if capacity < 1:
            raise ValueError("Queue capacity must be positive")
        if latest_only and wait_for_space:
            raise ValueError("Choose latest-only replacement or waiting, not both")
        self.classifier, self.store, self.confirmation = classifier, store, confirmation
        self.queue = queue.Queue(maxsize=capacity)
        self.latest_only = latest_only
        self.wait_for_space = wait_for_space
        self.closed = False
        self.fatal_error = None
        self.thread = threading.Thread(
            target=self._consume, daemon=True, name="clip-screening"
        )
        self.thread.start()

    def submit(self, candidate):
        if self.closed:
            raise RuntimeError("Screening queue is closed")
        if self.fatal_error:
            raise RuntimeError("Screening event journal failed") from self.fatal_error
        self.store.write(
            candidate,
            "queued",
            view_times_s={k: v["times_s"] for k, v in candidate["views"].items()},
        )
        self.store.retain(candidate["event_id"])
        try:
            self._enqueue(copy.deepcopy(candidate))
        except queue.Full:
            try:
                self.store.write(
                    candidate,
                    "not_run",
                    reason="screening_queue_full",
                    predictions={
                        t: unknown("Screening queue full") for t in candidate["tasks"]
                    },
                )
            finally:
                self.store.done(candidate["event_id"])
            return False
        except BaseException:
            self.store.done(candidate["event_id"])
            raise
        return True

    def _enqueue(self, candidate):
        if self.latest_only and self.queue.full():
            try:
                previous = self.queue.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    self.store.write(
                        previous,
                        "not_run",
                        reason="newer_live_evidence_available",
                        predictions={
                            t: unknown("Superseded by newer live evidence")
                            for t in previous["tasks"]
                        },
                    )
                finally:
                    self.store.done(previous["event_id"])
                    self.queue.task_done()
        if self.wait_for_space:
            # The worker always drains (including after release or a journal
            # failure), so this wait ends when the current window is screened.
            self.queue.put(candidate)
        else:
            self.queue.put_nowait(candidate)

    def _consume(self):
        while True:
            candidate = self.queue.get()
            try:
                if candidate is None:
                    return
                if self.store.released:
                    continue
                if self.fatal_error:
                    continue
                result = self.classifier.classify(candidate)
                if set(result["predictions"]) != set(candidate["tasks"]):
                    raise ValueError("Classifier omitted or added tasks")
                self.store.write(
                    candidate,
                    "screened",
                    predictions=result["predictions"],
                    inference_seconds=result["seconds"],
                )
                if self.confirmation is not None:
                    self.confirmation.submit(candidate, result, self.store)
            except Exception as exc:
                logger.exception("Classifier failed")
                try:
                    self.store.write(
                        candidate,
                        "screening_error",
                        reason=str(exc),
                        predictions={
                            t: unknown("Classifier failed") for t in candidate["tasks"]
                        },
                    )
                except BaseException as journal_error:
                    self.fatal_error = journal_error
            finally:
                try:
                    if candidate is not None:
                        self.store.done(candidate["event_id"])
                except Exception as exc:
                    self.fatal_error = exc
                    logger.exception("Could not release screening evidence")
                finally:
                    self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.join()
            self.queue.put(None)
            self.thread.join()
            if self.fatal_error:
                raise RuntimeError(
                    "Screening event journal failed"
                ) from self.fatal_error
