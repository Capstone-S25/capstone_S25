"""Backend frame adapter for the production YOLO/ByteTrack runtime."""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import UUID

from vision_runtime.evidence import enabled_tasks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1Frame:
    """One frame for the tracker, with the session it belongs to.

    Attributes:
        segment_id: Recording segment of the frame, or None for live frames.
        source_id: Camera or file source identifier.
        session_id: Tracking session; a new one restarts tracking.
        frame_index: Zero-based decoded frame index.
        timestamp_ms: Approximate UTC epoch milliseconds (media time for uploads).
        image: BGR uint8 pixel array with shape (height, width, 3).
    """

    segment_id: int | None
    source_id: str
    session_id: UUID
    frame_index: int
    timestamp_ms: int
    image: Any


# Which VLM objectives to ask about for each Stage 1 rule.
TASKS_BY_EVENT = {
    "final_scene_review": (
        "smoking",
        "littering",
        "loitering",
        "rough_sleeping",
        "unattended_items",
    ),
    # Putting something down is littering or leaving an item: often the bag itself is never
    # detected (dark bags, odd shapes), so the person's alert is the only chance to ask
    # about an unattended item.
    # Smoking too, the mirror of the hand_to_face note below: the person dedup
    # lets a reach-down or pose-action alert stand in for the smoking one on the
    # same person, so the smoking question is never asked. 11 ICVL held-out
    # smoking incidents were lost exactly this way (2026-11-18).
    "reach_down": ("littering", "unattended_items", "smoking"),
    "pose_action": ("littering", "unattended_items", "smoking"),
    "fast_motion": ("littering",),
    "static_object_left": ("littering", "unattended_items"),
    "object_appeared": ("littering", "unattended_items"),
    "unattended_object": ("unattended_items", "littering"),
    "sustained_presence": ("loitering", "smoking", "rough_sleeping"),
    "long_presence": ("loitering", "rough_sleeping", "smoking"),
    "near_camera": ("loitering",),
    # Littering too: the person dedup can let a smoking alert stand in for a reach-down
    # one on the same person (8 ICVL held-out littering incidents caught only this way,
    # 2026-11-10).
    "hand_to_face": ("smoking", "littering"),
    "periodic_review": (
        "smoking",
        "littering",
        "loitering",
        "rough_sleeping",
        "unattended_items",
    ),
}

# Past-frame windows saved per candidate, in seconds before the trigger frame.
# Littering needs at least eight views of the action; loitering and unattended
# items need about 20 seconds of history under the trained input protocol.
VIEW_WINDOWS_S = {"short": 4.0, "long": 20.0}


def load_gate():
    from vision_runtime.tracking import gate

    return gate


def default_config(gate, device="cpu", imgsz=640):
    """The calibrated Stage 1 settings (feat/stage1-integration a51198d).

    imgsz 1280 for wide shots where people are small: far lying people are only
    detected at 1280, at roughly 3x the CPU cost. STAGE1_MODEL_DIR and
    STAGE1_CROP_ENCODER override the bundled detector weights and crop-check
    encoder.
    """
    from vision_runtime.config import model_paths, models_root
    from vision_runtime.tracking.crop_check import weights_for

    weights = Path(os.environ.get("STAGE1_MODEL_DIR") or models_root() / "detectors")
    # Skip a crop check whose weights are missing instead of failing to start.
    checked = ("reach_down", "pose_action", "hand_to_face")
    missing = [rule for rule in checked if not weights_for(rule).is_file()]
    if missing:
        logger.warning(
            "Stage 1 crop check skipped (no weights): %s", ", ".join(missing)
        )
    return gate.GateConfig(
        det_weights=str(weights / "yolo26s.pt"),
        pose_weights=str(weights / "yolo26s-pose.pt"),
        rules=gate.DEFAULT_RULES + ("pose_action", "smoking"),
        inview_s=60.0,
        # Calibrated on the camera-disjoint split (notebooks/exp/eval/calib_chosen.json,
        # CALIB_LOG.md experiments 1-31 on feat/stage1-integration): 90.3 % of events on
        # the tuned cameras, 94.8 % on cameras never tuned on, at 98.5 alerts/h on
        # footage with no incident. Every change was priced at an equal alert rate; the
        # cheapest buy on the front is 7.7 alerts/h per point.
        loiter_s=30.0,
        loiter_radius_h=0.45,
        # Guards: a reach or pose cue needs a person at least 0.12 of frame height whose
        # hand has been tracked 1.5 s. Both were unset, which let tiny and half-second
        # detections fire.
        reach_min_h=0.12,
        hand_min_age_s=1.5,
        # Crop checks are stricter than the thresholds stored with the weights (0.2).
        pose_action_thr=0.4,
        crop_check_thr=0.3,
        # Stillness rules re-report a person still present after 45 s.
        loiter_repeat_s=45.0,
        inview_repeat_s=45.0,
        # Left at the calibrated None. 1.0 was tried and reverted: on the held-out cameras it buys
        # one event - unattended items 5/6 to 6/6, littering and loitering unchanged - for
        # +14.1 alerts/h, i.e. 30 per recall point against a 7.7 bar. The +5 littering it gains on
        # the tuned cameras does not appear on unseen ones (CALIB_LOG.md experiment 32).
        reach_refire_move_h=None,
        pose_action_cooldown_s=15.0,
        # Smoking: count hand raises to the face (3 in 10 s), then its own crop check.
        smoke_count_raises=True,
        smoke_min_hits=3,
        crop_check_rules=tuple(rule for rule in checked if rule not in missing),
        item_rule_skip=("umbrella",),  # cafe parasols, not left items
        still_lying_wh=1.1,  # people lying down are short boxes: skip the height guard
        # Stillness rules measure body size by the box's longer side (lying people in
        # any pose), and a person pass at 1280 px once a second feeds them far / lying
        # people the 640 px pass misses (CHAD lying 1/4 -> 3/4, ~+30 % CPU).
        still_size_longer_side=True,
        hires_every_s=1.0,
        imgsz=imgsz,
        device=device,
        crop_check_encoder=os.environ.get("STAGE1_CROP_ENCODER")
        or "local-dir:" + str(model_paths().base),
    )


class Stage1Detector:
    """Run Stage 1 on one source's selected frames, across segment boundaries.

    Tracker and rule state persist between segments, so a person who stays for a
    minute is one track rather than one per 5 s segment. A new recording session
    starts fresh.

    Args:
        source_id: Camera or file source identifier.
        out_dir: Candidate crops and window frames go in out_dir/source_id.
        fps: Rate of the frames passed to process(); the tracker and rules use it.
        cfg: A stage1_gate.GateConfig; defaults to default_config().
        gate: The stage1_gate module; defaults to load_gate().
        reference_frames: Save each session's first frame as the reference scene
            (needs OpenCV); off only for tests with a fake gate.
    """

    def __init__(
        self,
        source_id: str,
        out_dir: Path,
        *,
        fps: float,
        cfg=None,
        gate: ModuleType | None = None,
        safety_scan_seconds: float = 0,
        reference_frames: bool = True,
    ) -> None:
        if fps <= 0 or not math.isfinite(fps):
            raise ValueError("Stage 1 fps must be positive and finite")
        if not math.isfinite(safety_scan_seconds) or safety_scan_seconds < 0:
            raise ValueError("Safety scan interval must be finite and nonnegative")
        self.safety_scan_seconds = safety_scan_seconds
        self.enabled_tasks = frozenset(enabled_tasks())
        self.next_review_s = None
        self.gate = gate or load_gate()
        self.cfg = cfg or default_config(self.gate)
        self.source_id = source_id
        self.fps = fps
        self.perception = self.gate.Perception(self.cfg)
        self.writer = self.gate.HandoffWriter(
            Path(out_dir) / source_id,
            buffer_s=max(*VIEW_WINDOWS_S.values(), safety_scan_seconds) + 5.0,
        )
        self.stream = SimpleNamespace(seq_id=source_id, source=source_id, paths=None)
        # The first frame after each (re)start is the camera's reference scene;
        # the ensemble classifier compares loitering/unattended windows with it.
        self.reference_dir = Path(out_dir) / source_id / "reference"
        self.reference_path: str | None = None
        self.reference_frames = reference_frames
        self.session_id: UUID | None = None
        self.last_timestamp_ms: int | None = None
        self.continuity_epoch = 0

    def _reset(self, session_id: UUID) -> None:
        self._close_pool()
        same_session = session_id == self.session_id
        previous_start = getattr(self, "start_ms", None)
        previous_index = getattr(self, "index", 0)
        self.continuity_epoch += 1
        self.perception.reset(self.fps)
        self.engine = self.gate.RuleEngine(self.cfg, self.fps)
        self.writer.start(self.stream)
        self.session_id = session_id
        self.start_ms = previous_start if same_session else None
        self.index = previous_index if same_session else 0
        self.last_timestamp_ms = None
        self.next_review_s = None
        self.reference_path = None

    def _save_reference(self, frame: Stage1Frame) -> None:
        import cv2

        self.reference_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self.reference_dir
            / f"{frame.session_id.hex[:8]}_e{self.continuity_epoch}.jpg"
        )
        if not cv2.imwrite(str(path), frame.image):
            raise OSError(f"Could not write reference frame {path}")
        self.reference_path = str(path)

    def process(self, frame: Stage1Frame) -> list[dict]:
        """Observe one selected frame and return the candidates it triggers.

        Each candidate is a Stage 1 event (event_type, roi_box, crop_path,
        context_frame_path, vlm_prompt_hint, ...) plus timestamp_ms, the tasks to
        ask the VLM about and views: {"short"|"long": {"paths", "times_s"}}.
        """
        if frame.source_id != self.source_id:
            raise ValueError("A detector cannot mix sources")
        if frame.session_id != self.session_id:
            self._reset(frame.session_id)
        elif self.last_timestamp_ms is not None:
            gap = frame.timestamp_ms - self.last_timestamp_ms
            if gap < 0:
                raise ValueError("Stage 1 timestamps decreased")
            if gap == 0:
                return []
            if gap > max(2000, 3000 / self.fps):
                self._reset(frame.session_id)
        self.last_timestamp_ms = frame.timestamp_ms
        if self.start_ms is None:
            self.start_ms = frame.timestamp_ms
        t = (frame.timestamp_ms - self.start_ms) / 1000
        if self.reference_frames and self.reference_path is None:
            self._save_reference(frame)
        obs = self.perception.observe(frame.image, self.index, t)
        self.writer.observe(frame.image, t, self.index)
        candidates = []
        self.last_frame = frame
        events = list(self.engine.step(obs, frame.image))
        if self.safety_scan_seconds:
            if self.next_review_s is None:
                self.next_review_s = t + self.safety_scan_seconds
            if t + 1e-6 >= self.next_review_s:
                h, w = frame.image.shape[:2]
                events.append(
                    {
                        "event_id": f"periodic_review_scene_f{self.index}",
                        "event_type": "periodic_review",
                        "candidate_origin": "scheduled_scene_review",
                        "frame_index": self.index,
                        "timestamp_s": round(t, 2),
                        "roi_box": [0, 0, w, h],
                        "evidence_window_s": [max(0, t - 20), t],
                        "clip_window_s": [max(0, t - 20), t],
                        "vlm_prompt_hint": "Check the observed scene; no tracked target or ownership is established.",
                    }
                )
                while self.next_review_s <= t + 1e-6:
                    self.next_review_s += self.safety_scan_seconds
        candidates = self._events(events, frame, t)
        # Candidates are returned to the caller; the writer need not keep them.
        self.writer.manifest.clear()
        self.index += 1
        return candidates

    def _events(self, events, frame, observed_t):
        candidates = []
        for original in events:
            tasks = [
                task for task in TASKS_BY_EVENT.get(original["event_type"], ())
                if task in self.enabled_tasks
            ]
            if not tasks:
                continue
            event = dict(original)
            image = event.pop("_frame", frame.image)
            event_t = float(event.pop("_t", event["timestamp_s"]))
            event["trigger_timestamp_s"] = event_t
            event["available_at_s"] = observed_t
            event["observed_until_s"] = observed_t
            event["timestamp_s"] = event_t
            event["event_id"] = (
                f"{self.source_id}_{frame.session_id.hex[:8]}_{event['event_id']}"
            )
            event.update(
                source_id=self.source_id,
                session_id=str(frame.session_id),
                segment_id=frame.segment_id,
                timestamp_ms=self.start_ms + round(event_t * 1000),
                continuity_epoch=self.continuity_epoch,
                reference_frame_path=self.reference_path,
                timestamp_origin="decoder-estimated media time; decision availability recorded separately",
            )
            self.writer.save(event, image, self.stream, t=event_t)
            event["views"] = {}
            for name, span in VIEW_WINDOWS_S.items():
                if name == "short" and event["event_type"] in {
                    "periodic_review",
                    "final_scene_review",
                }:
                    span = max(span, self.safety_scan_seconds)
                if name == "long":
                    # Buffered frames are 1/buffer_fps apart, so the first one
                    # inside [t - 20, t] can be up to that much late and the
                    # view then misses the 19.5 s the classifier requires.
                    # Reach back one step (at most 0.5 s, the trained bound).
                    span += min(0.5, 1.0 / self.writer.buffer_fps)
                paths, times = self.writer.window_frames(
                    event, image, self.stream, event_t - span, event_t, tag="_" + name
                )
                event["views"][name] = {"paths": paths, "times_s": times}
            event["tasks"] = tasks
            candidates.append(event)
        self.writer.manifest.clear()
        return candidates

    def finish(self):
        frame = getattr(self, "last_frame", None)
        if frame is None or getattr(self, "finished", False):
            return []
        self.finished = True
        t = (frame.timestamp_ms - self.start_ms) / 1000
        try:
            events = self.engine.flush()
            # The end-of-clip whole-frame look is a scheduled scene review just
            # like periodic_review, so one switch governs both. On the 148 test
            # videos with littering, loitering and smoking enabled it caught no
            # incident a tracked rule missed, and 86 of the 88 windows that
            # reached review matched no label. With unattended_items enabled it
            # is the only route to 6 of the 9 labelled items, but as a
            # whole-frame window with no tracked target or established
            # ownership, which the strict standard does not count (3/9).
            if self.safety_scan_seconds:
                h, w = frame.image.shape[:2]
                events.append(
                    {
                        "event_id": "final_scene",
                        "event_type": "final_scene_review",
                        "candidate_origin": "scheduled_scene_review",
                        "timestamp_s": t,
                        "frame_index": self.index - 1,
                        "roi_box": [0, 0, w, h],
                    }
                )
            return self._events(events, frame, t)
        finally:
            self._close_pool()

    def _close_pool(self):
        engine = getattr(self, "engine", None)
        check = getattr(engine, "crop_check", None)
        if check:
            check["pool"].shutdown(wait=True, cancel_futures=True)
