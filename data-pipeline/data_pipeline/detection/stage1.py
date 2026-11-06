"""Backend frame adapter for the production YOLO/ByteTrack runtime."""

import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

# Which VLM objectives to ask about for each Stage 1 rule.
TASKS_BY_EVENT = {
    "final_scene_review": (
        "smoking",
        "littering",
        "loitering",
        "rough_sleeping",
        "unattended_items",
    ),
    "reach_down": ("littering",),
    "pose_action": ("littering",),
    "fast_motion": ("littering",),
    "static_object_left": ("littering", "unattended_items"),
    "object_appeared": ("littering", "unattended_items"),
    "unattended_object": ("unattended_items", "littering"),
    "sustained_presence": ("loitering", "smoking", "rough_sleeping"),
    "long_presence": ("loitering", "rough_sleeping"),
    "near_camera": ("loitering",),
    "hand_to_face": ("smoking",),
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


def default_config(gate, device="cpu"):
    from vision_runtime.config import model_paths, models_root

    weights = models_root() / "detectors"
    return gate.GateConfig(
        det_weights=str(weights / "yolo26s.pt"),
        pose_weights=str(weights / "yolo26s-pose.pt"),
        rules=gate.DEFAULT_RULES + ("pose_action",),
        inview_s=60.0,
        device=device,
        crop_check_encoder="local-dir:" + str(model_paths().base),
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
    ) -> None:
        if fps <= 0 or not math.isfinite(fps):
            raise ValueError("Stage 1 fps must be positive and finite")
        if not math.isfinite(safety_scan_seconds) or safety_scan_seconds < 0:
            raise ValueError("Safety scan interval must be finite and nonnegative")
        self.safety_scan_seconds = safety_scan_seconds
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

    def process(self, frame: DecodedFrame) -> list[dict]:
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
                paths, times = self.writer.window_frames(
                    event, image, self.stream, event_t - span, event_t, tag="_" + name
                )
                event["views"][name] = {"paths": paths, "times_s": times}
            event["tasks"] = list(TASKS_BY_EVENT.get(event["event_type"], ()))
            if event["tasks"]:
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
