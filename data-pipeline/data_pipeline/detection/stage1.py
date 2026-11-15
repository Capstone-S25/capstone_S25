"""Stage 1 (YOLO pre-filter) on selected frames, producing VLM candidates.

The gate itself (YOLO26s + pose + ByteTrack and the Stage 1 rules) lives in
notebooks/stage1_gate.py. Set STAGE1_GATE_DIR to load another copy. It needs
ultralytics and opencv in the environment.
"""

import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

GATE_DIR = Path(
    os.environ.get("STAGE1_GATE_DIR", Path(__file__).resolve().parents[3] / "notebooks")
)

# Which VLM objectives to ask about for each Stage 1 rule.
TASKS_BY_EVENT = {
    # Putting something down is littering or leaving an item: often the bag itself is never detected (dark bags,
    # odd shapes), so the person's alert is the only chance to ask about an unattended item.
    "reach_down": ("littering", "unattended_items"),
    "pose_action": ("littering", "unattended_items"),
    "fast_motion": ("littering",),
    "static_object_left": ("littering", "unattended_items"),
    "object_appeared": ("littering", "unattended_items"),
    "unattended_object": ("unattended_items", "littering"),
    "sustained_presence": ("loitering", "smoking", "rough_sleeping"),
    "long_presence": ("loitering", "rough_sleeping"),
    "near_camera": ("loitering",),
    # Littering too: the person dedup can let a smoking alert stand in for a reach-down one on the same person
    # (8 ICVL held-out littering incidents caught only this way, 2026-11-10).
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
# items need about 20 seconds of history (see anomaly_eval predict-clip).
VIEW_WINDOWS_S = {"short": 4.0, "long": 20.0}


def load_gate() -> ModuleType:
    """Import notebooks/stage1_gate.py (and its stage1_pose_action helper)."""
    if str(GATE_DIR) not in sys.path:
        sys.path.insert(0, str(GATE_DIR))
    import stage1_gate

    return stage1_gate


def default_config(gate: ModuleType, device: str = "cpu", imgsz: int = 640):
    """The Stage 1 settings chosen in the notebooks/ experiments (see notebooks/README.md).

    imgsz 1280 for wide shots where people are small: far lying people are only
    detected at 1280, at roughly 3x the CPU cost."""
    return gate.GateConfig(
        det_weights=str(
            Path(
                os.environ.get(
                    "STAGE1_MODEL_DIR", GATE_DIR.parent / "data/models/stage1"
                )
            )
            / "yolo26s.pt"
        ),
        pose_weights=str(
            Path(
                os.environ.get(
                    "STAGE1_MODEL_DIR", GATE_DIR.parent / "data/models/stage1"
                )
            )
            / "yolo26s-pose.pt"
        ),
        rules=gate.DEFAULT_RULES + ("pose_action", "smoking"),
        inview_s=60.0,
        # Calibrated on the camera-disjoint split (notebooks/exp/eval/calib_chosen.json,
        # CALIB_LOG.md experiments 1-31): 90.3 % of events on the tuned cameras, 94.8 % on
        # cameras never tuned on, at 98.5 alerts/h on footage with no incident. Every change was
        # priced at an equal alert rate; the cheapest buy on the front is 7.7 alerts/h per point.
        loiter_s=30.0,
        loiter_radius_h=0.45,
        # Guards: a reach or pose cue needs a person at least 0.12 of frame height whose hand has
        # been tracked 1.5 s. Both were unset, which let tiny and half-second detections fire.
        reach_min_h=0.12,
        hand_min_age_s=1.5,
        # Crop checks are stricter than the thresholds stored with the weights (0.2).
        pose_action_thr=0.4,
        crop_check_thr=0.3,
        # Stillness rules re-report a person still present after 45 s. Previously never repeated.
        loiter_repeat_s=45.0,
        inview_repeat_s=45.0,
        # KEPT AGAINST THE CALIBRATION, deliberately. calib_chosen says None; left at 1.0 because
        # the sweep optimises 983 incidents of which only 5 are left items, so it trades a bag for
        # a litter catch without noticing. Measured both ways: 1.0 takes unattended items from 5/6
        # to 6/6 on the held-out cameras (+5 littering on fit) for +4.6 alerts/h on fit, 9.2 per
        # recall point - above the 7.7 bar, accepted for bag coverage.
        reach_refire_move_h=1.0,
        pose_action_cooldown_s=15.0,
        # Smoking: count hand raises to the face (3 in 10 s), then its own crop check.
        smoke_count_raises=True,
        smoke_min_hits=3,
        crop_check_rules=("reach_down", "pose_action", "hand_to_face"),
        item_rule_skip=("umbrella",),  # cafe parasols, not left items
        still_lying_wh=1.1,  # people lying down are short boxes: skip the height guard
        # Stillness rules measure body size by the box's longer side (lying people in any
        # pose), and a person pass at 1280 px once a second feeds them far / lying people
        # the 640 px pass misses (CHAD lying 1/4 -> 3/4, other targets unchanged, ~+30 % CPU).
        still_size_longer_side=True,
        hires_every_s=1.0,
        imgsz=imgsz,
        device=device,
        # Crop check encoder: open_clip id or "local-dir:<MobileCLIP2-S2 folder>" for
        # offline machines; unset = timm/MobileCLIP2-S2-OpenCLIP from the HF hub/cache.
        crop_check_encoder=os.environ.get("STAGE1_CROP_ENCODER") or None,
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
        for event in events:
            event["event_id"] = (
                f"{self.source_id}_{frame.session_id.hex[:8]}_{event['event_id']}"
            )
            event["source_id"] = self.source_id
            event["observed_until_s"] = t
            event["continuity_epoch"] = self.continuity_epoch
            event["timestamp_origin"] = (
                "decoder-estimated media timeline; not camera capture latency"
            )
            event["session_id"] = str(frame.session_id)
            event["segment_id"] = frame.segment_id
            # A held alert (Stage 1 crop check) carries the frame and time it fired at;
            # observed_until_s stays the release time.
            image, te = event.pop("_frame", frame.image), event.pop("_t", t)
            event["timestamp_ms"] = frame.timestamp_ms - round((t - te) * 1000)
            self.writer.save(event, image, self.stream, t=te)
            event["views"] = {}
            for name, span in VIEW_WINDOWS_S.items():
                if event["event_type"] == "periodic_review" and name == "short":
                    span = max(span, self.safety_scan_seconds)
                paths, times = self.writer.window_frames(
                    event, image, self.stream, te - span, te, tag=f"_{name}"
                )
                event["views"][name] = {"paths": paths, "times_s": times}
            event["tasks"] = list(TASKS_BY_EVENT.get(event["event_type"], ()))
            candidates.append(event)
        # Candidates are returned to the caller; the writer need not keep them.
        self.writer.manifest.clear()
        self.index += 1
        return candidates
