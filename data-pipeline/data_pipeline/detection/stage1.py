"""Stage 1 (YOLO pre-filter) on selected frames, producing VLM candidates.

The gate itself (YOLO26s + pose + ByteTrack and the Stage 1 rules) lives in
notebooks/stage1_gate.py. Set STAGE1_GATE_DIR to load another copy. It needs
ultralytics and opencv in the environment.
"""

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


def default_config(gate: ModuleType, device: str = "cpu"):
    """The Stage 1 settings chosen in notebooks/STAGE1_PROGRESS.md."""
    return gate.GateConfig(
        det_weights=str(GATE_DIR / "yolo26s.pt"),
        pose_weights=str(GATE_DIR / "yolo26s-pose.pt"),
        rules=gate.DEFAULT_RULES + ("pose_action",),
        inview_s=60.0,
        device=device,
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
    ) -> None:
        self.gate = gate or load_gate()
        self.cfg = cfg or default_config(self.gate)
        self.source_id = source_id
        self.fps = fps
        self.perception = self.gate.Perception(self.cfg)
        self.writer = self.gate.HandoffWriter(
            Path(out_dir) / source_id,
            buffer_s=max(VIEW_WINDOWS_S.values()) + 5.0,
        )
        self.stream = SimpleNamespace(seq_id=source_id, source=source_id, paths=None)
        self.session_id: UUID | None = None

    def _reset(self, session_id: UUID) -> None:
        self.perception.reset(self.fps)
        self.engine = self.gate.RuleEngine(self.cfg, self.fps)
        self.writer.start(self.stream)
        self.session_id = session_id
        self.start_ms: int | None = None
        self.index = 0

    def process(self, frame: DecodedFrame) -> list[dict]:
        """Observe one selected frame and return the candidates it triggers.

        Each candidate is a Stage 1 event (event_type, roi_box, crop_path,
        context_frame_path, vlm_prompt_hint, ...) plus timestamp_ms, the tasks to
        ask the VLM about and views: {"short"|"long": {"paths", "times_s"}}.
        """
        if frame.session_id != self.session_id:
            self._reset(frame.session_id)
        if self.start_ms is None:
            self.start_ms = frame.timestamp_ms
        t = (frame.timestamp_ms - self.start_ms) / 1000
        obs = self.perception.observe(frame.image, self.index, t)
        self.writer.observe(frame.image, t, self.index)
        candidates = []
        for event in self.engine.step(obs):
            event["event_id"] = (
                f"{self.source_id}_{frame.session_id.hex[:8]}_{event['event_id']}"
            )
            event["source_id"] = self.source_id
            event["session_id"] = str(frame.session_id)
            event["segment_id"] = frame.segment_id
            event["timestamp_ms"] = frame.timestamp_ms
            self.writer.save(event, frame.image, self.stream, t=t)
            event["views"] = {}
            for name, span in VIEW_WINDOWS_S.items():
                paths, times = self.writer.window_frames(
                    event, frame.image, self.stream, t - span, t, tag=f"_{name}"
                )
                event["views"][name] = {"paths": paths, "times_s": times}
            event["tasks"] = list(TASKS_BY_EVENT.get(event["event_type"], ()))
            candidates.append(event)
        # Candidates are returned to the caller; the writer need not keep them.
        self.writer.manifest.clear()
        self.index += 1
        return candidates
