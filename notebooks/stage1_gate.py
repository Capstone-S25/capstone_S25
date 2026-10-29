"""Stage 1 of the YOLO -> VLM surveillance pipeline.

Perception (YOLO tracking, pose, static-foreground) is kept separate from the
rule engine: perception is the expensive part, the rules are the part you tune.
A benchmark can run perception once, cache the Observations, and replay the
rules under many configs in seconds (see `replay_rules`).
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

PERSON = 0
COCO_ITEMS = {24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase", 39: "bottle", 41: "cup"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_WRIST, R_WRIST = 9, 10
L_KNEE, R_KNEE = 13, 14
KP_CONF = 0.3

ALL_RULES = ("loiter", "speed", "smoking", "reach_down", "litter", "unattended")
# Off by default:
#   "litter" (static-object detection): could not be validated (littering clips are ~3 s long) and
#            fired ~15-130 times/min on normal footage.
#   "smoking" (pose hand-to-face): near chance on low-resolution CCTV and the largest source of
#            false alarms; smoking candidates come from sustained presence ("loiter") instead.
DEFAULT_RULES = ("loiter", "in_view", "speed", "reach_down", "unattended")


@dataclass
class GateConfig:
    det_weights: str = "yolo26n.pt"
    pose_weights: str | None = "yolo26n-pose.pt"
    imgsz: int = 640
    device: str = "cpu"                 # "cpu" for deployment timing; "mps"/"cuda" only to speed up offline runs
    det_conf: float = 0.10              # detections below new_track_conf only extend existing tracks
    person_conf: float = 0.40          # needed once to confirm a person track, not on every frame
    item_classes: tuple[int, ...] = tuple(COCO_ITEMS)
    rules: tuple[str, ...] = DEFAULT_RULES
    pose_every_n: int = 1

    tracker: str = "bytetrack"          # "bytetrack" | "botsort" | "botsort_reid"
    track_buffer_s: float = 4.0
    new_track_conf: float = 0.25

    stitch: bool = True
    stitch_max_gap_s: float = 8.0
    stitch_base_dist_h: float = 0.75
    stitch_walk_h_per_s: float = 1.0
    stitch_min_similarity: float = 0.55

    clip_pad_s: float = 2.5

    loiter_s: float = 20.0
    loiter_radius_h: float = 0.6
    loiter_release_h: float = 0.9
    loiter_repeat_s: float = 30.0       # re-send while the person stays (e.g. starts smoking later)

    # In view for a long time (wandering / pacing), moving or not. Should exceed how long people normally take
    # to pass through this camera: calibrate_inview() on its normal footage. Gaps up to the re-link window
    # (stitch_max_gap_s) do not restart the timer.
    inview_s: float = 120.0
    inview_area_h: float | None = 4.0   # only if they stayed within this many body heights of one area (None = anywhere)
    inview_repeat_s: float = 60.0

    speed_h_per_s: float = 1.2
    speed_window_s: float = 0.6

    hand_face_h: float = 0.20
    smoke_window_s: float = 10.0
    smoke_min_hits: int = 2
    smoke_cooldown_s: float = 30.0

    reach_margin_h: float = -0.15        # wrist within 0.15 body-heights above the knee counts
    reach_cooldown_s: float = 5.0
    reach_min_hits: int = 1             # low hand in at least this many frames ...
    reach_window_s: float = 0.6         # ... within this window (stretched to 2 frames at low fps)
    pose_min_h_px: float = 0.0          # pose rules skip people smaller than this (0 = off; tested 40-60 px, no gain)
    reach_skip_border: bool = True      # skip people cut off by the frame edge (their keypoints are unreliable)
    reach_kp_conf: float = 0.5          # keypoint confidence needed for the wrist and knee (0.3 let walkers through)
    reach_bend_ratio: float | None = None  # also require bending: box height <= this x the person's usual height

    bg_width: int = 480
    bg_long_s: float = 30.0
    bg_short_s: float = 1.0
    litter_static_s: float = 1.0
    litter_min_area: float = 1e-4       # fractions of frame area
    litter_max_area: float = 0.01
    litter_owner_lookback_s: float = 4.0
    litter_owner_dist_h: float = 1.0
    litter_person_margin: float = 0.25  # blobs inside a person box grown by this fraction are ignored
    litter_max_aspect: float = 4.0
    litter_min_novelty: float = 15.0    # mean grey-level difference from the long-term background
    litter_max_change: float = 8.0      # mean grey-level change since the blob became static

    unattended_s: float = 10.0
    owner_leave_h: float = 2.0
    object_grace_s: float = 10.0
    item_warmup_s: float = 2.0          # objects already there in the first seconds are part of the scene
    item_min_frames: int = 3            # an object must be detected this many times before it counts
    carry_margin: float = 0.3           # an item within a person's box widened by this fraction is being carried
    appear_min_conf: float = 0.35
    appear_classes: tuple[str, ...] = ("backpack", "handbag", "suitcase", "bottle", "cup")  # umbrellas: railings look like them

    # Per-camera prior: a person much taller (closer to the camera) than people usually are in this camera.
    # Off until calibrated with calibrate_near_camera() on that camera's normal footage.
    near_camera_px: float | None = None
    near_camera_cooldown_s: float = 10.0

    # One candidate per person: drop a new candidate if the same person (or an overlapping box, e.g. a split
    # detection) was already sent this recently, whichever rule fired.
    person_dedup_s: float = 2.5

    @property
    def uses_bg(self) -> bool:
        return "litter" in self.rules

    @property
    def uses_pose(self) -> bool:
        return bool(self.pose_weights) and bool({"smoking", "reach_down"} & set(self.rules))


# --------------------------------------------------------------------------- sequences

@dataclass
class Frame:
    index: int
    t: float
    image: np.ndarray
    path: str | None = None


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


class VideoSequence:
    def __init__(self, path, stride: int = 1, max_frames: int | None = None, seq_id: str | None = None):
        self.path = str(path)
        self.source = self.path
        self.seq_id = seq_id or Path(path).stem
        self.stride = max(1, stride)
        cap = cv2.VideoCapture(self.path)
        self.native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.total_frames = min(total, max_frames) if max_frames else total
        self.fps = self.native_fps / self.stride
        self.n_frames = math.ceil(self.total_frames / self.stride)
        self.duration_s = self.total_frames / self.native_fps

    def __len__(self):
        return self.n_frames

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        try:
            for i in range(self.total_frames):
                ok, img = cap.read()
                if not ok:
                    break
                if i % self.stride == 0:
                    yield Frame(i, i / self.native_fps, img)
        finally:
            cap.release()

    def paths_in_window(self, t0: float, t1: float) -> list[str]:
        return []


class ImageSequence:
    """Ordered image frames treated as a video. Timestamps default to index / fps."""

    def __init__(self, paths, fps: float | None = None, timestamps=None, seq_id: str | None = None, stride: int = 1):
        paths = [str(p) for p in paths]
        if timestamps is None:
            fps = fps or 10.0
            timestamps = [i / fps for i in range(len(paths))]
        else:
            timestamps = [float(x) - float(timestamps[0]) for x in timestamps]
            if fps is None:
                diffs = np.diff(timestamps)
                fps = 1.0 / float(np.median(diffs)) if len(diffs) else 1.0
        keep = range(0, len(paths), max(1, stride))
        self.paths = [paths[i] for i in keep]
        self.indices = list(keep)
        self.timestamps = [timestamps[i] for i in keep]
        self.fps = fps / max(1, stride)
        self.native_fps = fps
        self.seq_id = seq_id or (Path(paths[0]).parent.name if paths else "empty")
        self.source = str(Path(paths[0]).parent) if paths else ""
        self.n_frames = len(self.paths)
        self.duration_s = (self.timestamps[-1] + 1.0 / fps) if self.paths else 0.0

    def __len__(self):
        return self.n_frames

    def __iter__(self) -> Iterator[Frame]:
        for idx, t, p in zip(self.indices, self.timestamps, self.paths):
            img = cv2.imread(p)
            if img is not None:
                yield Frame(idx, t, img, p)

    def paths_in_window(self, t0: float, t1: float) -> list[str]:
        return [p for p, t in zip(self.paths, self.timestamps) if t0 <= t <= t1]


def list_images(folder) -> list[str]:
    return sorted((str(p) for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def split_on_scene_cuts(paths, thresh: float = 25.0) -> list[list[str]]:
    """Split an image list wherever consecutive frames differ by more than `thresh`
    (mean absolute grey-level difference at 160x90). Some datasets splice several
    clips into one folder; tracking and background models must restart at each cut."""
    segments, prev = [], None
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if g is None:
            continue
        g = cv2.resize(g, (160, 90)).astype(np.float32)
        if prev is None or float(np.mean(np.abs(g - prev))) > thresh:
            segments.append([])
        segments[-1].append(str(p))
        prev = g
    return segments


def load_sequence(src, fps: float = 10.0, stride: int = 1, max_frames: int | None = None):
    """Video file, image folder, single image, or list of image paths."""
    if isinstance(src, (list, tuple)):
        return ImageSequence(src, fps=fps, stride=stride)
    src = Path(src)
    if src.is_dir():
        paths = list_images(src)
        if max_frames:
            paths = paths[:max_frames]
        return ImageSequence(paths, fps=fps, stride=stride, seq_id=src.name)
    if src.suffix.lower() in VIDEO_EXTS:
        return VideoSequence(src, stride=stride, max_frames=max_frames)
    if src.suffix.lower() in IMAGE_EXTS:
        return ImageSequence([src], fps=1.0, seq_id=src.stem)
    raise ValueError(f"Unsupported source: {src}")


# --------------------------------------------------------------------------- geometry helpers

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def intersection_over(a, b) -> float:
    """Fraction of box `a` covered by box `b`."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def touches_border(box, frame_hw, margin: float = 0.01) -> bool:
    h, w = frame_hw
    return box[0] <= margin * w or box[1] <= margin * h or box[2] >= (1 - margin) * w or box[3] >= (1 - margin) * h


def expand_box(b, frac):
    px, py = (b[2] - b[0]) * frac, (b[3] - b[1]) * frac
    return [b[0] - px, b[1] - py, b[2] + px, b[3] + py]


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def foot(b):
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def appearance_hist(frame, box):
    """HSV colour histogram of upper and lower body separately (shirt vs trousers)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    x1, x2 = int(max(0, x1 + 0.15 * bw)), int(min(w, x2 - 0.15 * bw))
    y1, y2 = int(max(0, y1)), int(min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mid = hsv.shape[0] // 2
    parts = []
    for part in (hsv[:mid], hsv[mid:]):
        hist = cv2.calcHist([part], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        parts.append(hist.ravel())
    return np.concatenate(parts).astype(np.float32)


def hist_similarity(a, b) -> float:
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


# --------------------------------------------------------------------------- perception

@dataclass
class Observation:
    index: int
    t: float
    frame_hw: tuple[int, int]
    persons: list[dict]
    items: list[dict]
    blobs: list[dict]
    latency_ms: dict[str, float]
    path: str | None = None


def tracker_args(cfg: GateConfig, fps: float) -> dict:
    # Older Ultralytics versions scale track_buffer by frame_rate/30; pinning frame_rate to 30
    # makes track_buffer mean "frames" in every version (a 1 fps source would otherwise round
    # frame_rate to 0 and disable the buffer).
    botsort = cfg.tracker.startswith("botsort")
    args = {
        "tracker_type": "botsort" if botsort else "bytetrack",
        "track_high_thresh": cfg.new_track_conf,
        "track_low_thresh": 0.1,
        "new_track_thresh": cfg.new_track_conf,
        "track_buffer": max(1, int(round(cfg.track_buffer_s * fps))),
        "match_thresh": 0.8,
        "fuse_score": True,
        "frame_rate": 30,
    }
    if botsort:
        args.update({
            "gmc_method": "none",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": cfg.tracker == "botsort_reid",
            "model": "yolo26n-cls.pt",
            "device": "cpu",
        })
    return args


class StaticForeground:
    """Dual background model: pixels that are foreground for a slow (long-term) model
    but already background for a fast (short-term) model are *new and stationary*:
    dropped litter, a set-down bag. Plain MOG2 only finds moving pixels."""

    def __init__(self, cfg: GateConfig, fps: float, frame_hw):
        h, w = frame_hw
        self.scale = min(1.0, cfg.bg_width / w)
        self.size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        self.long = cv2.createBackgroundSubtractorMOG2(
            history=max(2, int(cfg.bg_long_s * fps)), varThreshold=25, detectShadows=True)
        self.short = cv2.createBackgroundSubtractorMOG2(
            history=max(2, int(cfg.bg_short_s * fps)), varThreshold=25, detectShadows=True)
        self.age = np.zeros((self.size[1], self.size[0]), np.float32)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_area = cfg.litter_min_area * 0.5
        self.max_area = cfg.litter_max_area * 2.0
        self.history: deque = deque()
        self.keep_s = 3.0 * max(cfg.litter_static_s, 1.0)
        self.t = 0.0

    def _gray_at(self, t: float):
        return min(self.history, key=lambda h: abs(h[0] - t))[1]

    def apply(self, frame, dt: float) -> list[dict]:
        small = cv2.GaussianBlur(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), (3, 3), 0)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.t += dt
        self.history.append((self.t, gray))
        while self.history[0][0] < self.t - self.keep_s:
            self.history.popleft()
        fg_long = self.long.apply(small)
        fg_short = self.short.apply(small)
        static = ((fg_long == 255) & (fg_short != 255)).astype(np.uint8)
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, self.kernel)
        self.age = np.where(static > 0, self.age + dt, 0.0).astype(np.float32)
        mask = (self.age > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total = float(self.size[0] * self.size[1])
        blobs = []
        background = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / total
            if not (self.min_area <= frac <= self.max_area):
                continue
            region = labels[y:y + bh, x:x + bw] == i
            age = float(self.age[y:y + bh, x:x + bw][region].max())
            if background is None:
                background = cv2.cvtColor(self.long.getBackgroundImage(), cv2.COLOR_BGR2GRAY).astype(np.float32)
            now = gray[y:y + bh, x:x + bw][region]
            # novelty: how different the blob is from the long-term background (a new object is)
            # change: how much the blob's pixels changed since it first became static (an object
            # lying on the ground does not change; a person's edge flickering in front of a door does)
            novelty = float(np.mean(np.abs(now - background[y:y + bh, x:x + bw][region])))
            change = float(np.mean(np.abs(now - self._gray_at(self.t - age + dt)[y:y + bh, x:x + bw][region])))
            s = self.scale
            blobs.append({"box": [x / s, y / s, (x + bw) / s, (y + bh) / s], "area_frac": frac, "age_s": age,
                          "novelty": novelty, "change": change})
        return blobs


class Perception:
    def __init__(self, cfg: GateConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.det = YOLO(cfg.det_weights)
        self.pose = YOLO(cfg.pose_weights) if cfg.uses_pose else None

    def reset(self, fps: float):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace

        self.fps = fps
        args = tracker_args(self.cfg, fps)
        self.tracker = TRACKER_MAP[args["tracker_type"]](args=IterableSimpleNamespace(**args))
        self.tracker.reset()
        self.bg = None
        self.confirmed: set[int] = set()
        self.n = 0
        self.last_t = None

    def observe(self, frame, index: int, t: float, path: str | None = None) -> Observation:
        cfg = self.cfg
        lat = {}
        t0 = time.perf_counter()
        res = self.det.predict(frame, imgsz=cfg.imgsz, conf=cfg.det_conf, device=cfg.device,
                               classes=[PERSON, *cfg.item_classes], verbose=False)[0]
        lat["det_ms"] = (time.perf_counter() - t0) * 1000
        self.tracker.update(res.boxes.cpu().numpy(), frame)
        # model.track() only returns *confirmed* tracks, and ByteTrack confirms a new track on
        # its second match. At low fps or on short clips most people never get confirmed, so
        # every track updated in this frame is used, confirmed or not.
        live = [s for s in self.tracker.tracked_stracks if s.frame_id == self.tracker.frame_id]
        persons, items = [], []
        for s in live:
            box, tid, cf, c = s.xyxy.tolist(), int(s.track_id), float(s.score), int(s.cls)
            if c == PERSON:
                if cf >= cfg.person_conf:
                    self.confirmed.add(tid)
                if tid in self.confirmed:
                    persons.append({"raw_id": tid, "box": box, "conf": cf})
            else:
                items.append({"raw_id": tid, "box": box, "conf": cf, "cls": COCO_ITEMS.get(c, res.names[c])})

        for p in persons:
            crowded = any(iou(p["box"], q["box"]) > 0.15 for q in persons if q is not p)
            p["hist"] = None if crowded else appearance_hist(frame, p["box"])
            p["kpts"] = None

        if self.pose is not None and persons and self.n % max(1, cfg.pose_every_n) == 0:
            t1 = time.perf_counter()
            pr = self.pose.predict(frame, imgsz=cfg.imgsz, conf=0.25, device=cfg.device, verbose=False)[0]
            lat["pose_ms"] = (time.perf_counter() - t1) * 1000
            if pr.keypoints is not None and pr.boxes is not None and len(pr.boxes):
                pboxes = pr.boxes.xyxy.cpu().numpy().tolist()
                kpts = pr.keypoints.data.cpu().numpy()
                for p in persons:
                    scores = [iou(p["box"], pb) for pb in pboxes]
                    j = int(np.argmax(scores))
                    if scores[j] >= 0.4:
                        p["kpts"] = kpts[j]

        blobs = []
        if cfg.uses_bg:
            t2 = time.perf_counter()
            if self.bg is None:
                self.bg = StaticForeground(cfg, self.fps, frame.shape[:2])
            dt = (t - self.last_t) if self.last_t is not None else 1.0 / self.fps
            blobs = self.bg.apply(frame, dt)
            lat["bg_ms"] = (time.perf_counter() - t2) * 1000

        lat["total_ms"] = (time.perf_counter() - t0) * 1000
        self.n += 1
        self.last_t = t
        return Observation(index, t, frame.shape[:2], persons, items, blobs, lat, path)


# --------------------------------------------------------------------------- occlusion: track stitching

class TrackStitcher:
    """Re-links a newly born track id to a recently lost one when position, size and
    clothing colour agree, so a person who is occluded for longer than the tracker
    buffer keeps one identity (and the loitering timer does not restart)."""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.alias: dict[int, int] = {}
        self.tracks: dict[int, dict] = {}
        self.n_stitched = 0

    def assign(self, persons: list[dict], t: float) -> list[int]:
        cids: list[int | None] = [None] * len(persons)
        used = set()
        for i, p in enumerate(persons):
            if p["raw_id"] in self.alias:
                cids[i] = self.alias[p["raw_id"]]
                used.add(cids[i])
        for i, p in enumerate(persons):
            if cids[i] is not None:
                continue
            c = self._match_lost(p, t, used) if self.cfg.stitch else None
            if c is None:
                c = p["raw_id"]
            else:
                self.n_stitched += 1
            self.alias[p["raw_id"]] = c
            cids[i] = c
            used.add(c)
        for p, c in zip(persons, cids):
            tr = self.tracks.setdefault(c, {"hist": None})
            tr.update(last_t=t, foot=foot(p["box"]), h=max(1.0, p["box"][3] - p["box"][1]))
            if p.get("hist") is not None:
                tr["hist"] = p["hist"] if tr["hist"] is None else 0.8 * tr["hist"] + 0.2 * p["hist"]
        horizon = 2 * self.cfg.stitch_max_gap_s
        for c in [c for c, tr in self.tracks.items() if t - tr["last_t"] > horizon]:
            del self.tracks[c]
        return cids

    def _match_lost(self, p, t, used):
        cfg = self.cfg
        if p.get("hist") is None:
            return None
        ph = max(1.0, p["box"][3] - p["box"][1])
        pf = foot(p["box"])
        best, best_score = None, -1e9
        for c, tr in self.tracks.items():
            gap = t - tr["last_t"]
            if c in used or gap <= 0 or gap > cfg.stitch_max_gap_s or tr["hist"] is None:
                continue
            ratio = ph / tr["h"]
            if not 0.6 <= ratio <= 1.67:
                continue
            dist = math.hypot(pf[0] - tr["foot"][0], pf[1] - tr["foot"][1]) / max(ph, tr["h"])
            if dist > cfg.stitch_base_dist_h + cfg.stitch_walk_h_per_s * gap:
                continue
            sim = hist_similarity(p["hist"], tr["hist"])
            if sim < cfg.stitch_min_similarity:
                continue
            score = sim - 0.05 * dist
            if score > best_score:
                best, best_score = c, score
        return best


# --------------------------------------------------------------------------- rule engine

HINTS = {
    "sustained_presence": "Person stayed in a small area for a long time. Decide: loitering, rough sleeping, smoking, or normal waiting.",
    "fast_motion": "Person moving unusually fast. Check for running, chasing, fleeing or fighting.",
    "hand_to_face": "Person repeatedly raised a hand to the face. Check for smoking (cigarette or vape at mouth, smoke) versus eating, drinking or a phone call.",
    "reach_down": "Person reached toward the ground. Check whether they discarded litter or set down an item.",
    "static_object_left": "A new stationary object appeared where a person just was. Check for littering or a dropped item.",
    "unattended_object": "Item has been away from its owner for a while. Check whether it is unattended.",
    "object_appeared": "A bag or object appeared on its own, away from people (thrown or dropped?). Check what happened and who it belongs to.",
    "near_camera": "Person is unusually close to the camera or in an unusual spot for this camera. Check what they are doing.",
    "long_presence": "Person has been in view for a long time, staying around one area. Check for loitering or other suspicious behaviour.",
}


def _kp(k, i):
    return (float(k[i, 0]), float(k[i, 1])) if k[i, 2] >= KP_CONF else None


class RuleEngine:
    def __init__(self, cfg: GateConfig, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.stitcher = TrackStitcher(cfg)
        self.people: dict[int, dict] = {}
        self.items: list[dict] = []
        self.flagged_blobs: list[list[float]] = []
        self.last_cids: list[int] = []
        self.sent: deque = deque()
        self.t0: float | None = None
        self.known_item_spots: list[list[float]] = []

    def step(self, obs: Observation) -> list[dict]:
        cfg, t = self.cfg, obs.t
        if self.t0 is None:
            self.t0 = t
        cids = self.stitcher.assign(obs.persons, t)
        self.last_cids = cids
        persons = list(zip(cids, obs.persons))
        events = []
        for cid, p in persons:
            st = self._state(cid, p, t)
            if "loiter" in cfg.rules:
                events += self._loiter(cid, st, p, obs)
            if "in_view" in cfg.rules and cfg.inview_s:
                events += self._inview(cid, st, p, obs)
            if "speed" in cfg.rules:
                events += self._speed(cid, st, p, obs)
            if p.get("kpts") is not None:
                if "smoking" in cfg.rules:
                    events += self._smoking(cid, st, p, obs)
                if "reach_down" in cfg.rules:
                    events += self._reach(cid, st, p, obs)
            if "near_camera" in cfg.rules and cfg.near_camera_px:
                events += self._near_camera(cid, st, p, obs)
        if "unattended" in cfg.rules:
            events += self._unattended(obs, persons)
        if "litter" in cfg.rules:
            events += self._litter(obs, persons)
        horizon = max(cfg.loiter_s, cfg.stitch_max_gap_s, cfg.litter_owner_lookback_s + cfg.litter_static_s) + 2
        for cid in [c for c, s in self.people.items() if t - s["last_t"] > horizon]:
            del self.people[cid]
        return self._dedup(events, t)

    def _dedup(self, events, t):
        while self.sent and t - self.sent[0][0] > self.cfg.person_dedup_s:
            self.sent.popleft()
        kept = []
        for e in events:
            box = e["roi_box"]
            if any(ent == e["entity_id"] or iou(box, b) >= 0.5 or
                   max(intersection_over(box, b), intersection_over(b, box)) >= 0.7 for _, ent, b in self.sent):
                continue
            self.sent.append((t, e["entity_id"], box))
            kept.append(e)
        return kept

    def _state(self, cid, p, t):
        st = self.people.get(cid)
        if st is None:
            st = self.people[cid] = {"first_t": t, "hist": deque(), "loiter_flagged": None, "speed_armed": True,
                                     "smoke_hits": deque(), "last_smoke": -1e9, "last_reach": -1e9, "last_near": -1e9,
                                     "view_pts": deque(), "last_inview": None}
        st["last_t"] = t
        st["box"] = p["box"]
        fx, fy = foot(p["box"])
        st["hist"].append((t, fx, fy, max(1.0, p["box"][3] - p["box"][1]), p["box"], p["raw_id"]))
        keep = max(self.cfg.loiter_s, self.cfg.litter_owner_lookback_s + self.cfg.litter_static_s) + 1
        while st["hist"] and st["hist"][0][0] < t - keep:
            st["hist"].popleft()
        return st

    def _event(self, etype, entity, obs, roi, **extra):
        t, pad = obs.t, self.cfg.clip_pad_s
        return {
            "event_id": f"{etype}_{entity}_f{obs.index}",
            "event_type": etype,
            "entity_id": str(entity),
            "frame_index": obs.index,
            "timestamp_s": round(t, 2),
            "clip_window_s": [round(max(0.0, t - pad), 2), round(t + pad, 2)],
            "roi_box": [int(v) for v in roi],
            "vlm_prompt_hint": HINTS[etype],
            **extra,
        }

    def _loiter(self, cid, st, p, obs):
        cfg, t = self.cfg, obs.t
        pts = [h for h in st["hist"] if h[0] >= t - cfg.loiter_s]
        # The window must be covered by observations, allowing for the frame spacing (a 1 fps
        # source has no frame exactly loiter_s ago).
        slack = 1.5 / self.fps
        if t - st["first_t"] < cfg.loiter_s or not pts or t - pts[0][0] < cfg.loiter_s - slack:
            return []
        xy = np.array([(h[1], h[2]) for h in pts])
        radius = float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))) / float(np.median([h[3] for h in pts]))
        if radius >= cfg.loiter_release_h:
            st["loiter_flagged"] = None
        if radius < cfg.loiter_radius_h and (st["loiter_flagged"] is None or t - st["loiter_flagged"] >= cfg.loiter_repeat_s):
            st["loiter_flagged"] = t
            return [self._event("sustained_presence", f"person{cid}", obs, p["box"],
                                evidence_window_s=[round(t - cfg.loiter_s, 2), round(t, 2)],
                                metrics={"norm_radius": round(radius, 3)})]
        return []

    def _speed(self, cid, st, p, obs):
        cfg, t = self.cfg, obs.t
        window = max(cfg.speed_window_s, 3.0 / self.fps)
        # Only the current tracker id's own, gap-free points: a re-link or an occlusion gap is a
        # position jump, not motion.
        pts = []
        for h in reversed(st["hist"]):
            if h[0] < t - window or h[5] != p["raw_id"] or (pts and pts[-1][0] - h[0] > 2.5 / self.fps):
                break
            pts.append(h)
        pts.reverse()
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.5 * window:
            return []
        steps = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) / b[3] for a, b in zip(pts, pts[1:])]
        v = float(np.median(steps))
        if v >= cfg.speed_h_per_s and st["speed_armed"]:
            st["speed_armed"] = False
            return [self._event("fast_motion", f"person{cid}", obs, p["box"], metrics={"speed_h_per_s": round(v, 2)})]
        if v < 0.6 * cfg.speed_h_per_s:
            st["speed_armed"] = True
        return []

    def _smoking(self, cid, st, p, obs):
        cfg, t, k = self.cfg, obs.t, p["kpts"]
        face_pts = [_kp(k, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)]
        face = face_pts[0] or (tuple(np.mean([f for f in face_pts if f], axis=0)) if any(face_pts) else None)
        wrists = [w for w in (_kp(k, L_WRIST), _kp(k, R_WRIST)) if w]
        if face is None or not wrists:
            return []
        h = max(1.0, p["box"][3] - p["box"][1])
        d = min(math.hypot(w[0] - face[0], w[1] - face[1]) for w in wrists) / h
        if d <= cfg.hand_face_h:
            st["smoke_hits"].append(t)
        while st["smoke_hits"] and st["smoke_hits"][0] < t - cfg.smoke_window_s:
            st["smoke_hits"].popleft()
        if len(st["smoke_hits"]) >= cfg.smoke_min_hits and t - st["last_smoke"] >= cfg.smoke_cooldown_s:
            st["last_smoke"] = t
            return [self._event("hand_to_face", f"person{cid}", obs, p["box"], metrics={"wrist_face_h": round(d, 3)})]
        return []

    def _reach(self, cid, st, p, obs):
        cfg, t, k = self.cfg, obs.t, p["kpts"]
        kc = cfg.reach_kp_conf
        knees = [(float(k[i, 0]), float(k[i, 1])) for i in (L_KNEE, R_KNEE) if k[i, 2] >= kc]
        wrists = [(float(k[i, 0]), float(k[i, 1])) for i in (L_WRIST, R_WRIST) if k[i, 2] >= kc]
        if not knees or not wrists:
            return []
        h = max(1.0, p["box"][3] - p["box"][1])
        if cfg.reach_bend_ratio is not None:
            usual = float(np.percentile([q[3] for q in st["hist"]], 75))
            if len(st["hist"]) < 3 or h > cfg.reach_bend_ratio * usual:
                return []
        if h < cfg.pose_min_h_px or (cfg.reach_skip_border and touches_border(p["box"], obs.frame_hw)):
            return []
        knee_y = float(np.mean([q[1] for q in knees]))
        hits = st.setdefault("reach_hits", deque())
        if max(w[1] for w in wrists) >= knee_y + cfg.reach_margin_h * h:
            hits.append(t)
        window = max(cfg.reach_window_s, 1.5 / self.fps)
        while hits and hits[0] < t - window:
            hits.popleft()
        if len(hits) >= cfg.reach_min_hits and t - st["last_reach"] >= cfg.reach_cooldown_s:
            st["last_reach"] = t
            return [self._event("reach_down", f"person{cid}", obs, p["box"])]
        return []

    def _inview(self, cid, st, p, obs):
        cfg, t = self.cfg, obs.t
        pts = st["view_pts"]
        if not pts or t - pts[-1][0] >= 1.0:
            fx, fy = foot(p["box"])
            pts.append((t, fx, fy, max(1.0, p["box"][3] - p["box"][1])))
        while pts and pts[0][0] < t - cfg.inview_s:
            pts.popleft()
        if t - st["first_t"] < cfg.inview_s:
            return []
        if st["last_inview"] is not None and t - st["last_inview"] < cfg.inview_repeat_s:
            return []
        metrics = {"time_in_view_s": round(t - st["first_t"], 1)}
        if cfg.inview_area_h is not None:
            xy = np.array([(q[1], q[2]) for q in pts])
            radius = float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))) / float(np.median([q[3] for q in pts]))
            if radius > cfg.inview_area_h:
                return []
            metrics["area_radius_h"] = round(radius, 2)
        st["last_inview"] = t
        return [self._event("long_presence", f"person{cid}", obs, p["box"],
                            evidence_window_s=[round(t - cfg.inview_s, 2), round(t, 2)], metrics=metrics)]

    def _near_camera(self, cid, st, p, obs):
        if p["box"][3] - p["box"][1] <= self.cfg.near_camera_px or obs.t - st["last_near"] < self.cfg.near_camera_cooldown_s:
            return []
        st["last_near"] = obs.t
        return [self._event("near_camera", f"person{cid}", obs, p["box"])]

    def _nearest_person(self, box, persons, max_h):
        c = center(box)
        best, best_d = None, 1e9
        for cid, p in persons:
            h = max(1.0, p["box"][3] - p["box"][1])
            d = min(math.hypot(c[0] - q[0], c[1] - q[1]) for q in (foot(p["box"]), center(p["box"]))) / h
            if d <= max_h and d < best_d:
                best, best_d = cid, d
        return best

    def _carrier(self, box, persons):
        """The person holding or wearing an item: the one whose box contains most of the item's box."""
        best, best_cover = None, 0.5
        for cid, p in persons:
            cover = intersection_over(box, expand_box(p["box"], self.cfg.carry_margin))
            if cover >= best_cover:
                best, best_cover = cid, cover
        return best

    def _unattended(self, obs, persons):
        cfg, t = self.cfg, obs.t
        visible = {cid: p for cid, p in persons}
        events = []
        for it in obs.items:
            entry = next((e for e in self.items if e["raw_id"] == it["raw_id"] or iou(e["box"], it["box"]) >= 0.3), None)
            if entry is None:
                known = any(iou(k, it["box"]) >= 0.3 for k in self.known_item_spots)
                late = t - self.t0 >= cfg.item_warmup_s and not known
                entry = {"raw_id": it["raw_id"], "cls": it["cls"], "origin": it["box"], "first_t": t, "seen": 0,
                         "owner": None, "owner_near_t": None, "flagged": False, "nobody_since": None,
                         "late": late, "appeared_alone": late and self._carrier(it["box"], persons) is None,
                         "appear_flagged": False, "moved": False, "max_conf": 0.0}
                self.items.append(entry)
            entry.update(box=it["box"], last_t=t, raw_id=it["raw_id"])
            entry["seen"] += 1
            entry["max_conf"] = max(entry["max_conf"], it["conf"])
            if not any(iou(k, it["box"]) >= 0.3 for k in self.known_item_spots):
                self.known_item_spots = (self.known_item_spots + [it["box"]])[-200:]
            carrier = self._carrier(it["box"], persons)
            if carrier is not None:
                # someone is holding or wearing it: attended, and it is theirs
                entry.update(origin=it["box"], owner=carrier, owner_near_t=t, flagged=False, nobody_since=None,
                             appeared_alone=False)
                continue
            diag = math.hypot(it["box"][2] - it["box"][0], it["box"][3] - it["box"][1])
            oc, nc = center(entry["origin"]), center(it["box"])
            if math.hypot(nc[0] - oc[0], nc[1] - oc[1]) > max(10.0, 0.5 * diag):
                entry.update(origin=it["box"], flagged=False, nobody_since=None, moved=True)
            if entry["owner"] is None:
                entry["owner"] = self._nearest_person(it["box"], persons, cfg.owner_leave_h)
                if entry["owner"] is not None:
                    entry["owner_near_t"] = t
            elif entry["owner"] in visible and self._nearest_person(it["box"], [(entry["owner"], visible[entry["owner"]])], cfg.owner_leave_h) is not None:
                entry["owner_near_t"] = t
            if self._nearest_person(it["box"], persons, cfg.owner_leave_h) is None:
                entry["nobody_since"] = entry["nobody_since"] if entry["nobody_since"] is not None else t
            else:
                entry["nobody_since"] = None
            if entry["seen"] < cfg.item_min_frames:
                continue
            # a landed (thrown / dropped) object stays put; carried objects move with their person
            if (entry["appeared_alone"] and not entry["appear_flagged"] and not entry["moved"]
                    and entry["cls"] in cfg.appear_classes and entry["max_conf"] >= cfg.appear_min_conf):
                entry["appear_flagged"] = True
                near = self._nearest_person(it["box"], persons, 4.0)
                roi = union_box(it["box"], visible[near]["box"]) if near is not None else it["box"]
                events.append(self._event("object_appeared", f"{it['cls']}{it['raw_id']}", obs, roi, item_class=it["cls"]))
            owner_gone = entry["owner"] is not None and t - entry["owner_near_t"] >= cfg.unattended_s
            ownerless = (entry["owner"] is None and entry["late"] and entry["nobody_since"] is not None
                         and t - entry["nobody_since"] >= cfg.unattended_s)
            if (owner_gone or ownerless) and not entry["flagged"]:
                entry["flagged"] = True
                events.append(self._event("unattended_object", f"{it['cls']}{it['raw_id']}", obs, it["box"],
                                          owner=f"person{entry['owner']}" if entry["owner"] is not None else None,
                                          item_class=it["cls"]))
        self.items = [e for e in self.items if t - e["last_t"] <= cfg.object_grace_s]
        return events

    def _litter(self, obs, persons):
        cfg, t = self.cfg, obs.t
        fh, fw = obs.frame_hw
        margin = 0.01 * max(fh, fw)
        near_people = [expand_box(p["box"], cfg.litter_person_margin) for _, p in persons]
        events = []
        for b in obs.blobs:
            if b["age_s"] < cfg.litter_static_s or not (cfg.litter_min_area <= b["area_frac"] <= cfg.litter_max_area):
                continue
            if b.get("novelty", 1e9) < cfg.litter_min_novelty or b.get("change", 0.0) > cfg.litter_max_change:
                continue
            box = b["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if max(bw, bh) > cfg.litter_max_aspect * max(1.0, min(bw, bh)):
                continue
            if box[0] <= margin or box[1] <= margin or box[2] >= fw - margin or box[3] >= fh - margin:
                continue
            # Anything still next to a person is their own body, shadow or bag; litter is only
            # reported once the person has moved away from it.
            if any(intersection_over(box, q) > 0 for q in near_people):
                continue
            if any(iou(box, f) > 0.2 or intersection_over(box, f) > 0.5 for f in self.flagged_blobs):
                continue
            bc = center(box)
            owner, owner_box, owner_t = None, None, -1e9
            for cid, st in self.people.items():
                for (ht, fx, fy, h, pbox, _) in st["hist"]:
                    if ht < t - cfg.litter_static_s - cfg.litter_owner_lookback_s or ht <= owner_t:
                        continue
                    pc = center(pbox)
                    if min(math.hypot(bc[0] - fx, bc[1] - fy), math.hypot(bc[0] - pc[0], bc[1] - pc[1])) / h <= cfg.litter_owner_dist_h:
                        owner, owner_box, owner_t = cid, pbox, ht
            if owner is None:
                continue
            self.flagged_blobs.append(box)
            events.append(self._event("static_object_left", f"blob{len(self.flagged_blobs)}", obs, union_box(box, owner_box),
                                      object_box=[int(v) for v in box], owner=f"person{owner}"))
        return events


# --------------------------------------------------------------------------- running

@dataclass
class RunResult:
    seq_id: str
    source: str
    fps: float
    n_frames: int
    duration_s: float
    events: list[dict]
    observations: list[Observation]
    wall_s: float
    n_stitched: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean([o.latency_ms["total_ms"] for o in self.observations])) if self.observations else 0.0


class Stage1Gate:
    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.perception = Perception(self.cfg)

    def run(self, seq, writer: "HandoffWriter | None" = None,
            on_frame: Callable | None = None, keep_observations: bool = True) -> RunResult:
        self.perception.reset(seq.fps)
        engine = RuleEngine(self.cfg, seq.fps)
        events, observations = [], []
        if writer is not None:
            writer.start(seq)
        t0 = time.perf_counter()
        for fr in seq:
            obs = self.perception.observe(fr.image, fr.index, fr.t, fr.path)
            if writer is not None:
                writer.observe(fr.image, fr.t, fr.index)
            evs = engine.step(obs)
            for e in evs:
                e["sequence_id"] = seq.seq_id
                e["source"] = seq.source
                e["event_id"] = f"{seq.seq_id}_{e['event_id']}".replace("/", "-").replace("#", "-")
                if writer is not None:
                    writer.save(e, fr.image, seq, t=fr.t)
            events += evs
            if keep_observations:
                observations.append(obs)
            if on_frame is not None:
                on_frame(fr, obs, evs, engine)
        return RunResult(seq.seq_id, seq.source, seq.fps, len(observations), seq.duration_s, events,
                         observations, time.perf_counter() - t0, engine.stitcher.n_stitched)


def calibrate_inview(results, percentile: float = 99.0, floor_s: float = 60.0, cfg: GateConfig | None = None) -> float:
    """in_view threshold (s) for one camera: longer than `percentile`% of people stay in view in its normal footage
    (with the same re-linking the engine uses), and never shorter than floor_s."""
    cfg = cfg or GateConfig()
    durations = []
    for r in results:
        eng = RuleEngine(cfg, r.fps)
        first, last = {}, {}
        for o in r.observations:
            eng.step(o)
            for cid in eng.last_cids:
                first.setdefault(cid, o.t)
                last[cid] = o.t
        durations += [last[c] - first[c] for c in first]
    return max(floor_s, float(np.percentile(durations, percentile))) if durations else floor_s


def calibrate_near_camera(results, percentile: float = 99.0) -> float:
    """Person-height threshold (px) for the near_camera rule, from one camera's normal footage (RunResults)."""
    heights = [p["box"][3] - p["box"][1] for r in results for o in r.observations for p in o.persons]
    return float(np.percentile(heights, percentile))


def replay_rules(result: RunResult, cfg: GateConfig) -> tuple[list[dict], int]:
    """Re-run only the rule engine on cached observations (no YOLO). Returns (events, n_stitched).
    Perception-side settings (weights, imgsz, tracker, bg model) are fixed by the original run."""
    engine = RuleEngine(cfg, result.fps)
    events = []
    for obs in result.observations:
        for e in engine.step(obs):
            e["sequence_id"] = result.seq_id
            events.append(e)
    return events, engine.stitcher.n_stitched


# --------------------------------------------------------------------------- handoff + summaries

class HandoffWriter:
    """Writes what the VLM sees for each candidate: a padded crop, an annotated context frame and a short
    window of earlier frames ending at the trigger frame.

    The window only uses frames up to the trigger time, so it works on a live stream. Image sequences
    reference their existing files. Video frames are not on disk, so the writer keeps a small buffer of
    recent frames (sampled at `buffer_fps`, JPEG-compressed, at most `buffer_s` old) and saves the
    selected ones. Events with an `evidence_window_s` (loitering, long presence) use that span, clamped
    to the buffer; others use the `pre_s` seconds before the trigger.
    """

    def __init__(self, out_dir, pad_ratio: float = 0.4, max_window_frames: int = 8,
                 buffer_fps: float = 2.0, buffer_s: float = 30.0, pre_s: float = 2.5, jpeg_quality: int = 90):
        self.out_dir = Path(out_dir)
        self.crops = self.out_dir / "crops"
        self.context = self.out_dir / "annotated_frames"
        self.windows = self.out_dir / "window_frames"
        for d in (self.crops, self.context, self.windows):
            d.mkdir(parents=True, exist_ok=True)
        self.pad_ratio = pad_ratio
        self.max_window_frames = max(1, max_window_frames)
        self.buffer_fps = buffer_fps
        self.buffer_s = buffer_s
        self.pre_s = pre_s
        self.jpeg_quality = jpeg_quality
        self.manifest: list[dict] = []
        self.buffer: deque = deque()
        self.use_buffer = True

    def start(self, seq) -> None:
        """Call at the start of each sequence so frames never leak between sequences."""
        self.buffer.clear()
        self.use_buffer = not getattr(seq, "paths", None)

    def observe(self, frame, t: float, index: int) -> None:
        """Offer every frame; a sampled subset is kept for later windows (videos only)."""
        if not self.use_buffer:
            return
        if self.buffer and t - self.buffer[-1][0] < 1.0 / self.buffer_fps - 1e-6:
            return
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            self.buffer.append((t, index, jpg))
        while self.buffer and self.buffer[0][0] < t - self.buffer_s:
            self.buffer.popleft()

    def _window(self, event: dict, t: float) -> tuple[float, float]:
        start = event["evidence_window_s"][0] if "evidence_window_s" in event else t - self.pre_s
        return max(0.0, start, t - self.buffer_s), t

    @staticmethod
    def _spread(items: list, n: int) -> list:
        """At most n items, evenly spread, always keeping the last one (the trigger frame)."""
        if len(items) <= n:
            return items
        keep = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
        return [items[i] for i in keep]

    def _window_frames(self, event: dict, frame, seq, t: float) -> tuple[list[str], list[float]]:
        w0, t = self._window(event, t)
        eps = 1e-6
        paths = getattr(seq, "paths", None)
        if paths:
            # Image sequences: reference the files that are already on disk.
            pairs = [(ts, p) for p, ts in zip(seq.paths, seq.timestamps) if w0 - eps <= ts <= t + eps]
            pairs = self._spread(pairs, self.max_window_frames)
            return [p for _, p in pairs], [round(ts, 2) for ts, _ in pairs]
        # Video: buffered earlier frames plus the trigger frame itself.
        earlier = [(bt, idx, jpg) for bt, idx, jpg in self.buffer if w0 - eps <= bt < t - eps]
        chosen = self._spread([*earlier, (t, event["frame_index"], None)], self.max_window_frames)
        out_paths, times = [], []
        for k, (bt, idx, jpg) in enumerate(chosen):
            p = self.windows / f"{event['event_id']}_w{k:02d}_f{idx}.jpg"
            if jpg is None:
                cv2.imwrite(str(p), frame)
            else:
                p.write_bytes(jpg.tobytes())
            out_paths.append(str(p))
            times.append(round(bt, 2))
        return out_paths, times

    def save(self, event: dict, frame, seq, t: float | None = None) -> None:
        """`t` is the trigger frame's exact time (event timestamps are rounded)."""
        t = float(event["timestamp_s"]) if t is None else float(t)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = event["roi_box"]
        px, py = (x2 - x1) * self.pad_ratio, (y2 - y1) * self.pad_ratio
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        crop_path = self.crops / f"{event['event_id']}_crop.jpg"
        cv2.imwrite(str(crop_path), frame[cy1:cy2, cx1:cx2])
        ctx = frame.copy()
        cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 3)
        ctx_path = self.context / f"{event['event_id']}_context.jpg"
        cv2.imwrite(str(ctx_path), ctx)
        event["crop_box"] = [cx1, cy1, cx2, cy2]
        event["crop_path"] = str(crop_path)
        event["context_frame_path"] = str(ctx_path)
        paths, times = self._window_frames(event, frame, seq, t)
        event["window_frame_paths"] = paths
        event["window_frame_times_s"] = times
        self.manifest.append(event)

    def write_manifest(self, name: str = "candidates_manifest.json") -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(self.manifest, indent=2))
        return path


def merge_windows(windows) -> list[list[float]]:
    merged = []
    for a, b in sorted(windows):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def workload_summary(result: RunResult, events: list[dict] | None = None) -> dict:
    events = result.events if events is None else events
    merged = merge_windows([e["clip_window_s"] for e in events])
    times = [o.t for o in result.observations]
    end = result.duration_s
    flagged_s = sum(max(0.0, min(b, end) - max(a, 0.0)) for a, b in merged)
    fwd = sum(any(a <= t <= b for a, b in merged) for t in times)
    by_type = {}
    for e in events:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
    return {
        "sequence": result.seq_id,
        "frames": result.n_frames,
        "duration_s": round(end, 1),
        "candidates": len(events),
        "flagged_s": round(flagged_s, 1),
        "review_reduction_%": round(100 * (1 - flagged_s / end), 1) if end else 0.0,
        "frames_forwarded_%": round(100 * fwd / len(times), 1) if times else 0.0,
        "mean_latency_ms": round(result.mean_latency_ms, 1),
        "throughput_fps": round(result.n_frames / result.wall_s, 1) if result.wall_s else 0.0,
        "tracks_stitched": result.n_stitched,
        **{f"n_{k}": v for k, v in sorted(by_type.items())},
    }


def draw_overlay(frame, obs: Observation, cids, events, recent: list[dict] | None = None):
    img = frame.copy()
    for cid, p in zip(cids, obs.persons):
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"P{cid}" if cid == p["raw_id"] else f"P{cid} (raw {p['raw_id']})"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if p.get("kpts") is not None:
            for x, y, c in p["kpts"]:
                if c >= KP_CONF:
                    cv2.circle(img, (int(x), int(y)), 3, (255, 200, 0), -1)
    for it in obs.items:
        x1, y1, x2, y2 = map(int, it["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(img, it["cls"], (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
    for b in obs.blobs:
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for e in events:
        x1, y1, x2, y2 = e["roi_box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    for i, e in enumerate((recent or [])[-4:]):
        cv2.putText(img, f"{e['timestamp_s']:.1f}s {e['event_type']} {e['entity_id']}", (12, 30 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img
