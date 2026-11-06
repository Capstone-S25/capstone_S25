"""Production gate extracted from Stage 1; benchmark/replay utilities excluded."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PERSON = 0

TWO_WHEELERS = (1, 3)

COCO_ITEMS = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = (0, 1, 2, 3, 4)

L_WRIST, R_WRIST = (9, 10)

L_HIP, R_HIP = (11, 12)

L_KNEE, R_KNEE = (13, 14)

KP_CONF = 0.3

ALL_RULES = ("loiter", "speed", "smoking", "reach_down", "litter", "unattended")

DEFAULT_RULES = ("loiter", "in_view", "reach_down", "unattended")


@dataclass
class GateConfig:
    det_weights: str = "yolo26n.pt"
    pose_weights: str | None = "yolo26n-pose.pt"
    imgsz: int = 640
    device: str = "cpu"
    det_conf: float = 0.1
    person_conf: float = 0.4
    item_classes: tuple[int, ...] = tuple(COCO_ITEMS)
    rules: tuple[str, ...] = DEFAULT_RULES
    pose_every_n: int = 1
    tracker: str = "bytetrack"
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
    loiter_repeat_s: float = float("inf")
    inview_s: float = 120.0
    inview_area_h: float | None = 4.0
    inview_repeat_s: float = float("inf")
    speed_h_per_s: float = 1.2
    speed_window_s: float = 0.6
    hand_face_h: float = 0.2
    smoke_window_s: float = 10.0
    smoke_min_hits: int = 2
    smoke_cooldown_s: float = 30.0
    reach_margin_h: float = -0.15
    reach_cooldown_s: float = 5.0
    reach_min_hits: int = 1
    reach_window_s: float = 0.6
    pose_min_h_px: float = 0.0
    reach_skip_border: bool = True
    reach_border_margin: float = 0.01
    still_border_margin: float | None = 0.05
    still_min_h: float | None = 0.1
    reach_min_h: float | None = None
    hand_min_age_s: float | None = None
    pose_border_margin: float | None = None
    reach_kp_conf: float = 0.5
    reach_bend_ratio: float | None = None
    reach_hand_pad_h: float | None = 0.25
    reach_refire_move_h: float | None = 1.0
    reach_max_speed_h: float | None = 0.6
    reach_speed_window_s: float = 1.0
    reach_skip_riders: bool = True
    reach_rider_classes: tuple[int, ...] = (1, 3)
    reach_rider_stopped_h: float | None = None
    reach_rider_stop_check: tuple[int, ...] = (1,)
    reach_skip_overlap: float | None = 0.5
    reach_skip_seated_s: float | None = 5.0
    reach_verifier: bool = True
    reach_verifier_weights: str | None = None
    reach_verifier_thr: float | None = None
    reach_verifier_ahead_s: float = 0.0
    bg_width: int = 480
    bg_long_s: float = 30.0
    bg_short_s: float = 1.0
    litter_static_s: float = 1.0
    litter_min_area: float = 0.0001
    litter_max_area: float = 0.01
    litter_owner_lookback_s: float = 4.0
    litter_owner_dist_h: float = 1.0
    litter_person_margin: float = 0.25
    litter_max_aspect: float = 4.0
    litter_min_novelty: float = 15.0
    litter_max_change: float = 8.0
    unattended_s: float = 10.0
    owner_leave_h: float = 2.0
    object_grace_s: float = 10.0
    item_warmup_s: float = 2.0
    item_min_frames: int = 3
    carry_margin: float = 0.3
    appear_min_conf: float = 0.35
    appear_classes: tuple[str, ...] = (
        "backpack",
        "handbag",
        "suitcase",
        "bottle",
        "cup",
    )
    item_detector: str | None = None
    item_prompts: tuple[str, ...] = (
        "bag",
        "backpack",
        "handbag",
        "suitcase",
        "box",
        "tote bag",
        "luggage",
        "case",
        "package",
    )
    item_period_s: float = 1.0
    item_conf: float = 0.2
    item_detector_min_passes: int = 2
    item_scenery: bool = True
    near_camera_px: float | None = None
    near_camera_cooldown_s: float = 10.0
    person_dedup_s: float = 2.5
    dedup_per_rule: bool = False
    pose_action_weights: str | None = None
    pose_action_thr: float | None = None
    pose_action_cooldown_s: float = 5.0
    crop_check: bool = True
    crop_check_rules: tuple[str, ...] = ("reach_down", "pose_action")
    crop_check_thr: float | None = None
    crop_check_weights: str | None = None
    crop_check_encoder: str | None = None

    @property
    def uses_bg(self) -> bool:
        return "litter" in self.rules

    @property
    def uses_pose(self) -> bool:
        return bool(self.pose_weights) and bool(
            {"smoking", "reach_down", "pose_action"} & set(self.rules)
        )


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
    return (
        box[0] <= margin * w
        or box[1] <= margin * h
        or box[2] >= (1 - margin) * w
        or (box[3] >= (1 - margin) * h)
    )


def expand_box(b, frac):
    px, py = ((b[2] - b[0]) * frac, (b[3] - b[1]) * frac)
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
    x1, x2 = (int(max(0, x1 + 0.15 * bw)), int(min(w, x2 - 0.15 * bw)))
    y1, y2 = (int(max(0, y1)), int(min(h, y2)))
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
    two_wheelers: list[list[float]] | None = None


def tracker_args(cfg: GateConfig, fps: float) -> dict:
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
        args.update(
            {
                "gmc_method": "none",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": cfg.tracker == "botsort_reid",
                "model": "yolo26n-cls.pt",
                "device": "cpu",
            }
        )
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
            history=max(2, int(cfg.bg_long_s * fps)),
            varThreshold=25,
            detectShadows=True,
        )
        self.short = cv2.createBackgroundSubtractorMOG2(
            history=max(2, int(cfg.bg_short_s * fps)),
            varThreshold=25,
            detectShadows=True,
        )
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
        small = cv2.GaussianBlur(
            cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), (3, 3), 0
        )
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
            if not self.min_area <= frac <= self.max_area:
                continue
            region = labels[y : y + bh, x : x + bw] == i
            age = float(self.age[y : y + bh, x : x + bw][region].max())
            if background is None:
                background = cv2.cvtColor(
                    self.long.getBackgroundImage(), cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
            now = gray[y : y + bh, x : x + bw][region]
            novelty = float(
                np.mean(np.abs(now - background[y : y + bh, x : x + bw][region]))
            )
            change = float(
                np.mean(
                    np.abs(
                        now
                        - self._gray_at(self.t - age + dt)[y : y + bh, x : x + bw][
                            region
                        ]
                    )
                )
            )
            s = self.scale
            blobs.append(
                {
                    "box": [x / s, y / s, (x + bw) / s, (y + bh) / s],
                    "area_frac": frac,
                    "age_s": age,
                    "novelty": novelty,
                    "change": change,
                }
            )
        return blobs


class Perception:
    def __init__(self, cfg: GateConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.det = YOLO(cfg.det_weights)
        self.pose = YOLO(cfg.pose_weights) if cfg.uses_pose else None
        self.item_model = None
        if cfg.item_detector:
            from ultralytics import YOLOE

            self.item_model = YOLOE(cfg.item_detector)
            self.item_model.set_classes(list(cfg.item_prompts))

    def reset(self, fps: float):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace

        self.fps = fps
        args = tracker_args(self.cfg, fps)
        self.tracker = TRACKER_MAP[args["tracker_type"]](
            args=IterableSimpleNamespace(**args)
        )
        self.tracker.reset()
        self.bg = None
        self.confirmed: set[int] = set()
        self.n = 0
        self.last_t = None
        self.item_tracks: list[dict] = []
        self.next_item_id = 0
        self.next_item_t = -math.inf

    def observe(
        self, frame, index: int, t: float, path: str | None = None
    ) -> Observation:
        cfg = self.cfg
        lat = {}
        t0 = time.perf_counter()
        classes = (
            [PERSON] if self.item_model is not None else [PERSON, *cfg.item_classes]
        )
        if cfg.reach_skip_riders:
            classes += list(cfg.reach_rider_classes)
        res = self.det.predict(
            frame,
            imgsz=cfg.imgsz,
            conf=cfg.det_conf,
            device=cfg.device,
            classes=classes,
            verbose=False,
        )[0]
        lat["det_ms"] = (time.perf_counter() - t0) * 1000
        two_wheelers = None
        if cfg.reach_skip_riders:
            cls = res.boxes.cls.cpu().numpy().astype(int)
            two_wheelers = [
                b.tolist() + [int(c)]
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), cls)
                if c in cfg.reach_rider_classes
            ]
            res = res[~np.isin(cls, list(cfg.reach_rider_classes))]
        self.tracker.update(res.boxes.cpu().numpy(), frame)
        live = [
            s
            for s in self.tracker.tracked_stracks
            if s.frame_id == self.tracker.frame_id
        ]
        persons, items = ([], [])
        for s in live:
            box, tid, cf, c = (
                s.xyxy.tolist(),
                int(s.track_id),
                float(s.score),
                int(s.cls),
            )
            if c == PERSON:
                if cf >= cfg.person_conf:
                    self.confirmed.add(tid)
                if tid in self.confirmed:
                    persons.append({"raw_id": tid, "box": box, "conf": cf})
            else:
                items.append(
                    {
                        "raw_id": tid,
                        "box": box,
                        "conf": cf,
                        "cls": COCO_ITEMS.get(c, res.names[c]),
                    }
                )
        if self.item_model is not None:
            items = []
            if t >= self.next_item_t - 1e-06:
                self.next_item_t = t + cfg.item_period_s
                t3 = time.perf_counter()
                items = self._item_pass(frame)
                lat["item_ms"] = (time.perf_counter() - t3) * 1000
        for p in persons:
            crowded = any(
                (iou(p["box"], q["box"]) > 0.15 for q in persons if q is not p)
            )
            p["hist"] = None if crowded else appearance_hist(frame, p["box"])
            p["kpts"] = None
        if (
            self.pose is not None
            and persons
            and (self.n % max(1, cfg.pose_every_n) == 0)
        ):
            t1 = time.perf_counter()
            pr = self.pose.predict(
                frame, imgsz=cfg.imgsz, conf=0.25, device=cfg.device, verbose=False
            )[0]
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
            dt = t - self.last_t if self.last_t is not None else 1.0 / self.fps
            blobs = self.bg.apply(frame, dt)
            lat["bg_ms"] = (time.perf_counter() - t2) * 1000
        lat["total_ms"] = (time.perf_counter() - t0) * 1000
        self.n += 1
        self.last_t = t
        return Observation(
            index, t, frame.shape[:2], persons, items, blobs, lat, path, two_wheelers
        )

    def _item_pass(self, frame) -> list[dict]:
        """Open-vocabulary objects, with ids from IoU matching to the previous passes."""
        cfg = self.cfg
        res = self.item_model.predict(
            frame, imgsz=cfg.imgsz, conf=cfg.item_conf, device=cfg.device, verbose=False
        )[0]
        items = []
        for b, c, k in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
        ):
            box = [float(v) for v in b]
            tr = max(self.item_tracks, key=lambda q: iou(q["box"], box), default=None)
            if tr is None or iou(tr["box"], box) < 0.3:
                tr = {"id": 100000 + self.next_item_id, "box": box}
                self.next_item_id += 1
                self.item_tracks = (self.item_tracks + [tr])[-500:]
            tr["box"] = box
            items.append(
                {
                    "raw_id": tr["id"],
                    "box": box,
                    "conf": float(c),
                    "cls": cfg.item_prompts[k],
                }
            )
        return items


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
            tr.update(
                last_t=t, foot=foot(p["box"]), h=max(1.0, p["box"][3] - p["box"][1])
            )
            if p.get("hist") is not None:
                tr["hist"] = (
                    p["hist"]
                    if tr["hist"] is None
                    else 0.8 * tr["hist"] + 0.2 * p["hist"]
                )
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
        best, best_score = (None, -1000000000.0)
        for c, tr in self.tracks.items():
            gap = t - tr["last_t"]
            if (
                c in used
                or gap <= 0
                or gap > cfg.stitch_max_gap_s
                or (tr["hist"] is None)
            ):
                continue
            ratio = ph / tr["h"]
            if not 0.6 <= ratio <= 1.67:
                continue
            dist = math.hypot(pf[0] - tr["foot"][0], pf[1] - tr["foot"][1]) / max(
                ph, tr["h"]
            )
            if dist > cfg.stitch_base_dist_h + cfg.stitch_walk_h_per_s * gap:
                continue
            sim = hist_similarity(p["hist"], tr["hist"])
            if sim < cfg.stitch_min_similarity:
                continue
            score = sim - 0.05 * dist
            if score > best_score:
                best, best_score = (c, score)
        return best


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
    "pose_action": "Person's body movement looks like dropping, tossing or throwing something. Check whether they littered or discarded an item.",
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
        self.appear_classes = set(cfg.appear_classes) | (
            set(cfg.item_prompts) if cfg.item_detector else set()
        )
        self.item_min_seen = (
            cfg.item_detector_min_passes if cfg.item_detector else cfg.item_min_frames
        )
        self.pose_action = None
        if "pose_action" in cfg.rules:
            from .pose import DEFAULT_WEIGHTS, PoseActionModel, TrackWindows

            model = PoseActionModel(cfg.pose_action_weights or DEFAULT_WEIGHTS)
            thr = (
                model.threshold if cfg.pose_action_thr is None else cfg.pose_action_thr
            )
            self.pose_action = {
                "model": model,
                "thr": thr,
                "windows": TrackWindows(),
                "last": {},
            }
        self.verifier = None
        if (
            cfg.reach_verifier or cfg.reach_verifier_weights
        ) and "reach_down" in cfg.rules:
            from .pose import DEFAULT_WEIGHTS, PoseActionModel

            model = PoseActionModel(
                cfg.reach_verifier_weights
                or DEFAULT_WEIGHTS.with_name("reach_verifier.npz")
            )
            self.verifier = {
                "model": model,
                "thr": model.threshold
                if cfg.reach_verifier_thr is None
                else cfg.reach_verifier_thr,
                "kp": defaultdict(deque),
                "heights": defaultdict(lambda: deque(maxlen=200)),
                "pending": [],
            }
        self.crop_check = None
        self.seq_id: str | None = None
        if cfg.crop_check:
            from concurrent.futures import ThreadPoolExecutor

            from .crop_check import CropCheck, weights_for

            checks = {
                r: CropCheck(
                    cfg.crop_check_weights
                    if r == "reach_down" and cfg.crop_check_weights
                    else weights_for(r),
                    cfg.crop_check_encoder,
                )
                for r in cfg.crop_check_rules
            }
            self.crop_check = {
                "checks": checks,
                "frames": deque(),
                "pending": [],
                "thr": {
                    r: c.threshold if cfg.crop_check_thr is None else cfg.crop_check_thr
                    for r, c in checks.items()
                },
                "keep_s": max((c.dt for c in checks.values()), default=0.0) + 0.5,
                "pool": ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="crop_check"
                ),
            }

    def _crop_check(self, events, obs, frame):
        """Drop crop_check_rules alerts whose crop scores below their rule's threshold. Without a frame (cached replay)
        the score comes from stage1_crop_check.SCORES at once; an alert with no score is kept. Live, a version-2 check
        holds the alert: the person's crops at t - dt and t are cut now and encoded on a background thread, the crop
        at t + dt is cut when it arrives, and the alert is released (or dropped) once its score is ready, so the frame
        loop never waits for the encoder."""
        from .crop_check import SCORES, crop

        cc, out = (self.crop_check, [])
        if frame is not None:
            if not cc.get("warm"):
                cc["warm"] = [
                    cc["pool"].submit(c._load)
                    for c in cc["checks"].values()
                    if hasattr(c, "_load")
                ]
            frames = cc["frames"]
            frames.append((obs.t, frame, {p["raw_id"]: p["box"] for p in obs.persons}))
            while frames and frames[0][0] < obs.t - cc["keep_s"]:
                frames.popleft()
            for x in cc["pending"]:
                if "score" not in x and x["due"] <= obs.t + 1e-06:
                    box = frames[-1][2].get(x["rid"])
                    after = (
                        None
                        if box is None
                        else crop(
                            frame, [box[i] + x["grow"][i] for i in range(4)]
                        ).copy()
                    )
                    x["score"] = cc["pool"].submit(
                        self._crop_score, x["check"], x["first"], after
                    )
            out += self._crop_release(lambda x: "score" in x and x["score"].done())
        for e in events:
            check = cc["checks"].get(e["event_type"])
            if check is None:
                out.append(e)
            elif frame is None:
                self._crop_keep(e, SCORES.get((self.seq_id, e["event_id"])), out)
            elif check.version < 2:
                self._crop_keep(
                    e, check.score(frame, e["roi_box"], e["event_type"]), out
                )
            else:
                roi = e["roi_box"]
                p = max(
                    obs.persons,
                    key=lambda q: intersection_over(q["box"], roi),
                    default=None,
                )
                now = crop(frame, roi).copy()
                if p is None or now.size == 0:
                    out.append(e)
                    continue
                grow = [roi[i] - p["box"][i] for i in range(4)]
                before = None
                cand = [
                    (abs(ft - (obs.t - check.dt)), f, boxes)
                    for ft, f, boxes in cc["frames"]
                    if abs(ft - (obs.t - check.dt)) <= 0.4 and p["raw_id"] in boxes
                ]
                if cand:
                    _, f, boxes = min(cand, key=lambda c: c[0])
                    b = boxes[p["raw_id"]]
                    before = crop(f, [b[i] + grow[i] for i in range(4)]).copy()
                first = cc["pool"].submit(
                    lambda c=check, b=before, n=now: (c.embed_crop(b), c.embed_crop(n))
                )
                cc["pending"].append(
                    {
                        "due": obs.t + check.dt,
                        "event": e,
                        "check": check,
                        "rid": p["raw_id"],
                        "grow": grow,
                        "first": first,
                        "frame": frame,
                        "t": obs.t,
                    }
                )
        return out

    @staticmethod
    def _crop_score(check, first, after_crop):
        before, now = first.result()
        return check.score_embeddings([before, now, check.embed_crop(after_crop)])

    def _crop_release(self, ready):
        """Release the held alerts for which ready(x) holds, keeping those that score at or above the threshold
        (an encoder error keeps the alert)."""
        cc, out, keep = (self.crop_check, [], [])
        for x in cc["pending"]:
            if not ready(x):
                keep.append(x)
                continue
            try:
                score = x["score"].result()
            except Exception as exc:
                score = None
                if not cc.get("warned"):
                    cc["warned"] = True
                    import warnings

                    warnings.warn(
                        f"Stage 1 crop check failed, alerts pass unchecked: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            x["event"]["_frame"], x["event"]["_t"] = (x["frame"], x["t"])
            self._crop_keep(x["event"], score, out)
        cc["pending"] = keep
        return out

    def flush(self) -> list[dict]:
        """End of a stream: score and release every alert the crop check still holds (the "after" crop is the last
        one seen, or missing), through the person dedup. Call after the last step()."""
        if not self.crop_check or not self.crop_check["pending"]:
            return []
        cc = self.crop_check
        for x in cc["pending"]:
            if "score" not in x:
                x["score"] = cc["pool"].submit(
                    self._crop_score, x["check"], x["first"], None
                )
        t = max((x["t"] for x in cc["pending"]))
        return self._dedup(self._crop_release(lambda x: True), t)

    def _crop_keep(self, e, score, out):
        if score is None or score >= self.crop_check["thr"][e["event_type"]]:
            if score is not None:
                e["crop_score"] = round(score, 3)
            out.append(e)

    def _verify_window(self, rid, t, frame_hw):
        from .pose import window_features

        v = self.verifier
        hist = [x for x in v["kp"].get(rid, ()) if t - 2.0 <= x[0] <= t + 1e-06]
        if not hist:
            return None
        return window_features(
            hist, float(np.median(v["heights"][rid])), t, 1.0, 8, frame_hw
        )[1]

    def _verify(self, events, obs):
        """Hold each reach_down for reach_verifier_ahead_s, then keep it only if the verifier agrees."""
        v, t, ahead = (self.verifier, obs.t, self.cfg.reach_verifier_ahead_s)
        for p in obs.persons:
            kp = v["kp"][p["raw_id"]]
            kp.append((t, p["box"], p.get("kpts")))
            while kp and kp[0][0] < t - 3.0:
                kp.popleft()
            v["heights"][p["raw_id"]].append(p["box"][3] - p["box"][1])
        for rid in [r for r, kp in v["kp"].items() if not kp or t - kp[-1][0] > 10.0]:
            del v["kp"][rid]
            v["heights"].pop(rid, None)
        out = []
        for e in events:
            rid = e.pop("_rid", None)
            if e["event_type"] != "reach_down" or rid is None:
                out.append(e)
                continue
            now = self._verify_window(rid, t, obs.frame_hw)
            if now is None:
                out.append(e)
            else:
                v["pending"].append((t + ahead, e, rid, now))
        due = [x for x in v["pending"] if x[0] <= t + 1e-06]
        v["pending"] = [x for x in v["pending"] if x[0] > t + 1e-06]
        for _, e, rid, now in due:
            f = now
            if ahead:
                later = self._verify_window(rid, t, obs.frame_hw)
                f = np.concatenate(
                    [now, later, [1.0]]
                    if later is not None
                    else [now, np.zeros_like(now), [0.0]]
                )
            score = float(v["model"].predict(f[None])[0])
            if score >= v["thr"]:
                e["verifier_score"] = round(score, 3)
                out.append(e)
        return out

    def step(self, obs: Observation, frame: np.ndarray | None = None) -> list[dict]:
        cfg, t = (self.cfg, obs.t)
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
                    new = self._reach(cid, st, p, obs)
                    if self.verifier:
                        for e in new:
                            e["_rid"] = p["raw_id"]
                    events += new
            if "near_camera" in cfg.rules and cfg.near_camera_px:
                events += self._near_camera(cid, st, p, obs)
        if "unattended" in cfg.rules:
            events += self._unattended(obs, persons)
        if "litter" in cfg.rules:
            events += self._litter(obs, persons)
        if self.pose_action:
            events += self._pose_action(
                obs, dict(((p["raw_id"], c) for c, p in persons))
            )
        horizon = (
            max(
                cfg.loiter_s,
                cfg.stitch_max_gap_s,
                cfg.litter_owner_lookback_s + cfg.litter_static_s,
            )
            + 2
        )
        for cid in [c for c, s in self.people.items() if t - s["last_t"] > horizon]:
            del self.people[cid]
        if self.verifier:
            events = self._verify(events, obs)
        if self.crop_check:
            events = self._crop_check(events, obs, frame)
        return self._dedup(events, t)

    def _dedup(self, events, t):
        """Drop an alert when the same person (or an overlapping box) already had one that happened up to
        person_dedup_s earlier. Times are the alerts' own timestamps, so an alert held by the crop check and
        released late is judged as if it had been sent when it fired."""
        dedup_s = self.cfg.person_dedup_s
        horizon = dedup_s + (
            self.crop_check["keep_s"] + 10.0 if self.crop_check else 0.0
        )
        if self.sent and t - min((x[4] for x in self.sent)) > horizon:
            self.sent = deque((x for x in self.sent if t - x[4] <= horizon))
        kept = []
        for e in events:
            box, ts_e = (e["roi_box"], e["timestamp_s"])
            kind = e["event_type"] if self.cfg.dedup_per_rule else None
            if any(
                (
                    k == kind
                    and 0.0 <= ts_e - ts + 1e-06 <= dedup_s + 2e-06
                    and (
                        ent == e["entity_id"]
                        or iou(box, b) >= 0.5
                        or max(intersection_over(box, b), intersection_over(b, box))
                        >= 0.7
                    )
                    for _, ent, b, k, ts in self.sent
                )
            ):
                continue
            self.sent.append((t, e["entity_id"], box, kind, ts_e))
            kept.append(e)
        return kept

    def _state(self, cid, p, t):
        st = self.people.get(cid)
        if st is None:
            st = self.people[cid] = {
                "first_t": t,
                "hist": deque(),
                "loiter_flagged": None,
                "speed_armed": True,
                "smoke_hits": deque(),
                "last_smoke": -1000000000.0,
                "last_reach": -1000000000.0,
                "last_near": -1000000000.0,
                "view_pts": deque(),
                "last_inview": None,
            }
        st["last_t"] = t
        st["box"] = p["box"]
        fx, fy = foot(p["box"])
        st["hist"].append(
            (t, fx, fy, max(1.0, p["box"][3] - p["box"][1]), p["box"], p["raw_id"])
        )
        keep = (
            max(
                self.cfg.loiter_s,
                self.cfg.litter_owner_lookback_s + self.cfg.litter_static_s,
            )
            + 1
        )
        while st["hist"] and st["hist"][0][0] < t - keep:
            st["hist"].popleft()
        return st

    def _event(self, etype, entity, obs, roi, **extra):
        t, pad = (obs.t, self.cfg.clip_pad_s)
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

    @staticmethod
    def _guard_ok(st, p, obs, min_h, border, min_age_s) -> bool:
        """The person-rule guards (GateConfig still_* / reach_min_h / hand_min_age_s / pose_border_margin)."""
        b = p["box"]
        if min_h is not None and b[3] - b[1] < min_h * obs.frame_hw[0]:
            return False
        if border is not None and touches_border(b, obs.frame_hw, border):
            return False
        return min_age_s is None or obs.t - st["first_t"] >= min_age_s

    def _loiter(self, cid, st, p, obs):
        cfg, t = (self.cfg, obs.t)
        pts = [h for h in st["hist"] if h[0] >= t - cfg.loiter_s]
        slack = 1.5 / self.fps
        if (
            t - st["first_t"] < cfg.loiter_s
            or not pts
            or t - pts[0][0] < cfg.loiter_s - slack
        ):
            return []
        xy = np.array([(h[1], h[2]) for h in pts])
        radius = float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))) / float(
            np.median([h[3] for h in pts])
        )
        if radius >= cfg.loiter_release_h:
            st["loiter_flagged"] = None
        if (
            radius < cfg.loiter_radius_h
            and (
                st["loiter_flagged"] is None
                or t - st["loiter_flagged"] >= cfg.loiter_repeat_s
            )
            and self._guard_ok(
                st, p, obs, cfg.still_min_h, cfg.still_border_margin, None
            )
        ):
            st["loiter_flagged"] = t
            return [
                self._event(
                    "sustained_presence",
                    f"person{cid}",
                    obs,
                    p["box"],
                    evidence_window_s=[round(t - cfg.loiter_s, 2), round(t, 2)],
                    metrics={"norm_radius": round(radius, 3)},
                )
            ]
        return []

    def _speed(self, cid, st, p, obs):
        cfg, t = (self.cfg, obs.t)
        window = max(cfg.speed_window_s, 3.0 / self.fps)
        pts = []
        for h in reversed(st["hist"]):
            if (
                h[0] < t - window
                or h[5] != p["raw_id"]
                or (pts and pts[-1][0] - h[0] > 2.5 / self.fps)
            ):
                break
            pts.append(h)
        pts.reverse()
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.5 * window:
            return []
        steps = [
            math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) / b[3]
            for a, b in zip(pts, pts[1:])
        ]
        v = float(np.median(steps))
        if v >= cfg.speed_h_per_s and st["speed_armed"]:
            st["speed_armed"] = False
            return [
                self._event(
                    "fast_motion",
                    f"person{cid}",
                    obs,
                    p["box"],
                    metrics={"speed_h_per_s": round(v, 2)},
                )
            ]
        if v < 0.6 * cfg.speed_h_per_s:
            st["speed_armed"] = True
        return []

    def _smoking(self, cid, st, p, obs):
        cfg, t, k = (self.cfg, obs.t, p["kpts"])
        face_pts = [_kp(k, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)]
        face = face_pts[0] or (
            tuple(np.mean([f for f in face_pts if f], axis=0))
            if any(face_pts)
            else None
        )
        wrists = [w for w in (_kp(k, L_WRIST), _kp(k, R_WRIST)) if w]
        if face is None or not wrists:
            return []
        h = max(1.0, p["box"][3] - p["box"][1])
        d = min((math.hypot(w[0] - face[0], w[1] - face[1]) for w in wrists)) / h
        if d <= cfg.hand_face_h:
            st["smoke_hits"].append(t)
        while st["smoke_hits"] and st["smoke_hits"][0] < t - cfg.smoke_window_s:
            st["smoke_hits"].popleft()
        if (
            len(st["smoke_hits"]) >= cfg.smoke_min_hits
            and t - st["last_smoke"] >= cfg.smoke_cooldown_s
        ):
            st["last_smoke"] = t
            return [
                self._event(
                    "hand_to_face",
                    f"person{cid}",
                    obs,
                    p["box"],
                    metrics={"wrist_face_h": round(d, 3)},
                )
            ]
        return []

    def _reach(self, cid, st, p, obs):
        cfg, t, k = (self.cfg, obs.t, p["kpts"])
        kc = cfg.reach_kp_conf
        knees = [
            (float(k[i, 0]), float(k[i, 1])) for i in (L_KNEE, R_KNEE) if k[i, 2] >= kc
        ]
        wrists = [
            (float(k[i, 0]), float(k[i, 1]))
            for i in (L_WRIST, R_WRIST)
            if k[i, 2] >= kc
        ]
        if not knees or not wrists:
            return []
        h = max(1.0, p["box"][3] - p["box"][1])
        if cfg.reach_bend_ratio is not None:
            usual = float(np.percentile([q[3] for q in st["hist"]], 75))
            if len(st["hist"]) < 3 or h > cfg.reach_bend_ratio * usual:
                return []
        if h < cfg.pose_min_h_px or (
            cfg.reach_skip_border
            and touches_border(p["box"], obs.frame_hw, cfg.reach_border_margin)
        ):
            return []
        if not self._guard_ok(st, p, obs, cfg.reach_min_h, None, cfg.hand_min_age_s):
            return []
        v = None
        pts = [
            q
            for q in st["hist"]
            if q[0] >= t - cfg.reach_speed_window_s and q[5] == p["raw_id"]
        ]
        if len(pts) >= 2 and pts[-1][0] - pts[0][0] >= 0.5 * cfg.reach_speed_window_s:
            a, b = (pts[0], pts[-1])
            v = (
                math.hypot(b[1] - a[1], b[2] - a[2])
                / (b[0] - a[0])
                / float(np.median([q[3] for q in pts]))
            )
        if cfg.reach_skip_riders:
            b = p["box"]
            lower = [b[0], (b[1] + b[3]) / 2, b[2], b[3]]
            riding = [
                int(w[4]) if len(w) >= 5 else -1
                for w in getattr(obs, "two_wheelers", None) or []
                if (len(w) < 5 or int(w[4]) in cfg.reach_rider_classes)
                and intersection_over(lower, w[:4]) >= 0.3
            ]
            stopped = (
                cfg.reach_rider_stopped_h is not None
                and v is not None
                and (v < cfg.reach_rider_stopped_h)
            )
            if any(
                (c not in cfg.reach_rider_stop_check or not stopped for c in riding)
            ):
                return []
        if cfg.reach_skip_overlap is not None and any(
            (
                intersection_over(p["box"], q["box"]) >= cfg.reach_skip_overlap
                for q in obs.persons
                if q is not p
            )
        ):
            return []
        if (
            cfg.reach_max_speed_h is not None
            and v is not None
            and (v > cfg.reach_max_speed_h)
        ):
            return []
        if cfg.reach_skip_seated_s is not None:
            hips = [float(k[i, 1]) for i in (L_HIP, R_HIP) if k[i, 2] >= kc]
            seated = (
                bool(hips)
                and float(np.mean([q[1] for q in knees])) - float(np.mean(hips))
                < 0.12 * h
            )
            if not seated:
                st["seated_since"] = None
            elif st.get("seated_since") is None:
                st["seated_since"] = t
            if seated and t - st["seated_since"] >= cfg.reach_skip_seated_s:
                return []
        knee_y = float(np.mean([q[1] for q in knees]))
        hits = st.setdefault("reach_hits", deque())
        if max((w[1] for w in wrists)) >= knee_y + cfg.reach_margin_h * h:
            hits.append(t)
        window = max(cfg.reach_window_s, 1.5 / self.fps)
        while hits and hits[0] < t - window:
            hits.popleft()
        since = t - st["last_reach"]
        ready = since >= cfg.reach_cooldown_s
        if (
            not ready
            and cfg.reach_refire_move_h is not None
            and (since >= 1.0)
            and ("last_reach_foot" in st)
        ):
            (fx, fy), (lx, ly) = (foot(p["box"]), st["last_reach_foot"])
            ready = math.hypot(fx - lx, fy - ly) >= cfg.reach_refire_move_h * h
        if len(hits) >= cfg.reach_min_hits and ready:
            st["last_reach"] = t
            st["last_reach_foot"] = foot(p["box"])
            roi = p["box"]
            if cfg.reach_hand_pad_h is not None:
                wx, wy = max(wrists, key=lambda w: w[1])
                r = cfg.reach_hand_pad_h * h
                fh, fw = obs.frame_hw
                roi = union_box(
                    roi,
                    [
                        max(0.0, wx - r),
                        max(0.0, wy - r),
                        min(fw, wx + r),
                        min(fh, wy + r),
                    ],
                )
            return [self._event("reach_down", f"person{cid}", obs, roi)]
        return []

    def _pose_action(self, obs, cid_of):
        pa, out = (self.pose_action, [])
        ws = pa["windows"].step(obs)
        if not ws:
            return out
        probs = pa["model"].predict(np.stack([w["sum"] for w in ws]))
        for w, prob in zip(ws, probs):
            rid = w["raw_id"]
            st = self.people.get(cid_of[rid])
            if st is not None and (
                not self._guard_ok(
                    st,
                    {"box": w["box"]},
                    obs,
                    self.cfg.reach_min_h,
                    self.cfg.pose_border_margin,
                    self.cfg.hand_min_age_s,
                )
            ):
                continue
            if (
                prob >= pa["thr"]
                and obs.t - pa["last"].get(rid, -1000000000.0)
                >= self.cfg.pose_action_cooldown_s
            ):
                pa["last"][rid] = obs.t
                out.append(
                    self._event(
                        "pose_action",
                        f"person{cid_of[rid]}",
                        obs,
                        w["box"],
                        score=round(float(prob), 3),
                    )
                )
        return out

    def _inview(self, cid, st, p, obs):
        cfg, t = (self.cfg, obs.t)
        pts = st["view_pts"]
        if not pts or t - pts[-1][0] >= 1.0:
            fx, fy = foot(p["box"])
            pts.append((t, fx, fy, max(1.0, p["box"][3] - p["box"][1])))
        while pts and pts[0][0] < t - cfg.inview_s:
            pts.popleft()
        if t - st["first_t"] < cfg.inview_s:
            return []
        if (
            st["last_inview"] is not None
            and t - st["last_inview"] < cfg.inview_repeat_s
        ):
            return []
        if not self._guard_ok(
            st, p, obs, cfg.still_min_h, cfg.still_border_margin, None
        ):
            return []
        metrics = {"time_in_view_s": round(t - st["first_t"], 1)}
        if cfg.inview_area_h is not None:
            xy = np.array([(q[1], q[2]) for q in pts])
            radius = float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))) / float(
                np.median([q[3] for q in pts])
            )
            if radius > cfg.inview_area_h:
                return []
            metrics["area_radius_h"] = round(radius, 2)
        st["last_inview"] = t
        return [
            self._event(
                "long_presence",
                f"person{cid}",
                obs,
                p["box"],
                evidence_window_s=[round(t - cfg.inview_s, 2), round(t, 2)],
                metrics=metrics,
            )
        ]

    def _near_camera(self, cid, st, p, obs):
        if (
            p["box"][3] - p["box"][1] <= self.cfg.near_camera_px
            or obs.t - st["last_near"] < self.cfg.near_camera_cooldown_s
        ):
            return []
        st["last_near"] = obs.t
        return [self._event("near_camera", f"person{cid}", obs, p["box"])]

    def _nearest_person(self, box, persons, max_h):
        c = center(box)
        best, best_d = (None, 1000000000.0)
        for cid, p in persons:
            h = max(1.0, p["box"][3] - p["box"][1])
            d = (
                min(
                    (
                        math.hypot(c[0] - q[0], c[1] - q[1])
                        for q in (foot(p["box"]), center(p["box"]))
                    )
                )
                / h
            )
            if d <= max_h and d < best_d:
                best, best_d = (cid, d)
        return best

    def _carrier(self, box, persons):
        """The person holding or wearing an item: the one whose box contains most of the item's box."""
        best, best_cover = (None, 0.5)
        for cid, p in persons:
            cover = intersection_over(box, expand_box(p["box"], self.cfg.carry_margin))
            if cover >= best_cover:
                best, best_cover = (cid, cover)
        return best

    def _unattended(self, obs, persons):
        cfg, t = (self.cfg, obs.t)
        visible = {cid: p for cid, p in persons}
        events = []
        for it in obs.items:
            entry = next(
                (
                    e
                    for e in self.items
                    if e["raw_id"] == it["raw_id"] or iou(e["box"], it["box"]) >= 0.3
                ),
                None,
            )
            if entry is None:
                known = any((iou(k, it["box"]) >= 0.3 for k in self.known_item_spots))
                late = t - self.t0 >= cfg.item_warmup_s and (not known)
                alone = self._carrier(it["box"], persons) is None
                entry = {
                    "raw_id": it["raw_id"],
                    "cls": it["cls"],
                    "origin": it["box"],
                    "first_t": t,
                    "seen": 0,
                    "owner": None,
                    "owner_near_t": None,
                    "flagged": False,
                    "nobody_since": None,
                    "late": late,
                    "appeared_alone": late and alone,
                    "appear_flagged": False,
                    "moved": False,
                    "max_conf": 0.0,
                    "scenery": cfg.item_scenery
                    and alone
                    and (
                        self._nearest_person(it["box"], persons, cfg.owner_leave_h)
                        is None
                    ),
                }
                self.items.append(entry)
            entry.update(box=it["box"], last_t=t, raw_id=it["raw_id"])
            entry["seen"] += 1
            entry["max_conf"] = max(entry["max_conf"], it["conf"])
            if not any((iou(k, it["box"]) >= 0.3 for k in self.known_item_spots)):
                self.known_item_spots = (self.known_item_spots + [it["box"]])[-200:]
            carrier = self._carrier(it["box"], persons)
            if entry["scenery"] and (not entry["moved"]):
                carrier = None
            if carrier is not None:
                entry.update(
                    origin=it["box"],
                    owner=carrier,
                    owner_near_t=t,
                    flagged=False,
                    nobody_since=None,
                    appeared_alone=False,
                )
                continue
            diag = math.hypot(it["box"][2] - it["box"][0], it["box"][3] - it["box"][1])
            oc, nc = (center(entry["origin"]), center(it["box"]))
            if math.hypot(nc[0] - oc[0], nc[1] - oc[1]) > max(10.0, 0.5 * diag):
                entry.update(
                    origin=it["box"], flagged=False, nobody_since=None, moved=True
                )
            if entry["owner"] is None and (
                not (entry["scenery"] and (not entry["moved"]))
            ):
                entry["owner"] = self._nearest_person(
                    it["box"], persons, cfg.owner_leave_h
                )
                if entry["owner"] is not None:
                    entry["owner_near_t"] = t
            elif (
                entry["owner"] in visible
                and self._nearest_person(
                    it["box"],
                    [(entry["owner"], visible[entry["owner"]])],
                    cfg.owner_leave_h,
                )
                is not None
            ):
                entry["owner_near_t"] = t
            if self._nearest_person(it["box"], persons, cfg.owner_leave_h) is None:
                entry["nobody_since"] = (
                    entry["nobody_since"] if entry["nobody_since"] is not None else t
                )
            else:
                entry["nobody_since"] = None
            if entry["seen"] < self.item_min_seen:
                continue
            if (
                entry["appeared_alone"]
                and (not entry["appear_flagged"])
                and (not entry["moved"])
                and (entry["cls"] in self.appear_classes)
                and (entry["max_conf"] >= cfg.appear_min_conf)
            ):
                entry["appear_flagged"] = True
                near = self._nearest_person(it["box"], persons, 4.0)
                roi = (
                    union_box(it["box"], visible[near]["box"])
                    if near is not None
                    else it["box"]
                )
                events.append(
                    self._event(
                        "object_appeared",
                        f"{it['cls']}{it['raw_id']}",
                        obs,
                        roi,
                        item_class=it["cls"],
                    )
                )
            owner_gone = (
                entry["owner"] is not None
                and t - entry["owner_near_t"] >= cfg.unattended_s
            )
            ownerless = (
                entry["owner"] is None
                and entry["late"]
                and (entry["nobody_since"] is not None)
                and (t - entry["nobody_since"] >= cfg.unattended_s)
            )
            if (owner_gone or ownerless) and (not entry["flagged"]):
                entry["flagged"] = True
                events.append(
                    self._event(
                        "unattended_object",
                        f"{it['cls']}{it['raw_id']}",
                        obs,
                        it["box"],
                        owner=f"person{entry['owner']}"
                        if entry["owner"] is not None
                        else None,
                        item_class=it["cls"],
                    )
                )
        self.items = [e for e in self.items if t - e["last_t"] <= cfg.object_grace_s]
        return events

    def _litter(self, obs, persons):
        cfg, t = (self.cfg, obs.t)
        fh, fw = obs.frame_hw
        margin = 0.01 * max(fh, fw)
        near_people = [
            expand_box(p["box"], cfg.litter_person_margin) for _, p in persons
        ]
        events = []
        for b in obs.blobs:
            if (
                b["age_s"] < cfg.litter_static_s
                or not cfg.litter_min_area <= b["area_frac"] <= cfg.litter_max_area
            ):
                continue
            if (
                b.get("novelty", 1000000000.0) < cfg.litter_min_novelty
                or b.get("change", 0.0) > cfg.litter_max_change
            ):
                continue
            box = b["box"]
            bw, bh = (box[2] - box[0], box[3] - box[1])
            if max(bw, bh) > cfg.litter_max_aspect * max(1.0, min(bw, bh)):
                continue
            if (
                box[0] <= margin
                or box[1] <= margin
                or box[2] >= fw - margin
                or (box[3] >= fh - margin)
            ):
                continue
            if any((intersection_over(box, q) > 0 for q in near_people)):
                continue
            if any(
                (
                    iou(box, f) > 0.2 or intersection_over(box, f) > 0.5
                    for f in self.flagged_blobs
                )
            ):
                continue
            bc = center(box)
            owner, owner_box, owner_t = (None, None, -1000000000.0)
            for cid, st in self.people.items():
                for ht, fx, fy, h, pbox, _ in st["hist"]:
                    if (
                        ht < t - cfg.litter_static_s - cfg.litter_owner_lookback_s
                        or ht <= owner_t
                    ):
                        continue
                    pc = center(pbox)
                    if (
                        min(
                            math.hypot(bc[0] - fx, bc[1] - fy),
                            math.hypot(bc[0] - pc[0], bc[1] - pc[1]),
                        )
                        / h
                        <= cfg.litter_owner_dist_h
                    ):
                        owner, owner_box, owner_t = (cid, pbox, ht)
            if owner is None:
                continue
            self.flagged_blobs.append(box)
            events.append(
                self._event(
                    "static_object_left",
                    f"blob{len(self.flagged_blobs)}",
                    obs,
                    union_box(box, owner_box),
                    object_box=[int(v) for v in box],
                    owner=f"person{owner}",
                )
            )
        return events


class HandoffWriter:
    """Writes what the VLM sees for each candidate: a padded crop, an annotated context frame and a short
    window of earlier frames ending at the trigger frame.

    The window only uses frames up to the trigger time, so it works on a live stream. Image sequences
    reference their existing files. Video frames are not on disk, so the writer keeps a small buffer of
    recent frames (sampled at `buffer_fps`, JPEG-compressed, at most `buffer_s` old) and saves the
    selected ones. Events with an `evidence_window_s` (loitering, long presence) use that span, clamped
    to the buffer; others use the `pre_s` seconds before the trigger.
    """

    def __init__(
        self,
        out_dir,
        pad_ratio: float = 0.4,
        max_window_frames: int = 8,
        buffer_fps: float = 2.0,
        buffer_s: float = 30.0,
        pre_s: float = 2.5,
        jpeg_quality: int = 90,
    ):
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
        self.next_buffer_t = None

    def start(self, seq) -> None:
        """Call at the start of each sequence so frames never leak between sequences."""
        self.buffer.clear()
        self.next_buffer_t = None
        self.use_buffer = not getattr(seq, "paths", None)

    def observe(self, frame, t: float, index: int) -> None:
        """Offer every frame; a sampled subset is kept for later windows (videos only)."""
        if not self.use_buffer:
            return
        if self.next_buffer_t is not None and t < self.next_buffer_t - 1e-06:
            return
        ok, jpg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if ok:
            self.buffer.append((t, index, jpg))
            if self.next_buffer_t is None:
                self.next_buffer_t = t
            while self.next_buffer_t <= t + 1e-06:
                self.next_buffer_t += 1.0 / self.buffer_fps
        while self.buffer and self.buffer[0][0] < t - self.buffer_s:
            self.buffer.popleft()

    def _window(self, event: dict, t: float) -> tuple[float, float]:
        start = (
            event["evidence_window_s"][0]
            if "evidence_window_s" in event
            else t - self.pre_s
        )
        return (max(0.0, start, t - self.buffer_s), t)

    @staticmethod
    def _spread(items: list, n: int) -> list:
        """At most n items, evenly spread, always keeping the last one (the trigger frame)."""
        if len(items) <= n:
            return items
        keep = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
        return [items[i] for i in keep]

    def _window_frames(
        self, event: dict, frame, seq, t: float
    ) -> tuple[list[str], list[float]]:
        return self.window_frames(event, frame, seq, *self._window(event, t))

    def window_frames(
        self,
        event: dict,
        frame,
        seq,
        start: float,
        t: float,
        max_frames: int | None = None,
        tag: str = "",
    ) -> tuple[list[str], list[float]]:
        """Past frames in [start, t] (clamped to the buffer), at most `max_frames` evenly spread, ending at the
        trigger frame. `tag` keeps the files of several windows for the same event apart."""
        w0 = max(0.0, start, t - self.buffer_s)
        n = self.max_window_frames if max_frames is None else max(1, max_frames)
        eps = 1e-06
        paths = getattr(seq, "paths", None)
        if paths:
            pairs = [
                (ts, p)
                for p, ts in zip(seq.paths, seq.timestamps)
                if w0 - eps <= ts <= t + eps
            ]
            pairs = self._spread(pairs, n)
            return ([p for _, p in pairs], [round(ts, 2) for ts, _ in pairs])
        earlier = [
            (bt, idx, jpg) for bt, idx, jpg in self.buffer if w0 - eps <= bt < t - eps
        ]
        chosen = self._spread([*earlier, (t, event["frame_index"], None)], n)
        out_paths, times = ([], [])
        for k, (bt, idx, jpg) in enumerate(chosen):
            p = self.windows / f"{event['event_id']}{tag}_w{k:02d}_f{idx}.jpg"
            if jpg is None:
                cv2.imwrite(str(p), frame)
            else:
                p.write_bytes(jpg.tobytes())
            out_paths.append(str(p))
            times.append(round(bt, 2))
        return (out_paths, times)

    def save(self, event: dict, frame, seq, t: float | None = None) -> None:
        """`t` is the trigger frame's exact time (event timestamps are rounded)."""
        t = float(event["timestamp_s"]) if t is None else float(t)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = event["roi_box"]
        px, py = ((x2 - x1) * self.pad_ratio, (y2 - y1) * self.pad_ratio)
        cx1, cy1 = (max(0, int(x1 - px)), max(0, int(y1 - py)))
        cx2, cy2 = (min(w, int(x2 + px)), min(h, int(y2 + py)))
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
