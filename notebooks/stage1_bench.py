"""Ground-truth loaders and correctness metrics for the Stage 1 gate.

The gate is judged on what the VLM will receive, at three levels:
  * event recall   - of the real anomalies (runs of positive frames), how many got at least
                     one candidate whose clip window overlaps them. A missed event here can
                     never be recovered by the VLM, so this is the number to protect.
  * crop recall    - as above, but the candidate's crop must also contain the GT actor/object.
  * candidate precision - fraction of candidates that are near a GT anomaly in time and space.
  * frame metrics  - each frame is "forwarded" if it falls inside any candidate window;
                     compared against per-frame labels. Review reduction = 1 - forwarded share.
A "person present" baseline (forward every frame with a person) is reported alongside, so you
can see what the rules add over simply detecting people.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stage1_gate import (GateConfig, ImageSequence, RunResult, Stage1Gate, VideoSequence, intersection_over,
                         list_images, merge_windows, replay_rules)


@dataclass
class GTSequence:
    category: str
    seq: object                      # VideoSequence | ImageSequence
    gt: dict[int, list[list[float]]]  # frame index -> positive boxes (xyxy px); absent = normal frame
    tag: str = ""                     # sub-label for breakdowns, e.g. littering action class


# --------------------------------------------------------------------------- dataset adapters

def littering_sequences(root, splits=("valid", "test"), fps: float = 10.0,
                        max_clips: int | None = None) -> list[GTSequence]:
    """Rebuilt littering clips (scripts/rebuild_littering_clips.py): <root>/<split>/<clip_id>/frame_NN.jpg.
    Every class (dropping, dumping, putting-down, throwing, tossing) is a littering action and every
    labelled box is the littering actor, so each clip is one positive event. The set measures recall
    only; measure false alarms on normal footage (e.g. Avenue training videos)."""
    root = Path(root)
    action = {}
    with open(root / "clips.csv") as fh:
        for row in csv.DictReader(fh):
            action[row["clip_id"]] = row["class"]
    out = []
    for split in splits:
        for clip_dir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            frames = list_images(clip_dir)
            h, w = cv2.imread(frames[0]).shape[:2]
            gt = {}
            for i, p in enumerate(frames):
                boxes = []
                for line in Path(p).with_suffix(".txt").read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        _, x, y, bw, bh = map(float, parts)
                        boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                if boxes:
                    gt[i] = boxes
            out.append(GTSequence("littering", ImageSequence(frames, fps=fps, seq_id=clip_dir.name), gt,
                                  tag=action[clip_dir.name]))
            if max_clips and len(out) >= max_clips:
                return out
    return out


_SMOKE_NAME = re.compile(r"frame_(c\d+)_(\d+)_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)-(\d+)_(AM|PM)")


def smoking_sequences(root, splits=("valid",), max_gap_s: float = 15.0, positive="Smoking") -> list[GTSequence]:
    """Roboflow shuffled the original CCTV frames (~1 fps) into train/valid. The original
    filename (images[].extra.name) encodes camera, date and wall-clock time, which is enough
    to rebuild the temporal sequences. A new sequence starts at any gap > max_gap_s.
    Use splits=("train", "valid") for more data while YOLO is COCO-pretrained; use only
    "valid" once you fine-tune on the train split."""
    frames = []
    for split in splits:
        ann_path = Path(root) / split / "_annotations.coco.json"
        data = json.loads(ann_path.read_text())
        cats = {c["id"]: c["name"] for c in data["categories"]}
        boxes = defaultdict(list)
        for a in data["annotations"]:
            if cats.get(a["category_id"]) == positive:
                x, y, w, h = a["bbox"]
                boxes[a["image_id"]].append([x, y, x + w, y + h])
        for im in data["images"]:
            m = _SMOKE_NAME.match(im.get("extra", {}).get("name", im["file_name"]))
            if not m:
                continue
            cam, _, date, hh, mm, ss, ms, ap = m.groups()
            hour = int(hh) % 12 + (12 if ap == "PM" else 0)
            ts = dt.datetime.strptime(date, "%Y-%m-%d").timestamp() + hour * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
            frames.append((cam, date, ts, str(ann_path.parent / im["file_name"]), boxes.get(im["id"], [])))
    groups = defaultdict(list)
    for f in frames:
        groups[(f[0], f[1])].append(f)
    out = []
    for (cam, date), fs in sorted(groups.items()):
        fs.sort(key=lambda f: f[2])
        runs, cur = [], [fs[0]]
        for prev, f in zip(fs, fs[1:]):
            if f[2] - prev[2] > max_gap_s:
                runs.append(cur)
                cur = []
            cur.append(f)
        runs.append(cur)
        for k, run in enumerate(runs):
            seq = ImageSequence([f[3] for f in run], timestamps=[f[2] for f in run], seq_id=f"{cam}_{date}#{k}")
            gt = {i: f[4] for i, f in enumerate(run) if f[4]}
            out.append(GTSequence("smoking", seq, gt))
    return out


def _mask_boxes(mask, min_area: int = 20):
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h, a in stats[1:] if a >= min_area]


def avenue_sequences(root, gt_dir, include_train_normals: bool = True, stride: int = 1,
                     max_videos: int | None = None) -> list[GTSequence]:
    """CUHK Avenue. Anomalies are running, throwing objects, wrong-direction walking and odd
    actions near the camera; it is a general-anomaly set, not a loitering set. Frame labels
    come from testing_label_mask/<n>_label.mat (official ground_truth_demo.zip); the
    data/loitering/*_vol/*.mat files are downsampled video volumes, not labels."""
    import scipy.io as sio

    out = []
    for avi in sorted((Path(root) / "testing_videos").glob("*.avi"))[:max_videos]:
        label = Path(gt_dir) / f"{int(avi.stem)}_label.mat"
        if not label.exists():
            print(f"skip {avi.name}: no {label.name}")
            continue
        masks = sio.loadmat(str(label))["volLabel"][0]
        gt = {i: _mask_boxes(m) for i, m in enumerate(masks) if m.any()}
        del masks
        out.append(GTSequence("avenue", VideoSequence(avi, stride=stride, seq_id=f"test/{avi.stem}"), gt))
    if include_train_normals:
        for avi in sorted((Path(root) / "training_videos").glob("*.avi"))[:max_videos]:
            out.append(GTSequence("avenue", VideoSequence(avi, stride=stride, seq_id=f"train/{avi.stem}"), {}))
    return out


# --------------------------------------------------------------------------- running

def run_perception(gate: Stage1Gate, gt_seqs: list[GTSequence], verbose: bool = True) -> list[tuple[GTSequence, RunResult]]:
    out = []
    for i, g in enumerate(gt_seqs):
        res = gate.run(g.seq)
        out.append((g, res))
        if verbose:
            print(f"\r[{g.category}] {i + 1}/{len(gt_seqs)} {g.seq.seq_id:<28} frames={res.n_frames:<5} "
                  f"{res.mean_latency_ms:6.1f} ms/frame", end="", flush=True)
    if verbose:
        print()
    return out


# --------------------------------------------------------------------------- metrics

def box_hit(roi, gt_box, pad_ratio: float = 0.4, min_cover: float = 0.3) -> bool:
    """Does the crop the VLM receives (ROI padded like HandoffWriter) contain enough of the GT box?"""
    x1, y1, x2, y2 = roi
    px, py = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    return intersection_over(gt_box, [x1 - px, y1 - py, x2 + px, y2 + py]) >= min_cover


def gt_events(times: list[float], idxs: list[int], gt: dict, fps: float, gap_s: float) -> list[dict]:
    events, cur = [], None
    for t, i in zip(times, idxs):
        if i not in gt:
            continue
        if cur is not None and t - cur["t1"] <= gap_s + 1.0 / fps:
            cur["t1"] = t
            cur["frames"].append((t, i))
        else:
            cur = {"t0": t, "t1": t, "frames": [(t, i)]}
            events.append(cur)
    return events


def evaluate_run(g: GTSequence, res: RunResult, events: list[dict] | None = None, tol_s: float = 1.0,
                 event_gap_s: float = 1.0, min_cover: float = 0.3) -> dict:
    events = res.events if events is None else events
    obs = res.observations
    times = [o.t for o in obs]
    idxs = [o.index for o in obs]
    merged = merge_windows([e["clip_window_s"] for e in events])
    fwd = np.array([any(a <= t <= b for a, b in merged) for t in times], dtype=bool)
    pos = np.array([i in g.gt for i in idxs], dtype=bool)
    person = np.array([len(o.persons) > 0 for o in obs], dtype=bool)

    pos_frames = [(t, i) for t, i in zip(times, idxs) if i in g.gt]
    cand_correct, by_type = 0, defaultdict(lambda: [0, 0])
    for e in events:
        ok = any(abs(t - e["timestamp_s"]) <= tol_s and any(box_hit(e["roi_box"], b, min_cover=min_cover) for b in g.gt[i])
                 for t, i in pos_frames)
        cand_correct += ok
        by_type[e["event_type"]][0] += 1
        by_type[e["event_type"]][1] += ok

    gevs = gt_events(times, idxs, g.gt, res.fps, event_gap_s)
    temporal_hits = spatial_hits = 0
    caught_by, missed = Counter(), []
    for ge in gevs:
        t_hit = any(a <= ge["t1"] and b >= ge["t0"] for a, b in merged)
        hitters = [e for e in events if ge["t0"] - tol_s <= e["timestamp_s"] <= ge["t1"] + tol_s and any(
            abs(t - e["timestamp_s"]) <= tol_s and any(box_hit(e["roi_box"], b, min_cover=min_cover) for b in g.gt[i])
            for t, i in ge["frames"])]
        temporal_hits += t_hit
        spatial_hits += bool(hitters)
        for e in {e["event_type"] for e in hitters}:
            caught_by[e] += 1
        if not hitters:
            missed.append({"sequence": res.seq_id, "t0": round(ge["t0"], 2), "t1": round(ge["t1"], 2),
                           "frames": len(ge["frames"]), "temporal_hit": t_hit})

    return {
        "category": g.category, "tag": g.tag, "sequence": res.seq_id, "frames": len(obs), "duration_s": res.duration_s,
        "tp": int((fwd & pos).sum()), "fp": int((fwd & ~pos).sum()), "tn": int((~fwd & ~pos).sum()), "fn": int((~fwd & pos).sum()),
        "pp_tp": int((person & pos).sum()), "pp_fp": int((person & ~pos).sum()), "pp_fn": int((~person & pos).sum()),
        "gt_events": len(gevs), "ev_temporal": temporal_hits, "ev_spatial": spatial_hits,
        "candidates": len(events), "cand_correct": cand_correct,
        "by_type": {k: tuple(v) for k, v in by_type.items()}, "caught_by": dict(caught_by), "missed": missed,
        "latency_ms": res.mean_latency_ms, "stitched": res.n_stitched,
    }


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else float("nan")


def summarize(rows: list[dict], name: str) -> dict:
    s = Counter()
    for r in rows:
        for k in ("frames", "tp", "fp", "tn", "fn", "pp_tp", "pp_fp", "pp_fn", "gt_events", "ev_temporal",
                  "ev_spatial", "candidates", "cand_correct", "stitched"):
            s[k] += r[k]
    minutes = sum(r["duration_s"] for r in rows) / 60.0
    fwd = s["tp"] + s["fp"]
    pp_fwd = s["pp_tp"] + s["pp_fp"]
    return {
        "run": name,
        "sequences": len(rows),
        "frames": s["frames"],
        "gt_events": s["gt_events"],
        "event_recall_%": _pct(s["ev_temporal"], s["gt_events"]),
        "crop_recall_%": _pct(s["ev_spatial"], s["gt_events"]),
        "candidates": s["candidates"],
        "candidate_precision_%": _pct(s["cand_correct"], s["candidates"]),
        "candidates_per_min": round(s["candidates"] / minutes, 1) if minutes else float("nan"),
        "frame_recall_%": _pct(s["tp"], s["tp"] + s["fn"]),
        "frame_precision_%": _pct(s["tp"], fwd),
        "review_reduction_%": round(100.0 - _pct(fwd, s["frames"]), 1),
        "baseline_person_frame_recall_%": _pct(s["pp_tp"], s["pp_tp"] + s["pp_fn"]),
        "baseline_person_reduction_%": round(100.0 - _pct(pp_fwd, s["frames"]), 1),
        "positive_frame_share_%": _pct(s["tp"] + s["fn"], s["frames"]),
        "ms_per_frame": round(float(np.mean([r["latency_ms"] for r in rows])), 1) if rows else float("nan"),
        "tracks_stitched": s["stitched"],
    }


def per_rule_table(rows: list[dict]) -> pd.DataFrame:
    n, ok, caught = Counter(), Counter(), Counter()
    for r in rows:
        for k, (a, b) in r["by_type"].items():
            n[k] += a
            ok[k] += b
        caught.update(r["caught_by"])
    total = sum(r["gt_events"] for r in rows)
    return pd.DataFrame([{"rule": k, "candidates": n[k], "precision_%": _pct(ok[k], n[k]),
                          "gt_events_caught": caught[k], "share_of_gt_events_%": _pct(caught[k], total)}
                         for k in sorted(n)])


def evaluate(results: list[tuple[GTSequence, RunResult]], cfg: GateConfig | None = None, name: str = "",
             **kw) -> tuple[dict, list[dict]]:
    """Score cached results. Pass `cfg` to replay the rules with different settings first."""
    rows = []
    for g, res in results:
        events = replay_rules(res, cfg)[0] if cfg is not None else None
        rows.append(evaluate_run(g, res, events, **kw))
    return summarize(rows, name), rows


def sweep(results, base_cfg: GateConfig, param: str, values, **kw) -> pd.DataFrame:
    out = []
    for v in values:
        summary, _ = evaluate(results, replace(base_cfg, **{param: v}), name=f"{param}={v}", **kw)
        out.append(summary)
    return pd.DataFrame(out)
