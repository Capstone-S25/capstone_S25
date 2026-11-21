r"""End to end on the CPU deployment bundle: video -> the gate -> the screening ensemble.

The bundle (capstone-cpu-deployment-20261117, feat/vision-app-integration 4d2ac65) ships this
gate verbatim - identical GateConfig, byte-identical cue weights, only import paths rewritten - so
running it measures the real cascade rather than a reimplementation. Its own entry points need
Linux (deploy.py uses fcntl; the Gemma server is an ELF binary) and its data-pipeline VLM path
shells out to anomaly_eval, which the bundle does not contain. This drives the two stages
in-process instead: the bundle's Stage1Detector for the gate and its EnsembleStrategy for
screening, so both are the shipped code with the shipped weights.

Frames are decoded with OpenCV on media time and thinned to --fps by the bundle's own
FrameSelector, which is what the live pipeline does. The gate config is the bundle's
default_config(), NOT calib_chosen.json: they agree on all eleven calibrated fields except
reach_refire_move_h (1.0 against None, a deviation the bundle documents), and the bundle also sets
inview_s 60, pose_action_cooldown_s 15, smoke_min_hits 3, still_lying_wh 1.1,
still_size_longer_side and hires_every_s 1.0.

Scoring, per labelled incident:
  gate      a candidate whose roi_box covers at least --cover of the labelled subject box inside
            the event window (the roi is padded around the hand, so intersection-over-subject is
            used rather than IoU)
  cascade   that candidate also asks about the incident's behaviour AND the ensemble answers
            "anomaly" for it
A candidate on no labelled incident is counted separately, with the ensemble's verdict, so the
screening stage's effect on the front end's false alarms is visible. False alarms are reported per
person and per clip, never per hour: the clips differ in length.

Usage (from notebooks/):
  ..\.venv\Scripts\python exp/integration/bundle_e2e.py --split test --limit 5 --out OUT
  ..\.venv\Scripts\python exp/integration/bundle_e2e.py --split test --scene icvl/02 --out OUT
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--bundle", type=Path, default=Path(r"C:\capstone\friend_bundle"))
p.add_argument("--data", type=Path, default=Path(r"C:\Users\Admin\Downloads\capstone_datasets"))
p.add_argument("--split", default="test", choices=["test", "fit"])
p.add_argument("--scene", action="append", help="limit to these scenes (repeatable)")
p.add_argument("--clip", action="append", help="limit to these clip stems (repeatable)")
p.add_argument("--anomaly", action="append", help="limit to these labelled anomalies")
p.add_argument("--limit", type=int, help="first N clips only")
p.add_argument("--fps", type=float, default=5.0)
p.add_argument("--device", default="cuda:0", help="cpu, or a CUDA device (same weights either way)")
p.add_argument("--cover", type=float, default=0.5, help="share of the subject box a roi must cover")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--localized-gate-only", action="store_true",
               help="score localized candidates and their routes without screening")
p.add_argument("--retry-duration-after-dedup", action="store_true",
               help="let a duration rule retry after deduplication drops its alert")
p.add_argument("--tracker", choices=("bytetrack", "botsort"))
p.add_argument("--main-detector", choices=("yolo", "faster-rcnn"), default="yolo")
p.add_argument("--detector-imgsz", type=int, help="override only the main detector resolution")
p.add_argument("--pose-imgsz", type=int, help="override only the pose detector resolution")
p.add_argument("--audit-perception", action="store_true",
               help="measure subject-box and keypoint coverage against labelled event frames")
ARGS = p.parse_args()

BUNDLE = ARGS.bundle.resolve()
OUT = ARGS.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)

# The bundle's own offline defaults, set before vision_runtime is imported.
os.environ.update(
    VISION_MODELS_DIR=str(BUNDLE / "models"),
    VISION_RUNTIME_DIR=str(OUT / "runtime"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(OUT / "runtime" / "yolo"),
)
# E2E_CODE_ROOT lets a checkout of the integration branch supply the pipeline
# code while the weights still come from the bundle (they are not in Git).
CODE = Path(os.environ.get("E2E_CODE_ROOT", str(BUNDLE))).resolve()
sys.path[:0] = [str(CODE), str(CODE / "data-pipeline"),
                str(BUNDLE), str(BUNDLE / "data-pipeline"),
                str(NB), str(NB / "exp" / "eval")]
print("code root:", CODE)
print("models:   ", BUNDLE / "models")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from data_pipeline.detection.stage1 import (Stage1Detector, Stage1Frame,  # noqa: E402
                                            default_config, load_gate)
from data_pipeline.video_ingestion.streaming.selection import FrameSelector  # noqa: E402
from vision_runtime.ensemble import EnsembleStrategy  # noqa: E402

TARGET_TASK = {"littering": "littering", "smoking": "smoking", "loitering": "loitering",
               "rough sleeping": "rough_sleeping", "unattended items": "unattended_items",
               "left item": "unattended_items"}
WINDOW = {"action": (1.0, 1.0), "dwell": (1.0, 1.0), "object": (1.0, 1e9), "pose": (1.0, 1.0)}


def clip_incidents(clip):
    """Labelled incidents for one clip, in calib.py's shape (imported, not reimplemented)."""
    from calib import incidents
    lab = json.loads((ARGS.data / clip["label"]).read_text(encoding="utf-8", errors="ignore"))
    return incidents(clip, lab)


def subject_box(inc, t):
    """The labelled subject box at time t, or None when it is not tracked then."""
    s = inc["subject"]
    if isinstance(s, list):
        return s
    ts = np.asarray(s["ts"])
    if not len(ts):
        return None
    j = int(np.argmin(np.abs(ts - t)))
    return s["boxes"][j] if abs(float(ts[j]) - t) <= 0.5 else None


def covers(roi, box, need):
    """Share of `box` that `roi` covers."""
    if box is None:
        return 0.0
    x1, y1 = max(roi[0], box[0]), max(roi[1], box[1])
    x2, y2 = min(roi[2], box[2]), min(roi[3], box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return (x2 - x1) * (y2 - y1) / area


def on_incident(cand, inc):
    lo, hi = WINDOW.get(inc["kind"], (1.0, 1.0))
    t = float(cand["timestamp_s"])
    t0 = inc["t0"] - lo
    t1 = (inc.get("t1") if inc["kind"] != "object" else inc.get("t1", 1e9)) + hi
    if not (t0 <= t <= t1):
        return False
    return covers(cand["roi_box"], subject_box(inc, t), ARGS.cover) >= ARGS.cover


def gate_config():
    """The bundle's default gate, with the optional open-vocabulary item pass when asked.

    GateConfig.item_detector is None by default, so the bag rules see only COCO's
    backpack / handbag / suitcase. E2E_ITEM_DETECTOR points it at YOLOE weights and
    E2E_ITEM_CONF overrides item_conf, so the pass can be measured against the
    default rather than assumed.
    """
    cfg = default_config(load_gate(), device=ARGS.device)
    base = {}
    if ARGS.tracker:
        base["tracker"] = ARGS.tracker
    if ARGS.retry_duration_after_dedup:
        base["still_retry_after_dedup"] = True
    if os.environ.get("E2E_IMGSZ"):
        base["imgsz"] = int(os.environ["E2E_IMGSZ"])
    if os.environ.get("E2E_APPEAR_CONF"):
        base["appear_min_conf"] = float(os.environ["E2E_APPEAR_CONF"])
    if os.environ.get("E2E_ITEM_MIN_FRAMES"):
        base["item_min_frames"] = int(os.environ["E2E_ITEM_MIN_FRAMES"])
    for var, field, cast in (("E2E_SMOKE_MIN_HITS","smoke_min_hits",int),
                             ("E2E_REACH_MIN_H","reach_min_h",float),
                             ("E2E_HAND_MIN_AGE","hand_min_age_s",float),
                             ("E2E_POSE_THR","pose_action_thr",float),
                             ("E2E_CROP_THR","crop_check_thr",float)):
        if os.environ.get(var):
            base[field] = cast(os.environ[var])
    if os.environ.get("E2E_CROP_THR_BY_RULE"):
        # "reach_down=0.45,pose_action=0.45" - leaves rules not named at their own
        # stored threshold, so hand_to_face keeps the value its weights were fitted with.
        base["crop_check_thr"] = {
            k: float(v) for k, v in
            (kv.split("=") for kv in os.environ["E2E_CROP_THR_BY_RULE"].split(","))}
    if os.environ.get("E2E_RULES"):
        # DEFAULT_RULES omits "litter", so uses_bg is False and static_object_left -
        # the only rule that localises an abandoned object without the detector
        # classifying it as a bag - never fires.
        base["rules"] = tuple(os.environ["E2E_RULES"].split(","))
    if os.environ.get("E2E_ADD_RULES"):
        base["rules"] = cfg.rules + tuple(
            r for r in os.environ["E2E_ADD_RULES"].split(",") if r not in cfg.rules)
    if os.environ.get("E2E_LITTER_MAX_AREA"):
        base["litter_max_area"] = float(os.environ["E2E_LITTER_MAX_AREA"])
    if os.environ.get("E2E_LITTER_LOOKBACK"):
        base["litter_owner_lookback_s"] = float(os.environ["E2E_LITTER_LOOKBACK"])
    if base:
        cfg = dataclasses.replace(cfg, **base)
    weights = os.environ.get("E2E_ITEM_DETECTOR")
    if not weights:
        return cfg
    over = {"item_detector": weights}
    if os.environ.get("E2E_ITEM_CONF"):
        over["item_conf"] = float(os.environ["E2E_ITEM_CONF"])
    if os.environ.get("E2E_ITEM_SCENERY"):
        over["item_scenery"] = os.environ["E2E_ITEM_SCENERY"] not in ("0", "false", "")
    return dataclasses.replace(cfg, **over)


@lru_cache(maxsize=1)
def faster_rcnn_model():
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    checkpoint = BUNDLE / "models/detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    return model.eval().to(ARGS.device)


class FasterRCNNDetector:
    """Translate COCO detections to the Box format used by the existing tracker."""

    def __init__(self, yolo, is_main_pass):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

        categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
        self.yolo = yolo
        self.is_main_pass = is_main_pass
        self.names = yolo.names
        ids_by_name = {name: key for key, name in self.names.items()}
        self.class_ids = [ids_by_name.get(name, -1) for name in categories]

    def predict(self, frame, *, classes, conf, **kwargs):
        import torch
        from ultralytics.engine.results import Results

        if not self.is_main_pass(kwargs["imgsz"], classes):
            return self.yolo.predict(frame, classes=classes, conf=conf, **kwargs)
        with torch.inference_mode():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).to(ARGS.device).float() / 255
            result = faster_rcnn_model()([image])[0]
            lookup = torch.tensor(self.class_ids, device=result["labels"].device)
            ids = lookup[result["labels"]]
            requested = torch.tensor(classes, device=ids.device)
            keep = torch.isin(ids, requested) & (result["scores"] >= conf)
            data = torch.cat((result["boxes"][keep], result["scores"][keep, None],
                              ids[keep, None].float()), dim=1)
        return [Results(frame, path="", names=self.names, boxes=data)]


def run_clip(clip, ens, jl):
    """Decode one clip, run the gate, classify every candidate. -> (incidents, candidates)."""
    video = ARGS.data / clip["video"]
    if not video.is_file():
        return None
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = Stage1Detector(clip["stem"], OUT / "candidates", fps=ARGS.fps,
                         cfg=gate_config(),
                         # The integrated pipeline (Stage1Pipeline) runs a whole-scene
                         # review every 10 s; this harness defaulted to none, so it
                         # scored one scene window per clip instead of one per 10 s.
                         safety_scan_seconds=0 if ARGS.localized_gate_only else
                         float(os.environ.get("E2E_SAFETY_SCAN", "0")))
    incs = clip_incidents(clip)
    audit = {}

    def main_pass(size, classes):
        return not (det.cfg.hires_every_s and size == det.cfg.hires_imgsz and classes == [0])

    if ARGS.main_detector == "faster-rcnn":
        det.perception.det = FasterRCNNDetector(det.perception.det, main_pass)

    def override_size(model, size, only_when=None):
        predict = model.predict

        def predict_at_size(*args, **kwargs):
            if only_when is None or only_when(kwargs["imgsz"], kwargs["classes"]):
                kwargs["imgsz"] = size
            return predict(*args, **kwargs)

        model.predict = predict_at_size

    if ARGS.detector_imgsz is not None:
        override_size(det.perception.det, ARGS.detector_imgsz, main_pass)
    if ARGS.pose_imgsz is not None and det.perception.pose is not None:
        override_size(det.perception.pose, ARGS.pose_imgsz)

    if ARGS.audit_perception:
        observe = det.perception.observe

        def audit_observation(*args, **kwargs):
            obs = observe(*args, **kwargs)
            for inc in incs:
                if not inc["t0"] - 1 <= obs.t <= inc["t1"] + 1:
                    continue
                box = subject_box(inc, obs.t)
                if box is None:
                    continue
                counts = audit.setdefault(inc["id"], Counter())
                counts["labelled_frames"] += 1
                subjects = obs.items if inc["kind"] == "object" else obs.persons
                if not subjects:
                    continue
                subject = max(subjects, key=lambda row: covers(row["box"], box, ARGS.cover))
                if covers(subject["box"], box, ARGS.cover) < ARGS.cover:
                    continue
                counts["matched_frames"] += 1
                counts[f'track_{subject["raw_id"]}'] += 1
                pose = subject.get("kpts")
                if pose is not None:
                    counts["pose_frames"] += 1
                    if max(pose[0:5, 2]) >= .35 and max(pose[9:11, 2]) >= .35:
                        counts["face_and_wrist_frames"] += 1
            return obs

        det.perception.observe = audit_observation
    sel, session, cands, i = FrameSelector(target_fps=ARGS.fps), uuid4(), [], 0
    t_gate = 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ms = round(i / vfps * 1000)
        i += 1
        if not sel.should_select(ms):
            continue
        t0 = time.perf_counter()
        cands += det.process(Stage1Frame(None, clip["stem"], session, i, ms, img))
        t_gate += time.perf_counter() - t0
    cands += det.finish()
    cap.release()
    frames = max(1, round(i / vfps * ARGS.fps))

    t_vlm = 0.0
    if ARGS.localized_gate_only:
        cands = [c for c in cands if c["event_type"] not in {"final_scene_review", "periodic_review"}]
    for c in cands:
        if ARGS.localized_gate_only:
            if (c["event_type"] in {"pose_action", "reach_down", "hand_to_face"}
                    and "loitering" not in c["tasks"] and c["timestamp_s"] >= 19.5):
                c["tasks"].append("loitering")
        else:
            t0 = time.perf_counter()
            try:
                c["screening"] = ens.classify(c)["predictions"]
            except Exception as e:
                c["screening"] = {"_error": f"{type(e).__name__}: {e}"}
            t_vlm += time.perf_counter() - t0
        c["on_incidents"] = [inc["id"] for inc in incs if on_incident(c, inc)]
        jl.write(json.dumps({k: v for k, v in c.items() if k != "views"}) + "\n")
    return {"clip": clip, "incidents": incs, "candidates": cands,
            "gate_s": t_gate, "vlm_s": t_vlm, "frames": frames,
            "perception_audit": audit,
            "duration_s": float(clip.get("duration_s") or 0.0)}


def main():
    clips = [c for c in json.loads((NB / "exp" / "eval" / "splits.json").read_text())["clips"]
             if c["split"] == ARGS.split
             and (not ARGS.clip or c["stem"] in ARGS.clip)
             and (not ARGS.scene or c["scene"] in ARGS.scene)
             and (not ARGS.anomaly or c["anomaly"] in ARGS.anomaly)]
    clips = clips[:ARGS.limit] if ARGS.limit else clips
    print(f"{len(clips)} clips on the {ARGS.split} side, device {ARGS.device}", flush=True)

    ens = None
    if not ARGS.localized_gate_only:
        ens = EnsembleStrategy(device=ARGS.device, directory=BUNDLE / "models"
                               / os.environ.get("E2E_ENSEMBLE", "ensemble-20261116"))
        print(f"screening model {ens.name}, operating point {ens.operating_point}, "
              f"held-out macro F1 {ens.metadata['held_out_macro_f1']:.3f}\n", flush=True)

    rows, t0 = [], time.time()
    with (OUT / "candidates.jsonl").open("w", encoding="utf-8") as jl:
        for k, clip in enumerate(clips, 1):
            r = run_clip(clip, ens, jl)
            if r is None:
                print(f"  [{k}/{len(clips)}] {clip['stem']}: video missing", flush=True)
                continue
            rows.append(r)
            hit = sum(1 for c in r["candidates"] if c["on_incidents"])
            print(f"  [{k}/{len(clips)}] {clip['stem']:<34} {clip['anomaly']:<16} "
                  f"{len(r['incidents'])} inc, {len(r['candidates'])} cand ({hit} on an incident), "
                  f"gate {r['gate_s']:.1f}s vlm {r['vlm_s']:.1f}s", flush=True)

    localized = None
    if ARGS.localized_gate_only:
        totals, matched, routed = Counter(), Counter(), Counter()
        for row in rows:
            for incident in row["incidents"]:
                target = incident["target"]
                totals[target] += 1
                hits = [c for c in row["candidates"] if incident["id"] in c["on_incidents"]]
                matched[target] += bool(hits)
                routed[target] += any(TARGET_TASK[target] in c["tasks"] for c in hits)
        localized = {target: {"total": totals[target], "matched": matched[target],
                              "routed": routed[target]} for target in sorted(totals)}
        print(json.dumps(localized, indent=2))
    else:
        summarize(rows, time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(
        {"clips": len(rows), "split": ARGS.split, "device": ARGS.device, "fps": ARGS.fps,
         "code_root": str(CODE), "screening_model": ens.name if ens else None,
         "operating_point": ens.operating_point if ens else None,
         "localized_gate_recall": localized,
         "perception_settings": {"main_detector": ARGS.main_detector, "tracker_override": ARGS.tracker, "detector_imgsz_override": ARGS.detector_imgsz,
                                 "pose_imgsz_override": ARGS.pose_imgsz},
         "per_clip": [{"stem": r["clip"]["stem"], "scene": r["clip"]["scene"],
                       "anomaly": r["clip"]["anomaly"], "incidents": len(r["incidents"]),
                       "candidates": len(r["candidates"]),
                       "perception_audit": r["perception_audit"],
                       "gate_s": round(r["gate_s"], 2), "vlm_s": round(r["vlm_s"], 2)}
                      for r in rows]}, indent=1))
    print(f"\nwrote {OUT / 'candidates.jsonl'} and {OUT / 'summary.json'}")


def summarize(rows, wall):
    gate_caught = Counter(); casc_caught = Counter(); total = Counter()
    why = Counter()
    for r in rows:
        for inc in r["incidents"]:
            tgt = inc["target"]; task = TARGET_TASK.get(tgt)
            total[tgt] += 1
            on = [c for c in r["candidates"] if inc["id"] in c["on_incidents"]]
            if not on:
                continue
            gate_caught[tgt] += 1
            asked = [c for c in on if task in (c.get("tasks") or ())]
            if not asked:
                why[f"{tgt}: gate fired but the rule never asks about {task}"] += 1
                continue
            verdicts = [c["screening"].get(task, {}) for c in asked]
            if any(v.get("decision") == "anomaly" for v in verdicts):
                casc_caught[tgt] += 1
            elif all(v.get("decision") == "unknown" for v in verdicts):
                why[f"{tgt}: screening had too little history"] += 1
            else:
                why[f"{tgt}: screening called it normal"] += 1

    print("\n" + "=" * 78)
    print("CASCADE RECALL  (gate = a candidate on the labelled subject; cascade = screening agreed)")
    print("=" * 78)
    print("%-18s %8s %14s %18s" % ("behaviour", "labelled", "gate caught", "gate + screening"))
    for tgt in sorted(total):
        g, c, n = gate_caught[tgt], casc_caught[tgt], total[tgt]
        print("%-18s %8d %8d %5.0f%% %12d %5.0f%%" % (tgt, n, g, 100*g/n, c, 100*c/n))
    n = sum(total.values()); g = sum(gate_caught.values()); c = sum(casc_caught.values())
    if n:
        print("%-18s %8d %8d %5.0f%% %12d %5.0f%%" % ("ALL", n, g, 100*g/n, c, 100*c/n))
    if why:
        print("\nwhere the cascade lost an event the gate had:")
        for k, v in why.most_common():
            print(f"    {v:4d}  {k}")

    # the false-alarm side, per candidate and per clip - never per hour.
    # A scheduled whole-scene review is not a gate alert: finish() emits one per clip, so on short
    # clips they would otherwise dominate the count. Reported separately.
    SCHEDULED = {"final_scene_review", "periodic_review"}

    def anomaly(c):
        return any(v.get("decision") == "anomaly"
                   for v in c["screening"].values() if isinstance(v, dict))

    alerts = [c for r in rows for c in r["candidates"] if c["event_type"] not in SCHEDULED]
    sched = [c for r in rows for c in r["candidates"] if c["event_type"] in SCHEDULED]
    off = [c for c in alerts if not c["on_incidents"]]
    kept = [c for c in off if anomaly(c)]
    allc = sum(len(r["candidates"]) for r in rows)
    n_clip = max(1, len(rows))
    print()
    print(f"FRONT-END LOAD over {len(rows)} clips")
    print(f"  gate rule alerts: {len(alerts)} ({len(alerts)/n_clip:.1f} per clip)")
    print(f"    on a labelled incident:  {len(alerts) - len(off)}")
    print(f"    on no labelled incident: {len(off)} ({len(off)/n_clip:.1f} per clip)")
    print(f"    of those, screening still called an anomaly: {len(kept)} "
          f"({100*len(kept)/max(1,len(off)):.0f}% kept, {len(kept)/n_clip:.1f} per clip)")
    print(f"  scheduled whole-scene reviews: {len(sched)}, "
          f"{sum(1 for c in sched if anomaly(c))} called an anomaly (one per clip from finish())")
    print("  (per clip, not per hour: the clips differ in length. Per person would need the "
          "one-alert-one-person attribution, which this harness does not do.)")
    by = Counter(c["event_type"] for c in off)
    print("  rules firing off-incident:", dict(by.most_common(8)))

    gs = sum(r["gate_s"] for r in rows); vs = sum(r["vlm_s"] for r in rows)
    fr = sum(r["frames"] for r in rows)
    print(f"\nCOST ({ARGS.device})\n  gate {gs:.0f}s over ~{fr} selected frames "
          f"({1000*gs/max(1,fr):.0f} ms/frame)")
    print(f"  screening {vs:.0f}s over {allc} candidates "
          f"({vs/max(1,allc):.2f} s/candidate)")
    print(f"  wall clock {wall:.0f}s")


if __name__ == "__main__":
    main()
