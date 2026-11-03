"""Fine-tune YOLO26s to see bags in CCTV better, keeping the 80 COCO classes so stage1_gate needs no change.

Data (data/bag_finetune/, YOLO format, 80 COCO names):
  roboflow   luggage_and_person v2 (CC BY 4.0), minus every image matched to ABODA (bag_data_dedup.py; ABODA is the
             test set). person -> 0; each luggage box -> the COCO bag class (24 backpack / 26 handbag / 28 suitcase)
             the original model predicts on it (IoU >= 0.5), else 28. Objects the dataset does not label (other COCO
             classes, unlabelled people and bags) are pseudo-labelled by the original yolo26s at conf >= PSEUDO_CONF.
  replay     coco128 (original COCO labels) and REPLAY_FRAMES CCTV frames from tune-only footage (Avenue odd training
             videos, IWDD train), pseudo-labelled by the original model, so people and other classes are not forgotten.
Train: yolo26s.pt init, backbone frozen (FREEZE layers), EPOCHS at 640 on MPS; val = Roboflow valid (non-ABODA).
Eval:  ABODA bag-found rate per frame (object_detectors.py frames and test, conf 0.2) and clutter detections, for the
       original and fine-tuned weights; then Stage 1 perception with the new weights (eval_gate).
Usage (from notebooks/): ../.venv/bin/python exp/newcam/bag_finetune.py build|train|eval|gate
"""
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
D = NB.parent / "data"
sys.path[:0] = [str(NB), str(HERE)]

from stage1_gate import iou  # noqa: E402

SRC = D / "roboflow_bags" / "luggage_and_person_v2"
OUT = D / "bag_finetune"
RUNS = NB / "bench_cache" / "bag_finetune_runs"
WEIGHTS = NB / "weights" / "yolo26s_bags.pt"
BAGS = (24, 26, 28)
PSEUDO_CONF = 0.4
REPLAY_FRAMES = 300
EPOCHS = 30
FREEZE = 10


def write_labels(path, rows):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def xyxy(c, x, y, w, h, W, H):
    return [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]


def to_yolo(b, W, H):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def build():
    from ultralytics import YOLO
    model = YOLO(str(NB / "yolo26s.pt"))
    names = model.names
    overlap = set(json.loads((D / "roboflow_bags" / "aboda_overlap.json").read_text()))
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {"roboflow": {}, "replay": 0}
    src_names = ["luggage", "person"]
    for split, dst in (("train", "train"), ("valid", "val")):
        (OUT / dst / "images").mkdir(parents=True)
        (OUT / dst / "labels").mkdir(parents=True)
        kept = dropped = 0
        for img_path in sorted((SRC / split / "images").glob("*.jpg")):
            if str(img_path.relative_to(D / "roboflow_bags")) in overlap:
                dropped += 1
                continue
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            gt = []
            lab = SRC / split / "labels" / f"{img_path.stem}.txt"
            for line in (lab.read_text().splitlines() if lab.exists() else []):
                p = line.split()
                if len(p) == 5:
                    gt.append((src_names[int(p[0])], xyxy(*map(float, p), W, H)))
            r = model.predict(img, imgsz=640, conf=0.05, verbose=False, device="mps")[0]
            preds = [(int(c), b.tolist(), float(s)) for c, b, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)]
            rows = []
            for kind, b in gt:
                if kind == "person":
                    rows.append((0, *to_yolo(b, W, H)))
                else:
                    best = max(((iou(b, pb), c) for c, pb, _ in preds if c in BAGS), default=(0, 28))
                    rows.append((best[1] if best[0] >= 0.5 else 28, *to_yolo(b, W, H)))
            gt_boxes = [b for _, b in gt]
            for c, pb, s in preds:
                if s >= PSEUDO_CONF and all(iou(pb, g) < 0.5 for g in gt_boxes):
                    rows.append((c, *to_yolo(pb, W, H)))
            shutil.copy(img_path, OUT / dst / "images" / img_path.name)
            write_labels(OUT / dst / "labels" / f"{img_path.stem}.txt", rows)
            kept += 1
        stats["roboflow"][dst] = {"kept": kept, "dropped_aboda": dropped}

    # replay: coco128 with original labels
    from ultralytics.utils.downloads import download
    coco = OUT / "_coco128"
    download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", dir=coco)
    for p in sorted((coco / "coco128" / "images" / "train2017").glob("*.jpg")):
        lab = coco / "coco128" / "labels" / "train2017" / f"{p.stem}.txt"
        shutil.copy(p, OUT / "train" / "images" / f"coco_{p.name}")
        shutil.copy(lab, OUT / "train" / "labels" / f"coco_{p.stem}.txt") if lab.exists() else \
            write_labels(OUT / "train" / "labels" / f"coco_{p.stem}.txt", [])
    shutil.rmtree(coco)

    # replay: CCTV frames from tune-only footage, pseudo-labelled
    sys.path.insert(0, str(NB / "exp" / "newcam"))
    from loaders import iwdd_sequences
    vids = sorted((D / "loitering" / "training_videos").glob("*.avi"))
    vids = [v for v in vids if int(v.stem) % 2 == 1] + [g.seq.path for g in iwdd_sequences("train")]
    rng = random.Random(0)
    for k in range(REPLAY_FRAMES):
        v = str(rng.choice(vids))
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, rng.randrange(max(1, n)))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        H, W = img.shape[:2]
        r = model.predict(img, imgsz=640, conf=PSEUDO_CONF, verbose=False, device="mps")[0]
        rows = [(int(c), *to_yolo(b.tolist(), W, H)) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
        name = f"cctv_{k:04d}"
        cv2.imwrite(str(OUT / "train" / "images" / f"{name}.jpg"), img)
        write_labels(OUT / "train" / "labels" / f"{name}.txt", rows)
        stats["replay"] += 1
    (OUT / "data.yaml").write_text(json.dumps({"path": str(OUT), "train": "train/images", "val": "val/images",
                                               "names": {int(k): v for k, v in names.items()}}, indent=1))
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def train():
    from ultralytics import YOLO
    model = YOLO(str(NB / "yolo26s.pt"))
    model.train(data=str(OUT / "data.yaml"), epochs=EPOCHS, imgsz=640, batch=16, device="mps", freeze=FREEZE,
                lr0=0.002, patience=10, project=str(RUNS), name="yolo26s_bags", exist_ok=True, plots=False,
                verbose=False, workers=4)
    best = RUNS / "yolo26s_bags" / "weights" / "best.pt"
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    print("wrote", WEIGHTS)


def evaluate():
    """ABODA bag-found rate and clutter, original vs fine-tuned (same frames and test as object_detectors.py)."""
    from object_detectors import COCO_BAGS, CONFS, frames
    from stage1_gate import intersection_over
    from ultralytics import YOLO
    pos, neg = frames()
    out = {}
    for name, w in (("yolo26s (original)", NB / "yolo26s.pt"), ("yolo26s fine-tuned on bags", WEIGHTS)):
        m = YOLO(str(w))
        found = {c: [] for c in CONFS}
        per_video = {}
        for v, t, img, box in pos:
            r = m.predict(img, imgsz=640, conf=min(CONFS), verbose=False, device="cpu")[0]
            confs = [float(s) for c, b, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf) if int(c) in COCO_BAGS and
                     (intersection_over(box, b.tolist()) >= 0.3 or intersection_over(b.tolist(), box) >= 0.3)]
            best = max(confs, default=0.0)
            for c in CONFS:
                found[c].append(best >= c)
            per_video.setdefault(v, []).append(best)
        clutter = {c: 0 for c in CONFS}
        for v, t, img in neg:
            r = m.predict(img, imgsz=640, conf=min(CONFS), verbose=False, device="cpu")[0]
            for c in CONFS:
                clutter[c] += int(sum(1 for k, s in zip(r.boxes.cls, r.boxes.conf) if int(k) in COCO_BAGS and s >= c))
        out[name] = {"bag_found_%frames": {str(c): round(100 * np.mean(f), 1) for c, f in found.items()},
                     "videos_bag_found_>=30%_frames@0.2": int(sum(np.mean(np.array(b) >= 0.2) >= 0.3 for b in per_video.values())),
                     "videos": len(per_video),
                     "clutter_bag_dets_per_frame": {str(c): round(clutter[c] / len(neg), 2) for c in CONFS}}
        print(name, json.dumps(out[name]), flush=True)
    (HERE / "bag_finetune_aboda.json").write_text(json.dumps(out, indent=1))


def gate():
    """Stage 1 with the fine-tuned detector: ABODA (unseen), littering held-out and Avenue, vs the cached originals."""
    import pickle
    from functools import partial

    from exp_harness import baseline
    from loaders import aboda_sequences
    from stage1_bench import avenue_sequences, evaluate_run, littering_sequences, run_perception, summarize
    from stage1_gate import DEFAULT_RULES, GateConfig, Stage1Gate, intersection_over

    cache = NB / "bench_cache" / "models" / "s_bags"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = GateConfig(det_weights=str(WEIGHTS), pose_weights=str(NB / "yolo26s-pose.pt"), device="mps")
    jobs = {"aboda": aboda_sequences,
            "litter_heldout": lambda: littering_sequences(D / "littering", splits=("valid", "test")),
            "avenue": lambda: avenue_sequences(D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" /
                                               "testing_label_mask", stride=2)}
    for name, build_seqs in jobs.items():
        if not (cache / f"{name}.pkl").exists():
            (cache / f"{name}.pkl").write_bytes(pickle.dumps(run_perception(Stage1Gate(cfg), build_seqs(), verbose=False)))
            print("perception", name, "done", flush=True)

    m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",)))
    lab = json.loads((HERE / "aboda_labels.json").read_text())
    num = lambda g: int(g.seq.seq_id.split("/")[-1])  # noqa: E731
    out = {}
    for label, d in (("original", NB / "bench_cache" / "models" / "s"), ("fine-tuned", cache)):
        load = lambda n, d=d: pickle.loads((d / f"{n}.pkl").read_bytes())  # noqa: E731
        caught = seen = total = 0
        for g, r in load("aboda"):
            for e in lab[r.seq_id.split("/")[-1]]["events"]:
                total += 1
                b = e["box"]
                caught += any(x["timestamp_s"] >= e["t_put_down"] - 2 and intersection_over(b, x["roi_box"]) >= 0.3
                              for x in m(r, "aboda"))
                seen += any(o.t >= e["t_unattended"] and any(intersection_over(b, it["box"]) >= 0.3 or
                                                            intersection_over(it["box"], b) >= 0.3 for it in o.items)
                            for o in r.observations)
        av = load("avenue")
        normal = [x for x in av if x[0].seq.seq_id.startswith("train/") and num(x[0]) % 2 == 0]
        test = [x for x in av if x[0].seq.seq_id.startswith("test/") and num(x[0]) % 2 == 0]
        lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in load("litter_heldout")], "l")
        a = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in test], "a")
        mins = sum(r.duration_s for _, r in normal) / 60
        items_fa = sum(len([e for e in m(r, "normal") if e["event_type"] in ("unattended_object", "object_appeared")])
                       for _, r in normal) / mins
        out[label] = {"aboda_bags_caught": f"{caught}/{total}", "aboda_bag_ever_detected": f"{seen}/{total}",
                      "littering_heldout_event/crop_%": f"{lit['event_recall_%']}/{lit['crop_recall_%']}",
                      "avenue_heldout_event/crop_%": f"{a['event_recall_%']}/{a['crop_recall_%']}",
                      "normal_FA_per_min": round(sum(len(m(r, "normal")) for _, r in normal) / mins, 2),
                      "normal_bag_alarms_per_min": round(items_fa, 2)}
        print(label, json.dumps(out[label]), flush=True)
    (HERE / "bag_finetune_gate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "train": train, "eval": evaluate, "gate": gate}[sys.argv[1]]()
