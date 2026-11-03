"""Can a different detector see the abandoned objects that COCO YOLO26s misses?  (ABODA, labelled by aboda_labels.json)

For each labelled event: frames every 1 s while the object is unattended (positives), plus frames every 2 s from the
start of the video until 3 s before anything is put down (clutter check: any "object" detection there is something
that is not an abandoned object, i.e. a source of false alarms).
Detectors: COCO yolo26s (bag classes: backpack, handbag, suitcase), open-vocabulary YOLOE-26s prompted with
object words, and YOLOv8s trained on Open Images V7 (bag classes below), each at 640 and 1280 input. Reports hit rate on the labelled object box at several confidence
thresholds, clutter detections per frame and CPU ms per frame.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/object_detectors.py [detector ...]   (default: all)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]
from loaders import ABODA  # noqa: E402
from stage1_gate import intersection_over  # noqa: E402

PROMPTS = ["bag", "backpack", "handbag", "suitcase", "box", "tote bag", "luggage", "case", "package"]
COCO_BAGS = {24, 26, 28}
# Open Images V7: Backpack, Box, Briefcase, Handbag, Luggage and bags, Plastic bag, Suitcase
OIV7_BAGS = {15, 62, 66, 237, 318, 394, 503}
CONFS = (0.05, 0.1, 0.2, 0.3)


def frames():
    lab = json.loads((HERE / "aboda_labels.json").read_text())
    pos, neg = [], []
    for v, l in lab.items():
        evs = l["events"]
        if not evs:
            continue
        cap = cv2.VideoCapture(str(ABODA / f"{v}.avi"))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

        def grab(t):
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * t)
            ok, img = cap.read()
            return img if ok else None

        for e in evs:
            end = e["t_removed"] if e["t_removed"] is not None else dur - 0.5
            for t in np.arange(e["t_unattended"], end, 1.0):
                img = grab(t)
                if img is not None:
                    pos.append((v, t, img, e["box"]))
        first = min(e["t_put_down"] for e in evs)
        for t in np.arange(1.0, max(1.0, first - 3.0), 2.0):
            img = grab(t)
            if img is not None:
                neg.append((v, t, img))
        cap.release()
    return pos, neg


def detectors():
    from ultralytics import YOLO, YOLOE
    coco = YOLO(str(NB / "yolo26s.pt"))
    ye = YOLOE(str(NB / "yoloe-26s-seg.pt"))
    ye.set_classes(PROMPTS)

    def run_coco(img, sz):
        r = coco.predict(img, imgsz=sz, conf=min(CONFS), verbose=False, device="cpu")[0]
        keep = [i for i, c in enumerate(r.boxes.cls.numpy().astype(int)) if c in COCO_BAGS]
        return r.boxes.xyxy.numpy()[keep], r.boxes.conf.numpy()[keep]

    def run_ye(img, sz):
        r = ye.predict(img, imgsz=sz, conf=min(CONFS), verbose=False, device="cpu")[0]
        return r.boxes.xyxy.numpy(), r.boxes.conf.numpy()

    oiv7 = YOLO("yolov8s-oiv7.pt")  # downloaded by Ultralytics on first use

    def run_oiv7(img, sz):
        r = oiv7.predict(img, imgsz=sz, conf=min(CONFS), verbose=False, device="cpu")[0]
        keep = [i for i, c in enumerate(r.boxes.cls.numpy().astype(int)) if c in OIV7_BAGS]
        return r.boxes.xyxy.numpy()[keep], r.boxes.conf.numpy()[keep]

    return {"coco_yolo26s": run_coco, "yoloe_26s_prompts": run_ye, "oiv7_yolov8s": run_oiv7}


def main():
    pos, neg = frames()
    print(f"{len(pos)} unattended-object frames, {len(neg)} clutter frames", flush=True)
    path = HERE / "object_detectors_aboda.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name, det in detectors().items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        for sz in (640, 1280):
            det(pos[0][2], sz)  # warm-up
            hits = {c: [] for c in CONFS}
            per_video = {}
            t0 = time.perf_counter()
            for v, t, img, box in pos:
                xyxy, conf = det(img, sz)
                best = max((c for b, c in zip(xyxy, conf) if intersection_over(box, list(b)) >= 0.3 or intersection_over(list(b), box) >= 0.3), default=0.0)
                for c in CONFS:
                    hits[c].append(best >= c)
                per_video.setdefault(v, []).append(best)
            ms = (time.perf_counter() - t0) * 1000 / len(pos)
            clutter = {c: 0 for c in CONFS}
            for v, t, img in neg:
                _, conf = det(img, sz)
                for c in CONFS:
                    clutter[c] += int((conf >= c).sum())
            key = f"{name}@{sz}"
            out[key] = {"cpu_ms_per_frame": round(ms, 1),
                        "object_found_%frames": {str(c): round(100 * np.mean(h), 1) for c, h in hits.items()},
                        "videos_where_found_>=30%_of_frames@0.1": int(sum(np.mean(np.array(b) >= 0.1) >= 0.3 for b in per_video.values())),
                        "videos": len(per_video),
                        "clutter_dets_per_frame": {str(c): round(clutter[c] / len(neg), 2) for c in CONFS},
                        "median_best_conf_per_video": {v: round(float(np.median(b)), 2) for v, b in per_video.items()}}
            print(key, json.dumps(out[key]), flush=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
