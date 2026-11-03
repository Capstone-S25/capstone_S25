"""How small can a person be before Stage 1 stops seeing them?

Reference people: YOLO26l at 1280 px input, confidence >= REF_CONF, on frames sampled every SAMPLE_S seconds from the
cached runs (no dataset labels every person). For each reference person: is there a person the gate tracked in that
frame (cached YOLO26s run at 640 px, confirmation rules included) overlapping it (IoU >= 0.5)? And if so, does it have
usable wrist and knee points (confidence >= 0.5, what reach_down needs)? Binned by the person's height in the 640 px
image the detector actually sees (height * 640 / longest frame side).
Datasets: ABODA, Avenue (test + train), CAVIAR, MEVA, IWDD test, littering held-out clips, plus a per-scene breakdown.
Usage (from notebooks/): ../.venv/bin/python exp/person_size.py   -> exp/person_size.json
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NB = Path(__file__).resolve().parents[1]
D = NB.parent / "data"
sys.path[:0] = [str(NB), str(NB / "exp" / "newcam")]

from stage1_gate import L_KNEE, L_WRIST, R_KNEE, R_WRIST, iou  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"
REF_CONF = 0.5
SAMPLE_S = 2.0
MAX_FRAMES_PER_VIDEO = 30
BINS = [0, 20, 30, 40, 50, 60, 80, 100, 150, 1e9]


def sources():
    from loaders import aboda_sequences, caviar_sequences, iwdd_sequences, meva_sequences
    from stage1_bench import avenue_sequences, littering_sequences
    yield "ABODA (indoor / station)", "aboda", aboda_sequences()
    yield "Avenue (walkway)", "avenue", avenue_sequences(
        D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" / "testing_label_mask", stride=2)
    yield "CAVIAR (lobby, low-res)", "caviar", caviar_sequences()
    yield "MEVA (wide outdoor / crowds)", "meva", meva_sequences()
    yield "IWDD (streets, dump sites)", "iwdd_test", iwdd_sequences("test")
    yield "littering clips", "litter_heldout", littering_sequences(D / "littering", splits=("valid", "test"))


def main():
    from ultralytics import YOLO
    ref = YOLO(str(NB / "yolo26l.pt"))
    rows = []
    for scene, cache_name, seqs in sources():
        results = {r.seq_id: r for _, r in pickle.loads((CACHE / f"{cache_name}.pkl").read_bytes())}
        for g in seqs:
            r = results.get(g.seq.seq_id)
            if r is None:
                continue
            by_index = {o.index: o for o in r.observations}
            next_t, taken = -1e9, 0
            for fr in g.seq:
                if fr.t < next_t or fr.index not in by_index or taken >= MAX_FRAMES_PER_VIDEO:
                    continue
                next_t, taken = fr.t + SAMPLE_S, taken + 1
                o = by_index[fr.index]
                scale = 640.0 / max(fr.image.shape[:2])
                res = ref.predict(fr.image, imgsz=1280, conf=REF_CONF, classes=[0], verbose=False, device="mps")[0]
                for b in res.boxes.xyxy.cpu().numpy().tolist():
                    h640 = (b[3] - b[1]) * scale
                    match = max(o.persons, key=lambda p: iou(p["box"], b), default=None)
                    tracked = match is not None and iou(match["box"], b) >= 0.5
                    pose_ok = False
                    if tracked and match.get("kpts") is not None:
                        k = match["kpts"]
                        pose_ok = (max(k[L_WRIST, 2], k[R_WRIST, 2]) >= 0.5 and max(k[L_KNEE, 2], k[R_KNEE, 2]) >= 0.5)
                    rows.append((scene, h640, tracked, pose_ok))
        print(scene, "done", sum(1 for x in rows if x[0] == scene), "reference people", flush=True)

    def table(subset):
        out = []
        for lo, hi in zip(BINS, BINS[1:]):
            sel = [x for x in subset if lo <= x[1] < hi]
            if not sel:
                continue
            out.append({"height_px_in_640_image": f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+", "people": len(sel),
                        "tracked_%": round(100 * np.mean([x[2] for x in sel]), 1),
                        "tracked_with_wrist_and_knee_%": round(100 * np.mean([x[3] for x in sel]), 1)})
        return out

    out = {"reference": f"YOLO26l @1280, conf >= {REF_CONF}, one frame per {SAMPLE_S:g} s (max {MAX_FRAMES_PER_VIDEO} per video)",
           "all scenes": table(rows)}
    scenes = defaultdict(list)
    for x in rows:
        scenes[x[0]].append(x)
    for s, sub in scenes.items():
        out[s] = {"people": len(sub), "median_height_px": round(float(np.median([x[1] for x in sub])), 1),
                  "tracked_%": round(100 * np.mean([x[2] for x in sub]), 1),
                  "tracked_with_wrist_and_knee_%": round(100 * np.mean([x[3] for x in sub]), 1)}
    print(json.dumps(out, indent=1))
    (NB / "exp" / "person_size.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
