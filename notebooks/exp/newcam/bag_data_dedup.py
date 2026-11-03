"""Find Roboflow bag images taken from ABODA (our unattended-item test set) so they are left out of training.

Every ABODA video is sampled once a second; each Roboflow image (train / valid / test) is compared with every sample as
a 64x48 grey thumbnail (zero-mean normalised correlation, after resizing both). Images whose best match is above
MATCH are written to data/roboflow_bags/aboda_overlap.json and excluded by bag_finetune.py.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/bag_data_dedup.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "data" / "roboflow_bags"
ABODA = HERE.parents[2] / "data" / "additional" / "aboda"
MATCH = 0.8


def thumb(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def main():
    refs, names = [], []
    for v in sorted(ABODA.glob("*.avi")):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(0, n, int(round(fps))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                refs.append(thumb(img))
                names.append(f"{v.stem}@{f / fps:.0f}s")
        cap.release()
    refs = np.stack(refs)
    hits, best_all = {}, []
    for p in sorted(R.glob("*/*/images/*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        sims = refs @ thumb(img)
        k = int(np.argmax(sims))
        best_all.append(float(sims[k]))
        if sims[k] >= MATCH:
            hits[str(p.relative_to(R))] = {"aboda": names[k], "similarity": round(float(sims[k]), 3)}
    (R / "aboda_overlap.json").write_text(json.dumps(hits, indent=1))
    hist = np.histogram(best_all, bins=[0, .5, .7, .8, .85, .9, .95, 1.01])[0]
    print(f"{len(refs)} ABODA samples; {len(best_all)} Roboflow images; {len(hits)} matched ABODA (>= {MATCH})")
    print("best-similarity histogram [0,.5,.7,.8,.85,.9,.95,1]:", hist.tolist())


if __name__ == "__main__":
    main()
