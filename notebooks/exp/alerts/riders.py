"""How many alerts are on people riding a motorbike / bicycle? (tune / train alert table)

At each alert frame, YOLO26s is run once more asking only for COCO bicycle (1) and motorcycle (3); a person "rides"
if a two-wheeler box covers >= RIDE_COVER of the lower half of their box. Adds `riding` to alerts_tune.pkl.
Usage (from notebooks/): ../.venv/bin/python exp/alerts/riders.py
"""
import pickle
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam")]

from compare_models import model_splits  # noqa: E402
from loaders import iwdd_sequences  # noqa: E402
from stage1_gate import intersection_over  # noqa: E402

RIDE_COVER = 0.3


def frame_at(path_or_seq, index):
    if isinstance(path_or_seq, str):
        cap = cv2.VideoCapture(path_or_seq)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        return img if ok else None
    seq = path_or_seq
    p = seq.paths[seq.indices.index(index)] if index in seq.indices else None
    return cv2.imread(p) if p else None


def main():
    from ultralytics import YOLO
    df = pickle.loads((HERE / "alerts_tune.pkl").read_bytes())
    T = model_splits("s", heldout=False)
    src = {r.seq_id: g.seq for k in ("littering", "avenue", "normal") for g, r in T[k]}
    src.update({g.seq.seq_id: g.seq.path for g in iwdd_sequences("train")})
    src = {k: (v.path if hasattr(v, "path") else v) for k, v in src.items()}
    model = YOLO(str(NB / "yolo26s.pt"))
    riding = []
    for _, row in df.iterrows():
        img = frame_at(src[row.video], int(row.frame_index))
        if img is None or not isinstance(row.roi, list):
            riding.append(False)
            continue
        r = model.predict(img, imgsz=640, conf=0.25, classes=[1, 3], verbose=False, device="mps")[0]
        x1, y1, x2, y2 = row.roi
        lower = [x1, (y1 + y2) / 2, x2, y2]
        riding.append(any(intersection_over(lower, b.tolist()) >= RIDE_COVER for b in r.boxes.xyxy))
    df["riding"] = riding
    (HERE / "alerts_tune.pkl").write_bytes(pickle.dumps(df))
    print(df.groupby(["source", "riding", "real"]).size().unstack(fill_value=0))
    print(df[df.riding].groupby(["rule", "real"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
