"""Phone boxes (COCO 67 cell phone) for every cached observation frame, so a "phone at the hand / ear" check on the
pose cue can be replayed on caches. Same frames and model as two_wheelers.py, at a low confidence (0.1) because a phone
at the ear is small; the confidence is stored so a threshold can be chosen later.
Each box is stored as [x1, y1, x2, y2, class, confidence].
Usage (from notebooks/): ../.venv/bin/python exp/alerts/phones.py NAME ...  (names as two_wheelers.py)
  -> bench_cache/models/s/phone_<NAME>.pkl
"""
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from two_wheelers import CACHE, NB, sources  # noqa: E402

CLASSES, CONF = [67], 0.1


def main():
    from ultralytics import YOLO
    model = YOLO(str(NB / "yolo26s.pt"))
    for name in sys.argv[1:]:
        out_path = CACHE / f"phone_{name}.pkl"
        if out_path.exists():
            continue
        t0, out = time.time(), {}
        for g in sources(name):
            boxes = {}
            for fr in g.seq:
                r = model.predict(fr.image, imgsz=640, conf=CONF, classes=CLASSES, verbose=False, device="mps")[0]
                if len(r.boxes):
                    boxes[fr.index] = [[float(v) for v in b] + [int(c), float(s)] for b, c, s in
                                       zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            out[g.seq.seq_id] = boxes
        out_path.write_bytes(pickle.dumps(out))
        print(f"{name}: {len(out)} sequences ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
