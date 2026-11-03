"""Bicycle / motorcycle boxes for every cached observation frame, so the "rider" option can be replayed on caches.

The live gate gets these from its own detector call (classes include COCO 1 bicycle and 3 motorcycle when
reach_skip_riders is on); cached runs predate that, so YOLO26s is re-run here on the same frames with only those two
classes (conf >= the detector's det_conf).
Each box is stored as [x1, y1, x2, y2, class, confidence].
Usage (from notebooks/): ../.venv/bin/python exp/alerts/two_wheelers.py NAME ...   NAME in
  litter_train litter_heldout avenue iwdd_train iwdd_test aboda   -> bench_cache/models/s/twowheel_<NAME>.pkl
"""
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
D = NB.parent / "data"
sys.path[:0] = [str(NB), str(NB / "exp" / "newcam")]

from stage1_gate import GateConfig  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"
CLASSES = [1, 3]


def sources(name):
    from loaders import aboda_sequences, iwdd_sequences
    from stage1_bench import avenue_sequences, littering_sequences
    if name.startswith("litter"):
        return littering_sequences(D / "littering", splits=("train",) if name == "litter_train" else ("valid", "test"))
    if name == "avenue":
        return avenue_sequences(D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" / "testing_label_mask",
                                stride=2)
    if name.startswith("iwdd"):
        return iwdd_sequences(name.split("_")[1])
    return aboda_sequences()


def main():
    from ultralytics import YOLO
    model = YOLO(str(NB / "yolo26s.pt"))
    conf = GateConfig().det_conf
    for name in sys.argv[1:]:
        out_path = CACHE / f"twowheel_{name}.pkl"
        if out_path.exists():
            continue
        t0, out = time.time(), {}
        for g in sources(name):
            boxes = {}
            for fr in g.seq:
                r = model.predict(fr.image, imgsz=640, conf=conf, classes=CLASSES, verbose=False, device="mps")[0]
                if len(r.boxes):
                    boxes[fr.index] = [[float(v) for v in b] + [int(c), float(s)] for b, c, s in
                                       zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            out[g.seq.seq_id] = boxes
        out_path.write_bytes(pickle.dumps(out))
        print(f"{name}: {len(out)} sequences ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
