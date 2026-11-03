"""Sheets for labelling WHO dumps in IWDD train videos (the dataset only gives the onset second).

For every positive train video with a tracked person near the onset: four frames (onset - 1 s, + 0.5 s, + 2 s,
+ 4 s), every tracked person boxed with a large raw track id. A reviewer writes the dumper's id(s) per video into
iwdd_actors_train.json ({"vid0001": [3, 7] or [] if the dumper is not tracked / not visible}).
Test videos are never rendered.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_actor_sheets.py   -> exp/newcam/actor_sheets/sheet_NN.jpg
"""
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from iwdd_eval import load  # noqa: E402
from loaders import iwdd_labels, iwdd_sequences  # noqa: E402

OUT = HERE / "actor_sheets"
OFFSETS = (-1.0, 0.5, 2.0, 4.0)
TW, TH = 480, 270
PER_SHEET = 5


def colour(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def tile(cap, o, label):
    cap.set(cv2.CAP_PROP_POS_FRAMES, o.index)
    ok, img = cap.read()
    if not ok:
        img = np.zeros((o.frame_hw[0], o.frame_hw[1], 3), np.uint8)
    for p in o.persons:
        x1, y1, x2, y2 = map(int, p["box"])
        c = colour(p["raw_id"])
        th = max(2, img.shape[0] // 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, th)
        s = str(p["raw_id"])
        scale = img.shape[0] / 360
        (tw, tht), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, max(2, th))
        cv2.rectangle(img, (x1, max(0, y1 - tht - 8)), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(img, s, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, c, max(2, th))
    img = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    labels = iwdd_labels("train")
    paths = {g.seq.seq_id: g.seq.path for g in iwdd_sequences("train")}
    rows, meta = [], {}
    for _, r in load("train"):
        lab = labels[r.seq_id]
        if not lab["positive"]:
            continue
        onset = lab["onset_s"]
        near = [o for o in r.observations if onset - 1 <= o.t <= onset + 5]
        if not any(o.persons for o in near):
            continue
        vid = r.seq_id.split("/")[-1]
        picks = [min(r.observations, key=lambda o, t=onset + d: abs(o.t - t)) for d in OFFSETS]
        cap = cv2.VideoCapture(paths[r.seq_id])
        tiles = [tile(cap, o, f"{vid} {o.t - onset:+.1f}s") for o in picks]
        cap.release()
        rows.append((vid, np.hstack(tiles)))
        meta[vid] = {"onset_s": onset, "kind": lab["kind"],
                     "ids_near_onset": sorted({p["raw_id"] for o in near for p in o.persons})}
    OUT.mkdir(exist_ok=True)
    for k in range(0, len(rows), PER_SHEET):
        chunk = rows[k:k + PER_SHEET]
        cv2.imwrite(str(OUT / f"sheet_{k // PER_SHEET:02d}.jpg"), np.vstack([im for _, im in chunk]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(rows)} videos on {(len(rows) + PER_SHEET - 1) // PER_SHEET} sheets in {OUT}")


if __name__ == "__main__":
    main()
