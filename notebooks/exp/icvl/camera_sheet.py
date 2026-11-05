"""One labelled frame per ICVL camera (deck slide 3): person markers from the head/feet points, coloured by label.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/camera_sheet.py OUT.jpg 01,02,03,04,05,07
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
NB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB / "exp" / "icvl"))
from inventory import ICVL, video_for, read_labels, GESTURE, EVENT, MOVE, POSTURE
OUT = Path(sys.argv[1])
inv = json.load(open(NB / "exp/icvl/inventory.json"))
want = sys.argv[2].split(",")
COL = {"littering": (40, 48, 220), "smoking": (40, 150, 255), "phoning": (200, 160, 60), "texting": (200, 160, 60),
       "still": (210, 120, 40), None: (235, 235, 235)}
tiles = []
for cam in want:
    vids = [v for v in inv["per_video"] if v["camera"] == cam]
    best = None
    for v in sorted(vids, key=lambda v: -(3 * v.get("smoking", 0) + 2 * v.get("littering", 0) + v["people"])):
        txt = (ICVL / v["video"]).with_suffix(".txt")
        if not txt.exists():
            cands = list((ICVL / cam).glob(Path(v["video"]).stem.split(".")[0] + "*.txt"))
            if not cands: continue
            txt = cands[0]
        rows = read_labels(txt)
        byf = defaultdict(list)
        for r in rows: byf[r[0]].append(r)
        def tag(r):
            if EVENT.get(r[10]) == "littering": return "littering"
            g = GESTURE.get(r[9])
            if g in ("smoking", "phoning", "texting"): return g
            return None
        score = lambda f: sum(3 if tag(r) in ("littering", "smoking") else 1 for r in byf[f])
        f = max(byf, key=score)
        best = (v, f, byf[f], tag); break
    v, f, people, tag = best
    cap = cv2.VideoCapture(str(ICVL / v["video"])); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read(); cap.release()
    if not ok: continue
    H, W = img.shape[:2]
    for r in people:
        hx, hy, fx, fy = r[1:5]
        h = max(10, fy - hy); w = 0.42 * h; cx = (hx + fx) / 2
        t = tag(r)
        lab = t or MOVE.get(r[8], "")
        c = COL[t] if t in COL else COL[None]
        th = max(2, int(H / 300))
        cv2.rectangle(img, (int(cx - w / 2), int(hy - 0.12 * h)), (int(cx + w / 2), int(fy)), c, th)
        fs = H / 900
        (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x0, y0 = int(cx - w / 2), int(hy - 0.12 * h) - 6
        cv2.rectangle(img, (x0, y0 - tht - 8), (x0 + tw + 8, y0 + 2), c, -1)
        cv2.putText(img, lab, (x0 + 4, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)
    tile = cv2.resize(img, (640, 360))
    cv2.rectangle(tile, (0, 0), (150, 30), (20, 20, 20), -1)
    cv2.putText(tile, f"Camera {cam}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
    print(cam, v["video"], f, len(people), [tag(r) for r in people])
while len(tiles) % 3: tiles.append(np.full_like(tiles[0], 30))
gap = 6
rows_ = []
for i in range(0, len(tiles), 3):
    row = tiles[i]
    for t in tiles[i + 1:i + 3]:
        row = np.hstack([row, np.full((360, gap, 3), 255, np.uint8), t])
    rows_.append(row)
sheet = rows_[0]
for r in rows_[1:]:
    sheet = np.vstack([sheet, np.full((gap, sheet.shape[1], 3), 255, np.uint8), r])
cv2.imwrite(str(OUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
print(sheet.shape)
