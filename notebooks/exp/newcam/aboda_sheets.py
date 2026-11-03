"""Contact sheets for labelling ABODA (no official annotation files).

For each video: one frame every STEP_S seconds, 4x3 per sheet, each tile at full resolution with its timestamp and a
100 px coordinate grid (so labellers can give boxes in original pixel coordinates).
Writes exp/newcam/aboda_sheets/<video>_<k>.jpg and index.json {video: {duration_s, fps, size, sheets: [{path, times}]}}.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ABODA = HERE.parents[2] / "data" / "additional" / "aboda"
OUT = HERE / "aboda_sheets"
STEP_S, COLS, ROWS = 2.0, 4, 3


def grid(img, t):
    img = img.copy()
    h, w = img.shape[:2]
    for x in range(100, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(100, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.rectangle(img, (w - 92, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:5.1f}s", (w - 88, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    index = {}
    for v in sorted(ABODA.glob("*.avi"), key=lambda p: int(re.sub(r"\D", "", p.stem))):
        cap = cv2.VideoCapture(str(v))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {int(round(k * STEP_S * fps)) for k in range(int(n / fps / STEP_S) + 1)}
        tiles, i = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in want:
                tiles.append((i / fps, grid(img, i / fps)))
            i += 1
        cap.release()
        h, w = tiles[0][1].shape[:2]
        sheets = []
        per = COLS * ROWS
        for k in range(0, len(tiles), per):
            chunk = tiles[k:k + per]
            canvas = np.zeros((ROWS * h, COLS * w, 3), np.uint8)
            for j, (_, im) in enumerate(chunk):
                canvas[(j // COLS) * h:(j // COLS + 1) * h, (j % COLS) * w:(j % COLS + 1) * w] = cv2.resize(im, (w, h))
            path = OUT / f"{v.stem}_{k // per:02d}.jpg"
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            sheets.append({"path": str(path), "times": [round(t, 1) for t, _ in chunk]})
        index[v.stem] = {"duration_s": round(n / fps, 1), "fps": fps, "size": [w, h], "sheets": sheets}
        print(v.stem, len(tiles), "frames,", len(sheets), "sheets")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
