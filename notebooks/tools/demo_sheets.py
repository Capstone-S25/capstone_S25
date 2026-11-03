"""One image per Stage 1 candidate for slides: the annotated frame, the crop sent to the VLM and the two
past-frame views (4 s for littering, 20 s for the duration tasks), with times and the VLM tasks.

Usage (from notebooks/): ../.venv/bin/python tools/demo_sheets.py vlm_handoff_demo
Reads <dir>/<source>/candidates.jsonl (data-pipeline stage1_offline.py or main.py --stage1), writes
<dir>/<source>/sheets/<event_id>.jpg.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = 1600
PAD = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    out = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    canvas = np.full((h, w, 3), 30, np.uint8)
    y, x = (h - out.shape[0]) // 2, (w - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def text(img, s, xy, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, xy, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def strip(view, label, w, h=150):
    paths, times = view["paths"], view["times_s"]
    n = max(1, len(paths))
    cw = (w - PAD * (n - 1)) // n
    row = np.full((h + 34, w, 3), 20, np.uint8)
    text(row, label, (0, 24), 0.65, (180, 220, 255))
    for k, (p, t) in enumerate(zip(paths, times)):
        img = cv2.imread(p)
        if img is None:
            continue
        tile = fit(img, cw, h)
        text(tile, f"{t - times[-1]:+.1f}s", (6, h - 10), 0.55)
        x = k * (cw + PAD)
        row[34:, x:x + cw] = tile
    return row


def sheet(c):
    ctx, crop = cv2.imread(c["context_frame_path"]), cv2.imread(c["crop_path"])
    top_h = 520
    left_w = int(W * 0.68)
    top = np.full((top_h, W, 3), 20, np.uint8)
    top[:, :left_w] = fit(ctx, left_w, top_h)
    top[:, left_w + PAD:] = fit(crop, W - left_w - PAD, top_h)
    head = np.full((70, W, 3), 20, np.uint8)
    vlm = ", ".join(f"{k}: {v.get('decision')}" for k, v in c.get("vlm", {}).items())
    text(head, f"{c['source_id']}  t={c['timestamp_s']:.1f}s  rule: {c['event_type']}", (0, 30), 0.9)
    text(head, f"VLM tasks: {', '.join(c['tasks'])}   {('VLM: ' + vlm) if vlm else ''}", (0, 60), 0.65, (180, 255, 180))
    parts = [head, top]
    for name, label in (("short", "short view (littering): last 4 s"), ("long", "long view (loitering / unattended): last 20 s")):
        if c["views"][name]["paths"]:
            parts.append(strip(c["views"][name], label, W))
    return np.vstack(parts)


def main(root):
    root = Path(root)
    for log in sorted(root.glob("*/candidates.jsonl")):
        out = log.parent / "sheets"
        out.mkdir(exist_ok=True)
        for line in log.read_text().splitlines():
            c = json.loads(line)
            cv2.imwrite(str(out / f"{c['event_id']}.jpg"), sheet(c), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(log.parent.name, len(list(out.glob("*.jpg"))), "sheets")


if __name__ == "__main__":
    main(sys.argv[1])
