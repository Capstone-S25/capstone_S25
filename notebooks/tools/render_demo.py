"""Render an annotated demo video of Stage 1 from cached detections.

Layout: the video on the left with people (stable ID colours + pose skeleton), candidates highlighted as they fire;
a "sent to the VLM" queue of crops on the right; a timeline at the bottom showing the true anomalies (red) against
the parts of the video Stage 1 sends to the VLM (orange).
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from stage1_gate import KP_CONF, GateConfig, RuleEngine

W, H = 1600, 900
VW, VH = 1180, 664
PANEL_X = VW + 20
TL_Y = VH + 70
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

LABELS = [("reach_down", "reached to the ground"), ("littering_act", "arm out / hand low"), ("fast", "moving fast"),
          ("dens", "unusual for this spot"), ("sustained", "stayed in one area"), ("unattended", "bag left alone"),
          ("neighbour_", "person next to the action"), ("context_", "person next to the action")]


def label_of(event_type: str) -> str:
    for key, text in LABELS:
        if key in event_type:
            return text
    return event_type.replace("_", " ")


def id_colour(cid) -> tuple[int, int, int]:
    h = (hash(str(cid)) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def text(img, s, org, scale=0.6, colour=(235, 235, 235), thick=1, box=False):
    if box:
        (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def merged(windows):
    out = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def render_sequence(writer, g, res, events, title: str, subtitle: str, cfg: GateConfig, hold_s: float = 1.2,
                    repeat_frames: int = 1, labels=None):
    """labels: list of (t0, t1, text) describing the labelled anomalies, shown while they happen."""
    obs = res.observations
    t0, t1 = obs[0].t, obs[-1].t + 1.0 / res.fps
    span = max(1e-6, t1 - t0)
    gt_times = [o.t for o in obs if o.index in g.gt]
    windows = merged([e["clip_window_s"] for e in events])
    forwarded = sum(any(a <= o.t <= b for a, b in windows) for o in obs) / len(obs)
    engine = RuleEngine(cfg, res.fps)
    queue = []
    by_time = sorted(events, key=lambda e: e["timestamp_s"])
    k = 0
    frames = {fr.index: fr.image for fr in g.seq}
    for o in obs:
        engine.step(o)
        img = frames[o.index]
        sx, sy = VW / img.shape[1], VH / img.shape[0]
        view = cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)
        lw = 2
        for cid, p in zip(engine.last_cids, o.persons):
            col = id_colour(cid)
            x1, y1, x2, y2 = int(p["box"][0] * sx), int(p["box"][1] * sy), int(p["box"][2] * sx), int(p["box"][3] * sy)
            cv2.rectangle(view, (x1, y1), (x2, y2), col, lw)
            text(view, f"ID {cid}", (x1, max(16, y1 - 6)), 0.5, col, 1, box=True)
            if p.get("kpts") is not None:
                kp = p["kpts"]
                for a, b in SKELETON:
                    if kp[a, 2] >= KP_CONF and kp[b, 2] >= KP_CONF:
                        cv2.line(view, (int(kp[a, 0] * sx), int(kp[a, 1] * sy)), (int(kp[b, 0] * sx), int(kp[b, 1] * sy)),
                                 col, 2, cv2.LINE_AA)
        for b in g.gt.get(o.index, []):
            cv2.rectangle(view, (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy)), (60, 60, 230), 1)
        while k < len(by_time) and by_time[k]["timestamp_s"] <= o.t + 1e-6:
            e = by_time[k]
            x1, y1, x2, y2 = e["roi_box"]
            px, py = (x2 - x1) * 0.4, (y2 - y1) * 0.4
            crop = img[max(0, int(y1 - py)):int(y2 + py), max(0, int(x1 - px)):int(x2 + px)]
            if crop.size:
                queue.append((e, crop.copy()))
            k += 1
        for e, _ in queue:
            if 0 <= o.t - e["timestamp_s"] <= hold_s:
                x1, y1, x2, y2 = e["roi_box"]
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(view, p1, p2, (40, 150, 255), 4)
                text(view, label_of(e["event_type"]), (p1[0], min(VH - 8, p2[1] + 24)), 0.7, (90, 170, 255), 2, box=True)

        canvas = np.full((H, W, 3), 28, np.uint8)
        canvas[40:40 + VH, 10:10 + VW] = view
        text(canvas, title, (12, 28), 0.8, (255, 255, 255), 2)
        now = [s for a, b, s in (labels or []) if a - 0.3 <= o.t - t0 <= b + 0.3]
        if now:
            text(canvas, "True anomaly: " + now[0], (16, 40 + VH - 14), 0.75, (110, 110, 255), 2, box=True)
        text(canvas, subtitle, (420, 28), 0.6, (190, 190, 190), 1)

        text(canvas, "Sent to the VLM", (PANEL_X, 70), 0.75, (90, 170, 255), 2)
        text(canvas, f"{len(queue)} candidates so far", (PANEL_X, 96), 0.5, (190, 190, 190), 1)
        y = 112
        for e, crop in reversed(queue[-4:]):
            th = 140
            tw = max(1, int(crop.shape[1] * th / crop.shape[0]))
            if tw > 180:
                tw, th = 180, max(1, int(crop.shape[0] * 180 / crop.shape[1]))
            canvas[y:y + th, PANEL_X:PANEL_X + tw] = cv2.resize(crop, (tw, th))
            text(canvas, f"{e['timestamp_s']:.1f}s", (PANEL_X + 190, y + 20), 0.55, (230, 230, 230), 1)
            words = label_of(e["event_type"]).split()
            for i in range(0, len(words), 2):
                text(canvas, " ".join(words[i:i + 2]), (PANEL_X + 190, y + 44 + 20 * (i // 2)), 0.5, (90, 170, 255), 1)
            y += th + 12

        x0, x1b = 10, 10 + VW
        tx = lambda t: int(x0 + (t - t0) / span * (x1b - x0))
        text(canvas, "true anomaly", (x0, TL_Y - 8), 0.5, (90, 90, 255), 1)
        cv2.rectangle(canvas, (x0, TL_Y), (x1b, TL_Y + 22), (55, 55, 55), -1)
        for t in gt_times:
            cv2.rectangle(canvas, (tx(t), TL_Y), (tx(t + 1.0 / res.fps) + 1, TL_Y + 22), (60, 60, 230), -1)
        rows = [("sent to the VLM", windows, (40, 150, 255), TL_Y + 50)]
        for label, wins, col, yy in rows:
            text(canvas, label, (x0, yy - 6), 0.5, col, 1)
            cv2.rectangle(canvas, (x0, yy), (x1b, yy + 22), (55, 55, 55), -1)
            for a, b in wins:
                if b >= t0 and a <= o.t:
                    cv2.rectangle(canvas, (tx(max(a, t0)), yy), (tx(min(b, o.t, t1)), yy + 22), col, -1)
        cv2.line(canvas, (tx(o.t), TL_Y - 4), (tx(o.t), rows[-1][3] + 26), (255, 255, 255), 2)
        text(canvas, f"{o.t - t0:5.1f}s / {span:.1f}s", (x1b - 150, TL_Y - 8), 0.55)
        text(canvas, "thin red box = labelled anomaly, orange = sent to VLM", (PANEL_X, TL_Y + 10), 0.45, (90, 90, 255), 1)
        text(canvas, f"VLM sees {100 * forwarded:.0f}% of this video", (PANEL_X, TL_Y + 40), 0.6, (255, 255, 255), 1)
        for _ in range(repeat_frames):
            writer.write(canvas)


def title_card(writer, lines, fps, seconds=2.0):
    card = np.full((H, W, 3), 28, np.uint8)
    for i, (s, scale, col) in enumerate(lines):
        size = cv2.getTextSize(s, FONT, scale, 2)[0]
        text(card, s, ((W - size[0]) // 2, 380 + 60 * i), scale, col, 2)
    for _ in range(int(fps * seconds)):
        writer.write(card)


def open_writer(path, fps):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not w.isOpened():
        raise RuntimeError("could not open H.264 writer")
    return w
