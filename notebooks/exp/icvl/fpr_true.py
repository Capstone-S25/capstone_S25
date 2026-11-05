"""How much of the per-person false positive rate is real? ICVL even (held-out), current defaults (crop check on).

A flagged person (no littering / smoking / still-20 s+ incident in their video, at least one alert on them) is set
aside when every alert on them is explained by the labels themselves:
  shared alert     the same alert also lands on another person during their labelled incident (one alert box
                   covering two people: a correct alert, not an extra false one)
  still by labels  a stays-put / long-in-view alert where the person's LABELLED feet stayed within the rule's radius
                   (loiter_radius_h body heights) for the 20 s before it: the gate measured stillness correctly; ICVL's
                   "stationary" label (the loitering stand-in) is stricter
What remains is sampled for a by-eye review (exp/icvl/fpr_true.py sheet -> demo/icvl_fp_review_<k>.jpg).
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/fpr_true.py           -> exp/icvl/fpr_true.json
  ../.venv/bin/python exp/icvl/fpr_true.py sheet N   -> contact sheets of N random remaining people (seed 0)
"""
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

import stage1_crop_check  # noqa: E402
from inventory import ICVL, video_for  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

for f in ("crop_scores_icvl.pkl", "crop_scores_pose_action_icvl.pkl"):
    stage1_crop_check.SCORES.update(pickle.loads((NB / "bench_cache" / "models" / "s" / f).read_bytes()))
STILL_RULES = ("sustained_presence", "long_presence")


def still_by_labels(tr, t, span=20.0, radius_h=CFG.loiter_radius_h):
    """Did the person's labelled feet stay within radius_h body heights of their mean over [t - span, t]?"""
    ts = np.array(tr["ts"])
    m = (ts >= t - span) & (ts <= t + 1e-6)
    if m.sum() < 5 or ts[m][0] > t - span + 2.0:
        return False
    b = np.array(tr["boxes"])[m]
    feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
    h = float(np.median(b[:, 3] - b[:, 1]))
    return float(np.max(np.hypot(*(feet - feet.mean(0)).T))) / max(h, 1.0) <= radius_h


def analyse():
    rows = []
    for g, res in load_runs("even"):
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        pos = {ep["pid"] for ev in EVENTS for ep in eps.get(ev, [])}
        for pid, tr in people.items():
            if pid in pos:
                continue
            mine = [a for a in alerts if on_track(a, tr)]
            if not mine:
                continue
            why = []
            for a in mine:
                shared = any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))
                             for ev in EVENTS for ep in eps.get(ev, []))
                still = a["event_type"] in STILL_RULES and still_by_labels(tr, a["timestamp_s"])
                why.append("shared alert" if shared else "still by labels" if still else "unexplained")
            kind = "unexplained" if "unexplained" in why else ("shared alert" if "shared alert" in why else "still by labels")
            rows.append({"seq": res.seq_id, "tag": g.tag, "fps": g.seq.native_fps, "pid": pid, "kind": kind,
                         "alerts": [{"t": a["timestamp_s"], "rule": a["event_type"], "roi_box": a["roi_box"], "why": w}
                                    for a, w in zip(mine, why)]})
    return rows


def main():
    rows = analyse()
    neg = sum(1 for g, res in load_runs("even") for pid in labels_for(g.tag, g.seq.native_fps)[1]
              if pid not in {ep["pid"] for ev in EVENTS for ep in labels_for(g.tag, g.seq.native_fps)[0].get(ev, [])})
    count = {k: sum(r["kind"] == k for r in rows) for k in ("shared alert", "still by labels", "unexplained")}
    out = {"people_without_incident": neg, "flagged": len(rows), **count,
           "fpr_raw_%": round(100 * len(rows) / neg, 1), "fpr_after_label_checks_%": round(100 * count["unexplained"] / neg, 1),
           "rows": rows}
    (HERE / "fpr_true.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


def sheet(n, seed=0, per_sheet=20):
    rows = [r for r in json.loads((HERE / "fpr_true.json").read_text())["rows"] if r["kind"] == "unexplained"]
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    tiles = []
    for k, r in enumerate(pick):
        a = next(x for x in r["alerts"] if x["why"] == "unexplained")
        cap = cv2.VideoCapture(str(video_for(ICVL / r["tag"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a["t"] * r["fps"])))
        ok, im = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = a["roi_box"]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cx, cy, hh = (x1 + x2) / 2, (y1 + y2) / 2, max(300, 1.8 * (y2 - y1))
        X1, Y1 = int(max(0, cx - 0.5 * hh)), int(max(0, cy - hh / 2))
        c = cv2.resize(im[Y1:int(min(im.shape[0], cy + hh / 2)), X1:int(min(im.shape[1], cx + 0.5 * hh))], (300, 300))
        cv2.rectangle(c, (0, 0), (300, 22), (20, 20, 20), -1)
        cv2.putText(c, f'#{k} {a["rule"]}', (5, 16), 0, 0.5, (90, 190, 255), 1)
        tiles.append(c)
        r["review_id"] = k
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        chunk += [np.zeros_like(tiles[0])] * (-len(chunk) % 5)
        cv2.imwrite(str(NB / "demo" / f"icvl_fp_review_{s // per_sheet}.jpg"),
                    np.vstack([np.hstack(chunk[i:i + 5]) for i in range(0, len(chunk), 5)]))
    (HERE / "fpr_true_review.json").write_text(json.dumps(
        [{"review_id": r.get("review_id"), "seq": r["seq"], "pid": r["pid"], "alerts": r["alerts"]} for r in pick], indent=1))
    print(len(tiles), "tiles")


if __name__ == "__main__":
    sheet(int(sys.argv[2])) if len(sys.argv) > 1 and sys.argv[1] == "sheet" else main()
