"""Why does the frozen gate miss MEVA put-downs?  Size of the actor, detection, keypoints, pose score vs threshold."""
import pickle, sys
from collections import Counter
from functools import partial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]
from loaders import meva_relabel
from stage1_bench import box_hit, gt_events
from stage1_gate import iou
from stage1_pose_action import PoseActionModel, TrackWindows

res = pickle.loads((NB / "bench_cache/models/s/meva.pkl").read_bytes())
model = PoseActionModel()
for act in (("person_puts_down_object",), ("person_picks_up_object",)):
    rows = []
    for g, r in meva_relabel(res, act):
        if not g.gt:
            continue
        by = {o.index: o for o in r.observations}
        # pose scores for every window, keyed by (t, raw_id)
        tw, scores = TrackWindows(), {}
        for o in r.observations:
            ws = tw.step(o)
            if ws:
                for w, p in zip(ws, model.predict(np.stack([w["sum"] for w in ws]))):
                    scores[(round(w["t"], 2), w["raw_id"])] = p
        fh = r.observations[0].frame_hw[0]
        for ge in gt_events([o.t for o in r.observations], [o.index for o in r.observations], g.gt, r.fps, 1.0):
            hs = [b[3] - b[1] for _, i in ge["frames"] for b in g.gt[i]]
            det = [p for _, i in ge["frames"] for p in by[i].persons if any(iou(p["box"], b) >= 0.4 for b in g.gt[i])]
            ids = {p["raw_id"] for p in det}
            kp = [p.get("kpts") is not None for p in det]
            best = max((s for (t, rid), s in scores.items() if rid in ids and ge["t0"] - 1.5 <= t <= ge["t1"] + 1.5), default=np.nan)
            rows.append({"h_px": np.median(hs), "h_frac": np.median(hs) / fh, "detected": bool(det),
                         "kpts_%": 100 * np.mean(kp) if kp else 0, "best_score": best, "people": np.mean([len(by[i].persons) for _, i in ge["frames"]])})
    h = np.array([x["h_px"] for x in rows]); s = np.array([x["best_score"] for x in rows], float)
    print(f"\n{act[0]}: {len(rows)} events")
    print(f"  actor height px: median {np.median(h):.0f} (pctl 25/75: {np.percentile(h, 25):.0f}/{np.percentile(h, 75):.0f}), {np.median([x['h_frac'] for x in rows]):.3f} of frame")
    print(f"  actor detected (IoU>=0.4) in some frame: {sum(x['detected'] for x in rows)}/{len(rows)}; keypoints on detected actor: {np.mean([x['kpts_%'] for x in rows if x['detected']]):.0f}% of frames")
    print(f"  people per frame: {np.mean([x['people'] for x in rows]):.1f}")
    print(f"  best pose score on actor near event: median {np.nanmedian(s):.2f}; >= thr {model.threshold:.2f}: {(s >= model.threshold).sum()}; >=0.5: {(s >= 0.5).sum()}; no window: {np.isnan(s).sum()}")
    for lo, hi in ((0, 100), (100, 200), (200, 2000)):
        sub = [x for x in rows if lo <= x["h_px"] < hi]
        if sub:
            print(f"   actor {lo}-{hi}px: {len(sub)} events, detected {sum(x['detected'] for x in sub)}, median score {np.nanmedian([x['best_score'] for x in sub]):.2f}")
