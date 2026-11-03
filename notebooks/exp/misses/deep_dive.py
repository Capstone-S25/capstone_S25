"""Follow-ups to miss_analysis.py (tune half only).

smoking:   merge flickering per-frame labels into episodes (labelled frames of one sequence <= EPISODE_GAP_S apart) and
           ask whether each episode was ever sent with a good crop. An episode is "dwell-testable" when some detected
           person covering the label stays in the sequence for >= loiter_s (the dwell rule cannot fire otherwise).
littering: for clips where a candidate went to the wrong person, look at the actor (person box best matching the GT
           box): detected?  touching the frame border?  its best pose_action score vs the best bystander score.
"""
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "gt_review"), str(NB / "exp" / "actions")]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview, iou, touches_border  # noqa: E402
from miss_analysis import methods  # noqa: E402

EPISODE_GAP_S = 30.0


def episodes(g, r):
    pos = [(o.t, o.index) for o in r.observations if o.index in g.gt]
    out, cur = [], None
    for t, i in pos:
        if cur and t - cur[-1][0] <= EPISODE_GAP_S:
            cur.append((t, i))
        else:
            cur = [(t, i)]
            out.append(cur)
    return out


def presence(r):
    """raw track id -> (first t, last t)."""
    span = {}
    for o in r.observations:
        for p in o.persons:
            a, _ = span.get(p["raw_id"], (o.t, o.t))
            span[p["raw_id"]] = (a, o.t)
    return span


def smoking(T, method, loiter_s):
    rows = []
    for g, r in T["smoking"]:
        ev = method(r, "smoking")
        span = presence(r)
        by = {o.index: o for o in r.observations}
        for ep in episodes(g, r):
            caught = any(abs(t - e["timestamp_s"]) <= 1.0 and any(box_hit(e["roi_box"], b) for b in g.gt[i])
                         for e in ev for t, i in ep)
            ids = {p["raw_id"] for _, i in ep for p in by[i].persons if any(box_hit(p["box"], b) for b in g.gt[i])}
            stay = max((span[k][1] - span[k][0] for k in ids), default=0.0)
            rows.append({"caught": caught, "testable": stay >= loiter_s, "stay": stay, "seq_dur": r.duration_s,
                         "frames": len(ep), "by": {e["event_type"] for e in ev
                                                    if any(abs(t - e["timestamp_s"]) <= 1.0 and any(box_hit(e["roi_box"], b) for b in g.gt[i]) for t, i in ep)}})
    n = len(rows)
    print(f"\nsmoking tune: {n} episodes (labels <= {EPISODE_GAP_S:.0f}s apart merged; was {sum(len(g.gt) and 1 for g, _ in T['smoking'])} seqs with labels)")
    print(f"  caught with a good crop: {sum(r['caught'] for r in rows)} / {n} = {100 * np.mean([r['caught'] for r in rows]):.1f}%")
    te = [r for r in rows if r["testable"]]
    nt = [r for r in rows if not r["testable"]]
    print(f"  smoker visible >= {loiter_s:.0f}s (dwell can fire): {len(te)} episodes, caught {100 * np.mean([r['caught'] for r in te]) if te else float('nan'):.1f}%")
    print(f"  smoker visible <  {loiter_s:.0f}s: {len(nt)} episodes, caught {100 * np.mean([r['caught'] for r in nt]) if nt else float('nan'):.1f}%"
          f"  (median visible {np.median([r['stay'] for r in nt]):.1f}s, median sequence {np.median([r['seq_dur'] for r in nt]):.1f}s)")
    print("  caught by:", Counter(x for r in rows for x in r["by"]).most_common())
    return rows


def littering(T, method):
    import pose_seq as P
    import pose_seq_final as F
    st = F.pickle.loads(F.FROZEN.read_bytes())
    oof, meta = st["oof"]["littering"], st["meta"]
    stats = defaultdict(list)
    for g, r in T["littering"]:
        ev = method(r, "littering")
        by = {o.index: o for o in r.observations}
        hit = any(any(box_hit(e["roi_box"], b) for i in g.gt for b in g.gt[i] if abs(by[i].t - e["timestamp_s"]) <= 1.0) for e in ev)
        if hit or not ev:
            continue
        actor = Counter()
        border = 0
        for i, bs in g.gt.items():
            for p in by[i].persons:
                if max(iou(p["box"], b) for b in bs) >= 0.4:
                    actor[p["raw_id"]] += 1
                    border += touches_border(p["box"], by[i].frame_hw)
        aid = actor.most_common(1)[0][0] if actor else None
        sc = defaultdict(float)
        for (t, rid, box), s in zip(meta[r.seq_id], oof[r.seq_id]):
            sc[rid] = max(sc[rid], s)
        a_s = sc.get(aid, 0.0) if aid is not None else None
        b_s = max((v for k, v in sc.items() if k != aid), default=0.0)
        n_act = sum(actor.values())
        sent_to = Counter("actor" if any(iou(e["roi_box"], p["box"]) > 0.5 and p["raw_id"] == aid for o in r.observations
                                         if abs(o.t - e["timestamp_s"]) < 0.06 for p in o.persons) else "other" for e in ev)
        stats["tag"].append(g.tag)
        stats["actor_tracked"].append(aid is not None)
        stats["actor_border_%"].append(100 * border / max(1, n_act))
        stats["actor_score"].append(a_s if a_s is not None else np.nan)
        stats["bystander_score"].append(b_s)
        stats["sent_to"].append(dict(sent_to))
        stats["types"].append(Counter(e["event_type"] for e in ev))
        stats["n_people"].append(np.mean([len(o.persons) for o in r.observations]))
        stats["actor_h_frac"].append(np.median([b[3] - b[1] for bs in g.gt.values() for b in bs]) / r.observations[0].frame_hw[0])
    n = len(stats["tag"])
    thr = st["thr"]
    a = np.array(stats["actor_score"], float)
    b = np.array(stats["bystander_score"], float)
    print(f"\nlittering tune: {n} clips where candidates went only to the wrong place")
    print("  by action:", Counter(stats["tag"]).most_common())
    print(f"  actor tracked: {sum(stats['actor_tracked'])}/{n}; actor box touching frame border (median % of frames): {np.median(stats['actor_border_%']):.0f}%;"
          f" actor height median {np.median(stats['actor_h_frac']):.2f} of frame; people/frame {np.mean(stats['n_people']):.1f}")
    print(f"  actor best pose score: median {np.nanmedian(a):.2f}; above threshold {thr:.2f}: {(a >= thr).sum()}; bystander best above: {(b >= thr).sum()};"
          f" actor > bystander: {(a > b).sum()}/{n}")
    print("  candidate types:", sum(stats["types"], Counter()).most_common())
    print("  candidates went to:", sum((Counter(s) for s in stats["sent_to"]), Counter()))
    return stats


def main():
    T = model_splits("s", heldout=False)
    M = methods(T)
    cfg = GateConfig()
    smoking(T, M["smoking"], cfg.loiter_s)
    littering(T, M["littering"])


if __name__ == "__main__":
    main()
