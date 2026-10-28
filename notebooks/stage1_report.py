"""Tables and figures for the Stage 1 report notebook (yolo_poc_v3.ipynb).

Everything here works from cached perception results (bench_cache/*.pkl), so the report
renders in seconds without re-running YOLO.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage1_bench import box_hit, evaluate, evaluate_run, summarize
from stage1_gate import GateConfig, RuleEngine, replay_rules

HEADLINE_COLS = {
    "gt_events": "labelled events",
    "event_recall_%": "Event recall %",
    "crop_recall_%": "Crop recall %",
    "candidates": "Candidates",
    "candidate_precision_%": "Candidate precision %",
    "candidates_per_min": "Candidates / min",
    "review_reduction_%": "Review reduction %",
    "baseline_person_reduction_%": "Person-baseline reduction %",
    "positive_frame_share_%": "Anomalous frames %",
}


def headline_table(sets, cfg: GateConfig) -> tuple[pd.DataFrame, dict]:
    """sets: list of (name, results, rules) or (name, results, rules, camera_cfg). Returns the table and per-sequence rows."""
    rows, details = [], {}
    for name, results, rules, *camera in sets:
        summary, seq_rows = evaluate(results, replace(camera[0] if camera else cfg, rules=rules), name=name)
        rows.append(summary)
        details[name] = seq_rows
    table = pd.DataFrame(rows).set_index("run")[list(HEADLINE_COLS)].rename(columns=HEADLINE_COLS)
    return table, details


def recall_by_tag(rows) -> pd.DataFrame:
    out = []
    for tag in sorted({r["tag"] for r in rows}):
        s = summarize([r for r in rows if r["tag"] == tag], tag)
        out.append({"action": tag, "clips": s["sequences"], "event recall %": s["event_recall_%"],
                    "crop recall %": s["crop_recall_%"]})
    return pd.DataFrame(out).set_index("action")


def rule_table(rows) -> pd.DataFrame:
    n, ok, caught = Counter(), Counter(), Counter()
    for r in rows:
        for k, (a, b) in r["by_type"].items():
            n[k] += a
            ok[k] += b
        caught.update(r["caught_by"])
    total = sum(r["gt_events"] for r in rows)
    return pd.DataFrame([{"event type": k, "candidates": n[k],
                          "precision %": round(100 * ok[k] / n[k], 1) if n[k] else np.nan,
                          "GT events caught": caught[k],
                          "share of GT events %": round(100 * caught[k] / total, 1) if total else np.nan}
                         for k in sorted(n)]).set_index("event type")


def false_alarm_rate(results, cfg: GateConfig) -> pd.DataFrame:
    """Candidates per minute on footage with no labelled anomaly (normal footage)."""
    out = Counter()
    minutes = 0.0
    for _, res in results:
        events, _ = replay_rules(res, cfg)
        out.update(e["event_type"] for e in events)
        minutes += res.duration_s / 60.0
    rows = [{"event type": k, "candidates": v, "per minute": round(v / minutes, 2)} for k, v in sorted(out.items())]
    rows.append({"event type": "all", "candidates": sum(out.values()), "per minute": round(sum(out.values()) / minutes, 2)})
    return pd.DataFrame(rows).set_index("event type")


def operating_curve(ax, results, cfg: GateConfig, param: str, values, title: str, label_fmt="{:g}"):
    rec, crop, red = [], [], []
    for v in values:
        s, _ = evaluate(results, replace(cfg, **{param: v}))
        rec.append(s["event_recall_%"])
        crop.append(s["crop_recall_%"])
        red.append(s["review_reduction_%"])
    ax.plot(red, rec, "o-", label="event recall")
    ax.plot(red, crop, "s--", label="crop recall")
    for x, y, v in zip(red, rec, values):
        ax.annotate(label_fmt.format(v), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("review reduction % (higher = less video to the VLM)")
    ax.set_ylabel("recall %")
    ax.set_title(f"{title}\n(sweep of {param})", fontsize=10)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def load_frame(gtseq, index):
    for fr in gtseq.seq:
        if fr.index == index:
            return fr.image
    raise KeyError(index)


def draw_case(gtseq, res, cfg: GateConfig, t: float, title: str):
    """Frame nearest to t with ground truth (green) and the candidates fired within ±clip_pad_s (red)."""
    obs = min(res.observations, key=lambda o: abs(o.t - t))
    img = load_frame(gtseq, obs.index).copy()
    thick = max(2, img.shape[1] // 400)
    for b in gtseq.gt.get(obs.index, []):
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), thick)
    events, _ = replay_rules(res, cfg)
    for e in events:
        if abs(e["timestamp_s"] - t) <= cfg.clip_pad_s:
            x1, y1, x2, y2 = e["roi_box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thick)
            cv2.putText(img, e["event_type"], (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        img.shape[1] / 1400, (0, 0, 255), thick)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), title


def show_grid(cases, cols: int = 3, width: float = 16):
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width, width / cols * 0.62 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, title) in zip(np.atleast_1d(axes).ravel(), cases):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.show()


def id_fragmentation(results, cfg: GateConfig) -> dict:
    n_ids = n_stitched = 0
    for _, res in results:
        eng = RuleEngine(cfg, res.fps)
        ids = set()
        for o in res.observations:
            eng.step(o)
            ids.update(eng.last_cids)
        n_ids += len(ids)
        n_stitched += eng.stitcher.n_stitched
    return {"distinct person IDs": n_ids, "re-links": n_stitched}


def pick_example(results, cfg: GateConfig, kind: str):
    """First (gtseq, result, time) where a candidate's crop hits the ground truth ("hit") or an event is missed ("miss")."""
    for g, r in results:
        events, _ = replay_rules(r, cfg)
        if kind == "miss":
            missed = evaluate_run(g, r, events)["missed"]
            if missed:
                return g, r, missed[0]["t0"]
            continue
        positives = [(o.t, o.index) for o in r.observations if o.index in g.gt]
        for e in events:
            if any(abs(t - e["timestamp_s"]) <= 1.0 and any(box_hit(e["roi_box"], b) for b in g.gt[i]) for t, i in positives):
                return g, r, e["timestamp_s"]
    return None


def _inside(box, hw, margin=0.03):
    h, w = hw
    return box[0] > margin * w and box[1] > margin * h and box[2] < (1 - margin) * w and box[3] < (1 - margin) * h


def find_relinks(results, cfg: GateConfig, limit: int = 3, min_gap_s: float = 1.0):
    """Re-links where the person was fully inside the frame both when lost and when found again:
    (gtseq, result, lost_obs, lost_box, found_obs, found_box, canonical_id, new_raw_id)."""
    found = []
    for g, res in results:
        eng = RuleEngine(cfg, res.fps)
        last = {}
        for o in res.observations:
            before = eng.stitcher.n_stitched
            eng.step(o)
            if eng.stitcher.n_stitched > before:
                for cid, p in zip(eng.last_cids, o.persons):
                    if cid == p["raw_id"] or cid not in last:
                        continue
                    lost_obs, lost_box = last[cid]
                    if o.t - lost_obs.t >= min_gap_s and _inside(lost_box, o.frame_hw) and _inside(p["box"], o.frame_hw):
                        found.append((g, res, lost_obs, lost_box, o, p["box"], cid, p["raw_id"]))
            for cid, p in zip(eng.last_cids, o.persons):
                last[cid] = (o, p["box"])
            if len(found) >= limit:
                return found
    return found


def relink_panel(example):
    g, res, lost, lost_box, back, back_box, cid, raw = example
    panels = []
    for o, box, label in ((lost, lost_box, f"last seen as person {cid}"),
                          (back, back_box, f"returns as new tracker id {raw} -> re-linked to person {cid}")):
        img = load_frame(g, o.index).copy()
        for p in o.persons:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"person {cid}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"{g.seq.seq_id}, t={o.t:.1f}s: {label}"))
    return panels
