"""Score Yuchuan's VLM stage (CPU bundle capstone-mobileclip2-s2-cpu, MobileCLIP2-S2 strategy classifier) on the Stage 1
candidates it would receive on ICVL (real CCTV; the VLM never saw ICVL in training).

Candidates are gate alerts (current defaults), at most one per video per group, picked with a fixed seed:
  littering   an alert on a labelled litterer within 3 s of the drop (reach_down, pose_action, object_appeared)
  smoking     an alert on a labelled smoker during the episode
  lingering   a stays-put / long-in-view alert on a person still for 20 s+, at least 21 s into the video
              (the VLM's loitering answer needs 20 s of history)
  normal      an alert on a labelled person in a video with no labelled event at all (false alerts the VLM should
              reject); reach_down / pose_action ones, plus stays-put ones at least 21 s in
Every candidate is asked all five tasks; each task is scored on its positive group against the normal group.

Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/vlm_eval.py prepare             -> demo/vlm_eval/candidates.json (+ views)
  ../.venv/bin/python exp/icvl/vlm_eval.py classify <bundle>   -> demo/vlm_eval/predictions.jsonl (resumable)
  ../.venv/bin/python exp/icvl/vlm_eval.py score               -> exp/icvl/vlm_eval.json
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(HERE)]

from make_vlm_demo import ALL_TASKS, CFG, RUNS, classify, export, load_predictions  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

OUT = NB / "demo" / "vlm_eval"
SIZES = {"littering": 40, "smoking": 25, "lingering": 20, "normal_reach_pose": 30, "normal_stay": 15}
PERSON_RULES = ("reach_down", "pose_action", "object_appeared")
STAY_RULES = ("sustained_presence", "long_presence")
MIN_HISTORY_S = 21.0
TASK_GROUP = {"littering": "littering", "smoking": "smoking", "loitering": "lingering"}


def pools():
    pool = defaultdict(list)
    for sid, (g, r) in sorted(RUNS.items()):
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        alerts = replay_rules(r, CFG)[0]
        quiet = not any(episodes.get(ev) for ev in EVENTS)
        for a in alerts:
            on = lambda ev: any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))  # noqa: E731
                                for ep in episodes.get(ev, []))
            et, late = a["event_type"], a["timestamp_s"] >= MIN_HISTORY_S
            if et in PERSON_RULES and on("littering"):
                pool["littering"].append((sid, a))
            if on("smoking"):
                pool["smoking"].append((sid, a))
            if et in STAY_RULES and late and on("lingering 20s+"):
                pool["lingering"].append((sid, a))
            if quiet and any(on_track(a, tr) for tr in people.values()):
                if et in ("reach_down", "pose_action"):
                    pool["normal_reach_pose"].append((sid, a))
                elif et in STAY_RULES and late:
                    pool["normal_stay"].append((sid, a))
    return pool


def pick(pool, n, rng):
    """At most one alert per video, spread over cameras."""
    by_video = defaultdict(list)
    for sid, a in pool:
        by_video[sid].append(a)
    videos = sorted(by_video)
    rng.shuffle(videos)
    by_cam = defaultdict(list)
    for v in videos:
        by_cam[v.split("/")[1]].append(v)
    out = []
    while len(out) < n and any(by_cam.values()):
        for cam in sorted(by_cam):
            if by_cam[cam] and len(out) < n:
                v = by_cam[cam].pop()
                out.append((v, rng.choice(by_video[v])))
    return out


def prepare():
    rng = random.Random(0)
    pool = pools()
    items = []
    for group, n in SIZES.items():
        chosen = pick(pool[group], n, rng)
        print(group, len(pool[group]), "alerts in pool,", len(chosen), "picked", flush=True)
        items += [(sid, a, {"group": group}) for sid, a in chosen]
    # one alert can be in two groups (a smoker who also stays put): keep the first
    seen, uniq = set(), []
    for sid, a, extra in items:
        if (sid, a["event_id"]) not in seen:
            seen.add((sid, a["event_id"]))
            uniq.append((sid, a, extra))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(export(uniq, OUT), indent=1, default=float))


def score():
    cands = json.loads((OUT / "candidates.json").read_text())
    preds = load_predictions(OUT / "predictions.jsonl")
    out = {"candidates": len(cands), "classified": len(preds), "tasks": {}}
    normal = [e for e in cands if e["group"].startswith("normal") and e["event_id"] in preds]
    for task in ALL_TASKS:
        group = TASK_GROUP.get(task)
        rows = {}
        for name, members in ((group, [e for e in cands if e["group"] == group and e["event_id"] in preds]),
                              ("normal", normal)):
            if not name:
                continue
            d = [preds[e["event_id"]][task]["decision"] for e in members]
            rows[name] = {"n": len(d), "anomaly": d.count("anomaly"), "normal": d.count("normal"),
                          "unknown": d.count("unknown")}
        if group:
            pos = [preds[e["event_id"]][task].get("score") for e in cands
                   if e["group"] == group and e["event_id"] in preds and preds[e["event_id"]][task]["decision"] != "unknown"]
            neg = [preds[e["event_id"]][task].get("score") for e in normal if preds[e["event_id"]][task]["decision"] != "unknown"]
            if pos and neg:
                # AUC: chance that a real event scores above a false alert (0.5 = no better than a coin)
                rows["auc"] = round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)
        out["tasks"][task] = rows
    by_rule = defaultdict(lambda: defaultdict(int))
    for e in normal:
        for task in ALL_TASKS:
            by_rule[e["event_type"]][f"{task} anomaly"] += preds[e["event_id"]][task]["decision"] == "anomaly"
        by_rule[e["event_type"]]["n"] += 1
    out["normal_alerts_flagged_by_rule"] = {k: dict(v) for k, v in by_rule.items()}
    out["seconds_per_candidate"] = None
    (HERE / "vlm_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prepare": prepare, "classify": lambda: classify(sys.argv[2], "mobileclip2-s2", OUT / "candidates.json",
                                                      OUT / "predictions.jsonl"),
     "score": score}[cmd]()
