"""Score the cached Stage 1 run on ICVL (real CCTV, never used for tuning) against its per-person labels.

Labelled events (per person, from the annotation layers):
  smoking     gesture == smoking; runs of one person merged across gaps <= 15 s (as smoking_sequences)
  littering   event == littering; gaps <= 1 s merged
  lingering   loitering proxy: movement stationary and posture standing / sitting continuously >= 20 s (also >= 60 s)
Person boxes come from the labelled head and feet points (height = feet - head, padded 12 % above the head,
width 0.45 x height). ICVL does not label every person (43 labelled vs 48 confident YOLO people on 25 sampled frames).
Counted BY PERSON: every alert counts (current defaults, rules + pose cue); an incident is caught when an alert lands
on that person (box_hit, as the bench) within the incident [t0 - w, t1 + w]
(w = 3 s for littering, 1 s otherwise; see WINDOW_S). Alerts are split into
  on an incident of that event / on another labelled event / on a labelled person doing nothing labelled /
  on nobody labelled (an unlabelled person or not a person).
Usage (from notebooks/): ../.venv/bin/python exp/icvl/score.py   -> exp/icvl/score.json
  Settings sweep: score.py --sweep NAME '<json list of GateConfig overrides>' [--half odd|even]
  -> exp/icvl/sweep_<NAME>.json (one compact row per setting; odd videos = ICVL tune half, even = final check)
"""
import json
import pickle
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

NB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(NB), str(Path(__file__).parent)]

from inventory import ICVL, read_labels  # noqa: E402
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s" / "icvl.pkl"
TOL_S = 1.0
GAP_S = {"smoking": 15.0, "littering": 1.0, "lingering 20s+": 1.0, "lingering 60s+": 1.0}
MIN_S = {"lingering 20s+": 20.0, "lingering 60s+": 60.0}
# how far before / after the labelled incident an alert on that person still counts. ICVL labels a drop very tightly
# (~2 s), while the act the gate sees (bend, release, straighten) runs a few seconds either side; 3 s keeps the same act.
WINDOW_S = {"littering": 3.0}
EVENTS = list(GAP_S)


def person_box(r):
    hx, hy, fx, fy = r[1:5]
    h = max(fy - hy, 1)
    top, cx = hy - 0.12 * h, (hx + fx) / 2
    w = 0.45 * (fy - top)
    return [cx - w / 2, top, cx + w / 2, fy]


def flags(r):
    still = r[8] == 0 and r[7] in (0, 1)
    return {"smoking": r[9] == 2, "littering": r[10] == 1, "lingering 20s+": still, "lingering 60s+": still}


def labels_for(tag, fps):
    """-> (episodes per event, all labelled boxes as sorted (t, box) per person)."""
    rows = read_labels(ICVL / tag)
    track = defaultdict(list)
    on = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t, pid = r[0] / fps, r[6]
        track[pid].append((t, person_box(r)))
        for ev, f in flags(r).items():
            if f:
                on[ev][pid].append((t, person_box(r)))
    episodes = defaultdict(list)
    for ev, people in on.items():
        for pid, pts in people.items():
            pts.sort(key=lambda x: x[0])
            cur = [pts[0]]
            for p in pts[1:]:
                if p[0] - cur[-1][0] > GAP_S[ev]:
                    episodes[ev].append((pid, cur))
                    cur = []
                cur.append(p)
            episodes[ev].append((pid, cur))
    for ev in episodes:
        episodes[ev] = [{"pid": pid, "t0": pts[0][0], "t1": pts[-1][0], "ts": [p[0] for p in pts], "boxes": [p[1] for p in pts]}
                        for pid, pts in episodes[ev] if pts[-1][0] - pts[0][0] >= MIN_S.get(ev, 0)]
    people = {pid: {"ts": [p[0] for p in sorted(v, key=lambda x: x[0])], "boxes": [p[1] for p in sorted(v, key=lambda x: x[0])]}
              for pid, v in track.items()}
    return episodes, people


def on_track(alert, tr, t0=-1e9, t1=1e9, window_s=TOL_S):
    """Is the alert on this labelled person (any labelled frame within TOL_S of the alert, inside [t0, t1] +- window_s)?"""
    t = alert["timestamp_s"]
    if not (t0 - window_s <= t <= t1 + window_s):
        return False
    lo, hi = bisect_left(tr["ts"], t - TOL_S), bisect_right(tr["ts"], t + TOL_S)
    return any(box_hit(alert["roi_box"], b) for b in tr["boxes"][lo:hi])


def load_runs(half=None):
    """The cached ICVL run, with bicycle / motorcycle boxes attached when two_wheelers.py has made them."""
    runs = pickle.loads(CACHE.read_bytes())
    wheels = CACHE.parent / "twowheel_icvl.pkl"
    if wheels.exists():
        boxes = pickle.loads(wheels.read_bytes())
        for g, r in runs:
            b = boxes.get(g.seq.seq_id, {})
            for o in r.observations:
                o.two_wheelers = b.get(o.index, [])
    if half is not None:
        runs = runs[1::2] if half == "odd" else runs[0::2]
    return runs


def score(runs, cfg, filt=None):
    """filt(res, alerts) -> alerts: an optional post-filter (e.g. a verifier) applied before scoring."""
    tot = {ev: Counter() for ev in EVENTS}
    kinds, by_rule, per_cam = Counter(), defaultdict(Counter), defaultdict(Counter)
    minutes, quiet_minutes, quiet_alerts = 0.0, 0.0, 0
    for g, res in runs:
        cam = res.seq_id.split("/")[1]
        alerts = replay_rules(res, cfg)[0]
        if filt is not None:
            alerts = filt(res, alerts)
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        minutes += res.duration_s / 60
        per_cam[cam]["minutes"] += res.duration_s / 60
        per_cam[cam]["alerts"] += len(alerts)
        if not any(episodes.get(ev) for ev in EVENTS):
            quiet_minutes += res.duration_s / 60
            quiet_alerts += len(alerts)
        hit_any = [False] * len(alerts)
        for ev in EVENTS:
            eps = episodes.get(ev, [])
            tot[ev]["incidents"] += len(eps)
            hit = [False] * len(alerts)
            for ep in eps:
                caught = False
                for k, a in enumerate(alerts):
                    if on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S)):
                        hit[k] = caught = True
                tot[ev]["caught"] += caught
                per_cam[cam][f"{ev} incidents"] += 1
                per_cam[cam][f"{ev} caught"] += caught
            tot[ev]["alerts_on_incident"] += sum(hit)
            for k, h in enumerate(hit):
                if h:
                    by_rule[alerts[k]["event_type"]][f"on {ev}"] += 1
            hit_any = [x or y for x, y in zip(hit_any, hit)]
        for k, a in enumerate(alerts):
            rule = a["event_type"]
            by_rule[rule]["alerts"] += 1
            if hit_any[k]:
                kind = "on a labelled event"
            elif any(on_track(a, tr) for tr in people.values()):
                kind = "on a labelled person, nothing labelled"
            else:
                kind = "on nobody labelled"
            kinds[kind] += 1
            by_rule[rule][kind] += 1
    n = sum(kinds.values())
    out = {"videos": len(runs), "hours": round(minutes / 60, 2), "alerts": n,
           "alerts_per_min": round(n / minutes, 2) if minutes else None,
           "videos_with_no_labelled_event": {"minutes": round(quiet_minutes, 1), "alerts": quiet_alerts,
                                             "alerts_per_min": round(quiet_alerts / quiet_minutes, 2) if quiet_minutes else None},
           "events": {ev: {**c, "caught_%": round(100 * c["caught"] / c["incidents"], 1) if c["incidents"] else None,
                           "precision_%": round(100 * c["alerts_on_incident"] / n, 1) if n else None}
                      for ev, c in tot.items()},
           "alert_kinds": dict(kinds), "by_rule": {k: dict(v) for k, v in by_rule.items()},
           "per_camera": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sorted(per_cam.items())}}
    return out


def compact(out, kw):
    return {"setting": kw or "current", "alerts": out["alerts"], "quiet_alerts_min": out["videos_with_no_labelled_event"]["alerts_per_min"],
            "on_event": out["alert_kinds"].get("on a labelled event", 0),
            **{ev: f'{c["caught"]}/{c["incidents"]}' for ev, c in out["events"].items()},
            "reach_down_alerts": out["by_rule"].get("reach_down", {}).get("alerts", 0)}


def main():
    base = dict(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
    if "--sweep" in sys.argv:
        name, settings = sys.argv[sys.argv.index("--sweep") + 1], json.loads(sys.argv[sys.argv.index("--sweep") + 2])
        half = sys.argv[sys.argv.index("--half") + 1] if "--half" in sys.argv else None
        runs, rows = load_runs(half), []
        for kw in settings:
            kw = {k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}
            rows.append(compact(score(runs, GateConfig(**{**base, **kw})), kw))
            print(json.dumps(rows[-1], default=str), flush=True)
        (Path(__file__).parent / f"sweep_{name}.json").write_text(json.dumps(rows, indent=1, default=str))
        return
    out = score(load_runs(), GateConfig(**base))
    (Path(__file__).parent / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
