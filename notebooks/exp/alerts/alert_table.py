"""One row per Stage 1 alert on TUNE / TRAIN data, labelled real / false, with simple facts about the alerted person.

Alerts come from replaying the current defaults (rules + pose cue) frame by frame, so each alert's person, track
state and frame are known. Labels:
  littering tune clips      real if the crop covers the labelled litterer within 0.5 s (stage1_bench.box_hit),
                            else false (a bystander in the same clip)
  Avenue tune throwing      real if on a labelled throwing incident (label review, running excluded), else false
  Avenue tune normal videos false
  IWDD train normal videos  false
  IWDD train dumping videos with a labelled dumper (iwdd_actors_train.json): real if on the dumper's track in
                            [onset - 1, onset + 10 s]; false if more than 10 s from the onset; otherwise left out
Facts per alert: rule, person height (fraction of frame), speed (body heights / s over the last 1 s), track age,
people in frame, nearest other person (body heights), touches the frame edge, mean keypoint confidence, how low the
wrist is relative to the knee, pose-cue score (latest window of that person), source and video.
Usage (from notebooks/): ../.venv/bin/python exp/alerts/alert_table.py   -> exp/alerts/alerts_tune.pkl (pandas)
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam"), str(NB / "exp" / "gt_review")]

import pandas as pd  # noqa: E402
from compare_models import model_splits  # noqa: E402
from counts_precision import throwing_labels  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from scoped import scoped  # noqa: E402
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import (DEFAULT_RULES, L_KNEE, L_WRIST, R_KNEE, R_WRIST, GateConfig, RuleEngine,  # noqa: E402
                         touches_border)

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
ACTORS = json.loads((NB / "exp" / "newcam" / "iwdd_actors_train.json").read_text())["actors"]


def facts(eng, obs, e, cid_of):
    """Facts about the alerted person at the alert frame (None-filled when the alert is not on a person)."""
    out = {"rule": e["event_type"], "t": e["timestamp_s"], "n_people": len(obs.persons)}
    cid = int(e["entity_id"][6:]) if e["entity_id"].startswith("person") else None
    p = next((q for c, q in zip(eng.last_cids, obs.persons) if c == cid), None)
    if p is None:
        return out, None
    b = p["box"]
    h = max(1.0, b[3] - b[1])
    st = eng.people.get(cid, {})
    hist = [q for q in st.get("hist", []) if q[0] >= obs.t - 1.0 and q[5] == p["raw_id"]]
    speed = None
    if len(hist) >= 2 and hist[-1][0] > hist[0][0]:
        speed = math.hypot(hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2]) / (hist[-1][0] - hist[0][0]) / h
    others = [q for q in obs.persons if q is not p]
    near = min((math.hypot((q["box"][0] + q["box"][2] - b[0] - b[2]) / 2, q["box"][3] - b[3]) / h for q in others),
               default=None)
    k = p.get("kpts")
    low = kc = None
    if k is not None:
        kc = float(np.mean(k[:, 2]))
        knees = [k[i, 1] for i in (L_KNEE, R_KNEE) if k[i, 2] >= 0.3]
        wr = [k[i, 1] for i in (L_WRIST, R_WRIST) if k[i, 2] >= 0.3]
        if knees and wr:
            low = (max(wr) - float(np.mean(knees))) / h
    pa = eng.pose_action
    score = None
    if pa is not None:
        score = pa.get("last_score", {}).get(p["raw_id"])
    out.update(height=h / obs.frame_hw[0], speed=speed, track_age=obs.t - st.get("first_t", obs.t), near_h=near,
               edge=touches_border(b, obs.frame_hw), kp_conf=kc, wrist_low=low, pose_score=score, raw_id=p["raw_id"])
    return out, p


class Recorder(RuleEngine):
    """RuleEngine that also remembers each person's latest pose-cue score."""

    def _pose_action(self, obs, cid_of):
        pa = self.pose_action
        ws = pa["windows"].step(obs)
        out = []
        if not ws:
            return out
        probs = pa["model"].predict(np.stack([w["sum"] for w in ws]))
        scores = pa.setdefault("last_score", {})
        for w, prob in zip(ws, probs):
            rid = w["raw_id"]
            scores[rid] = float(prob)
            if prob >= pa["thr"] and obs.t - pa["last"].get(rid, -1e9) >= self.cfg.pose_action_cooldown_s:
                pa["last"][rid] = obs.t
                out.append(self._event("pose_action", f"person{cid_of[rid]}", obs, w["box"], score=round(float(prob), 3)))
        return out


def replay(r):
    eng = Recorder(CFG, r.fps)
    rows = []
    for o in r.observations:
        for e in eng.step(o):
            f, _ = facts(eng, o, e, None)
            f.update(roi=e["roi_box"], frame_index=e["frame_index"])
            rows.append(f)
    return rows


def main():
    T = model_splits("s", heldout=False)
    table = []

    def add(rows, source, video):
        for f in rows:
            f.update(source=source, video=video)
            table.append(f)

    for g, r in T["littering"]:
        obs_t = {o.index: o.t for o in r.observations}
        rows = replay(r)
        for f in rows:
            near = [i for i in g.gt if i in obs_t and abs(obs_t[i] - f["t"]) <= 0.5]
            f["real"] = any(box_hit(f["roi"], b, min_cover=0.5) for i in near for b in g.gt[i])
        add(rows, "littering clips", r.seq_id)
    for g, r in scoped(T["avenue"], keep=("in",), path=throwing_labels()):
        obs_t = {o.index: o.t for o in r.observations}
        rows = replay(r)
        for f in rows:
            near = [i for i in g.gt if i in obs_t and abs(obs_t[i] - f["t"]) <= 0.5]
            f["real"] = any(box_hit(f["roi"], b, min_cover=0.5) for i in near for b in g.gt[i])
        add(rows, "walkway (Avenue) incidents", r.seq_id)
    for _, r in T["normal"]:
        rows = replay(r)
        for f in rows:
            f["real"] = False
        add(rows, "walkway (Avenue) normal", r.seq_id)
    labels = iwdd_labels("train")
    for _, r in pickle.loads((NB / "bench_cache/models/s/iwdd_train.pkl").read_bytes()):
        lab, vid = labels[r.seq_id], r.seq_id.split("/")[-1]
        rows = replay(r)
        if not lab["positive"]:
            for f in rows:
                f["real"] = False
            add(rows, "dump sites normal", r.seq_id)
            continue
        if vid not in ACTORS:
            continue
        kept = []
        for f in rows:
            dt = f["t"] - lab["onset_s"]
            if -1 <= dt <= 10:
                f["real"] = f.get("raw_id") in ACTORS[vid]
                kept.append(f)
            elif abs(dt) > 10:
                f["real"] = False
                kept.append(f)
        add(kept, "dump sites dumping", r.seq_id)
    df = pd.DataFrame(table)
    (HERE / "alerts_tune.pkl").write_bytes(pickle.dumps(df))
    print(df.groupby(["source", "real"]).size().unstack(fill_value=0))
    print(df.groupby(["rule", "real"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
