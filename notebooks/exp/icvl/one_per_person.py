"""How many VLM calls does capping alerts per person save, and what does it cost in caught incidents?

Every gate alert currently becomes a VLM candidate. Each variant is a post-filter on the gate's alerts, keyed on the
alert's entity_id (the re-linked track):
  once           first alert per person only
  once_per_rule  first alert per person per rule (a person standing still who later reaches down is sent twice)
  gap_<N>        at most one alert per person every N seconds, any rule
  rule_gap_<N>   at most one alert per person per rule every N seconds
  still_once     stays put / long in view at most once per person; other rules unchanged
Writes exp/icvl/one_per_person.json. Usage (from notebooks/): ../.venv/bin/python exp/icvl/one_per_person.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from score import load_runs, score  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), pose_action_cooldown_s=15.0)


def capped(key, gap_s=float("inf")):
    def filt(res, alerts):
        last, kept = {}, []
        for a in sorted(alerts, key=lambda a: a["timestamp_s"]):
            k = key(a)
            if k in last and a["timestamp_s"] - last[k] < gap_s:
                continue
            last[k] = a["timestamp_s"]
            kept.append(a)
        return kept
    return filt


VARIANTS = {"current": None,
            "once": capped(lambda a: a["entity_id"]),
            "once_per_rule": capped(lambda a: (a["entity_id"], a["event_type"])),
            **{f"gap_{n}": capped(lambda a: a["entity_id"], n) for n in (30, 60, 120)},
            **{f"rule_gap_{n}": capped(lambda a: (a["entity_id"], a["event_type"]), n) for n in (30, 60, 120)},
            "still_once": capped(lambda a: (a["entity_id"], "still") if a["event_type"] in STILL else (a["event_id"],))}
STILL = ("sustained_presence", "long_presence")


def main():
    runs, rows = load_runs(), []
    only = sys.argv[1:]
    for name, filt in VARIANTS.items():
        if only and name not in only:
            continue
        out = score(runs, CFG, filt)
        rows.append({"variant": name, "alerts": out["alerts"],
                     "false_alerts": out["alert_kinds"].get("on a labelled person, nothing labelled", 0)
                     + out["alert_kinds"].get("on nobody labelled", 0),
                     "quiet_alerts_min": out["videos_with_no_labelled_event"]["alerts_per_min"],
                     **{ev: f'{c["caught"]}/{c["incidents"]}' for ev, c in out["events"].items()}})
        print(json.dumps(rows[-1]), flush=True)
    (HERE / "one_per_person.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
