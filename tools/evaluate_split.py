"""End-to-end evaluation of the camera backend on the test side of the team split.

    python tools/evaluate_split.py run SPLITS DATASETS OUT --jobs 6 \
        --gemma-url http://127.0.0.1:8090/v1 --model ensemble-20261119
    python tools/evaluate_split.py score SPLITS DATASETS OUT

SPLITS is notebooks/exp/eval/splits.json (or a rebuilt copy), DATASETS the
capstone_datasets folder it was built from. `run` plays every distinct test
video through data-pipeline/main.py exactly as a camera would see it (real-time
pacing, 5 fps streaming, rules, screening classifier, Gemma, incidents in
PostgreSQL), one process per video, `--jobs` at a time. `score` reads each
run's event journal and the label files and reports every stage:

  stream     frames dropped, wall time against video time
  triggers   candidate windows per minute; labelled events that got at least
             one window asking about their behaviour
  screening  window level: AUC, recall and share of negatives flagged
  incidents  event recall, incident precision, false incidents per hour,
             time from event start to first alert
  gemma      reviews done or skipped; agreement with the labels

A behaviour is scored only on videos labelled for it: the label files are
complete for their own anomaly and say nothing about the others, so an alert
for another behaviour there is counted separately as unverifiable.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = {
    "smoking": "smoking",
    "littering": "littering",
    "loitering": "loitering",
    "rough sleeping": "rough_sleeping",
    "unattended items": "unattended_items",
}
TASKS = list(TASK.values())
# Seconds after a labelled interval in which an alert still counts: the
# screening window ends at the trigger, and a dropped item stays visible.
TOLERANCE = {"littering": 10.0}
DEFAULT_TOLERANCE = 5.0
# Window spans, as the incident recorder uses them (short view for littering).
SPAN = {"littering": 4.0}
DEFAULT_SPAN = 20.0
QUIET = 30.0  # IncidentRecorder default
# Duration-defined behaviours on duration rules open an incident whatever the
# screening classifier says (vision_runtime.service.bypasses_screening); kept
# here so journals from code before that change score the same way.
BYPASS = {"loitering", "rough_sleeping"}
BYPASS_RULES = {"long_presence", "sustained_presence"}


def opens(row, task):
    """True when this window opens or extends an incident for `task`."""
    p = (row.get("predictions") or {}).get(task)
    if p is None:
        return False
    if p.get("decision") == "anomaly":
        return True
    return task in BYPASS and row.get("kind") in BYPASS_RULES


def test_videos(splits, datasets):
    """{video_key: {"path": Path, "labels": [clip, ...]}} for the test side."""
    clips = json.loads(Path(splits).read_text())["clips"]
    videos = {}
    for clip in clips:
        if clip["split"] != "test":
            continue
        path = Path(datasets) / clip["video"]
        if not path.exists():
            continue
        entry = videos.setdefault(clip["video_key"], {"path": path, "labels": []})
        entry["labels"].append(clip)
    return videos


def source_id(key):
    return "t" + hashlib.sha1(key.encode()).hexdigest()[:10]


def run_one(key, entry, args):
    sid = source_id(key)
    out = Path(args.out) / sid
    done = out / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(ROOT / "data-pipeline/main.py"),
        "--video",
        str(entry["path"].resolve()),
        "--source-id",
        sid,
        "--device",
        "cpu",
        "--out",
        str(out / "ai"),
        "--recordings-dir",
        str(out / "recordings"),
    ]
    if args.model:
        command += ["--model", args.model]
    if args.gemma_url:
        command += ["--gemma-url", args.gemma_url, "--gemma-model", args.gemma_model]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "DATABASE_URL": args.database_url,
    }
    started = time.time()
    with (out / "pipeline.log").open("w") as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=log)
    result = {
        "video_key": key,
        "source_id": sid,
        "video": str(entry["path"]),
        "exit_code": code,
        "wall_s": round(time.time() - started, 1),
        "duration_s": max(c["duration_s"] for c in entry["labels"]),
    }
    if code == 0:
        done.write_text(json.dumps(result, indent=1))
    return result


def run(args):
    videos = test_videos(args.splits, args.datasets)
    order = sorted(videos, key=lambda k: -videos[k]["labels"][0]["duration_s"])
    print(f"{len(order)} test videos, {args.jobs} at a time", flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_one, k, videos[k], args) for k in order]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            print(
                f"[{n}/{len(order)}] {Path(r['video']).name} exit {r['exit_code']} "
                f"{r['wall_s']:.0f} s for {r['duration_s']:.0f} s of video",
                flush=True,
            )


# ---------------------------------------------------------------- scoring


def intervals(task, label):
    """Labelled (start, end) seconds in which `task` is happening."""
    out = []
    for e in label.get("events", []):
        if task == "littering":
            a, b = e.get("t_littering_s") or (e["t_start_s"], e["t_end_s"])
        elif task == "loitering":
            a, b = e.get("t_loitering_from_s", e["t_start_s"]), e["t_end_s"]
        elif task == "unattended_items":
            a = e.get("t_unattended_s")
            b = e.get("t_removed_s") or label.get("duration_s")
            if a is None:
                continue
        else:
            a, b = e["t_start_s"], e["t_end_s"]
        out.append((float(a), float(b)))
    return out


def journal(out):
    """Last state of every event in a run's journal, oldest first."""
    rows = {}
    for path in sorted(Path(out).glob("ai/**/events.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            state = rows.setdefault(row["event_id"], {})
            state.update(row)
            state.setdefault("statuses", []).append(row["status"])
    return sorted(rows.values(), key=lambda r: r.get("timestamp_s", 0))


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def num(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def score(args):
    videos = test_videos(args.splits, args.datasets)
    labels_root = Path(args.datasets)
    stream = collections.Counter()
    kinds = collections.Counter()
    minutes = 0.0
    per = {t: collections.defaultdict(list) for t in TASKS}
    unverifiable = collections.Counter()
    unverifiable_minutes = collections.Counter()
    missing = []
    for key, entry in videos.items():
        out = Path(args.out) / source_id(key)
        done = out / "done.json"
        if not done.exists():
            missing.append(entry["path"].name)
            continue
        meta = json.loads(done.read_text())
        duration = meta["duration_s"]
        minutes += duration / 60
        stream["videos"] += 1
        stream["video_s"] += duration
        stream["wall_s"] += meta["wall_s"]
        log = (out / "pipeline.log").read_text(errors="ignore")
        drops = [int(x) for x in re.findall(r"dropped_frames=(\d+)", log)]
        stream["dropped"] += drops[-1] if drops else 0
        rows = journal(out)
        labelled = {}
        for clip in entry["labels"]:
            task = TASK[clip["anomaly"]]
            label = json.loads((labels_root / clip["label"]).read_text())
            labelled.setdefault(task, []).extend(intervals(task, label))
        screened = [r for r in rows if r.get("predictions")]
        for r in screened:
            kinds[r["kind"]] += 1
        for task in TASKS:
            alerts = [(r["timestamp_s"], r) for r in screened if opens(r, task)]
            if task not in labelled:
                unverifiable[task] += len(merge(task, [t for t, _ in alerts]))
                unverifiable_minutes[task] += duration / 60
                continue
            spans = labelled[task]
            tol = TOLERANCE.get(task, DEFAULT_TOLERANCE)

            def inside(t, spans=spans, tol=tol):
                return any(a <= t <= b + tol for a, b in spans)

            def near(t, spans=spans, tol=tol, task=task):
                span = SPAN.get(task, DEFAULT_SPAN)
                return any(a - span <= t <= b + tol + span for a, b in spans)

            d = per[task]
            d["minutes"].append(duration / 60)
            # Stage: triggers and screening, window level.
            for r in screened:
                p = r["predictions"].get(task)
                if p is None:
                    continue
                t = r["timestamp_s"]
                if p.get("decision") == "unknown":
                    d["unknown"].append(1)
                    continue
                d["unknown"].append(0)
                if inside(t):
                    d["pos"].append(p["score"] - p["threshold"])
                    d["pos_flag"].append(p["decision"] == "anomaly")
                elif not near(t):
                    d["neg"].append(p["score"] - p["threshold"])
                    d["neg_flag"].append(p["decision"] == "anomaly")
            for a, b in spans:
                asked = [
                    r
                    for r in screened
                    if task in r["predictions"] and a <= r["timestamp_s"] <= b + tol
                ]
                known = [
                    r
                    for r in asked
                    if r["predictions"][task].get("decision") != "unknown"
                ]
                hits = [r["timestamp_s"] for r in asked if opens(r, task)]
                flagged = [
                    t
                    for t, r in zip(hits, [r for r in asked if opens(r, task)])
                    if r["predictions"][task].get("decision") == "anomaly"
                ]
                # Stricter: the window itself overlaps the event.
                overlap = [
                    t for t in hits if t <= b + min(tol, SPAN.get(task, DEFAULT_SPAN))
                ]
                reviews = [
                    r["confirmation"][task].get("decision")
                    for r in rows
                    if task in (r.get("confirmation") or {})
                    and a <= r["timestamp_s"] <= b + tol
                ]
                d["events"].append(
                    {
                        "asked": bool(asked),
                        "known": bool(known),
                        "found": bool(hits),
                        "by_screening": bool(flagged),
                        "overlap": bool(overlap),
                        "delay": (min(hits) - a) if hits else None,
                        "gemma_reviewed": bool(reviews),
                        "gemma_confirmed": "anomaly" in reviews,
                    }
                )
            # Stage: incidents.
            for incident in merge(task, [t for t, _ in alerts]):
                d["incidents"].append(any(inside(t) for t in incident))
            # Stage: Gemma on the windows it reviewed.
            for r in rows:
                c = (r.get("confirmation") or {}).get(task)
                if "confirmation_not_run" in r.get("statuses", []) and task in (
                    r.get("confirmation_tasks") or []
                ):
                    d["gemma_skipped"].append(1)
                if not c:
                    continue
                truth = inside(r["timestamp_s"])
                d["gemma"].append((truth, c.get("decision")))
    report(
        args, stream, kinds, minutes, per, unverifiable, unverifiable_minutes, missing
    )


def merge(task, times):
    """Alert times grouped into incidents the way IncidentRecorder does."""
    span = SPAN.get(task, DEFAULT_SPAN)
    incidents = []
    for t in sorted(times):
        if incidents and t - span - incidents[-1][-1] <= QUIET:
            incidents[-1].append(t)
        else:
            incidents.append([t])
    return incidents


def report(
    args, stream, kinds, minutes, per, unverifiable, unverifiable_minutes, missing
):
    lines = [f"# End-to-end evaluation: {Path(args.out).name}", ""]
    lines.append(
        f"{stream['videos']} test videos, {minutes:.0f} min of footage"
        + (
            f"; {len(missing)} without a finished run: {', '.join(missing[:5])}"
            if missing
            else ""
        )
    )
    lines += ["", "## Stream", ""]
    lines.append(
        f"Frames dropped: {stream['dropped']}. Wall time {stream['wall_s'] / 60:.0f} min for "
        f"{stream['video_s'] / 60:.0f} min of video (real-time pacing; the excess is model "
        "loading and the Gemma queue draining after each video)."
    )
    lines += ["", "## Triggers", ""]
    total = sum(kinds.values())
    lines.append(
        f"{total} candidate windows screened ({total / max(minutes, 1e-9):.1f} per minute): "
        + ", ".join(f"{k} {v}" for k, v in kinds.most_common())
    )
    lines += [
        "",
        "| Behaviour | Events | Asked about | Scored (not unknown) | Windows unknown |",
        "|---|---|---|---|---|",
    ]
    for task in TASKS:
        d = per[task]
        ev = d["events"]
        if not ev:
            continue
        lines.append(
            f"| {task} | {len(ev)} | {pct(sum(e['asked'] for e in ev) / len(ev))} | "
            f"{pct(sum(e['known'] for e in ev) / len(ev))} | "
            f"{pct(sum(d['unknown']) / len(d['unknown']) if d['unknown'] else None)} |"
        )
    lines += [
        "",
        "## Screening classifier (window level)",
        "",
        "| Behaviour | Positive windows | Negative windows | AUC | Recall | Negatives flagged |",
        "|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        d = per[task]
        if not d["events"]:
            continue
        lines.append(
            f"| {task} | {len(d['pos'])} | {len(d['neg'])} | {num(auc(d['pos'], d['neg']))} | "
            f"{pct(sum(d['pos_flag']) / len(d['pos_flag']) if d['pos_flag'] else None)} | "
            f"{pct(sum(d['neg_flag']) / len(d['neg_flag']) if d['neg_flag'] else None)} |"
        )
    lines += [
        "",
        "## Incidents",
        "",
        "| Behaviour | Events found | Found by screening | Window overlaps event | Incidents | Correct incidents | False incidents per hour | Median time to alert |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        d = per[task]
        ev = d["events"]
        if not ev:
            continue
        delays = sorted(e["delay"] for e in ev if e["delay"] is not None)
        hours = sum(d["minutes"]) / 60
        wrong = sum(not ok for ok in d["incidents"])

        def share(key, ev=ev):
            n = sum(e[key] for e in ev)
            return f"{n}/{len(ev)} ({pct(n / len(ev))})"

        lines.append(
            f"| {task} | {share('found')} | {share('by_screening')} | {share('overlap')} | "
            f"{len(d['incidents'])} | "
            f"{pct(sum(d['incidents']) / len(d['incidents']) if d['incidents'] else None)} | "
            f"{wrong / hours:.1f} | "
            f"{(str(round(delays[len(delays) // 2])) + ' s') if delays else '—'} |"
        )
    lines += [
        "",
        "Alerts on videos not labelled for that behaviour (cannot be judged):",
        ", ".join(
            f"{t} {unverifiable[t]} in {unverifiable_minutes[t]:.0f} min"
            for t in TASKS
            if unverifiable_minutes[t]
        ),
        "",
        "## Gemma review",
        "",
        "Per window: windows Gemma reviewed (screening flagged them, or a duration "
        "rule bypassed screening) and the share it called an anomaly. Per event: "
        "labelled events with at least one review, and those with at least one "
        "confirming review.",
        "",
        "| Behaviour | Reviewed | Skipped | Confirms on labelled windows | Confirms on unlabelled windows | Events reviewed | Events confirmed |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        d = per[task]
        g = d["gemma"]
        if not d["events"]:
            continue
        pos = [dec for truth, dec in g if truth]
        neg = [dec for truth, dec in g if not truth]
        lines.append(
            f"| {task} | {len(g)} | {len(d['gemma_skipped'])} | "
            f"{pct(sum(x == 'anomaly' for x in pos) / len(pos) if pos else None)} of {len(pos)} | "
            f"{pct(sum(x == 'anomaly' for x in neg) / len(neg) if neg else None)} of {len(neg)} | "
            f"{sum(e['gemma_reviewed'] for e in d['events'])}/{len(d['events'])} | "
            f"{sum(e['gemma_confirmed'] for e in d['events'])}/"
            f"{sum(e['gemma_reviewed'] for e in d['events'])} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    (Path(args.out) / "report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "score"):
        p = sub.add_parser(name)
        p.add_argument("splits")
        p.add_argument("datasets")
        p.add_argument("out")
    r = sub.choices["run"]
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--model")
    r.add_argument("--gemma-url")
    r.add_argument("--gemma-model", default="gemma4-e4b")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args()
    (run if args.action == "run" else score)(args)


if __name__ == "__main__":
    main()
