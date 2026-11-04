"""Demo video of the VLM stage (Yuchuan's fine-tuned CLIP classifier, anomaly_eval predict-clip) judging the Stage 1
candidates on ICVL clips: the true events from icvl_works.mp4 and the false alerts from icvl_misses.mp4.

  prepare   replay the cached gate on each clip and save every candidate exactly as the data pipeline hands it to
            the VLM (crop, context frame, "short" 4 s and "long" 20 s views, tasks from TASKS_BY_EVENT)
            -> demo/vlm_demo/candidates.json
  classify  run Yuchuan's CPU bundle classifier on every candidate, all five tasks -> demo/vlm_demo/predictions.json
  render    -> demo/vlm_demo.mp4: each candidate box is labelled with the VLM's verdict on the tasks the pipeline
            asks for that rule (green = anomaly, grey = normal)

Usage (from notebooks/): ../.venv/bin/python tools/make_vlm_demo.py prepare | classify <bundle dir> | render
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

NB = Path(__file__).resolve().parents[1]
ROOT = NB.parent
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp" / "icvl"), str(ROOT / "data-pipeline")]

from data_pipeline.detection.stage1 import TASKS_BY_EVENT, VIEW_WINDOWS_S  # noqa: E402
from make_icvl_showcase import CFG, NOTE, RUNS, card  # noqa: E402
from make_showcase import GREY, WHITE  # noqa: E402
from render_demo import open_writer, render_sequence  # noqa: E402
from stage1_gate import HandoffWriter, replay_rules  # noqa: E402

OUT = NB / "demo" / "vlm_demo"
FPS = 10
# (clip, window (s), what is really happening)
CLIPS = [("01/AA_014_01_20170829_01", (0.0, 12.0), "littering: bag put down at the shelter"),
         ("01/AA_001_01_20170829_01", (0.0, 11.0), "littering: thrown"),
         ("01/AA_052_01_20170829_01", (1.0, 15.0), "littering: bag dropped"),
         ("05/AA_005_05_20140802_03", (0.0, 17.0), "smoking"),
         ("08/AA-006-08-20140524-02", (4.0, 36.0), "standing still in a corridor"),
         ("01/AA-016-01-20140502-01", (5.0, 13.0), "nothing: walking with something in hand"),
         ("03/AR-004-03-20151003-01", (0.0, 7.0), "nothing: phone to the face"),
         ("02/AR-021-02-20151031-01", (0.0, 7.0), "nothing: pushing a bicycle")]


def export(items, out_dir):
    """items: (seq_id, alert, extra fields). Saves each alert as the data pipeline hands it to the VLM (crop, context
    frame, "short" and "long" views ending at the trigger frame) -> list of candidates."""
    by_seq = {}
    for sid, a, extra in items:
        by_seq.setdefault(sid, []).append((a, extra))
    cands = []
    for k, (sid, alerts) in enumerate(by_seq.items()):
        g, r = RUNS[sid]
        name = sid.split("/", 1)[1].replace("/", "_")
        by_frame = {a["frame_index"]: (a, extra) for a, extra in alerts}
        last = max(by_frame)
        writer = HandoffWriter(Path(out_dir) / name, buffer_s=max(VIEW_WINDOWS_S.values()) + 5.0)
        stream = SimpleNamespace(seq_id=sid, source=sid, paths=None)
        writer.start(stream)
        obs = {o.index: o for o in r.observations if o.index <= last}
        for fr in g.seq:
            if fr.index > last:
                break
            o = obs.get(fr.index)
            if o is None:
                continue
            writer.observe(fr.image, o.t, o.index)
            if o.index not in by_frame:
                continue
            a, extra = by_frame[o.index]
            e = dict(a, event_id=f"{name}_{a['event_id']}")
            writer.save(e, fr.image, stream, t=o.t)
            e["views"] = {}
            for view, span in VIEW_WINDOWS_S.items():
                paths, times = writer.window_frames(e, fr.image, stream, o.t - span, o.t, tag=f"_{view}")
                e["views"][view] = {"paths": paths, "times_s": times}
            e.update(seq=sid.split("/", 1)[1], tasks=list(TASKS_BY_EVENT.get(e["event_type"], ())), **extra)
            cands.append(e)
        if k % 20 == 0:
            print(k, "videos exported", flush=True)
    return cands


def prepare():
    items = []
    for seq, (w0, w1), truth in CLIPS:
        r = RUNS[f"icvl/{seq}"][1]
        items += [(f"icvl/{seq}", a, {"truth": truth}) for a in replay_rules(r, CFG)[0] if w0 <= a["timestamp_s"] <= w1]
    (OUT / "candidates.json").write_text(json.dumps(export(items, OUT), indent=1, default=float))


ALL_TASKS = ["littering", "smoking", "loitering", "rough_sleeping", "unattended_items"]
# Runs inside the bundle's own code (its resident pipeline classifier, anomaly_eval.reasoning.backends.ClipStrategy).
# One JSON line per candidate, appended as it finishes; candidates already in the output are skipped.
DRIVER = """
import json, os, sys
bundle, cands, out, model = sys.argv[1:5]
sys.path.insert(0, bundle)
from anomaly_eval.reasoning.backends import ClipStrategy
done = {json.loads(l)["event_id"] for l in open(out)} if os.path.exists(out) else set()
todo = [e for e in json.load(open(cands)) if e["event_id"] not in done]
engine = ClipStrategy(model=model, device="cpu")
with open(out, "a") as fh:
    for i, e in enumerate(todo):
        r = engine.classify({"views": e["views"], "observed_until_s": e["timestamp_s"], "tasks": %r})
        fh.write(json.dumps({"event_id": e["event_id"], "seconds": r["seconds"], "predictions": r["predictions"]},
                            default=str) + "\\n")
        fh.flush()
        print(i + 1, "/", len(todo), e["event_id"], round(r["seconds"], 1), "s", flush=True)
""" % ALL_TASKS


def classify(bundle, model, cands=OUT / "candidates.json", out=OUT / "predictions.jsonl"):
    """Yuchuan's CPU bundle (capstone-mobileclip2-s2-cpu): every candidate is asked all five tasks."""
    bundle = Path(bundle).resolve()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_CACHE": str(bundle / "data/models/huggingface/hub"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    subprocess.run([sys.executable, "-I", "-c", DRIVER, str(bundle), str(cands), str(out), model], check=True, env=env)


def load_predictions(path=OUT / "predictions.jsonl"):
    return {d["event_id"]: d["predictions"] for d in map(json.loads, Path(path).read_text().splitlines())}


def verdict(e, pred):
    """'VLM: littering YES, unattended items no' for the tasks the pipeline asks about this rule."""
    parts, any_yes = [], False
    for task in e["tasks"]:
        d = pred[task].get("decision")
        any_yes |= d == "anomaly"
        parts.append(f"{task.replace('_', ' ')} {'YES' if d == 'anomaly' else 'no' if d == 'normal' else '?'}")
    return "VLM: " + ", ".join(parts), any_yes


def render():
    cands = json.loads((OUT / "candidates.json").read_text())
    data = {"model": "mobileclip2-s2", "predictions": load_predictions()}
    w = open_writer(NB / "demo" / "vlm_demo.mp4", FPS)
    card(w, "Stage 1 + VLM on real CCTV (ICVL)", ["Stage 1 sends a candidate (orange); the VLM then answers",
                                                  f"the questions the pipeline asks for that rule ({data['model']})",
                                                  "Green = VLM says anomaly, grey = VLM says normal"], 5.0)
    for seq, window, truth in CLIPS:
        mine = [e for e in cands if e["seq"] == seq]
        for e in mine:
            e["caption"], yes = verdict(e, data["predictions"][e["event_id"]])
            e["colour"] = (80, 200, 80) if yes else (170, 170, 170)
        card(w, f"Really: {truth}", [f"{len(mine)} candidate(s) from Stage 1"], 2.5, WHITE if truth[:7] != "nothing" else GREY)
        g, r = RUNS[f"icvl/{seq}"]
        g.gt = {}
        render_sequence(w, g, r, mine, f"Really: {truth}", f"{seq}  |  {NOTE} + VLM", CFG, window=window,
                        repeat_frames=1 if window[1] - window[0] > 20 else 2, hold_s=3.0)
    w.release()
    print("wrote", NB / "demo" / "vlm_demo.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prepare": prepare, "classify": lambda: classify(sys.argv[2], "mobileclip2-s2"), "render": render}[cmd]()
