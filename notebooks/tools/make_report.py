"""Generates notebooks/yolo_poc_v3.ipynb (Stage 1 progress report).

Usage (from notebooks/):  ../.venv/bin/python tools/make_report.py yolo_poc_v3.ipynb
Then execute it with nbclient (see STAGE1_PROGRESS.md).
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Stage 1 (YOLO pre-filter): progress update

My part of the project is Stage 1: the cheap YOLO part that runs on every frame and decides which moments are worth sending to the VLM.
The VLM is slow, so the less video it has to look at the better. But if Stage 1 misses an event, the VLM never sees it,
so **missing events is the thing I care about most**, and too many false alarms comes second.

This notebook covers what Stage 1 does now, how I test it, the results, and what I tried this week to push recall up (including what didn't work).
Everything runs from cached YOLO outputs, so the notebook runs in under a minute.

**Demo videos** (open these to see it running): `demo/loitering_demo.mp4` and `demo/littering_demo.mp4`.
""")

md(r"""
## 1. What Stage 1 does

```
video ──► YOLO26s (people + bags/bottles) + ByteTrack tracking + YOLO26s-pose (body keypoints)      ~100-125 ms per frame on my laptop CPU
      ──► simple rules  ──►  "candidates": time window + crop of the person + a hint for the VLM  ──►  VLM (Stage 2)
```

The rules I have right now:

| Rule | What it looks for | Meant for |
|---|---|---|
| `reach_down` | a wrist goes down to about knee height (person bending to the ground). Ignores people cut off by the frame edge and unsure keypoints | littering |
| `sustained_presence` | a person stays in roughly the same spot for 20 s (re-sent every 30 s if they keep staying) | loitering, smoking, rough sleeping |
| `long_presence` (new) | the same person stays **in view** for a long time, even if they walk around, as long as they stay within ~4 body-heights of one area (re-sent every 60 s) | loitering, pacing, wandering |
| `fast_motion` | a person moves faster than 1.2 body-heights per second | running, chasing |
| `unattended_object` | a bag whose owner has walked away, or with nobody around, for more than 10 s. A bag someone is carrying or wearing never counts | unattended items |
| `object_appeared` (new) | a bag that suddenly appears on its own, away from people, and stays put (e.g. it was thrown or dropped) | thrown / dropped items, littering |

All distances are in body-heights rather than pixels, so the same setting works whether people are near or far.
If several rules fire on the same person within 2.5 s, only one candidate is sent, so the VLM doesn't get the same person twice.

So loitering has two time limits: a short one for standing still (20 s) and a longer one for just being around (`long_presence`).
The longer one has to be longer than people normally take to walk through the scene, and that depends on the camera,
so I set it from a few minutes of that camera's normal footage (right now that's only done for the loitering camera).
""")

md(r"""
## 2. How I test it

Datasets:
* **Littering**: 330 short clips (30 frames each) of 5 actions: dropping, dumping, putting-down, throwing, tossing. The person littering is boxed in every frame.
  I had to rebuild this dataset first (see section 7).
* **Loitering**: the videos in our `data/loitering` folder (these come from the CUHK Avenue dataset). The labelled events are things like
  running, throwing stuff and odd behaviour in front of the camera. The training videos have no events at all, so I use them as **normal footage** to count false alarms.
* **Smoking**: factory CCTV at about 1 frame per second, with smokers boxed.

What I measure:
* **Event recall**: out of all the labelled events, how many got at least one candidate at the right time. *This is the main number.*
* **Crop recall**: same, but the crop sent to the VLM must also contain the right person (not someone standing next to them).
* **Video skipped**: how much of the video the VLM doesn't have to look at.
* **False alarms per minute**: candidates on normal footage where nothing happens.

To keep myself honest I split every dataset in half. I tuned all thresholds on one half and only ran the final numbers on the other half once.
I also compare against a **random guess**: pick a random person at the same moments my rule fires. If my rule isn't better than that, it's not really detecting anything.
""")

code(r"""
import pickle
from dataclasses import replace
from functools import partial
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, "exp")
from exp_harness import load_splits
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview
sys.path.insert(0, "tools")
from compare_models import model_splits
from stage1_report import (draw_case, false_alarm_rate, find_relinks, headline_table, id_fragmentation,
                           pick_example, recall_by_tag, relink_panel, show_grid)
from random_control import lift

pd.set_option("display.precision", 1)
MODEL = "{MODEL}"                                                         # YOLO26 size used for this report
TUNE, TEST = model_splits(MODEL, heldout=False), model_splits(MODEL, heldout=True)   # tune half / held-out half

# in-view threshold for the loitering camera, set from the TUNE half of its normal footage only
LOITER_CAM = GateConfig(inview_s=calibrate_inview([r for _, r in TUNE["normal"]]))
CFG = GateConfig()                                       # other cameras: in-view at the default 120 s
print(f"model YOLO26{MODEL}; loitering camera in-view threshold {LOITER_CAM.inview_s:.0f} s")
print({k: len(v) for k, v in TEST.items()}, "held-out sequences")
""")

md(r"""
## 3. Results (held-out half)
""")

code(r"""
table, details = headline_table([
    ("Littering", TEST["littering"], DEFAULT_RULES),
    ("Loitering", TEST["avenue"], DEFAULT_RULES, LOITER_CAM),
    ("Smoking", TEST["smoking"], DEFAULT_RULES),
], CFG)
table.T
""")

md(r"""
{RESULTS_TEXT}
""")

md(r"""
### Is it better than guessing?
""")

code(r"""
m, c, s = lift(partial(__import__("exp_harness").baseline, cfg=CFG), TEST["littering"])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(["random person,\nsame number of tries", "my rules"], [c, m], color=["#bbbbbb", "#f39c12"])
ax.errorbar([0], [c], yerr=[s], color="black", capsize=6)
for x, v, top in [(0, c, c + s), (1, m, m)]:
    ax.text(x, top + 3, f"{v:.0f}%", ha="center")
ax.set_ylabel("littering crop recall %"); ax.set_ylim(0, 100); ax.set_title("Littering: does the rule pick the right person?")
plt.tight_layout(); plt.show()
""")

md(r"""
{CONTROL_TEXT}
""")

md(r"""
### Littering by action
""")

code(r"""
recall_by_tag(details["Littering"])
""")

md(r"""
{ACTION_TEXT}
""")

md(r"""
### False alarms on normal footage
""")

code(r"""
false_alarm_rate(TEST["normal"], LOITER_CAM)
""")

md(r"""
{FA_TEXT}
""")

md(r"""
### Examples
Green box = labelled person/event, red box = what Stage 1 sent to the VLM. One hit and one miss per dataset.
""")

code(r"""
cases = []
for name, res, cfg in [("Littering", TEST["littering"], CFG), ("Loitering", TEST["avenue"], LOITER_CAM),
                       ("Smoking", TEST["smoking"], CFG)]:
    for kind in ("hit", "miss"):
        found = pick_example(res, cfg, kind)
        if found:
            g, r, t = found
            cases.append(draw_case(g, r, cfg, t, f"{name} {kind}: {g.seq.seq_id} at {t:.1f}s"))
show_grid(cases, cols=2, width=14)
""")

md(r"""
## 4. Trying to push recall up

{EXPERIMENT_TEXT}
""")

code(r"""
pd.read_csv("exp/heldout_results.csv").set_index("method").rename(columns=lambda c: c.replace("avenue", "loitering")).T
""")

md(r"""
{ROUND3_TEXT}
""")

code(r"""
pd.read_csv("exp/round3_changes.csv").set_index("version").T
""")

md(r"""
{EXPERIMENT_RESULT_TEXT}
""")

md(r"""
## 5. Occlusion: keeping the same ID when someone is blocked

For rules like loitering and bag ownership, a person needs to keep the same ID even if they walk behind a pillar for a bit.
ByteTrack keeps a lost person for 4 s, and I also keep low-confidence (half-hidden) detections of people it already knows.
On top of that I added **re-linking**: if a new ID appears near where someone disappeared in the last 8 s, with about the same height
and similar clothing colours, it gets the old ID back.

The table compares the same videos with re-linking off and on. Fewer distinct IDs means fewer people got split into several IDs.
The picture shows one real case: same person (red) before and after being blocked.
""")

code(r"""
rows = []
for stitch in (False, True):
    c = replace(LOITER_CAM, stitch=stitch)
    summary, _ = headline_table([("x", TEST["avenue"], DEFAULT_RULES, c)], c)
    rows.append({"re-linking": "on" if stitch else "off", **id_fragmentation(TEST["avenue"], c),
                 "event recall %": summary.iloc[0]["Event recall %"]})
display(pd.DataFrame(rows).set_index("re-linking"))
ex = find_relinks(TEST["avenue"], LOITER_CAM, limit=1)
if ex:
    show_grid(relink_panel(ex[0]), cols=2, width=13)
""")

md(r"""
{OCCLUSION_TEXT}
""")

md(r"""
## 6. Bigger YOLO models: is it worth it?
I re-ran everything with the small (`s`) and medium (`m`) versions of YOLO26 and YOLO26-pose, using exactly the same rules,
and compared them with the nano (`n`) version.
""")

code(r"""
pd.read_csv("exp/model_comparison.csv").set_index(["half", "model"]).T
""")

md(r"""
Speed on my laptop CPU (Apple M3 Pro), one frame at a time, detection + tracking + pose:
""")

code(r"""
lat = pd.read_json("bench_cache/latency.json")
lat.pivot(index="source", columns="model", values="total ms / frame").rename_axis(None, axis=1).add_suffix(" ms/frame")
""")

md(r"""
{MODEL_TEXT}
""")

md(r"""
## 7. Data problems I had to fix

* **Littering dataset naming.** Every image was called `video_<N>_<frame>`, but each action had its own `video_1..12`, so one filename
  matched up to 30 different frames. I rebuilt the 330 clips by matching each frame to the most similar frame that comes after it
  (script: `scripts/rebuild_littering_clips.py`, output in `data/littering/{train,valid,test}/`). Every match was clear-cut, and I spot-checked 48 clips by eye.
  The original Roboflow split had **only dropping and dumping in valid/test**, so I re-split it with 46/10/10 clips per action.
* **Loitering labels.** The `.mat` files in the folder turned out to be video data, not labels, so I downloaded the official frame labels (`data/loitering/avenue_gt/`).
* **Smoking frames** were shuffled by Roboflow. I put them back in order using the timestamps in the original filenames.
* **Tracker bug on low-fps video.** `model.track()` only returns tracks it has confirmed (seen twice), so at 1 fps about half the smokers never reached my rules.
  I now run the tracker myself and use unconfirmed tracks too, which brought untracked smokers down from ~50% to 5%.
""")

md(r"""
## 8. What's next

{NEXT_TEXT}
""")

md(r"""
## Appendix: run it on any video or folder of frames
""")

code(r"""
RUN_DEMO = False
SOURCE = "../data/littering/test/dumping_v01_5"     # a video file, a folder of frames, or one image

if RUN_DEMO:
    from stage1_gate import HandoffWriter, Stage1Gate, load_sequence, workload_summary
    seq = load_sequence(SOURCE, fps=10.0)
    writer = HandoffWriter("vlm_handoff_v3")
    result = Stage1Gate(CFG).run(seq, writer=writer)
    print("manifest for the VLM:", writer.write_manifest())
    display(pd.DataFrame([workload_summary(result)]))
""")


def build(path, texts):
    out = []
    for c in cells:
        c = dict(c)
        src = "".join(c["source"])
        for k, v in texts.items():
            src = src.replace("{" + k + "}", v.strip("\n"))
        c["source"] = src.splitlines(keepends=True)
        out.append(c)
    nb = {"cells": out,
          "metadata": {"kernelspec": {"display_name": ".venv (3.12.5)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.5"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell{i}"
    json.dump(nb, open(path, "w"), indent=1)


if __name__ == "__main__":
    texts = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    build(sys.argv[1], texts)
    print("wrote", sys.argv[1])
