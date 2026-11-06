# Stage 1: the YOLO gate

Stage 1 watches CCTV on a CPU and decides which moments are worth sending to the VLM (Stage 2). It is tuned for
recall: it should miss as little as possible, and the VLM is there to remove false alarms. Each alert is drawn around
one person (or one item), and that person's crop plus a short clip window is what the VLM receives.

Owner: Ian Varella. Branch: `feat/stage1-integration`.

## How it works

```
video, 5 fps
  -> YOLO26s detector (OpenVINO) + ByteTrack            people and items, with track ids
  -> YOLO26s-pose                                       17 body points per person
  -> rules, per tracked person                          candidate alerts
  -> crop check (reach-down and pose-cue alerts only)   drops alerts that look harmless in the pixels
  -> VLM
```

| Rule | Fires when | Aimed at |
|---|---|---|
| Reach-down | The wrist drops well below the hip (within 0.15 body heights of the knee) while the person is slow or still | Littering, dumping |
| Pose cue | A small trained model on 1 s of body points scores a throw, toss or hand-to-mouth motion | Littering, smoking |
| Long in view | The same person has been in view for 60 s | Loitering |
| Stays put | The same person has stayed within 0.6 body heights of one spot for 20 s | Loitering, smoking |
| Item left alone | A bag or other item has been away from the person who brought it for 10 s | Unattended items |

Two learned checks sit on top of the rules. The **reach verifier** is a small MLP on the same 1 s of body points
that re-checks each reach-down. The **crop check** is a frozen MobileCLIP2-S2 image encoder with a logistic regression,
scoring crops at t-1 s, t and t+1 s.

## Results on ICVL

ICVL is real outdoor CCTV from 11 cameras. Settings were tuned on the odd-numbered videos; the numbers below are
the even-numbered half (471 videos, 5.3 h). Both halves share the same cameras, so this is "tuned on one half,
reported on the other", not a test on unseen cameras.

| Event | Caught | Median time to first alert |
|---|---|---|
| Littering | 240 / 267 (90 %) | 0.3 s |
| &nbsp;&nbsp;put down while standing still | 207 / 222 (93 %) | 0.4 s |
| &nbsp;&nbsp;dropped while walking | 33 / 45 (73 %) | 0.0 s |
| Smoking | 81 / 116 (70 %) | 3.7 s |
| Loitering (in view 60 s+) | 66 / 70 (94 %) | 13.3 s |

* Normal people flagged at least once: 223 / 1,801 (12.4 %).
* 70 % of alerts are on a person during one of their labelled incidents.
* 0.92 alerts per minute on footage with no incident.
* Speed (Apple M3 Pro, OpenVINO): detector 36 ms, pose 39 ms, tracking and rules about 1 ms; gate total 70 ms per
  frame (p95 86 ms) against a 200 ms budget at 5 fps. The crop check takes about 220 ms per crop on one thread
  (464 ms in `timing.json`, where it shared the CPU with the gate) and only runs on reach-down and pose-cue alerts.

The numbers come from `exp/icvl/caught_table.py` and `exp/icvl/timing.py`.

Demo: [demo/icvl_activities.mp4](demo/icvl_activities.mp4) shows one caught example each of casual littering (0:00),
dumping (0:09), smoking (0:20) and loitering (0:29). Thick orange box = sent to the VLM; thin red box = the person
ICVL labels as doing the event.

## Files

| Path | What it is |
|---|---|
| `stage1_gate.py` | The gate: `GateConfig` (every setting, with the reason for its value), the rules and `replay_rules` |
| `stage1_pose_action.py` | The pose cue: features from 1 s of body points and the MLP ensemble |
| `stage1_crop_check.py` | The crop check (MobileCLIP2-S2 + logistic regression) |
| `stage1_bench.py` | Dataset loaders, the perception run and recall metrics |
| `stage1_report.py`, `yolo_poc_v3.ipynb` | Tables and figures for the earlier benchmark report (littering clips, Avenue, smoking) |
| `exp_harness.py` | Scores candidate rules on cached detections without re-running YOLO |
| `weights/` | Trained pose cue, reach verifier and crop-check weights (small `.npz` files) |
| `exp/icvl/` | ICVL scoring: `score.py` (splits and matching), `fpr_attrib.py` (one alert to one person) and one script per question asked, each writing a `.json` next to it |
| `exp/integration/` | Gate + VLM end to end: candidates for both halves in `vcal/`, scoring and calibration scripts |
| `exp/actions/` | Pose cue training and the action-model comparison (VideoMAE, X-CLIP, pose sequences) |
| `exp/newcam/`, `exp/misses/`, `exp/alerts/`, `exp/gt_review/` | New-camera tests (ABODA, IWDD), miss analysis, false-alert analysis, label scoping |
| `exp/*.py` | Earlier recall experiments on the littering, Avenue and smoking benchmarks |
| `tools/` | Demo videos, the report notebook builder, timing and model comparison |
| `demo/` | The demo video and figures |
| `yolo_poc.ipynb`, `yolo_poc_v2.ipynb`, `plswork.ipynb`, `new.py` | Early prototypes from before the gate was split into modules; kept for history |

Every script's docstring ends with a `Usage` line. Scripts are run from this folder, for example:

```bash
cd notebooks
../.venv/bin/python exp/icvl/caught_table.py even
```

## What is not in git

* `bench_cache/` (2.6 GB): cached detections, tracks and body points for every benchmark video. Every scoring script
  replays these, so results can be re-scored in seconds without YOLO.
* `../data/`: the datasets (ICVL 26 GB, littering clips, CUHK Avenue, smoking) and the OpenVINO model exports.
* YOLO `.pt` weights: the stock Ultralytics ones download themselves.
* Older demo videos: rebuilt by `tools/make_icvl_showcase.py`, `tools/make_showcase.py` and `tools/make_demo.py`.

## Tests

From the repository root: `python -m unittest tests.test_stage1`.
