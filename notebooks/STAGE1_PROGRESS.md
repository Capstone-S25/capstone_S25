# Stage 1 (YOLO gate): progress log

Read this first when resuming. It is updated at each milestone.

## Current task (in progress)
**Raise Stage 1 recall, keeping false alarms in check.** The user asked to experiment with methods; recall matters most.

* Harness: `notebooks/exp_harness.py`. It scores a candidate generator `method(result, dataset)` from cached detections
  (no YOLO re-run), about 5 s per evaluation. It uses only the tune half unless `STAGE1_HELDOUT=1`.
  * Tune / held-out splits: littering 230 train clips / 100 valid+test clips; Avenue odd / even test videos;
    normal footage odd / even Avenue training videos; smoking alternate camera-days.
  * Baseline (current rules, tune half): littering event 75.7 / crop 52.2; Avenue event 76.0 / crop 60.0 (reduction 43.2%);
    smoking event 43.1 / crop 17.2; false alarms on normal video 7.05 per minute.
* Five explorer agents were launched in parallel (workflow run `wf_9f8ac28a-0d7`, script under
  `~/.claude/projects/-Users-ianvarella-capstone-S25/.../workflows/scripts/stage1-recall-explore-wf_9f8ac28a-0d7.js`):
  `littering_act` (arm swing, extension, bend, crouch), `pose_anomaly` (one-class normal-pose model fitted on normal tune
  videos), `motion_anomaly` (speed, direction, thrown items, near-camera), `smoking_dwell` (pose-free dwell variants),
  and `crop_policy` (better crops for the right person). Each writes `<scratchpad>/exp/<name>.py` with `make_method(**params)` and `DEFAULTS`.
  Snapshots are copied to `notebooks/exp/explorer_work/`.
* **Explorers finished** (results in `notebooks/exp/explorer_results.json`; modules copied to `notebooks/exp/*.py`;
  combination scoring in `notebooks/exp/combo_eval.py` → `combo_tune.csv`; `exp_harness.dedup()` merges duplicate crops).
  Tune-half results (with dedup):
  base 52 littering crop / Avenue 76 event / smoking 43 event / 6.9 FA per min;
  **base + littering_act(A, extend_h=0.18) + motion_anomaly + crop_policy(trigger_wrist=None)**: littering crop 80,
  Avenue 100/100, smoking 44/24, FA 13.9; adding smoking_dwell(A_lean) gives smoking 64/41 at FA 15.5.
  Caveats: smoking_dwell fires on anyone still for 1 s, so its littering gain is a 3 s clip artefact (don't count it);
  dwell crops are larger (median 8% of frame); motion_anomaly "density" is a per-camera model fitted on normal footage;
  pose_anomaly has domain-shift risk (smoking reduction 55% → 34%), so it is not chosen.
  NEXT: adversarial review of the 4 chosen modules, then the held-out measurement, then integration.
* **Next steps:**
  1. Collect the explorer results. If the session was lost, the modules are in `notebooks/exp/explorer_work/`;
     re-score them with `exp_harness.score(combine(baseline, make_method(**DEFAULTS)))`.
  2. Pick and combine the winners under a false-alarm budget of about 10–15 per minute on normal video (tune half).
  3. Run an adversarial review: causality (no future frames), no ground-truth use, no tuning on held-out data.
  4. Measure **once** on the held-out half (`STAGE1_HELDOUT=1`).
  5. Move the chosen rules into `stage1_gate.py` (RuleEngine), add unit tests, re-run
     `notebooks/tools/make_report.py` → `yolo_poc_v3.ipynb`, and execute it with nbclient.

## Recall experiment: outcome (done)
* Adversarial review (`exp/review_results.json`): littering_act was REJECTED. A rate-matched random-person control does as well
  on the 3 s all-positive clips. smoking_dwell gains are first-seen / sequence-boundary artefacts. crop_policy is really
  "crop another random person". motion_anomaly's gain is essentially "person very close to the camera" (per-camera).
  A leak was fixed: the motion scene model now always fits on tune normal videos.
* Held-out results (`exp/heldout_eval.py` → `heldout_results.csv`, `heldout_control.json`):
  current rules: littering crop 55 (random control 42.4 ± 3.9, lift +12.6); loitering(Avenue) 90/55, reduction 41.6%; FA 8.4/min.
  With a "very tall box" near-camera rule: loitering 100/85, reduction 22.6%, FA 11.7. The motion module is the same
  (100/85, FA 12.3). Act cues: 63 littering crop, but lift falls to +7.0.
* DECISION: ship a simple per-camera `near_camera` rule (person box taller than the 99th percentile of person heights in that
  camera's normal footage). Drop the other modules; report them as tried and rejected, with the reasons.

## Status 2026-10-21 05:15: all four requests done
* `near_camera` rule is integrated in `stage1_gate.py` (`calibrate_near_camera()`; off unless calibrated per camera). 15 unit tests pass.
* The report `yolo_poc_v3.ipynb` was rewritten in the student's voice, with Avenue renamed to Loitering, held-out numbers and a
  "trying to push recall up" section. Generator: `tools/make_report.py` + `tools/report_text.json` (commentary text).
  Execute with nbclient.
* Demo videos: `demo/loitering_demo.mp4`, `demo/littering_demo.mp4` (`tools/make_demo.py`, `tools/render_demo.py`).
  They use tune-half footage and the final config (current rules + near_camera on the loitering camera).

## Status 2026-10-21 (later): re-send fix + in-view rule
* `near_camera_cooldown_s` changed 3 → 10. A person-level dedup is now inside RuleEngine (`person_dedup_s`=2.5: same entity,
  IoU ≥ 0.5 or containment ≥ 0.7). New `in_view` rule (`long_presence` events: `inview_s`, `inview_area_h`=4,
  `inview_repeat_s`=60, `calibrate_inview()` = max(60 s, 99th percentile of visit length in normal footage)). 18 tests pass.
* Final numbers (`exp/final_eval.py` → `exp/final_results.csv`), held-out half: loitering events 90 → 100%, right person 55 → 80%,
  video skipped 42 → 27%, false alarms 8.2 → 10.4 per minute. The in-view rule adds about 0.2 per minute and has no measurable
  recall effect yet (videos are 60 s or shorter). Report and demos regenerated (the loitering demo sends 18 candidates, VLM sees 82%).

## Status 2026-10-21 (evening): user feedback round 3 (in progress)
Requests: (1) try yolo26s/m plus their pose models; (2) remove near_camera; (3) don't flag a bag as unattended while it is carried,
and handle a thrown bag that lands; (4) stop reach_down firing on walkers; (5) catch "threw it and left";
(6) demo: say what each true anomaly is, remove the old-rules bar, drop "(new)".
Done so far:
* near_camera removed from DEFAULT_RULES (code kept, off).
* Bags: carrier = person whose box widened by 30% contains at least 50% of the bag → attended. New `object_appeared` cue:
  a bag first seen away from people after a 2 s warm-up, at a spot not seen before, that stays still, confidence ≥ 0.35,
  umbrellas excluded. Ownerless unattended: nobody near for unattended_s. Normal false alarms from object_appeared went 7 → 1;
  all 3 thrown-bag landings in the loitering tune videos are caught.
* reach_down: keypoint confidence 0.5 and people touching the frame edge are skipped. The 2-frame requirement and the
  bending requirement were tested and rejected (too much recall loss; see `exp/ablate_v2.py`, `exp/reach_variants.py`).
* Tune half now: littering 73.5/51.3, loitering 68/48 (skipped 52%), smoking 42/18, FA 5.3 per min. 22 tests pass.
* `GateConfig.device` was added. `tools/run_models.py s m` is running (MPS), writing `bench_cache/models/{s,m}/*.pkl`.
* The bag fixes were tightened: carrier = person box widened by 30%; object_appeared requires the object to stay still,
  confidence ≥ 0.35 and no umbrellas.
* Demo updated: `tools/demo_annotations.json` describes test/09's 6 GT events (shown on screen); old-rules bar removed;
  "(new)" dropped. `tools/make_demo.py [n|s|m]`. The n demo reaches 5/6 anomalies, 4 with the right person, and the VLM sees 53%.
* Held-out n with the final rules: littering 73/55 (lift +14.4), loitering 90/70 (skipped 45.7%), smoking 28/9.9, FA 6.8 per min.
  `exp/round3_changes.csv` compares this with the last version.
* The report generator was restructured: MODEL placeholder, `model_splits` from `tools/compare_models.py`, new sections ROUND3_TEXT
  and MODEL_TEXT; the text JSON still needs those and refreshed numbers.
NEXT: when `bench_cache/models/m/smoke.pkl` exists, run `tools/compare_models.py` (n s m), then
`tools/latency.py bench_cache/latency.json n s m` (CPU idle), pick the model, then `tools/make_demo.py <size>`,
update `tools/report_text.json`, build and execute the report.

## Status 2026-10-21 09:30: round 3 nearly done
* CPU latency (`bench_cache/latency.json`): n 56-63 ms, s 100-125 ms, m 182-232 ms per frame.
* Chosen model: **YOLO26s** (held-out: littering 72/56, loitering 80/60 with 59% skipped, smoking 37/13, FA 5.9 per min).
  Report and demo were regenerated with s (`tools/report_text.json` MODEL="s"; `tools/make_demo.py s`).
* The first m run hung after the Mac slept (MPS). It was restarted with caffeinate for avenue+smoke (`tools/run_models.py m`).
  When it is done: `tools/compare_models.py n s m`, update MODEL_TEXT in `tools/report_text.json` if m changes the picture,
  then rebuild and execute the report.

## Status 2026-10-21 ~10:00: round 3 done
* m finished. Held-out results: littering 71/57 (lift +20.6), loitering 80/60, smoking 43/14, FA 6.5, but only 4-5 fps on CPU.
  Kept **s**. `exp/model_comparison.csv` has n/s/m. Report rebuilt and executed. 22 tests pass.

## 2026-10-27: action models + label re-scoping (in progress, branch feat/yolo-stage1)
Requests: (1) continue experimenting, mainly action detection models (the professor suggested them; DeepStream needs NVIDIA,
so it's out for CPU); (2) the user disagrees with some Avenue GT events (e.g. a man with a bag walking into frame).
Plan: write the scope rules first (`exp/gt_review/SCOPE_RULES.md`); render every Avenue test GT event as a sheet
(`exp/gt_review/`); have agents describe and tag each event in/out of scope blind to Stage 1 results; then the user/group
confirms. Report recall on all labels and on in-scope labels.
Action experiments (`exp/actions/`): X-CLIP zero-shot on person tubes, a skeleton-sequence classifier on the pose we
already have, and VideoMAE/Kinetics; judged by crop recall vs the random control, FA on normal video and CPU latency.

* Label review done (`exp/gt_review/events.csv`, raw `review_raw.json`): 45 events → 29 agreed in, 8 agreed out (all
  "walking normally toward or close to camera", incl. the backpack man), 4 unsure, 4 disagree. The `group_decision`
  column is empty for the group to fill in. `exp/gt_review/scoped.py` filters GT; `scoped_recall.csv`:
  held-out (s model) all labels 80/60, lenient 78/67, strict 80/67; tune strict 93/86.
* Action experiments: the first run died when the internet dropped; resumed the workflow (run wf_f393197a-a97).

* Action experiments done (`exp/actions/action_results.json`; modules xclip.py, pose_seq.py, videomae.py; the cache is
  gitignored). Tune half, s model, baseline lit 55.7 crop (lift +10.9), FA 5.09:
  - pose_seq mlp_sum@FA1.0 as an add-on: lit 97/87, avenue 84/64, smoking 56.6/22.6, FA 6.07; fair lift alone +19.0 vs
    reach_down +9.9; 0.24 ms/call. Caveats: partly a size prior (within-clip AUC 0.747 vs taller-box 0.707), and the
    threshold is camera-specific (13 bystander candidates/min on the littering camera vs reach_down 16).
  - videomae throw>=0.10 add-on: lit crop 79.6, lift +17.2, FA 5.68, but 577 ms/call (not real-time). Smoking fails (AUROC 0.49).
  - xclip litter_negs m=0.2: alone lift +16.7, add-on FA 10 (2x); 233 ms/call (about 2 cores). Avenue gains are a rate effect.
  - The stock random control can be gamed by firing on everyone (it samples with replacement); the stricter control is
    in pose_seq.diagnose().
  NEXT: adversarial review of pose_seq, then held-out once, then integrate as an optional rule.

* pose_seq review (`exp/actions/pose_seq_review.json`, both reviewers "ok with caveats"): the single-seed 87% was lucky
  (79.6–87.0 across seeds); held-out littering clips share source videos with tune clips; the features identify the
  camera, so the threshold must be calibrated per camera.
* Frozen and measured once on held-out (2026-10-27, `exp/actions/pose_seq_final.py freeze|heldout|export`): a 5-seed
  MLP ensemble, with one threshold (0.833) set on the averaged out-of-fold tune normal scores for 1 false alarm/min.
  - Held-out, current rules → rules + pose cue: littering event 72→97, crop 56→86; loitering 80/60→80/65 (strict labels
    80/66.7, unchanged); smoking 36.8/12.6→43.7/15.7; normal false alarms 5.87→6.45/min.
  - The cue alone beats "tallest person at the same moments" by +20 crop points (64 vs 44) on held-out. Per action
    (rules→with cue): dropping 35→80, throwing 30→80, tossing 35→80, putting-down 85→90, dumping 95→100.
  - Results in `exp/actions/pose_seq_final_{tune,held-out}.json`.
* Integrated as the optional rule `"pose_action"` (not in DEFAULT_RULES): `stage1_pose_action.py` holds the window
  features and a numpy ensemble (weights `notebooks/weights/pose_action_mlp.npz`, 144 KB, no torch at run time);
  `GateConfig.pose_action_thr/pose_action_cooldown_s`. Replaying held-out through the gate reproduces the numbers
  exactly; it adds ~0.14 ms/frame. `pose_seq.py` now imports the features from the shared module. 24 tests pass.
  Use `GateConfig(rules=DEFAULT_RULES + ("pose_action",))`. Note that tune numbers with the final model are in-sample.

## 2026-10-27/28: what limits recall, and unseen cameras (user: "happy for you to try these out")
* Miss analysis, tune half (`exp/misses/miss_analysis.py` → `misses_tune.csv`): YOLO missed the person in ~1/684 events on
  the original datasets. Littering misses are mostly wrong-person crops (32/39; the actor's pose score never clears the
  threshold, a bystander's does). Loitering misses: 9/10 are events the label review tagged out/unsure/disputed.
  Smoking is limited by the data (~1 fps, median 15.6 s sequences; smoker visible >=20 s in a small minority of episodes).
* Rejected on tune: wider "context" crops (gain is just the bigger box; control gains more), shorter dwell on the Avenue
  camera (10 s: smoking episode recall 37→41%, Avenue normal FA 5→16/min). Dwell must be per camera
  (`exp/misses/sweep_*.csv`). Planned held-out check: loiter_s 8 on the smoking camera only.
* Unseen cameras (`exp/newcam/`; caches `bench_cache/models/s/{caviar,aboda,meva}*.pkl`):
  - MEVA (11 clips, 55 min, annotations downloaded to `data/meva_annotations` from gitlab.kitware.com): right person for
    only ~21–23% of 52 put-downs; pose_action catches 1/52 (does not generalise). Labels are aligned; YOLO at 640
    misses dense/seated/small people (0/22 actors <100 px). Raw YOLO on a crowd frame: 0 persons >=0.4 at 640, 44 >=0.1 at 1280.
  - CAVIAR (384x288): YOLO never sees the 4 left bags (9–22 px), at 640 or 1280; 1280 even finds fewer people on this
    low-res video. Old background rule fires 0 times (still-timer resets on noisy video). New persistence rule
    `static_v2` (hold 8 s, >=60% of frames, novelty>=30, burst guard <=4 spots/2 s) gets 3/4 CAVIAR bags (in-sample, n=4)
    at ~1/min on CAVIAR normal and 0/min on Avenue normal.
  - ABODA: labelled by 2 agent labellers + reconciler (`aboda_labels.json`, 11 events + 1 ambiguous in video11).
    YOLO bag rules: right crop in 4/11; YOLO sees the bag at all in 4/10 videos. static_v2 + burst guard: ~0.4/min,
    catches 1. ABODA is now development data for static_v2 (its failures were used to design the burst guard).
  - Running: MEVA perception variants (conf25, img1280) and COCO-vs-YOLOE (open-vocabulary) bag detection on ABODA
    (`object_detectors.py`). Ultralytics auto-installed `clip` and `ftfy` into .venv for YOLOE.
  - Adversarial review of these findings: workflow wf_b583d2b1-846 (results summarised below).

### Corrections after the review + chance baselines (2026-10-28)
* `exp/chance.py` → `exp/chance_results.json`: every dataset scored against a rate-matched baseline (same number of
  candidates, random tracked person at the same moments / uniform times), with the lenient box_hit and a strict test
  (>=50% GT covered, crop <= 16x GT area). Held-out/unseen, strict, rules+pose vs same-times chance:
  littering 86 vs ~60 (+26), loitering 50 vs ~21 (+29), ABODA bags 45 vs 9 (+36), MEVA put-down 19 vs ~13 (+6),
  **smoking 22 vs 23 (no better than chance)**, CAVIAR tiny bags 0 vs 0.
* MEVA "event %" (~80%) is pure chance at ~19 candidates/min — never quote it. pose_action on MEVA = chance, and its
  candidates take dedup slots from reach_down (23→21% crop).
* static_v2 (background persistence) was inflated by the lenient crop test (huge padded crops): tight crop on only 1/4
  CAVIAR bags, and no genuine ABODA catch. Background subtraction still does not deliver.
* Smoking: the real limit is track fragmentation at the dataset's irregular ~1 fps (gaps > stitch window); deep_dive's
  "testable" used raw ids (19) vs stitched (38). Longer stitch windows / shorter dwell (`exp/fixes.py stitch`) give at
  most +1–2 points over chance on tune; held-out dwell rule 7% vs 6% chance. Conclusion: this dataset cannot test a
  dwell-based smoking rule.
* `dedup_per_rule` (new GateConfig flag, default False): more candidates, no recall gain, less above chance → keep off.
* YOLOE-26s (open-vocabulary, prompted "bag, backpack, box, …") once per second for objects (`exp/newcam/yoloe_items.py`):
  ABODA right crop 4→5/11, bag alarms 0.48→0.67/min, Avenue normal bag alarms 0.59→0. Once YOLOE sees a bag the rules
  catch it 5/5; the 6 misses are small dark bags no detector found (even at 1280). MEVA perception variants
  (`meva__{conf25,img1280,img1280_conf25}.pkl`): imgsz 1280 raises tracked persons/frame 4.4→5.9 and put-down strict
  crop recall with pose_action 19→31% (chance ~13–18%, n=52, ±2 pts per event); pick-ups stay at chance; person_conf
  0.25 alone does nothing. 1280 costs ~3x CPU, so only worth it per camera (wide 1080p views with small people).
  Note: chance.py seeding was non-deterministic (python string hash salt) until fixed with crc32. Detector test
  (`object_detectors_aboda.json`): bag found in 50% of frames (YOLOE@640) vs 18% (COCO@640), ~2x CPU.

## 2026-11-03: IWDD (MIVIA illegal waste dumping) as a new littering test (user: "do your thing")
* Data: all of MIVIA-IWDD-500 now in `data/training_sources_20261015/mivia_iwdd` (train 400 = 200 dumping / 200
  normal, test 100 = 50/50; user downloaded the train split by hand, every file matched its pinned SHA-256, plus
  vid0325/vid0346 that the original acquisition lacked). Labels are weak: one onset second per positive, static/dynamic,
  day/night, no boxes. Mixed cameras 240p–4K, much of it news/municipal footage with pixelated people.
* Code (`exp/newcam/`): `loaders.iwdd_labels/iwdd_sequences` (~10 fps like the littering clips), `run_perception.py
  iwdd_test iwdd_train`, `iwdd_eval.py` (hit = candidate in [onset-1, onset+10 s]; chance = same number of candidates at
  uniform random times), `iwdd_diag.py`. Tuning/diagnosis on train only; test scored once with the UNTUNED gate.
* **Held-out test (untuned current gate):** rules+pose 28/50 = 56% (95% CI 42–70) vs chance 42%; rules 52 vs 37.
  Train is harder: 37 vs 32. Box on the dumper/dumped waste in ~26/28 test hit videos (visual check), but almost all
  scenes contain one person, so crop correctness is nearly free here. Normal IWDD videos: 70–76% get >=1 candidate
  (~15–17/min) — these are hard negatives (crowds, a fall, someone carrying a box, the same dump-site cameras).
* Why misses (train): 68/200 no tracked person near the onset (pixelated, behind/inside a car; 1280 input + conf 0.1
  recovers few: 9→13 of 70 at >=0.4), 58/200 person tracked but no rule fires (frame-edge people skipped by reach_down,
  unloading a boot at hip height, throwing), 51/200 clips end <3 s after the onset. Test is easier (8/50 no person,
  6/50 short).
* Rejected on train (none run on test):
  - pose_action retrained with IWDD weak labels (`iwdd_pose.py`, "all" and one MIL round, same 1/min Avenue budget):
    hit 38→52% but chance 33→48%, lead +4.8→+3.8; window AUC 0.71 but within-video 0.57–0.59, video-level 0.54;
    littering tune event 97→93–94%.
  - YOLOE waste objects once a second (`iwdd_objects.py eval`, 17 prompts): object rules alone 8.0% vs chance 8.1%
    (dump sites are already full of bags).
  - Background-subtraction static_v2 (`iwdd_objects.py bg`): alone below chance, normal-video candidates 14→35/min
    (handheld/edited footage); with burst guard almost silent.
  - "Person at a stopped vehicle" cue (`iwdd_vehicle.py`, COCO vehicles once a second): alone = chance at every dwell /
    margin (people next to parked cars are just as common in normal videos).
  - reach_down sweep (`iwdd_tune.py` → `iwdd_tune_train.csv`): looser hand-height margins raise Avenue normal FA from
    5 to 100–128/min with no lead over chance; reach_skip_border=False / kp_conf 0.3 add ~1 point (2 videos), noise.
* Simple timers for comparison (whole frame; same 11 s hit window): a frame every 10 s → 65/68% (train/test) at
  7.2/min; a frame every 10 s **only while someone is tracked** → 46/62% at 5–6/min on IWDD normals and 7/min on
  Avenue normal (gate: 37/56% at 15–17/min on IWDD normals, ~5–6/min Avenue); every 5 s while tracked → 70/78% at
  9–11/min (Avenue 12.5/min). Caveats: whole frames (the VLM must find a small dumper), and IWDD clips are short and
  activity-dense; on an empty dump site a presence timer sends nothing.
* Conclusion: the current gate beats chance on IWDD held-out, modestly. The remaining misses are mostly invisible
  dumpers and short clips — not something rules or the pose cue can fix. Recommendation (team decision, VLM budget):
  an opt-in "presence heartbeat" (a frame every ~10 s while people are in view) as a recall floor on dump-site
  cameras. No GateConfig default changed.

## New requests (2026-10-21, after the session-limit break)
1. Finish the recall work: re-run the adversarial review (`wf_0880d8a2-aa0` died at the session limit before running),
   then the held-out measurement, then integration.
2. Rename "Avenue" to "Loitering" everywhere in the report (it lives in `data/loitering`).
3. Rewrite the report so it reads like the student wrote it: first person, plainer, less polished.
4. Make something substantial and visual for the professor's walk-around tomorrow, that he can point at on screen
   (plan: annotated demo video(s) plus an HTML dashboard of the review queue and timeline).

## Done so far
* `stage1_gate.py` (perception and rules), `stage1_bench.py` (loaders and metrics), `stage1_report.py` (report figures),
  `tests/test_stage1.py` (14 tests, `python -m unittest tests.test_stage1`).
* `yolo_poc_v3.ipynb` is the report notebook, generated by `notebooks/tools/make_report.py` and executed with outputs saved.
* Littering dataset rebuilt: `scripts/rebuild_littering_clips.py` → `data/littering/{train,valid,test}/<clip>/`
  (330 clips, 66 per action, 46/10/10 per action). The raw export `Littering/` is kept as the input.
* Avenue frame labels: `data/loitering/avenue_gt/`. The "loitering" folder is really CUHK Avenue.
* Default rules: loiter (20 s, re-sent every 30 s), speed (1.2 body heights per second), reach_down (margin -0.15), unattended.
  Off by default: `smoking` (pose hand-to-face; weak, user asked not to mention it) and `litter` (background subtraction).
* Background subtraction for littering was tested with a short-clip variant: 0 of 37 detections were real objects
  (two independent reviewers each). Conclusion: don't use it for littering on this data; keep it for unattended items
  or motion gating. Tool: `notebooks/tools/spawn_shortclip_experiment.py`.
* Caches: `notebooks/bench_cache/` (`avenue.pkl`, `smoke.pkl`, `litter_clips.pkl` = held-out, `litter_clips_train.pkl` = tune,
  `latency.json`). `notebooks/tools/run_perception.py` and `notebooks/tools/latency.py` regenerate them.

## User preferences learned
* The report notebook should be report-style and focused, for the professor.
* Smoking: use "person stays in an area"; the VLM does the reasoning. Don't present the pose hand-to-face rule.
* Recall comes first; false positives second.

## 2026-11-03: frame rate for the data pipeline (feat/data-pipeline selects 2 fps)

`exp/newcam/fps_check.py` re-runs YOLO26s perception at lower rates (tracking depends on fps, so caches are
not subsampled) and scores the unchanged rules + pose on held-out data (`exp/newcam/fps_check.json`):

| rules + pose | ~10 fps (benchmark) | 5 fps | 2 fps |
|---|---|---|---|
| littering held-out event / crop % | 97 / 86 | 96 / 86 | 87 / 69 |
| Avenue held-out event / crop % | 80 / 65 | 75 / 45 | 82 / 47 |
| Avenue normal false alarms / min | 6.5 | 6.6 | 8.8 |
| IWDD test hit@10s vs chance % | 56 vs 42 | 38 vs 29 | 34 vs 25 |

* 2 fps fragments tracks: wrong-person crops and more false alarms. 5 fps recovers littering and the false-alarm
  rate. Avenue crop stays lower (small n, 17 held-out events) and IWDD candidates drop (17 -> 11/min) with a
  smaller lead over chance, so some rules still depend on frame counts; not yet investigated.
* CPU cost at 5 fps: ~100 ms/frame (YOLO26s + pose, p95 101 ms) on the sample clip, about half a core.
* Recommendation: run the pipeline with `--target-fps 5` when Stage 1 is on. The VLM views are sampled from the
  writer's own 2 fps buffer, so the VLM input does not change.
* Integration lives on `feat/stage1-integration` (data-pipeline/data_pipeline/detection/).

### Why 5 fps differs (diagnosed on tune / train data only; no gate change)

Rule by rule at 12.5 vs 5 fps (rule alone; littering tune clips, Avenue tune normals, IWDD train):

* **No frame-count bugs in the rules.** Rules use seconds; a half-frame slack on the speed window changed
  nothing at 5 fps (12 vs 12 Avenue events) and was dropped. `test_fast_motion_at_5_fps` checks the rule fires.
* **Fast motion** loses most (Avenue tune 56/40 -> 12/4 event/crop %): at 0.2 s spacing a runner's boxes stop
  overlapping, ByteTrack starts a new id and the speed history resets. A net-displacement speed did not help
  (still 0.2 s jumps break the track); rejected. This is a tracker limit at low fps.
* **Loitering false alarms double** on Avenue tune normals (2.9 -> 5.9/min): fewer frames give the tracker fewer
  chances to drop a standing person, so more people really reach 20 s. Thinning points to a fixed rate did
  not change it (rejected). Treat as a per-camera dwell setting at the deployment fps. Held-out overall false
  alarms were unchanged (6.5 vs 6.6/min).
* **pose_action** keeps littering recall (tune alone 95 -> 94 %) but its features shift: the 1 s window is
  resampled to 8 points by sample-and-hold, so at 5 fps every other velocity is 0 and the next doubled.
  Avenue tune normal 0 -> 1.2/min (0 is in-sample). Fix if needed: retrain on 10 + 5 fps windows.
* **IWDD train**: hit@10s 37.0 vs chance 32.3 (full) and 33.5 vs 27.4 (5 fps); negatives 15.2 -> 10.1/min.
  The lead over chance holds (+4.7 -> +6.1); 5 fps is a lower candidate rate (fewer fast_motion and reach_down
  frames), not a loss of signal. The held-out drop (56 vs 42 -> 38 vs 29) is the same effect plus n = 50 noise.
* **pose_action retrained on 10 + 5 fps windows** (`exp/actions/pose_mixed_fps.py`; tune OOF and one held-out
  run in `pose_mixed_fps_{tune,heldout}.json`). Held-out littering event/crop, cue alone and rules +:
  at 5 fps current 82/67 and 96/86 vs mixed 79/64 and 93/80; at 10 fps current 81/64 and 97/86 vs mixed
  85/68 and 98/85. **Rejected**: the current cue already holds at 5 fps on held-out, so the feature shift
  costs no littering recall; weights not exported, `weights/pose_action_mlp.npz` unchanged.
* Demo material: `data-pipeline/stage1_offline.py --target-fps 5` on IWDD test vid0075/0332/0359/0395 (all caught
  within ~1 s of the onset), vid0208 (54 s normal, no candidates) and the pipeline's sample clip;
  `tools/demo_sheets.py` renders one slide image per candidate (output in gitignored `vlm_handoff_demo/`).

## 2026-11-03: abandoned bags (ABODA 6/11 caught) — can a different detector see the missed bags?
* Bottleneck is detection, not rules: once a detector sees the bag the rules catch it (YOLOE 5/5).
* YOLOv8s trained on Open Images V7 (601 classes; bag-like: Backpack, Box, Briefcase, Handbag, Luggage and bags,
  Plastic bag, Suitcase) added to `exp/newcam/object_detectors.py`: **0% of unattended-bag frames at 640, 0.7% at
  1280** (COCO yolo26s 29%, YOLOE 50%). It does not even find the people in these 640x480 frames. Rejected.
* MEVA annotations in hand box 1 bag and 45 "other" objects over 11 clips: too few to fine-tune on.
* Remaining route: fine-tune YOLO26s on CCTV bag boxes from other sources (candidates: Roboflow Universe CC BY 4.0
  abandoned-luggage sets, PETS 2006 train-station footage auto-labelled with YOLOE then checked), keeping ABODA test-only.

## 2026-11-04: training round (presentation moved to Friday)
* **Littering, source-grouped CV** (`exp/actions/pose_grouped_cv.py` → `pose_grouped_cv.json`): every held-out clip's
  source video also had train clips. Re-training the pose cue with 5 folds grouped by (action, source video) over all
  330 clips: rules + cue 96.4 / 90.0 % event / crop (frozen cue on the old held-out: 97 / 86; grouped CV on those same
  100 clips: 97 / 92). The shared sources did not inflate the result.
* **IWDD dumper labels** (`exp/newcam/iwdd_actor_sheets.py`, `iwdd_actor_labels.py` → `iwdd_actors_train.json`):
  157 train videos with someone tracked near the onset reviewed by eye; 105 labelled, 32 ambiguous, 20 dumper never
  tracked. Test videos not reviewed.
* **Pose cue retrained with dumper labels** (`exp/newcam/iwdd_pose_actor.py`; train OOF in `iwdd_pose_actor_train.json`):
  dumper-vs-bystander AUC 0.59 → 0.78, first in-window candidate on the dumper 17/18 → 49/57 videos, but IWDD hit@10s
  vs chance 38.0 vs 33.2 → 46.5 vs 42.7 (lead +4.8 → +3.8) and littering tune crop 83 → 77. **Rejected as default**
  (no gain over chance, hurts littering); test split untouched. Model kept in exp/actions/cache/iwdd_pose_actor.pkl.
* **Bag fine-tune data**: Roboflow luggage_and_person v2 and luggage_bag v3 (CC BY 4.0) in `data/roboflow_bags/`.
  `exp/newcam/bag_data_dedup.py`: 456 images are ABODA frames (all of luggage_bag v3, 295 of the other) → excluded.
  `exp/newcam/bag_finetune.py build`: 1974 train + 439 val Roboflow images, + coco128 and 300 tune-only CCTV frames
  as replay, 80 COCO classes kept (luggage → the COCO bag class the original model sees, else suitcase; unlabelled
  objects pseudo-labelled at conf 0.4).
* **Bag fine-tune** (`exp/newcam/bag_finetune.py`, weights `weights/yolo26s_bags.pt`, gitignored *.pt): early-stopped at
  11 epochs, val mAP50 0.83. ABODA bag-found 18 → 27 % of frames (conf 0.2), clutter 0.21 → 0.07 per frame
  (`bag_finetune_aboda.json`). Stage 1 end to end (`bag_finetune_gate.json`): ABODA 8/11 → 8/11, littering held-out
  97/86 → 99/86, Avenue 80/65 → 80/65, normal FA 6.45 → 5.67/min. **Not adopted**: the Roboflow data turned out to be
  mostly phone photos (1,221 of 2,413 images have no bag; only 97 have a CCTV-sized one), so the gain is not a bag gain.
* **ABODA misses checked by eye**: labels are right. Videos 2–4 (one outdoor camera, dark bag on patterned paving,
  23–52 px) are found by no detector (COCO, YOLOE, Open Images, fine-tuned). Videos 7–9 are seen by COCO only as
  "dining table" / low-confidence suitcase (0.16–0.26, under our 0.25 track / 0.35 appear thresholds).
* **Bag metric**: "bags caught" = a candidate whose crop covers the bag from 2 s before put-down on (8/11 for the
  current gate; 4 via bag rules, 4 via reach_down / pose / fast_motion on the person leaving it). The 6/11 in
  chance_results.json is the stricter "crop covers the bag while it is unattended" test. Quote 8/11 with the split.
* **YOLOE item pass in the gate** (new `GateConfig.item_detector` / `item_prompts` / `item_period_s` / `item_conf` /
  `item_detector_min_passes`, off by default; `exp/newcam/yoloe_gate.py` → `yoloe_gate.json`): ABODA 8/11 → 8/11
  (bag rules 4 → 5, adds video9), normal bag alarms 0.59 → 0/min, normal FA 5.77 → 4.99/min, but Avenue held-out crop
  65 → 55 % (thrown objects no longer seen as COCO items). Keep as a per-camera option for bag-heavy scenes, not default.
* **ABODA misses diagnosed** (the bag-leaver is tracked with pose in all three): video3 reach_down fired on the right
  person 0.5 s before the put-down but the person-box crop missed the bag; videos 2 and 4 the reach fired 3.4 s / 3 s
  earlier on the same person and the 5 s cooldown blocked the put-down itself.
* **reach_down options** (`GateConfig.reach_hand_pad_h`, `reach_refire_move_h`, default off; `exp/newcam/reach_options.py`):
  on tune data every setting is neutral (littering 97.8/96.5, Avenue 68/60, normal FA 5.09–5.29), so the most
  conservative {hand pad 0.25 h, re-fire after moving 1.0 h} was fixed a priori and scored once: **ABODA 8/11 → 10/11**
  (video2, video3 now caught; video4 still missed), held-out littering 97/86 → 97/85, Avenue 80/65 → 80/65, normal FA
  6.45 → 6.26/min. Caveat: the fix was motivated by looking at these ABODA misses, so ABODA is no longer a clean test
  for it; the tune data only shows it costs nothing.
* **Bigger YOLO on ABODA bags** (`exp/newcam/yolo_sizes_aboda.json`, same frames as object_detectors.py, conf 0.2):
  s 18 % / m 30 % / l 31 % of unattended-bag frames, CPU 49 / 104 / 131 ms. The gain is only on bags s already sees
  (videos 1, 8, 10); videos 2–6 and 9 stay at 0 for every size, m loses video 7. Not worth 2x CPU.
* **CORRECTION (bag scoring)**: the ABODA "bags caught" test above had no end time, so a reach_down when the owner
  came back to PICK UP the bag counted as a catch (both video6 bags: candidates at 134.3 s / 206.3 s, removed at
  133 s / 207 s). Now a candidate must fall in [put-down − 2 s, removal − 2 s] (`reach_options.py`, `yoloe_gate.py`).
  Corrected: current gate **6/11**, with the reach options **8/11** (adds video2, video3; checked by eye: a man
  bending with the bag in hand), YOLOE pass 6/11. Still missed: video4 (cooldown), both video6 bags (dark night
  corridor, no rule fires at the put-down). video5 is borderline (owner walking away 3 s later, crop just touches
  the bag). The 8/11 and 10/11 quoted earlier today were wrong; comparisons between arms are unaffected.
* **reach_down on walkers** (user spotted ABODA video2: a man walking with a heavy bag hanging from his hand fires
  "reached to the ground" — from a steep camera a hanging arm reaches knee height in the image). New
  `GateConfig.reach_max_speed_h` (skip people faster than this many body heights/s over `reach_speed_window_s`=1 s).
  Chosen on tune / train only: 0.45 keeps littering tune 97.8/96.5 and Avenue tune 68/60 (0.3 drops Avenue crop to 56),
  cuts reach_down false alarms on IWDD train normal videos 5.26 → 4.34/min (videos with any: 43 → 36.5 %) and raises
  the IWDD train lead over chance +5.0 → +6.8. ABODA unchanged at 8/11; the walking fire at 65.6 s in video2 is gone,
  the real put-down bend at 69.0 s stays. Off by default like the other reach options.
* **Reach options now ON by default** (reach_hand_pad_h 0.25, reach_refire_move_h 1.0, reach_max_speed_h 0.45).
  Scored once (`exp/final_defaults.py` → `final_defaults.json`), old → new: littering held-out 97/86 → 97/85, loitering
  80/65 → 80/65, smoking 43.7/15.4 → 43.7/15.4, normal FA 6.45 → 6.26/min, IWDD test hit@10s vs chance 56.0 vs 42.1 →
  52.0 vs 38.6 (same +13 lead, fewer candidates), ABODA bags 6/11 → 8/11. Demo videos re-rendered with the defaults
  (`tools/make_showcase.py`): ABODA video2 moved from the misses reel to the works reel ("caught through the person"),
  video6 (night corridor) is now the bag miss; title cards say "footage not used for tuning".
* **reach_down on people standing with arms down** (user: ABODA video6, ID 20 at 102.8 s). Keypoints are right; the
  rule's absolute test (lowest wrist within 0.15 h above the knee) is met by hanging arms from a steep camera
  (wrists ~0.60–0.65 h, knees ~0.72–0.80 h). Tune / train sweeps (`exp/newcam/reach_margin.py`):
  - tighter absolute margin (`reach_margin_tune.json`): −0.10 halves IWDD train reach alarms (4.3 → 2.2/min) but the
    IWDD train lead over chance falls +6.8 → +2.5 and littering reach-alone 65/35 → 54/22. **Rejected** (recall first).
  - relative drop vs the person's own 5 s median (`reach_relative_tune.json`): 0.05 → reach alarms 4.3 → 3.9/min, lead
    +6.8 → +6.0; but it does not remove the reported fire (new track, < 1 s history → fallback) and ABODA bags fall
    8 → 7 (people who arrive carrying a bag already have a low hand). **Rejected**; code not kept.
  Accept these standing-still reach alarms as VLM work; recall matters more.
* **fast_motion off by default** (`exp/newcam/fast_motion_sweep.py` → `fast_motion_tune.json`, tune / train only):
  alone it is at chance for IWDD dumping (9.0 vs 8.4 %); off cuts IWDD train normal alerts 14.9 → 8.3/min and the
  lead over chance goes +6.8 → +7.5; littering tune unchanged. Avenue tune drops 68/60 → 52/40, but the label review
  shows Avenue's in-scope incidents are only RUNNING (13) and THROWING (15), no loitering; on throwing only, off loses
  nothing (tune 6/6, held-out 10/10). Running is not one of the four project events. `DEFAULT_RULES` no longer has
  "speed" (rule kept). IWDD test confirms: 52.0 vs 38.6 → 44.0 vs 31.2 (lead +13.4 → +12.8) with normal-video
  alerts 16.5 → 9.5/min (videos with any 72 → 58 %).
* **CORRECTION**: "loitering 80 / 65 %" (Avenue) in earlier reports measured running + throwing, not loitering. No
  dataset we have labels loitering, so loitering is unmeasured. Avenue throwing is now reported under littering.
* `exp/counts_precision.py` (raw counts, recall, precision; Avenue throwing-only) and `exp/per_rule_fa.py` re-run with
  the new defaults: walkway alerts 6.26 → 4.89/min, dump sites 16.5 → 9.45/min. Demo: Avenue test/04 (two runners)
  replaced by test/12 (throwing).
* **Alert budget** (`exp/newcam/alert_budget.py` → `alert_budget_tune.json`, tune / train only): `loiter_repeat_s`
  30 → 60 s cuts walkway alerts 4.11 → 3.72/min with no loss anywhere (120 s same) → **new default 60 s**. A longer
  one-alert-per-person window (`person_dedup_s` 5–30 s) cuts dump-site alerts up to 8.3 → 5.4/min but loses real
  catches (IWDD lead +7.5 → +4.5, smoking 55 → 43 %, Avenue throw crop 100 → 83 %): the alert that matters often
  comes seconds after an earlier one on the same person (approach, then the act). **Rejected.** Most extra alerts are
  different people doing ordinary things; reducing them needs a learned re-ranker, not spacing.

## 2026-11-04 (evening): why the false alerts happen (tune / train only)
* `exp/alerts/alert_table.py` → `alerts_tune.pkl`: every alert of the current defaults on littering tune clips, Avenue
  tune throwing incidents + normals, IWDD train (normal videos; dumping videos with a labelled dumper), labelled real /
  false with facts about the person. 1,353 alerts; dump-site normal videos alone give 752 false (reach_down 422, pose
  167, stays-put 80, bag left alone 64). Dump-site false vs real reach_down: 7 vs 1 people in frame, nearest person
  0.7 vs 1.7 body heights; 10 of 112 videos hold 35 % of the false alerts.
* Looked at samples by eye: (1) riders on motorbikes / scooters (seated: hands at knee height) — 148 of 752;
  (2) walkers in dense street crowds, overlapping people (keypoints mixed between people: 36 % of false non-rider
  reaches vs 3 % of real); (3) people seated / crouched doing ordinary things, bin workers; (4) the bag rules firing on
  a fixed scene object that adopts a passer-by as "owner"; (5) static non-people detected as people ("stays put").
* Fixes tried (`exp/alerts/sweep.py`, options default off):
  - `reach_skip_overlap` 0.5: dump-site alerts 8.27 → 7.66/min (−7 %), lead +7.6 → +7.3, littering same. Candidate.
  - `item_scenery`: a bag first seen with nobody near never adopts an owner: 8.27 → 7.93/min, lead +7.8, ABODA 8/11
    unchanged (false bag-left-alone 9 → 6). Candidate.
  - `reach_skip_riders` (all two-wheelers): 8.27 → 6.71/min (−19 %) but loses 3 real dumping videos (two cyclists
    stopping to dump bags; one with bin bags mis-detected as motorbikes). Re-running with motorcycle-only
    (`reach_rider_classes` = (3,)), detection classes now cached (`exp/alerts/two_wheelers.py`).
  - Rejected: pose_action_thr 0.88–0.95 (littering crop 96.5 → 86.5 at 0.92 for −11 % alerts); a "too still to be a
    person" filter on stays-put (real smokers sway as little as static fakes, p10 0.006 vs 0.005); crowd suppression
    (littering does happen in crowds).
* `reach_skip_seated_s` 5 s (skip people seated — knee within 0.12 h of the hip — for 5 s; a dumper crouches briefly):
  dump-site alerts 8.27 → 7.87/min, lead unchanged; 1–3 s lose dumping videos.
* Motorcycle-only riders (`reach_rider_classes` = (3,), detection classes re-cached): 8.27 → 6.86/min, loses 1 dumping
  video (all two-wheelers lost 3: two cyclists stopping to dump, one bin-bag mis-detection). Live gate checked: the
  detector also returns motorcycles, kept out of the bag items.
* **New defaults**: reach_skip_riders (motorcycles), reach_skip_overlap 0.5, reach_skip_seated_s 5, item_scenery. Tune /
  train: dump-site alerts 8.27 → 5.87/min (−29 %), lead +7.6 → +7.3, littering / throws / smoking unchanged.
  Held-out once (`counts_precision.json`, `per_rule_fa.json`; two-wheeler boxes attached via `exp/alerts/cached.py`):
  IWDD test 44.0 vs 31.2 → 44.0 vs 31.2 (no loss) with normal-video alerts 9.39 → 6.30/min (−33 %); dumping alerts
  230 → 168, precision 10 → 14 %; walkway 4.89 → 4.50/min; littering 97/85, throws 10/10, bags 8/11 unchanged;
  smoking 159 → 150 of 364 (from loiter_repeat 30 → 60 s).
* Pose-cue false alerts at dump sites (167) are mostly real throwing-like motions (catch in a car park, beach clean-up,
  workers loading a truck): only the VLM can tell a thrown ball from thrown litter. Left alone.
* **Per-event precision now counts only alerts ROUTED to that event** (`exp/counts_precision.py`): littering 49 %,
  walkway throws 45 % (was 24), dumping 15 % (14), unattended items 15 % (11; item rules alone 62 % but 4/11 caught),
  smoking 25 % at 10 % recall (13 right person of 364).
* **CORRECTION (smoking)**: smoking's own rule (stays put) catches only 10 % of episodes; the 41–44 % "caught" before
  was mostly reach_down firing on smokers at ~1 fps. The smoking set is 169 sequences, median 16 s at 0.9 fps —
  shorter than the 20 s dwell — so smoking cannot be measured or tuned on it.
* **Routing fix** (data-pipeline `TASKS_BY_EVENT`): reach_down and pose_action now also ask the VLM about unattended
  items, since half the caught bags (4/8) come from the person putting them down while the bag itself is never detected.
* **Counting decision (user)**: precision / recall counted BY PERSON — an alert is a catch if it puts the incident person
  or bag in front of the VLM at the right time, whichever rule fired; every alert counts towards precision
  (`counts_precision.json`). The routed version is kept as `counts_precision_routed.json` (`--routed`).
  Open question for the team: the VLM should answer all five questions per alert plus an open "anything else unusual?"
  (Gemma), so events outside the five reach human review; Stage 1 only flags what its rules and the 10 s scene review catch.

## 2026-11-04: how small can a person be? (scope: close to mid-range cameras first)
* `exp/person_size.py` → `person_size.json`: reference people = YOLO26l @1280 px (conf >= 0.5), one frame per 2 s on
  ABODA, Avenue, CAVIAR, MEVA, IWDD test, littering clips (10.3k people). Height measured in the 640 px detector input.
  Tracked by the gate: < 20 px 24 %, 20–30 46 %, 30–40 55 %, 40–50 82 %, 50–60 86 %, 60–80 93 %, 150+ 98 %.
  Tracked with confident wrist + knee: 0 % below 30 px, 24 % at 40–50, 50 % at 50–60, 66 % at 60–80, 43–58 % at
  80–150 (occluded wrists while walking), 87 % at 150+.
* Scope: tracking-based rules from ~40 px, pose-based rules from ~60 px (reliable ~150 px). Placement guidance: person
  >= 1/6 of frame height on 16:9 (>= 1/8 on 4:3), ideally >= 1/4. Matches DORI (IEC 62676-4): detect ~40 px,
  observe ~105 px for a 1.7 m person. Wide crowded halls (ABODA video11, MEVA) are outside this envelope; plan:
  per-camera higher input size / tiling after the CPU speed-up (YOLO26s @1280 triples confident people in video11).

## 2026-11-04: ICVL — real CCTV, never used for tuning (`exp/icvl/`)
* ICVL: 946 videos / 941 with per-person labels, 9.97 h, 11 cameras (01–12, no 09), 1280x720 – 1920x1080. Labels per
  person per frame: head + feet points, posture / movement / gesture / event. No loitering, unattended-item or rough-
  sleeping labels (2 lying episodes). `inventory.py` → `inventory.json`.
* `run.py`: current defaults (YOLO26s + pose @640, 5 fps) on all 941 → `bench_cache/models/s/icvl.pkl` (~80 min, MPS).
* `score.py` → `score.json`, counted BY PERSON. Boxes from head/feet (width 0.45 x height). Incidents: smoking
  (gaps <= 15 s merged), littering, lingering = standing/sitting stationary >= 20 s / >= 60 s (loitering PROXY).
  Caught = alert on the person within [t0 - w, t1 + w], w = 3 s littering (ICVL labels only ~2 s of the drop; 51 of 60
  "near" alerts in a first check were on the litterer within 3 s), 1 s otherwise.
  | event | incidents | caught |
  | littering | 521 | 459 (88 %) — reach_down 435 |
  | smoking | 161 | 123 (76 %) — pose_action 97, reach_down 81, stays put 53 |
  | lingering 20 s+ | 126 | 125 (99 %) — stays put alone 93 |
  | lingering 60 s+ | 27 | 27 (100 %) |
  3450 alerts (5.8/min): 1685 (49 %) on a labelled event, 1610 on a labelled person doing nothing labelled, 155 on
  nobody labelled (ICVL misses some people: 43 labelled vs 48 YOLO on 25 sampled frames). Videos with no labelled
  event: 3.1 alerts/min. ICVL is event-dense (acted), so precision here is optimistic; alerts/min is what transfers.
* `breakdown.py` → `breakdown.json` (what people were doing at off-event alerts): reach_down 1943 alerts, 1033 off
  event — 528 walking, **287 cyclists (sitting / bicycling)**, 65 phoning; pose_action 943, 384 off event — phoning /
  texting 174 (arm raised to the face, like smoking). Bicycle skip was left out on IWDD (cyclists stop to dump); the
  ICVL cache only has motorcycle boxes, so a bicycle skip needs a detector re-run.

## 2026-11-05 (overnight): cutting false alerts with ICVL (tune = ODD videos, final check = EVEN videos)
* Correction: the first rider-skip test on ICVL showed no effect because `score.load_runs` looked for the bicycle
  boxes in the wrong folder; fixed (`exp/alerts/two_wheelers.py icvl` → `twowheel_icvl.pkl`).
* **Cyclist skip** (`reach_rider_classes` (3,) → (1, 3)): ICVL odd quiet-footage alerts 3.10 → 2.01/min, all alerts
  1411 → 1252, reach_down 887 → 727, littering 226 → 224/254, smoking / lingering unchanged; cyclist reach_down false
  alerts 155 → 10. Tune sets: littering / throws / smoking unchanged, IWDD train dumping 32.5 → 31.5 % (vid0033: a
  cyclist bending to dump beside the bike; vid0447 was a chance hit by a passing cyclist). New option
  `reach_rider_stopped_h` (still check a stopped rider) does not save vid0033 (its track is too short to measure
  speed) and changes nothing else; kept off. **New default (1, 3)**, pending the held-out check.
* Rejected on ICVL odd (`sweep_*.json`): reach_margin_h −0.13 / −0.11 (littering −16 / −41), reach_min_hits 2 (−35),
  refire / cooldown changes (≤ 4 % fewer alerts), reach_kp_conf 0.6 / 0.7 (≤ 2 %), reach_max_speed_h 0.35
  (littering −19: ICVL litterers drop while walking), pose_action_thr 0.9 / 0.95 (ICVL −12 % alerts, but littering
  tune clips right person 96.5 → 76.5 at 0.95; throws / tosses need the pose cue), person_dedup_s 5 / 10 (littering
  −10 / −27). Dwell rules on people < 30 px: only 12 alerts (1 %) — not worth an option.
* **Chance check for ICVL littering** (`exp/icvl/chance.py`): a labelled person who never litters gets an alert in a
  random 7.6 s window 33 % of the time (walker reach_down alerts), vs 88 % for litterers at their drop. Real signal,
  but report caught-vs-chance, not caught alone.
* Stays-put / long-presence alerts on "nobody labelled" (46 on odd): by eye they are real people ICVL did not label
  (far away at the top edge, people sitting on the lobby floor), not phantom detections.
* **Reach verifier** (`exp/icvl/reach_verifier.py`): MLP on the pose_action 1 s summary at the reach_down moment,
  trained on ICVL odd only. ICVL even: alerts 1907 → 1363 (−29 %), quiet 1.96 → 1.33/min, littering 232 → 214/267,
  smoking 93 → 84/116. But on the original tune sets littering clips right person 96.5 → 84.3 and IWDD lead 6.9 → 4.8:
  it learned ICVL's "let go while walking". **Rejected as trained**; retraining on all tune sources
  (`reach_verifier_mixed.py`: ICVL odd + littering train + IWDD train dumpers + Avenue normals).
* **Reach verifier trained on all tune sources** (`exp/icvl/reach_verifier_mixed.py`; ICVL odd + littering train +
  IWDD train dumpers + Avenue tune normals; 30k windows; 5-fold out-of-fold by sequence, `reach_verifier_mixed_oof.json`).
  OOF on top of the cyclist skip, thr 0.05: littering tune 96.5 → 95.7 right person, throws 100, smoking same, IWDD
  lead 6.9 → 6.8, dump-site 5.81 → 5.36/min, walkway 3.72 → 3.53, ICVL odd 1252 → 1118 alerts (littering 224 → 221).
  Higher thresholds cost littering clips / IWDD. **Adopted at 0.05** (`weights/reach_verifier.npz`, final model on all
  rows; `GateConfig.reach_verifier` True by default; in-gate version checked against the offline filter).
  - 1 s look-ahead (hold the alert, also score the next second; `AHEAD_S=1`): same trade-off curve → not worth the delay.
  - Applying the verifier to pose_action alerts too (`TAG=_v2`, `sweep_pose`): −2 to −5 % alerts, littering clips
    right person 95.7 → 94.3 already at 0.05 (the pose cue is what catches throws) → **rejected**.
* **pose_action_cooldown_s 5 → 15 s** (per track): tune sets unchanged except smoking clips 53.1 → 52.4, dump sites
  5.42 → 4.99/min, ICVL odd 1150 → 1070 alerts, no catch lost on ICVL odd; 20 / 30 s add little. **Adopted.**
  `loiter_repeat_s` 120 / 300: −1 % → left at 60 s.
* **Held-out check, once** (old defaults → new: cyclist skip + reach verifier + pose cooldown 15 s):
  | ICVL even (never used) | alerts 2039 → 1569 (−23 %), quiet footage 3.10 → 1.61/min (−48 %), on a labelled event
    54 → 58 %; littering 233 → 231/267, smoking 93 → 89/116 (the cooldown: −4, none lost on odd), lingering 81/81, 17/17 |
  | littering clips | 97/85 → 96/84 caught / right person, alerts 184 → 174 |
  | walkway throws | 10/10, alerts 41 → 40 |
  | smoking set | caught 150 → 136, alerts 332 → 290 (the set's own rule catches ~10 %; most catches were reach_down by chance) |
  | IWDD test dumping | hit@10s 44.0 vs 31.2 chance → 40.0 vs 28.0 (lead 12.8 → 12.0), normal videos 6.30 → 5.37/min |
  | ABODA bags | 8/11 → 8/11, alerts 114 → 109 |
* All of ICVL with the new defaults (`score.json`; the odd half trained the verifier): 3450 → 2639 alerts (4.4/min),
  quiet footage 3.1 → 1.62/min, 54 % on a labelled event; littering 455/521 (chance 31 %), smoking 118/161,
  lingering 124/126 and 27/27. Remaining off-event alerts: mostly reach_down on walkers and the pose cue on people
  raising a hand (phone / texting, same motion as smoking).
* **Chance on ICVL for smoking and lingering** (`chance.walkers(event=...)`, all videos, new defaults): smoking 118/161
  (73 %) caught vs 45 % for non-smokers in a 13 s window. Lingering: any-rule chance is 99.6 % for a 40 s window
  (anyone in view that long gets some alert), so the by-person 124/126 is not meaningful; the stays-put rule alone
  catches 108/126 (20 s+) and 27/27 (60 s+) at 0.39 alerts/min on quiet footage, but also fires in 71 % of 40 s windows
  on other long-staying people (ICVL's "stationary" label is strict; many near-still people are labelled walking).
  Reported as "the lingering timer works"; loitering vs waiting still needs per-camera settings and the VLM.
* Held-out walkway (Avenue normal) unchanged at 4.5 alerts/min (2.7 of it stays-put on people standing still);
  IWDD test normal 6.3 → 5.37/min (`per_rule_fa.json`).
* Remaining reach_down false alerts (ICVL even, 306), by eye: people carrying a bag / paper / phone low, people
  standing with arms down under a high camera; verifier scores 0.1–0.5, where real drops also score → VLM work.
* Slides updated (ICVL slide, new held-out numbers, false-alert fixes, smoking / loitering with chance).

## 2026-11-04 — false positives per person, recall push, crop check

**New metric: per-person false positive rate** (`exp/icvl/fpr_sweep.py`): of ICVL people with no littering / smoking /
still-20 s+ incident in their video, the share flagged at least once. Deck settings (pose cooldown 15 s) were 28.7 %
on all of ICVL (949/3306), 29.2 % on the even half. Why (`exp/icvl/fp_people.py`, contact sheet
`demo/icvl_fp_people.jpg`): first flagged by reach-down 68 % (carrying a bag / papers / phone low), pose cue 21 %
(phone calls, texting), stays put / long in view 10 % (waiting, pacing on the phone; ICVL labels them walking).
Running / cycling people are rarely flagged (8 %); phone users 61 %, people standing still under 20 s 62 %.
Misses and false alarms explained in `exp/icvl/fp_fn.py` (`fp_fn.json`): littering misses are mostly the wrist not
reaching knee height (drops from waist height) and the walking-speed skip; smoking misses are mostly very short
labelled stretches (27 of 40 under 5 s; smokers who smoke 5 s+ in total: 72/80 caught).

* **reach_max_speed_h 0.45 → 0.6** (tune on odd, confirm on even + tune sets): ICVL even littering 231 → 245/267,
  quiet 1.61 → 1.70/min; tune sets unchanged (dump site 4.99 → 5.22/min). **Adopted.** Looser wrist margin with a
  stricter verifier looked better on ICVL (255/267) but lost an Avenue throw, smoking clips 52 → 40-49 % → rejected.
  Pose-cue threshold, dedup, keypoint confidence, reach window: no gain. Per-person FPR rose to 32.7 % (even).
* Tried for the false positives and **rejected**: per-person caps on sends (lose catches: the first alert is often
  before the act; `exp/icvl/one_per_person.py`), carried-bag check (YOLO sees a bag at the hand on 5 % of alerts),
  how long the hand stays low (drops and walkers identical), new static object on the ground after a reach
  (`ground_blobs.py`; follows 56 % of drops but also 29 % of walkers), YOLO phone boxes for the pose cue
  (`exp/alerts/phones.py`; phones found near 10 % of either group), pose-cue verifier trained with phone / texting as
  hard negatives (`exp/icvl/pose_verifier.py`; −1 point FPR, littering clips 97.8 → 93.9), reach verifier / pose-cue
  thresholds alone (trade recall ~1:1; pose cue 0.95 drops littering clips to 85 %), 2 low-hand frames (−19 drops),
  tighter stays-put radius 0.4 (−1.3 points, −1 lingering, −2 smokers; optional).
* **Crop check** (`stage1_crop_check.py`, `GateConfig.crop_check`, off by default): the pixels decide what body points
  cannot. Every reach_down alert's person crop (box + hand square) at t − 1 s, t and t + 1 s is embedded with a frozen
  MobileCLIP2-S2 image encoder (the public timm weights also in Yuchuan's bundle) and scored by a logistic regression
  trained on ICVL odd reach alerts (`exp/icvl/crop_v2.py`; one-crop v1 in `hand_probe.py`, `weights/crop_check_v1.npz`).
  Runs before the person dedup; live, the alert is held 1 s and the handoff uses the alert's own frame. Held-out
  ICVL even AUC 0.874 (one crop: 0.831).
  | ICVL even (held-out) | people flagged | littering | smoking | quiet /min |
  | no check | 32.7 % | 245/267 | 90/116 | 1.70 |
  | v1 one crop @0.3 | 26.3 % | 244 | 89 | 1.31 |
  | **v2 before/now/after @0.2 (default thr)** | **25.2 %** | 244 | 89 | 1.30 |
  | v2 @0.3 | 21.9 % | 242 | 88 | 1.17 |
  Tune sets (`exp/alerts/crop_v2_check.py`): v2 @0.2 littering clips 97.8 unchanged, throws 6/6, smoking clips
  52.4 → 51.3, IWDD lead 6.3 → 5.4; @0.25 and @0.3 lose 1 of 6 Avenue throws. Cut-offs chosen on odd out-of-fold
  scores agree with even. Cost: 3 encoder passes per reach_down alert (~0.28 s each on one core) and a 1 s delay.
  Live path checked against the replay on 12 even videos (scores identical); found and fixed: a held alert was
  suppressed by a later alert on the same person sent during the hold → the dedup now only lets earlier alerts
  suppress (no change to any replayed number).
* **Pose-cue crop check** (same before / now / after recipe on pose_action alerts, `crop_v2.py ... pose_action`,
  `weights/crop_check_pose_action.npz`, threshold 0.2; ICVL even AUC 0.836): on top of the reach check, ICVL even
  people flagged 25.2 → **23.9 %** (littering 244, smoking 89, lingering 81 → 80/81); tune sets unchanged (littering
  clips 97.8, throws 6/6, smoking clips 51.3 → 51.0, IWDD lead 5.4). **In the gate**: `crop_check_rules` = reach_down
  + pose_action, one weights file per rule, shared encoder; replay reproduces the numbers exactly; live path vs replay
  on 20 even videos: same alerts kept bar one borderline score.
  Stricter options (ICVL even): reach 0.3 / pose 0.2 → 20.4 % (littering 242, smoking 88) but 1 of 6 Avenue throws
  lost; reach 0.3 / pose 0.3 → 18.4 % (smoking 86).
* Rejected: stacking the crop score with the reach verifier score (odd OOF AUC 0.918 → 0.916); training the crop
  check on ICVL odd + all tune sources (`exp/alerts/crop_rows.py`, `exp/icvl/crop_mixed.py`, out-of-fold): worse
  everywhere at the same cut-offs (throws 6 → 4 of 6, littering clips 97.8 → 95.7, smoking clips 51 → 44, ICVL even
  smoking 89 → 84); the tune sets' person-level labels are coarser than ICVL's.
* **Where it stands** (ICVL even, held-out; all fixes): people with no incident flagged 32.7 → 23.9 % with
  `crop_check=True`; it is **off by default** pending a decision: it adds open_clip + MobileCLIP2-S2 to the gate
  (Yuchuan's VLM stage loads the same encoder), ~0.85 s of one core per reach / pose alert, and a 1 s hold.
  Data-pipeline: `Stage1Detector` passes frames to the engine and uses a held alert's own frame for the handoff
  (`_frame` / `_t`); its tests' fake engine takes the optional frame. Nothing committed.
* **Crop check off the frame loop**: the encoder runs on one background thread (`RuleEngine` holds the alert; crops at
  t − 1 s and t are cut at once and encoded, the t + 1 s crop when it arrives, the alert is released when its score
  is ready; the encoder is warmed up on the first live frame). `RuleEngine.flush()` releases alerts still held at the
  end of a stream (`Stage1Gate.run` calls it). Engine step on 20 even ICVL videos with real frames: max 19 ms (was
  5.8 s at the first alert, ~0.63 s at each). The person dedup now compares the alerts' own timestamps (an alert
  released late is judged as if sent when it fired); no replayed number changes (ICVL all 2863 alerts, 480/521,
  121/161; even with the check 23.9 %). Live vs replay: 29 vs 28 kept (the same borderline score).
  Cost (M3 Pro, 1 core, uncontended): 211 ms per crop, ~0.63 s per alert; 1.2 checks/min on event-free footage
  (~1.3 % of a core), 4.2/min on busy ICVL (~4.4 %). Tests: `CropCheckTests` (hold / release / drop / flush with a
  stub check). Data-pipeline: a held alert's `timestamp_ms` is its firing time; alerts held at a session reset are
  still lost (≤ ~1.2 s window: 1 s hold + the last crop encoding).
* **`crop_check` on by default** (decided 4 Nov). Needs open_clip_torch (in requirements/base.txt and
  cpu-deployment.txt) and the MobileCLIP2-S2 weights (HF hub / cache, or `STAGE1_CROP_ENCODER=local-dir:<folder>` in
  the data pipeline); if the encoder cannot load, a warning is raised once and alerts pass unchecked. To do: the
  data-pipeline `stage1` extra in `data-pipeline/pyproject.toml` should add open_clip_torch==3.3.0 (needs `uv lock`).
  Replays of cached runs are unaffected unless `stage1_crop_check.SCORES` is filled (no score = alert kept).
