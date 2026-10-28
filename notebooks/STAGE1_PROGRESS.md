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
