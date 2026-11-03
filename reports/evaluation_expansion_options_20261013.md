# Why the current cohorts are small, and how to expand them

This is an audit of expansion options, not a new inference experiment. Existing evaluation results and slides retain their original cohorts.

## Two constraints: sampling and event-label coverage

The final quality runs removed the recording-level preparation cap and inference limit. However, CAVIAR video extraction still used the default **four endpoints per reviewed interval**. Nine configured intervals therefore produced 36 endpoint frames and 36 past-window images. The six source recordings were split into three development and three test groups, with only one positive source sequence per split. See [the extraction loop](../anomaly_eval/datasets/prepare.py) and [reviewed intervals](../anomaly_eval/config/caviar_review_intervals.json).

This is a deliberately sparse preliminary cohort, rather than the maximum amount of test material available from these videos. Its limited size should not be attributed solely to missing data or computing capacity.

The source annotation timing permits more samples from the same intervals:

| Endpoints requested per interval | Planned endpoints | Distinct XML frame indices across the intervals | Positive source groups per split |
|---|---:|---:|---:|
| 4, used in the current run | 36 | 36 | 1 |
| 20 | 180 | 157 | 1 |
| 50 | 450 | 348 | 1 |

The last two rows are feasibility counts calculated from the configured intervals and 25 fps XML frame indexing. They are **not extracted or scored inputs**. Decoded-pixel deduplication can reduce counts further. Short intervals cannot supply arbitrarily many distinct frames. Past windows share the endpoint events and should not be counted as additional independent events.

The existing standalone preparation command already exposes the sampling setting:

```bash
.venv/bin/python -m anomaly_eval prepare \
  --per-group 0 --per-interval 20 \
  --output data/evaluation_caviar20
```

Use a fresh output directory, then point a new run at it with `--prepared-dir` and `--skip-prepare`. The unified runner currently has no `--per-interval` argument, so this separate preparation step is necessary to change extraction density.

## More independent unattended-item examples are already local

- `LeftBag_BehindChair` and `LeftBox` have local videos and XML annotations but are absent from the current interval configuration. Review their positive intervals, attended placement and retrieval controls before importing them.
- The 11 ABODA videos are downloaded. The publisher distributes abandoned-object scenarios, but this checkout has no reviewed frame/event boundaries for them. Reviewing positive intervals and normal/hard-negative periods can provide additional recordings and scene conditions. [ABODA publisher repository](https://github.com/kevinlin311tw/ABODA).
- All 16 videos under `data/raw/datasets/unattended items/` are byte-identical to their counterparts in `data/additional/`: 11 ABODA videos and five CAVIAR leaving-object videos. These copies must not be counted as additional recordings.

Increasing extraction density helps characterize each existing sequence. Adding reviewed recordings is more valuable for testing generalization.

## The other three tasks are useful material, with different evidence levels

| Target | Available material | Useful test now | What is needed for target-event scoring |
|---|---|---|---|
| Littering | 6,721 images; dropping, dumping, putting-down, throwing and tossing boxes | Annotated action recognition or an explicitly described weak-label diagnostic | Review actual rubbish disposal and normal/hard-negative examples; repair filename aliases; split by source recording rather than blindly retaining the export split |
| Loitering | 1,675 images, 95 videos; the Roboflow boxes identify people | Person/presence diagnostics and temporal review candidates | Define a duration/zone rule, connect source videos or tracks to frames, label qualifying intervals and ordinary passage/waiting controls |
| Rough sleeping | 224 image files, 112 unique hashes; synthetic laying/standing data | Lying/fallen-person or pose diagnostics | Actual public-area sleeping/lying-with-bedding examples under the current definition, plus task-specific controls and reviewed source identity |

“Unusable” is too broad. These sources are excluded from the **confirmed five-event accuracy tables**, but they can be used for explicitly named diagnostic tests and as annotation candidates. Their current label names cannot automatically establish the requested event.

The rough-sleeping [source README](<../data/raw/datasets/rough sleeping/hugging_face/README.md>) explicitly describes a **synthetic fall/pose/incident dataset**. Its labels are laying/fallen and standing/walking. A mixed scene containing a standing person is not a sleeping-negative scene when another person is lying down. The current benchmark definition also requires public-area and bedding evidence; testing lying-person recognition would be a different operational target.

All 37 local Avenue MAT files were rechecked using their schemas: they contain `vol` arrays of type `uint8`, rather than event ground truth. This does not mean Avenue has no labels: the [official Avenue page](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) provides a separate ground-truth download for general abnormal-event detection. That can support a broader anomaly benchmark after importing the annotations; loitering-specific intervals still need to be identified instead of treating every abnormal event as loitering.

The prepared dataset metadata lists 7,200 proxy candidates collected, while `proxy_development.csv` saves only the first 1,000. This separate bounded preview export is another limit to expose when building a larger diagnostic review set; it did not limit confirmed smoking/CAVIAR inference.

## Proposed expansion order

1. Review the two unused CAVIAR clips and the existing ABODA videos; add event/control intervals while retaining recording groups and duplicate checks. Use more densely sampled past windows where temporal evidence is useful.
2. Review a recording-diverse littering subset, preserving the distinction between rubbish disposal and generic actions. Add enough matched normal examples to measure false alarms.
3. Run explicitly labelled action/person/lying diagnostics on the remaining material. Report positive recognition rates when only positives are available; do not present unsupported precision, FPR or deployment conclusions.
4. Import duration-labelled loitering and genuine rough-sleeping examples, or explicitly change the operational target to the proxy behavior being tested.
5. Freeze prompts and thresholds on development material, then reserve fresh recordings/cameras for the enlarged final test. The already-inspected inputs are development evidence for subsequent protocol choices.

The bottleneck is trustworthy event labels and independent source coverage. GPU capacity is sufficient for considerably larger comparisons.
