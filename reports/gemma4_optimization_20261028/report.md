# Gemma 4 second-stage optimization — 28 October 2026

Completed variants: 5/5. Existing baseline and first-stage optimized results are preserved. This is a retrospective study using previously inspected holdouts.

Three new input policies test native 1,120-token detail, explicit competing explanations at 560 tokens, and full-scene plus automatic person/object crops at 560 tokens. Crops use existing label-free Faster R-CNN detections, not annotation boxes. Smoking crops emphasize the upper body. Cropped sequences use four chronological time points, each followed by an optional detail crop; other native routes retain up to eight time points.

The original primary-development screen chooses a new-policy finalist per task. That finalist, baseline and the supported prior route are compared on the entire primary development set. Unsupported sheet layouts are excluded. Three methods compare full-development source-AP selection, a per-source development safeguard, and equal baseline/selected standardized-score blending. All routes, blend constants and four operating-point thresholds are frozen before new held-out inference. XRF development supplies a separate secondary cutoff only.

| Model | Method | All-source macro AP | Common-source macro AP | smoking | littering | loitering | rough_sleeping | unattended_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-e2b | baseline | 0.6503 | 0.6714 | 0.6775 | 0.5605 | 0.2921 | 0.9509 | 0.7706 |
| gemma4-e2b | optimized | 0.6347 | 0.6557 | 0.5912 | 0.5486 | 0.2808 | 0.9522 | 0.8009 |
| gemma4-e4b | baseline | 0.6887 | 0.7112 | 0.6857 | 0.7144 | 0.2470 | 0.9656 | 0.8309 |
| gemma4-e4b | optimized | 0.6788 | 0.7010 | 0.6712 | 0.7385 | 0.1844 | 0.9791 | 0.8208 |
| gemma4-12b | baseline | 0.6852 | 0.7072 | 0.7467 | 0.7259 | 0.2843 | 0.9401 | 0.7291 |
| gemma4-12b | optimized | 0.7224 | 0.7444 | 0.7467 | 0.7617 | 0.3609 | 0.9779 | 0.7651 |
| gemma4-26b-a4b | baseline | 0.6968 | 0.7247 | 0.7272 | 0.8221 | 0.2366 | 0.9450 | 0.7531 |
| gemma4-26b-a4b | optimized | 0.7108 | 0.7387 | 0.7272 | 0.8294 | 0.2995 | 0.9450 | 0.7531 |
| gemma4-31b | baseline | 0.7395 | 0.7678 | 0.7299 | 0.9131 | 0.2662 | 0.9635 | 0.8248 |
| gemma4-31b | optimized | 0.7394 | 0.7674 | 0.7294 | 0.9131 | 0.2662 | 0.9635 | 0.8248 |
| gemma4-e2b | full_dev_selected | 0.6553 | 0.6760 | 0.6716 | 0.5710 | 0.2808 | 0.9522 | 0.8009 |
| gemma4-e2b | source_safeguarded | 0.6492 | 0.6700 | 0.6716 | 0.5710 | 0.2808 | 0.9522 | 0.7706 |
| gemma4-e2b | baseline_blend | 0.6554 | 0.6772 | 0.6861 | 0.5632 | 0.2873 | 0.9542 | 0.7861 |
| gemma4-e4b | full_dev_selected | 0.6879 | 0.7107 | 0.6860 | 0.7504 | 0.2101 | 0.9791 | 0.8139 |
| gemma4-e4b | source_safeguarded | 0.6953 | 0.7180 | 0.6860 | 0.7504 | 0.2470 | 0.9791 | 0.8139 |
| gemma4-e4b | baseline_blend | 0.6942 | 0.7177 | 0.7136 | 0.7510 | 0.2263 | 0.9741 | 0.8060 |
| gemma4-12b | full_dev_selected | 0.7348 | 0.7597 | 0.7616 | 0.7815 | 0.3879 | 0.9779 | 0.7651 |
| gemma4-12b | source_safeguarded | 0.7318 | 0.7538 | 0.7467 | 0.7815 | 0.3879 | 0.9779 | 0.7651 |
| gemma4-12b | baseline_blend | 0.7402 | 0.7640 | 0.7579 | 0.7947 | 0.4017 | 0.9615 | 0.7854 |
| gemma4-26b-a4b | full_dev_selected | 0.7178 | 0.7446 | 0.7622 | 0.8294 | 0.2995 | 0.9450 | 0.7531 |
| gemma4-26b-a4b | source_safeguarded | 0.7178 | 0.7446 | 0.7622 | 0.8294 | 0.2995 | 0.9450 | 0.7531 |
| gemma4-26b-a4b | baseline_blend | 0.7149 | 0.7435 | 0.7511 | 0.8475 | 0.2777 | 0.9450 | 0.7531 |
| gemma4-31b | full_dev_selected | 0.7485 | 0.7752 | 0.7588 | 0.9131 | 0.2806 | 0.9654 | 0.8248 |
| gemma4-31b | source_safeguarded | 0.7485 | 0.7752 | 0.7588 | 0.9131 | 0.2806 | 0.9654 | 0.8248 |
| gemma4-31b | baseline_blend | 0.7483 | 0.7763 | 0.7519 | 0.9131 | 0.2890 | 0.9625 | 0.8248 |

AP averages sources within each of the five equally weighted objectives. Common-source macro excludes the weak XRF smoking proxy, matching the previous 3,792-case comparison; all-source macro includes all 4,032 held-out cases. Complete AUROC, AP, precision, recall, F1, FPR, accuracy, balanced accuracy and confusion counts are in metrics.csv. A 5% development FPR constraint is not a test guarantee.

Scores are imported from the previous stage only after an exact-path numerical development parity check (maximum allowed margin drift 0.02). Raw caches bind model receipts, prior score hashes, detector hashes, prompts, implementation and the frozen manifest. No Gemma weights are trained. These are classification/clip experiments, not continuous detector/tracker event validation. Reused holdouts, weak labels and limited camera/owner evidence constrain generalization claims.

## Operating-point comparison

Each metric below first averages sources within a task, then the five tasks equally. These are task/source macro values, not pooled case metrics. All rows use each method's frozen every-source primary-development FPR-budget threshold. AP measures ranking and can improve while F1 or recall falls. Test sources exceeding 5% FPR and sources with zero recall are reported explicitly. All four operating points, both the 4,032-case all-source and 3,792-case common-source cohorts, confusion totals and micro F1 are in `operating_summary.csv`. The blend's raw-zero point is a centered-score control.

| Model | Method | AP | AUROC | F1 | Recall | FPR | Sources >5% FPR | Zero-recall sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-e2b | baseline | 0.6503 | 0.7497 | 0.3227 | 0.2869 | 0.0284 | 2/11 | 5/11 |
| gemma4-e2b | optimized | 0.6347 | 0.7419 | 0.3294 | 0.2729 | 0.0275 | 2/11 | 4/11 |
| gemma4-e4b | baseline | 0.6887 | 0.7746 | 0.3661 | 0.3410 | 0.1151 | 4/11 | 3/11 |
| gemma4-e4b | optimized | 0.6788 | 0.7776 | 0.4355 | 0.4060 | 0.1279 | 5/11 | 2/11 |
| gemma4-12b | baseline | 0.6852 | 0.7735 | 0.3287 | 0.3211 | 0.0431 | 4/11 | 3/11 |
| gemma4-12b | optimized | 0.7224 | 0.8244 | 0.4676 | 0.4330 | 0.0564 | 5/11 | 2/11 |
| gemma4-26b-a4b | baseline | 0.6968 | 0.7888 | 0.4159 | 0.4079 | 0.0620 | 4/11 | 3/11 |
| gemma4-26b-a4b | optimized | 0.7108 | 0.7993 | 0.5481 | 0.5399 | 0.0900 | 5/11 | 3/11 |
| gemma4-31b | baseline | 0.7395 | 0.8249 | 0.4213 | 0.3874 | 0.0737 | 4/11 | 2/11 |
| gemma4-31b | optimized | 0.7394 | 0.8256 | 0.4258 | 0.3880 | 0.0711 | 4/11 | 2/11 |
| gemma4-e2b | full_dev_selected | 0.6553 | 0.7613 | 0.3286 | 0.2821 | 0.0169 | 2/11 | 4/11 |
| gemma4-e2b | source_safeguarded | 0.6492 | 0.7537 | 0.3222 | 0.2722 | 0.0166 | 2/11 | 4/11 |
| gemma4-e2b | baseline_blend | 0.6554 | 0.7568 | 0.3244 | 0.2896 | 0.0162 | 1/11 | 5/11 |
| gemma4-e4b | full_dev_selected | 0.6879 | 0.7760 | 0.4342 | 0.4181 | 0.1085 | 6/11 | 0/11 |
| gemma4-e4b | source_safeguarded | 0.6953 | 0.7838 | 0.4057 | 0.3999 | 0.1054 | 6/11 | 1/11 |
| gemma4-e4b | baseline_blend | 0.6942 | 0.7788 | 0.4301 | 0.4248 | 0.1287 | 6/11 | 1/11 |
| gemma4-12b | full_dev_selected | 0.7348 | 0.8261 | 0.3991 | 0.3518 | 0.0307 | 1/11 | 3/11 |
| gemma4-12b | source_safeguarded | 0.7318 | 0.8251 | 0.3873 | 0.3563 | 0.0359 | 3/11 | 3/11 |
| gemma4-12b | baseline_blend | 0.7402 | 0.8234 | 0.4161 | 0.3741 | 0.0168 | 2/11 | 2/11 |
| gemma4-26b-a4b | full_dev_selected | 0.7178 | 0.8105 | 0.5450 | 0.5413 | 0.0914 | 5/11 | 3/11 |
| gemma4-26b-a4b | source_safeguarded | 0.7178 | 0.8105 | 0.5450 | 0.5413 | 0.0914 | 5/11 | 3/11 |
| gemma4-26b-a4b | baseline_blend | 0.7149 | 0.8091 | 0.5170 | 0.4982 | 0.0828 | 6/11 | 3/11 |
| gemma4-31b | full_dev_selected | 0.7485 | 0.8471 | 0.4670 | 0.4250 | 0.0966 | 4/11 | 1/11 |
| gemma4-31b | source_safeguarded | 0.7485 | 0.8471 | 0.4670 | 0.4250 | 0.0966 | 4/11 | 1/11 |
| gemma4-31b | baseline_blend | 0.7483 | 0.8451 | 0.4351 | 0.4049 | 0.0896 | 5/11 | 2/11 |

gemma4-e2b: compared with the previous optimized route, full-development selection changes all-source macro AP by +0.0206 and macro F1 at this operating point by -0.0008. These are descriptive retrospective differences, not a deployment promotion.

gemma4-e4b: compared with the previous optimized route, full-development selection changes all-source macro AP by +0.0091 and macro F1 at this operating point by -0.0013. These are descriptive retrospective differences, not a deployment promotion.

gemma4-12b: compared with the previous optimized route, full-development selection changes all-source macro AP by +0.0123 and macro F1 at this operating point by -0.0685. These are descriptive retrospective differences, not a deployment promotion.

gemma4-26b-a4b: compared with the previous optimized route, full-development selection changes all-source macro AP by +0.0070 and macro F1 at this operating point by -0.0030. These are descriptive retrospective differences, not a deployment promotion.

gemma4-31b: compared with the previous optimized route, full-development selection changes all-source macro AP by +0.0091 and macro F1 at this operating point by +0.0412. These are descriptive retrospective differences, not a deployment promotion.

## Matched incumbent comparison

The common-source cohort contains the same 3,792 held-out IDs, labels, sources and groups as the earlier CLIP/SmolVLM comparison. Its original identity audit is retained in the previous report. Different training, crops, input processing and precision make these pipeline comparisons. The additional XRF gesture proxy is excluded from every row below and from `common_source_comparison.csv`.

| Incumbent | Stage | Common-source macro AP |
| --- | --- | --- |
| mobileclip2-s2 | dev_selected_strategy | 0.7572 |
| siglip2-b16-224 | dev_selected_strategy | 0.7911 |
| smolvlm-500m | original | 0.6125 |
| smolvlm-500m | routed | 0.6555 |

The highest observed standalone Gemma common-source macro AP across the reported methods is 0.7763 (gemma4-31b, baseline_blend); the incumbent reference is 0.7911 (siglip2-b16-224, dev_selected_strategy). This describes the retrospective results; choosing an implementation from held-out scores does not create a development-selected deployment policy.

## Source-level failure checks

All rows below use full-development selection and its frozen every-source development FPR-budget threshold. A ranking gain does not imply useful event recall. The CCTV smoking cohort has one camera, Street Scene loitering has six held-out recording groups from one camera, reviewed unattended items have two scene groups, and MEVA labels abandonment-action windows rather than persistent ownership state.

| Model | CCTV AP | CCTV recall | CCTV FPR | Loiter AP | Loiter recall | MEVA AP | MEVA recall | Reviewed-item FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-e2b | 0.5767 | 0.0000 | 0.0000 | 0.2808 | 0.0000 | 0.6114 | 0.0000 | 0.0028 |
| gemma4-e4b | 0.6702 | 0.0814 | 0.0356 | 0.2101 | 0.0909 | 0.6315 | 0.1333 | 0.5208 |
| gemma4-12b | 0.6980 | 0.1298 | 0.0356 | 0.3879 | 0.0000 | 0.5442 | 0.0000 | 0.0347 |
| gemma4-26b-a4b | 0.6889 | 0.0000 | 0.0000 | 0.2995 | 0.0000 | 0.5313 | 0.0667 | 0.0625 |
| gemma4-31b | 0.6801 | 0.0636 | 0.0632 | 0.2806 | 0.0909 | 0.6933 | 0.2667 | 0.3069 |

The frozen manifest has 6,004 cases: 1,812 primary development, 160 secondary source-development and 4,032 held out. The previous report's `dataset_counts.csv` gives the complete source/split denominators. Only the queried objective is annotated; other objectives are not automatically negative. Public-sleep labels describe provisional visible sleeping/bedding, without inferring housing status. Smoking primary development contains single images, so temporal crop/detail behavior remains weakly validated.

E2B, E4B and 12B use native BF16 Transformers inference. The 26B-A4B and 31B variants use official Google QAT Q4_0 GGUF releases through the pinned CUDA llama.cpp runtime. This is a pipeline comparison, with size, architecture, precision and runtime differences. GPU inference has no deliberate pacing; no latency benchmark is included. The Yes-minus-No margin is a ranking score, not a calibrated probability.

`ap_uncertainty.csv` and `ap_uncertainty.json`, generated with the separate uncertainty command, give paired observed-group bootstrap intervals against the original baseline. They do not estimate new-camera performance, remove weak-label uncertainty or undo repeated holdout inspection. Sources with fewer than three groups have no interval. No Gemma weights were updated; these experiments optimize input representation, prompts, development routing, score blending and thresholds. Continuous detector/tracker event recall and false alerts per hour still require a separate prospective evaluation.
