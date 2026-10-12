# Corrected evaluation run

Run: `evaluation_20261012T040605_929193Z`. Execution status: **error**.

Mode: **evaluation**. Unpaced inference: no duty-cycle or camera-rate sleeps. CPU threads and memory limits still apply.

Requested tasks: littering, loitering, rough_sleeping, smoking, unattended_items. Input protocols: frame, window. Sample cap per task/protocol/split: all.

Measurements describe the recorded host. CPU throttling does not emulate a Ryzen AI Max+ 395.

All models in each task/protocol see the same images and source groups. Quality scores include failures and abstentions. Continuous-score operating thresholds are chosen on development data; VLM hard decisions have no invented AUC.

Available dataset coverage below describes prepared data, before any inference sample cap. Unrequested tasks/protocols are not evaluated.

| Task / input | Available data |
|---|---|
| littering | Needs verified event and matched normal labels |
| loitering | Needs verified event and matched normal labels |
| rough_sleeping | Needs verified event and matched normal labels |
| smoking / frame | Development only; counts {'dev': {'normal': 120, 'smoking': 116}, 'test': {}} |
| unattended_items / frame | Dev/test available; counts {'dev': {'normal': 16, 'unattended_items': 4}, 'test': {'normal': 12, 'unattended_items': 4}} |
| unattended_items / window | Dev/test available; counts {'dev': {'normal': 16, 'unattended_items': 4}, 'test': {'normal': 12, 'unattended_items': 4}} |

Input deduplication: 104 aliases removed from 3454 collected rows, using source moments and decoded RGB pixels. Provenance is saved in `evaluation.json`.

| Task / input / split | Selected / available inputs | Selected labels | Source groups | Groups per label |
|---|---:|---|---:|---|
| smoking_frame / dev | 236 / 236 | {'normal': 120, 'smoking': 116} | 16 | {'normal': 16, 'smoking': 16} |
| unattended_items_frame / dev | 20 / 20 | {'normal': 16, 'unattended_items': 4} | 3 | {'normal': 3, 'unattended_items': 1} |
| unattended_items_frame / test | 16 / 16 | {'normal': 12, 'unattended_items': 4} | 3 | {'normal': 3, 'unattended_items': 1} |
| unattended_items_window / dev | 20 / 20 | {'normal': 16, 'unattended_items': 4} | 3 | {'normal': 3, 'unattended_items': 1} |
| unattended_items_window / test | 16 / 16 | {'normal': 12, 'unattended_items': 4} | 3 | {'normal': 3, 'unattended_items': 1} |

## Measured model comparisons

| Task / input / split | Model | Coverage | Binary F1 | FPR | p95 seconds | RSS MiB | Host budget |
|---|---|---:|---:|---:|---:|---:|---|
| smoking_frame / dev | mobileclip2-s2 | 1.000 | 0.034 | 0.000 | 0.059 | 997.293 | passes sampled frame budget |
| smoking_frame / dev | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.048 | 1274.996 | passes sampled frame budget |
| smoking_frame / dev | smolvlm-500m | 0.987 | 0.162 | 0.075 | 2.629 | 3159.496 | fails/unknown |
| smoking_frame / dev | minicpm-v4 | NA | NA | NA | NA | 18407.941 | fails/unknown |
| unattended_items_frame / dev | mobileclip2-s2 | 1.000 | 0.000 | 0.000 | 0.056 | 1004.738 | passes sampled frame budget |
| unattended_items_frame / dev | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.045 | 1271.832 | passes sampled frame budget |
| unattended_items_frame / dev | smolvlm-500m | 1.000 | 0.333 | 1.000 | 3.058 | 3182.285 | fails/unknown |
| unattended_items_frame / dev | minicpm-v4 | 1.000 | 0.000 | 0.000 | 2.695 | 18304.012 | fails/unknown |
| unattended_items_frame / test | mobileclip2-s2 | 1.000 | 0.000 | 0.000 | 0.060 | 1001.355 | passes sampled frame budget |
| unattended_items_frame / test | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.044 | 1271.949 | passes sampled frame budget |
| unattended_items_frame / test | smolvlm-500m | 1.000 | 0.400 | 1.000 | 3.093 | 3180.434 | fails/unknown |
| unattended_items_frame / test | minicpm-v4 | 1.000 | 0.000 | 0.000 | 2.730 | 18333.914 | fails/unknown |
| unattended_items_window / dev | mobileclip2-s2 | 1.000 | 0.000 | 0.250 | 0.068 | 1195.867 | passes sampled frame budget |
| unattended_items_window / dev | siglip2-b16-224 | 1.000 | 0.000 | 0.062 | 0.057 | 1269.988 | passes sampled frame budget |
| unattended_items_window / dev | smolvlm-500m | 1.000 | 0.333 | 1.000 | 3.124 | 3188.676 | fails/unknown |
| unattended_items_window / dev | minicpm-v4 | 1.000 | 0.000 | 0.000 | 5.322 | 18857.254 | fails/unknown |
| unattended_items_window / test | mobileclip2-s2 | 1.000 | 0.250 | 0.250 | 0.065 | 1139.875 | passes sampled frame budget |
| unattended_items_window / test | siglip2-b16-224 | 1.000 | 0.000 | 0.250 | 0.056 | 1270.855 | passes sampled frame budget |
| unattended_items_window / test | smolvlm-500m | 1.000 | 0.353 | 0.833 | 3.098 | 3184.883 | fails/unknown |
| unattended_items_window / test | minicpm-v4 | 1.000 | 0.000 | 0.000 | 5.374 | 18820.234 | fails/unknown |

## Response diagnostics

Counts cover measured inference attempts, including repeats. Exact labels are case-insensitive; unknown and invalid responses remain separate. JSON label compliance includes structured `unknown` answers. Constant categorical predictions do not demonstrate event discrimination.

| Task / input / split | Model | Categorical prediction counts | JSON label fraction | Abstention reasons |
|---|---|---|---:|---|
| smoking_frame / dev | mobileclip2-s2 | {'normal': 702, 'smoking': 6} | NA | {} |
| smoking_frame / dev | siglip2-b16-224 | {'normal': 708} | NA | {} |
| smoking_frame / dev | smolvlm-500m | {'__unanswered__': 9, 'normal': 639, 'smoking': 60} | 0.000 | {'invalid_response': 9} |
| smoking_frame / dev | minicpm-v4 | unrecorded | NA | {} |
| unattended_items_frame / dev | mobileclip2-s2 | {'normal': 60} | NA | {} |
| unattended_items_frame / dev | siglip2-b16-224 | {'normal': 60} | NA | {} |
| unattended_items_frame / dev | smolvlm-500m | {'unattended_items': 60} | 0.000 | {} |
| unattended_items_frame / dev | minicpm-v4 | {'normal': 60} | 0.000 | {} |
| unattended_items_frame / test | mobileclip2-s2 | {'normal': 48} | NA | {} |
| unattended_items_frame / test | siglip2-b16-224 | {'normal': 48} | NA | {} |
| unattended_items_frame / test | smolvlm-500m | {'unattended_items': 48} | 0.000 | {} |
| unattended_items_frame / test | minicpm-v4 | {'normal': 48} | 0.000 | {} |
| unattended_items_window / dev | mobileclip2-s2 | {'unattended_items': 12, 'normal': 48} | NA | {} |
| unattended_items_window / dev | siglip2-b16-224 | {'normal': 57, 'unattended_items': 3} | NA | {} |
| unattended_items_window / dev | smolvlm-500m | {'unattended_items': 60} | 0.000 | {} |
| unattended_items_window / dev | minicpm-v4 | {'normal': 60} | 0.800 | {} |
| unattended_items_window / test | mobileclip2-s2 | {'normal': 36, 'unattended_items': 12} | NA | {} |
| unattended_items_window / test | siglip2-b16-224 | {'unattended_items': 9, 'normal': 39} | NA | {} |
| unattended_items_window / test | smolvlm-500m | {'unattended_items': 39, 'normal': 9} | 0.000 | {} |
| unattended_items_window / test | minicpm-v4 | {'normal': 48} | 1.000 | {} |

Constant-prediction baselines (quality only):

```json
{
  "smoking_frame/dev": {
    "always_normal": {
      "accuracy": 0.5084745762711864,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.4915254237288136,
      "binary_f1": 0.6590909090909091
    }
  },
  "unattended_items_frame/dev": {
    "always_normal": {
      "accuracy": 0.8,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.2,
      "binary_f1": 0.3333333333333333
    }
  },
  "unattended_items_frame/test": {
    "always_normal": {
      "accuracy": 0.75,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.25,
      "binary_f1": 0.4
    }
  },
  "unattended_items_window/dev": {
    "always_normal": {
      "accuracy": 0.8,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.2,
      "binary_f1": 0.3333333333333333
    }
  },
  "unattended_items_window/test": {
    "always_normal": {
      "accuracy": 0.75,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.25,
      "binary_f1": 0.4
    }
  }
}
```

Resource screen: one stream sampled at 1 frames/s, p95 inference ≤ 1s, worker/process-tree RSS cap 24576 MiB. This excludes queueing and camera decoding; verify on the deployment box.

Inference latency excludes pacing sleeps. Active throughput, wall throughput and pacing sleep seconds are saved separately in `results.json` and `comparison.csv`; deployment duty-cycle overhead is included in the host-budget screen.

Raw model argmax and development-calibrated operating points are separate in `calibration.json`. Group-bootstrap intervals and paired differences are in each run's `improvements.json`. Unsupported tasks are not zero-scored or pooled into a model ranking.

A final choice requires adequate recall at an acceptable false-alert rate on verified events from the deployment domain. A small staged set cannot establish a five-task deployment winner.

## Limitations

- Historical benchmark groups are development only.
- CAVIAR is a small staged single-camera set; source-sequence holdout is not a new-camera holdout.
- Window contact sheets are a separate past-context protocol, not native video input.
- Proxy pose/person/action labels do not establish rough sleeping, loitering or littering events.
- ABODA, D2-CDNet and Roboflow candidates need event/normal review before task scoring.
