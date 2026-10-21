# Corrected evaluation run

Run: `evaluation_20261011T120315_062384Z`. Execution status: **complete_for_available_tasks**.

Mode: **quick**. Unpaced inference: no duty-cycle or camera-rate sleeps. CPU threads and memory limits still apply.

Requested tasks: littering, loitering, rough_sleeping, smoking, unattended_items. Input protocols: frame. Sample cap per task/protocol/split: 8.

Measurements describe the recorded host. CPU throttling does not emulate a Ryzen AI Max+ 395.

All models in each task/protocol see the same images and source groups. Quality scores include failures and abstentions. Continuous-score operating thresholds are chosen on development data; VLM hard decisions have no invented AUC.

Available dataset coverage below describes prepared data, before any inference sample cap. Unrequested tasks/protocols are not evaluated.

| Task / input | Available data |
|---|---|
| littering | Needs verified event and matched normal labels |
| loitering | Needs verified event and matched normal labels |
| rough_sleeping | Needs verified event and matched normal labels |
| smoking / frame | Development only; counts {'dev': {'smoking': 116, 'normal': 120}, 'test': {}} |
| unattended_items / frame | Dev/test available; counts {'dev': {'normal': 24, 'unattended_items': 8}, 'test': {'normal': 24, 'unattended_items': 8}} |
| unattended_items / window | Dev/test available; counts {'dev': {'normal': 16, 'unattended_items': 4}, 'test': {'normal': 12, 'unattended_items': 4}} |

This is a subset/pilot measurement. Reduced label and recording counts limit quality conclusions, confidence intervals and false-positive calibration; use an uncapped evaluation for model selection.

| Task / input / split | Selected / available inputs | Selected labels | Source groups |
|---|---:|---|---:|
| smoking_frame / dev | 8 / 236 | {'normal': 4, 'smoking': 4} | 4 |
| unattended_items_frame / dev | 8 / 32 | {'normal': 6, 'unattended_items': 2} | 3 |
| unattended_items_frame / test | 8 / 32 | {'normal': 6, 'unattended_items': 2} | 3 |

## Measured model comparisons

| Task / input / split | Model | Coverage | Binary F1 | FPR | p95 seconds | RSS MiB | Host budget |
|---|---|---:|---:|---:|---:|---:|---|
| smoking_frame / dev | mobileclip2-s2 | 1.000 | 0.000 | 0.000 | 0.055 | 929.898 | passes sampled frame budget |
| smoking_frame / dev | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.061 | 1266.840 | passes sampled frame budget |
| smoking_frame / dev | smolvlm-500m | 0.000 | 0.000 | 0.000 | 3.418 | 3164.281 | fails/unknown |
| smoking_frame / dev | minicpm-v4 | 1.000 | 0.000 | 0.000 | 5.941 | 18088.754 | fails/unknown |
| unattended_items_frame / dev | mobileclip2-s2 | 1.000 | 0.000 | 0.000 | 0.051 | 1225.855 | passes sampled frame budget |
| unattended_items_frame / dev | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.052 | 1264.164 | passes sampled frame budget |
| unattended_items_frame / dev | smolvlm-500m | 0.000 | 0.000 | 0.000 | 3.854 | 3173.523 | fails/unknown |
| unattended_items_frame / dev | minicpm-v4 | 1.000 | 0.000 | 0.000 | 3.297 | 18891.340 | fails/unknown |
| unattended_items_frame / test | mobileclip2-s2 | 1.000 | 0.000 | 0.000 | 0.054 | 929.188 | passes sampled frame budget |
| unattended_items_frame / test | siglip2-b16-224 | 1.000 | 0.000 | 0.000 | 0.056 | 1264.297 | passes sampled frame budget |
| unattended_items_frame / test | smolvlm-500m | 0.000 | 0.000 | 0.000 | 3.840 | 3171.484 | fails/unknown |
| unattended_items_frame / test | minicpm-v4 | 1.000 | 0.000 | 0.000 | 3.244 | 18516.430 | fails/unknown |

Constant-prediction baselines (quality only):

```json
{
  "smoking_frame/dev": {
    "always_normal": {
      "accuracy": 0.5,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.5,
      "binary_f1": 0.6666666666666666
    }
  },
  "unattended_items_frame/dev": {
    "always_normal": {
      "accuracy": 0.75,
      "binary_f1": 0.0
    },
    "always_anomaly": {
      "accuracy": 0.25,
      "binary_f1": 0.4
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
