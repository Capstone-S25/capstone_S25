"""Performance and classification metrics without an extra scientific dependency."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def roc_auc(pairs: list[tuple[bool, float]]) -> float | None:
    """Rank-sum AUC with average ranks for tied scores."""
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[1])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        rank_sum += rank * sum(label for label, _ in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(pairs: list[tuple[bool, float]]) -> float | None:
    """Noninterpolated AP, treating tied scores as one operating threshold."""
    positives = sum(label for label, _ in pairs)
    if not positives or positives == len(pairs):
        return None
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    index, true_positives, ap = 0, 0, 0.0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        added = sum(label for label, _ in ordered[index:end])
        true_positives += added
        ap += added / positives * true_positives / end
        index = end
    return ap


def detection_metrics(
    rows: list[dict], labels: list[str], scored_labels: list[str] | None = None
) -> dict:
    """All-label recall includes unanswered cases; conditional accuracy is separate.

    Macro classes are fixed from ground-truth support, never from model predictions.
    Pass scored_labels from the full dataset when bootstrapping. An unanswered normal
    is neither a true negative nor a false alert; its count and coverage are explicit.
    """
    labeled = [
        r for r in rows if r.get("repeat", 0) == 0 and r.get("label") is not None
    ]
    if any(r["label"] not in labels for r in labeled):
        raise ValueError("Unknown ground-truth class")
    valid = [
        r for r in labeled if r.get("status") == "ok" and r.get("prediction") in labels
    ]
    count = len(labeled)
    supported = [label for label in labels if any(r["label"] == label for r in labeled)]
    active = supported if scored_labels is None else scored_labels
    if not set(active) <= set(labels):
        raise ValueError("Scored classes must belong to definitions")
    matrix = {
        actual: {pred: 0 for pred in labels + ["__unanswered__"]} for actual in labels
    }
    for r in labeled:
        pred = r.get("prediction") if r.get("status") == "ok" else None
        matrix[r["label"]][pred if pred in labels else "__unanswered__"] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in labels if a != label)
        support = sum(matrix[label].values())
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else None
        per_class[label] = {
            "precision": precision if support or fp else None,
            "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if support or fp else None,
            "support": support,
            "unanswered": matrix[label]["__unanswered__"],
        }
    tp = sum(r["label"] != "normal" and r["prediction"] != "normal" for r in valid)
    fp = sum(r["label"] == "normal" and r["prediction"] != "normal" for r in valid)
    tn = sum(r["label"] == "normal" and r["prediction"] == "normal" for r in valid)
    positives = sum(r["label"] != "normal" for r in labeled)
    negatives = count - positives
    fn = positives - tp  # includes positive abstentions and failures
    unanswered_normal = matrix.get("normal", {}).get("__unanswered__", 0)
    pairs = [
        (r["label"] != "normal", r["anomaly_score"])
        for r in labeled
        if r.get("anomaly_score") is not None and math.isfinite(r["anomaly_score"])
    ]
    correct = sum(r["label"] == r["prediction"] for r in valid)
    return {
        "metric_version": 2,
        "labeled_samples": count,
        "valid_predictions": len(valid),
        "scored_labels": active,
        "unsupported_labels": [l for l in labels if l not in supported],
        "prediction_coverage": len(valid) / count if count else None,
        "abstentions": sum(r.get("status") == "abstain" for r in labeled),
        "errors": sum(r.get("status") == "error" for r in labeled),
        "accuracy_including_failures": correct / count if count else None,
        "accuracy": correct / len(valid) if valid else None,
        "conditional_accuracy": correct / len(valid) if valid else None,
        "macro_f1": statistics.mean(per_class[l]["f1"] or 0.0 for l in active)
        if active
        else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "binary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "unanswered_normal": unanswered_normal,
            "precision": tp / (tp + fp) if tp + fp else 0.0 if positives else None,
            "recall": tp / positives if positives else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if positives or fp else None,
            "false_positive_rate": fp / negatives if negatives else None,
            "specificity": tn / negatives if negatives else None,
            "roc_auc": roc_auc(pairs),
            "average_precision": average_precision(pairs),
            "auc_samples": len(pairs),
            "score_coverage": len(pairs) / count if count else None,
        },
    }


def select_threshold(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import (
        precision_recall_curve,
    )

    y, score = _validate(y_true, y_score)
    if len(np.unique(y)) != 2:
        raise ValueError(
            "Threshold calibration requires both positive and negative development frames"
        )
    p, r, thresholds = precision_recall_curve(y, score)
    f = np.divide(
        2 * p[:-1] * r[:-1],
        p[:-1] + r[:-1],
        out=np.zeros_like(p[:-1]),
        where=(p[:-1] + r[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f))])


def _validate(y_true, y_score):
    import numpy as np

    y, score = np.asarray(y_true), np.asarray(y_score, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, aligned one-dimensional labels and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError(
            "Evaluation needs original binary labels and finite scores, not soft targets"
        )
    return y.astype(np.int8), score


def frame_metrics(y_true, y_score, threshold: float | None = None) -> dict:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    y, score = _validate(y_true, y_score)
    both = len(np.unique(y)) == 2
    # Oracle is diagnostic only. Test callers MUST pass a development-selected threshold.
    oracle = threshold is None
    if threshold is None:
        threshold = select_threshold(y, score) if both else 0.5
    if not np.isfinite(threshold):
        raise ValueError("Operating threshold must be finite")
    pred = score >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {
        "n": len(y),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, score)) if both else None,
        "aupr": float(average_precision_score(y, score)) if both else None,
        "threshold": float(threshold),
        "threshold_source": "oracle_diagnostic" if oracle else "development",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(np.mean(pred == y)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        # Compatibility fields always describe the SAME applied threshold.
        "f1_opt": f1,
        "precision_opt": precision,
        "recall_opt": recall,
        "threshold_opt": float(threshold),
        "acc_at_f1opt": float(np.mean(pred == y)),
    }
