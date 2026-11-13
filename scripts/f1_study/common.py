"""Shared data, metrics and reporting for the November F1 study.

Every model is scored on the same held-out sources as the October strategy
study. Thresholds come from training/development data only; the "best
threshold" column is a diagnostic that shows how much F1 is lost to threshold
transfer, never a selection rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/strategy_study_20261016"
OUT = ROOT / "results/f1_study_20261111"
TASKS = ["smoking", "littering", "loitering", "rough_sleeping", "unattended_items"]

# Held-out evidence, by task. Names match the October report.
TEST_SOURCES = {
    ("smoking", "kaggle_sujaykapadnis_smoking"): "Smoking: photo holdout",
    ("smoking", "hmdb51_action_transfer"): "Smoking: HMDB action clips",
    ("smoking", "smoking_cctv"): "Smoking: CCTV camera c1",
    ("smoking", "stanford40"): "Smoking: Stanford40 photos",
    ("littering", "mivia_iwdd"): "Littering: IWDD official test",
    ("loitering", "street_scene"): "Loitering: Street Scene",
    ("rough_sleeping", "commons_visible_sleeping"): "Sleeping: old photos",
    ("rough_sleeping", "commons_new_author_transfer"): "Sleeping: new-author photos",
    ("unattended_items", "unattended_reviewed"): "Unattended: held-out cameras",
    ("unattended_items", "meva_abandon_action_transfer"): "Unattended: MEVA windows",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_rows():
    """All train/dev rows and every held-out row, tagged with `split`."""
    rows = list(
        read_jsonl(DATA / "samples.jsonl")
    )  # train, dev and the photo holdout (split=test)
    for row in read_jsonl(DATA / "historical.jsonl"):
        rows.append({**row, "split": "test"})
    for name in ["hmdb_screened", "meva_screened", "rough_screened"]:
        rows.extend(read_jsonl(DATA / f"{name}.jsonl"))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample ids across manifests")
    for row in rows:
        if row["task"] in REFERENCE_TASKS:
            row["reference"] = reference_frame(row["views"][0])
    return rows


# Tasks whose windows come from longer videos of a fixed camera. Their earliest
# extracted frame is a label-free picture of the scene "before", the way a
# deployed camera can keep a reference frame of its own empty scene.
REFERENCE_TASKS = {"unattended_items", "loitering"}


_REFERENCES: dict[str, str] = {}


def reference_frame(view):
    folder = str(Path(view).parent)
    if folder not in _REFERENCES:
        names = sorted(
            p.name for p in (ROOT / folder).iterdir() if p.suffix in {".png", ".jpg"}
        )
        _REFERENCES[folder] = f"{folder}/{names[0]}"
    return _REFERENCES[folder]


def counts(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, bool)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn


def f1_at(y, score, threshold):
    tp, fp, fn, tn = counts(y, np.asarray(score) >= threshold)
    return {
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(y, score):
    """Threshold maximizing F1; ties prefer the lower false-positive rate."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    s, t = score[order], y[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    f1 = 2 * tp / np.maximum(1, tp + fp + t.sum())
    # Only cut between distinct scores.
    last = np.r_[s[1:] != s[:-1], True]
    f1 = np.where(last, f1, -1)
    i = int(np.argmax(f1 - 1e-9 * fp))
    return float(s[i]), float(f1[i])


def evaluate(rows, scores, thresholds):
    """Per-source metrics for held-out rows. `thresholds` maps task -> value."""
    by_source = defaultdict(list)
    for row, score in zip(rows, scores):
        key = (row["task"], row["source"])
        if row["split"] in {"test", "external"} and key in TEST_SOURCES:
            by_source[key].append((row["target"], float(score)))
    out = {}
    for key, name in TEST_SOURCES.items():
        if key not in by_source:
            continue
        y, s = map(np.asarray, zip(*by_source[key]))
        metrics = f1_at(y, s, thresholds[key[0]])
        out[name] = {
            "task": key[0],
            "n": len(y),
            "positive": int(y.sum()),
            "ap": float(average_precision_score(y, s)),
            **metrics,
            "best_threshold_f1": best_f1_threshold(y, s)[1],
        }
    return out


def macro(results, field="f1"):
    """Average over sources within a task, then over tasks."""
    per_task = defaultdict(list)
    for value in results.values():
        per_task[value["task"]].append(value[field])
    return float(np.mean([np.mean(v) for v in per_task.values()]))


def table(named_results, fields=("ap", "f1", "fpr", "best_threshold_f1")):
    names = list(TEST_SOURCES.values())
    header = "| Evidence | n (+) | " + " | ".join(
        f"{label} {field}" for label in named_results for field in fields
    )
    lines = [header + " |", "|" + "---|" * (2 + len(fields) * len(named_results))]
    for name in names:
        cells = []
        sizes = None
        for results in named_results.values():
            value = results.get(name)
            if value is None:
                cells += ["—"] * len(fields)
                continue
            sizes = f"{value['n']} ({value['positive']})"
            cells += [f"{value[f]:.3f}" for f in fields]
        lines.append(f"| {name} | {sizes} | " + " | ".join(cells) + " |")
    for field in fields:
        pass
    summary = []
    for label, results in named_results.items():
        summary.append(
            f"{label}: macro AP {macro(results, 'ap'):.3f}, macro F1 {macro(results):.3f}, "
            f"best-threshold macro F1 {macro(results, 'best_threshold_f1'):.3f}"
        )
    return "\n".join(lines) + "\n\n" + "\n".join(summary)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
