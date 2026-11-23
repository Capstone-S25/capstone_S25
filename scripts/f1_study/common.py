"""Shared data, metrics and reporting for the November F1 study.

Every model is scored on the same held-out sources as the October strategy
study. Thresholds come from training/development data only; the "best
threshold" column is a diagnostic that shows how much F1 is lost to threshold
transfer, never a selection rule.
"""

from __future__ import annotations

import json
import os
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


def detector_half(row, by="video"):
    """Which smoking-detector fold a fit-side new-site row trains: the c4
    detector takes half "a", the c2 detector half "b", so each row can be
    scored by a detector that has not seen it.

    by="video" (the first round-3 detectors, suffix f) splits a camera's
    recordings between the halves, so the other detector has still seen the
    camera and its scores on those rows are too low to calibrate a threshold.
    by="camera" (suffix g) keeps each camera, room subject or cluster whole.
    """
    import hashlib

    unit = (row.get("video") if by == "video" else None) or row["group_id"]
    digest = hashlib.sha1(f"detector|{row['source']}|{unit}".encode()).hexdigest()
    return "a" if int(digest, 16) % 2 == 0 else "b"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


# F1_SPLIT=ian: the team's fit / test split (notebooks/exp/eval/splits.json on
# feat/vision-app-integration), held out by camera or scene. Its test side
# must never be fitted: ICVL cameras 02 and 10 move to the shift side for
# every ICVL task (and 05, 08, 11 join the fit), and fitting rows from its
# ABODA test scenes are dropped. Benchmark test rows stay, since they are
# only ever scored.
IAN_TEST_ICVL = {"02", "10"}
# The split treats each ABODA video as its own scene, but they share cameras:
# walkway 2-4, entrance 5-6, classroom 7-10 (lobby 1 stands alone). Its test
# side holds 2, 3, 5, 7, 8 and 10, so every video on those three cameras is
# kept out of fitting, or the "unseen camera" would have been seen.
IAN_TEST_ABODA = {f"video{n}" for n in range(2, 11)}
# F1_FIT_ENTRANCE=1: the team drops ABODA video5 from its test side, so the
# entrance camera (videos 5 and 6, already in the October dev pool) is fitted.
if os.environ.get("F1_FIT_ENTRANCE"):
    IAN_TEST_ABODA -= {"video5", "video6"}
# Round 4 kept every D2-CDNet video for scoring, but the split puts "sofa"
# (its own indoor scene, four left items) on the fit side.
IAN_FIT_SCENES = {("d2cdnet", "sofa")}


def ian_site(row):
    if (row["source"], row.get("camera")) in IAN_FIT_SCENES:
        return "fit"
    if not row["source"].startswith("icvl_"):
        return row["site"]
    return "shift" if row["camera"] in IAN_TEST_ICVL else "fit"


def in_ian_test(row):
    """True for a fitting row whose footage is on the team split's test side."""
    if row.get("split") not in {"train", "dev"}:
        return False
    import re

    text = row.get("source_id", "") + " " + " ".join(row.get("views", [])[:1])
    if "aboda" not in text.lower():
        return False
    video = re.search(r"(video\d+)\b", text)
    return bool(video) and video.group(1) in IAN_TEST_ABODA


def load_rows():
    """All train/dev rows and every held-out row, tagged with `split`."""
    rows = list(
        read_jsonl(DATA / "samples.jsonl")
    )  # train, dev and the photo holdout (split=test)
    for row in read_jsonl(DATA / "historical.jsonl"):
        rows.append({**row, "split": "test"})
    for name in ["hmdb_screened", "meva_screened", "rough_screened"]:
        rows.extend(read_jsonl(DATA / f"{name}.jsonl"))
    if os.environ.get("F1_ROUND3"):
        # New-site rows (round3_sources.py). Their split is "site_fit" or
        # "site_shift", so the earlier protocols ignore them.
        for row in read_jsonl(OUT / "round3/rows.jsonl"):
            rows.append({**row, "split": "site_" + row["site"]})
    if os.environ.get("F1_ROUND3") and os.environ.get("F1_ROUND4"):
        # The Drive's new labelled videos (round4_sources.py), same splits.
        for row in read_jsonl(OUT / "round4/rows.jsonl"):
            if os.environ.get("F1_SPLIT") == "ian":
                row = {**row, "site": ian_site(row)}
            rows.append({**row, "split": "site_" + row["site"]})
    if os.environ.get("F1_SPLIT") == "ian":
        rows = [r for r in rows if not in_ian_test(r)]
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
