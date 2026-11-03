"""Inputs, development selection and immutable definitions for Gemma 4 tests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.training.data import TASKS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/gemma4_evaluation_20261027"
SOURCES = {
    "main": ROOT / "results/smolvlm_finetuning_20261019/samples.jsonl",
    "additional_transfer": ROOT
    / "results/smolvlm_finetuning_20261019/additional_transfer/samples.jsonl",
    "xrf55": ROOT / "data/generalization_pilot_20261020/xrf55/samples.jsonl",
}
MODELS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "gemma4-12b": "google/gemma-4-12B-it",
    "gemma4-26b-a4b": "google/gemma-4-26B-A4B-it",
    "gemma4-31b": "google/gemma-4-31B-it",
}


@dataclass(frozen=True)
class Policy:
    name: str
    prompt: str = "definition"
    image_tokens: int = 280
    layout: str = "native"
    max_views: int = 8


POLICIES = (
    Policy("definition_native_280"),
    Policy("evidence_native_280", "evidence"),
    Policy("evidence_native_560", "evidence", 560),
    Policy("evidence_sheet_1120", "evidence", 1120, "sheet"),
)
BASELINE = POLICIES[0]
CONFUSERS = {
    "smoking": "Distinguish smoking from drinking, eating, coughing and using a phone. Hand-to-mouth motion alone is insufficient evidence of a smoking object.",
    "littering": "Look for a person releasing or discarding rubbish and the rubbish remaining on the ground. Distinguish disposal from carrying, picking up, bin use and benign placement; rubbish already on the ground alone does not prove disposal.",
    "loitering": "Use the ordered observations to assess whether the same person remains in the public area. Distinguish a brief pause, purposeful walking and ordinary activity from sustained remaining. Do not invent elapsed duration or stable identity.",
    "rough_sleeping": "Look for apparent sleep or a lying person with bedding in a public setting. Distinguish awake resting, walking, empty bedding and indoor sleep. Do not infer housing status or a medical condition.",
    "unattended_items": "Look for an item left behind, its continued presence and an associated person moving away or absent. Distinguish carrying, attended placement and retrieval. An isolated bag or a person outside a crop does not establish ownership or abandonment.",
}


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def view_indices(row, maximum=8):
    count = len(row["views"])
    if not count:
        raise ValueError("An example must contain at least one image")
    return np.linspace(0, count - 1, min(count, maximum)).round().astype(int).tolist()


def question(row, policy):
    """Never expose source names, labels, review notes or split to the model."""
    definitions = json.loads((ROOT / "anomaly_eval/config/anomalies.json").read_text())
    indices = view_indices(row, policy.max_views)
    context = (
        "This image shows one moment."
        if len(indices) == 1
        else "These images show one scene or clip, ordered from earliest to latest."
    )
    if policy.layout == "sheet" and len(indices) > 1:
        context = "The numbered panels show one scene or clip in chronological reading order, from left to right and then top to bottom."
    if row.get("timestamps") and len(row["timestamps"]) == len(row["views"]):
        context += (
            " Frame timestamps in seconds: "
            + ", ".join(f"{row['timestamps'][i]:g}" for i in indices)
            + "."
        )
    elif len(indices) > 1:
        context += " Exact elapsed duration is not supplied."
    prompt = f"{context} Target: {row['task'].replace('_', ' ')}. Definition: {definitions[row['task']]} "
    if policy.prompt == "evidence":
        prompt += CONFUSERS[row["task"]] + " "
    elif policy.prompt != "definition":
        raise ValueError("Unknown prompt policy")
    return (
        prompt
        + "Use only the provided visual evidence; do not invent duration or ownership. Does this evidence match the target? Answer Yes or No."
    )


def development_screen(rows, per_task=64):
    """Round-robin classes/sources/groups; selection receives development only."""
    if any(row["cohort"] != "main" or row["split"] != "dev" for row in rows):
        raise ValueError("Screening must use primary development rows only")
    selected = []
    for task in TASKS:
        cells = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["task"] == task:
                cells[row["source"], row["target"]][row["group_id"]].append(row)
        queues = []
        for key in sorted(cells):
            groups = cells[key]
            queue = []
            while any(groups.values()):
                for group in sorted(groups):
                    if groups[group]:
                        groups[group].sort(
                            key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
                        )
                        queue.append(groups[group].pop())
            queues.append(queue)
        count = 0
        while any(queues) and count < per_task:
            for queue in queues:
                if queue and count < per_task:
                    selected.append(queue.pop(0))
                    count += 1
    return sorted(selected, key=lambda row: row["id"])


def source_ap(rows, scores, task):
    values = []
    for source in sorted({r["source"] for r in rows if r["task"] == task}):
        members = [r for r in rows if r["task"] == task and r["source"] == source]
        labels = [r["target"] for r in members]
        if set(labels) == {0, 1}:
            values.append(
                float(
                    average_precision_score(labels, [scores[r["id"]] for r in members])
                )
            )
    if not values:
        raise ValueError(f"No two-class development source for {task}")
    return float(np.mean(values))


def choose_routes(dev, predictions):
    if any(r["split"] != "dev" or r["cohort"] != "main" for r in dev):
        raise ValueError(
            "Route selection rejects test, external and source-development data"
        )
    routes = {}
    table = {}
    for task in TASKS:
        table[task] = {
            p.name: source_ap(dev, predictions[p.name], task) for p in POLICIES
        }
        # Prefer the simpler predeclared baseline on a tie.
        routes[task] = max(
            POLICIES, key=lambda p: (table[task][p.name], -POLICIES.index(p))
        ).name
    return {
        "routes": routes,
        "development_source_ap": table,
        "selection_uses_test": False,
    }


def prepare(output, verify=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acquired = {name: read_rows(path) for name, path in SOURCES.items()}
    identifiers = set()
    groups = {}
    pixels = {}
    records = []
    for cohort, examples in acquired.items():
        for row in examples:
            partition = row["split"]
            if row["task"] not in TASKS or row["target"] not in (0, 1):
                raise ValueError("Require one known binary target per sample")
            if row["group_id"] in groups and groups[row["group_id"]] != partition:
                raise ValueError("Source group crosses inherited splits")
            groups[row["group_id"]] = partition
            for pixel in row["pixel_hashes"]:
                if pixel in pixels and pixels[pixel] != partition:
                    raise ValueError("Decoded pixels cross inherited splits")
                pixels[pixel] = partition
            if partition == "train":
                continue
            if row["id"] in identifiers:
                raise ValueError("Duplicate evaluation identity")
            identifiers.add(row["id"])
            records.append({**row, "cohort": cohort})
    records.sort(key=lambda row: row["id"])
    manifest = output / "samples.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError("Refuse to change the frozen evaluation manifest")
    if verify:
        from concurrent.futures import ThreadPoolExecutor

        from anomaly_eval.data import image_pixel_sha256

        expected = {
            view: pixel
            for row in records
            for view, pixel in zip(row["views"], row["pixel_hashes"])
        }
        if any(len(row["views"]) != len(row["pixel_hashes"]) for row in records):
            raise ValueError("Incomplete image fingerprints")

        def one(item):
            path, pixel = item
            if image_pixel_sha256(ROOT / path) != pixel:
                raise ValueError("Input image changed: " + path)

        with ThreadPoolExecutor(12) as pool:
            list(pool.map(one, expected.items()))
    manifest.write_text(content)
    screen = development_screen(
        [r for r in records if r["cohort"] == "main" and r["split"] == "dev"]
    )
    protocol = {
        "schema": 1,
        "status": "frozen_before_gemma_scores",
        "manifest_sha256": digest(manifest),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in SOURCES.values()},
        "definitions_sha256": digest(ROOT / "anomaly_eval/config/anomalies.json"),
        "models": MODELS,
        "policies": [asdict(p) for p in POLICIES],
        "baseline": BASELINE.name,
        "development_screen_ids": [r["id"] for r in screen],
        "development_screen_cases": len(screen),
        "counts": dict(
            Counter(
                f"{r['cohort']}/{r['split']}/{r['task']}/{r['source']}" for r in records
            )
        ),
        "cases": len(records),
        "unique_images": len({p for r in records for p in r["views"]}),
        "scoring": "First final-answer Yes-minus-No logit margin, thinking disabled, token probability mass retained; no synthetic confidence or unknown-class negatives",
        "selection": "Per-task mean source AP on fixed primary development screen; full development data calibrate selected and baseline routes before any held-out scoring",
        "operating_points": "Raw0, dev maximumF1, pooled dev FPR<=5%, and every-primary-source dev FPR<=5%; save all transfer failures",
        "xrf55": "Source-dev recalibration is secondary; never chooses the primary route. Its source test remains held out until both calibrations are frozen.",
        "precision": "BF16 native Transformers for E2B/E4B/12B; official Google QAT Q4_0 GGUF for 26B/31B owing to32GBGPU. Report backend and precision explicitly; do not attribute all differences to size.",
        "partial_labels": "Only the annotated objective is supervised/evaluated; other anomalies are unlabelled, not automatically negative",
        "qualification": "Retrospective source-held-out historical cohorts; weak XRF smoking gesture labels; provisional sleep/duration/ownership labels; MEVA abandonment-action proxy; no deployment guarantee",
        "no_latency_testing": True,
        "no_model_training": True,
        "code_sha256": digest(Path(__file__)),
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol is already frozen")
    atomic_json(path, protocol)
    return protocol
