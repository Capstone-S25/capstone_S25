"""Assemble disjoint model results from exactly matching accuracy cohorts.

Original inference artifacts remain unchanged. The assembled report is quality
analysis; timings from separate jobs are observational measurements.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from anomaly_eval.analysis import analyze, calibrate, generate, write_report
from anomaly_eval.benchmark import write_comparison, write_json
from anomaly_eval.data import file_sha256, fingerprint
from anomaly_eval.models import DEFAULT_MODELS
from anomaly_eval.paths import portable_path, resolve_path


def combine(sources, output, bootstrap=2000):
    states = [
        json.loads((source / "evaluation.json").read_text()) for source in sources
    ]
    first = states[0]
    keys = [(r["cohort"], r["split"]) for r in first["benchmarks"]]
    for state in states:
        if state["status"] != "complete_for_available_tasks":
            raise ValueError("Every source must be a completed accuracy run")
        if state["dataset_fingerprint"] != first["dataset_fingerprint"]:
            raise ValueError("Source datasets differ")
        if [(r["cohort"], r["split"]) for r in state["benchmarks"]] != keys:
            raise ValueError("Source task/split cohorts differ")

    # Validate all inputs before creating any assembled artifacts.
    cohorts = []
    names = None
    for index, (cohort, split) in enumerate(keys):
        directories = [
            resolve_path(state["benchmarks"][index]["output"], source)
            for source, state in zip(sources, states)
        ]
        runs = [
            json.loads((directory / "run.json").read_text())
            for directory in directories
        ]
        reference = runs[0]
        collected = []
        for run, directory in zip(runs, directories):
            for key in (
                "definitions",
                "sample_identity",
                "samples",
                "code_sha256",
                "metric_version",
                "evaluation_split",
            ):
                if run[key] != reference[key]:
                    raise ValueError(f"Mismatched {key}: {cohort}/{split}")
            for key in (
                "seed",
                "threads",
                "cpu_cores",
                "duty_cycle",
                "max_samples_per_second",
                "repeats",
                "max_side",
                "image_policy",
                "max_new_tokens",
                "device",
                "dtype",
                "batch_size",
                "workers",
            ):
                if run["options"].get(key) != reference["options"].get(key):
                    raise ValueError(
                        f"Mismatched inference setting {key}: {cohort}/{split}"
                    )
            for name in run["models"]:
                summary = json.loads((directory / name / "summary.json").read_text())
                if summary["status"] != "ok":
                    raise ValueError(f"Failed source model: {name}")
                collected.append((name, directory, summary))
        available = [name for name, _, _ in collected]
        if len(available) != len(set(available)):
            raise ValueError("Source runs contain overlapping model names")
        order = [name for name in DEFAULT_MODELS if name in available] + sorted(
            set(available) - set(DEFAULT_MODELS)
        )
        if names is not None and names != order:
            raise ValueError("Model support differs across cohorts")
        names = order
        cohorts.append((cohort, split, reference, runs, collected))

    output.mkdir(parents=True, exist_ok=False)
    state = copy.deepcopy(first)
    state["benchmarks"] = []
    state["options"]["models"] = names
    state["options"]["output"] = str(output)
    state["assembly"] = {
        "kind": "quality_analysis_from_matching_inference_runs",
        "script_sha256": file_sha256(Path(__file__)),
        "sources": [str(source) for source in sources],
        "note": "Independent jobs; resource timings are observational, not a controlled latency ranking.",
    }
    state["steps"] = [step for original in states for step in original["steps"]]
    for cohort, split, reference, runs, collected in cohorts:
        destination = output / cohort / split
        destination.mkdir(parents=True)
        metadata = copy.deepcopy(reference)
        metadata["models"] = names
        metadata["options"]["models"] = names
        metadata["options"]["output"] = portable_path(destination)
        metadata["assembled_source_fingerprints"] = [
            run["experiment_fingerprint"] for run in runs
        ]
        metadata["experiment_fingerprint"] = fingerprint(
            {
                "sources": metadata["assembled_source_fingerprints"],
                "models": names,
                "assembly": state["assembly"],
            }
        )
        write_json(destination / "run.json", metadata)
        summaries = []
        for name in names:
            _, directory, summary = next(item for item in collected if item[0] == name)
            model_output = destination / name
            model_output.mkdir()
            for filename in ("summary.json", "predictions.jsonl", "worker.log"):
                original = directory / name / filename
                if original.exists():
                    shutil.copy2(original, model_output / filename)
            summaries.append(summary)
        write_comparison(destination, summaries)
        analyze(destination, bootstrap, reference["options"]["seed"])
        generate(destination, destination / "figures")
        selection = reference["selection"]
        state["benchmarks"].append(
            {
                "cohort": cohort,
                "split": split,
                "output": portable_path(destination, output),
                "selection": selection,
            }
        )
    for cohort in sorted({name for name, _ in keys}):
        if (cohort, "dev") in keys and (cohort, "test") in keys:
            calibrate(
                output / cohort / "dev",
                output / cohort / "test",
                first["options"]["target_fpr"],
            )
    if first.get("vadclip", {}).get("status") == "ok":
        shutil.copytree(sources[0] / "vadclip", output / "vadclip")
    state["experiment_fingerprint"] = fingerprint(state["assembly"])
    write_json(output / "evaluation.json", state)
    write_report(output, state, argparse.Namespace(**state["options"]))
    report = output / "report.md"
    report.write_text(
        report.read_text().replace(
            "# Corrected evaluation run\n",
            "# Combined GPU accuracy comparison\n\n" + state["assembly"]["note"] + "\n",
            1,
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    result = combine(
        [source.resolve() for source in args.sources],
        args.output.resolve(),
        args.bootstrap,
    )
    print(f"Report: {result / 'report.md'}")


if __name__ == "__main__":
    main()
