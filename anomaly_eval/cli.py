"""One CLI for CPU anomaly-model experiments and dataset utilities."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .paths import ROOT_FIELDS, configure

COMMANDS = {
    "predict-clip": (
        "anomaly_eval.training.predict",
        "main",
        "Classify images with the development-selected trained CLIP bundle",
    ),
    "train-clip": (
        "anomaly_eval.training.clip",
        "main",
        "Train source-grouped CLIP probes and last-stage visual adapters",
    ),
    "run": (
        "anomaly_eval.runner",
        "main",
        "Prepare, benchmark, calibrate and report available tasks",
    ),
    "fetch": (
        "anomaly_eval.datasets.fetch",
        "main",
        "Fetch ABODA/CAVIAR/D2-CDNet/Roboflow sources",
    ),
    "fetch-drive": (
        "anomaly_eval.datasets.drive",
        "main",
        "Fetch the original Drive corpus with resume and pacing",
    ),
    "prepare": (
        "anomaly_eval.datasets.prepare",
        "main",
        "Build reviewed task manifests and coverage reports",
    ),
    "benchmark": (
        "anomaly_eval.benchmark",
        "main",
        "Run isolated CPU model workers on a manifest",
    ),
    "analyze": (
        "anomaly_eval.analysis",
        "main",
        "Bootstrap, calibrate and plot saved predictions",
    ),
    "inventory": (
        "anomaly_eval.datasets.tools",
        "inventory_main",
        "Inventory images or sample videos for label review",
    ),
    "review": (
        "anomaly_eval.datasets.tools",
        "review_main",
        "Extract bounded ABODA/D2 review candidates",
    ),
    "export": (
        "anomaly_eval.datasets.tools",
        "export_main",
        "Export a portable data bundle from a recorded run",
    ),
    "audit": (
        "anomaly_eval.audit",
        "main",
        "Check bundle integrity and source diversity",
    ),
}
VAD_STAGES = {
    "extract": "extract_features",
    "train": "train",
    "evaluate": "evaluate",
    "baselines": "baselines",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        epilog="Global workspace flags precede the command. COMMAND --help shows command options.",
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show command help")
    parser.add_argument(
        "--workspace-config",
        type=Path,
        help="Server JSON; relative locations resolve against that file",
    )
    for token, field in ROOT_FIELDS.items():
        parser.add_argument(
            "--" + token.lower().replace("_", "-"), dest=field, type=Path
        )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD_ROOT=NEW_ROOT",
        help="Explicitly relocate legacy absolute paths",
    )
    parser.add_argument("--version", action="version", version="anomaly-eval 0.1.0")
    parser.add_argument(
        "command",
        choices=[*COMMANDS, "vadclip", "paths"],
        nargs="?",
        help="Operation to perform",
    )
    args, rest = parser.parse_known_args(argv)
    if args.help and args.command:
        rest.append("--help")
    try:
        workspace = configure(
            args.workspace_config,
            {field: getattr(args, field) for field in ROOT_FIELDS.values()},
            args.path_map,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "paths":
        import json

        print(
            json.dumps(
                {**workspace.as_dict(), "model_cache": str(workspace.model_cache)},
                indent=2,
            )
        )
        return 0
    if args.command == "vadclip":
        if not rest or rest[0] in {"-h", "--help"}:
            print(
                "Usage: anomaly-eval vadclip {extract,train,evaluate,baselines} [options]"
            )
            return 0
        stage = rest.pop(0)
        if stage not in VAD_STAGES:
            parser.error("Unknown VadCLIP stage")
        module = "anomaly_eval.vadclip." + VAD_STAGES[stage]
        function = "main"
    else:
        module, function, _ = COMMANDS[args.command]
    # Scientific/model dependencies are loaded only by the selected operation.
    try:
        return getattr(importlib.import_module(module), function)(rest) or 0
    except PermissionError as exc:
        print(
            f"anomaly-eval: permission denied: {exc.filename or str(exc)}\n"
            "Choose paths your account can access with --data-root, --results-root "
            "and --cache-root; results and cache directories must be writable.\n"
            "For a read-only dataset, use run --prepared-dir with a writable directory.",
            file=sys.stderr,
        )
        return 1
