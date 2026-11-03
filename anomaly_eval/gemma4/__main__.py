"""One entry point for Gemma acquisition, evaluation and optimization."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "fetch": ("fetch", "main"),
    "fetch-runtime": ("fetch", "runtime_main"),
    "fetch-parallel": ("download", "main"),
    "preflight": ("preflight", "main"),
    "suite": ("suite", "main"),
    "evaluate": ("study", "main"),
    "report": ("report", "main"),
    "verify": ("verify", "main"),
    "optimize": ("optimization", "main"),
    "uncertainty": ("uncertainty", "main"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            __doc__
            + "\nUsage: anomaly-eval gemma4 {"
            + ",".join(COMMANDS)
            + "} [options]"
        )
        return 0
    command = args.pop(0)
    if command not in COMMANDS:
        raise SystemExit("Unknown Gemma command: " + command)
    module, function = COMMANDS[command]
    return (
        getattr(importlib.import_module("anomaly_eval.gemma4." + module), function)(
            args
        )
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
