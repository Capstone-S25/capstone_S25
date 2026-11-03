"""One resident backend per process, JSON lines on stdin/stdout; diagnostics on stderr."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import logging
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["clip", "gemma"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention", choices=["sdpa", "math", "eager"], default="sdpa"
    )
    parser.add_argument(
        "--operating-point",
        choices=["raw", "dev_max_f1", "dev_fpr_budget"],
        default="dev_max_f1",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-format", choices=["auto", "prompt"], default="auto")
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    protocol = sys.stdout

    def send(value):
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    with contextlib.redirect_stdout(sys.stderr):
        try:
            from .backends import ClipStrategy, GemmaGGUF, GemmaReasoner

            started = time.perf_counter()
            backend = (
                ClipStrategy
                if args.backend == "clip"
                else GemmaGGUF
                if args.model in {"gemma4-26b-a4b", "gemma4-31b"}
                else GemmaReasoner
            )
            engine = backend(**vars(args))
            versions = {"python": sys.version.split()[0]}
            for package in [
                "torch",
                "torchvision",
                "transformers",
                "numpy",
                "Pillow",
                "open_clip_torch",
            ]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            send(
                {
                    "type": "ready",
                    "metadata": engine.metadata,
                    "load_seconds": time.perf_counter() - started,
                    "runtime_versions": versions,
                }
            )
        except Exception as exc:
            logger.exception("Resident model request/startup failed")
            send({"type": "startup_error", "error": str(exc)})
            return 1

        def stop(_signal, _frame):
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, stop)
        try:
            for line in sys.stdin:
                ident = None
                try:
                    request = json.loads(line)
                    ident = request["id"]
                    if request.get("method") == "close":
                        break
                    value = engine.classify(request["candidate"])
                    send({"id": ident, "result": value})
                except Exception as exc:
                    logger.exception("Resident model request/startup failed")
                    send(
                        {
                            "id": ident,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
        finally:
            close = getattr(engine, "close", None)
            if close is not None:
                close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
