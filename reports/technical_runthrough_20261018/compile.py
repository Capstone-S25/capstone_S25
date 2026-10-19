"""Compile this self-contained LaTeX source tree with Tectonic or XeLaTeX."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "model_evaluation_runthrough_20261018"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Install the pinned Linux compiler locally if none is available",
    )
    args = parser.parse_args()
    candidates = [
        HERE / "data/tools/tectonic/tectonic",
        HERE.parents[1] / "data/tools/tectonic/tectonic",
    ]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None and shutil.which("tectonic"):
        binary = Path(shutil.which("tectonic"))
    if binary is None and args.bootstrap:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "tools/bootstrap_report_latex.py"),
                "--prefix",
                str(HERE / "data/tools/tectonic"),
            ],
            check=True,
        )
        binary = candidates[0]
    if binary is not None:
        env = {**os.environ, "XDG_CACHE_HOME": str(binary.parent / "cache")}
        command = [
            str(binary),
            "--untrusted",
            "--keep-logs",
            "--keep-intermediates",
            STEM + ".tex",
        ]
        subprocess.run(command, cwd=HERE, env=env, check=True)
    elif shutil.which("xelatex"):
        for _ in range(3):
            subprocess.run(
                [
                    "xelatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-no-shell-escape",
                    STEM + ".tex",
                ],
                cwd=HERE,
                check=True,
            )
    else:
        raise SystemExit(
            "Install Tectonic/XeLaTeX, or use python compile.py --bootstrap on Linux x86_64."
        )
    print("PDF:", HERE / (STEM + ".pdf"))


if __name__ == "__main__":
    main()
