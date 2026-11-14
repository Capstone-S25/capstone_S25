"""Calibrate one camera's thresholds on its own ordinary footage.

    python tools/calibrate_site.py --camera camera_01
    python tools/calibrate_site.py --camera camera_01 --journal runtime/cameras/camera_01/<session>/events.jsonl
    python tools/calibrate_site.py --camera camera_01 --clear

Reads the screening scores the camera backend already journaled (every window
it scored, flagged or not) and raises that camera's threshold per behaviour
until at most `--share` of those windows would be flagged. Use it on footage
in which the behaviour did NOT happen: a real incident in the calibration
footage would be learnt as ordinary. No model is loaded and nothing is sent
anywhere. Restart the camera backend afterwards.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision_runtime import site  # noqa: E402
from vision_runtime.classifier import ENSEMBLES, default_classifier  # noqa: E402
from vision_runtime.config import configure_runtime, models_root, runtime_root  # noqa: E402
from vision_runtime.ensemble import operating_point  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="The camera's --source-id")
    parser.add_argument(
        "--journal",
        type=Path,
        nargs="*",
        help="events.jsonl files to use; default: every session of this camera",
    )
    parser.add_argument(
        "--share",
        type=float,
        default=0.02,
        help="Share of the ordinary windows that may still be flagged (default 2%%)",
    )
    parser.add_argument(
        "--minimum", type=int, default=30, help="Windows needed per behaviour"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Remove this camera's calibration"
    )
    args = parser.parse_args()
    configure_runtime()
    if args.clear:
        removed = site.clear(args.camera)
        print("Removed." if removed else "This camera had no calibration.")
        return
    model = default_classifier()
    if model not in ENSEMBLES:
        raise SystemExit("Site calibration needs an ensemble classifier")
    bundle = json.loads((models_root() / model / "ensemble.json").read_text())
    point = operating_point(bundle, bundle["tasks"])
    thresholds = {t: float(v["threshold"]) for t, v in bundle["tasks"].items()}
    files = args.journal or sorted(
        (runtime_root() / "cameras" / args.camera).glob("*/events.jsonl")
    )
    if not files:
        raise SystemExit(
            f"No event journal for {args.camera}. Run the camera backend on "
            "ordinary footage first, or pass --journal."
        )
    scores = site.journal_scores(files, model)
    tasks = site.calibrate(scores, thresholds, args.share, args.minimum)
    print(
        f"{args.camera}: {len(files)} journal file(s), model {model}"
        + (f", {point}" if point else "")
    )
    for task in thresholds:
        count = len(scores.get(task, []))
        if task not in tasks:
            print(f"  {task:17s} {count:5d} windows: too few, left unchanged")
            continue
        value = tasks[task]
        change = (
            f"threshold {value['model_threshold']:.2f} -> {value['threshold']:.2f}"
            if value["threshold"] > value["model_threshold"]
            else "threshold unchanged"
        )
        print(
            f"  {task:17s} {count:5d} windows: flagged {value['flagged_before']:.1%} "
            f"-> {value['flagged_after']:.1%}, {change}"
        )
    if not tasks:
        raise SystemExit("Nothing to calibrate: not enough scored windows.")
    path = site.save(args.camera, model, point, tasks, args.share)
    print(f"Saved to {path}. Restart the camera backend to apply it.")


if __name__ == "__main__":
    main()
