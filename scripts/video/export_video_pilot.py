"""Export explicit research checkpoint identities and operational limitations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    bundle = {
        "status": "research_only",
        "tasks": ["smoking", "unattended_items"],
        "models": {},
        "source_guard": json.loads((data / "source_guard.json").read_text()),
        "old_results_preserved": True,
        "new_visual_backbone_finetuning": False,
        "detectors": {
            "closed": json.loads((data / "detector.json").read_text()),
            "grounding": json.loads((data / "grounding_all.json").read_text()),
        },
    }
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = run / model
        selection = json.loads((directory / "joint_selection.json").read_text())
        feature = json.loads((data / "features" / model / "identity.json").read_text())
        tasks = {}
        for task, entry in selection["tasks"].items():
            primary = entry["primary"]
            path = directory / task / (primary["name"] + ".pt")
            tasks[task] = {
                "candidate": primary["name"],
                "checkpoint": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "thresholds": primary["thresholds"],
                "selection": primary["selection"],
            }
        bundle["models"][model] = {
            "base_revision": feature["base_revision"],
            "feature_identity": str(
                (data / "features" / model / "identity.json").relative_to(ROOT)
            ),
            "event_policy": str((directory / "event_policies.json").relative_to(ROOT)),
            "tasks": tasks,
        }
    (run / "model_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (run / "model_card.md").write_text("""# Video pilot research model card

New smoking and unattended-item heads use frozen pretrained MobileCLIP2-S2 or
SigLIP2-B/16-224 representations. `model_bundle.json` records exact selected
checkpoints, hashes, base revisions, detector settings and frozen policies.
There is no automatic replacement of the previous five-objective model bundle.

Inputs are RGB scene/automatic person and item crops, with 16 uniform views for
offline HMDB action clips and 2 Hz causal 20-second windows for persistent-item
recordings. Static smoking images have one view. CLIP feature extraction and
head training use float32; no latency claim is made. Person supervision uses
only training-camera source boxes matched to automatic crops.

Global alerting uses development-fitted thresholds and frozen persistence,
release and cooldown. AVSS's thirty-second stationary-item/absent-owner gate is
a separate known-source-criteria diagnostic. Its ownership relation is a
proximity hypothesis, without independent ownership labels or robust appearance
re-identification. Missing initial owner history can suppress a genuine event.

The first pilot improves MobileCLIP CCTV smoking ranking modestly and regresses
SigLIP's selected route there. Unattended score and threshold transfer remain
weak. A state-only rule detects both clean fresh AVSS events without observed
normal false or repeated alerts, but normal exposure is only 2.76 minutes and
median delay is about 35 seconds. This is not deployment validation.

Historical labels and error triage are provisional assistant review. HMDB uses
primary action labels and is now a retrospective adaptation test. Two AVSS
sequences form the primary fresh cohort; a third has a censored annotation end.
Unknown time never supplies normal labels/exposure. Event alert onset matching
is strict, so ambiguous historical transition intervals require independent
review before interpreting misses.

Loitering, littering and public sleep have error triage in this pilot, without
new temporal checkpoints. Housing status is never a visual label.

See [results](report.md), [verification](verification.json), and the repository
[workflow guide](../../docs/video_pilot.md).
""")
    print("RESEARCH BUNDLE EXPORTED", flush=True)


if __name__ == "__main__":
    main()
