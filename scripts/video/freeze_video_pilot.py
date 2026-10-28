"""Freeze the combined development-only selection before opening fresh scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.video.train_video_pilot import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        output = args.run / model
        if (output / "joint_selection.json").exists():
            raise ValueError("Final selection already frozen")
        joint = json.loads((output / "selection.json").read_text())
        for task, name in [
            ("smoking", "person_selection"),
            ("unattended_items", "grounding_selection"),
        ]:
            extra = json.loads((output / (name + ".json")).read_text())
            joint["tasks"][task]["candidates"].extend(extra["candidates"])
            joint["tasks"][task]["primary"] = max(
                joint["tasks"][task]["candidates"],
                key=lambda c: (
                    c["selection"]["worst_source_ap"],
                    c["selection"]["mean_source_ap"],
                ),
            )
        joint["pre_score_amendments"] = [
            "training person supervision and training-only negative mining",
            "open-vocabulary causal geometry; CLIP visual features fixed",
        ]
        save(output / "joint_selection.json", joint)
        print(
            "FINAL SELECTION FROZEN",
            model,
            {k: v["primary"]["name"] for k, v in joint["tasks"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
