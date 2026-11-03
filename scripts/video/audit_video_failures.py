"""Export reproducible, source-balanced retrospective error-review boards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/strategy_study_20261016"
RUN = ROOT / "results/strategy_study_20261016"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in [
        "samples",
        "historical",
        "hmdb_screened",
        "meva_screened",
        "rough_screened",
    ]:
        for row in map(json.loads, (DATA / (name + ".jsonl")).read_text().splitlines()):
            manifest[row["id"]] = row
    queue, summaries = [], []
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = RUN / model / "transfer_evaluation"
        policies = json.loads((directory / "frozen_policies.json").read_text())
        groups = defaultdict(list)
        for tag in ["historical", "hmdb", "meva", "rough", "photo"]:
            for row in map(
                json.loads,
                (directory / (tag + "_predictions.jsonl")).read_text().splitlines(),
            ):
                if row["family"] == "selected":
                    groups[row["task"], row["source"]].append(row)
        for (task, source), rows in groups.items():
            threshold = policies[task]["selected"]["thresholds"]["dev_fpr_budget"]
            for target in [0, 1]:
                values = [r["score"] for r in rows if r["target"] == target]
                summaries.append(
                    {
                        "model": model,
                        "task": task,
                        "source": source,
                        "target": target,
                        "n": len(values),
                        "threshold": threshold,
                        "score_quantiles": np.quantile(
                            values, [0, 0.25, 0.5, 0.75, 1]
                        ).tolist(),
                        "errors": sum((s >= threshold) != bool(target) for s in values),
                    }
                )
            errors = defaultdict(list)
            for row in rows:
                if (row["score"] >= threshold) != bool(row["target"]):
                    errors[
                        "false_negative" if row["target"] else "false_positive"
                    ].append(row)
            for kind, candidates in errors.items():
                # Round-robin across recording groups, then score severity.
                ordered = sorted(
                    candidates,
                    key=lambda r: r["score"],
                    reverse=kind == "false_positive",
                )
                seen, chosen = set(), []
                for row in ordered:
                    if row["group_id"] not in seen:
                        chosen.append(row)
                        seen.add(row["group_id"])
                    if len(chosen) == 4:
                        break
                for row in ordered:
                    if len(chosen) == min(4, len(ordered)):
                        break
                    if row not in chosen:
                        chosen.append(row)
                for row in chosen:
                    original = manifest[row["id"]]
                    queue.append(
                        {
                            **row,
                            "model": model,
                            "error": kind,
                            "threshold": threshold,
                            "views": original["views"],
                            "evidence": original["evidence"],
                            "review_status": "pending",
                            "reviewer": None,
                            "categories": [],
                            "notes": "",
                        }
                    )
    (args.output / "review_queue.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in queue)
    )
    (args.output / "source_score_audit.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    categories = [
        "label_ambiguous",
        "target_not_resolvable",
        "region_missed",
        "temporal_evidence_missing",
        "normal_action_confusion",
        "scene_score_shift",
        "uncertain",
    ]
    (args.output / "review_policy.json").write_text(
        json.dumps(
            {
                "purpose": "Retrospective diagnostic selection; never final evaluation",
                "sampling": "up to four errors per source/task/type/model; recording-diverse first, severity next",
                "categories": categories,
                "independent_human_verification": False,
                "rule": "Do not infer error causes from scores alone; retain uncertain cases",
            },
            indent=2,
        )
        + "\n"
    )
    for task in sorted({r["task"] for r in queue}):
        rows = [r for r in queue if r["task"] == task]
        for page in range(0, len(rows), 12):
            selected = rows[page : page + 12]
            board = Image.new("RGB", (1200, len(selected) * 150), "white")
            draw = ImageDraw.Draw(board)
            for i, row in enumerate(selected):
                paths = row["views"]
                indices = sorted({0, len(paths) // 2, len(paths) - 1})
                for j, index in enumerate(indices):
                    with Image.open(ROOT / paths[index]) as im:
                        im = ImageOps.contain(im.convert("RGB"), (220, 122))
                        board.paste(im, (j * 230, i * 150 + 24))
                draw.text(
                    (0, i * 150 + 4),
                    f"{page + i}: {row['model']} {row['source']} {row['error']}",
                    fill="black",
                )
                draw.text(
                    (700, i * 150 + 40),
                    f"label={row['target']} score={row['score']:.3f} threshold={row['threshold']:.3f}",
                    fill="black",
                )
                draw.text((700, i * 150 + 65), row["id"][:65], fill="black")
            board.save(args.output / f"{task}_{page // 12:02d}.jpg", quality=92)
    print("ERROR AUDIT EXPORTED", len(queue), "review cases", flush=True)


if __name__ == "__main__":
    main()
