"""How good are Gemma's confirmation verdicts, as the application asks for them?

Sends windows through the runtime's own GemmaClient (JSON verdict plus scene
summary) and reports precision and recall of "anomaly" per source, for the
strict (gatekeeper) and balanced (second-opinion) prompt wordings.

    python gemma_confirm_eval.py --split dev  --styles strict balanced
    python gemma_confirm_eval.py --split test --styles balanced

Needs the integration worktree on the path (for vision_runtime) and a running
OpenAI-compatible Gemma server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
from common import OUT, ROOT, TEST_SOURCES, load_rows

sys.path.insert(0, str(ROOT / ".local/worktrees/vision-app-integration"))
from vision_runtime.gemma import GemmaClient

PER_CELL = {"dev": 25, "test": 40}


def sample(rows, split):
    random.seed(0)
    cells = defaultdict(list)
    for r in rows:
        held = (
            r["split"] in {"test", "external"}
            and (r["task"], r["source"]) in TEST_SOURCES
        )
        pool = r["split"] in {"train", "dev"}
        if (split == "test" and held) or (split == "dev" and pool):
            cells[r["task"], r["source"], r["target"]].append(r)
    chosen = []
    for key, group in sorted(cells.items()):
        random.shuffle(group)
        chosen += group[: PER_CELL[split]]
    # The runtime rejects a window that repeats a frame (clipped at a video's start).
    return [r for r in chosen if len(set(r["views"])) == len(r["views"])]


def candidate(row):
    views = row["views"]
    span = 4.0 if row["task"] == "littering" else 20.0
    times = (
        np.linspace(0, span, len(views)).round(2).tolist() if len(views) > 1 else [0.0]
    )
    view = {"paths": [str(ROOT / p) for p in views], "times_s": times}
    return {
        "event_id": row["id"],
        "timestamp_s": times[-1],
        "tasks": [row["task"]],
        "candidate_origin": "scheduled_scene_review",
        "views": {"short": view, "long": view},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--styles", nargs="+", default=["strict", "balanced"])
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", default="gemma4-e4b")
    args = parser.parse_args()
    rows = sample(load_rows(), args.split)
    target = OUT / "gemma_video" / f"confirm_{args.split}.json"
    saved = json.loads(target.read_text()) if target.exists() else {}
    client = GemmaClient(args.url, args.model, timeout=600)
    for style in args.styles:
        os.environ["GEMMA_PROMPT_STYLE"] = style
        done = saved.setdefault(style, {})
        started = time.perf_counter()
        for n, row in enumerate(rows):
            if row["id"] in done:
                continue
            out = client.classify(candidate(row))
            value = out["predictions"][row["task"]]
            done[row["id"]] = {
                "decision": value.get("model_decision_before_guard", value["decision"]),
                "evidence": value.get("evidence"),
                "summary": out["summary"],
                "seconds": out["seconds"],
            }
            if n % 50 == 0:
                print(
                    style,
                    n,
                    len(rows),
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )
                target.write_text(json.dumps(saved))
        target.write_text(json.dumps(saved))
    print(f"\n{args.split}: {len(rows)} windows")
    print(
        "| Source | n (pos.) | "
        + " | ".join(f"{s}: P / R / F1 (unknown)" for s in args.styles)
        + " |"
    )
    print("|---|---|" + "---|" * len(args.styles))
    by = defaultdict(list)
    for r in rows:
        by[r["task"], r["source"]].append(r)
    totals = {s: defaultdict(list) for s in args.styles}
    for (task, source), group in sorted(by.items()):
        cells = []
        for style in args.styles:
            d = [saved[style][r["id"]]["decision"] for r in group]
            y = np.array([r["target"] for r in group])
            pred = np.array([x == "anomaly" for x in d])
            tp, fp, fn = (
                (pred & (y == 1)).sum(),
                (pred & (y == 0)).sum(),
                (~pred & (y == 1)).sum(),
            )
            p, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            unknown = np.mean([x == "unknown" for x in d])
            cells.append(f"{p:.2f} / {rec:.2f} / {f1:.2f} ({unknown:.0%})")
            totals[style][task].append(f1)
        print(
            f"| {task}: {source} | {len(group)} ({int(sum(r['target'] for r in group))}) | "
            + " | ".join(cells)
            + " |"
        )
    for style in args.styles:
        macro = np.mean([np.mean(v) for v in totals[style].values()])
        print(f"{style}: macro F1 {macro:.3f}")


if __name__ == "__main__":
    main()
