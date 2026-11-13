"""Summarize gemma_video.py: selected frames against denser video sampling."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from common import OUT, best_f1_threshold, f1_at, save_json
from sklearn.metrics import average_precision_score, roc_auc_score

NAMES = {
    "hmdb51_action_transfer": "Smoking: HMDB clips",
    "mivia_iwdd": "Littering: IWDD test",
    "unattended_reviewed": "Unattended: held-out cameras",
    "meva_abandon_action_transfer": "Unattended: MEVA windows",
}


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    return {
        "n": len(y),
        "positive": int(y.sum()),
        "ap": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        # The label-free operating point: the model answers "Yes".
        "f1_yes": f1_at(y, s, 0.0)["f1"],
        "recall_yes": f1_at(y, s, 0.0)["recall"],
        "fpr_yes": f1_at(y, s, 0.0)["fpr"],
        "best_f1": best_f1_threshold(y, s)[1],
    }


def macro(per_source, field):
    by_task = defaultdict(list)
    for name, value in per_source.items():
        by_task[name.split(":")[0]].append(value[field])
    return float(np.mean([np.mean(v) for v in by_task.values()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="e4b")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    folder = OUT / "gemma_video"
    rows = [
        json.loads(line)
        for line in (folder / "manifest.jsonl").read_text().splitlines()
    ]
    scores = json.loads((folder / f"scores_{args.tag}.json").read_text())
    done = [c for c, v in scores.items() if len(v["scores"]) == len(rows)]
    report = {}
    for condition in done:
        per_source = {}
        for source, name in NAMES.items():
            part = [r for r in rows if r["source"] == source]
            per_source[name] = metrics(
                [r["target"] for r in part],
                [scores[condition]["scores"][r["id"]] for r in part],
            )
        frames = [
            len(r["selected"])
            if condition.startswith("selected")
            else int(condition.split("@")[0].removeprefix("dense"))
            for r in rows
        ]
        tokens = int(condition.split("@")[1])
        report[condition] = {
            "sources": per_source,
            "macro_ap": macro(per_source, "ap"),
            "macro_f1_yes": macro(per_source, "f1_yes"),
            "macro_best_f1": macro(per_source, "best_f1"),
            "mean_frames": float(np.mean(frames)),
            "mean_visual_tokens": float(np.mean(frames)) * tokens,
            "seconds": float(np.mean(list(scores[condition]["seconds"].values()))),
        }
    # Paired bootstrap over recording groups: macro AP against the selected frames.
    rng = np.random.default_rng(0)
    groups = sorted({r["group_id"] for r in rows})
    index = defaultdict(list)
    for i, r in enumerate(rows):
        index[r["group_id"]].append(i)

    def macro_ap(condition, idx):
        per = {}
        for source, name in NAMES.items():
            part = [i for i in idx if rows[i]["source"] == source]
            y = [rows[i]["target"] for i in part]
            if len(set(y)) < 2:
                return None
            per[name] = {
                "ap": average_precision_score(
                    y, [scores[condition]["scores"][rows[i]["id"]] for i in part]
                )
            }
        return macro(per, "ap")

    deltas = defaultdict(list)
    for _ in range(args.resamples):
        idx = [i for g in rng.choice(groups, len(groups)) for i in index[g]]
        base = macro_ap("selected@280", idx)
        if base is None:
            continue
        for condition in done:
            if condition != "selected@280":
                value = macro_ap(condition, idx)
                if value is not None:
                    deltas[condition].append(value - base)
    for condition, values in deltas.items():
        report[condition]["macro_ap_gain"] = {
            "mean": float(np.mean(values)),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
    save_json(folder / f"report_{args.tag}.json", report)
    header = "| Condition | Frames | Visual tokens | s/window | Macro AP | AP gain (95% CI) | F1 at Yes | Best F1 | "
    header += " | ".join(f"{n} AP" for n in NAMES.values()) + " |"
    print(header)
    print("|" + "---|" * (8 + len(NAMES)))
    for condition in done:
        r = report[condition]
        gain = r.get("macro_ap_gain")
        gain = (
            f"{gain['mean']:+.3f} [{gain['low']:+.3f}, {gain['high']:+.3f}]"
            if gain
            else "—"
        )
        print(
            f"| {condition} | {r['mean_frames']:.1f} | {r['mean_visual_tokens']:.0f} | {r['seconds']:.2f} | "
            f"{r['macro_ap']:.3f} | {gain} | {r['macro_f1_yes']:.3f} | {r['macro_best_f1']:.3f} | "
            + " | ".join(f"{r['sources'][n]['ap']:.3f}" for n in NAMES.values())
            + " |"
        )


if __name__ == "__main__":
    main()
