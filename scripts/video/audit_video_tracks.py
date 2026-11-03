"""Check automatic person/item tracks against existing CAVIAR source boxes."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from anomaly_eval.training.strategies import overlap

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    tracks = {
        r["path"]: r["tracks"]
        for r in map(
            json.loads, (args.tracks or data / "tracks.jsonl").read_text().splitlines()
        )
    }
    records = [
        r
        for r in map(json.loads, (data / "recordings.jsonl").read_text().splitlines())
        if r["id"].startswith("caviar_")
    ]
    results = []
    for record in records:
        directory = ROOT / Path(record["video"]).parent
        xml = next(directory.glob("*gt.xml"))
        frames = {
            int(f.attrib["number"]): f for f in ET.parse(xml).getroot().iter("frame")
        }
        with Image.open(ROOT / record["frames"][0]["path"]) as image:
            w, h = image.size
        for kind in ["person", "item"]:
            for cutoff in [0.3, 0.5]:
                tp = fp = fn = switches = 0
                previous = {}
                for frame in record["frames"]:
                    truth = []
                    if frame["index"] not in frames:
                        continue
                    for obj in frames[frame["index"]].findall("objectlist/object"):
                        role = obj.findtext("hypothesislist/hypothesis/role")
                        if (role == "leaving object") != (kind == "item"):
                            continue
                        b = obj.find("box").attrib
                        x = float(b["xc"])
                        y = float(b["yc"])
                        bw = float(b["w"])
                        bh = float(b["h"])
                        truth.append(
                            (
                                obj.attrib["id"],
                                [
                                    (x - bw / 2) / w,
                                    (y - bh / 2) / h,
                                    (x + bw / 2) / w,
                                    (y + bh / 2) / h,
                                ],
                            )
                        )
                    predicted = [t for t in tracks[frame["path"]] if t["kind"] == kind]
                    cost = np.asarray(
                        [
                            [1 - overlap(a, b["box"]) for b in predicted]
                            for _, a in truth
                        ]
                    )
                    matched = []
                    if len(truth) and len(predicted):
                        gi, pi = linear_sum_assignment(cost)
                        matched = [
                            (g, p) for g, p in zip(gi, pi) if 1 - cost[g, p] >= cutoff
                        ]
                    tp += len(matched)
                    fp += len(predicted) - len(matched)
                    fn += len(truth) - len(matched)
                    for g, p in matched:
                        key = truth[g][0]
                        pid = predicted[p]["id"]
                        if (
                            key in previous
                            and frame["time"] - previous[key][1] <= 1.1
                            and previous[key][0] != pid
                        ):
                            switches += 1
                        previous[key] = (pid, frame["time"])
                results.append(
                    {
                        "recording_id": record["id"],
                        "kind": kind,
                        "iou": cutoff,
                        "matched": tp,
                        "unmatched_tracks": fp,
                        "missed_source_boxes": fn,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "reference_match_fraction": tp / (tp + fp) if tp + fp else None,
                        "local_id_switches": switches,
                    }
                )
    aggregate = []
    for kind in ["person", "item"]:
        for cutoff in [0.3, 0.5]:
            values = [r for r in results if r["kind"] == kind and r["iou"] == cutoff]
            sums = {
                k: sum(r[k] for r in values)
                for k in [
                    "matched",
                    "unmatched_tracks",
                    "missed_source_boxes",
                    "local_id_switches",
                ]
            }
            aggregate.append(
                {
                    "kind": kind,
                    "iou": cutoff,
                    **sums,
                    "recall": sums["matched"]
                    / max(1, sums["matched"] + sums["missed_source_boxes"]),
                    "reference_match_fraction": sums["matched"]
                    / max(1, sums["matched"] + sums["unmatched_tracks"]),
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "per_recording": results,
                "qualification": "Training-camera diagnostic, not official MOT evaluation. Published leaving-object roles identify only event-item boxes; other carried bags/items are not exhaustively annotated. Unmatched item tracks cannot be counted as proven false positives and reference-match fraction is not item precision. IoU Hungarian matching at sampled frames; ID switches counted only across <=1.1-second matched gaps. No source boxes choose crops/tracks. Ownership has no independent ground truth.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
