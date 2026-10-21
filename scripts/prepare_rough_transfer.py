"""Reserve newly reviewed public-sleep photos with unseen authors and pixels."""

from __future__ import annotations

import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anomaly_eval.data import image_pixel_sha256
from scripts.prepare_strategy_data import signature

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/strategy_study_20261016"
SOURCE = ROOT / "data/strategy_sources_20261016/commons_human"


def author(row):
    value = row["metadata"].get("Artist", {}).get("value", "")
    return (
        re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", "", value))).strip().lower()
    )


def main():
    path = DATA / "rough_screened.jsonl"
    if path.exists():
        raise ValueError("New photo transfer cohort already frozen")
    rows = json.loads((SOURCE / "downloaded_with_flickr.json").read_text())
    review = json.loads((SOURCE / "review_labels.json").read_text())
    assert (
        hashlib.sha256(
            (SOURCE / "downloaded_with_flickr.json").read_bytes()
        ).hexdigest()
        == review["candidate_manifest_sha256"]
    )
    old = [
        json.loads(line)
        for line in (ROOT / "data/clip_training_20261015/samples.jsonl")
        .read_text()
        .splitlines()
    ]
    old_ids = {r["source_id"] for r in old if r["source"] == "commons_visible_sleeping"}
    old_candidates = json.loads(
        (
            ROOT / "data/training_sources_20261015/commons_review/candidates.json"
        ).read_text()
    )
    previous_authors = {author(r) for r in old_candidates if r["id"] in old_ids}
    references = json.loads((DATA / "reference_phashes.json").read_text())
    missing = sorted({p for r in old for p in r["views"]} - set(references))
    with ThreadPoolExecutor(12) as pool:
        references.update(
            zip(missing, pool.map(lambda p: signature(ROOT / p), missing))
        )
    retained, rejected = [], []
    for row in rows:
        label = review["labels_by_page_id"].get(row["id"])
        if label not in [0, 1] or "path" not in row:
            continue
        image = ROOT / row["path"]
        assert (
            hashlib.sha256(image.read_bytes()).hexdigest()
            == review["image_sha256"][row["id"]]
        )
        a = author(row)
        image_signature = signature(image)
        distance = min(
            (image_signature ^ value).bit_count() for value in references.values()
        )
        if not a or a in previous_authors or distance <= 6:
            rejected.append(
                {
                    "id": row["id"],
                    "reason": "unknown_or_previously_used_author"
                    if not a or a in previous_authors
                    else "related_old_image",
                    "phash_distance": distance,
                }
            )
            continue
        retained.append(
            {
                "id": "commons_new:" + row["id"],
                "task": "rough_sleeping",
                "target": label,
                "source": "commons_new_author_transfer",
                "group_id": "commons_author:" + a,
                "views": [row["path"]],
                "split": "test",
                "pixel_hashes": [image_pixel_sha256(image)],
                "evidence": "provisional_assistant_visual_review",
                "qualification": "New author/photo holdout; provisional apparent public-sleep labels, not independently human verified; no housing-status inference",
                "source_page": row["source_page"],
            }
        )
    assert {r["target"] for r in retained} == {0, 1}
    path.write_text("".join(json.dumps(r) + "\n" for r in retained))
    (DATA / "rough_screened.audit.json").write_text(
        json.dumps(
            {
                "reviewed_positive": sum(
                    v == 1 for v in review["labels_by_page_id"].values()
                ),
                "reviewed_normal": sum(
                    v == 0 for v in review["labels_by_page_id"].values()
                ),
                "retained": len(retained),
                "positive": sum(r["target"] for r in retained),
                "authors": len({r["group_id"] for r in retained}),
                "excluded": rejected,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "NEW AUTHOR ROUGH TRANSFER",
        len(retained),
        "positive",
        sum(r["target"] for r in retained),
        "excluded",
        len(rejected),
        flush=True,
    )


if __name__ == "__main__":
    main()
