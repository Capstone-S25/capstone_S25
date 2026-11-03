"""Purge known title-part aliases across modeling folds before final fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def canonical_title(group):
    title = group.removeprefix("hmdb_title:")
    title = re.sub(
        r"(?:[_ -])(?:part|pt|clip|scene|episode)[_ -]*\d+.*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^a-z0-9]", "", title.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    source = data / "verified_windows.jsonl"
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row["source"] == "hmdb51_action_transfer":
            groups[canonical_title(row["group_id"])].append(row)
    excluded = set()
    collisions = []
    priority = {"train": 0, "dev": 1, "test": 2}
    for key, values in groups.items():
        splits = {r["split"] for r in values}
        if len(splits) == 1:
            continue
        retained = max(splits, key=lambda s: priority[s])
        removed = [r for r in values if r["split"] != retained]
        excluded.update(r["id"] for r in removed)
        collisions.append(
            {
                "canonical_title": key,
                "source_groups": sorted({r["group_id"] for r in values}),
                "retained_fold": retained,
                "excluded_ids": [r["id"] for r in removed],
            }
        )
    retained = [r for r in rows if r["id"] not in excluded]
    destination = data / "modeling_windows.jsonl"
    content = "".join(json.dumps(r) + "\n" for r in retained)
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Modeling inputs already frozen")
    destination.write_text(content)
    audit = {
        "status": "pass",
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "modeling_manifest_sha256": hashlib.sha256(
            destination.read_bytes()
        ).hexdigest(),
        "rows": len(retained),
        "excluded": len(excluded),
        "collisions": collisions,
        "policy": "Canonicalize explicit part/pt/clip/scene/episode-number suffixes; retain higher holdout fold and purge lower-fold clips; no model scores or labels select exclusions",
        "qualification": "Known filename aliases screened; this does not establish independently verified film/actor identity for every web clip",
    }
    (data / "source_guard.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
