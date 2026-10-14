"""Build portable, hashed task manifests with source-group-separated development/test data.

Previously benchmarked test_data images become development data. Lying/person/action
proxies are excluded from confirmed task tests. Add reviewed images via --review-csv:
path,label,task,group_id,source,evidence,review_note (relative to that CSV).
Videos are sampled as both endpoint frames and chronological past-only contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from anomaly_eval.data import (
    DEDUPLICATION_POLICY,
    file_sha256,
    fingerprint,
    image_pixel_sha256,
    read_csv,
    read_definitions,
    source_moment_identity,
)
from anomaly_eval.datasets.fetch import CAVIAR_SEQUENCES
from anomaly_eval.datasets.labels import (
    collect_littering,
    collect_loitering,
    collect_rough_sleeping,
    collect_smoking,
)
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    resolve_arguments,
    resolve_path,
)

FIELDS = [
    "path",
    "label",
    "split",
    "group_id",
    "source",
    "task",
    "evidence",
    "sha256",
    "pixel_sha256",
    "review_note",
    "protocol",
    "timestamp_seconds",
    "origin",
]
PREPARED_SCHEMA = 3


def deduplicate_examples(rows, historical_groups=()):
    """Merge encoding aliases/source moments and connect their recording groups.

    Prefer checksum-verified bundle images when both sources contain a moment,
    so decoder/PNG differences on another host do not change the chosen input.
    """
    rows = [dict(row) for row in rows]
    parents = list(range(len(rows)))
    group_parents = {r["group_id"]: r["group_id"] for r in rows}

    def find(parents, key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(parents, first, second):
        a, b = find(parents, first), find(parents, second)
        parents[max(a, b)] = min(a, b)

    known = {}
    recordings = {}
    pixels = {}
    for index, row in enumerate(rows):
        path = Path(row["path"]).resolve()
        if path not in pixels:
            pixels[path] = image_pixel_sha256(path)
        row["pixel_sha256"] = pixels[path]
        moment = source_moment_identity(row.get("origin"), row["protocol"])
        if moment:
            if moment[1] in recordings:
                join(group_parents, row["group_id"], recordings[moment[1]])
            recordings[moment[1]] = row["group_id"]
        for kind, value in (
            ("bytes", row["sha256"]),
            ("pixels", row["pixel_sha256"]),
            ("moment", moment),
        ):
            if value is None:
                continue
            key = row["task"], row["protocol"], kind, value
            if key in known:
                other = rows[known[key]]
                if other["label"] != row["label"]:
                    raise ValueError(
                        f"Conflicting task labels for duplicate {kind} identity"
                    )
                join(parents, index, known[key])
                join(group_parents, row["group_id"], other["group_id"])
            else:
                known[key] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        row["group_id"] = find(group_parents, row["group_id"])
        components[find(parents, index)].append(row)
    bundle = Workspace.current().bundle_dir
    kept, aliases = [], []
    for members in components.values():
        representative = min(
            members,
            key=lambda row: (
                not Path(row["path"]).resolve().is_relative_to(bundle),
                portable_path(Path(row["path"])),
            ),
        )
        kept.append(representative)
        for row in members:
            if row is not representative:
                aliases.append(
                    {
                        "kept": portable_path(Path(representative["path"])),
                        "dropped": portable_path(Path(row["path"])),
                        "task": row["task"],
                        "protocol": row["protocol"],
                        "origin": row.get("origin"),
                    }
                )
    kept.sort(
        key=lambda row: (
            row["task"],
            row["protocol"],
            row["group_id"],
            row["pixel_sha256"],
        )
    )
    historical = {
        find(group_parents, g) for g in historical_groups if g in group_parents
    }
    return (
        kept,
        historical,
        {
            "policy": DEDUPLICATION_POLICY,
            "input_rows": len(rows),
            "unique_rows": len(kept),
            "removed_rows": len(aliases),
            "aliases": aliases,
        },
    )


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            item = {k: row.get(k, "") for k in FIELDS}
            item["path"] = portable_path(Path(row["path"]), path.parent)
            w.writerow(item)


def assign_splits(rows, seed=42, historical_groups=()):
    """Stratify group label sets, keep all samples from a source recording together.

    With two independent positive groups, reserve one for dev and one for test.
    One group cannot estimate held-out performance and remains development only.
    """
    historical = set(historical_groups)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    strata = defaultdict(list)
    for key, samples in groups.items():
        strata[tuple(sorted({r["label"] for r in samples}))].append(key)
    rng = random.Random(seed)
    mapping = {g: "dev" for g in historical}
    for signature in sorted(strata):
        keys = strata[signature]
        keys = sorted(k for k in keys if k not in historical)
        rng.shuffle(keys)
        if len(keys) < 2:
            mapping.update({k: "dev" for k in keys})
            continue
        # Zero-shot image models require no training split. Training of temporal
        # adapters uses its separate frozen three-way smoking protocol.
        n_test = max(1, round(0.4 * len(keys)))
        mapping.update({k: "test" if i < n_test else "dev" for i, k in enumerate(keys)})
    for r in rows:
        r["split"] = mapping[r["group_id"]]
    return rows


def read_reviews(path):
    rows = []
    definitions = read_definitions(Workspace.current().definitions)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r = {k: v.strip() if isinstance(v, str) else v for k, v in r.items()}
            if not all(
                r.get(k)
                for k in (
                    "path",
                    "label",
                    "task",
                    "group_id",
                    "source",
                    "evidence",
                    "review_note",
                )
            ):
                raise ValueError(
                    "Reviewed rows require path,label,task,group_id,source,evidence,review_note"
                )
            if (
                r["task"] not in definitions
                or r["task"] == "normal"
                or r["label"] not in {"normal", r["task"]}
            ):
                raise ValueError(
                    "Reviewed labels must be normal or the named anomaly task"
                )
            if r["evidence"] not in {"human_review", "sequence_annotation_review"}:
                raise ValueError(
                    "Review imports need explicit human/sequence review evidence"
                )
            r["path"] = str(resolve_path(r["path"], path.parent))
            if not Path(r["path"]).is_file():
                raise ValueError("Missing reviewed image")
            r["protocol"] = r.get("protocol") or "frame"
            if r["protocol"] not in {"frame", "window"}:
                raise ValueError("Reviewed protocol must be frame or window")
            r["sha256"] = file_sha256(Path(r["path"]))
            rows.append(r)
    return rows


def _frame(video, timestamp, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-ss",
            f"{timestamp:.6f}",
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )


def caviar_examples(root, output, intervals, per_interval=4):
    from PIL import Image, ImageDraw

    rows = []
    video_hashes = {}
    reviewed = json.loads(intervals.read_text())
    for index, interval in enumerate(reviewed["intervals"]):
        sequence = interval["sequence"]
        video = root / "caviar" / sequence / (sequence + ".mpg")
        annotation = root / "caviar" / sequence / CAVIAR_SEQUENCES[sequence]
        if not video.is_file() or not annotation.is_file():
            continue
        if video not in video_hashes:
            video_hashes[video] = file_sha256(video)
        # XML uses zero-based frame indices at 25 fps; ffmpeg samples at PTS.
        xml = ET.parse(annotation).getroot()
        for i in range(per_interval):
            t = interval["start"] + (interval["end"] - interval["start"]) * i / max(
                1, per_interval - 1
            )
            frame = xml.find(f"frame[@number='{round(t * 25)}']")
            if frame is None:
                raise ValueError(f"Missing annotation at {sequence}:{t}")
            if interval["label"] == "unattended_items" and not any(
                r.text == "leaving object" for r in frame.iter("role")
            ):
                raise ValueError(
                    "Reviewed unattended interval has no left-object annotation"
                )
            prefix = f"caviar_{sequence}_{index}_{i}"
            # Past-only evidence; equal window inputs for all CLIP/VLM adapters.
            paths = []
            for j in range(4):
                p = output / "frames" / (prefix + f"_{j}.png")
                _frame(video, max(0, t - 6 + 2 * j), p)
                paths.append(p)
            single = paths[-1]
            with Image.open(single) as im:
                w, h = im.size
            sheet = Image.new("RGB", (w * 2, h * 2 + 32), "white")
            draw = ImageDraw.Draw(sheet)
            for j, p in enumerate(paths):
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB"), (j % 2 * w, 32 + j // 2 * h))
            draw.text(
                (6, 8),
                "Chronological: top left, top right, bottom left, bottom right",
                fill="black",
            )
            window = output / "windows" / (prefix + ".png")
            window.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(window)
            for protocol, p in (("frame", single), ("window", window)):
                rows.append(
                    {
                        "path": str(p.resolve()),
                        "label": interval["label"],
                        "group_id": "caviar:" + sequence,
                        "source": "caviar_inria_single_camera",
                        "task": "unattended_items",
                        "evidence": "sequence_annotation_review",
                        "sha256": file_sha256(p),
                        "review_note": interval["note"],
                        "protocol": protocol,
                        "timestamp_seconds": t,
                        "origin": "caviar://"
                        + sequence
                        + "/"
                        + video.name
                        + "#"
                        + video_hashes[video]
                        + "@"
                        + str(t),
                    }
                )
    return rows


def prepare(
    raw, additional, output, review_csv=None, per_group=8, seed=42, per_interval=4
):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    proxy = []
    issues = []
    historical = set()
    # Old bundle is always development, including all known source recording groups.
    bundle = Workspace.current().bundle_dir / "MANIFEST.csv"
    if bundle.is_file():
        for r in read_csv(bundle):
            p = bundle.parent / r["file"]
            historical.add(r["group_id"])
            item = {
                "path": str(p.resolve()),
                "label": r["label"],
                "group_id": r["group_id"],
                "source": "historical_bundle",
                "task": r["label"] if r["label"] != "normal" else "smoking",
                "protocol": "frame",
                "evidence": "source_annotation"
                if r["label"] in {"smoking", "normal"}
                else "proxy",
                "sha256": file_sha256(p),
                "review_note": "Previously evaluated: development only",
                "split": "dev",
            }
            if item["sha256"] != r["sha256"]:
                raise ValueError("Historical bundle checksum mismatch")
            (rows if item["evidence"] != "proxy" else proxy).append(item)
    try:
        pos, neg = collect_smoking(raw)
        for item in pos + neg:
            rows.append(
                {
                    "path": str(item.path),
                    "label": item.label,
                    "group_id": item.group_id,
                    "source": "smoking_coco",
                    "task": "smoking",
                    "protocol": "frame",
                    "evidence": "source_annotation",
                    "sha256": file_sha256(item.path),
                }
            )
    except ValueError as e:
        issues.append("smoking: " + str(e))
    for collector in (collect_littering, collect_rough_sleeping, collect_loitering):
        try:
            for item in collector(raw):
                proxy.append(
                    {
                        "path": str(item.path),
                        "label": item.label,
                        "group_id": item.group_id,
                        "source": "raw_proxy",
                        "task": item.label,
                        "protocol": "frame",
                        "evidence": "proxy",
                        "sha256": file_sha256(item.path),
                        "split": "dev",
                    }
                )
        except ValueError as e:
            issues.append(collector.__name__ + ": " + str(e))
    rows += caviar_examples(
        additional,
        output / "derived",
        Workspace.current().config_dir / "caviar_review_intervals.json",
        per_interval,
    )
    # Portable curated CAVIAR inputs remain usable after cloning, without raw videos.
    curated = Workspace.current().bundle_dir / "EVALUATION.csv"
    if curated.is_file():
        for item in read_csv(curated):
            if item["source"] != "caviar_inria_single_camera":
                continue
            item["path"] = str(resolve_path(item["path"], curated.parent))
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Curated CAVIAR checksum mismatch")
            rows.append(item)
    if review_csv:
        rows += read_reviews(review_csv)
    for row in rows:
        if row.get("timestamp_seconds") not in (None, ""):
            row["timestamp_seconds"] = str(float(row["timestamp_seconds"]))
    rows, hist, deduplication = deduplicate_examples(rows, historical)
    # Make one common group assignment across endpoint and window protocols/tasks.
    assign_splits(rows, seed, hist)
    rng = random.Random(seed)
    rng.shuffle(rows)
    counts = Counter()
    selected = []
    for r in rows:
        key = (r["task"], r["protocol"], r["group_id"], r["label"])
        if not per_group or counts[key] < per_group:
            selected.append(r)
            counts[key] += 1
    write_manifest(output / "confirmed.csv", selected)
    write_manifest(output / "proxy_development.csv", proxy[:1000])
    coverage = {}
    manifests = {}
    tasks = [
        k for k in read_definitions(Workspace.current().definitions) if k != "normal"
    ]
    for task in tasks:
        coverage[task] = {}
        for protocol in ("frame", "window"):
            subset = [
                r for r in selected if r["task"] == task and r["protocol"] == protocol
            ]
            if not subset:
                continue
            name = f"{task}_{protocol}"
            path = output / (name + ".csv")
            write_manifest(path, subset)
            manifests[name] = path.name
            bysplit = {
                split: dict(Counter(r["label"] for r in subset if r["split"] == split))
                for split in ("dev", "test")
            }
            groups = {
                split: len({r["group_id"] for r in subset if r["split"] == split})
                for split in ("dev", "test")
            }
            groups_per_label = {
                split: {
                    label: len(
                        {
                            r["group_id"]
                            for r in subset
                            if r["split"] == split and r["label"] == label
                        }
                    )
                    for label in ("normal", task)
                }
                for split in ("dev", "test")
            }
            eligible = all(
                bysplit[s].get(task, 0) > 0 and bysplit[s].get("normal", 0) > 0
                for s in ("dev", "test")
            )
            coverage[task][protocol] = {
                "counts": bysplit,
                "groups": groups,
                "groups_per_label": groups_per_label,
                "eligible_for_calibrated_test": eligible,
            }
    result = {
        "schema": PREPARED_SCHEMA,
        "deduplication": deduplication,
        "seed": seed,
        "manifests": manifests,
        "coverage": coverage,
        "issues": issues,
        "confirmed_rows": len(selected),
        "sampling": {
            "per_group": per_group,
            "available_unique_rows": len(rows),
            "selected_rows": len(selected),
        },
        "proxy_development_rows": len(proxy),
        "fingerprint": fingerprint(
            [
                {k: r.get(k) for k in FIELDS if k != "path"}
                for r in sorted(
                    selected, key=lambda x: (x["task"], x["protocol"], x["sha256"])
                )
            ]
        ),
        "limitations": [
            "Historical benchmark groups are development only.",
            "CAVIAR is a small staged single-camera set; source-sequence holdout is not a new-camera holdout.",
            "Window contact sheets are a separate past-context protocol, not native video input.",
            "Proxy pose/person/action labels do not establish rough sleeping, loitering or littering events.",
            "ABODA, D2-CDNet and Roboflow candidates need event/normal review before task scoring.",
        ],
    }
    (output / "dataset.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw", type=Path, default=Workspace.current().data_dir / "raw" / "datasets"
    )
    p.add_argument(
        "--additional", type=Path, default=Workspace.current().data_dir / "additional"
    )
    p.add_argument(
        "--output", type=Path, default=Workspace.current().data_dir / "evaluation"
    )
    p.add_argument("--review-csv", type=Path)
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--per-interval", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = resolve_arguments(p.parse_args(argv))
    if a.per_group < 0 or a.per_interval < 1:
        p.error("per-group must be nonnegative and per-interval positive")
    result = prepare(
        a.raw.resolve(),
        a.additional.resolve(),
        a.output.resolve(),
        a.review_csv,
        a.per_group,
        a.seed,
        a.per_interval,
    )
    print(json.dumps(result, indent=2))
    return 0
