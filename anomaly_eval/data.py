"""Shared manifest and anomaly-definition validation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from anomaly_eval.paths import (
    PACKAGE_ROOT,
    portable_path,
    resolve_path,
)

DEFAULT_DEFINITIONS = PACKAGE_ROOT / "config" / "anomalies.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str | None
    split: str
    group_id: str
    origin: str | None = None
    pixel_sha256: str | None = None
    protocol: str = "frame"


DEDUPLICATION_POLICY = "source_moment_and_exif_rgb_v1"


def image_pixel_sha256(path: Path) -> str:
    """Hash model-visible decoded RGB pixels, independently of image encoding."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(f"exif_rgb_v1:{image.width}:{image.height}:".encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


def source_moment_identity(origin: str | None, protocol: str = "frame"):
    """Recognize CAVIAR's checksum-qualified source time at exporter precision.

    Other provenance strings remain descriptive; a path to a whole video alone
    is not an identity for a frame. Window inputs use the fixed exporter protocol.
    """
    if not origin or not origin.startswith("caviar://"):
        return None
    match = re.fullmatch(r"(caviar://[^#@]+)#([0-9a-fA-F]{64})@([^@]+)", origin)
    if not match:
        raise ValueError(
            "CAVIAR origins require source URI, video SHA-256 and timestamp"
        )
    try:
        timestamp = Decimal(match[3])
        if not timestamp.is_finite() or timestamp < 0:
            raise ValueError("CAVIAR source timestamp must be finite and nonnegative")
        timestamp = timestamp.quantize(Decimal("0.000001"))
    except InvalidOperation as exc:
        raise ValueError("Invalid CAVIAR source timestamp") from exc
    return protocol, match[1] + "#" + match[2].lower(), str(timestamp)


def read_definitions(path: Path) -> dict[str, str]:
    definitions = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(definitions, dict)
        or "normal" not in definitions
        or len(definitions) < 2
    ):
        raise ValueError(
            "Definitions must map 'normal' and at least one anomaly to descriptions"
        )
    if any(
        not isinstance(key, str)
        or not key.strip()
        or key != key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in definitions.items()
    ):
        raise ValueError("Definition names and descriptions must be nonempty strings")
    if len({key.casefold() for key in definitions}) != len(definitions):
        raise ValueError("Definition names must be unique ignoring capitalization")
    return definitions


def read_manifest(path: Path, definitions: dict[str, str]) -> list[Sample]:
    samples = []
    seen_paths: set[Path] = set()
    content_splits: dict[str, set[str]] = {}
    pixel_inputs = {}
    source_moments = {}
    groups: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError(
                "Manifest needs path,label,split columns (group_id is recommended)"
            )
        for line, row in enumerate(reader, start=2):
            if None in row or any(
                row[key] is None for key in ("path", "label", "split")
            ):
                raise ValueError(f"Manifest line {line}: malformed CSV row")
            if not row["path"].strip():
                raise ValueError(f"Manifest line {line}: empty image path")
            image_path = resolve_path(row["path"].strip(), path.parent)
            label = row["label"].strip() or None
            split = row["split"].strip() or "test"
            group_id = (row.get("group_id") or "").strip() or portable_path(
                image_path, path.parent
            )
            if split not in {"train", "dev", "test", "reference", "ignore"}:
                raise ValueError(
                    f"Manifest line {line}: split must be train, dev, test, reference, or ignore"
                )
            if label is not None and label not in definitions:
                raise ValueError(f"Manifest line {line}: unknown label {label!r}")
            if split == "ignore":
                continue
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                raise ValueError(
                    f"Manifest line {line}: missing or unsupported image {image_path}"
                )
            if image_path in seen_paths:
                raise ValueError(f"Manifest line {line}: duplicate image {image_path}")
            if split == "reference" and label is None:
                raise ValueError(
                    f"Manifest line {line}: reference images need verified labels"
                )
            digest = file_sha256(image_path)
            if row.get("sha256") and row["sha256"].strip() != digest:
                raise ValueError(f"Manifest line {line}: image SHA-256 mismatch")
            pixel_digest = image_pixel_sha256(image_path)
            if row.get("pixel_sha256") and row["pixel_sha256"].strip() != pixel_digest:
                raise ValueError(
                    f"Manifest line {line}: decoded pixel SHA-256 mismatch"
                )
            origin = (row.get("origin") or "").strip() or None
            protocol = (row.get("protocol") or "frame").strip()
            moment = source_moment_identity(origin, protocol)
            for identity, known, kind in (
                (pixel_digest, pixel_inputs, "decoded image pixels"),
                (moment, source_moments, "source moment"),
            ):
                if identity is None:
                    continue
                if identity in known:
                    prefix = (
                        "Content leakage"
                        if known[identity] != split
                        else "Duplicate input"
                    )
                    raise ValueError(
                        f"{prefix}: repeated {kind}; rebuild dataset preparation"
                    )
                known[identity] = split
            content_splits.setdefault(digest, set()).add(split)
            seen_paths.add(image_path)
            groups.setdefault(group_id, set()).add(split)
            samples.append(
                Sample(
                    image_path, label, split, group_id, origin, pixel_digest, protocol
                )
            )
    leaking = [group for group, splits in groups.items() if len(splits) > 1]
    if leaking:
        raise ValueError(
            f"Reference/test leakage or train/dev/test leakage: source groups cross splits: {leaking[:3]}"
        )
    if any(len(splits) > 1 for splits in content_splits.values()):
        raise ValueError("Content leakage: identical image bytes cross splits")
    if not any(sample.split in {"test", "dev"} for sample in samples):
        raise ValueError("Manifest has no test or development images")
    return samples


def select_test_samples(
    samples: list[Sample], limit: int | None, seed: int
) -> list[Sample]:
    """Repeatable capped cohort retaining label support and source-group diversity.

    Label quotas approximate the original proportions, with at least one example
    per present label. Within each label, source groups contribute in round-robin
    order so a long recording cannot crowd out all other recordings.
    """
    selected = sorted(
        (sample for sample in samples if sample.split == "test"),
        key=lambda sample: (sample.group_id, portable_path(sample.path)),
    )
    rng = random.Random(seed)
    if limit is None or limit >= len(selected):
        rng.shuffle(selected)
        return selected
    if limit < 1:
        raise ValueError("Sample limit must be positive or unset")
    counts = Counter(s.label for s in selected)
    labels = sorted(counts, key=lambda label: label or "")
    if limit < len(labels):
        raise ValueError(
            f"Sample limit {limit} cannot retain all {len(labels)} present labels"
        )
    quotas = dict.fromkeys(labels, 1)
    targets = {label: counts[label] * limit / len(selected) for label in labels}
    for _ in range(limit - len(labels)):
        label = max(
            (label for label in labels if quotas[label] < counts[label]),
            key=lambda label: targets[label] - quotas[label],
        )
        quotas[label] += 1
    capped = []
    group_order = sorted({s.group_id for s in selected})
    rng.shuffle(group_order)
    for label in labels:
        groups = defaultdict(list)
        for sample in selected:
            if sample.label == label:
                groups[sample.group_id].append(sample)
        # A shared order retains matched event/control recordings when both
        # labels have support in the same groups and their quotas permit it.
        order = [group for group in group_order if group in groups]
        for pool in groups.values():
            rng.shuffle(pool)
        remaining = quotas[label]
        while remaining:
            for group in order:
                if groups[group]:
                    capped.append(groups[group].pop())
                    remaining -= 1
                    if not remaining:
                        break
    rng.shuffle(capped)
    return capped


def selection_summary(available: list[Sample], selected: list[Sample]) -> dict:
    return {
        "available_samples": len(available),
        "selected_samples": len(selected),
        "available_label_counts": dict(
            Counter(s.label or "unlabeled" for s in available)
        ),
        "selected_label_counts": dict(
            Counter(s.label or "unlabeled" for s in selected)
        ),
        "available_groups": len({s.group_id for s in available}),
        "selected_groups": len({s.group_id for s in selected}),
        "available_groups_per_label": {
            label: len(
                {s.group_id for s in available if (s.label or "unlabeled") == label}
            )
            for label in sorted({s.label or "unlabeled" for s in available})
        },
        "selected_groups_per_label": {
            label: len(
                {s.group_id for s in selected if (s.label or "unlabeled") == label}
            )
            for label in sorted({s.label or "unlabeled" for s in selected})
        },
        "policy": "label_proportional_group_round_robin_v1",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_original(path: Path) -> Path | None:
    """Resolve an original remote path; conflicting contents require manual review."""
    path = path.resolve()
    candidates = [path] if path.is_file() else []
    for parent in path.parents:
        receipt = parent / "download_receipt.json"
        if not receipt.is_file():
            continue
        entries = json.loads(receipt.read_text()).get("renamed_files", [])
        for entry in entries:
            original = Path(entry["original_path"].replace("\\", "/"))
            actual = Path(entry["path"].replace("\\", "/"))
            if (parent / original).resolve() == path:
                local = (parent / actual).resolve()
                if local.is_file():
                    candidates.append(local)
        break
    unique = {file_sha256(p): p for p in sorted(set(candidates))}
    if len(unique) > 1:
        raise ValueError(f"Ambiguous original filename has differing contents: {path}")
    return next(iter(unique.values()), None)


def original_name(path: Path) -> str:
    for parent in path.resolve().parents:
        receipt = parent / "download_receipt.json"
        if receipt.is_file():
            for entry in json.loads(receipt.read_text()).get("renamed_files", []):
                actual = (parent / entry["path"].replace("\\", "/")).resolve()
                if actual == path.resolve():
                    return Path(entry["original_path"].replace("\\", "/")).name
            break
    return path.name


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def code_hashes() -> dict:
    return {
        "anomaly_eval/" + p.relative_to(PACKAGE_ROOT).as_posix(): file_sha256(p)
        for p in sorted(PACKAGE_ROOT.rglob("*"))
        if p.suffix in {".py", ".json"}
    }


def sample_identity(samples) -> list[dict]:
    return [
        {
            "path": portable_path(s.path),
            "sha256": file_sha256(s.path),
            "pixel_sha256": s.pixel_sha256 or image_pixel_sha256(s.path),
            "origin": s.origin,
            "source_moment": source_moment_identity(s.origin, s.protocol),
            "protocol": s.protocol,
            "label": s.label,
            "split": s.split,
            "group_id": s.group_id,
        }
        for s in samples
    ]


def experiment_options(options: dict) -> dict:
    """Locations and CLI plumbing are provenance, not scoring/runtime parameters."""
    location_keys = {
        "prepared_dir",
        "output",
        "manifest",
        "definitions",
        "cache_dir",
        "data_dir",
        "worker_config",
        "workspace",
        "list_models",
    }
    return {key: value for key, value in options.items() if key not in location_keys}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
