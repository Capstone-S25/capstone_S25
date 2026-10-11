"""Derive image-level anomaly labels from YOLO/COCO annotation files.

The benchmark deliberately does not import detection boxes; this module is the
single place that turns per-box annotation files into *image-level* labels so a
balanced multi-anomaly manifest can be built for head-to-head model evaluation.

Collectors use local annotations and download receipts (no network) and takes
its dataset root as an argument so it can be pointed at a small fixture tree in
tests without touching the real data on disk.

Label strategy per class (see notebooks/anomaly_benchmark.ipynb for caveats):

* ``smoking``        -- >=1 "Smoking" COCO box in smoking-video.v1i.coco.
* ``normal``         -- a smoking-video frame with a "Not Smoking" box and no
                        "Smoking" box (a clean, person-present normal).
* ``littering``      -- an image whose matching YOLO label file is non-empty.
* ``rough_sleeping`` -- a kaggle laying_dataset image labelled only as class 0
                        ("laying"); scenes that also contain a "standing" person
                        are excluded to keep the positive set unambiguous.
* ``loitering``      -- every frame of the ATMGUARD-Loitering-Tracking corpus,
                        treated as a dataset-level proxy (the COCO file carries
                        only "person" boxes; loitering is inherently temporal).

``unattended_items`` has no labels in the repo and is intentionally not produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from anomaly_eval.data import (
    IMAGE_EXTENSIONS,
    original_name,
    resolve_original,
)
from anomaly_eval.paths import Workspace

# Classes that can be labelled from the available datasets (order is stable and
# used as the manifest row order). ``normal`` is the clean-negative class.
MANIFEST_CLASSES = ("smoking", "littering", "rough_sleeping", "loitering", "normal")


@dataclass(frozen=True)
class LabeledImage:
    """One image with a derived label and a diversity group id."""

    path: Path
    label: str
    group_id: str


# --------------------------------------------------------------------------- #
# Annotation parsing helpers (pure, testable).
# --------------------------------------------------------------------------- #
def load_coco(coco_path: Path) -> tuple[dict[int, str], dict[int, str], list[dict]]:
    """Parse a COCO ``_annotations.coco.json`` into its three core tables.

    Returns ``(categories, images_by_id, annotations)`` where ``images_by_id``
    maps image id -> relative file name as stored in the file.
    """
    data = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    categories = {
        c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])
    }
    images = {i["id"]: i.get("file_name", "") for i in data.get("images", [])}
    return categories, images, list(data.get("annotations", []))


def coco_positive_image_ids(
    categories: dict[int, str], annotations: list[dict], positive_names: set[str]
) -> set[int]:
    """Image ids that carry at least one box whose category name is in the set."""
    wanted = {cid for cid, name in categories.items() if name in positive_names}
    return {a["image_id"] for a in annotations if a.get("category_id") in wanted}


def yolo_has_box(label_path: Path) -> bool:
    """True when a YOLO label file exists and contains at least one box line."""
    if not label_path.is_file():
        return False
    return any(
        line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()
    )


def yolo_class_tokens(label_path: Path) -> set[str]:
    """The set of class-id tokens present in a YOLO label file (empty if absent)."""
    if not label_path.is_file():
        return set()
    return {
        line.split()[0]
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split()
    }


# --------------------------------------------------------------------------- #
# Diversity group keys (derived from file names only, so they are testable).
# --------------------------------------------------------------------------- #
_SMOKE_FRAME_RE = re.compile(r"frame_(c\d+)_(\d+)_([\d]{4}-[\d]{2}-[\d]{2})")
_LITTER_VIDEO_RE = re.compile(r"video_(\d+)")


def smoking_video_group(path: Path) -> str:
    """Group a smoking-video frame by (camera, date): one recording session."""
    match = _SMOKE_FRAME_RE.search(Path(path).name)
    if not match:
        return f"smoke:{Path(path).name}"
    camera, _, date = match.groups()
    return f"smoke:{camera}:{date}"


def littering_group(path: Path) -> str:
    """Group a littering frame by its source video id (``video_<n>`` prefix)."""
    match = _LITTER_VIDEO_RE.search(Path(path).name)
    if not match:
        return f"litter:{Path(path).name}"
    return f"litter:video_{match.group(1)}"


# --------------------------------------------------------------------------- #
# Per-category collectors. Each returns LabeledImage lists rooted at ``root``.
# --------------------------------------------------------------------------- #
def collect_smoking(
    root: Path | None = None,
) -> tuple[list[LabeledImage], list[LabeledImage]]:
    """Return ``(smoking_positives, normals)`` from the smoking-video COCO set.

    A frame is a *smoking* positive when it has any "Smoking" box (a mixed
    Smoking + Not-Smoking frame counts as smoking). Otherwise, if it has a
    "Not Smoking" box, it is a clean *normal*.
    """
    root = (root or Workspace.current().data_dir / "raw" / "datasets").resolve()
    base = root / "smoking" / "smoking-video.v1i.coco"
    positives: list[LabeledImage] = []
    normals: list[LabeledImage] = []
    for split in ("train", "valid"):
        coco_path = resolve_original(base / split / "_annotations.coco.json")
        if coco_path is None:
            continue
        categories, images, annotations = load_coco(coco_path)
        smoking_ids = coco_positive_image_ids(categories, annotations, {"Smoking"})
        not_smoking_ids = coco_positive_image_ids(
            categories, annotations, {"Not Smoking"}
        )
        for image_id, file_name in images.items():
            path = resolve_original(coco_path.parent / file_name)
            if path is None:
                continue
            group = smoking_video_group(path)
            if image_id in smoking_ids:
                positives.append(LabeledImage(path, "smoking", group))
            elif image_id in not_smoking_ids:
                normals.append(LabeledImage(path, "normal", group))
    return positives, normals


def collect_littering(root: Path | None = None) -> list[LabeledImage]:
    """Littering positives: images with a non-empty matching YOLO label file.

    Only the ``train`` and ``test`` splits are used; the ``valid`` split ships
    labels but no images in this dataset.
    """
    root = (root or Workspace.current().data_dir / "raw" / "datasets").resolve()
    base = root / "littering" / "Littering"
    positives: list[LabeledImage] = []
    for split in ("train", "valid", "test"):
        image_dir, label_dir = base / split / "images", base / split / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            original_image = resolve_original(image_dir / original_name(image))
            if original_image is None:
                continue
            label_path = resolve_original(
                label_dir / (Path(original_name(image)).stem + ".txt")
            )
            if label_path is not None and yolo_has_box(label_path):
                positives.append(
                    LabeledImage(image.resolve(), "littering", littering_group(image))
                )
    return positives


def collect_rough_sleeping(root: Path | None = None) -> list[LabeledImage]:
    """Rough-sleeping positives: kaggle laying_dataset images labelled only as
    class 0 ("laying"). Images that also contain a "standing" person are skipped
    so the positive set is unambiguous."""
    root = (root or Workspace.current().data_dir / "raw" / "datasets").resolve()
    base = root / "rough sleeping" / "kaggle" / "laying_dataset"
    image_dir, label_dir = base / "images", base / "labels"
    positives: list[LabeledImage] = []
    if not image_dir.is_dir():
        return positives
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if resolve_original(image_dir / original_name(image)) is None:
            continue
        label_path = resolve_original(
            label_dir / (Path(original_name(image)).stem + ".txt")
        )
        tokens = yolo_class_tokens(label_path) if label_path else set()
        if tokens == {"0"}:
            positives.append(
                LabeledImage(
                    image.resolve(),
                    "rough_sleeping",
                    f"rough:{original_name(image).split('.rf.')[0]}",
                )
            )
    return positives


def collect_loitering(root: Path | None = None) -> list[LabeledImage]:
    """Loitering positives: every frame of the ATMGUARD-Loitering-Tracking corpus.

    This is a dataset-level proxy, not a box-derived label -- the COCO file only
    carries "person" boxes and loitering is a temporal behaviour that cannot be
    confirmed from a single frame. The caveat is surfaced in the notebook.
    """
    root = (root or Workspace.current().data_dir / "raw" / "datasets").resolve()
    base = root / "loitering" / "Roboflow Dataset (New Workspace)"
    positives: list[LabeledImage] = []
    for split in ("train", "valid", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        for image in sorted(split_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                positives.append(
                    LabeledImage(image.resolve(), "loitering", "loit:ATM_scene_unknown")
                )
    return positives
