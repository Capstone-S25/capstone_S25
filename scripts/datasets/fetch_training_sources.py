"""Fetch public research sources with receipts; never execute downloaded code."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/training_sources_20261015"
STREET_URL = "https://zenodo.org/records/10870472/files/StreetScene.zip?download=1"


class HTTPRangeFile(io.RawIOBase):
    """Seekable, validated HTTP ranges, avoiding a 49 GB full ZIP download."""

    request_lock = threading.Lock()
    next_request = 0.0

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.buffer_start, self.buffer = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        start = self.position
        if self.buffer_start <= start and end <= self.buffer_start + len(self.buffer):
            self.position = end
            return self.buffer[start - self.buffer_start : end - self.buffer_start]

        fetch_end = min(self.size, max(end, start + 524288))
        for attempt in range(7):
            # Dataset hosting limits only; model training/inference is unpaced.
            with self.request_lock:
                delay = max(0, HTTPRangeFile.next_request - time.monotonic())
                HTTPRangeFile.next_request = (
                    max(time.monotonic(), HTTPRangeFile.next_request) + 0.6
                )
            if delay:
                time.sleep(delay)
            response = self.session.get(
                self.url,
                headers={"Range": f"bytes={start}-{fetch_end - 1}"},
                timeout=(20, 120),
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            delay = float(response.headers.get("Retry-After", min(30, 2**attempt)))
            response.close()
            time.sleep(delay)
        response.raise_for_status()
        expected = f"bytes {start}-{fetch_end - 1}/{self.size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected
        ):
            raise ValueError("Server did not honor the requested ZIP byte range")
        value = response.content
        if len(value) != fetch_end - start:
            raise ValueError("Incomplete ZIP range")
        self.buffer_start, self.buffer = start, value
        self.position = end
        return value[: end - start]


def street_index():
    destination = DEST / "street_scene"
    destination.mkdir(parents=True, exist_ok=True)
    record = fetch(
        "https://zenodo.org/api/records/10870471", destination / "record.json"
    )
    info = json.loads(record.read_text())["files"][0]
    with zipfile.ZipFile(HTTPRangeFile(STREET_URL, info["size"])) as archive:
        rows = [
            {"name": x.filename, "size": x.file_size, "compressed": x.compress_size}
            for x in archive.infolist()
        ]
        (destination / "zip_index.json").write_text(json.dumps(rows, indent=2) + "\n")
        textfiles = [
            x
            for x in archive.infolist()
            if x.filename.lower().endswith((".txt", ".csv", ".md"))
        ]
        print(
            "ZIP entries", len(rows), "text files", len(textfiles), rows[:4], flush=True
        )
        for entry in textfiles:
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path traversal")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
        print("Text samples", [x.filename for x in textfiles[:12]], flush=True)
    fetch(
        "https://www.merl.com/publications/docs/TR2019-078.pdf",
        destination / "paper.pdf",
    )


# Paper Tables 4/5 use 1-based anomaly indices; released files use 0-based IDs.
LOITER_TRACKS = {
    2: [2, 3, 4],
    5: [1, 3, 4, 6, 7, 10, 11, 12, 13],
    11: [1],
    12: [1, 2, 5],
    13: [2, 5, 7, 8, 10],
    17: [2, 3, 8, 10, 12],
    18: [5, 8, 11],
    21: [7],
    22: [1, 3, 4],
    26: [5],
    30: [2],
    35: [3, 5],
}


def street_frames():
    import threading

    destination = DEST / "street_scene"
    records, required = [], set()
    all_entries = json.loads((destination / "zip_index.json").read_text())
    available = {r["name"] for r in all_entries}
    for video in range(1, 36):
        tag = f"Test{video:03d}"
        base = f"StreetScene/Test/{tag}"
        gt = destination / base / (tag + "_gt.txt")
        positives, all_anomalies = set(), set()
        tracks = {x - 1 for x in LOITER_TRACKS.get(video, [])}
        for row in gt.read_text().splitlines():
            name, track, *_ = row.split()
            frame = int(Path(name).stem)
            all_anomalies.add(frame)
            if int(track) in tracks:
                positives.add(frame)
        indices = sorted(
            int(Path(r["name"]).stem)
            for r in all_entries
            if r["name"].startswith(base + "/") and r["name"].endswith(".jpg")
        )
        # A 15-second grid avoids treating near-identical windows as new events.
        # Include the end of each target track to retain short labelled events.
        target_ends = {}
        for row in gt.read_text().splitlines():
            name, track, *_ = row.split()
            if int(track) in tracks:
                target_ends[track] = int(Path(name).stem)
        for frame in sorted(set(indices[::225]) | set(target_ends.values())):
            label = int(frame in positives)
            if not label and frame in all_anomalies:
                continue  # Other anomaly annotations are not target-negative labels.
            # Require a full 20-second past history; no future frames.
            if frame < 301:
                continue
            views = [
                max(1, frame - 300),
                max(1, frame - 200),
                max(1, frame - 100),
                frame,
            ]
            names = [f"{base}/{v:05d}.jpg" for v in views]
            if not all(name in available for name in names):
                continue
            required.update(names)
            records.append(
                {
                    "task": "loitering",
                    "target": label,
                    "source": "street_scene",
                    "group_id": "street:" + tag,
                    "views": [str((destination / n).relative_to(ROOT)) for n in names],
                    "source_id": f"{tag}@{frame}",
                    "evidence": "published_track_annotations_and_anomaly_type_table",
                    "qualification": "Supervised research re-split of original Test videos; single camera; not the official unsupervised benchmark",
                }
            )
    (destination / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
    thread_local = threading.local()
    size = json.loads((destination / "record.json").read_text())["files"][0]["size"]

    def one(name):
        path = destination / name
        if path.exists() and path.stat().st_size:
            return
        if not hasattr(thread_local, "archive"):
            thread_local.archive = zipfile.ZipFile(HTTPRangeFile(STREET_URL, size))
        value = thread_local.archive.read(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    print(
        "Street sampled windows",
        len(records),
        "unique frames",
        len(required),
        flush=True,
    )
    # Spread transfers over recordings so a failed transfer does not select only early videos.
    by_video = {}
    for name in sorted(required):
        by_video.setdefault(str(Path(name).parent), []).append(name)
    order = []
    for i in range(max(map(len, by_video.values()))):
        order.extend(names[i] for names in by_video.values() if i < len(names))
    with ThreadPoolExecutor(3) as pool:
        for i, _ in enumerate(pool.map(one, order), 1):
            if i % 200 == 0 or i == len(required):
                print("Street frames", i, "/", len(required), flush=True)


def commons_candidates():
    """Candidate retrieval only: categories/search terms never create labels."""
    import re

    destination = DEST / "commons_review"
    destination.mkdir(parents=True, exist_ok=True)
    queries = [
        "sleeping people outdoors filetype:bitmap",
        "sleeping street filetype:bitmap",
        "sleeping sidewalk filetype:bitmap",
        "sleeping bench filetype:bitmap",
        "sleeping blanket outdoor filetype:bitmap",
        "people sitting bench filetype:bitmap",
        "people lying grass filetype:bitmap",
        "people sleeping bed filetype:bitmap",
    ]
    found = {}
    for query in queries:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 35,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 800,
        }
        with requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers={
                "User-Agent": "CapstoneAnomalyResearch/1.0 (research dataset audit)"
            },
            timeout=45,
        ) as response:
            response.raise_for_status()
            for page in response.json().get("query", {}).get("pages", {}).values():
                info = page.get("imageinfo", [{}])[0]
                url = info.get("thumburl", info.get("url", ""))
                if not re.search(r"\.(jpg|jpeg|png)(?:$|\?)", url, re.IGNORECASE):
                    continue
                metadata = info.get("extmetadata", {})
                license = metadata.get("LicenseShortName", {}).get("value", "")
                if not any(x in license for x in ["CC BY", "CC0", "Public domain"]):
                    continue
                key = str(page["pageid"])
                if key not in found:
                    found[key] = {
                        "id": key,
                        "title": page["title"],
                        "url": url,
                        "source_page": info.get("descriptionurl"),
                        "metadata": metadata,
                        "retrieval_queries": [],
                    }
                found[key]["retrieval_queries"].append(query)
    rows = list(found.values())
    for i, row in enumerate(rows):
        row["path"] = str(
            (destination / "images" / f"{i:03d}_{row['id']}.jpg").relative_to(ROOT)
        )
        row["review_index"] = i
    (destination / "candidates.json").write_text(json.dumps(rows, indent=2) + "\n")

    def one(row):
        try:
            fetch(row["url"], ROOT / row["path"])
        except requests.RequestException as exc:
            print("Candidate unavailable", row["id"], type(exc).__name__, flush=True)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, rows))
    print(
        "Commons candidates",
        len(rows),
        "unlabelled; requires visual review",
        flush=True,
    )


def commons_retry():
    destination = DEST / "commons_review"
    rows = json.loads((destination / "candidates.json").read_text())

    def one(row):
        try:
            fetch(row["url"], ROOT / row["path"])
        except requests.RequestException as exc:
            print("Unavailable", row["id"], type(exc).__name__, flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, rows))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".receipt.json")
    if path.exists() and receipt.exists():
        if digest(path) == json.loads(receipt.read_text())["sha256"]:
            return path
        raise ValueError(f"Changed downloaded source: {path}")
    temporary = path.with_name(path.name + ".part")
    with requests.get(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "CapstoneAnomalyResearch/1.0 (research dataset audit)",
        },
        stream=True,
        timeout=(20, 120),
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
        if (
            response.headers.get("Content-Length")
            and response.headers.get("Content-Encoding", "identity") == "identity"
        ):
            assert temporary.stat().st_size == int(response.headers["Content-Length"])
    temporary.replace(path)
    receipt.write_text(
        json.dumps(
            {"url": url, "bytes": path.stat().st_size, "sha256": digest(path)}, indent=2
        )
        + "\n"
    )
    print("Downloaded", path.relative_to(ROOT), path.stat().st_size, flush=True)
    return path


def stanford():
    destination = DEST / "stanford40"
    names = ["JPEGImages", "XMLAnnotations", "ImageSplits"]

    def one(name):
        url = f"http://vision.stanford.edu/Datasets/Stanford40_{name}.zip"
        path = fetch(url, destination / (name + ".zip"))
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Archive path traversal")
                if entry.is_dir():
                    continue
                if target.exists() and target.stat().st_size == entry.file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
        print("Extracted", name, flush=True)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one, names))
    fetch(
        "http://vision.stanford.edu/Datasets/40actions.html",
        destination / "SOURCE.html",
    )


def stanford_mirror():
    destination = DEST / "stanford40"
    path = fetch(
        "https://huggingface.co/datasets/zrchen03/Stanford40_Dataset/resolve/main/Stanford40.zip",
        destination / "Stanford40_mirror.zip",
    )
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path traversal")
            if entry.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    fetch(
        "http://vision.stanford.edu/Datasets/40actions.html",
        destination / "SOURCE.html",
    )
    print("Stanford40 mirror extracted", flush=True)


def iwdd_list():
    import gdown

    destination = DEST / "mivia_iwdd"
    destination.mkdir(parents=True, exist_ok=True)
    folders = {
        "train": "1DgXQnvK3keGP9BKB-cqpixnKaNLl2746",
        "test": "1OnA3rHn5uKAMEuHsQc11oxueTT-TQQwJ",
    }
    for split, folder in folders.items():
        path = destination / (split + "_files.json")
        if path.exists():
            print(split, "metadata cached", flush=True)
            continue
        files = gdown.download_folder(
            id=folder,
            output=str(destination / split),
            quiet=True,
            skip_download=True,
            use_cookies=False,
        )
        rows = [dict(zip(file._fields, file)) for file in files]
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(split, len(rows), rows[:4], flush=True)
    fetch(
        "https://raw.githubusercontent.com/MiviaLab/IllegalWasteDumping/main/README.md",
        destination / "SOURCE.md",
    )


def iwdd_download():
    import gdown

    destination = DEST / "mivia_iwdd"
    rows = []
    failures = []
    for split in ["train", "test"]:
        rows.extend(json.loads((destination / (split + "_files.json")).read_text()))

    def one(row):
        path = Path(row["local_path"])
        if (
            path.exists()
            and path.stat().st_size
            and path.with_name(path.name + ".receipt.json").exists()
        ):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = str(path) + ".part"
        try:
            result = gdown.download(
                id=row["id"],
                output=temporary,
                quiet=True,
                use_cookies=False,
                resume=True,
                retries=1,
                timeout=(20, 120),
            )
        except (
            gdown.exceptions.FileURLRetrievalError,
            requests.RequestException,
            OSError,
        ) as exc:
            failures.append(
                {"path": row["path"], "id": row["id"], "reason": str(exc)[:400]}
            )
            print("IWDD unavailable", row["path"], flush=True)
            return
        if result is None:
            raise RuntimeError(f"Public Drive download unavailable: {row['path']}")
        Path(temporary).replace(path)
        path.with_name(path.name + ".receipt.json").write_text(
            json.dumps(
                {
                    "google_drive_id": row["id"],
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                },
                indent=2,
            )
            + "\n"
        )

    # Fetch small annotations first so scene grouping can be fixed before training.
    for suffix in [".json", ".mp4"]:
        selected = [row for row in rows if row["path"].endswith(suffix)]
        with ThreadPoolExecutor(12 if suffix == ".json" else 6) as pool:
            for i, _ in enumerate(pool.map(one, selected), 1):
                if i % 25 == 0 or i == len(selected):
                    print("IWDD", suffix, i, "/", len(selected), flush=True)
    fetch(
        "https://raw.githubusercontent.com/MiviaLab/IllegalWasteDumping/main/README.md",
        destination / "SOURCE.md",
    )
    (destination / "unavailable_files.json").write_text(
        json.dumps(failures, indent=2) + "\n"
    )


def iwdd_direct():
    destination = DEST / "mivia_iwdd"
    rows = []
    failures = []
    for split in ["test", "train"]:
        rows.extend(
            x
            for x in json.loads((destination / (split + "_files.json")).read_text())
            if x["path"].endswith(".mp4")
        )

    def one(row):
        path = Path(row["local_path"])
        if (
            path.exists()
            and path.stat().st_size
            and path.with_name(path.name + ".receipt.json").exists()
        ):
            return
        try:
            with requests.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": row["id"]},
                headers={
                    "User-Agent": "CapstoneAnomalyResearch/1.0 (research dataset audit)"
                },
                stream=True,
                timeout=(20, 120),
            ) as response:
                response.raise_for_status()
                if not any(
                    x in response.headers.get("Content-Type", "")
                    for x in ["video/", "application/octet-stream"]
                ):
                    failures.append(
                        {
                            "path": row["path"],
                            "reason": "Public endpoint returned a permission/quota HTML page",
                        }
                    )
                    return
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".part")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                temporary.replace(path)
                path.with_name(path.name + ".receipt.json").write_text(
                    json.dumps(
                        {
                            "google_drive_id": row["id"],
                            "bytes": path.stat().st_size,
                            "sha256": digest(path),
                            "method": "anonymous public download endpoint",
                        },
                        indent=2,
                    )
                    + "\n"
                )
        except requests.RequestException as exc:
            failures.append({"path": row["path"], "reason": str(exc)[:200]})

    with ThreadPoolExecutor(2) as pool:
        for i, _ in enumerate(pool.map(one, rows), 1):
            if i % 25 == 0:
                print(
                    "Public IWDD",
                    i,
                    "/",
                    len(rows),
                    "unavailable",
                    len(failures),
                    flush=True,
                )
    (destination / "unavailable_files_direct.json").write_text(
        json.dumps(failures, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        choices=[
            "stanford40",
            "stanford40-mirror",
            "iwdd-list",
            "iwdd-download",
            "iwdd-direct",
            "street-index",
            "street-frames",
            "commons-candidates",
            "commons-retry",
        ],
    )
    args = parser.parse_args()
    {
        "stanford40": stanford,
        "stanford40-mirror": stanford_mirror,
        "iwdd-list": iwdd_list,
        "iwdd-download": iwdd_download,
        "iwdd-direct": iwdd_direct,
        "street-index": street_index,
        "street-frames": street_frames,
        "commons-candidates": commons_candidates,
        "commons-retry": commons_retry,
    }[args.source]()


if __name__ == "__main__":
    main()
