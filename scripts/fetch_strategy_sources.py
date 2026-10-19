"""Public source acquisition with immutable receipts and unlabelled review queues."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/strategy_sources_20261016"


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_name(target.name + ".receipt.json")
    if target.exists() and receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() != saved["sha256"]:
            raise ValueError("Cached public source changed: " + str(target))
        return saved
    if not target.exists():
        for attempt in range(4):
            response = requests.get(
                url,
                stream=True,
                timeout=(20, 90),
                headers={"User-Agent": "CapstoneAnomalyResearch/1.0 dataset research"},
            )
            if response.status_code == 429:
                delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                response.close()
                if delay > 30:
                    raise RuntimeError(
                        f"Host requested deferred retry after {delay} seconds"
                    )
                time.sleep(delay)
                continue
            response.raise_for_status()
            with target.with_suffix(target.suffix + ".part").open("wb") as stream:
                for chunk in response.iter_content(1048576):
                    stream.write(chunk)
            target.with_suffix(target.suffix + ".part").replace(target)
            response.close()
            break
    receipt = {
        "url": url,
        "path": str(target.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def public_small():
    urls = {
        "smoking_kaggle.zip": "https://www.kaggle.com/api/v1/datasets/download/sujaykapadnis/smoking"
    }
    for name in [
        "annotations_loitering.csv",
        "gt_df.csv",
        "train_df.csv",
        "test_df.csv",
        "X_train.csv",
        "X_test.csv",
    ]:
        urls["thermal/" + name] = (
            "https://raw.githubusercontent.com/johnnynunez/RS-WACV24_Loitering/main/dataset/"
            + name
        )
    receipts = [fetch(url, OUT / name) for name, url in urls.items()]
    (OUT / "small_sources_receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )


def commons():
    destination = OUT / "commons_candidates"
    destination.mkdir(exist_ok=True)
    old = json.loads(
        (
            ROOT / "data/training_sources_20261015/commons_review/candidates.json"
        ).read_text()
    )
    seen = {str(r["id"]) for r in old}
    queries = [
        "sleeping street",
        "sleeping sidewalk",
        "sleeping bench",
        "sleeping blanket outdoor",
        "sleeping pavement",
        "people resting park",
        "people lying grass",
        "people sleeping bed",
    ]
    found = {}
    for query in queries:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query + " filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "gsroffset": 25,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 800,
        }
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            timeout=45,
            headers={"User-Agent": "CapstoneAnomalyResearch/1.0 dataset research"},
        )
        response.raise_for_status()
        for page in response.json().get("query", {}).get("pages", {}).values():
            key = str(page["pageid"])
            info = page.get("imageinfo", [{}])[0]
            url = info.get("thumburl", info.get("url", ""))
            metadata = info.get("extmetadata", {})
            license_name = metadata.get("LicenseShortName", {}).get("value", "")
            if (
                key in seen
                or not any(s in license_name for s in ["CC BY", "CC0", "Public domain"])
                or not re.search(r"\.(jpg|jpeg|png)(?:$|\?)", url, re.IGNORECASE)
            ):
                continue
            if key not in found:
                found[key] = {
                    "id": key,
                    "title": page["title"],
                    "url": url,
                    "source_page": info.get("descriptionurl"),
                    "metadata": metadata,
                    "retrieval_queries": [],
                    "label": None,
                    "qualification": "Unlabelled candidate; search/category does not establish visible sleeping",
                }
            found[key]["retrieval_queries"].append(query)
    rows = list(found.values())

    def one(row):
        target = destination / "images" / (row["id"] + ".jpg")
        try:
            row["receipt"] = fetch(row["url"], target)
            row["path"] = str(target.relative_to(ROOT))
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            row["download_error"] = str(exc)[:250]
        return row

    (destination / "candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, rows))
    (destination / "candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        "COMMONS CANDIDATES",
        len(rows),
        "downloaded",
        sum("path" in r for r in rows),
        "still unlabelled",
        flush=True,
    )


def commons_human():
    destination = OUT / "commons_human"
    rows = json.loads((destination / "candidates.json").read_text())
    stop = threading.Event()
    lock = threading.Lock()
    next_request = [0.0]

    def one(row):
        if stop.is_set():
            row["download_status"] = "deferred_host_retry"
            return row
        with lock:
            delay = max(0, next_request[0] - time.monotonic())
            next_request[0] = max(time.monotonic(), next_request[0]) + 0.4
        if delay:
            time.sleep(delay)
        try:
            target = destination / "images" / (row["id"] + ".jpg")
            row["receipt"] = fetch(row["thumburl"], target)
            row["path"] = str(target.relative_to(ROOT))
        except RuntimeError as exc:
            stop.set()
            row["download_status"] = str(exc)
        except (requests.RequestException, OSError, ValueError) as exc:
            row["download_status"] = str(exc)[:250]
        return row

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, rows))
    (destination / "downloaded.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        "HUMAN CATEGORY PHOTOS",
        len(rows),
        "downloaded",
        sum("path" in r for r in rows),
        "unlabelled pending visual review",
        flush=True,
    )


def flickr_copies():
    destination = OUT / "commons_human"
    rows = json.loads((destination / "downloaded.json").read_text())
    for row in rows:
        if "path" in row:
            continue
        text = (
            row["metadata"].get("Credit", {}).get("value", "")
            + " "
            + row["metadata"].get("ImageDescription", {}).get("value", "")
        )
        urls = re.findall(
            r'https?://(?:www\.)?flickr.com/(?:photos|gp)/[^\s"<>]+', text
        )
        if not urls:
            continue
        try:
            response = requests.get(
                "https://www.flickr.com/services/oembed",
                params={"url": urls[0], "format": "json"},
                timeout=(15, 30),
            )
            response.raise_for_status()
            metadata = response.json()
            if not any(
                s in metadata.get("license", "")
                for s in ["CC BY", "CC0", "Public domain"]
            ):
                continue
            row["flickr_oembed"] = metadata
            target = destination / "images" / (row["id"] + ".jpg")
            row["receipt"] = fetch(metadata["url"], target)
            row["path"] = str(target.relative_to(ROOT))
            row["download_status"] = (
                "Downloaded author-linked public Flickr copy; matching open license verified"
            )
            print("flickr copy", row["id"], flush=True)
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            row["flickr_error"] = str(exc)[:250]
        time.sleep(0.5)
    (destination / "downloaded_with_flickr.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(
        "AUTHOR COPIES COMPLETE",
        sum("path" in r for r in rows),
        "downloaded human candidates",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["public-small", "commons", "commons-human", "flickr-copies"]
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    {
        "public-small": public_small,
        "commons": commons,
        "commons-human": commons_human,
        "flickr-copies": flickr_copies,
    }[args.command]()
