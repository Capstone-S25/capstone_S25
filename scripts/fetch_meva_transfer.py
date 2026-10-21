"""Acquire annotated abandoned-package clips from the public MEVA S3 release."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/strategy_sources_20261016"
BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"


def main():
    destination = SOURCE / "meva"
    destination.mkdir(exist_ok=True)
    candidates = json.loads(
        (SOURCE / "metadata/meva_abandon_activity_files.json").read_text()
    )
    records = []
    for candidate in candidates:
        name = (
            Path(candidate["archive_path"])
            .name.replace("-activities.yml", "")
            .replace(".activities.yml", "")
        )
        intervals = []
        for item in yaml.safe_load(candidate["text"]):
            act = item.get("act", {})
            if any("abandon" in name for name in act.get("act2", {})):
                intervals.extend(span["tsr0"] for span in act["timespan"])
        if not intervals:
            continue
        date, hour = name[:10], name[11:13]
        prefix = f"drops-123-r13/{date}/{hour}/{name}"
        response = requests.get(
            BUCKET, params={"list-type": "2", "prefix": prefix}, timeout=40
        )
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [
            (
                node.findtext("s:Key", namespaces=ns),
                int(node.findtext("s:Size", namespaces=ns)),
            )
            for node in tree.findall("s:Contents", ns)
        ]
        videos = [(key, size) for key, size in keys if key.endswith(".avi")]
        if len(videos) != 1:
            records.append(
                {"name": name, "status": "missing_or_ambiguous", "matches": keys}
            )
            continue
        key, size = videos[0]
        records.append(
            {
                "name": name,
                "s3_key": key,
                "bytes": size,
                "url": BUCKET + key,
                "intervals_original_frames": intervals,
                "annotation_path": candidate["archive_path"],
                "annotation_text": candidate["text"],
                "status": "pending",
            }
        )
    (destination / "index.json").write_text(json.dumps(records, indent=2) + "\n")

    def one(row):
        if row["status"] != "pending":
            return row
        target = destination / (row["name"] + ".avi")
        try:
            if not target.exists() or target.stat().st_size != row["bytes"]:
                with requests.get(
                    row["url"], stream=True, timeout=(20, 120)
                ) as response:
                    response.raise_for_status()
                    with target.with_suffix(".part").open("wb") as stream:
                        for value in response.iter_content(1048576):
                            stream.write(value)
                target.with_suffix(".part").replace(target)
            assert target.stat().st_size == row["bytes"]
            row.update(
                status="downloaded",
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            print("downloaded", row["name"], row["bytes"], flush=True)
        except (requests.RequestException, OSError, ValueError, AssertionError) as exc:
            row.update(status="failed", error=str(exc)[:300])
        return row

    with ThreadPoolExecutor(6) as pool:
        records = list(pool.map(one, records))
    (destination / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        "MEVA FETCH COMPLETE",
        {
            status: sum(r["status"] == status for r in records)
            for status in {r["status"] for r in records}
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
