"""Fetch the requested public item datasets, preserving URLs, versions and file hashes.

Roboflow exports require ROBOFLOW_API_KEY and explicit version numbers. D2-CDNet
is foreground segmentation data, retained as auxiliary data, never event labels.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path

from anomaly_eval.data import file_sha256
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    resolve_arguments,
)

CAVIAR = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/"
CAVIAR_SEQUENCES = {
    "LeftBag": "lb1gt.xml",
    "LeftBag_AtChair": "lb2gt.xml",
    "LeftBag_BehindChair": "lbbcgt.xml",
    "LeftBox": "lbgt.xml",
    "LeftBag_PickedUp": "lbpugt.xml",
    "Walk1": "wk1gt.xml",
    "Walk2": "wk2gt.xml",
    "Walk3": "wk3gt.xml",
}
ROBOFLOW = {
    "roboflow_suspicious": ("abandoned-object", "abandoned-object-detection-9fijk"),
    "roboflow_objects": ("getting", "abandoned-w7arq"),
}
SOURCES = ["aboda", "caviar", "d2_cdnet", *ROBOFLOW]


def download(
    url: str,
    path: Path,
    *,
    expected_sha: str | None = None,
    expected_bytes: int | None = None,
    receipt_url: str | None = None,
) -> dict:
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_name(path.name + ".download.json")
    if path.is_file() and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        digest = file_sha256(path)
        if (
            previous.get("url") == (receipt_url or url)
            and previous.get("sha256") == digest
            and (not expected_sha or digest == expected_sha)
        ):
            return previous
    temporary = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                # Restart incomplete files: no unverified range concatenation.
                with temporary.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            out.write(chunk)
            digest = file_sha256(temporary)
            if expected_sha and digest != expected_sha:
                raise ValueError("Downloaded content differs from the source checksum")
            if (
                expected_bytes is not None
                and temporary.stat().st_size != expected_bytes
            ):
                raise ValueError("Downloaded content differs from the source size")
            temporary.replace(path)
            receipt = {
                "url": receipt_url or url,
                "path": portable_path(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            return receipt
        except (requests.RequestException, OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def extract_safe(archive: Path, destination: Path) -> None:
    """Extract regular files only, reject links, absolute paths and traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def check(name):
        target = (root / name).resolve()
        if not target.is_relative_to(root) or Path(name).is_absolute() or "\\" in name:
            raise ValueError(f"Unsafe archive member: {name}")
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                check(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains a symlink")
            z.extractall(root)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                target = check(member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as source, target.open("wb") as out:
                        import shutil

                        shutil.copyfileobj(source, out)
                else:
                    raise ValueError("Archive contains a nonregular member")


def fetch(
    root: Path, sources: list[str], d2_shards: list[str], versions: dict[str, int]
) -> dict:
    import requests

    root.mkdir(parents=True, exist_ok=True)
    receipt_file = root / "fetch_receipt.json"
    receipt = (
        json.loads(receipt_file.read_text())
        if receipt_file.is_file()
        else {"sources": {}, "schema": 1}
    )
    for source in sources:
        status = {"status": "error", "files": []}
        receipt["sources"][source] = status
        try:
            if source == "aboda":
                api = "https://api.github.com/repos/kevinlin311tw/ABODA/git/trees/master?recursive=1"
                r = requests.get(api, timeout=45)
                r.raise_for_status()
                tree = r.json()
                status.update(
                    homepage="https://github.com/kevinlin311tw/ABODA",
                    revision=tree["sha"],
                    annotation_policy="No frame annotations in the repository. Sequence review required.",
                )
                for entry in tree["tree"]:
                    if entry["type"] == "blob" and Path(entry["path"]).suffix in {
                        ".avi",
                        ".md",
                    }:
                        url = f"https://raw.githubusercontent.com/kevinlin311tw/ABODA/{tree['sha']}/{entry['path']}"
                        print(f"Fetching ABODA {entry['path']}", flush=True)
                        status["files"].append(
                            download(
                                url,
                                root / source / entry["path"],
                                expected_bytes=entry["size"],
                            )
                        )
            elif source == "caviar":
                status.update(
                    homepage=CAVIAR,
                    license="CC BY-SA; acknowledge EC CAVIAR/IST 2001 37540",
                    annotation_policy="XML roles/situations retained; item presence alone is not an unattended label.",
                )
                status["files"].append(
                    download(
                        CAVIAR + "gt_file_format.txt",
                        root / source / "gt_file_format.txt",
                    )
                )
                for sequence, annotation in CAVIAR_SEQUENCES.items():
                    for name in (sequence + ".mpg", annotation):
                        print(f"Fetching CAVIAR {name}", flush=True)
                        status["files"].append(
                            download(
                                CAVIAR + sequence + "/" + name,
                                root / source / sequence / name,
                            )
                        )
            elif source == "d2_cdnet":
                repo = "tauseefakhund/D2-CDNet"
                r = requests.get(
                    f"https://huggingface.co/api/datasets/{repo}", timeout=45
                )
                r.raise_for_status()
                revision = r.json()["sha"]
                status.update(
                    homepage=f"https://huggingface.co/datasets/{repo}",
                    revision=revision,
                    annotation_policy="Foreground masks, NOT unattended-item event labels. Auxiliary review only.",
                )
                status["files"].append(
                    download(
                        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md",
                        root / source / "README.md",
                    )
                )
                r = requests.get(
                    f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/raw_tar",
                    timeout=45,
                )
                r.raise_for_status()
                catalog = {Path(x["path"]).stem: x for x in r.json()}
                (root / source / "catalog.json").write_text(
                    json.dumps(catalog, indent=2)
                )
                for shard in d2_shards:
                    entry = catalog[shard]
                    print(
                        f"Fetching D2-CDNet {shard} ({entry['size'] / 1e9:.2f} GB)",
                        flush=True,
                    )
                    status["files"].append(
                        download(
                            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry['path']}",
                            root / source / Path(entry["path"]).name,
                            expected_sha=entry.get("lfs", {}).get("oid"),
                            expected_bytes=entry["size"],
                        )
                    )
                if not d2_shards:
                    status["note"] = (
                        "Metadata fetched. Use --d2-shards intermittentObjectMotion to fetch the 3.85 GB auxiliary shard."
                    )
            else:
                workspace, project = ROBOFLOW[source]
                status.update(
                    homepage=f"https://universe.roboflow.com/{workspace}/{project}",
                    annotation_policy="Object classes are review candidates, not event labels.",
                )
                key = os.environ.get("ROBOFLOW_API_KEY")
                if not key or source not in versions:
                    status.update(
                        status="needs_credentials",
                        note="Set ROBOFLOW_API_KEY and supply --roboflow-version source=VERSION from the export page.",
                    )
                    continue
                # Official dataset export API. Never persist keys or signed export URLs.
                version = versions[source]
                r = requests.get(
                    f"https://api.roboflow.com/{workspace}/{project}/{version}/coco",
                    params={"api_key": key},
                    timeout=60,
                )
                r.raise_for_status()
                link = r.json()["export"]["link"]
                target = root / source / f"version_{version}.zip"
                try:
                    item = download(
                        link,
                        target,
                        receipt_url=status["homepage"] + f"/dataset/{version}",
                    )
                except Exception:  # noqa: BLE001 -- redact credential-bearing download failures
                    raise RuntimeError(
                        "Roboflow export download failed; check access and selected version"
                    ) from None
                item["url"] = (
                    f"https://universe.roboflow.com/{workspace}/{project}/dataset/{version}"
                )
                target.with_name(target.name + ".download.json").write_text(
                    json.dumps(item, indent=2)
                )
                status.update(version=version)
                status["files"].append(item)
                extract_safe(target, root / source / f"version_{version}")
            status["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 -- isolate sources and redact URL credentials
            # Requests exceptions may include API keys/signed URLs; omit their text.
            status.update(
                status="error",
                error=f"{type(exc).__name__}: source fetch failed; check source availability",
            )
        finally:
            (root / "fetch_receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Workspace.current().data_dir / "additional"
    )
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    p.add_argument(
        "--d2-shards",
        nargs="*",
        default=[],
        help="Optional multi-GB foreground data shards",
    )
    p.add_argument(
        "--roboflow-version", action="append", default=[], metavar="SOURCE=VERSION"
    )
    a = resolve_arguments(p.parse_args(argv))
    versions = {}
    for value in a.roboflow_version:
        name, number = value.split("=")
        versions[name] = int(number)
        if name not in ROBOFLOW or versions[name] < 1:
            p.error("Unknown Roboflow source or invalid version")
    result = fetch(a.output.resolve(), a.sources, a.d2_shards, versions)
    for name, status in result["sources"].items():
        print(name, status["status"], status.get("note", status.get("error", "")))
    return int(any(s["status"] == "error" for s in result["sources"].values()))
