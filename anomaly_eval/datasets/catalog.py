"""Catalog-driven acquisition of the recorded research corpus.

Add sources and pinned assets in JSON/JSONL without changing Python. Metadata,
review queues and acquired bytes are distinct from prepared training examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from anomaly_eval.paths import PACKAGE_ROOT, Workspace, resolve_arguments

from .transfer import Asset, Options, TransferError, Transfers, atomic_json

DEFAULT_CATALOG = PACKAGE_ROOT / "config/dataset_sources.json"
SUCCESS = {"downloaded", "cached", "ok"}


def load_catalog(path, sources=None, profile="research"):
    """Validate the entire selected plan before writing or connecting anywhere."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    catalog = json.loads(raw)
    if catalog.get("schema") != 1:
        raise ValueError("Unsupported catalog schema")
    by_id = {row["id"]: row for row in catalog["sources"]}
    if len(by_id) != len(catalog["sources"]):
        raise ValueError("Duplicate source IDs")
    if sources == ["all"]:
        names = list(by_id)
    elif sources:
        names = sources
    else:
        names = catalog["profiles"][profile]
    if len(names) != len(set(names)) or set(names) - set(by_id):
        raise ValueError("Unknown or repeated source IDs")
    assets, selected, destinations, reserved_paths = [], [], set(), set()
    identity = hashlib.sha256(raw)
    validator = Transfers(Path("/tmp/catalog-validation"))
    for name in names:
        source = by_id[name]
        selected.append(source)
        if not source.get("manifest"):
            continue
        manifest = path.parent / source["manifest"]
        data = manifest.read_bytes()
        if (
            source.get("manifest_sha256")
            and hashlib.sha256(data).hexdigest() != source["manifest_sha256"]
        ):
            raise ValueError(f"Manifest checksum differs: {name}")
        identity.update(data)
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source", name) != name:
                raise ValueError("Manifest source differs from catalog")
            row["source"] = name
            row["urls"] = tuple(row.get("urls", ()))
            asset = Asset(**row)
            validator.target(asset)
            normalized = str(PurePosixPath(asset.path))
            sidecars = {
                str(
                    PurePosixPath(normalized).with_name(
                        "." + PurePosixPath(normalized).name + ".fetch.part"
                    )
                )
            }
            if (
                not normalized
                or normalized == "."
                or normalized in reserved_paths
                or sidecars & destinations
            ):
                raise ValueError("Duplicate or empty destination in selected manifests")
            destinations.add(normalized)
            reserved_paths.update(sidecars | {normalized})
            if asset.kind not in {"http", "gdrive", "zip_member"}:
                raise ValueError("Unsupported transfer kind")
            if not asset.urls or any(
                urlsplit(url).scheme not in {"https", "http"} or urlsplit(url).username
                for url in asset.urls
            ):
                raise ValueError(
                    "An asset requires public HTTP(S) URLs without embedded credentials"
                )
            if asset.sha256 is not None and not re.fullmatch(
                r"[a-f0-9]{64}", asset.sha256
            ):
                raise ValueError("Invalid SHA-256 pin")
            if asset.size is not None and (
                not isinstance(asset.size, int) or asset.size < 0
            ):
                raise ValueError("Invalid file size")
            if asset.kind == "gdrive" and not asset.file_id:
                raise ValueError("Drive assets need a file ID")
            if asset.kind == "zip_member" and (
                not asset.member or not asset.archive_size
            ):
                raise ValueError("ZIP assets need a member and archive size")
            assets.append(asset)
    for destination in destinations:
        if any(
            str(parent) in reserved_paths
            for parent in PurePosixPath(destination).parents
            if str(parent) != "."
        ):
            raise ValueError(
                "A file destination conflicts with another file or its sidecar directory"
            )
    return catalog, selected, assets, identity.hexdigest()


def run_assets(assets, root, options, *, report=None, catalog_sha256=None):
    """Keep a small work queue; checkpoint each result even if other files fail."""
    root = Path(root).resolve()
    report = Path(report or root / "fetch_all_receipt.json")
    root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    journal = report.with_name(report.stem + f".{run_id}.journal.jsonl")
    cancel = threading.Event()
    client = Transfers(root, options, cancel)
    rows = []
    started = time.monotonic()
    last_checkpoint = started
    interrupted = False

    def snapshot():
        counts = dict(Counter(row["status"] for row in rows))
        by_source = {
            name: dict(Counter(row["status"] for row in rows if row["source"] == name))
            for name in {a.source for a in assets}
        }
        return {
            "schema": 2,
            "run_id": run_id,
            "catalog_sha256": catalog_sha256,
            "data_root": str(root),
            "options": asdict(options),
            "planned_files": len(assets),
            "processed_files": len(rows),
            "counts": counts,
            "sources": by_source,
            "files": sorted(rows, key=lambda row: (row["source"], row["path"])),
            "bytes_received_http": client.bytes_received,
            "seconds": round(time.monotonic() - started, 3),
            "interrupted": interrupted,
            "journal": str(journal),
            "complete": not interrupted
            and len(rows) == len(assets)
            and all(row["status"] in SUCCESS for row in rows),
        }

    def execute(asset):
        try:
            return client.one(asset)
        except TransferError as exc:
            return {
                "source": asset.source,
                "path": asset.path,
                "status": exc.status,
                "error": str(exc),
                "retry_after_seconds": round(exc.delay, 3),
            }
        except Exception as exc:  # noqa: BLE001 -- isolate files; never expose credential-bearing exception text
            return {
                "source": asset.source,
                "path": asset.path,
                "status": "failed",
                "error": type(exc).__name__ + ": transfer failed",
            }

    pool = ThreadPoolExecutor(max_workers=options.workers)
    iterator = iter(assets)
    pending = {}
    with journal.open("a", encoding="utf-8") as stream:

        def record(row):
            nonlocal last_checkpoint
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if time.monotonic() - last_checkpoint >= 5:
                atomic_json(report, snapshot())
                last_checkpoint = time.monotonic()
                print(
                    f"Processed {len(rows)}/{len(assets)}: {dict(Counter(r['status'] for r in rows))}",
                    flush=True,
                )

        try:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < options.workers * 2:
                    asset = next(iterator, None)
                    if asset is None:
                        exhausted = True
                    else:
                        pending[pool.submit(execute, asset)] = asset
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    record(future.result())
        except KeyboardInterrupt:
            interrupted = True
            cancel.set()
            for future in pending:
                future.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if interrupted:
                for future, asset in pending.items():
                    record(
                        future.result()
                        if not future.cancelled()
                        else {
                            "source": asset.source,
                            "path": asset.path,
                            "status": "cancelled",
                        }
                    )
            client.close()
            result = snapshot()
            atomic_json(report, result)
    return result


def run_external(source, args, versions):
    """Reuse the existing tested Drive/export providers without shell commands."""
    base = {"id": source["id"], "instructions": source.get("instructions")}
    if (
        args.cached_only
        or args.limit
        or args.retry_failed
        or args.max_download_gb is not None
    ):
        return {
            **base,
            "status": "deferred",
            "reason": "External provider cannot honor this file audit/limit/budget; select it separately",
        }
    try:
        if source.get("provider") == "drive":
            from .drive import main as drive_main

            code = drive_main(
                [
                    "--url",
                    source["homepage"],
                    "--output",
                    str(args.output / "raw"),
                    "--extract-zip",
                    "--continue-on-error",
                    "--anonymous-listing",
                    "--workers",
                    str(min(args.workers, args.per_host)),
                    "--retries",
                    str(args.retries),
                    "--request-interval",
                    "0",
                    "--name-conflicts",
                    "rename",
                ]
            )
            return {
                **base,
                "status": "ok" if code == 0 else "partial",
                "receipt": str(args.output / "raw/download_receipt.json"),
            }
        if source.get("provider") == "roboflow":
            from .fetch import fetch

            row = fetch(args.output / "additional", [source["id"]], [], versions)[
                "sources"
            ][source["id"]]
            return {
                **base,
                "status": row["status"],
                "note": row.get("note", row.get("error")),
            }
        return {**base, "status": "requires_separate_fetch"}
    except (Exception, SystemExit) as exc:  # noqa: BLE001 -- external source isolation and credential redaction
        return {
            **base,
            "status": "failed",
            "error": type(exc).__name__ + ": external source fetch failed",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Workspace.current().data_dir,
        help="Data root; historical relative paths are preserved",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Catalog source IDs, or all (including manual/credential sources)",
    )
    parser.add_argument("--profile", default="research")
    parser.add_argument(
        "--list", action="store_true", help="List catalog sources; no network or writes"
    )
    parser.add_argument(
        "--plan",
        "--dry-run",
        dest="plan",
        action="store_true",
        help="Print the selected acquisition plan; no network or writes",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-host", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15)
    parser.add_argument("--read-timeout", type=float, default=60)
    parser.add_argument("--min-free-gb", type=float, default=0.25)
    parser.add_argument(
        "--max-download-gb",
        type=float,
        help="Conservative sum of uncached file sizes; excludes retry/ZIP-index overhead",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Audit cached files without any network requests or per-file receipt writes",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute cached SHA-256 even if the saved stat is unchanged",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected files (explicit bounded run)",
    )
    parser.add_argument(
        "--retry-failed",
        type=Path,
        metavar="RECEIPT",
        help="Retry only non-successful files from a previous receipt",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--roboflow-version", action="append", default=[], metavar="SOURCE=VERSION"
    )
    args = resolve_arguments(parser.parse_args(argv))
    versions = {}
    for value in args.roboflow_version:
        try:
            name, number = value.split("=", 1)
            if (
                name not in {"roboflow_suspicious", "roboflow_objects"}
                or int(number) < 1
            ):
                raise ValueError
            versions[name] = int(number)
        except ValueError:
            parser.error("Roboflow versions must be known SOURCE=positive integer")
    if (
        args.workers < 1
        or args.per_host < 1
        or args.retries < 0
        or args.min_free_gb < 0
        or min(args.connect_timeout, args.read_timeout) <= 0
        or (args.limit is not None and args.limit < 1)
        or (args.max_download_gb is not None and args.max_download_gb < 0)
        or not all(
            math.isfinite(value)
            for value in [
                args.min_free_gb,
                args.connect_timeout,
                args.read_timeout,
                args.max_download_gb if args.max_download_gb is not None else 0,
            ]
        )
    ):
        parser.error(
            "Concurrency, timeout and limit values must be positive; retries and budgets must be non-negative"
        )
    try:
        catalog, selected, assets, identity = load_catalog(
            args.catalog, args.sources, args.profile
        )
    except (ValueError, KeyError, TypeError, OSError, TransferError) as exc:
        parser.error(str(exc))
    if args.list:
        for row in catalog["sources"]:
            print(
                f"{row['id']:24} {row.get('availability', 'catalogued'):20} {row['license']['id']}"
            )
        return 0
    if args.retry_failed:
        previous = json.loads(args.retry_failed.read_text())
        keys = {
            (row["source"], row["path"])
            for row in previous["files"]
            if row["status"] not in SUCCESS
        }
        # A hard interruption may leave planned files absent from the receipt;
        # regular reruns cover those too. This option deliberately retries recorded failures only.
        assets = [asset for asset in assets if (asset.source, asset.path) in keys]
    if args.limit:
        assets = assets[: args.limit]
    external = [row for row in selected if not row.get("manifest")]
    if args.plan:
        print(
            json.dumps(
                {
                    "catalog_sha256": identity,
                    "data_root": str(args.output),
                    "files": len(assets),
                    "pinned_payload_bytes": sum(a.size or 0 for a in assets),
                    "unknown_size_files": sum(a.size is None for a in assets),
                    "sources": dict(Counter(a.source for a in assets)),
                    "external_sources": [
                        {"id": r["id"], "instructions": r.get("instructions")}
                        for r in external
                    ],
                },
                indent=2,
            )
        )
        return 0
    options = Options(
        workers=args.workers,
        per_host=args.per_host,
        retries=args.retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        min_free_bytes=int(args.min_free_gb * 1024**3),
        max_download_bytes=int(args.max_download_gb * 1024**3)
        if args.max_download_gb is not None
        else None,
        verify=args.verify,
        cached_only=args.cached_only,
    )
    fetch_started = time.monotonic()
    result = run_assets(
        assets, args.output, options, report=args.report, catalog_sha256=identity
    )
    result["bounded_run"] = bool(args.limit or args.retry_failed)
    result["pinned_seconds"] = result["seconds"]
    pinned_complete = result["complete"]
    result["external_sources"] = [
        {"id": row["id"], "status": "pending"} for row in external
    ]
    report = args.report or args.output / "fetch_all_receipt.json"
    if external:
        result["complete"] = False
    atomic_json(report, result)
    if not result["interrupted"]:
        try:
            for index, source in enumerate(external):
                result["external_sources"][index] = run_external(source, args, versions)
                atomic_json(report, result)
        except KeyboardInterrupt:
            result["interrupted"] = True
    result["complete"] = (
        pinned_complete
        and not result["interrupted"]
        and all(row["status"] in SUCCESS for row in result["external_sources"])
    )
    result["seconds"] = round(time.monotonic() - fetch_started, 3)
    atomic_json(report, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "planned_files",
                    "processed_files",
                    "counts",
                    "bytes_received_http",
                    "seconds",
                    "bounded_run",
                    "external_sources",
                    "complete",
                ]
            },
            indent=2,
        )
    )
    return 130 if result["interrupted"] else 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
