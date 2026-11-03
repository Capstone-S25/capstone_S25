"""Generate source/license tables and file-level attribution from the catalog."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anomaly_eval.datasets.catalog import DEFAULT_CATALOG, load_catalog
from anomaly_eval.datasets.transfer import atomic_json, digest


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(catalog_path, output):
    catalog, sources, assets, _ = load_catalog(catalog_path, ["all"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_by_id = {row["id"]: row for row in sources}
    rows = []
    for source in sources:
        rows.append(
            {
                "id": source["id"],
                "title": source["title"],
                "homepage": source["homepage"],
                "license": source["license"]["id"],
                "license_url": source["license"].get("url"),
                "license_evidence": source["license"].get("evidence_url"),
                "license_notes": source["license"].get("notes"),
                "checked_on": source["license"].get("checked_on"),
                "revision": source.get("revision"),
                "asset_count": source.get("asset_count", 0),
                "pinned_payload_bytes": source.get("pinned_payload_bytes", 0),
                "purpose": source.get("purpose"),
                "manifest": source.get("manifest"),
                "instructions": source.get("instructions"),
            }
        )
    write_csv(output / "dataset_sources.csv", list(rows[0]), rows)
    downloads = []
    attribution = []
    for asset in assets:
        source = source_by_id[asset.source]
        attrs = asset.attribution
        for index, url in enumerate(asset.urls):
            downloads.append(
                {
                    "source": asset.source,
                    "path": asset.path,
                    "url": url,
                    "mirror_index": index,
                    "sha256": asset.sha256,
                    "bytes": asset.size,
                    "kind": asset.kind,
                    "archive_member": asset.member,
                    "source_page": attrs.get("source_page") or source["homepage"],
                    "license": attrs.get("LicenseShortName") or source["license"]["id"],
                    "license_url": attrs.get("LicenseUrl")
                    or source["license"].get("url"),
                }
            )
        if attrs:
            attribution.append(
                {
                    "path": asset.path,
                    "sha256": asset.sha256,
                    "download_url": asset.urls[0],
                    **attrs,
                }
            )
    write_csv(output / "dataset_downloads.csv", list(downloads[0]), downloads)
    if attribution:
        fields = list(dict.fromkeys(k for row in attribution for k in row))
        write_csv(output / "commons_attribution.csv", fields, attribution)
    lines = [
        "# Dataset sources, downloads and licenses",
        "",
        (
            f"Catalog reviewed on {catalog['updated_on']}. The canonical machine-readable registry is "
            "[`anomaly_eval/config/dataset_sources.json`](../anomaly_eval/config/dataset_sources.json). "
            "It includes author homepages, versions, download manifests, license evidence and limitations."
        ),
        "",
        (
            "The later additions include Stanford40, MIVIA IWDD, Street Scene, smoking image transfer data, "
            "thermal trajectories, Commons/Flickr photographs, HMDB51, MEVA, AVSS, small-item component footage and "
            "the XRF55 RGB public pilot. Acquiring files does not make all of them eligible labeled training examples."
        ),
        "",
        (
            f"The catalog records **{len(sources)} source entries** and **{len(assets):,} pinned file assets** "
            f"(**{sum(a.size or 0 for a in assets):,} bytes** of file payload). "
            "These are acquisition artifacts, not independent examples. Archives may contain unused classes or duplicate mirrors; "
            "Street Scene entries are selected frames from related videos. The original Drive source and gated Roboflow exports "
            "use separate live providers, so their contents are not included in this pinned-file count."
        ),
        "",
        "## Consolidated tables",
        "",
        "- [Source and license CSV](dataset_sources.csv): one row per source.",
        "- [All pinned download links and hashes](dataset_downloads.csv): one row per file/mirror, with source page and license.",
        (
            "- [Commons/Flickr author and license attribution](commons_attribution.csv): one row per acquired photograph, "
            "including artist/credit text and original HTML links, original page, license URL, restrictions and exact byte hash."
        ),
        "- [Downloader usage and recovery](dataset_fetching.md).",
        "",
        "## Source inventory",
        "",
        "| Source | Recorded assets | License declaration / evidence | Use |",
        "| --- | ---: | --- | --- |",
    ]
    for source in sources:
        lic = source["license"]
        link = lic.get("url") or lic.get("evidence_url") or source["homepage"]
        lines.append(
            f"| [{source['title']}]({source['homepage']}) | {source.get('asset_count', 'live provider')} | "
            f"[{lic['id']}]({link}) | {source.get('purpose', '').replace('|', '/')} |"
        )
    lines += [
        "",
        "## License scope and attribution",
        "",
        (
            "A public URL is not a license grant. Entries marked `not_specified`, `not_verified` or `upstream unresolved` "
            "retain that uncertainty. A repository code license or mirror declaration does not establish rights to an "
            "underlying video collection. The registry records source statements and evidence; it does not grant new rights."
        ),
        "",
        (
            "Per-file Commons terms take precedence over the aggregate row. Preserve the creator, original source page and "
            "license link when reusing a photograph; record crop/resize or other modifications in downstream exports. "
            "The acquisition manifest describes original downloaded bytes. Review labels and model predictions are project artifacts."
        ),
        "",
    ]
    for source in sources:
        lines += [f"### {source['title']}", "", source["license"].get("notes", ""), ""]
        if source.get("citation"):
            lines += ["Requested/reference citation: " + source["citation"], ""]
        for field in [
            "revision",
            "mirrors",
            "author_repository",
            "annotation_repository",
            "annotation_revision",
            "license_snapshot",
            "annotation_snapshot",
            "instructions",
        ]:
            if source.get(field):
                value = source[field]
                lines += [
                    f"{field.replace('_', ' ').capitalize()}: {', '.join(value) if isinstance(value, list) else value}",
                    "",
                ]
        if source.get("unavailable_files"):
            lines += [
                f"Historical unavailable files: {len(source['unavailable_files'])}; original public links remain recorded in the JSON registry.",
                "",
            ]
        for component in source.get("upstream_components", []):
            name = component["name"]
            if component.get("url"):
                name = f"[{name}]({component['url']})"
            lines += [
                (
                    f"- {name}: {component['license']}. {component.get('qualification', '')} "
                    f"Evidence: {component.get('evidence', 'export notices')}"
                )
            ]
        if source.get("upstream_components"):
            lines.append("")
    lines += [
        "## Provenance and updating",
        "",
        catalog["checksum_basis"],
        "",
        (
            "Historical date-specific fetch scripts and scientific receipts are preserved. The unified downloader writes "
            "new receipts under `DATA_ROOT/.fetch_state/` and keeps the previous `.receipt.json` / `.download.json` files. "
            "XRF55 release terms, Kaggle API license metadata, D2-CDNet mirror terms and original MEVA activity annotations "
            "are retained under `anomaly_eval/config/dataset_evidence/`."
        ),
        "",
        (
            "After reviewing edits to the catalog/manifests, regenerate these tables with "
            "`python scripts/datasets/export_dataset_sources.py`. Use `--update-manifest-hashes` only when intentionally publishing "
            "a revised acquisition plan. This does not revise historical experiment splits or results."
        ),
        "",
    ]
    (output / "dataset_sources.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "sources": len(sources),
        "assets": len(assets),
        "photographs": len(attribution),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[2] / "docs"
    )
    parser.add_argument("--update-manifest-hashes", action="store_true")
    args = parser.parse_args()
    if args.update_manifest_hashes:
        value = json.loads(args.catalog.read_text())
        for source in value["sources"]:
            if source.get("manifest"):
                path = args.catalog.parent / source["manifest"]
                rows = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                source.update(
                    manifest_sha256=digest(path),
                    asset_count=len(rows),
                    pinned_payload_bytes=sum(row.get("size") or 0 for row in rows),
                )
        atomic_json(args.catalog, value)
    print(json.dumps(export(args.catalog, args.output), indent=2))


if __name__ == "__main__":
    main()
