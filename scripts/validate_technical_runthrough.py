"""Validate and package the detailed LaTeX report without running models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/technical_runthrough_20261018"
STEM = "model_evaluation_runthrough_20261018"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUT, STEM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261018", "20261019"], default="20261018")
    args = parser.parse_args()
    OUT = ROOT / f"reports/technical_runthrough_{args.date}"
    STEM = f"model_evaluation_runthrough_{args.date}"
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    for path, value in manifest["source_files_sha256"].items():
        assert digest(ROOT / path) == value, path
    for path, value in manifest["latex_files_sha256"].items():
        assert digest(OUT / path) == value, path
    for path, value in manifest["figures_sha256"].items():
        assert digest(OUT / "figures" / path) == value, path
    for path, value in manifest["numeric_exports_sha256"].items():
        assert digest(OUT / "numeric_exports" / path) == value, path
    assert (
        digest(
            ROOT / manifest.get("builder_path", "scripts/build_technical_runthrough.py")
        )
        == manifest["builder_sha256"]
    )
    expected = {
        "strategy_all_candidates": 330,
        "pilot_all_heads": 50,
        "latest_all_policies": 38,
        "historical_audit": 15,
        "historical_vadclip_ablation": 6,
        "pilot_avss_primary": 6,
    }
    if args.date == "20261019":
        expected.update(smol_all_metrics=60, smol_history=9, smol_ap=10, smol_counts=5)
    assert all(manifest["table_rows"][k] == n for k, n in expected.items())
    assert manifest["table_rows"]["source_registry"] == len(
        manifest["source_files_sha256"]
    )
    pdf = OUT / (STEM + ".pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    for token in [
        "116/136",
        "85.3%",
        "48.5%",
        "0.9645",
        "330",
        "182",
        "right-censored",
        "provisional",
        "retrospective",
    ]:
        assert token in text, token
    assert "??" not in text, "Unresolved PDF references"
    if args.date == "20261019":
        for token in [
            "4,341,760",
            "70.8%",
            "0.1822",
            "0.6933",
            "187",
            "20,971",
            "3,792",
        ]:
            assert token in text, token
    log = (OUT / (STEM + ".log")).read_text()
    for token in [
        "Overfull",
        "Undefined control sequence",
        "undefined references",
        "Missing character",
        "LaTeX Error",
    ]:
        assert token not in log, token
    console = (OUT / "compile_console.log").read_text()
    assert "error:" not in console
    assert "already defined" not in console
    # Check all local LaTeX dependencies and every explicit narrative artifact.
    inputs = figures = evidence = 0
    for p in OUT.rglob("*.tex"):
        value = p.read_text()
        for name in re.findall(r"\\input\{([^}]+)\}", value):
            target = OUT / (name if name.endswith(".tex") else name + ".tex")
            assert target.exists(), (p, name)
            inputs += 1
        for name in re.findall(r"\\fig\{([^}]+)\}", value):
            assert (OUT / "figures" / name).exists()
            figures += 1
        if p.parent.name == "chapters":
            for name in re.findall(r"\\evidence\{([^}]+)\}", value):
                assert (ROOT / name).exists(), name
                assert name in manifest["source_files_sha256"], name
                evidence += 1
    bbox_text = subprocess.check_output(
        ["pdftotext", "-bbox", str(pdf), "-"], text=True
    )
    tree = ET.fromstring(bbox_text)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    boundaries = []
    for i, page in enumerate(tree.findall(".//x:page", ns), 1):
        w, h = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//x:word", ns):
            a = word.attrib
            if (
                float(a["xMin"]) < 18
                or float(a["xMax"]) > w - 18
                or float(a["yMin"]) < 15
                or float(a["yMax"]) > h - 15
            ):
                boundaries.append((i, word.text, a))
    assert not boundaries, boundaries[:5]
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    font_lines = fonts.splitlines()[2:]
    assert font_lines and all(
        re.search(r"\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+$", line)
        for line in font_lines
    )
    narrative_words = sum(
        len(p.read_text().split()) for p in (OUT / "chapters").glob("*.tex")
    )
    validation = {
        "status": "complete",
        "report_date": manifest["report_date"],
        "experiment_cutoff": manifest["experiment_cutoff"],
        "pdf_pages": pages,
        "narrative_source_words_approximate": narrative_words,
        "generated_tables": len(manifest["table_rows"]),
        "generated_table_rows": sum(manifest["table_rows"].values()),
        "source_files_checked": len(manifest["source_files_sha256"]),
        "latex_input_links_checked": inputs,
        "narrative_artifact_links_checked": evidence,
        "scientific_figures": figures,
        "no_unresolved_references": True,
        "no_overfull_boxes": True,
        "fonts_embedded": True,
        "text_within_page_boundaries": True,
        "no_new_model_inference": True,
        "pdf_sha256": digest(pdf),
        "source_manifest_sha256": digest(OUT / "source_manifest.json"),
        "compiler": json.loads(
            (ROOT / "data/tools/tectonic/install_receipt.json").read_text()
        ),
    }
    package_check = OUT / "source_package_check.json"
    if package_check.exists():
        check = json.loads(package_check.read_text())
        assert check["status"] == "pass"
        assert check["source_manifest_sha256"] == digest(OUT / "source_manifest.json")
        assert check["pdf_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        validation["extracted_source_package_compiled_independently"] = True
        validation["independent_pdf_text_matches"] = True
    visual = OUT / "previews/review.json"
    if visual.exists():
        review = json.loads(visual.read_text())
        assert review["pdf_sha256"] == digest(pdf)
        assert review["pages_rendered"] == pages
        validation["pages_rendered_and_reviewed_in_contact_sheets"] = pages
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    archive = ROOT / "reports" / (STEM + "_source.zip")
    included = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or any(
                v in p.parts for v in ["__pycache__", "previews", "data"]
            ):
                continue
            if p.name == "package_manifest.json":
                continue
            if (
                p.suffix in {".tex", ".json", ".csv", ".md", ".py"}
                or p.parent.name == "figures"
            ):
                name = OUT.name + "/" + str(p.relative_to(OUT))
                zipped.write(p, name)
                included.append(name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert OUT.name + "/" + STEM + ".tex" in zipped.namelist()
        for relative, expected_hash in manifest["latex_files_sha256"].items():
            assert (
                hashlib.sha256(zipped.read(OUT.name + "/" + relative)).hexdigest()
                == expected_hash
            )
        for relative, expected_hash in manifest["figures_sha256"].items():
            assert (
                hashlib.sha256(
                    zipped.read(OUT.name + "/figures/" + relative)
                ).hexdigest()
                == expected_hash
            )
    (OUT / "package_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(archive.relative_to(ROOT)),
                "sha256": digest(archive),
                "files": included,
                "pdf_in_archive": False,
                "description": "Self-contained LaTeX sources, generated tables/figures, numeric exports, provenance and compiler helpers. PDF supplied separately.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Validated {pages} PDF pages, {len(manifest['table_rows'])} tables and {len(manifest['source_files_sha256'])} source identities; packaged {len(included)} files."
    )


if __name__ == "__main__":
    main()
