"""Export a measured RAM report to PDF with camera-scaling and queue charts.

Use uv run --locked --with reportlab --with matplotlib python -m
benchmarks.memory_pdf <run-directory> --output <report.pdf>.
"""

import argparse
import csv
import io
import re
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from benchmarks.memory_report import write_report

WIDTH = A4[0] - 36 * mm
MIB = 1024 * 1024
NAVY = colors.HexColor("#15334B")
TEAL = colors.HexColor("#087F8C")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleRAM",
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=NAVY,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "HeadingRAM",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=9,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyRAM",
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.4,
            textColor=NAVY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletRAM", parent=styles["BodyRAM"], leftIndent=10, bulletIndent=0
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallRAM",
            parent=styles["BodyRAM"],
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#516473"),
        )
    )
    styles.add(
        ParagraphStyle(
            "CellRAM", parent=styles["BodyRAM"], fontSize=8, leading=10.5, spaceAfter=0
        )
    )
    styles.add(
        ParagraphStyle(
            "HeaderRAM",
            parent=styles["CellRAM"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    return styles


def _inline(text):
    value = escape(text)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    return re.sub(r"`([^`]+)`", r"<font face='Courier' size='8'>\1</font>", value)


def _table(rows, styles):
    weights = [1.0] * len(rows[0])
    weights[0] = 1.8
    if rows[0][-1] in ("Accounting", "Production treatment"):
        weights[-1] = 3
    if rows[0][0] == "Component":
        weights = [2, 1]
    if rows[0][1] == "Cameras":
        weights[1] = 0.55
    cells = [
        [
            Paragraph(_inline(cell), styles["HeaderRAM" if index == 0 else "CellRAM"])
            for cell in row
        ]
        for index, row in enumerate(rows)
    ]
    table = Table(
        cells,
        colWidths=[WIDTH * weight / sum(weights) for weight in weights],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#EEF5F7"), colors.white],
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#D3DFE5")),
            ]
        )
    )
    return table


def _image(figure, height):
    buffer = io.BytesIO()
    figure.savefig(
        buffer, format="png", dpi=190, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=WIDTH, height=height)


def _scaling_chart(data):
    by_key = {(r["mode"], r["cameras"]): r for r in data["scenarios"]}
    rows = [by_key[("full", n)] for n in (1, 2, 4)]
    figure, axis = plt.subplots(figsize=(7.1, 2.6), layout="constrained")
    axis.bar(
        [i - 0.18 for i in range(3)],
        [r["pipeline_mean_mib"] for r in rows],
        width=0.34,
        color="#087F8C",
        label="Typical application RAM",
    )
    bars = axis.bar(
        [i + 0.18 for i in range(3)],
        [r["pipeline_peak_mib"] for r in rows],
        width=0.34,
        color="#15334B",
        label="Peak application RAM",
    )
    peak = max(r["pipeline_peak_mib"] for r in rows)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            row["pipeline_peak_mib"] + peak * 0.025,
            f"{row['pipeline_peak_mib']:.1f}",
            ha="center",
            fontsize=8,
        )
    axis.axhline(
        by_key[("baseline", 0)]["pipeline_mean_mib"],
        color="#8295A3",
        linestyle="--",
        linewidth=1,
        label="Shared Python baseline",
    )
    axis.set(
        xticks=range(3),
        xticklabels=["1 camera", "2 cameras", "4 cameras"],
        ylabel="Application RAM (MiB)",
        ylim=(0, peak * 1.2),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.15)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2
    )
    return _image(figure, 65 * mm)


def _time_chart(root, run):
    with (root / run["case"] / "process.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    elapsed = [float(r["elapsed_seconds"]) for r in rows]
    memory = [float(r["app_rss_bytes"]) / MIB for r in rows]
    figure, axis = plt.subplots(figsize=(7.1, 2.4), layout="constrained")
    axis.plot(elapsed, memory, color="#15334B", linewidth=1.3, label="Application RAM")
    axis.plot(
        elapsed,
        [float(r["queued_image_bytes"]) / MIB for r in rows],
        color="#CE7D27",
        label="Queued image data",
    )
    axis.axvspan(0, run["warmup_seconds"], color="#DDE7EB", alpha=0.6, label="Warm-up")
    axis.axvline(
        run["warmup_seconds"] + run["measurement_seconds"],
        color="#087F8C",
        linestyle="--",
        linewidth=1,
        label="Stop input / drain",
    )
    axis.set(
        xlabel="Elapsed seconds, including startup and drain",
        ylabel="MiB",
        ylim=(0, max(memory) * 1.12),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    axis.legend(
        frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2
    )
    return _image(figure, 61 * mm)


def write_pdf(root, output):
    root, output = Path(root), Path(output)
    data = write_report(root)
    styles = _styles()
    lines = (root / "report.md").read_text(encoding="utf-8").splitlines()
    story, section, index = [], "", 0
    page_sections = {
        "Pipeline RAM and camera scaling",
        "Whole-system RAM",
        "Production ingestion accounting estimate",
        "Queue stability and stress",
        "Workload, method and limits",
    }
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), styles["TitleRAM"]))
        elif line.startswith("## "):
            section = line[3:]
            if section in page_sections:
                story.append(PageBreak())
            story.append(Paragraph(_inline(section), styles["HeadingRAM"]))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            story.extend([_table(rows, styles), Spacer(1, 8)])
            if section == "Pipeline RAM and camera scaling":
                story.extend([_scaling_chart(data), Spacer(1, 8)])
            elif section == "Queue stability and stress" and rows[0][0] == "Case":
                for run in data["special_runs"]:
                    story.append(
                        KeepTogether(
                            [
                                Paragraph(
                                    f"Four cameras: {escape(run['label'].lower())}",
                                    styles["HeadingRAM"],
                                ),
                                _time_chart(root, run),
                            ]
                        )
                    )
            continue
        elif line.startswith("- "):
            story.append(
                Paragraph(_inline(line[2:]), styles["BulletRAM"], bulletText="•")
            )
        else:
            story.append(Paragraph(_inline(line), styles["BodyRAM"]))
        index += 1

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D3DFE5"))
        canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#516473"))
        canvas.drawString(
            18 * mm,
            10 * mm,
            f"Bounded-queue RAM benchmark | {data['date']} | No model inference",
        )
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=21 * mm,
        title=f"Pipeline RAM benchmark - {data['date']}",
        author="Data pipeline performance study",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_pdf(args.root, args.output)
    print(args.output.resolve())
