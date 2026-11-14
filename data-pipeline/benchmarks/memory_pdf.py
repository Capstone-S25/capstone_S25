"""Export benchmark results to PDF using ReportLab and Matplotlib.

Run with: uv run --locked --with reportlab --with matplotlib python -m
benchmarks.memory_pdf <run-directory> --output <report.pdf>
"""

import argparse
import csv
import io
import json
from html import escape
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from benchmarks.memory_report import _aggregate, _direct_camera_services

WIDTH = A4[0] - 36 * mm
NAVY = colors.HexColor("#15334B")
TEAL = colors.HexColor("#087F8C")
GRAY = colors.HexColor("#516473")
MIB = 1024 * 1024


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "Subsection",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Copy",
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            textColor=NAVY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallCopy",
            parent=styles["Copy"],
            fontSize=8.2,
            leading=11,
            textColor=GRAY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Cell", parent=styles["Copy"], fontSize=8, leading=10.2, spaceAfter=0
        )
    )
    styles.add(
        ParagraphStyle(
            "HeaderCell",
            parent=styles["Cell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle("Eyebrow", fontSize=8, leading=11, textColor=TEAL, spaceAfter=8)
    )
    styles.add(
        ParagraphStyle(
            "Metric",
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=TEAL,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle("MetricLabel", parent=styles["SmallCopy"], alignment=TA_CENTER)
    )
    return styles


def _p(text, styles, style="Copy"):
    return Paragraph(text, styles[style])


def _heading(number, title, styles):
    return [
        _p(f"DATA PIPELINE  /  MEMORY STUDY  /  {number}", styles, "Eyebrow"),
        _p(title, styles, "Section"),
    ]


def _table(headers, rows, weights, styles):
    data = [[_p(escape(str(cell)), styles, "HeaderCell") for cell in headers]]
    data += [[_p(escape(str(cell)), styles, "Cell") for cell in row] for row in rows]
    table = Table(
        data,
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
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
                ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#D3DFE5")),
            ]
        )
    )
    return table


def _figure(figure, height):
    buffer = io.BytesIO()
    figure.savefig(
        buffer, format="png", dpi=190, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=WIDTH, height=height)


def _scaling_chart(by_key):
    figure, ax = plt.subplots(figsize=(7.1, 2.7), layout="constrained")
    x = list(range(3))
    means = [by_key[("full", n)]["app_rss_mean_mib"] for n in (1, 2, 4)]
    peaks = [by_key[("full", n)]["app_rss_peak_mib"] for n in (1, 2, 4)]
    ax.bar(
        [v - 0.18 for v in x],
        means,
        width=0.34,
        color="#087F8C",
        label="Typical steady RSS",
    )
    bars = ax.bar(
        [v + 0.18 for v in x],
        peaks,
        width=0.34,
        color="#15334B",
        label="Highest sampled RSS",
    )
    for bar, value in zip(bars, peaks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
    ax.axhline(
        by_key[("baseline", 0)]["app_rss_mean_mib"],
        color="#8295A3",
        linestyle="--",
        linewidth=1,
        label="Shared Python baseline",
    )
    ax.set(
        xticks=x,
        xticklabels=["1 camera", "2 cameras", "4 cameras"],
        ylabel="Summed RSS (MiB)",
        ylim=(0, 230),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _figure(figure, 66 * mm)


def _time_chart(root, run, *, slow):
    with (root / run["case"] / "process.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    x = [float(row["elapsed_seconds"]) for row in rows]
    rss = [float(row["app_rss_bytes"]) / MIB for row in rows]
    figure, ax = plt.subplots(figsize=(7.1, 2.35), layout="constrained")
    ax.plot(x, rss, color="#15334B", linewidth=1.3, label="Application RSS")
    if slow:
        ax.plot(
            x,
            [float(row["queued_image_bytes"]) / MIB for row in rows],
            color="#CE7D27",
            linewidth=1.5,
            label="Queued image payload",
        )
    ax.axvspan(0, run["warmup_seconds"], color="#DDE7EB", alpha=0.6, label="Warm-up")
    ax.axvline(
        run["warmup_seconds"] + run["measurement_seconds"],
        color="#087F8C",
        linestyle="--",
        linewidth=1,
        label="Stop input / drain",
    )
    ax.set(
        xlabel="Elapsed seconds (including startup and drain)",
        ylabel="MiB",
        ylim=(0, 350 if slow else 225),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="lower left",
        ncols=2,
        bbox_to_anchor=(0, 1.02),
        borderaxespad=0,
    )
    return _figure(figure, 56 * mm)


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D3DFE5"))
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(
        18 * mm,
        10 * mm,
        "Pipeline memory benchmark | 14 November 2026 | 640 x 360 | No model inference",
    )
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def write_pdf(root, output):
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if any(run["status"] != "completed" for run in results):
        raise ValueError("PDF input contains failed benchmark runs")
    aggregate = _aggregate(results)
    by_key = {(row["mode"], row["cameras"]): row for row in aggregate}
    baseline = by_key[("baseline", 0)]
    four = by_key[("full", 4)]
    production = _direct_camera_services(results)
    soak = next(run for run in results if "_soak" in run["case"])
    slow = next(run for run in results if run["consumer_delay_seconds"])
    queue_size = results[0].get("queue_size", 0)
    queue_description = (
        f"bounded drop-oldest queue with {queue_size} waiting frames"
        if queue_size
        else "historical unbounded FIFO queue"
    )
    slow_dropped = sum(c.get("dropped_frames", 0) for c in slow["cameras_detail"])
    queue_conclusion = (
        f"The bounded queues dropped {slow_dropped} oldest frames on overflow."
        if queue_size
        else "A continuously growing unbounded queue has no fixed memory ceiling."
    )
    fast_peak = max(
        run["metrics"]["app_rss_peak_mib"]
        for run in results
        if run["mode"] == "full"
        and run["cameras"] == 4
        and not run["consumer_delay_seconds"]
    )
    styles = _styles()
    story = [
        _p("PERFORMANCE REPORT  /  14 NOVEMBER 2026", styles, "Eyebrow"),
        _p("Pipeline memory<br/>benchmark", styles, "ReportTitle"),
    ]
    story.append(
        _p(
            "Component costs, camera-count scaling, Docker overhead and frame-queue behavior on one Windows machine.",
            styles,
        )
    )
    cards = Table(
        [
            [
                _p(f"{fast_peak:.0f} MiB", styles, "Metric"),
                _p(f"{fast_peak * 1.25:.0f} MiB", styles, "Metric"),
                _p(
                    f"{slow['metrics']['queued_image_late_slope_mib_per_minute']:.1f}",
                    styles,
                    "Metric",
                ),
            ],
            [
                _p("4-camera fast-consumer peak", styles, "MetricLabel"),
                _p("App-only allowance, +25%", styles, "MetricLabel"),
                _p("MiB/min queue growth when slow", styles, "MetricLabel"),
            ],
        ],
        colWidths=[WIDTH / 3] * 3,
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF5F7")),
                ("TOPPADDING", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
            ]
        )
    )
    story.extend(
        [
            cards,
            Spacer(1, 10),
            _p("Application scaling: one shared Python process", styles, "Subsection"),
            _scaling_chart(by_key),
        ]
    )
    story.append(
        _p(
            "Chart: three 60-second comparison runs per camera count. The longer stability run reached 200.0 MiB; it is included in the headline peak above.",
            styles,
            "SmallCopy",
        )
    )
    story.append(_p("What the measurements establish", styles, "Subsection"))
    story.append(
        _p(
            "Recording adds about <b>22.2 MiB</b> and streaming, including decoding, about <b>10.3 MiB</b> above the shared Python baseline. Four fast-consumer cameras completed a 10-minute run without sustained queue growth.",
            styles,
        )
    )
    story.append(
        _p(
            f"Deliberately slow consumers reached <b>{slow['metrics']['app_rss_peak_mib']:.1f} MiB</b> and <b>{slow['metrics']['queued_frames_peak']} queued frames</b>. {queue_conclusion}",
            styles,
        )
    )
    story.append(
        _p(
            "Scope: H.264, 640 x 360, 25 source FPS, up to 2 selected FPS per camera. No YOLO, VLM or GPU inference. The 250 MiB allowance is for the measured application only, not a complete machine or deployment recommendation.",
            styles,
            "SmallCopy",
        )
    )

    story.append(
        _p(
            f"<b>Production adjustment:</b> Direct camera RTSP connections do not need MediaMTX or mock publishers. Excluding MediaMTX reduces the known-container reference budget from {production['test_container_total_mib']:.1f} to {production['production_known_container_estimate_mib']:.1f} MiB (PostgreSQL only). Application-only figures above are unchanged; see page 5.",
            styles,
            "SmallCopy",
        )
    )
    story.extend([PageBreak(), *_heading("02", "Workload and method", styles)])
    profile = environment["input"]
    story.append(
        _table(
            ["Setting", "Measured configuration"],
            [
                [
                    "Machine",
                    f"{environment['host_ram_mib'] / 1024:.2f} GiB physical RAM; {environment['os']}",
                ],
                [
                    "Runtime",
                    f"Python {environment['python']}; pipeline on Windows, PostgreSQL and MediaMTX in Docker",
                ],
                [
                    "Video",
                    f"{profile['codec']}, {profile['width']} x {profile['height']}, {profile['source_fps']} source FPS, RTSP/TCP",
                ],
                [
                    "Topology",
                    f"One shared Python process; recording + streaming workers and a separate {queue_description} per camera; one FFmpeg recording process per camera",
                ],
                [
                    "Consumer",
                    "One counting consumer per camera, with no inference model; stress run adds 0.75 s per frame per consumer",
                ],
                [
                    "Normal trials",
                    "Baseline; recording only; streaming only; full pipeline at 1, 2 and 4 cameras. Three fresh-process repeats per case.",
                ],
                [
                    "Timing",
                    "10 s warm-up + 60 s measurement per normal trial; 10-minute four-camera stability measurement; one slow-consumer trial",
                ],
                [
                    "Sampling",
                    "Application RSS/USS approximately every 0.2 s; host every 1 s; Docker CLI approximately every 5-8 s",
                ],
            ],
            [1, 3.3],
            styles,
        )
    )
    for title, text in [
        (
            "Controlled input",
            "Independent connections read the same looping sample-video feed. A separate benchmark publisher normalizes timestamps, uses one-second H.264 keyframes and no B-frames. Existing camera streams and services remain running. This simulates ingestion instances, not four distinct physical camera scenes.",
        ),
        (
            "What is counted",
            "The recording path copies compressed video without decoding. Streaming includes PyAV decoding, selection and BGR conversion. The offline recorded-segment decoder is not used. All recording trials write real segments and database metadata.",
        ),
        (
            "How statistics are reported",
            "Typical steady memory is the median of the three per-run measurement-window means. Peaks are the highest sampled value across full runs, including startup and draining. Memory held by the sampler itself is included in the measured application; small long-run drift may include retained sample data.",
        ),
        (
            "Units and ownership",
            "MiB = 1,048,576 bytes; GiB = 1,024 MiB. RSS is resident process memory; USS is private resident memory. Adding RSS across processes may count shared executable/library pages repeatedly. Workers are threads, so component costs are estimated from isolated-run differences, not exact per-thread ownership.",
        ),
    ]:
        story.extend([_p(title, styles, "Subsection"), _p(text, styles)])

    story.extend(
        [PageBreak(), *_heading("03", "Components and camera scaling", styles)]
    )
    story.append(
        _p(
            "Application totals: Python plus its recording FFmpeg children. Docker services and the separate benchmark coordinator are excluded from these figures.",
            styles,
        )
    )
    story.append(
        _table(
            [
                "Scenario",
                "Cams",
                "Python mean MiB",
                "FFmpeg mean MiB",
                "App mean RSS MiB",
                "App peak RSS MiB",
                "App mean USS MiB",
            ],
            [
                [
                    row["mode"],
                    row["cameras"],
                    *[
                        f"{row[key]:.1f}"
                        for key in (
                            "python_rss_mean_mib",
                            "ffmpeg_rss_mean_mib",
                            "app_rss_mean_mib",
                            "app_rss_peak_mib",
                            "app_uss_mean_mib",
                        )
                    ],
                ]
                for row in aggregate
            ],
            [1.15, 0.55, 1, 1, 1, 1, 1],
            styles,
        )
    )
    story.append(_p("Additional memory for one camera", styles, "Subsection"))
    story.append(
        _p(
            f"Subtract the {baseline['app_rss_mean_mib']:.1f} MiB imported Python baseline. The recording and streaming results are comparative increments; the baseline is shared and counted once.",
            styles,
        )
    )
    story.append(
        _table(
            ["Component", "Additional steady RSS (MiB)", "Additional steady USS (MiB)"],
            [
                [
                    label,
                    f"{by_key[(mode, 1)]['app_rss_mean_mib'] - baseline['app_rss_mean_mib']:.1f}",
                    f"{by_key[(mode, 1)]['app_uss_mean_mib'] - baseline['app_uss_mean_mib']:.1f}",
                ]
                for mode, label in [
                    ("recording", "Recording, including FFmpeg"),
                    ("streaming", "Streaming, including decoding"),
                    ("full", "Combined pipeline"),
                ]
            ],
            [1.5, 1, 1],
            styles,
        )
    )
    one, two = by_key[("full", 1)], by_key[("full", 2)]
    increment = two["app_rss_mean_mib"] - one["app_rss_mean_mib"]
    predicted = one["app_rss_mean_mib"] + 3 * increment
    story.extend(
        [
            _p("Scaling estimate and validation", styles, "Subsection"),
            _p(
                f"Going from one to two cameras added {increment:.1f} MiB steady summed RSS. Applying that increment predicts <b>{predicted:.1f} MiB</b> for four cameras. The measured four-camera mean was <b>{four['app_rss_mean_mib']:.1f} MiB</b>, a prediction error of {(predicted / four['app_rss_mean_mib'] - 1) * 100:+.1f}%.",
                styles,
            ),
        ]
    )
    story.append(
        _p(
            f"Across fast four-camera runs, the highest application RSS was {fast_peak:.1f} MiB. Adding 25% gives a provisional <b>{fast_peak * 1.25:.0f} MiB application-only allowance</b>. Add a deliberate queue budget and future model requirements separately. This does not include the operating system or Docker services.",
            styles,
        )
    )
    story.append(
        _p(
            f"Four cameras averaged {four['app_uss_mean_mib']:.1f} MiB private resident memory, versus {four['app_rss_mean_mib']:.1f} MiB summed RSS. Shared-page accounting is one reason RSS totals exceed unique memory. Whole-host measurements on the next page already include both.",
            styles,
            "SmallCopy",
        )
    )

    story.extend(
        [PageBreak(), *_heading("04", "Docker and whole-machine memory", styles)]
    )
    story.append(
        _p(
            "Host used RAM is total physical RAM minus available RAM. It already includes Python, FFmpeg, Docker/WSL, the OS, monitoring and unrelated applications. <b>Do not add container or process figures to host usage.</b>",
            styles,
        )
    )
    story.append(
        _table(
            [
                "Scenario",
                "Cams",
                "Host used mean GiB",
                "Host available min GiB",
                "WSL RSS mean MiB",
                "Docker Desktop RSS mean MiB",
            ],
            [
                [
                    row["mode"],
                    row["cameras"],
                    f"{row['host_used_mean_gib']:.2f}",
                    f"{row['host_available_min_gib']:.2f}",
                    f"{row['wsl_rss_mean_mib']:.1f}",
                    f"{row['desktop_rss_mean_mib']:.1f}",
                ]
                for row in aggregate
            ],
            [1.1, 0.55, 1, 1, 1, 1.3],
            styles,
        )
    )
    story.append(_p("Container working memory", styles, "Subsection"))
    names = sorted({name for run in results for name in run["containers"]})
    containers = []
    for name in names:
        idle = [
            run["containers"][name]["mean_mib"]
            for run in results
            if run["mode"] == "baseline"
        ]
        active = [
            run["containers"][name]["mean_mib"]
            for run in results
            if run["mode"] == "full"
            and run["cameras"] == 4
            and "_soak" not in run["case"]
            and not run["consumer_delay_seconds"]
        ]
        peak = max(run["containers"][name]["peak_mib"] for run in results)
        containers.append(
            [name, f"{median(idle):.1f}", f"{median(active):.1f}", f"{peak:.1f}"]
        )
    story.append(
        _table(
            ["Container", "Idle mean MiB", "4-camera mean MiB", "Highest observed MiB"],
            containers,
            [1.6, 1, 1, 1],
            styles,
        )
    )
    for title, text in [
        (
            "Interpret the layers separately",
            "Docker CLI memory statistics on Linux exclude reclaimable cache. Container values overlap the WSL VM and host totals. WSL figures include all visible vmmem processes, not necessarily Docker alone; Docker Desktop figures sum visible Windows processes and may themselves share pages.",
        ),
        (
            "Host usage is not a standalone requirement",
            f"The machine averaged {four['host_used_mean_gib']:.2f} GiB used during the four-camera comparison windows, with a minimum {four['host_available_min_gib']:.2f} GiB available. Existing Windows applications and VM caching contribute to this load; it cannot all be assigned to the pipeline. Services were left running, so there is no Docker-off or OS-only baseline.",
        ),
        (
            "Separate test infrastructure from deployment",
            f"The local mock publishers averaged about {four['mock_publisher_rss_mean_mib']:.1f} MiB summed RSS. The mock CCTV server and publishers are included in host figures, but real cameras may remove those costs. The Docker VM configuration is a memory cap, not its measured consumption.",
        ),
    ]:
        story.extend([_p(title, styles, "Subsection"), _p(text, styles)])

    story.extend(
        [PageBreak(), *_heading("05", "Direct-camera production estimate", styles)]
    )
    story.append(
        _p(
            "<b>Production flow: camera RTSP directly to recording and streaming workers.</b> The camera performs encoding and publishing. MediaMTX is optional relay infrastructure and is excluded from this production budget.",
            styles,
        )
    )
    story.append(
        _p(
            "The following is a labelled accounting estimate derived from the four-camera trials, not a new direct-camera measurement. The original test results remain unchanged on page 4 and in the raw CSV/JSON data.",
            styles,
        )
    )
    story.append(
        _table(
            ["Component", "Test observation", "Production treatment"],
            [
                [
                    "PostgreSQL container",
                    f"{production['postgres_mib']:.1f} MiB mean working memory",
                    "Keep as the measured reference",
                ],
                [
                    "MediaMTX mock relay",
                    f"{production['excluded_mtx_mib']:.1f} MiB mean working memory",
                    "Exclude: 0 MiB for this service",
                ],
                [
                    "Known-container subtotal",
                    f"{production['test_container_total_mib']:.1f} MiB (database + relay)",
                    f"About {production['production_known_container_estimate_mib']:.1f} MiB (database only)",
                ],
                [
                    "Local mock-video publishers",
                    f"{four['mock_publisher_rss_mean_mib']:.1f} MiB summed RSS",
                    "Exclude; real cameras publish their own streams",
                ],
                [
                    "Recording FFmpeg processes",
                    "Included in application figures",
                    "Keep; these record incoming video on the server",
                ],
                [
                    "Docker runtime / VM",
                    "Included in measured host usage",
                    "Keep an allowance while production services use Docker",
                ],
            ],
            [1.2, 1.2, 1.65],
            styles,
        )
    )
    for title, text in [
        (
            "What is reduced",
            f"Remove MediaMTX's approximately <b>{production['excluded_mtx_mib']:.1f} MiB</b> mean working memory from the known-container budget. The two measured containers previously totalled {production['test_container_total_mib']:.1f} MiB; the remaining PostgreSQL reference is {production['production_known_container_estimate_mib']:.1f} MiB. This subtotal does not include Docker/VM overhead or future model-serving containers.",
        ),
        (
            "What does not change",
            "The measured 200.0 MiB application peak and the provisional 250 MiB application allowance already exclude MediaMTX and the mock publishers. Do not subtract their memory from those figures again. Server-side recording FFmpeg and the streaming decoder are still required.",
        ),
        (
            "Why the host total is not simply reduced",
            "Container working memory, process RSS and WSL allocation are different measurements with shared and cached memory. Removing the relay does not prove an identical physical host-RAM saving. Docker is still needed for PostgreSQL, so its whole VM/desktop overhead cannot be removed. Historical host totals remain as measured.",
        ),
        (
            "Remaining verification",
            "Benchmark the intended direct camera URLs, actual resolution/codec and deployed container layout to measure real savings. The production estimate here excludes mock-only services, but still needs OS/runtime, model, queue and operating-headroom budgets.",
        ),
    ]:
        story.extend([_p(title, styles, "Subsection"), _p(text, styles)])

    story.extend([PageBreak(), *_heading("06", "Stability and queue backlogs", styles)])
    story.append(
        _p("Four cameras, fast consumers: 10-minute measurement", styles, "Subsection")
    )
    story.append(_time_chart(root, soak, slow=False))
    story.append(
        _p(
            f"Consumed {sum(c['consumed_frames'] for c in soak['cameras_detail']):,} selected frames; dropped {sum(c.get('dropped_frames', 0) for c in soak['cameras_detail'])} on overflow. Peak RSS was {soak['metrics']['app_rss_peak_mib']:.1f} MiB; at most {soak['metrics']['queued_frames_peak']} frames were queued. Late queue-image growth was {soak['metrics']['queued_image_late_slope_mib_per_minute']:.2f} MiB/min. Late RSS growth was {soak['metrics']['app_rss_late_slope_mib_per_minute']:.2f} MiB/min and includes profiling overhead; this is not a proof of leak-free behavior indefinitely.",
            styles,
            "SmallCopy",
        )
    )
    story.append(
        _p("Four cameras, slow consumers: 0.75 s per frame each", styles, "Subsection")
    )
    story.append(_time_chart(root, slow, slow=True))
    story.append(
        _table(
            ["Stress outcome", "Measured value"],
            [
                [
                    "Peak application RSS",
                    f"{slow['metrics']['app_rss_peak_mib']:.1f} MiB",
                ],
                [
                    "Peak backlog",
                    f"{slow['metrics']['queued_frames_peak']} frames / {slow['metrics']['queued_image_bytes_peak_mib']:.1f} MiB image data",
                ],
                [
                    "Late queued-image growth",
                    f"{slow['metrics']['queued_image_late_slope_mib_per_minute']:.1f} MiB per minute",
                ],
                ["Dropped frames on overflow", str(slow_dropped)],
                [
                    "Run time including drain",
                    f"{slow['elapsed_seconds_including_drain']:.1f} s for {slow['warmup_seconds']:g} s warm-up + {slow['measurement_seconds']:g} s measurement",
                ],
            ],
            [1, 1.6],
            styles,
        )
    )
    story.append(
        _p(
            f"A single 640 x 360 BGR image uses 0.659 MiB before metadata and decoder buffers. {queue_conclusion} A completed finite stress run does not establish safety or detection coverage for indefinite operation.",
            styles,
            "SmallCopy",
        )
    )

    story.extend(
        [PageBreak(), *_heading("07", "Limits, findings and reproduction", styles)]
    )
    for title, text in [
        (
            "Model and resolution limits",
            "These are 640 x 360 ingestion results, not 1080p/4K hardware sizing. Queued image payload grows with pixel count, but total decoding memory cannot reliably be scaled by that factor alone. YOLO/VLM weights, inference workspaces and GPU memory are absent and may dominate the eventual budget.",
        ),
        (
            "Consumer and deployment topology",
            "The experiment uses a separate counting consumer per camera. One future shared YOLO consumer must sustain up to 8 selected FPS for four cameras (roughly 125 ms per frame on average including other work). That model and queue arrangement must be benchmarked. The pipeline currently runs on Windows while services run in Docker; a fully containerized/Linux deployment also needs measurement.",
        ),
        (
            "Sampling and background activity",
            "Sampled peaks may miss shorter spikes. The in-process sampler retains rows until each run ends, contributing small memory growth. Background applications, source encoding complexity, shared library pages and Docker/WSL caches affect RAM. Three repeats reduce run-to-run uncertainty but do not establish a universal capacity guarantee.",
        ),
    ]:
        story.extend([_p(title, styles, "Subsection"), _p(text, styles)])
    attempt = environment.get("excluded_attempt")
    if attempt:
        story.extend(
            [
                _p(
                    "Input-timing finding: excluded initial attempt",
                    styles,
                    "Subsection",
                ),
                _p(
                    f"A previous four-camera attempt against the existing mock feed stopped after {attempt['elapsed_seconds']:.1f} seconds with <font face='Courier' size='8'>{escape(attempt['reason'])}</font>. Its memory results were excluded. All reported comparisons use a separate timestamp-normalized feed. This is not a production fix; investigate that input-timing failure separately.",
                    styles,
                ),
            ]
        )
    story.extend(
        [
            _p("Reproduce the suite", styles, "Subsection"),
            _p("With the database migrated and the mock RTSP server running:", styles),
            _p("<font face='Courier'>npm run benchmark:memory:suite</font>", styles),
        ]
    )
    story.append(
        _p(
            "The suite starts and stops its own publisher, records real video/metadata, runs three repeats per case, performs a 10-minute stability measurement and a slow-consumer test. Existing streams and containers are left running. Allow approximately 35 minutes.",
            styles,
        )
    )
    story.append(_p("Artifacts and provenance", styles, "Subsection"))
    story.append(
        _p(
            f"Completed scenarios: {len(results)}. Raw run directory:<br/><font face='Courier' size='7.5'>recordings/benchmarks/{escape(root.name)}/</font>",
            styles,
        )
    )
    story.append(
        _p(
            "Each case contains process.csv, host.csv, docker.csv, summary.json and worker.log. The run directory also contains results.json, environment.json, aggregate.csv and the Markdown report. The source summary is reports/memory-benchmark-2026-11-14.md; the companion table is reports/memory-benchmark-2026-11-14.csv.",
            styles,
            "SmallCopy",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=21 * mm,
        title="Pipeline memory benchmark - 14 November 2026",
        author="Data pipeline performance study",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_pdf(args.root, args.output)
    print(args.output.resolve())
