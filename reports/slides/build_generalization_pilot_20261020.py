"""Build a focused, source-bound addendum deck for the generalization pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pptx import Presentation
from reportlab import rl_config

from reports.slides import build_progress_presentation as prior

d = prior.d
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "results/generalization_pilot_20261020"
STEM = "model_generalization_pilot_20261020"
MODELS = ("mobileclip2-s2", "siglip2-b16-224")
SHORT = {MODELS[0]: "MobileCLIP", MODELS[1]: "SigLIP2"}


def load(path):
    return d.read(path)


def tab(title, subtitle, headers, rows, widths, takeaway, notes):
    return prior.tab(
        "Generalization pilot / 20 October",
        title,
        subtitle,
        "20 Oct frozen protocols, predictions and verification",
        headers,
        rows,
        widths,
        takeaway,
        notes,
        row_h=32,
        font=12,
        warn=True,
    )


def paired(title, left, right, takeaway, notes):
    return prior.paired(
        "Generalization pilot / 20 October",
        title,
        "New public-source evidence and full-stream comparisons",
        "20 Oct verified artifacts; 19 Oct history preserved",
        left,
        right,
        takeaway,
        notes,
        warn=True,
    )


def pct(x):
    return f"{100 * x:.1f}%"


def build():
    d.SLIDES.clear()
    d.SOURCES.clear()
    events = load(RUN / "results.json")
    smoke = {m: load(RUN / "smoking" / m / "results.json") for m in MODELS}
    smol = {
        m: load(RUN / "smolvlm_transfer" / m / "results.json")
        for m in ("original", "lora")
    }
    verification = load(RUN / "verification.json")
    native = load(RUN / "native_feature_validation.json")
    assert verification["status"] == native["status"] == "passed"
    for path in (
        RUN / "protocol.json",
        RUN / "smoking/protocol.json",
        RUN / "smolvlm_transfer/protocol.json",
        ROOT / "data/generalization_pilot_20261020/source_audit.json",
        ROOT / "data/generalization_pilot_20261020/xrf55/preparation.json",
    ):
        load(path)
    paired(
        "Test the complete system and the source shift",
        (
            "Completed in this pilot",
            "740 public action clips acquired\n36 CLIP classifier fits\nOriginal / LoRA SmolVLM transfer\n10 complete streams, 3 detector arms",
        ),
        (
            "What the results establish",
            "Tiling worsens primary event detection\nSmolVLM LoRA regresses proxy AP\nSource-specific CLIP results follow\nNo automatic model promotion",
        ),
        "More data and better box recall do not guarantee better target-camera performance.",
        "Focused addendum to the 19 October paper and 45-slide history. These are actual new experiments. The data source is scripted smoking-action/gesture evidence, not independently verified CCTV smoking. No new backbone or SmolVLM training and no latency test are claimed.",
    )
    tab(
        "Public data acquired, with explicit label limits",
        "XRF55 release v6 • smoking action + drinking / eating / phone controls",
        ["Split", "Scene blocks", "Subject aliases", "Clips", "Positive / control"],
        [
            ["Train", "1 + 2", "15", "340", "85 / 255"],
            ["Development", "3", "2", "160", "40 / 120"],
            ["Test", "4", "3", "240", "60 / 180"],
        ],
        [130, 170, 180, 150, 246],
        "Masked training actors: treat this as a scripted gesture proxy; cigarettes were not verified.",
        "Author repository airslab2020/XRF55-repo; public Kaggle RGB/depth/IR release version 6. There are 886,570,077 bytes and 11,840 sixteen-view JPEG inputs. The release README records CC BY-NC 4.0. Numeric subject aliases are conservatively treated as global. Seventeen scene-subject training groups collapse to 15 numeric groups. One test scene and three groups do not represent 240 independent people. Assistant visual review used training clips only. PETS links were unavailable; synthetic and rendered method-output luggage media were rejected. Other exclusions remain in source_audit.json.",
    )
    paired(
        "Same selection rules across old, new and mixed data",
        (
            "Thirty-six classifier fits",
            "2 frozen pretrained CLIP encoders\n3 training-source families\n2 full / contextual-crop recipes\nC = 0.1, 1, 10; group-balanced fits",
        ),
        (
            "Development-only decisions",
            "Same 1,474 dev cases / 4 sources\nOne cutoff: each source FPR ≤5%\nSelect by worst-source recall\nFreeze both models before test",
        ),
        "Old-only and new-only differ in training data; both receive combined-source dev supervision.",
        "Old-only2812/new-only340/mixed3152 training cases. Frozen visual encoders, alpha0 only. Float32 CUDA features; normalized last/mean/max/delta pooling and automatic COCO crops. All36 logistic regression fits converged;0 new visual fine-tunes. Selection tie-breaks mean recall, worst AP, mean AP. 1916 held-out cases per encoder include1676 previously inspected historical cases and240 new proxy cases. Source-aware development FPR does not guarantee held-out FPR.",
    )
    for model in MODELS:
        rows = []
        for source, label in (
            ("smoking_cctv", "Historical CCTV"),
            ("xrf55_rgb_public", "New action proxy"),
        ):
            for family in ("old_only", "new_only", "mixed"):
                m = smoke[model]["results"][family + "/" + source]["operating_points"][
                    "dev_fpr_budget"
                ]
                rows.append(
                    [
                        label,
                        family.replace("_", " "),
                        f"{m['ap']:.4f}",
                        pct(m["recall"]),
                        pct(m["fpr"]),
                    ]
                )
        tab(
            SHORT[model] + ": data addition has source-specific effects",
            "Frozen every-source development cutoff • CCTV n=646; proxy n=240",
            ["Held-out source", "Training family", "AP", "Recall", "FPR"],
            rows,
            [244, 190, 120, 160, 162],
            "AP is ranking. Recall and FPR measure each family's frozen operating point.",
            "All recipe/C choices are development-only. Historical CCTV is retrospective. XRF55 is a one-scene, three-subject action-proxy holdout. All source cohorts and raw/maxF1/budget points are in the report CSV; photo-source results must not be pooled into CCTV accuracy. The genuine pretrained text control and historical last-frame native reference are also retained. This comparison selects within each matched family, not a single fixed head.",
        )
    rows = []
    for route in ("original", "lora"):
        for key, name in (
            ("new_source_calibrated", "New source dev"),
            ("original19oct_cutoffs", "19 Oct dev"),
        ):
            m = smol[route][key]["dev_fpr_budget"]
            rows.append(
                [route, name, f"{m['ap']:.4f}", pct(m["recall"]), pct(m["fpr"])]
            )
    tab(
        "SmolVLM: prior LoRA does not improve this source",
        "Matched original / prior adapter • eight views each • no new training",
        ["Route", "Cutoff calibration", "AP", "Recall", "FPR"],
        rows,
        [160, 240, 140, 165, 171],
        "Both older budget cutoffs miss all positives; new-source calibration still exceeds 5% test FPR.",
        "Original AP is 0.3855 versus 0.3573 for LoRA. The existing smoking route is seed 42 / learning rate 0.0001, with 4,341,760 language parameters trained previously. Both development routes were calibrated before any new-source test forward. There is no scene-4 route selection or new fine-tuning. Eight uniform views are the validated prompt maximum; CLIP uses sixteen. Native-versus-cached margin errors are below 0.002. This result concerns scripted action source transfer, not true CCTV smoking accuracy.",
    )
    rows = []
    for mode in ("full", "item_tiles", "all_tiles"):
        m = next(
            r
            for r in events["aggregate"]
            if r["mode"] == mode and r["evaluation_group"] == "avss_primary"
        )
        rows.append(
            [
                mode.replace("_", " "),
                f"{m['detected_events']}/{m['events']}",
                str(m["normal_false_alerts"]),
                str(m["duplicate_alerts"]),
                f"{m['median_delay_seconds']:.2f} s",
            ]
        )
    tab(
        "Full-stream tiling reverses the component gain",
        "AVSS EASY / MEDIUM primary • unchanged detector, tracker and alarm rules",
        ["Detector arm", "Events found", "Normal FP", "Duplicates", "Median delay"],
        rows,
        [230, 165, 160, 150, 171],
        "Retain full-frame baseline. Better CAVIAR box recall did not improve the event pipeline.",
        "Ten recordings are processed from time zero to EOF at 2 Hz: 2,512 images and 12,560 views across three arms. Full-frame views plus four 60% corner tiles use class-wise NMS at IoU 0.5. Item-only tiles retain full-frame person proposals. The unchanged rule requires known ownership, 30 seconds of stationarity, 0.5 seconds of owner absence and two seconds of alarm persistence. Primary normal exposure is 205.739 seconds, or 3.43 minutes; its zero-alert one-sided 95% Poisson upper bound is 52.42/hour. Saved results also align the earlier t >= 20 exposure of 165.739 seconds. Forty added seconds are the same footage. Delay measures onset to alarm, not inference latency.",
    )
    rows = []
    for group, label in (
        ("historical_provisional", "ABODA provisional"),
        ("avss_censored", "AVSS HARD censored"),
    ):
        for mode in ("full", "item_tiles", "all_tiles"):
            m = next(
                r
                for r in events["aggregate"]
                if r["mode"] == mode and r["evaluation_group"] == group
            )
            rows.append(
                [
                    label,
                    mode.replace("_", " "),
                    f"{m['detected_events']}/{m['events']}",
                    str(m["normal_false_alerts"]),
                    str(m["duplicate_alerts"]),
                    str(m["unknown_time_alerts"]),
                ]
            )
    tab(
        "Keep provisional and censored evidence separate",
        "Seven ABODA streams share two scene blocks; AVSS is one conservative source block",
        ["Evidence", "Arm", "Found", "FP", "Repeat", "Unknown"],
        rows,
        [250, 190, 110, 100, 110, 116],
        "Ten recordings are not ten independent cameras. Unknown-time alarms remain unscored.",
        "ABODA has five provisional events and 359 normal seconds; full-frame finds 4/5, while each tile arm finds 2/5. These intervals are not independently reviewed. Right-censored HARD remains separate from the two primary events. All three arms detect its annotated interval, but repeated alarms persist; all-tiles adds a normal false alert. MEDIUM item-track IDs increase from 284 to 497 and EASY from 162 to 371, consistent with spurious or fragmented associations. These counts are not ground-truth identity-switch rates. No rule was retuned on these test streams.",
    )
    paired(
        "Replay and native forwards bind the results",
        (
            "Numerical / data checks",
            "7,536 stream decisions replayed\nAll 36 candidates and 6 choices\nAll selected / reference test scores\n11,840 decoded new inputs checked",
        ),
        (
            "Native GPU validation",
            "Both CLIP encoders, full + crop\nFixed train / dev / test cases\nOriginal and LoRA SmolVLM paths\nSource and code identities saved",
        ),
        "Passed verification checks computation and identity; it does not strengthen proxy labels.",
        "Verification.json and native_feature_validation.json passed before report build.192 unit tests passed previously, including5newtilefusioninvariants; currentchecklog isstoredwithrun. Exact decoded-pixel disjointness cannotruleoutperceptual/correlated repetition. No latency benchmark, camera-rate simulation or CPU-only quality cap. Inference uses availableGPU. Convex fitting/metric replay usesCPU.",
    )
    tab(
        "Next evidence needed for the five objectives",
        "No target is promoted from this small public-source pilot",
        ["Objective", "Next concrete data / modelling target"],
        [
            [
                "Smoking",
                "Visible-cigarette CCTV evidence; distant drinking / eating / phone negatives",
            ],
            [
                "Unattended items",
                "Owner/item histories; coherent association and gap recovery; long normal streams",
            ],
            [
                "Littering",
                "Annotated release/disposal events and putting-down / picking-up controls",
            ],
            [
                "Loitering",
                "Duration and zone-labelled person tracks; camera/day holdouts",
            ],
            [
                "Apparent public sleep",
                "Temporal resting/wake controls and new locations; visual definition only",
            ],
        ],
        [235, 641],
        "Prioritize independent event evidence and a frozen acceptance test over a larger parameter search.",
        "No new littering, loitering or public-sleep scores are claimed. Prior results and failed transfers remain in the 19 October report. About three normal camera-hours with zero alerts would give a 95% Poisson upper bound near one/hour, subject to that model's assumptions; this does not replace camera diversity. Documentation: reports/generalization_pilot_20261020/report.md and runthrough.pdf. Reproduction: docs/generalization_pilot.md. The existing 19 October paper and 45 slides remain preserved.",
    )
    for i, slide in enumerate(d.SLIDES, 1):
        for item in slide["items"]:
            if item["kind"] != "text":
                continue
            item["value"] = item["value"].replace(
                "PROGRESS 17 OCT 2026", "PILOT 20 OCT 2026"
            )
            if item["x"] == 882 and item["y"] == 512:
                item["value"] = f"{i:02d}"
            assert (
                len(item["value"].splitlines()) * item["leading"] <= item["h"] + 0.01
            ), item
    d.check_geometry(d.SLIDES)


def main():
    build()
    rl_config.canvas_basefontname = "DejaVu"
    d.render_pptx(d.SLIDES, OUT / (STEM + ".pptx"))
    presentation = Presentation(OUT / (STEM + ".pptx"))
    presentation.core_properties.title = (
        "Public-source generalization pilot — 20 October 2026"
    )
    presentation.core_properties.subject = "CLIP source ablations, SmolVLM transfer and full-stream detector/tracker results"
    presentation.save(OUT / (STEM + ".pptx"))
    d.render_pdf(d.SLIDES, OUT / (STEM + ".pdf"))
    (OUT / (STEM + "_notes.md")).write_text(
        "\n\n".join(
            ["# Generalization pilot — speaker notes"]
            + [
                f"## {i}. {s['title']}\n\n{s['notes']}"
                for i, s in enumerate(d.SLIDES, 1)
            ]
        )
        + "\n"
    )
    (OUT / (STEM + "_manifest.json")).write_text(
        json.dumps(
            {
                "date": "2026-10-20",
                "slides": len(d.SLIDES),
                "sources_sha256": d.SOURCES,
                "builder_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "layout": d.SLIDES,
                "qualification": "Focused pilot addendum; prior full-history deck preserved.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Built", len(d.SLIDES), "pilot slides, editable PPTX, PDF and notes")


if __name__ == "__main__":
    main()
