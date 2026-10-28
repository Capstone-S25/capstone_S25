"""Extend the preserved technical report with the completed SmolVLM study.

Only saved results are read. No model inference or training runs here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

import build_technical_runthrough as b

ROOT = b.ROOT
OUT = ROOT / "reports/technical_runthrough_20261019"
STEM = "model_evaluation_runthrough_20261019"
STUDY = "results/smolvlm_finetuning_20261019"
SOURCE_NAMES = {
    "kaggle_sujaykapadnis_smoking": "Smoking photos (Kaggle)",
    "smoking_cctv": "Smoking CCTV c1",
    "mivia_iwdd": "Littering / IWDD",
    "street_scene": "Loitering / Street Scene",
    "commons_visible_sleeping": "Public sleep / historical",
    "unattended_reviewed": "Unattended / historical",
    "stanford40": "Smoking / Stanford40",
    "hmdb51_action_transfer": "Smoking / HMDB",
    "commons_new_author_transfer": "Public sleep / new authors",
    "meva_abandon_action_transfer": "MEVA abandonment action",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_context(cells):
    """Read the previously verified identical-ID CLIP pipeline comparisons."""
    context = {}
    for model in b.MODELS:
        data = b.read(
            f"results/strategy_study_20261016/{model}/transfer_evaluation/per_source_results.json"
        )
        for cell in cells:
            matches = [v for k, v in data.items() if k.endswith("/" + cell["source"])]
            assert len(matches) == 1, cell["source"]
            row = matches[0]["selected"]["raw"]
            assert (row["n"], row["positive"]) == (cell["n"], cell["positive"])
            context[model, cell["source"]] = row["ap"]
    return context


def smol_tables():
    protocol = b.read(STUDY + "/protocol.json")
    data = b.read(STUDY + "/report_data.json")
    verification = b.read(STUDY + "/verification.json")
    run = b.read(STUDY + "/run.json")
    assert run["status"] == "complete" and verification["status"] == "passed"
    cells = data["cells"]
    assert len(cells) == 60
    source_order = list(SOURCE_NAMES)
    lookup = {(r["seed"], r["source"], r["method"]): r for r in cells}
    primary = [lookup[42, s, "routed"] for s in source_order]
    assert sum(r["n"] for r in primary) == 3792
    b.table(
        "smol_counts",
        "SmolVLM main partitions: normal/positive counts, not total/positive.",
        ["Target", "Train N/+", "Dev N/+", "Test N/+", "External N/+"],
        [
            [b.TASK_NAMES[t]]
            + [
                f"{protocol['counts'].get(f'{s}/{t}/0', 0)}/{protocol['counts'].get(f'{s}/{t}/1', 0)}"
                if any(k.startswith(f"{s}/{t}/") for k in protocol["counts"])
                else "--"
                for s in ["train", "dev", "test", "external"]
            ]
            for t in b.TASKS
        ],
        [1.7, 1, 1, 1, 1.1],
    )
    b.table(
        "smol_settings",
        "Actual SmolVLM training and scoring settings.",
        ["Setting", "Recorded value"],
        [
            ["Adaptation", "Language projections only; 4,341,760 trainable parameters"],
            ["LoRA", "Rank 8; alpha 16; dropout 0.05; no bias"],
            ["Projections", "q/k/v/o and gate/up/down in 32 language layers"],
            [
                "Frozen",
                "Vision tower, multimodal connector, embeddings, vocabulary head",
            ],
            [
                "Inputs",
                "512-pixel gray letterbox; 64 visual tokens/image; no image splitting",
            ],
            [
                "Views",
                "1 static; 4 ordered past views for temporal windows; 8 littering clip views",
            ],
            [
                "Loss",
                "First-answer-token full-vocabulary cross entropy; all input positions masked",
            ],
            ["Optimizer", "AdamW; weight decay 0.01; gradient norm clip 1.0"],
            ["Schedule", "5% warmup; linear decay; at most 6 epochs; patience 2"],
            ["Primary search", "Seed 42; learning rates 0.00003 and 0.0001"],
            [
                "Replication",
                "Seed 43; only the primary winning joint learning rate 0.0001",
            ],
            ["Batch", "8 examples/microbatch; accumulation 4; effective batch 32"],
            [
                "Coverage/epoch",
                "3,973 unique examples once + 3,973 balanced resamples = 7,946 visits",
            ],
            [
                "Numerics",
                "BF16 training; frozen visual tokens and all selection/final scores float32; TF32 off",
            ],
            [
                "Score",
                "First-answer logit(Yes) minus logit(No); not a calibrated probability",
            ],
        ],
        [1.4, 4.7],
    )
    histories, states = [], []
    for seed, lr in [(42, "3e-05"), (42, "0.0001"), (43, "0.0001")]:
        route = f"seed_{seed}/lr_{lr}"
        state = b.read(f"{STUDY}/{route}/state.json")
        history = b.read(f"{STUDY}/{route}/history.json")
        assert state["epochs_completed"] == 3 and state["best_epoch"] == 1
        states.append(
            [
                seed,
                lr,
                state["best_epoch"],
                state["epochs_completed"],
                b.num(state["best_dev_macro_ap"], 4),
            ]
        )
        for h in history:
            assert h["visits"] == 7946 and h["unique_training_examples"] == 3973
            histories.append(
                [
                    seed,
                    lr,
                    h["epoch"],
                    b.num(h["loss"], 4),
                    b.num(h["dev_macro_ap"], 4),
                    h["visits"],
                ]
            )
        b.register(f"{STUDY}/{route}/adapter/adapter_config.json")
    b.table(
        "smol_checkpoints",
        "All three completed candidates; selection is based on development data.",
        ["Seed", "Learning rate", "Best epoch", "Epochs run", "Dev macro AP"],
        states,
        [0.6, 1.4, 1, 1, 1.3],
    )
    b.table(
        "smol_history",
        "Complete epoch histories. Falling training loss does not imply better source transfer.",
        ["Seed", "Learning rate", "Epoch", "Train loss", "Dev macro AP", "Visits"],
        histories,
        [0.6, 1.3, 0.7, 1, 1.3, 0.9],
    )
    selection = b.read(STUDY + "/seed_42/selection.json")
    b.read(STUDY + "/seed_43/selection.json")
    short_route = lambda r: (
        "Original" if r == "original" else ("Low LR" if "3e-05" in r else "High LR")
    )
    b.table(
        "smol_routes",
        "Primary per-task routes and exact development thresholds; raw margin is zero for every route.",
        ["Target", "Route", "Dev AP", "Dev-F1 cutoff", "Dev-5% cutoff"],
        [
            [
                b.TASK_NAMES[t],
                short_route(r),
                b.num(selection["route_dev_source_ap"][t], 4),
                b.num(selection["thresholds"][r][t]["dev_max_f1"], 6),
                b.num(selection["thresholds"][r][t]["dev_fpr_budget"], 6),
            ]
            for t, r in selection["task_routes"].items()
        ],
        [1.7, 0.9, 0.8, 1.3, 1.3],
    )
    b.table(
        "smol_ap",
        "Matched primary held-out AP: identical inputs, definitions and scoring; historical source holdouts.",
        ["Source / task", "Total / +", "Constant AP", "Original", "Joint", "Routed"],
        [
            [
                SOURCE_NAMES[s],
                f"{lookup[42, s, 'original']['n']}/{lookup[42, s, 'original']['positive']}",
                b.num(lookup[42, s, "original"]["constant_score_ap"], 4),
            ]
            + [
                b.num(lookup[42, s, m]["ap"], 4)
                for m in ["original", "joint", "routed"]
            ]
            for s in source_order
        ],
        [2.5, 1, 1, 0.9, 0.9, 0.9],
    )
    b.table(
        "smol_threshold_transfer",
        "At the target-specific pooled development FPR budget of 5%; no held-out FPR guarantee.",
        [
            "Source / task",
            "Orig rec.",
            "Orig FPR",
            "Route rec.",
            "Route FPR",
            "Route TP/FP",
        ],
        [
            [SOURCE_NAMES[s]]
            + [
                b.pct(lookup[42, s, m][f"{k}_at_dev_5pct"])
                for m in ["original", "routed"]
                for k in ["recall", "fpr"]
            ]
            + [
                f"{lookup[42, s, 'routed']['tp_at_dev_5pct']}/{lookup[42, s, 'routed']['fp_at_dev_5pct']}"
            ]
            for s in source_order
        ],
        [2.4, 0.9, 0.9, 0.9, 0.9, 1.1],
    )
    b.table(
        "smol_raw",
        "Predeclared raw margin-zero decisions; primary routed versus matched original.",
        [
            "Source / task",
            "Orig rec.",
            "Orig FPR",
            "Route rec.",
            "Route FPR",
            "Route F1",
        ],
        [
            [SOURCE_NAMES[s]]
            + [
                b.pct(lookup[42, s, m][f"{k}_at_raw"])
                for m in ["original", "routed"]
                for k in ["recall", "fpr"]
            ]
            + [b.num(lookup[42, s, "routed"]["f1_at_raw"], 4)]
            for s in source_order
        ],
        [2.4, 0.9, 0.9, 0.9, 0.9, 1.1],
    )
    b.table(
        "smol_seeds",
        "Joint-adapter AP under the fixed winning learning rate: optimization replication on the same sources.",
        ["Source / task", "Seed 42", "Seed 43"],
        [
            [SOURCE_NAMES[s]]
            + [b.num(lookup[seed, s, "joint"]["ap"], 4) for seed in [42, 43]]
            for s in source_order
        ],
        [3, 1, 1],
    )
    context = clip_context(primary)
    provenance = b.read(STUDY + "/clip_comparison_provenance.json")
    assert provenance["exact_heldout_labels_checked"] == 3792
    b.table(
        "smol_clip_context",
        "Same 3,792 held-out identities and labels: different trained pipelines, not an equal-input architecture ablation.",
        ["Source / task", "MobileCLIP", "SigLIP", "Smol routed"],
        [
            [SOURCE_NAMES[s]]
            + [b.num(context[m, s], 4) for m in b.MODELS]
            + [b.num(lookup[42, s, "routed"]["ap"], 4)]
            for s in source_order
        ],
        [2.7, 1, 1, 1.1],
    )
    repeated = b.read(STUDY + "/repeated_gain_routes.json")
    b.table(
        "smol_repeated_gains",
        "Exploratory retrospective routing safeguard; not a prospectively validated selection rule.",
        ["Target", "Primary dev gain", "Seed-43 dev gain", "Optional route"],
        [
            [
                b.TASK_NAMES[t],
                b.num(v["primary_development_ap_gain"], 4),
                b.num(v["secondary_development_ap_gain"], 4),
                short_route(repeated["task_routes"][t]),
            ]
            for t, v in repeated["development_gains"].items()
        ],
        [1.7, 1.3, 1.3, 1.2],
    )
    b.table(
        "smol_all_metrics",
        "All 60 cohort/method/seed summaries. O=original, J=joint, R=per-task routed; AUC denotes AUROC. Rec./FPR use the dev-5% point.",
        ["Source", "Seed", "M", "AP", "AUC", "Raw F1", "Dev-F1 F1", "Rec.", "FPR"],
        [
            [
                SOURCE_NAMES[r["source"]],
                r["seed"],
                r["method"][0].upper(),
                b.num(r["ap"], 4),
                b.num(r["auroc"], 4),
                b.num(r["f1_at_raw"], 4),
                b.num(r["f1_at_dev_f1"], 4),
                b.pct(r["recall_at_dev_5pct"]),
                b.pct(r["fpr_at_dev_5pct"]),
            ]
            for r in cells
        ],
        [2.6, 0.6, 0.6, 0.9, 0.9, 0.9, 1, 0.9, 0.9],
    )
    for name in ["learning_curves", "matched_ap"]:
        shutil.copyfile(
            b.register(f"{STUDY}/figures/{name}.pdf"), OUT / f"figures/smol_{name}.pdf"
        )
    shutil.copyfile(
        b.register(STUDY + "/metrics.csv"),
        OUT / "numeric_exports/smolvlm_finetuning_20261019_metrics.csv",
    )
    for name in [
        "environment.json",
        "input_label_audit.json",
        "implementation_manifest.json",
        "model_bundle.json",
        "portable_adapter_export.json",
        "response_mass.json",
        "software_checks.json",
        "cli_checks.json",
        "repeated_gain_protocol.json",
        "additional_transfer/protocol.json",
    ]:
        b.register(f"{STUDY}/{name}")


def historical_table_updates():
    original = b.table

    def revised(name, caption, headings, data, *args, **kwargs):
        data = [list(row) for row in data]
        if name == "chronology":
            data[-1][1:] = [
                "First full technical report",
                "83-page account through 17 October; no new training for documentation.",
            ]
            data.append(
                [
                    "19 Oct",
                    "SmolVLM LoRA and updated documentation",
                    "Three trained adapters; ten held-out cohorts; repeat-seed audit; no latency testing.",
                ]
            )
        if name == "models":
            for row in data:
                if "SmolVLM" in str(row[0]):
                    row[-1] = (
                        "Original prompted comparison; 19 Oct language-layer LoRA with binary scoring"
                    )
        if name == "verification":
            data.append(
                [
                    "SmolVLM LoRA",
                    "187 tests; six CLI checks",
                    "7,248 dev + 15,168 held-out scores; 20,971 decoded images; three changed adapters; 20 native GPU cases.",
                ]
            )
        return original(name, caption, headings, data, *args, **kwargs)

    b.table = revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    b.OUT = OUT
    b.SOURCES.clear()
    b.TABLES.clear()
    for name in ["generated", "figures", "numeric_exports", "tools"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    historical_table_updates()
    b.baseline_tables()
    b.expanded_tables()
    fine = b.fine_tables()
    transfer = b.strategy_tables()
    b.pilot_tables()
    component = b.latest_tables()
    b.figures(fine, transfer, component)
    for stage in [
        "clip_finetuning_20261015",
        "strategy_study_20261016",
        "video_pilot_20261016",
    ]:
        shutil.copyfile(
            b.register(f"results/{stage}/comparison.csv"),
            OUT / f"numeric_exports/{stage}_comparison.csv",
        )
    shutil.copyfile(
        b.register("results/video_pilot_20261016/events.csv"),
        OUT / "numeric_exports/video_pilot_20261016_events.csv",
    )
    shutil.copyfile(
        b.register("scripts/reporting/bootstrap_report_latex.py"),
        OUT / "tools/bootstrap_report_latex.py",
    )
    smol_tables()
    b.register("scripts/reporting/build_technical_runthrough.py")
    b.evidence_registry()
    manifest = {
        "report_date": "2026-10-19",
        "timezone": "Asia/Jakarta",
        "experiment_cutoff": "2026-10-19",
        "no_new_model_inference": True,
        "source_files_sha256": dict(sorted(b.SOURCES.items())),
        "table_rows": b.TABLES,
        "latex_files_sha256": {
            str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob("*.tex"))
        },
        "builder_path": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": digest(Path(__file__)),
        "figures_sha256": {
            p.name: digest(p) for p in sorted((OUT / "figures").glob("*.pdf"))
        },
        "numeric_exports_sha256": {
            p.name: digest(p) for p in sorted((OUT / "numeric_exports").glob("*.csv"))
        },
        "figure_build_packages": {
            n: importlib.metadata.version(n) for n in ["matplotlib", "numpy"]
        },
        "numeric_exports": "Preserved study CSVs; main counts are normal/positive unless explicitly total/positive.",
        "primary_smol_metric_rows": 60,
        "smol_heldout_unique_examples": 3792,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        compiler = ROOT / "data/tools/tectonic/tectonic"
        env = {**os.environ, "XDG_CACHE_HOME": str(compiler.parent / "cache")}
        with (OUT / "compile_console.log").open("w") as log:
            subprocess.run(
                [
                    str(compiler),
                    "--untrusted",
                    "--keep-logs",
                    "--keep-intermediates",
                    STEM + ".tex",
                ],
                cwd=OUT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print(
        f"Generated {len(b.TABLES)} tables and six figures from {len(b.SOURCES)} source identities."
    )


if __name__ == "__main__":
    main()
