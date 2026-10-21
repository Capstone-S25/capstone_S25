"""Build the complete progress deck, including the 19 October SmolVLM study."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from reportlab import rl_config

from reports.slides import build_progress_presentation as prior

d = prior.d
ROOT, OUT = prior.ROOT, prior.OUT
STEM = "model_evaluation_progress_20261019"
STUDY = "results/smolvlm_finetuning_20261019"
MAIN = [
    "smoking_cctv",
    "mivia_iwdd",
    "street_scene",
    "commons_visible_sleeping",
    "unattended_reviewed",
]
NAMES = {
    "kaggle_sujaykapadnis_smoking": "Smoking photos / Kaggle",
    "smoking_cctv": "Smoking / CCTV",
    "mivia_iwdd": "Littering / IWDD",
    "street_scene": "Loitering / Street Scene",
    "commons_visible_sleeping": "Public sleep / historical",
    "unattended_reviewed": "Unattended / historical",
    "stanford40": "Smoking / Stanford40",
    "hmdb51_action_transfer": "Smoking / HMDB",
    "commons_new_author_transfer": "Public sleep / new authors",
    "meva_abandon_action_transfer": "MEVA abandonment action",
}
new, paired, tab, brief = prior.new, prior.paired, prior.tab, prior.brief


def load(name):
    return prior.load(STUDY + "/" + name)


def summary():
    s = new(
        "Findings / updated executive summary",
        "Five-anomaly model evaluation",
        "Data, testing and iterative improvement • completed studies through 19 October 2026",
        "Preserved results + SmolVLM matched-control evaluation",
        "The full history remains visible. Six CLIP visual fine-tunes and three actual SmolVLM language-layer LoRA runs changed model weights. The 330 strategy configurations, 50 temporal heads and 38 smoking policies are separate studies, repeatedly using some sources. SmolVLM has 3,792 unique held-out examples across ten cohorts, not 3,792 independent events. Four historical target rankings improve with primary per-task routing, but loitering regresses and unattended FPR is 70.8% at the development-budget cutoff. No five-event deployment winner is established; no new latency benchmark is run.",
    )
    d.box(s, 42, 166, 876, 101, d.NAVY)
    d.text(
        s,
        61,
        181,
        834,
        76,
        "SmolVLM learns useful task rankings.\nCamera transfer and false alarms remain unresolved.",
        26,
        d.WHITE,
        True,
    )
    for x, value, label, detail in [
        (42, "6 + 3", "Changed-weight runs", "6 CLIP visual + 3 SmolVLM LoRA"),
        (
            340,
            "3,792",
            "SmolVLM held-out cases",
            "10 source/task cohorts; retrospective",
        ),
        (638, "70.8%", "Unattended endpoint FPR", "94.6% recall at dev-budget cutoff"),
    ]:
        d.stat(s, x, 282, 280, 142, value, label, detail)
    brief(
        s,
        "Next decisive evidence: independently reviewed event histories and long normal streams on new cameras.",
        d.ORANGE,
    )


def models():
    tab(
        "Models / current comparison scope",
        "Adaptation now includes SmolVLM language weights",
        "Five requested model families • protocols and trainable components remain explicit",
        "Completed CLIP, SmolVLM and separate VadCLIP artifacts",
        ["Model", "Completed adaptation / evaluation", "Current comparison scope"],
        [
            [
                "MobileCLIP2-S2",
                "Visual fine-tuning; region / temporal heads",
                "Five targets; actual visual updates",
            ],
            [
                "SigLIP2-B/16-224",
                "Visual fine-tuning; region / temporal heads",
                "Five targets; actual visual updates",
            ],
            [
                "SmolVLM-500M",
                "NEW: language-layer LoRA; binary margins",
                "Five targets; matched original control",
            ],
            [
                "MiniCPM-V 4.0",
                "Deterministic prompted image responses",
                "Historical expanded image comparison",
            ],
            [
                "VadCLIP",
                "Separate supervised offline smoking adapter",
                "Different split / supervision; unchanged",
            ],
        ],
        [202, 370, 304],
        "Three SmolVLM adapters were trained. Its frozen vision tower differs from the earlier CLIP visual updates.",
        "SmolVLM-500M-Instruct is adapted through q/k/v/o and gate/up/down language projections. No new MiniCPM or VadCLIP training is claimed. The older four-model common-image comparison remains historical. New SmolVLM AP uses first-token binary margins, not hard JSON categories; its original matched baseline is essential. VadCLIP remains a separately supervised 1,056-frame smoking protocol.",
        font=12.5,
    )


def chronology():
    tab(
        "Work completed / dated progression",
        "From common prompts to controlled model adaptation",
        "Counts describe separate studies and reuse some sources",
        "Run, selection and verification artifacts",
        ["Stage", "Completed work", "Evidence / qualification"],
        [
            [
                "13–14 Oct",
                "13,400 then 55,156 predictions",
                "Four-model comparison; qualified labels",
            ],
            [
                "15–16 Oct",
                "6 actual CLIP visual fine-tunes",
                "2 encoders × 3 seeds; changed weights",
            ],
            [
                "16 Oct",
                "330 configurations; 50 video heads",
                "Crops, source transfer, tracks and events",
            ],
            [
                "17 Oct",
                "38 policies; state replay; 830 views",
                "Ensembles, memory and fixed tiling",
            ],
            [
                "18 Oct",
                "83-page full technical report",
                "Documentation; no new training that day",
            ],
            [
                "19 Oct",
                "3 SmolVLM LoRA runs; 10 cohorts",
                "Seed replication; threshold failures retained",
            ],
        ],
        [110, 365, 401],
        "The updated report and deck preserve earlier versions. No new inference-speed or latency test was added.",
        "The SmolVLM round reuses collected and previously inspected source data. Its three runs are seed42 at learning rates 3e-5 and 1e-4, plus seed43 at the primary winning rate 1e-4. All choose epoch one after three completed epochs. More configurations, crops, overlapping windows or seeds are not more independent cameras/events.",
        row_h=34,
        font=12.5,
    )


def smol_content():
    data = load("report_data.json")
    run = load("run.json")
    assert run["status"] == "complete"
    protocol = load("protocol.json")
    load("seed_42/selection.json")
    repeat = load("repeated_gain_routes.json")
    verification = load("verification.json")
    software = load("software_checks.json")
    provenance = load("clip_comparison_provenance.json")
    assert provenance["exact_heldout_labels_checked"] == 3792
    for name in [
        "metrics.csv",
        "model_bundle.json",
        "response_mass.json",
        "repeated_gain_protocol.json",
        "additional_transfer/protocol.json",
        "implementation_manifest.json",
        "portable_adapter_export.json",
    ]:
        prior.register(STUDY + "/" + name)
    for path in [
        "reports/smolvlm_finetuning_findings_20261019.md",
        "docs/smolvlm_finetuning.md",
        "reports/technical_runthrough_20261019/source_manifest.json",
    ]:
        prior.register(path)
    cells = data["cells"]
    lookup = {(r["seed"], r["source"], r["method"]): r for r in cells}
    assert len(cells) == 60
    assert sum(lookup[42, s, "routed"]["n"] for s in NAMES) == 3792
    ap = lambda s, m="routed", seed=42: f"{lookup[seed, s, m]['ap']:.4f}"
    pct = lambda r, k: f"{r[k]:.1%}"
    count = lambda s: (
        f"{lookup[42, s, 'original']['n']} / {lookup[42, s, 'original']['positive']}"
    )

    paired(
        "19 Oct / controlled SmolVLM experiment",
        "Use the original model as a matched weight control",
        "One task query per example • real first-answer scores • same inputs before / after adaptation",
        "SmolVLM protocol + original / adapter predictions",
        (
            "Matched within this study",
            "Same accepted target definitions\nSame 512-pixel letterbox images\nSame ordered views and visual tokens\nSame binary Yes / No task query\nSame first-token logit margin",
        ),
        (
            "Comparison limits",
            "Earlier study generated six-label JSON\nPrompt / preprocessing also changed\nNo learned unknown or explanations\nNo boxes or tracked identities added\nRetrospective source holdouts",
        ),
        "Attribute adaptation gains to the matched original-to-LoRA comparison; preserve the old generation results.",
        "Score = logit(Yes) minus logit(No). This is a genuine continuous ranking score and is not calibrated event probability. Yes+No probability mass is already high before training: minimum 98.4%, median 99.9% on development. The original and all adapted paths share the frozen image encoder and connector. No other task is fabricated as negative. The new interface differs from the earlier free-generated JSON protocol.",
        warn=True,
    )

    rows = []
    for t, name in [
        ("smoking", "Smoking"),
        ("littering", "Littering"),
        ("loitering", "Loitering"),
        ("rough_sleeping", "Public sleep"),
        ("unattended_items", "Unattended"),
    ]:
        rows.append(
            [name]
            + [
                f"{protocol['counts'].get(f'{s}/{t}/0', 0)} / {protocol['counts'].get(f'{s}/{t}/1', 0)}"
                for s in ["train", "dev", "test"]
            ]
        )
    tab(
        "19 Oct / supervised data",
        "All five tasks are trained; evidence is uneven",
        "3,973 train • 1,812 development • 2,025 test • 806 historical external photos",
        "Frozen SmolVLM protocol / counts",
        ["Objective", "Train normal / +", "Dev normal / +", "Test normal / +"],
        rows,
        [240, 212, 212, 212],
        "8,616 main examples use 13,596 distinct images. Unknown objectives are masked; source groups stay together.",
        "These are normal/positive counts, unlike the total/positive header in the Markdown summary. Smoking test comprises 646 CCTV frames and 224 Kaggle photos; Stanford40 adds 806 external photos with 140 positives. Training uses exactly the existing 3,973 strategy examples, not additional new downloads. Loitering has one physical camera despite multiple recording groups. Sleep labels are provisional. Unattended endpoints are correlated, with two test source families.",
        font=13,
    )

    paired(
        "19 Oct / transfer and source guards",
        "Ten cohorts cover 3,792 unique held-out examples",
        "Main holdouts plus 961 additional transfer cases • no transfer labels enter training or selection",
        "Additional-transfer protocol + identity verification",
        (
            "Additional transfer",
            "HMDB: 906 clips, 109 smoking\nNew-author sleep: 28 photos, 17 +\nMEVA: 27 windows, 15 actions\n7,375 extra distinct decoded images\n20,971 total decoded images checked",
        ),
        (
            "What the data can establish",
            "HMDB: movie / web action transfer\nSleep: provisional visible context\nMEVA: action, not persistent state\nExact pixels / groups checked\nOld holdouts remain retrospective",
        ),
        "More frames and seeds do not add independent cameras. This round reuses previously collected data.",
        "HMDB other-primary-action controls are non-exhaustive negatives. Eight-view clip smoking differs from single-image smoking training. Twenty-eight public-sleep cases have 25 author groups. MEVA has 15 action windows and 12 pre-event controls and cannot establish persistent ownership. Existing historical tests have been inspected before. No new dataset is downloaded for this LoRA round.",
        warn=True,
    )

    paired(
        "19 Oct / actual LoRA training",
        "Adapt 4.34 million language parameters",
        "32 language layers × 7 projections • rank 8 • alpha 16 • dropout 0.05",
        "Training implementation + finite changed-adapter checks",
        (
            "What changes",
            "q/k/v/o + gate/up/down projections\n224 adapted language modules\n4,341,760 trainable parameters\nAnswer-only full-vocabulary loss\n~17 MB saved weights per adapter",
        ),
        (
            "Frozen inputs and optimization",
            "Vision + connector + embeddings frozen\n64 cached visual tokens / image\n1 static / 4 temporal / 8 clip views\nBF16 training; F32 final scoring\nAdamW; effective batch 32",
        ),
        "This is model-weight adaptation. It does not update the image encoder or learn tracking and ownership.",
        "LoRA also freezes the vocabulary output head. Every epoch visits 3,973 unique training examples once, then 3,973 balanced resamples; 7,946 visits total. The resampled half balances task/class/source/group, while the combined epoch retains original imbalance. AdamW weight decay 0.01, 5% warmup, linear decay, gradient clipping 1, microbatch 8 and accumulation 4. CUDA accuracy is unpaced; there is no new latency benchmark.",
    )

    candidates = []
    histories = []
    for seed, lr in [(42, "3e-05"), (42, "0.0001"), (43, "0.0001")]:
        state = load(f"seed_{seed}/lr_{lr}/state.json")
        h = load(f"seed_{seed}/lr_{lr}/history.json")
        candidates.append(
            [
                str(seed),
                lr,
                str(state["best_epoch"]),
                str(state["epochs_completed"]),
                f"{state['best_dev_macro_ap']:.4f}",
            ]
        )
        histories.append((seed, lr, h))
    tab(
        "19 Oct / checkpoint selection",
        "Three trained candidates all select epoch one",
        "Selection: mean source AP within each task, then equal macro average over five tasks",
        "Three state files + primary frozen selection",
        ["Seed", "Learning rate", "Best epoch", "Epochs run", "Dev macro AP"],
        candidates,
        [110, 210, 165, 165, 226],
        "Original dev AP 0.5657 → primary joint 0.7026. Task routing uses the low LR only for public sleep.",
        "Seed 42 compares two rates. Seed 43 repeats only the winning high rate; it is not a full repeated grid. Planned maximum six epochs, patience two; each completes three epochs and selects epoch one. Routes and thresholds precede their held-out predictions. Joint means one high-rate adapter for all tasks. Routed means per-task original/adapter choices using development AP. Do not substitute a better test seed.",
        row_h=54,
        font=15,
    )

    s = new(
        "19 Oct / optimization behavior",
        "Training loss falls while development AP declines",
        "Primary high learning rate: 7,946 example visits per epoch • selected checkpoint = epoch 1",
        "seed_42/lr_0.0001/history.json",
        "All three runs select epoch one. The primary high-rate training loss is 0.4789, 0.2895, 0.2235, while development macro AP is 0.7026, 0.6485, 0.6276. More epochs fit the training corpus but worsen this source-transfer diagnostic. Nine actual epochs total across candidates yield 71,514 visits to the same 3,973 training examples. Frozen visual extraction avoids repeating unchanged computation.",
    )
    d.table(
        s,
        42,
        170,
        [150, 230, 250, 246],
        ["Epoch", "Train loss", "Dev macro AP", "Checkpoint status"],
        [
            [
                str(h["epoch"]),
                f"{h['loss']:.4f}",
                f"{h['dev_macro_ap']:.4f}",
                "SELECTED" if h["epoch"] == 1 else "Retained history",
            ]
            for h in histories[1][2]
        ],
        row_h=52,
        font=16,
        header_h=40,
    )
    d.text(
        s,
        58,
        381,
        844,
        34,
        "Source transfer deteriorates before the planned six-epoch limit.",
        18,
        d.ORANGE,
        True,
    )
    brief(
        s,
        "Early stopping protects the selected checkpoint; the lowest-loss final weights are not promoted.",
        d.ORANGE,
    )

    main = ["kaggle_sujaykapadnis_smoking"] + MAIN
    tab(
        "19 Oct / matched primary ranking",
        "LoRA improves selected rankings and regresses on loitering",
        "Primary seed 42 • identical inputs and binary scoring • AP is ranking, not accuracy",
        "SmolVLM metrics.csv / original, joint and routed",
        ["Task / source", "Total / +", "Original AP", "Joint AP", "Routed AP"],
        [[NAMES[s], count(s), ap(s, "original"), ap(s, "joint"), ap(s)] for s in main],
        [292, 125, 153, 153, 153],
        "Joint sleep regresses; low-LR routing helps. CCTV AP 0.6225 is only slightly above constant AP 0.6084.",
        "Four of five historical target rankings improve under per-task routing, counting CCTV as the smoking target. Loitering falls from 0.4250 to 0.1822. Historical sleep has six positives: joint falls from 0.7437 to 0.6327 while the routed low-learning-rate adapter reaches 0.7885. Unattended ranking rises from 0.4762 to 0.6933 but decisions remain problematic. Photo smoking has different prevalence and domain from CCTV. The constant-score CCTV AP is 393/646 = 0.6084.",
        row_h=34,
        font=13,
        warn=True,
    )

    transfer = [
        "stanford40",
        "hmdb51_action_transfer",
        "commons_new_author_transfer",
        "meva_abandon_action_transfer",
    ]
    tab(
        "19 Oct / other-source transfer",
        "Photo and action-source gains do not validate events",
        "No transfer example is used for SmolVLM training, route selection or calibration",
        "SmolVLM additional-transfer / source summaries",
        ["Source", "Total / +", "Original AP", "Joint AP", "Routed AP"],
        [
            [NAMES[s], count(s), ap(s, "original"), ap(s, "joint"), ap(s)]
            for s in transfer
        ],
        [292, 125, 153, 153, 153],
        "All four routed APs rise. Labels remain source-specific: action clips, provisional sleep, and abandonment actions.",
        "Stanford40 is a historical external photograph holdout. HMDB includes 906 clips with 109 smoking positives and 797 other primary actions; controls are not exhaustive negatives. New-author sleep has 28 cases and 25 author groups. The 27 MEVA windows do not prove persistent unattended state. These sources were inspected in earlier experiments, so transfer is retrospective. No continuous event false-alert rate is measured here.",
        row_h=48,
        font=13,
        warn=True,
    )

    tab(
        "19 Oct / fixed decisions",
        "The development FPR cutoff fails on unattended items",
        "Each target cutoff allows ≤5% pooled development FPR; held-out FPR can exceed it",
        "Primary routed scores / frozen development-budget cutoffs",
        ["Task / source", "Recall", "FPR", "True / false positives"],
        [
            [
                NAMES[s],
                pct(lookup[42, s, "routed"], "recall_at_dev_5pct"),
                pct(lookup[42, s, "routed"], "fpr_at_dev_5pct"),
                f"{lookup[42, s, 'routed']['tp_at_dev_5pct']} / {lookup[42, s, 'routed']['fp_at_dev_5pct']}",
            ]
            for s in MAIN
        ],
        [330, 140, 140, 266],
        "Unattended: 510 / 720 normals flagged. CCTV: only 7 / 393 positives found; raw margin zero finds none.",
        "At the development-budget point unattended recall is 94.6% and FPR is 70.8%; original values are 20.2% and 1.9%. This is not a uniformly superior decision tradeoff. Littering recall/FPR is 30%/8%; loitering 9.1%/6.2%; historical sleep 33.3%/8.3%. MEVA is 66.7%/58.3%. Endpoint FPR is not hourly alert rate. Cutoffs are fitted per target on pooled development normals, not every source separately.",
        font=13.5,
        warn=True,
    )

    tab(
        "19 Oct / optimization replication",
        "The second seed repeats the loitering failure",
        "Fixed primary winning learning rate • one joint adapter per seed • same held-out sources",
        "Seed 42 / 43 joint-adapter source AP",
        ["Task / source", "Seed 42 joint AP", "Seed 43 joint AP"],
        [[NAMES[s], ap(s, "joint", 42), ap(s, "joint", 43)] for s in MAIN],
        [410, 233, 233],
        "Sleep varies greatly on six positives. Optimization replication adds no independent camera evidence.",
        "Seed 43 is a limited stability check and is never selected using test outcomes. Sleep joint AP 0.6327 versus 0.9167 is sensitive on 18 cases. The primary per-task sleep route is a different low-LR adapter, AP 0.7885. HMDB joint AP 0.7837 versus 0.7862 and unattended AP 0.6933 versus 0.7052 broadly repeat. Loitering worsens from 0.1822 to 0.1314. Existing source/annotation limits remain.",
        font=14,
        warn=True,
    )

    context = {}
    for model in prior.MODELS:
        values = prior.load(
            f"results/strategy_study_20261016/{model}/transfer_evaluation/per_source_results.json"
        )
        for source in NAMES:
            matching = [
                v["selected"]["raw"]
                for k, v in values.items()
                if k.endswith("/" + source)
            ]
            assert len(matching) == 1
            row = matching[0]
            assert row["n"] == lookup[42, source, "routed"]["n"]
            context[model, source] = row["ap"]
    tab(
        "19 Oct / context on identical cases",
        "CLIP strategies lead on most historical cohorts",
        "Same 3,792 held-out IDs / labels; crops, resolution and training differ across pipelines",
        "16 Oct selected CLIP strategy routes + SmolVLM primary routes",
        ["Task / source", "MobileCLIP AP", "SigLIP AP", "Smol routed AP"],
        [
            [NAMES[s]] + [f"{context[m, s]:.4f}" for m in prior.MODELS] + [ap(s)]
            for s in MAIN + ["hmdb51_action_transfer"]
        ],
        [330, 182, 182, 182],
        "SmolVLM is stronger on 906-case HMDB transfer. Later CLIP video heads use a different 205-case protocol.",
        "This compares complete pipelines, not an architecture-only ablation. Same learning/development splits are 3,973/1,812. The matched original SmolVLM is the clean weight-adaptation control. Later SigLIP ensemble AP 0.9645 uses 530 HMDB training clips, 205 test clips and 16 views; comparing that directly to SmolVLM transfer on 906 cases with eight views would confound supervision and examples. Historical VadCLIP also has a different split.",
        row_h=34,
        font=13,
        warn=True,
    )

    tab(
        "19 Oct / optional retrospective safeguard",
        "Require a development gain in both optimization seeds",
        "Rule introduced after primary held-out inspection • optional, exploratory and retrospective",
        "repeated_gain_protocol.json + repeated_gain_routes.json",
        ["Target", "Primary dev gain", "Seed-43 dev gain", "Optional route"],
        [
            [
                t.replace("_", " ").title(),
                f"{v['primary_development_ap_gain']:+.4f}",
                f"{v['secondary_development_ap_gain']:+.4f}",
                "Original"
                if repeat["task_routes"][t] == "original"
                else ("Low LR" if "3e-05" in repeat["task_routes"][t] else "High LR"),
            ]
            for t, v in repeat["development_gains"].items()
        ],
        [244, 190, 196, 246],
        "Only loitering returns to the original. Primary results stay unchanged; unattended false positives remain.",
        repeat["qualification"]
        + " The function reads only development scores, but the decision to invent it was made after primary held-out inspection and the first secondary development epoch. Restored targets reuse original development thresholds; others retain primary routes. Seed 43 repeats the high learning rate, so this is not exact replication of the low-LR sleep route. CLI option: --repeated-gains.",
        font=13.5,
        warn=True,
    )

    paired(
        "19 Oct / verification and inference artifacts",
        "Check actual checkpoints and native image execution",
        "Saved research bundle includes adapter weights, portable base identity, task routes and cutoffs",
        "SmolVLM verification.json + software_checks.json",
        (
            "Experiment reconstruction",
            f"{verification['development_predictions_replayed']:,} dev scores replayed\n{verification['heldout_predictions_replayed']:,} held-out scores replayed\n20,971 decoded images checked\n3 finite changed adapters checked\n20 native GPU / cached comparisons",
        ),
        (
            "Software and saved output",
            f"{software['unit_tests_passed']} unit tests passed\n6 real-input / missing-context checks\nMax native score error < 0.000073\nPortable adapters + processor files\nResearch bundle; no event certification",
        ),
        "Functional CLI checks verify scores and guards. They are not six successful accuracy predictions.",
        "Largest native-versus-saved error 7.2479e-5 is below the 0.002 tolerance. Four actual native CLI scoring cases and two missing-timestamp temporal guards pass. Some chosen positive inputs are classified absent under strict cutoffs, consistent with low recall. PEFT export changes machine-local base config to public pinned identity, leaving weight bytes unchanged. Source hashes, response mass and exact cutoffs are preserved. No new latency test is performed.",
    )


def closing():
    tab(
        "Current assessment / all five objectives",
        "No single model reliably covers all five events",
        "Use source-qualified candidates; do not replace event evidence with a model confidence score",
        "All preserved studies through 19 October",
        ["Objective", "Useful evidence so far", "Main remaining gap"],
        [
            [
                "Smoking",
                "Photo learning; action ensembles; crops",
                "CCTV low-FPR recall; tiny people",
            ],
            [
                "Littering",
                "CLIP / LoRA learn IWDD dumping actions",
                "Small disposal vs placing; online onset",
            ],
            [
                "Loitering",
                "Geometry / provided tracks; original control",
                "LoRA regression; identities / new cameras",
            ],
            [
                "Public sleep",
                "Photo probes; low-LR VLM route",
                "Tiny provisional sets; persistent video",
            ],
            [
                "Unattended",
                "Grounded items + explicit state rule",
                "Owner truth; threshold shift; normal hours",
            ],
        ],
        [165, 355, 356],
        "Ranking gains are real within their protocols. Reliable five-event camera transfer is still unproven.",
        "SmolVLM provides a useful language-adaptation experiment but does not supersede all CLIP strategies. Keep the loiter regression, joint-sleep regression and unattended FPR failure visible. Tiling improves component recall, not verified event performance. AVSS still has only two clean primary events and 2.76 normal minutes. MiniCPM and VadCLIP are historical baselines, with different scope.",
        font=12.5,
        warn=True,
    )
    paired(
        "Findings / retain gains and failures",
        "Generalization needs evidence beyond another training run",
        "All tested interventions remain auditable, including unsuccessful choices",
        "Preserved checkpoints, source metrics and event traces",
        (
            "Useful progress",
            "Correct labels and source-group guards\nTrue CLIP and language-weight updates\nDumping and photo ranking gains\nMatched original controls and replays\nExplicit item state and tiled recall",
        ),
        (
            "Limits that repeated",
            "Photo success does not settle CCTV\nHigher AP can still yield high FPR\nLater epochs can overfit quickly\nSeed agreement adds no new cameras\nSparse views lack owner / dwell truth",
        ),
        "The optional LoRA safeguard is retrospective. A new frozen source holdout is needed to validate it.",
        "The study makes failure mechanisms more concrete: few source families, tiny positive cohorts, small person/object detail, incomplete ownership/duration labels and threshold domain shift. More epochs, generic language explanations or extra correlated frames alone do not address those evidence gaps. Neither current research routes nor the optional safeguard are validated deployment choices.",
        warn=True,
    )
    tab(
        "Next steps / generalization gates",
        "Freeze new camera evidence before selecting another model",
        "Existing definitions retained • improve source diversity, temporal labels and realistic controls",
        "Current observed failure modes",
        ["Priority", "Concrete work", "Acceptance evidence"],
        [
            [
                "1. Label / collect",
                "New cameras; full event / owner histories",
                "Independent review + matched controls",
            ],
            [
                "2. Components",
                "Person / item coverage; identity; persistence",
                "Automatic inputs; explicit error audit",
            ],
            [
                "3. Adapt",
                "Source-balanced confirmed examples; early stop",
                "Original control; multiple source folds",
            ],
            [
                "4. Validate",
                "Freeze route / cutoff; untouched long streams",
                "Recall, delay, FAR, repeats and bounds",
            ],
        ],
        [130, 375, 371],
        "Keep unknown when the required visible, duration or ownership evidence is missing.",
        "Acquire smoking controls (drinking, eating, coughing); disposal versus placing/pickup; legitimate waiting and complete dwell tracks; persistent sleep versus awake rest; owner-associated items with retrieval/occlusion. Use unseen camera/day blocks and review uncertainty. Evaluate controlled low-LR/regularized adapters or vision adaptation only after adding compatible evidence. Do not claim normal alert safety from a few zero-alert minutes.",
        row_h=50,
        font=12.5,
    )
    paired(
        "Deliverables / reproducibility",
        "Updated paper, editable slides and source-linked findings",
        "Complete historical account plus detailed SmolVLM methods, results and limitations",
        "Technical runthrough + per-stage data / source manifests",
        (
            "Detailed technical paper",
            "All original data / testing stages\nNew full SmolVLM training chapter\nAll 60 method / seed / source cells\nExact cutoffs and epoch histories\nLaTeX source + portable source ZIP",
        ),
        (
            "Presentation and evidence",
            "Editable PPTX + matching vector PDF\nUpdated one-slide summary\nSpeaker notes + source identities\nRendered previews + layout checks\nEarlier report / slides preserved",
        ),
        "This documentation build reads completed results. It runs no new model evaluation or latency benchmark.",
        "Updated paper: reports/technical_runthrough_20261019/model_evaluation_runthrough_20261019.pdf. SmolVLM findings: reports/smolvlm_finetuning_findings_20261019.md; guide: docs/smolvlm_finetuning.md. Rebuild using scripts/build_technical_runthrough_20261019.py and reports/slides/build_progress_presentation_20261019.py. The paper and slides cite stage-specific sources and preserve failed interventions.",
    )


def build_content():
    prior.build_content()
    history = deepcopy(d.SLIDES[1:28])
    d.SLIDES.clear()
    summary()
    d.SLIDES.append(history[0])
    models()
    chronology()
    for s in history[3:]:
        s["notes"] = s["notes"].replace(
            "These models have not been rerun or fine-tuned in subsequent CLIP studies.",
            "At the 14 October cutoff these are original prompted models. A separate 19 October study later adapts SmolVLM language weights under a new matched binary protocol.",
        )
        d.SLIDES.append(s)
    smol_content()
    closing()
    assert len(d.SLIDES) == 45, len(d.SLIDES)
    for i, s in enumerate(d.SLIDES, 1):
        for item in s["items"]:
            if item["kind"] != "text":
                continue
            if item["y"] == 514 and item["x"] == 42:
                item["value"] = item["value"].replace(
                    "PROGRESS 17 OCT 2026", "PROGRESS 19 OCT 2026"
                )
            if item["x"] == 882 and item["y"] == 512:
                item["value"] = f"{i:02d}"
        for item in s["items"]:
            if item["kind"] == "text":
                assert (
                    len(item["value"].splitlines()) * item["leading"]
                    <= item["h"] + 0.01
                ), (s["title"], item)
    d.check_geometry(d.SLIDES)


def main():
    build_content()
    # Use an embedded font for the initial PDF graphics state as well as all text.
    rl_config.canvas_basefontname = "DejaVu"
    for stem, slides in [
        (STEM, d.SLIDES),
        ("summary_slide_progress_20261019", [deepcopy(d.SLIDES[0])]),
    ]:
        path = OUT / (stem + ".pptx")
        d.render_pptx(slides, path)
        ppt = Presentation(path)
        ppt.core_properties.title = "Five-anomaly evaluation and adaptation progress"
        ppt.core_properties.subject = (
            "Data, methods, results and improvements through 19 October 2026"
        )
        ppt.save(path)
        d.render_pdf(slides, OUT / (stem + ".pdf"))
    notes = [
        "# Evaluation and adaptation progress — speaker notes",
        "",
        "Updated 19 October 2026 (Asia/Jakarta). Historical protocols stay separate.",
        "",
    ]
    for i, s in enumerate(d.SLIDES, 1):
        notes += [f"## {i:02d}. {s['title']}", "", s["notes"], ""]
    (OUT / "progress_speaker_notes_20261019.md").write_text("\n".join(notes))
    builders = [
        Path(__file__),
        OUT / "build_progress_presentation.py",
        OUT / "build_presentation.py",
    ]
    manifest = {
        "date": "2026-10-19",
        "timezone": "Asia/Jakarta",
        "slide_count": len(d.SLIDES),
        "sources_sha256": dict(sorted(d.SOURCES.items())),
        "builders_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in builders
        },
        "layout": d.SLIDES,
        "pdf_rendering": "Independent vector PDF from same measured layout; embedded fonts.",
        "no_new_model_inference": True,
    }
    (OUT / "progress_source_manifest_20261019.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Created {len(d.SLIDES)} slides, PDF/PPTX, summary, notes and source identities."
    )


if __name__ == "__main__":
    main()
