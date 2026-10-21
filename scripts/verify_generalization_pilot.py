"""Replay source splits, stream detections/alerts, and classifier selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anomaly_eval.training.stream_tiles import MODES, TILES, fuse
from anomaly_eval.training.video import (
    CausalTracker,
    event_metrics,
    make_alerts,
    unattended_state,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generalization_pilot_20261020"
OUT = ROOT / "results/generalization_pilot_20261020"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def equal(actual, expected):
    if isinstance(actual, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for k, v in actual.items():
            equal(v, expected[k])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        assert np.isclose(actual, expected, rtol=1e-7, atol=1e-6), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def streams():
    from scripts.run_generalization_pilot import check_inputs

    check_inputs()
    completion = json.loads((OUT / "detection_complete.json").read_text())
    assert completion["raw_sha256"] == digest(DATA / "raw_tile_predictions.jsonl")

    protocol = json.loads((OUT / "protocol.json").read_text())
    by_path = defaultdict(list)
    for r in rows(DATA / "raw_tile_predictions.jsonl"):
        by_path[r["path"]].append(r)
    assert len(by_path) == protocol["unique_input_images"]
    assert sum(map(len, by_path.values())) == protocol["gpu_views"]
    for path, views in by_path.items():
        assert len(views) == 5 and {tuple(v["tile"]) for v in views} == set(TILES), path
    records = rows(DATA / "recordings.jsonl")
    results = json.loads((OUT / "results.json").read_text())
    expected = {(r["mode"], r["recording_id"]): r for r in results["rows"]}
    saved_traces = {
        (r["mode"], r["recording_id"], r["time"]): r for r in rows(OUT / "traces.jsonl")
    }
    emitted_saved = rows(OUT / "alerts.jsonl")
    counts = 0
    for mode in MODES:
        for record in records:
            tracker = CausalTracker(new_threshold=0.25, low_threshold=0.2)
            times, scores = [], []
            for frame in record["frames"]:
                live = tracker.update(frame["time"], fuse(by_path[frame["path"]], mode))
                equal(live, saved_traces[mode, record["id"], frame["time"]]["tracks"])
                score = unattended_state(live)
                assert (
                    score == saved_traces[mode, record["id"], frame["time"]]["eligible"]
                )
                times.append(frame["time"])
                scores.append(float(score))
                counts += 1
            alerts = make_alerts(times, scores, 0.5)
            equal(
                alerts,
                [
                    r["time"]
                    for r in emitted_saved
                    if r["mode"] == mode and r["recording_id"] == record["id"]
                ],
            )
            events = [
                (a, b)
                for a, b, label in record["intervals"]
                if label == "unattended_items"
            ]
            normal = [
                (a, b) for a, b, label in record["intervals"] if label == "normal"
            ]
            equal(
                event_metrics(alerts, events, normal),
                expected[mode, record["id"]]["metrics"],
            )
            equal(
                event_metrics(
                    [t for t in alerts if t >= 20],
                    events,
                    [(max(a, 20), b) for a, b in normal if b > 20],
                ),
                expected[mode, record["id"]]["aligned_previous_t20_metrics"],
            )
    # Recompute model-visible bytes against prepared identities; no image cache
    # is silently trusted merely because detector completion exists.
    for r in rows(DATA / "input_identities.jsonl"):
        path = ROOT / r["path"]
        assert digest(path) == r["sha256"]
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            actual = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        assert actual == r["rgb_sha256"]
    for aggregate in results["aggregate"]:
        group = [
            r["metrics"]
            for r in results["rows"]
            if r["mode"] == aggregate["mode"]
            and r["evaluation_group"] == aggregate["evaluation_group"]
        ]
        assert aggregate["recordings"] == len(group)
        for key in (
            "events",
            "detected_events",
            "normal_false_alerts",
            "duplicate_alerts",
            "unknown_time_alerts",
            "normal_seconds",
            "reviewed_seconds",
        ):
            equal(sum(r[key] for r in group), aggregate[key])
        delays = [v for r in group for v in r["delays_seconds"]]
        equal(
            float(np.median(delays)) if delays else None,
            aggregate["median_delay_seconds"],
        )
        from anomaly_eval.training.event_memory import poisson_upper_rate

        equal(
            aggregate["detected_events"] / aggregate["events"],
            aggregate["event_recall"],
        )
        equal(
            3600 * aggregate["normal_false_alerts"] / aggregate["normal_seconds"],
            aggregate["false_alerts_per_normal_camera_hour"],
        )
        equal(
            poisson_upper_rate(
                aggregate["normal_false_alerts"], aggregate["normal_seconds"]
            ),
            aggregate["false_rate_one_sided_95pct_upper_per_hour"],
        )
    return {
        "status": "passed",
        "trace_decisions_recomputed": counts,
        "views_checked": protocol["gpu_views"],
        "image_files_checked": len(by_path),
        "recordings": len(records),
        "arms": len(MODES),
    }


def smoking():
    import torch

    from anomaly_eval.data import image_pixel_sha256
    from anomaly_eval.training.clip import binary_metrics, select_points
    from scripts.evaluate_strategy_transfer import matrix
    from scripts.run_strategy_screen import predict
    from scripts.train_source_generalization import FAMILIES, MODELS, context, metrics

    data = DATA / "xrf55"
    records = rows(data / "samples.jsonl")
    identities = rows(data / "input_identities.jsonl")
    pixels = {}
    groups = {}
    for r in records:
        assert digest(ROOT / r["video"]) == r["sha256"]
        assert groups.setdefault(r["group_id"], r["split"]) == r["split"]
        for h in r["pixel_hashes"]:
            assert pixels.setdefault(h, r["split"]) == r["split"]
    for r in identities:
        path = ROOT / r["path"]
        assert digest(path) == r["sha256"]
        assert image_pixel_sha256(path) == r["rgb_sha256"]
    predictions_replayed = 0
    selections = 0
    for model in MODELS:
        directory = OUT / "smoking" / model
        sel = json.loads((directory / "selection.json").read_text())
        dev_predictions = json.loads(
            (directory / "development_predictions.json").read_text()
        )
        old, fresh, features = context(model, include_historical=True)
        indexed = {r["id"]: r for r in old + fresh}
        heads = torch.load(
            directory / "heads.pt", weights_only=True, map_location="cpu"
        )["heads"]
        assert digest(directory / "heads.pt") == sel["heads_sha256"]
        assert sel["protocol_sha256"] == digest(OUT / "smoking/protocol.json")
        for stage in ("full", "crop"):
            cache = data / "features" / model / f"xrf55_{stage}_alpha_0.npz"
            identity = json.loads(cache.with_suffix(".json").read_text())
            assert identity["manifest_sha256"] == digest(data / "samples.jsonl")
            assert identity["model"] == model and identity["alpha"] == 0
            assert identity["stage"] == stage
            with np.load(cache) as value:
                assert set(value["paths"]) == {p for r in records for p in r["views"]}
                np.testing.assert_allclose(
                    np.linalg.norm(value["features"], axis=1), 1, atol=2e-5
                )
        for name, entry in dev_predictions.items():
            dev = [indexed[i] for i in entry["ids"]]
            assert all(r["split"] == "dev" for r in dev)
            scores = predict(
                heads[name],
                matrix(dev, sel["candidates"][name]["recipe"], features, {}),
            )
            np.testing.assert_allclose(scores, entry["scores"], atol=1e-5, rtol=1e-6)
            threshold, cells, key = metrics(dev, scores)
            points = select_points([r["target"] for r in dev], scores)
            points["dev_fpr_budget"] = threshold
            equal(points, sel["candidates"][name]["thresholds"])
            assert sel["candidates"][name]["converged"]
            equal(cells, sel["candidates"][name]["development"])
            assert np.isclose(
                threshold, sel["candidates"][name]["thresholds"]["dev_fpr_budget"]
            )
            equal(list(key), sel["candidates"][name]["selection_key"])
            assert all(c["fpr"] <= 0.05 + 1e-12 for c in cells.values())
            predictions_replayed += len(scores)
        for family in FAMILIES:
            best = max(
                (k for k, v in sel["candidates"].items() if v["family"] == family),
                key=lambda k: sel["candidates"][k]["selection_key"],
            )
            assert best == sel["selected"][family]
            selections += 1
        predictions = rows(directory / "predictions.jsonl")
        result = json.loads((directory / "results.json").read_text())
        for family, name in sel["selected"].items():
            group = [r for r in predictions if r["family"] == family]
            examples = [indexed[r["id"]] for r in group]
            assert all(r["split"] in {"test", "external"} for r in examples)
            scores = predict(
                heads[name],
                matrix(examples, sel["candidates"][name]["recipe"], features, {}),
            )
            np.testing.assert_allclose(
                scores, [r["score"] for r in group], atol=1e-5, rtol=1e-6
            )
            predictions_replayed += len(group)
            for source in {r["source"] for r in group}:
                subset = [r for r in group if r["source"] == source]
                for point, cutoff in sel["candidates"][name]["thresholds"].items():
                    actual = binary_metrics(
                        [r["target"] for r in subset],
                        [r["score"] for r in subset],
                        cutoff,
                    )
                    equal(
                        actual,
                        result["results"][family + "/" + source]["operating_points"][
                            point
                        ],
                    )
        # Independent reconstruction also covers both immutable reference heads.
        import scripts.train_source_generalization as study

        original_candidates = torch.load(
            ROOT / "results/strategy_study_20261016" / model / "smoking_candidates.pt",
            weights_only=True,
            map_location="cpu",
        )["candidates"]
        text = next(
            c["head"]
            for c in original_candidates
            if c["selection"]["classifier"] == "zero_shot"
        )
        legacy_dir = ROOT / "results/strategy_study_20261016" / model
        legacy_policy = json.loads((legacy_dir / "selection.json").read_text())[
            "tasks"
        ]["smoking"]
        legacy = torch.load(
            legacy_dir / "selected_heads.pt", weights_only=True, map_location="cpu"
        )["heads"]["smoking"]
        for family in ("pretrained_text", "legacy_native"):
            group = [r for r in predictions if r["family"] == family]
            examples = [indexed[r["id"]] for r in group]
            if family == "pretrained_text":
                direction = text["weight"].numpy().reshape(-1)
                scores = np.array(
                    [
                        np.mean(
                            [features["full", 0][p] @ direction for p in r["views"]]
                        )
                        for r in examples
                    ]
                )
                points = {"raw": 0}
            else:
                native = [{**r, "views": r["views"][-1:]} for r in examples]
                scores = predict(
                    legacy, matrix(native, legacy_policy["strategy"], features, {})
                )
                points = legacy_policy["thresholds"]
            np.testing.assert_allclose(
                scores, [r["score"] for r in group], atol=1e-5, rtol=1e-6
            )
            predictions_replayed += len(group)
            for source in {r["source"] for r in group}:
                ids = [i for i, r in enumerate(group) if r["source"] == source]
                for point, cutoff in points.items():
                    equal(
                        binary_metrics(
                            [group[i]["target"] for i in ids], scores[ids], cutoff
                        ),
                        result["results"][family + "/" + source]["operating_points"][
                            point
                        ],
                    )
        study.check()
    return {
        "status": "passed",
        "examples": len(records),
        "decoded_images_checked": len(identities),
        "source_videos_checked": len(records),
        "development_and_selected_heldout_scores_replayed": predictions_replayed,
        "family_selections_reconstructed": selections,
        "source_and_subject_groups_disjoint": True,
    }


def smolvlm():
    from anomaly_eval.training.clip import binary_metrics, select_points

    directory = OUT / "smolvlm_transfer"
    protocol = json.loads((directory / "protocol.json").read_text())
    assert digest(directory / "samples.jsonl") == protocol["manifest_sha256"]
    assert (
        digest(ROOT / "scripts/evaluate_smolvlm_source_pilot.py")
        == protocol["code_sha256"]
    )
    examples = {r["id"]: r for r in rows(directory / "samples.jsonl")}
    source = {r["id"]: r for r in rows(DATA / "xrf55/samples.jsonl")}
    for r in examples.values():
        assert len(r["views"]) == len(r["pixel_hashes"]) == len(r["frames"]) == 8
        for key in ("views", "pixel_hashes", "frames", "timestamps"):
            equal(
                r[key],
                [source[r["id"]][key][i] for i in protocol["source_view_indices"]],
            )
    cache = directory / "visual_cache"
    index = json.loads((cache / "index.json").read_text())
    assert index["tokens_sha256"] == digest(cache / "tokens.npy")
    checked = 0
    for name in ("original", "lora"):
        path = directory / name
        dev = rows(path / "dev_predictions.jsonl")
        test = rows(path / "test_predictions.jsonl")
        assert len(dev) == 160 and len(test) == 240
        assert all(examples[r["id"]]["split"] == "dev" for r in dev)
        assert all(examples[r["id"]]["split"] == "test" for r in test)
        calibration = json.loads((path / "calibration.json").read_text())
        points = select_points([r["target"] for r in dev], [r["score"] for r in dev])
        equal(points, calibration["new_source_thresholds"])
        result = json.loads((path / "results.json").read_text())
        assert result["native_cached_score_abs_error"] < 0.002
        for output, cutoffs in (
            ("new_source_calibrated", points),
            ("original19oct_cutoffs", calibration["original19oct_thresholds"]),
        ):
            for point, cutoff in cutoffs.items():
                equal(
                    binary_metrics(
                        [r["target"] for r in test], [r["score"] for r in test], cutoff
                    ),
                    result[output][point],
                )
        checked += len(dev) + len(test)
    return {
        "status": "passed",
        "score_records_checked": checked,
        "frozen_original_and_adapter_routes": 2,
        "new_training": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "streams"), default="all")
    args = parser.parse_args()
    result = {"streams": streams(), "no_latency_testing": True}
    if args.stage == "all":
        result.update(smoking=smoking(), smolvlm=smolvlm())
    result["status"] = "passed"
    result["artifact_hashes"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            OUT / "protocol.json",
            OUT / "results.json",
            OUT / "smoking/protocol.json",
            DATA / "source_audit.json",
        ]
    }
    filename = (
        "verification.json" if args.stage == "all" else "stream_verification.json"
    )
    (OUT / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
