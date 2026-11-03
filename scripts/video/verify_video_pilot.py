"""Replay saved pilot metrics, selections, checkpoint scores and live encodings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import Encoder, binary_metrics, select_points
from scripts.video.evaluate_video_pilot import summarize_events
from scripts.video.extract_video_pilot import Views, regions
from scripts.video.guard_video_sources import canonical_title
from scripts.video.train_grounded_pilot import grounded_features
from scripts.video.train_person_pilot import instance_arrays, score_person
from scripts.video.train_video_pilot import batch_arrays, score_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert read(data / "data_verification.json")["status"] == "pass"
    rows = [
        r
        for n in ["modeling_windows.jsonl", "verified_fresh_avss.jsonl"]
        for r in map(json.loads, (data / n).read_text().splitlines())
    ]
    index = {r["id"]: r for r in rows}
    assert len(index) == len(rows)
    film_folds = defaultdict(set)
    for r in rows:
        if r["source"] == "hmdb51_action_transfer":
            film_folds[canonical_title(r["group_id"])].add(r["split"])
    assert all(len(v) == 1 for v in film_folds.values())
    fresh_records = {
        r["id"]: r
        for r in map(
            json.loads, (data / "fresh_recordings.jsonl").read_text().splitlines()
        )
    }
    old_records = {
        r["id"]: r
        for r in map(json.loads, (data / "recordings.jsonl").read_text().splitlines())
        if r["mode"] == "continuous"
    }
    tracks = {
        r["path"]: r["tracks"]
        for r in map(json.loads, (data / "tracks.jsonl").read_text().splitlines())
    }
    audit = {
        "status": "complete",
        "predictions_replayed": 0,
        "candidate_selections_verified": 0,
        "saved_head_checks": [],
        "live_feature_checks": [],
        "known_title_alias_folds_isolated": True,
        "checkpoints": {},
        "data_verification_sha256": hashlib.sha256(
            (data / "data_verification.json").read_bytes()
        ).hexdigest(),
    }
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = run / model
        selection = read(directory / "joint_selection.json")
        results = read(directory / "results.json")
        policies = read(directory / "event_policies.json")
        assert (
            selection["manifest_sha256"]
            == hashlib.sha256(
                (data / "modeling_windows.jsonl").read_bytes()
            ).hexdigest()
        )
        assert (directory / "joint_selection.json").stat().st_mtime_ns < (
            directory / "predictions.jsonl"
        ).stat().st_mtime_ns
        assert (directory / "event_policies.json").stat().st_mtime_ns < (
            directory / "predictions.jsonl"
        ).stat().st_mtime_ns
        with np.load(data / "features" / model / "features.npz") as archive:
            features = {k: archive[k] for k in archive.files}
        grounded = grounded_features(data, features)
        assert all(
            np.isfinite(features[k]).all()
            for k in ["full", "crop_mean", "crop_max", "geometry", "roi_features"]
        )
        assert np.allclose(np.linalg.norm(features["full"], axis=1), 1, atol=1e-4)
        dev = read(directory / "development_predictions.json")
        for task, sidecar in [
            ("smoking", "person_selection"),
            ("unattended_items", "grounding_selection"),
        ]:
            for name, value in read(directory / (sidecar + ".json"))[
                "development_predictions"
            ].items():
                dev[task + "/" + name] = value
        predictions = [
            json.loads(s)
            for s in (directory / "predictions.jsonl").read_text().splitlines()
        ]
        by_head = defaultdict(list)
        for prediction in predictions:
            row = index[prediction["id"]]
            assert all(
                row[k] == prediction[k]
                for k in [
                    "target",
                    "source",
                    "split",
                    "recording_id",
                    "task",
                    "group_id",
                ]
            )
            assert row["split"] in {"test", "external", "transfer", "fresh_test"}
            by_head[row["task"], prediction["candidate"]].append(prediction)
        for task, value in selection["tasks"].items():
            best = max(
                value["candidates"],
                key=lambda c: (
                    c["selection"]["worst_source_ap"],
                    c["selection"]["mean_source_ap"],
                ),
            )
            assert best["name"] == value["primary"]["name"]
            for candidate in value["candidates"]:
                name = candidate["name"]
                d = dev[task + "/" + name]
                assert (
                    select_points(
                        [index[key]["target"] for key in d["ids"]], d["scores"]
                    )
                    == candidate["thresholds"]
                )
                assert all(index[key]["split"] == "dev" for key in d["ids"])
                path = directory / task / (name + ".pt")
                head = torch.load(path, weights_only=True, map_location="cpu")
                audit["checkpoints"][model + "/" + task + "/" + name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                audit["candidate_selections_verified"] += 1
                values = by_head[task, name]
                assert len(values) == len({r["id"] for r in values})
                chosen = []
                for target in [0, 1]:
                    candidates = [r for r in values if r["target"] == target]
                    if candidates:
                        chosen.append(candidates[0])
                fresh = [r for r in values if r["split"] == "fresh_test"]
                if fresh:
                    chosen.append(fresh[0])
                sample_rows = [index[r["id"]] for r in chosen]
                ff = (
                    grounded
                    if head.get("geometry_source") == "grounding_dino"
                    else features
                )
                actual = (
                    score_person(head, instance_arrays(sample_rows, ff))
                    if head["kind"] == "person_mil"
                    else score_checkpoint(
                        head,
                        batch_arrays(sample_rows, ff),
                        "cuda" if args.live else "cpu",
                    )
                )
                delta = float(
                    np.max(np.abs(actual - np.asarray([r["score"] for r in chosen])))
                )
                assert delta < 0.005, (model, task, name, delta)
                audit["saved_head_checks"].append(
                    {
                        "model": model,
                        "task": task,
                        "candidate": name,
                        "cases": len(chosen),
                        "max_score_delta": delta,
                    }
                )
                for source, cohort in results["clip_metrics"][task][name].items():
                    if source == "avss2007_clean_alignment_primary":
                        vv = [
                            r
                            for r in values
                            if r["recording_id"] in fresh_records
                            and not fresh_records[r["recording_id"]][
                                "right_censored_annotation"
                            ]
                        ]
                    elif source == "avss2007_right_censored_diagnostic":
                        vv = [
                            r
                            for r in values
                            if r["recording_id"] in fresh_records
                            and fresh_records[r["recording_id"]][
                                "right_censored_annotation"
                            ]
                        ]
                    else:
                        vv = [
                            r
                            for r in values
                            if r["source"] == source and r["target"] >= 0
                        ]
                    for op, expected in cohort["operating_points"].items():
                        metric = binary_metrics(
                            [r["target"] for r in vv],
                            [r["score"] for r in vv],
                            candidate["thresholds"][op],
                        )
                        assert all(
                            np.isclose(metric[k], expected[k], atol=1e-12, rtol=0)
                            for k in expected
                        ), (model, name, source, op)
                if (
                    task == "unattended_items"
                    and name in policies[task]["event_policies"]
                ):
                    cutoff = policies[task]["event_policies"][name]["threshold"]
                    rr = [index[r["id"]] for r in values]
                    ss = [r["score"] for r in values]
                    for source, records in [
                        ("historical", old_records),
                        ("fresh_avss", fresh_records),
                    ]:
                        replay = summarize_events(rr, ss, records, cutoff)
                        assert replay == results["event_metrics"][task][name][source]
            audit["predictions_replayed"] += sum(
                len(v) for (t, n), v in by_head.items() if t == task
            )
        if args.live:
            # Re-encode actual image bytes and all automatic crops, independently
            # of the cached feature rows used by training/evaluation.
            encoder = Encoder(model, "cuda").eval()
            path_index = {p: i for i, p in enumerate(features["paths"].tolist())}
            cases = [
                next(
                    r
                    for r in rows
                    if r["source"] == "smoking_cctv"
                    and r["split"] == "test"
                    and r["target"] == 1
                ),
                next(
                    r for r in rows if r["split"] == "fresh_test" and r["target"] == 1
                ),
            ]
            for case in cases:
                p = case["views"][-1]
                tasks = [(p, None)] + [(p, b) for b in regions(tracks[p])]
                dataset = Views(tasks, encoder)
                batch = torch.stack([dataset[i] for i in range(len(dataset))]).cuda()
                with torch.inference_mode():
                    encoded = (
                        encoder.backbone.encode_image(batch, normalize=True)
                        if model == "mobileclip2-s2"
                        else torch.nn.functional.normalize(
                            encoder.backbone.get_image_features(pixel_values=batch),
                            dim=-1,
                        )
                    )
                encoded = encoded.cpu().numpy()
                i = path_index[p]
                local = encoded[1:] if len(encoded) > 1 else encoded[:1]
                delta = max(
                    float(np.max(np.abs(encoded[0] - features["full"][i]))),
                    float(np.max(np.abs(local.mean(0) - features["crop_mean"][i]))),
                    float(np.max(np.abs(local.max(0) - features["crop_max"][i]))),
                )
                assert delta < 0.002, (model, p, delta)
                audit["live_feature_checks"].append(
                    {
                        "model": model,
                        "image": p,
                        "automatic_regions": len(tasks) - 1,
                        "max_feature_delta": delta,
                    }
                )
            del encoder
            torch.cuda.empty_cache()
        print("PILOT VERIFIED", model, len(predictions), "predictions", flush=True)
    snapshots = run / "final_source_snapshots"
    sources = [
        ROOT / "anomaly_eval/training/video.py",
        ROOT / "anomaly_eval/training/temporal.py",
        ROOT / "tests/test_video_pilot.py",
        ROOT / "reports/video_pilot_protocol_20261016.md",
    ]
    sources += list((ROOT / "scripts/video").glob("*pilot.py"))
    sources += [
        ROOT / "scripts/video" / name
        for name in [
            "guard_video_sources.py",
            "verify_video_data.py",
            "audit_video_failures.py",
            "audit_smoking_regions.py",
            "audit_video_tracks.py",
        ]
    ]
    hashes = {}
    for source in sources:
        rel = source.relative_to(ROOT)
        destination = snapshots / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    audit["final_source_snapshot_sha256"] = hashes
    audit["source_note"] = (
        "Final reproducible workflow sources; feature/tracker metadata retain their earlier execution hashes. Source additions after extraction preserve default feature behavior, checked by replay."
    )
    (run / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("FULL PILOT VERIFICATION COMPLETE", audit["predictions_replayed"], flush=True)


if __name__ == "__main__":
    main()
