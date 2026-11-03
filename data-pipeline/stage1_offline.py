"""Run Stage 1 and the VLM on a video file without recording or the database.

Frames go through the same decoder and 5 fps selection as the live pipeline;
candidates land in candidates/<source_id>/candidates.jsonl.

Usage: python stage1_offline.py VIDEO --target-fps 5 --vlm cascade
"""

import argparse
import hashlib
import json
import logging
import math
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from data_pipeline.detection.reasoning import add_arguments, build_service
from data_pipeline.detection.stage1 import Stage1Detector, default_config, load_gate
from data_pipeline.video_ingestion.decoding.decoder import decode_segment
from data_pipeline.video_ingestion.decoding.selection import select_frames
from data_pipeline.video_ingestion.scheduling.worker import SegmentJob


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument(
        "--replay-candidates",
        type=Path,
        help="Replay existing evidence through resident models without detection",
    )
    parser.add_argument("--target-fps", type=float, default=5)
    add_arguments(parser, project_dir.parent)
    parser.add_argument("--out", type=Path, default=project_dir / "candidates")
    args = parser.parse_args()
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error("target-fps must be positive and finite")
    if bool(args.video) == bool(args.replay_candidates):
        parser.error("Provide one video or --replay-candidates")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    source = (args.video or args.replay_candidates).resolve(strict=True)
    sources = [
        project_dir / "data_pipeline/detection" / name
        for name in ["reasoning.py", "stage1.py"]
    ]
    sources += list((project_dir.parent / "anomaly_eval/reasoning").glob("*.py"))
    sources += [
        project_dir.parent / "notebooks/stage1_gate.py",
        project_dir.parent / "notebooks/stage1_pose_action.py",
        Path(__file__),
    ]
    manifest = {
        "mode": "candidate_replay" if args.replay_candidates else "video_integration",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "code_sha256": {
            str(p.relative_to(project_dir.parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    service = build_service(args, project_dir.parent, args.out)
    ready = time.perf_counter()
    service.start()
    ready_seconds = time.perf_counter() - ready
    count = 0
    started = time.perf_counter()
    try:
        if args.replay_candidates:
            for line in source.read_text().splitlines():
                candidate = json.loads(line)
                if candidate.get("status") in {"queued", "rejected"}:
                    continue
                for key in [
                    "status",
                    "vlm",
                    "inference",
                    "queue_seconds",
                    "service_seconds",
                    "response_seconds",
                ]:
                    candidate.pop(key, None)
                service.submit(candidate)
                count += 1
        else:
            source_id = source.stem
            job = SegmentJob(0, source_id, uuid4(), source.name, 0, 0)
            gate = load_gate()
            detector = Stage1Detector(
                source_id,
                args.out,
                fps=args.target_fps,
                cfg=default_config(gate, args.stage1_device),
                gate=gate,
                safety_scan_seconds=args.safety_scan_seconds,
            )
            with closing(decode_segment(job, source.parent)) as frames:
                for frame in select_frames(frames, target_fps=args.target_fps):
                    count += 1
                    for candidate in detector.process(frame):
                        admitted = service.submit(candidate)
                        logging.info(
                            "t=%6.1fs  %-18s tasks=%s admitted=%s",
                            candidate["timestamp_s"],
                            candidate["event_type"],
                            ",".join(candidate["tasks"]),
                            admitted,
                        )
    finally:
        service.close()
    elapsed = time.perf_counter() - started
    (args.out / "run_timing.json").write_text(
        json.dumps(
            {
                "mode": manifest["mode"],
                "model_ready_seconds": ready_seconds,
                "processing_and_drain_seconds": elapsed,
                "selected_frames_or_replayed_candidates": count,
            },
            indent=2,
        )
        + "\n"
    )
    logging.info("Total processing + drain %.3f seconds", elapsed)
    logging.info("%d inputs, candidates in %s", count, args.out)


if __name__ == "__main__":
    main()
