"""Run Stage 1 and the VLM on a video file without recording or the database.

Frames go through the same decoder and 2 fps selection as the live pipeline;
candidates land in candidates/<source_id>/candidates.jsonl.

Usage: python stage1_offline.py VIDEO [--target-fps 5] [--vlm predict-clip]
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from data_pipeline.detection.stage1 import Stage1Detector
from data_pipeline.detection.vlm import CandidateLog, NoVLM, PredictClipVLM
from data_pipeline.video_ingestion.decoding.decoder import decode_segment
from data_pipeline.video_ingestion.decoding.selection import select_frames
from data_pipeline.video_ingestion.scheduling.worker import SegmentJob


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--target-fps", type=float, default=2)
    parser.add_argument("--vlm", choices=["none", "predict-clip"], default="none")
    parser.add_argument("--vlm-python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=project_dir / "candidates")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    video = args.video.resolve()
    source_id = video.stem
    job = SegmentJob(0, source_id, uuid4(), video.name, 0, 0)
    detector = Stage1Detector(source_id, args.out, fps=args.target_fps)
    if args.vlm == "predict-clip":
        vlm = PredictClipVLM(args.vlm_python, project_dir.parent)
    else:
        vlm = NoVLM()
    log = CandidateLog(args.out, vlm)
    count = 0
    with closing(decode_segment(job, video.parent)) as frames:
        for frame in select_frames(frames, target_fps=args.target_fps):
            count += 1
            for candidate in detector.process(frame):
                record = log(candidate)
                logging.info(
                    "t=%6.1fs  %-18s tasks=%s  vlm=%s",
                    record["timestamp_ms"] / 1000,
                    record["event_type"],
                    ",".join(record["tasks"]),
                    {k: v.get("decision") for k, v in record["vlm"].items()},
                )
    logging.info("%d frames, candidates in %s", count, args.out / source_id)


if __name__ == "__main__":
    main()
