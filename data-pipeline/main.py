import argparse
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline.database.session import create_engine_and_session_factory
from data_pipeline.pipeline.runner import run_pipeline
from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame
from data_pipeline.video_ingestion.sources import SourceType, VideoSource


def _parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record, schedule, and decode a local video or CCTV RTSP stream."
    )
    parser.add_argument(
        "--source",
        choices=[SourceType.FILE.value, SourceType.RTSP.value],
        default=SourceType.FILE.value,
        help="Input type: file (default) or rtsp for a CCTV stream.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=project_dir / "samples/videos/loitering.avi",
        help="Video file path; defaults to samples/videos/loitering.avi.",
    )
    parser.add_argument(
        "--rtsp-url",
        default="rtsp://localhost:8554/camera_01",
        help="CCTV stream URL; defaults to the local mock camera.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Maximum selected media FPS (default: 5 with Stage 1; otherwise 2).",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Run the Stage 1 YOLO pre-filter on selected frames (needs ultralytics).",
    )
    from data_pipeline.detection.reasoning import add_arguments

    add_arguments(parser, project_dir.parent)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed recording, scheduling, and decoding messages.",
    )
    parser.add_argument(
        "--candidate-dir", type=Path, default=project_dir / "candidates"
    )
    parser.add_argument(
        "--recordings-dir", type=Path, default=project_dir / "recordings"
    )
    parser.add_argument("--segment-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.segment_seconds < 1:
        parser.error("segment-seconds must be positive")
    if args.target_fps is None:
        args.target_fps = 5.0 if args.stage1 else 2.0
    if args.vlm != "none" and not args.stage1:
        parser.error("--vlm requires --stage1")
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error("--target-fps must be positive and finite")
    return args


def _build_video_source(args: argparse.Namespace) -> VideoSource:
    if args.source == SourceType.FILE:
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        source = VideoSource(
            source_id="test_video",
            name="Local sample video",
            source_type=SourceType.FILE,
            location=str(video_path),
        )
    else:
        source = VideoSource(
            source_id="camera_01",
            name="CCTV RTSP stream",
            source_type=SourceType.RTSP,
            location=args.rtsp_url,
        )
    return source


def _build_stage1_hook(
    args: argparse.Namespace, source: VideoSource, out_dir: Path
) -> Callable[[DecodedFrame], None]:
    # Imported here so the pipeline runs without the YOLO dependencies.
    from data_pipeline.detection.reasoning import build_service
    from data_pipeline.detection.stage1 import Stage1Detector, default_config, load_gate

    gate = load_gate()
    detector = Stage1Detector(
        source.source_id,
        out_dir,
        fps=args.target_fps,
        cfg=default_config(gate, args.stage1_device),
        gate=gate,
        safety_scan_seconds=args.safety_scan_seconds,
    )
    service = build_service(args, Path(__file__).resolve().parents[1], out_dir)
    service.start()

    def on_frame(frame: DecodedFrame) -> None:
        for candidate in detector.process(frame):
            service.submit(candidate)

    on_frame.close = service.close
    return on_frame


def _load_database_url(project_dir: Path) -> str:
    load_dotenv(project_dir / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL in data-pipeline/.env or your environment")
    return database_url


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    args = _parse_args(project_dir)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    database_url = _load_database_url(project_dir)

    source = _build_video_source(args)

    storage_root = args.recordings_dir
    on_frame = None
    engine, session_factory = create_engine_and_session_factory(database_url)
    try:
        on_frame = (
            _build_stage1_hook(args, source, args.candidate_dir)
            if args.stage1
            else None
        )
        run_pipeline(
            source,
            storage_root,
            session_factory=session_factory,
            target_fps=args.target_fps,
            segment_seconds=args.segment_seconds,
            on_frame=on_frame,
        )
    finally:
        try:
            if on_frame is not None:
                on_frame.close()
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
