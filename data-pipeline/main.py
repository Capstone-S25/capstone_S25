import argparse
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline.database.session import create_engine_and_session_factory
from data_pipeline.pipeline.processing import process_frame
from data_pipeline.pipeline.runner import MAX_CAMERA_COUNT, run_pipeline
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.frame_queue import DEFAULT_FRAME_QUEUE_SIZE

DEFAULT_RTSP_URL = "rtsp://localhost:8554/camera_01"


def _parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record up to four cameras and share one round-robin frame consumer."
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
        action="append",
        help="CCTV stream URL; repeat for up to four cameras. Defaults to the local mock camera.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=2,
        help="Maximum selected frames per second per camera/file (default: 2).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_FRAME_QUEUE_SIZE,
        help=f"Maximum waiting frames per camera/file; drops oldest when full (default: {DEFAULT_FRAME_QUEUE_SIZE}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed recording and streaming messages.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error("--target-fps must be positive and finite")
    if args.queue_size <= 0:
        parser.error("--queue-size must be a positive integer")
    if args.rtsp_url and len(args.rtsp_url) > MAX_CAMERA_COUNT:
        parser.error(f"--rtsp-url supports at most {MAX_CAMERA_COUNT} cameras")
    if args.source != SourceType.RTSP and args.rtsp_url and len(args.rtsp_url) > 1:
        parser.error("Multiple --rtsp-url arguments require --source rtsp")
    return args


def _build_video_sources(args: argparse.Namespace) -> list[VideoSource]:
    if args.source == SourceType.FILE:
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return [
            VideoSource(
                source_id="test_video",
                name="Local sample video",
                source_type=SourceType.FILE,
                location=str(video_path),
            )
        ]
    return [
        VideoSource(
            source_id=f"camera_{index:02d}",
            name=f"CCTV RTSP stream {index}",
            source_type=SourceType.RTSP,
            location=url,
        )
        for index, url in enumerate(args.rtsp_url or [DEFAULT_RTSP_URL], start=1)
    ]


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

    sources = _build_video_sources(args)

    storage_root = project_dir / "recordings"
    engine, session_factory = create_engine_and_session_factory(database_url)

    try:
        run_pipeline(
            sources,
            storage_root,
            session_factory=session_factory,
            frame_handler=process_frame,
            target_fps=args.target_fps,
            queue_size=args.queue_size,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
