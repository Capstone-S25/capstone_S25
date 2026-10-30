import argparse
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline.database.session import create_engine_and_session_factory
from data_pipeline.pipeline.runner import run_pipeline
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
        default=2,
        help="Maximum selected frames per second of video time (default: 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed recording, scheduling, and decoding messages.",
    )
    args = parser.parse_args()
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

    storage_root = project_dir / "recordings"
    engine, session_factory = create_engine_and_session_factory(database_url)

    try:
        run_pipeline(
            source,
            storage_root,
            session_factory=session_factory,
            target_fps=args.target_fps,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
