import argparse
import logging
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

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
        default=None,
        help="Local video file path; required for --source file.",
    )
    parser.add_argument(
        "--rtsp-url",
        default="rtsp://localhost:8554/camera_01",
        help="CCTV stream URL; defaults to the local mock camera.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5,
        help="Maximum selected frames per second of video time (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed recording, scheduling, and decoding messages.",
    )
    parser.add_argument(
        "--record-only", action="store_true", help="Record/decode without AI models"
    )
    parser.add_argument(
        "--model",
        choices=["ensemble-20261111", "mobileclip2-s2", "siglip2-b16-224"],
        default="ensemble-20261111",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-id", default="camera_01")
    parser.add_argument("--segment-seconds", type=int, default=2)
    parser.add_argument(
        "--gemma-url",
        default=os.environ.get("GEMMA_BASE_URL"),
        help="OpenAI-compatible base URL ending /v1",
    )
    parser.add_argument(
        "--gemma-model", default=os.environ.get("GEMMA_MODEL", "gemma4-26b-a4b")
    )
    parser.add_argument(
        "--out", type=Path, help="Directory for AI evidence and event journal"
    )
    parser.add_argument(
        "--recordings-dir", type=Path, help="Backend recordings directory"
    )
    args = parser.parse_args()
    if args.source == "file" and args.video is None:
        parser.error("Provide --video for a file source")
    if args.segment_seconds < 1:
        parser.error("segment-seconds must be positive")
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error("--target-fps must be positive and finite")
    return args


def _build_video_source(args: argparse.Namespace) -> VideoSource:
    if args.source == SourceType.FILE:
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        source = VideoSource(
            source_id=args.source_id,
            name="Local sample video",
            source_type=SourceType.FILE,
            location=str(video_path),
        )
    else:
        source = VideoSource(
            source_id=args.source_id,
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
    load_dotenv(project_dir.parent / ".env", override=False)
    args = _parse_args(project_dir)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    database_url = _load_database_url(project_dir)

    source = _build_video_source(args)

    storage_root = args.recordings_dir or project_dir / "recordings"
    engine, session_factory = create_engine_and_session_factory(database_url)

    sys.path.insert(0, str(project_dir.parent))
    with ExitStack() as cleanup:
        cleanup.callback(engine.dispose)
        on_frame = None
        if not args.record_only:
            from vision_runtime.classifier import load_classifier
            from vision_runtime.config import configure_runtime, runtime_root
            from vision_runtime.gemma import GemmaClient
            from vision_runtime.service import ConfirmationQueue

            from data_pipeline.detection.integration import VisionPipeline

            configure_runtime()
            classifier = load_classifier(args.model, args.device)
            confirmation = None
            if args.gemma_url:
                confirmation = ConfirmationQueue(
                    GemmaClient(args.gemma_url, args.gemma_model)
                )
                cleanup.callback(confirmation.close)
            output = (
                args.out or runtime_root() / "cameras" / args.source_id / uuid4().hex
            )
            vision = VisionPipeline(
                source.source_id,
                output,
                classifier,
                confirmation,
                fps=args.target_fps,
                device=args.device,
            )
            cleanup.callback(vision.close)
            on_frame = vision.on_frame
            logging.info("AI event journal | %s", output / "events.jsonl")
        run_pipeline(
            source,
            storage_root,
            session_factory=session_factory,
            target_fps=args.target_fps,
            segment_seconds=args.segment_seconds,
            on_frame=on_frame,
        )


if __name__ == "__main__":
    main()
