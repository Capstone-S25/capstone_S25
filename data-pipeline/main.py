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
from data_pipeline.pipeline.processing import process_frame
from data_pipeline.pipeline.runner import MAX_CAMERA_COUNT, run_pipeline
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.frame_queue import DEFAULT_FRAME_QUEUE_SIZE

DEFAULT_RTSP_URL = "rtsp://localhost:8554/camera_01"


def _parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record up to four cameras (or one video file), share one "
        "round-robin frame consumer, and screen each camera's frames."
    )
    parser.add_argument(
        "--source",
        choices=[SourceType.FILE.value, SourceType.RTSP.value],
        default=SourceType.FILE.value,
        help="Input type: file (default) or rtsp for CCTV streams.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Local video file path; required for --source file.",
    )
    parser.add_argument(
        "--rtsp-url",
        action="append",
        help="CCTV stream URL; repeat for up to four cameras. Defaults to the local mock camera.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5,
        help="Maximum selected frames per second per camera/file (default: 5).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_FRAME_QUEUE_SIZE,
        help="Maximum waiting frames per camera/file; drops the oldest when full "
        f"(default: {DEFAULT_FRAME_QUEUE_SIZE}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed recording and streaming messages.",
    )
    parser.add_argument(
        "--record-only", action="store_true", help="Record and stream without AI models"
    )
    parser.add_argument(
        "--model",
        choices=[
            "ensemble-20261116",
            "ensemble-20261114",
            "ensemble-20261113",
            "ensemble-20261111",
            "mobileclip2-s2",
            "siglip2-b16-224",
        ],
        default=None,
        help="Screening classifier; default: the newest installed ensemble",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--source-id",
        default="camera_01",
        help="Camera ID; with several --rtsp-url, IDs are this plus _1, _2, ...",
    )
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
    parser.add_argument(
        "--no-incidents",
        action="store_true",
        help="Journal AI events only; do not write incidents to the database",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Instead of the screening classifier, send Stage 1 candidates to the "
        "--vlm candidate service (data_pipeline.detection.reasoning).",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=project_dir / "candidates",
        help="Candidate journals for --stage1",
    )
    from data_pipeline.detection.reasoning import add_arguments

    add_arguments(parser, project_dir.parent)
    args = parser.parse_args()
    if args.vlm != "none" and not args.stage1:
        parser.error("--vlm requires --stage1")
    if args.stage1 and args.record_only:
        parser.error("Choose --stage1 or --record-only")
    if args.source == SourceType.FILE and args.video is None:
        parser.error("Provide --video for a file source")
    if args.segment_seconds < 1:
        parser.error("segment-seconds must be positive")
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error("--target-fps must be positive and finite")
    if args.queue_size <= 0:
        parser.error("--queue-size must be a positive integer")
    if args.rtsp_url and len(args.rtsp_url) > MAX_CAMERA_COUNT:
        parser.error(f"--rtsp-url supports at most {MAX_CAMERA_COUNT} cameras")
    if args.source != SourceType.RTSP and args.rtsp_url:
        parser.error("--rtsp-url requires --source rtsp")
    return args


def _build_video_sources(args: argparse.Namespace) -> list[VideoSource]:
    if args.source == SourceType.FILE:
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return [
            VideoSource(
                source_id=args.source_id,
                name="Local video",
                source_type=SourceType.FILE,
                location=str(video_path),
            )
        ]
    urls = args.rtsp_url or [DEFAULT_RTSP_URL]
    return [
        VideoSource(
            source_id=args.source_id if len(urls) == 1 else f"{args.source_id}_{index}",
            name=f"CCTV RTSP stream {index}",
            source_type=SourceType.RTSP,
            location=url,
        )
        for index, url in enumerate(urls, start=1)
    ]


def _load_database_url(project_dir: Path) -> str:
    load_dotenv(project_dir / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL in data-pipeline/.env or your environment")
    return database_url


def _build_stage1_hook(args, sources, out_dir):
    """Stage 1 per camera, all candidates into one service from build_service().

    The returned frame handler has a close() that flushes the service.
    """
    # Imported here so the pipeline runs without the YOLO dependencies.
    from data_pipeline.detection.reasoning import build_service
    from data_pipeline.detection.stage1 import (
        Stage1Detector,
        Stage1Frame,
        default_config,
        load_gate,
    )

    gate = load_gate()
    detectors = {
        source.source_id: Stage1Detector(
            source.source_id,
            out_dir,
            fps=args.target_fps,
            cfg=default_config(gate, args.stage1_device, args.stage1_imgsz),
            gate=gate,
            safety_scan_seconds=args.safety_scan_seconds,
        )
        for source in sources
    }
    sessions = {source_id: uuid4() for source_id in detectors}
    service = build_service(args, Path(__file__).resolve().parents[1], out_dir)
    service.start()

    def on_frame(frame) -> None:
        frame = Stage1Frame(
            None,
            frame.source_id,
            sessions[frame.source_id],
            frame.frame_index,
            frame.timestamp_ms,
            frame.image,
        )
        for candidate in detectors[frame.source_id].process(frame):
            service.submit(candidate)

    def close() -> None:
        try:
            service.close()
        finally:
            for detector in detectors.values():
                detector._close_pool()

    on_frame.close = close
    return on_frame


def _build_vision(args, sources, session_factory, storage_root, cleanup):
    """One VisionPipeline per camera behind a single frame handler."""
    from vision_runtime.classifier import load_classifier
    from vision_runtime.config import configure_runtime, runtime_root
    from vision_runtime.gemma import GemmaClient
    from vision_runtime.service import ConfirmationQueue

    from data_pipeline.detection.integration import (
        CameraRouter,
        SerializedClassifier,
        VisionPipeline,
    )

    configure_runtime()
    classifier = load_classifier(args.model, args.device)
    if len(sources) > 1:
        classifier = SerializedClassifier(classifier)
    confirmation = None
    if args.gemma_url:
        confirmation = ConfirmationQueue(GemmaClient(args.gemma_url, args.gemma_model))
        cleanup.callback(confirmation.close)
    recorder = None
    if not args.no_incidents:
        from data_pipeline.detection.incidents import IncidentRecorder

        recorder = IncidentRecorder(session_factory, storage_root)
    run = uuid4().hex
    pipelines = {}
    for source in sources:
        if args.out and len(sources) == 1:
            output = args.out
        elif args.out:
            output = args.out / source.source_id
        else:
            output = runtime_root() / "cameras" / source.source_id / run
        pipelines[source.source_id] = VisionPipeline(
            source.source_id,
            output,
            classifier,
            confirmation,
            fps=args.target_fps,
            device=args.device,
            on_event=recorder,
        )
        logging.info(
            "AI event journal | %s | %s", source.source_id, output / "events.jsonl"
        )
    router = CameraRouter(pipelines)
    cleanup.callback(router.close)
    return router


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

    sources = _build_video_sources(args)

    storage_root = args.recordings_dir or project_dir / "recordings"
    engine, session_factory = create_engine_and_session_factory(database_url)

    sys.path.insert(0, str(project_dir.parent))
    with ExitStack() as cleanup:
        cleanup.callback(engine.dispose)
        handler = process_frame
        if args.stage1:
            handler = _build_stage1_hook(args, sources, args.candidate_dir)
            cleanup.callback(handler.close)
        elif not args.record_only:
            handler = _build_vision(
                args, sources, session_factory, storage_root, cleanup
            )
        run_pipeline(
            sources,
            storage_root,
            session_factory=session_factory,
            frame_handler=handler,
            target_fps=args.target_fps,
            queue_size=args.queue_size,
            segment_seconds=args.segment_seconds,
        )


if __name__ == "__main__":
    main()
