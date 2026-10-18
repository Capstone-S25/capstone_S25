import os
from pathlib import Path

from dotenv import load_dotenv

from database.session import create_engine_and_session_factory
from video_ingestion.recording.worker import run_recording_worker
from video_ingestion.sources import SourceType, VideoSource

# TODO: currently for testing only


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env", override=False)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL in data-pipeline/.env or your environment")

    video_path = Path(r"C:\Users\ongxu\Videos\test\2023-03-22 09-32-18.mp4")
    if not video_path.is_file():
        raise FileNotFoundError(f"Test video not found: {video_path}")

    source = VideoSource(
        source_id="test_video",
        name="Local test video",
        source_type=SourceType.FILE,
        location=str(video_path),
    )
    storage_root = project_dir / "recordings"
    engine, session_factory = create_engine_and_session_factory(database_url)

    try:
        print(f"Recording: {video_path}")
        print(f"Output directory: {storage_root / source.source_id}")
        print("Press Ctrl+C to stop recording gracefully.")
        run_recording_worker(
            source,
            storage_root,
            segment_seconds=5,
            session_factory=session_factory,
            retention_hours=24,
        )
        print("Recording worker finished.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
