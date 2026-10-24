import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

from dotenv import load_dotenv
from sqlalchemy.orm import Session, sessionmaker

from database.models.video_segments import SegmentStatus, VideoSegment
from database.repositories.video_segments import update_video_segment_status
from database.session import create_engine_and_session_factory
from video_ingestion.decoding.decoder import decode_segment
from video_ingestion.recording.worker import run_recording_worker
from video_ingestion.scheduling.worker import SegmentJob, SegmentScheduler
from video_ingestion.sources import SourceType, VideoSource

# TODO: currently for testing only


def _decode_job(
    job: SegmentJob,
    storage_root: Path,
    session_factory: sessionmaker[Session],
) -> int:
    """Decode a claimed job and persist its demo processing result.

    No database transaction stays open during decoding. Failures, including
    Ctrl+C, mark the job failed before propagating. Successful decoding counts
    as processed for this demo; later this should include downstream analysis.
    """
    status = SegmentStatus.FAILED
    frame_count = 0
    try:
        with closing(decode_segment(job, storage_root)) as frames:
            for frame in frames:
                frame_count += 1
                if frame_count == 1:
                    print(
                        f"Decoding segment {job.segment_id}: "
                        f"shape={frame.image.shape}, timestamp_ms={frame.timestamp_ms}",
                        flush=True,
                    )
                # Future frame validation and model processing go here.

        if frame_count == 0:
            raise ValueError(f"Segment {job.segment_id} contains no decoded frames")
        status = SegmentStatus.PROCESSED
    finally:
        # The scheduler already committed the claim; update it in a new transaction.
        with session_factory.begin() as db:
            segment = db.get(VideoSegment, job.segment_id)
            if segment is None:
                raise RuntimeError(f"Segment {job.segment_id} no longer exists")
            update_video_segment_status(db, segment, status=status)

    return frame_count


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
        print("Press Ctrl+C to stop recording, scheduling, and decoding.")
        scheduler = SegmentScheduler(
            source_ids=[source.source_id], session_factory=session_factory
        )
        stop_event = Event()

        # Sessions are created per transaction; the engine pool can serve both threads.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="recorder"
        ) as executor:
            recording = executor.submit(
                run_recording_worker,
                source,
                storage_root,
                segment_seconds=5,
                session_factory=session_factory,
                retention_hours=24,
                stop_event=stop_event,
            )
            try:
                while True:
                    # Snapshot before querying so completion cannot hide a final insert.
                    recording_finished = recording.done()
                    if recording_finished:
                        recording.result()  # Propagate recording failures to the caller.

                    job = scheduler.get_next_job()
                    if job is not None:
                        print(f"Scheduled segment: {job}", flush=True)
                        frame_count = _decode_job(job, storage_root, session_factory)
                        print(
                            f"Decoded segment {job.segment_id}: {frame_count} frames",
                            flush=True,
                        )
                    elif recording_finished:
                        # The recorder has flushed its last segments and no work remains.
                        break
                    else:
                        stop_event.wait(0.25)
            except KeyboardInterrupt:
                print("Stopping recording, scheduling, and decoding...")
            finally:
                stop_event.set()
                # Join before disposing the engine; the recorder may register final rows.
                recording.result()

        print("Recording, scheduling, and decoding finished.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
