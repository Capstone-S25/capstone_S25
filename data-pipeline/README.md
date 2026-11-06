# Camera data pipeline

The backend records files or RTSP streams with FFmpeg, registers finalized
segments in PostgreSQL, claims jobs, decodes selected frames, and calls the
vision runtime. Use the shared Python environment from the
[application setup](../README.md) for AI-enabled runs.

## Database and execution

From the repository root, with its virtual environment activated:

```bash
cp data-pipeline/.env.example data-pipeline/.env
docker compose -f data-pipeline/infra/compose.yaml up -d --wait postgres
cd data-pipeline
python -m alembic upgrade head
python main.py --video /path/to/input.mp4 --source-id camera_01
```

For RTSP, replace `--video` with
`--source rtsp --rtsp-url rtsp://camera-address/stream`.
The backend loads environment variables first, root `.env` next, and backend
`.env` last without overwriting existing values. Root model/runtime paths remain
relative to the repository even when invoked here.

`--record-only` runs ingestion, scheduling and decoding without the AI models.
`--recordings-dir` selects recording storage. `--out` selects the AI evidence
directory, whose `events.jsonl` reports screening/confirmation transitions.
Use a stable, unique `--source-id` for each camera and only one active tracker
process per camera. Scheduler support for multiple source IDs does not provide
distributed ownership of a camera's tracker.

Press Ctrl+C for shutdown. The recorder finalizes its current output, tracking
flushes pending crop checks and a final scene review, and AI queues drain.
An active Gemma request can therefore delay graceful shutdown until its timeout.

## Queue semantics

`video_segments` is the existing persistent job queue:

```text
finalized recording → pending → processing → processed / failed
```

Registration is idempotent by relative path. Claims use PostgreSQL
`FOR UPDATE SKIP LOCKED`; jobs are oldest first within a source, and the
scheduler cycles across source IDs. `processed` means decoding and frame
callbacks completed. AI event screening and Gemma confirmation may still be
running. Their separate journal states are documented in
[the integration audit](../docs/integration.md).

Recording metadata has a retention deadline, but no automatic video-file
deletion worker is provided. Pending segment metadata survives a restart.
Jobs left `processing` after a crash have no automatic lease expiry/requeue.
Recovery and retry policy require additional operational work.

## Backend-only development

The retained `uv.lock` covers the lightweight backend and its tests; it does
not install Torch, detectors or the full application. From this directory:

```bash
uv sync --locked --dev
uv run --locked python main.py --record-only --video /path/to/input.mp4
uv run --locked pytest --cov=data_pipeline --cov-report=term-missing
```

Set `TEST_DATABASE_URL` to a disposable PostgreSQL database to include database
tests. Without it they skip locally and fail in CI. GitHub Actions supplies its
own database service. The root environment can run the same tests with
`pytest -q` from the repository root.

Optional npm shortcuts start the database, run uv, and launch a mock MediaMTX
camera. `npm start -- --record-only --video /path/to/input.mp4` uses the
backend-only uv environment. AI-enabled execution uses the root environment
and `python main.py` instead. To publish a mock video, set `SAMPLE_VIDEO` to
your own local file before `npm run camera:stream`; no sample video is tracked.

Stop the database while preserving its volume:

```bash
docker compose -f infra/compose.yaml down
```
