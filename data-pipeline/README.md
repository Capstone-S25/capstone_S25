# Camera data pipeline

The backend records a file or up to four RTSP streams with FFmpeg and
registers finalized segments in PostgreSQL. Alongside the recorder, a
streaming worker per camera decodes the input live and queues selected
frames; one consumer takes them round-robin across cameras and calls the
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
`--source rtsp --rtsp-url rtsp://camera-address/stream`; repeat `--rtsp-url`
for up to four cameras (IDs `camera_01_1`, `camera_01_2`, ... from
`--source-id`). Each camera gets its own tracker and event journal; the
screening classifier and Gemma are shared.
The backend loads environment variables first, root `.env` next, and backend
`.env` last without overwriting existing values. Root model/runtime paths remain
relative to the repository even when invoked here.

`--record-only` records and streams without the AI models (frames are only
logged). `--target-fps` (default 5) caps the frames selected per second per
camera, and `--queue-size` (default 8) the frames waiting per camera: when the
AI falls behind, the oldest waiting frame is dropped and the queue reports how
many were dropped at shutdown. Files are read at their own playback speed, like
a live camera.
`--recordings-dir` selects recording storage. `--out` selects the AI evidence
directory, whose `events.jsonl` reports screening/confirmation transitions.
Use a stable, unique `--source-id` for each camera and only one active tracker
process per camera.

Press Ctrl+C for shutdown. The recorder finalizes its current output, tracking
flushes pending crop checks and a final scene review, and AI queues drain.
An active Gemma request can therefore delay graceful shutdown until its timeout.

## Recording and frames

Recording and frame selection are independent: segments are registered in
`video_segments` for retention and evidence, while the AI sees frames from the
streaming worker as they are decoded. `video_segments` is no longer a job
queue for the AI. AI event screening and Gemma confirmation run after the
frame handler returns; their journal states are documented in
[the integration audit](../docs/integration.md).

Recording metadata has a retention deadline, but no automatic video-file
deletion worker is provided.

## Run the API

The dashboard API needs the database running and migrations applied. To
fill the database with 30 fake incidents for development:

```bash
npm run db:seed
```

Running it again replaces the previous seed data and leaves other incidents
alone. Start the API with:

```bash
npm run api
```

Browse and try the endpoints at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Login is temporary:
every request runs as user 1, with the role from `DEV_USER_ROLE`.

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
