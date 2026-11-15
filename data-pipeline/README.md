# Data pipeline

A Python project for the YOLO-VLM anomaly detection pipeline.

## First-time setup

Install [Node.js](https://nodejs.org/) with npm. Run project commands from
`data-pipeline/`. npm provides shortcuts; uv manages Python packages.
No `npm install` is needed.

### 1. Install uv

uv installs the Python version and packages this project needs.

**Linux/macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal, then run `uv --version` to check the installation.

### 2. Install Python and dependencies

```bash
npm run setup
```

This installs Python 3.12 and locked dependencies, and creates `.env` from
`.env.example` if needed. Existing settings are preserved. Run setup again
after pulling dependency changes; manual environment activation is not needed.

### 3. Install FFmpeg

Video ingestion uses `ffmpeg` to record video segments. The executable must be available on `PATH`.

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (Homebrew):**

Install [Homebrew](https://brew.sh/) if needed, then run:

```bash
brew install ffmpeg
```

**Windows (PowerShell):**

```powershell
winget install --id Gyan.FFmpeg --exact --source winget
```

Verify that FFmpeg is available:

```powershell
ffmpeg -version
```

### 4. Install Docker

Docker with Docker Compose is required for the local development database. Install [Docker Desktop](https://docs.docker.com/get-started/get-docker/) for Windows, macOS, or a supported Linux distribution.
Start Docker Desktop before running the database commands.

Verify that Docker is running and Compose is installed:

```bash
docker info
docker compose version
```

See [Setup local development database](#setup-local-development-database) for startup and connection instructions.

## Environment configuration

`npm run setup` creates `.env` if needed. Review its settings before running
migrations or the app.

| Variable            | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `DATABASE_URL`      | PostgreSQL connection used by the app and migrations. |
| `TEST_DATABASE_URL` | PostgreSQL connection used by database tests.         |

## Setup local development database

Run these commands from the project root (`data-pipeline/`).
Start the PostgreSQL 18 development container in the background:

```bash
npm run db:up
```

This waits until PostgreSQL is healthy before returning.

To open an interactive SQL shell inside the container:

```bash
npm run db:shell
```

To stop and remove the container while keeping data:

```bash
npm run db:down
```

To reset the database and **delete all data in its volume**:

```bash
npm run db:down:force
```

After resetting, run `npm run db:up` and `npm run db:migrate` to recreate the schema.

## Database migrations

Run these commands from the project root with the local PostgreSQL container running. The development connection is configured in `alembic.ini`.

Apply all migrations after initial setup or pulling or creating new migration files:

```bash
npm run db:migrate
```

## Run the app

Run the sample video at `samples/videos/loitering.avi`:

```bash
npm start
```

To use another local video:

```bash
npm start -- --video "path/to/video.mp4"
```

To use the mock CCTV stream, first start the server and publisher as described
in [Mock RTSP camera](#mock-rtsp-camera), then run in another terminal:

```bash
npm start -- --source rtsp
```

For a real CCTV camera, provide its stream URL:

```bash
npm start -- --source rtsp --rtsp-url "rtsp://camera-address1/stream" --rtsp-url "rtsp://camera-address2/stream" ..
```

Pass arguments after `npm start --`:

| Argument       | Options / value         | Default                           | Description                                                                                            |
| -------------- | ----------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--source`     | `file` or `rtsp`        | `file`                            | Choose a local video or CCTV stream.                                                                   |
| `--video`      | File path               | `samples/videos/loitering.avi`    | Video to read when `--source file`.                                                                    |
| `--rtsp-url`   | RTSP URL, repeatable    | `rtsp://localhost:8554/camera_01` | Repeat for up to four cameras when `--source rtsp`; order sets camera IDs and initial selection order. |
| `--target-fps` | Positive, finite number | `2`                               | Maximum frames selected per second of video time for each camera/file.                                 |
| `--queue-size` | Positive integer        | `8`                               | Maximum waiting frames per camera/file; drops the oldest waiting frame when full.                      |
| `--verbose`    | Flag                    | Off                               | Show detailed recording and streaming messages.                                                        |
| `-h`, `--help` | Flag                    | —                                 | Show available arguments and exit.                                                                     |

## Mock RTSP camera

With Docker running, start the MediaMTX server in one terminal:

```bash
npm run camera:server
```

Wait for the server to report that its RTSP listener is open, then publish the
test video from a second terminal:

```bash
npm run camera:stream
```

The sample video at `samples/videos/loitering.avi` loops continuously at
`rtsp://localhost:8554/camera_01` over TCP, encoded as H.264 for streaming.
Preview the live stream in your browser at
[http://localhost:8888/camera_01/](http://localhost:8888/camera_01/)
while the server and publisher are running.

To stop the mock CCTV Docker server:

```bash
npm run camera:stop
```

## Memory benchmark

The dated reports currently under `reports/` describe the older setup with one
consumer per camera. Keep them as historical results; the updated shared-consumer
pipeline needs a fresh suite run before publishing new measurements.

With the database running and migrations applied, measure one source for 30 seconds:

```bash
npm run benchmark:memory
```

The benchmark samples Python and its FFmpeg child every 0.1 seconds and saves
`summary.json` and `memory.csv` under `recordings/benchmarks/`. It records real
video segments and database metadata under a separate benchmark source ID.
Memory is reported in MiB. PostgreSQL, Docker, and YOLO are excluded; the consumer
only counts frames. Summed RSS can count shared memory pages more than once.

The benchmark accepts `--source`, `--video`, `--rtsp-url`, `--target-fps`, and `--queue-size` as
above, plus `--duration` (seconds, default 30) and `--consumer-delay` (seconds per
frame, default 0). For example, simulate a consumer slower than the 2 FPS input:

```bash
npm run benchmark:memory -- --duration 30 --consumer-delay 0.75
```

Input stops at the next callback after the duration limit, then the queue drains.
The full run can therefore take longer than the requested duration. Compare
runs using the same video resolution, FPS, and duration.

For component measurements, 1/2/4 camera scaling, and whole-machine RAM including
PostgreSQL and Docker, start the mock RTSP server and database, then run:

```bash
npm run benchmark:memory:suite
```

This runs 20 fresh-process scenarios: three repeats each of idle, recording-only,
streaming-only (including decoding), and recording + streaming at 1/2/4 cameras;
then a 10-minute four-camera stability run and a four-camera stress run with
0.75 seconds of processing per frame in the single shared consumer. Normal runs use a 10-second
warm-up plus 60-second measurement. Allow about 35 minutes. Each camera selects
2 FPS and has eight waiting frames, dropping the oldest on overflow.

Results include `report.md`, `report_data.json`, `aggregate.csv`, and raw
process/host/container samples under `recordings/benchmarks/`. Application RAM
uses RSS (resident process RAM) for Python plus recording FFmpeg. Whole-machine
RAM is measured separately as total minus available RAM, in MiB/GiB. Component
estimates subtract a shared Python baseline; they are comparative estimates
because threads share memory. No decoding-only scenario or USS sampling is used.
It leaves existing Docker services running and records real video and metadata.
One shared counting consumer reads camera queues in round-robin order; it does
not load YOLO or VLM models. Each simulated camera
opens independent connections to the same RTSP stream. By default, the suite
starts and stops its own publisher on a separate path, normalizing timestamps
and using one-second H.264 keyframes. Existing camera streams are left untouched.

Suite options: `--warmup`, `--duration`, `--repeats`, `--queue-size` (default 8), `--soak-duration` (0 to skip),
`--skip-stress`, `--output`, and `--rtsp-url` (repeat for distinct camera URLs;
otherwise the dedicated benchmark feed is used). Docker and WSL figures overlap
host RAM and must not be added to whole-machine measurements. Production
accounting excludes MediaMTX and mock publishers while retaining PostgreSQL,
recording FFmpeg, and Docker overhead. Raw host totals still include the test
infrastructure; exact host RAM without MediaMTX has not been measured.

The suite checks queue bounds, complete draining, produced = consumed + dropped,
and recording process counts. Failed runs are excluded from estimates and their
logs are retained. The sample workload is 640 x 360 without YOLO/VLM; these
measurements do not establish complete production hardware requirements.

To export a suite run to PDF with charts (reporting packages are temporary):

```bash
uv run --locked --with reportlab --with matplotlib python -m benchmarks.memory_pdf "<run-directory>" --output reports/memory-benchmark.pdf
```

The exporter regenerates the Markdown, CSV, and JSON summaries from the raw
measurements and uses the recorded run date for the PDF. Published reports are
named `reports/memory-benchmark-YYYY-MM-DD` with `.pdf`, `.md`, `.csv`, and `.json`
extensions.

## Linting and formatting

Linting and formatting are executed when save file is executed. To make changes from the terminal:

```bash
npm run lint
npm run format
```

## Testing

Start the local database and create `.env` as above to run PostgreSQL tests:

```bash
npm run db:up
npm test -- tests/data_pipeline/database -q
```

Without a test URL, database integration tests skip locally. GitHub Actions
starts a disposable PostgreSQL service and sets its database URLs directly in
the workflow; CI does not need a `.env` file. Missing `TEST_DATABASE_URL` fails
database tests in CI instead of silently skipping them.

Run the test suite with:

```bash
npm test
```

To run a single test file:

```bash
npm test -- tests/data_pipeline/video_ingestion/recording/worker/test_prepare_session.py
```

To see which parts of the pipeline were exercised by tests:

```bash
npm test -- --cov=data_pipeline --cov-report=term-missing
```

Tests live in `tests/`, with filenames starting with `test_`.
