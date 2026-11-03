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
npm start -- --source rtsp --rtsp-url "rtsp://camera-address/stream"
```

This starts PostgreSQL and waits until healthy before launching the app.
Migrations must already be applied. Press Ctrl+C to stop recording; PostgreSQL
stays running until you run `npm run db:down`.

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
