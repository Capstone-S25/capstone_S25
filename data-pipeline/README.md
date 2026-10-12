# Data pipeline

A Python project for the YOLO-VLM anomaly detection pipeline.

## First-time setup

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
cd data-pipeline
uv python install 3.12
uv sync --locked
```

Use `uv run` before Python commands to use this environment; no manual activation is needed. Run `uv sync --locked` again after pulling changes that update dependencies.

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

See [Local development database](#local-development-database) for startup and connection instructions.

## Local development database

Run these commands from the project root (`data-pipeline/`).
Start the PostgreSQL 18 development container in the background:

```bash
docker compose -f infra/compose.yaml up -d postgres
```

Check the container status and whether PostgreSQL is accepting connections:

```bash
docker compose -f infra/compose.yaml exec postgres pg_isready -U capstone -d capstone
```

To inspect startup errors:

```bash
docker compose -f infra/compose.yaml logs postgres
```

To open an interactive SQL shell inside the container:

```bash
docker compose -f infra/compose.yaml exec postgres psql -U capstone -d capstone
```

To stop and remove the container:

```bash
docker compose -f infra/compose.yaml down
# To delete database volume:
docker compose -v -f infra/compose.yaml down
```

## Add dependencies

To add a production package:

```bash
uv add <package-name>
```

To add a development package:

```bash
uv add --dev <package-name>
```

Commit both `pyproject.toml` and `uv.lock` after changing dependencies.

## Linting and formatting

Linting and formatting are executed when save file is executed. To make changes from the terminal:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

To run a single test file:

```bash
uv run pytest tests/video_ingestion/recording_worker/test_prepare_session.py
```

To see which parts of the pipeline were exercised by tests:

```bash
uv run pytest --cov=video_ingestion --cov-report=term-missing
```

Tests live in `tests/`, with filenames starting with `test_`.
