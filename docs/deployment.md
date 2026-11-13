# CPU deployment entry point

`deploy.py` handles setup, readiness checks and startup. It uses the existing
model importer; it contains no dataset acquisition, training or evaluation code.
Run it from a complete checkout of `feat/vision-app-integration`.

## Default upload demo

Prerequisites: Python **3.12** with `venv`/`ensurepip`, an **x86_64 Linux**
machine, network access for Python package installation, and the separately
transferred model archive. The pinned wheel configuration also accepts x86_64
Windows, but Linux is the tested platform. macOS/ARM and other Python versions
require a separately validated dependency set. A GPU is not required.

```bash
python3.12 deploy.py setup --bundle /path/to/capstone-mobileclip2-s2-cpu-20261103.tar.gz
python3.12 deploy.py start
```

Open `http://localhost:8501`. Startup runs headlessly without Streamlit's
first-run email prompt. Stop it with Ctrl+C. To change the localhost port:

```bash
python3.12 deploy.py start --port 8502
```

For a built-in/USB webcam, select **Live camera**, select an index (usually 0),
then **Start camera**. The device belongs to the Streamlit host; this is not
browser webcam streaming from a remote client. Local webcam capture uses the
already installed OpenCV and needs no PostgreSQL/Docker/FFmpeg executable.
Stop/Clear controls, bounded latest-frame work, evidence age, temporary rolling
evidence and disconnect shutdown are described in the README. Operating-system
camera permissions/device access must be available. Physical camera hardware
has not been validated on this development host.

Setup performs the following steps:

1. Checks the interpreter/platform, archive checksum and available environment
   storage. A fresh environment needs at least 3 GiB free; allow additional
   space for models, pip cache and runtime media.
2. Creates `.venv`, installs CPU-only Torch/torchvision from the official CPU
   index and installs the direct pinned application requirements. It registers
   this checkout as an editable application.
3. Creates `.env` from `.env.example` if absent. Existing values and comments
   are preserved. Paths containing spaces and external model folders work.
4. Imports only the 15 allowlisted model files, verifying sizes, hashes,
   pretrained revision, completeness and training definitions. Existing model
   storage is verified and preserved; it is never silently replaced.
5. Runs actual CPU classifier inference and checks temporal abstention, YOLO,
   pose, ByteTrack and the fitted proposal checks on small synthetic inputs.
   This is execution validation, not a model accuracy benchmark.
6. If requested, prepares the camera database and verifies its migration head.

The archive checksum is taken from `--sha256`, the adjacent `.sha256` sidecar,
or the tool's pinned checksum for the original 3 November archive, in that order.
For a newly built compatible bundle, supply its trusted checksum explicitly.
Archive integrity does not replace trusting the source of that checksum.

## Reruns and environment recovery

```bash
python3.12 deploy.py setup
python3.12 deploy.py check
python3.12 deploy.py check --smoke
```

Repeat setup reuses dependencies only if both the installation receipt and
current package versions/checkout match. Missing or changed direct dependencies
trigger installation again. Successful dependency installation is recorded
before model/service checks, so a later failure does not require downloading
everything again. A receipt alone cannot bypass environment verification.
Concurrent setups for one checkout are prevented by an OS-released lock.

`check` hashes the complete model file set and checks installed direct package
versions. `check --smoke` additionally loads models and executes the CPU checks.
`start` checks package versions, required files/sizes and configuration, then
starts the app; it does not reinstall, rehash all weights or rerun inference.
Use `check` after transferring/modifying weights.

An existing non-venv directory is rejected rather than cleared. To choose
another environment, use the same path on every command:

```bash
python3.12 deploy.py setup --venv .venv-cpu --bundle /path/to/transfer.tar.gz
python3.12 deploy.py start --venv .venv-cpu
```

Logs/receipts and pip cache live in ignored `.local/deploy/`. Model paths and
runtime paths follow `VISION_MODELS_DIR` and `VISION_RUNTIME_DIR` from the
environment or `.env`. Relative configured paths resolve against the checkout.
Do not rename a mismatched model directory to make verification pass: preserve
it and import a trusted bundle into a new configured location.

System Python, FFmpeg, Docker, and OS libraries are not installed through sudo
or a remote shell script. Install missing prerequisites through the OS package
manager. Linux OpenCV may need the OS's GL and GLib runtime libraries. Python
packages use direct pins; transitive dependencies are resolved by pip and are
not a complete frozen lock or offline wheelhouse.

## Optional camera recording and database

Install FFmpeg on PATH. With Docker Compose available:

```bash
python3.12 deploy.py setup --bundle /path/to/transfer.tar.gz --with-camera
python3.12 deploy.py check --camera
python3.12 deploy.py start --camera --rtsp-url rtsp://camera-address/stream --source-id camera_01
```

For a local recording fixture instead of RTSP:

```bash
python3.12 deploy.py start --camera --video /path/to/input.mp4 --source-id camera_01
```

With no configured connection, setup appends the retained development Compose
URL to `.env`, starts its PostgreSQL service, waits for health and applies the
included migrations. This service binds to localhost and uses the existing
development credentials. It is not production database provisioning. The
installer does not delete/reset volumes or stop an existing database on failure.

For an existing PostgreSQL service, set `DATABASE_URL` in the process environment,
root `.env`, or backend `.env` before `setup --with-camera`. A connection other
than the exact development Compose URL is treated as external: Docker is not
started, but migrations **are applied to that configured database**. Store
credentials in the environment/configuration, not command-line arguments.
Schema readiness checks confirm the Alembic migration head without running AI.

The optional backend-only uv/npm shortcuts remain separate; AI-enabled startup
uses the root environment selected by this deployment tool.

## Optional Gemma endpoint

Serve Gemma independently, then configure its compatible endpoint:

```bash
python3.12 deploy.py setup --gemma-url http://model-server:8080/v1 --gemma-model gemma4-26b-a4b
python3.12 deploy.py check --gemma
python3.12 deploy.py start
```

The tool appends missing Gemma settings. A conflicting existing value causes an
error; edit `.env` explicitly to change it. Use `GEMMA_API_KEY` in configuration
if authentication is required. `check --gemma` checks `/models` and the configured
alias without sending images; it does not establish image/schema compatibility.
Normal setup does not contact the endpoint or download/start Gemma weights.
Actual inference sends selected evidence only when review runs in the app.

CPU MobileCLIP is the default. A working installation does not establish the
two-second inference target or sufficient RAM for a locally served large Gemma
variant. Existing model licenses and measured runtime limitations still apply.

## Transfer bundle and full start (13 November 2026)

`tools/build_deployment.py` assembles one folder with the application at the
current commit, all models (including `ensemble-20261113`), the built
dashboard, Python wheels for an offline install, and optionally Gemma 4 E4B
(4-bit) with a CPU build of the llama.cpp server. Its `START_HERE.md`
(`docs/START_HERE.md` here) is the operator guide.

```bash
python tools/build_deployment.py dist/capstone-cpu-deployment-20261113 \
    --ensemble-export /path/to/bundle_v9 --hf-cache ~/.cache/huggingface/hub \
    --gemma-weights /path/to/gemma-4-E4B-it-qat-q4_0 --llama /path/to/llama-b11925 \
    --wheels --archive
```

What `deploy.py` does with such a folder:

- `setup` installs from `wheels/` without internet access, verifies the
  ensemble folder, and writes `GEMMA_BASE_URL`, `GEMMA_MODEL`, `GEMMA_LOCAL=1`
  and longer Gemma timeouts when `gemma/` is present and its server runs here
  (`--no-gemma` skips this; an existing Gemma setting is never overwritten).
- `start --dashboard --camera --rtsp-url ...` (or `--video ...`) runs the
  bundled Gemma server, the API with the dashboard on port 8000, and the
  camera backend together. Ctrl+C stops the camera backend first so queued
  Gemma reviews can finish, then the rest. The Gemma server's output goes to
  `.local/deploy/logs/gemma_server.log`.
- `start --dashboard` alone serves the API and dashboard; plain `start` is
  still the Streamlit upload page.

Gemma reviews only windows the screening classifier flags, and one behaviour
on one camera at most once per `GEMMA_CONFIRM_INTERVAL_S` (60 s of video), so
a CPU model that needs tens of seconds per answer keeps up.

## Validation on 10 November 2026

A fresh Python 3.12 environment installed successfully with Torch 2.8.0+cpu
and no CUDA build. A separate model copy was imported and exercised by real
classifier/tracker inference. Repeat setup skipped package installation and
preserved the model files. Startup from outside the checkout passed browser
upload/inference/download checks. Camera setup migrated an empty private
PostgreSQL database, and the camera launcher processed a short video through
a terminal screened event. **116 tests**, lint and formatting passed.

The Gemma model-list/auth check was exercised against a local HTTP fixture.
Docker Compose configuration and flags were validated; the camera test used an
external test database, so this run did not launch the Compose container or a
new Gemma model server. Windows remains untested. Raw receipts are kept outside
Git in the original workspace's `.local/deployment_validation_20261110/`.

The subsequent live-camera extension passed **117 tests**, with 8 PostgreSQL
tests skipped because no disposable database was configured. Nine new camera
tests cover device failures, stop/disconnect cleanup, concurrent camera refusal,
timestamps, rolling retention, pending confirmation and latest-evidence work.
A real browser exercised a paced video source through the actual CPU models,
camera-index selection, stop/restart/clear, unavailable-device handling, JSON
export, uploads and mobile layout. A final export check confirmed that a frozen
download remains valid across live fragment refreshes. The development host has
no physical webcam; built-in/USB hardware and OS permissions need destination
validation. These are execution checks, not a new accuracy or latency benchmark.
Receipts are in `.local/live_camera_validation_20261110/` in the original
workspace. The production Streamlit configuration disables source-file watching;
restart the application after changing code/configuration.
