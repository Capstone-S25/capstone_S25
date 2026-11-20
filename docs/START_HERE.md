# Transfer this entire folder

This is the CPU deployment of `feat/vision-app-integration`. It contains the
application, every model file, the built dashboard, Python packages for an
offline install, and Gemma 4 E4B with a CPU server. `DEPLOYMENT_MANIFEST.json`
lists what this copy includes and the commit it was built from. Nothing else
needs to be copied or downloaded.

## What the target machine needs

| Need | For | Notes |
|---|---|---|
| x86_64 Linux, Python 3.12 with `venv` | everything | Validated on Ubuntu 24.04. |
| About 12 GB free disk beyond this folder | everything | The Python environment takes about 3 GB; recordings need the rest. |
| FFmpeg on `PATH` | cameras and video files | `sudo apt install ffmpeg` |
| PostgreSQL | cameras and the dashboard | Any one of: Docker Compose (setup starts a database); or `sudo apt install postgresql postgresql-16-pgvector` followed by `sudo -u postgres psql -c "CREATE USER capstone PASSWORD 'password'" -c "CREATE DATABASE capstone OWNER capstone"` and `sudo -u postgres psql -d capstone -c "CREATE EXTENSION IF NOT EXISTS vector"`; or an existing server, with `DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname` in `.env` before setup. |
| glibc 2.38 or newer, `libgomp1`, `libssl3` | bundled Gemma only | Ubuntu 24.04 has them. On an older system setup says so and continues without Gemma. |
| RAM | everything | Screening about 2.5 GB, Gemma about 9 GB; 32 GB is ample. |

No GPU, Node.js or internet connection is required.

## 1. Copy and verify

Copy the folder (or extract the `.tar`, which keeps hidden files) and run
commands inside it:

```bash
python3.12 verify_transfer.py
```

This checks the size and SHA-256 of every file. If you received the `.tar`,
compare it with its `.sha256` file first.

## 2. Install

```bash
python3.12 deploy.py setup --with-camera
python3.12 deploy.py check --camera --dashboard --smoke
```

Setup creates `.venv` from the bundled wheels, creates `.env`, verifies the
models, runs a real CPU inference, prepares the database and, if this system
can run it, configures the bundled Gemma. It can be rerun after a failed step
and never overwrites existing settings. Leave out `--with-camera` if this
machine only serves the dashboard (no FFmpeg needed); add `--no-gemma` to skip
Gemma. If the `models/` folder is missing, setup and start print a warning and
carry on: the dashboard runs, cameras only record, and no incidents are opened.

## 3. Create the first sign-in

Nobody can open the dashboard until an admin account exists. Once is enough;
the admin then adds staff from the dashboard. It asks for a password (at least
8 characters with upper and lower case, a number and a symbol):

```bash
python3.12 deploy.py create-admin --name "Your Name" --email you@example.com --staff-id S001
```

## 4. Start

The whole system for one camera (Gemma, the API with login and the dashboard,
and the camera backend), all supervised by this one command:

```bash
python3.12 deploy.py start --dashboard --camera --rtsp-url rtsp://camera-address/stream --source-id camera_01
```

A recorded file instead of a camera (played at its real speed):

```bash
python3.12 deploy.py start --dashboard --camera --video /path/to/video.mp4 --source-id camera_01
```
To show staff invite emails (for example in a demo), start the email inbox
first; the email settings in `.env.example` already point at it:

```bash
docker compose -f data-pipeline/infra/compose.yaml up -d mailpit
```

Invites then arrive at <http://localhost:8025> instead of the start output.
Open <http://localhost:8000/login> and sign in. Incidents appear in the review queue
with their evidence clip. Add `--host 0.0.0.0` to reach the dashboard from
another machine. Sign-in is required, but the server speaks plain HTTP, so
passwords and the session cookie cross the network unencrypted: use a trusted
network, or put an HTTPS proxy in front and set `COOKIE_SECURE=true` in `.env`.
Press Ctrl+C once to stop: the camera backend finishes its queued Gemma
reviews, then everything exits.

Other modes:

```bash
python3.12 deploy.py start                      # dashboard, API and login only
python3.12 deploy.py start --camera --video F   # camera backend only (writes incidents)
```

Use one `--source-id` per camera and one camera backend process per camera.

## What happens to a window

1. YOLO26s tracking and rules, plus a scan every 10 s, propose windows.
2. The screening classifier (`ensemble-20261116`, CPU) scores each window for
   smoking, littering, loitering, rough sleeping and unattended items.
3. A window scored as an anomaly opens an incident with a short clip.
4. Gemma reviews that window: it answers per behaviour and writes one or two
   sentences describing the scene. Its verdict and description are attached
   to the incident. It reviews flagged windows only, and the same behaviour on
   the same camera at most every 30 s, and never while its previous review is
   still running (`GEMMA_CONFIRM_INTERVAL_S`).

Gemma does not decide whether an incident exists; its verdict and description
are a second opinion for the reviewer. Expect roughly 15 to 40 s per review on
CPU, depending on the processor. How much weight its verdict deserves, from
592 held-out windows:

| Behaviour | When Gemma confirms | When Gemma does not confirm |
|---|---|---|
| Smoking in close view, rough sleeping | Usually right | Usually right |
| Unattended item on a nearby camera, littering | Trust it (it was right 8–10 times in 10) | It often just missed it: it confirmed 1 in 3 real unattended items and 1 in 5 littering events |
| Loitering, smoking at CCTV distance, small distant items | It never confirmed these | Carries no information; rely on the clip |

## Settings worth knowing (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `GEMMA_LOCAL_THREADS` | half the cores | CPU threads for Gemma. Screening uses the rest. |
| `GEMMA_CONFIRM` | `flagged` | `all` reviews every window; only sensible with a GPU server. |
| `GEMMA_CONFIRM_INTERVAL_S` | `30` | Minimum video seconds between reviews of one behaviour on one camera. A review is also skipped while the previous one is still running. |
| `GEMMA_BASE_URL` | bundled server | Point it at another OpenAI-compatible server and remove `GEMMA_LOCAL=1` to use a GPU machine instead. |
| `VISION_CLASSIFIER` | newest installed | `ensemble-20261113`, `ensemble-20261111` or `mobileclip2-s2` select earlier classifiers. |
| `VISION_OPERATING_POINT` | `balanced` | `low_false_alarm` flags fewer ordinary windows and misses more unattended items. |
| `COOKIE_SECURE` | `false` (written by setup) | `true` only behind HTTPS; on plain HTTP away from localhost the browser would drop the login cookie. |
| `SMTP_HOST`, `SMTP_*`, `APP_URL` | unset | Where staff invite emails go and the sign-in link in them. Unset, the email (with its temporary password) is printed in the start output instead. |

## If one camera keeps raising the same false alarm

The classifier's thresholds were set on other people's cameras. When a camera
has something in view that scores high all day, calibrate it on its own
footage. Let it run on footage in which the behaviour does **not** happen
(five minutes or more; a recorded file through `--video` counts), stop it,
then:

```bash
python3.12 deploy.py calibrate --source-id camera_01
```

It reads the scores already journaled for that camera and raises its
threshold for each behaviour until at most 2% of those windows would have
been flagged. It prints what changed; nothing is lowered, no labels are
needed, and nothing leaves the machine. Start the system again to apply it.
`--clear` removes the calibration. A stricter threshold also misses weaker
real events on that camera, so use it for cameras with a false-alarm problem,
not everywhere.

## Limits

- Cameras, rules, hotspots and reports in the dashboard are not connected yet;
  they show demonstration data. Review buttons and login are connected.
- Scores and "confidence" rank windows; they are not measured precision.
- Model licences differ: see `docs/model_sources.md`. The MobileCLIP2 encoders
  are under Apple's research licence (non-commercial research use).
- More detail: `README.md`, `docs/deployment.md`, `docs/integration.md`,
  `docs/f1_study.md`.
