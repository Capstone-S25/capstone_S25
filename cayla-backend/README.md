# CCTV Incident Platform: Backend

The backend API for **InSight**, a CCTV incident platform. It handles login and sessions, staff accounts (with emailed temporary passwords), locations and cameras, monitoring rules, system settings, and a full audit trail of every change.

**Stack:** Python 3.12 · FastAPI · psycopg 3 · PostgreSQL 16 with pgvector · Docker Compose · Mailpit (fake email server for development)

---

## Contents

1. [What's in the box](#1-whats-in-the-box)
2. [Project layout](#2-project-layout)
3. [Quick start with Docker (recommended)](#3-quick-start-with-docker-recommended)
4. [Running without Docker](#4-running-without-docker)
5. [Using the API](#5-using-the-api)
6. [How login, sessions and roles work](#6-how-login-sessions-and-roles-work)
7. [Adding staff (invite flow)](#7-adding-staff-invite-flow)
8. [Email](#8-email)
9. [Monitoring rules and the rules engine](#9-monitoring-rules-and-the-rules-engine)
10. [Audit trail](#10-audit-trail)
11. [Configuration reference](#11-configuration-reference)
12. [Database and migrations](#12-database-and-migrations)
13. [Running the tests](#13-running-the-tests)
14. [Extending the backend](#14-extending-the-backend)
15. [Build status and roadmap](#15-build-status-and-roadmap)
16. [Design decisions to confirm with the team](#16-design-decisions-to-confirm-with-the-team)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What's in the box

When you run `docker compose up`, you get three services:

| Service   | What it is                                      | Address on your machine            |
|-----------|-------------------------------------------------|------------------------------------|
| `api`     | The FastAPI backend                             | http://localhost:8000              |
| `db`      | PostgreSQL 16 + pgvector                        | `localhost:5433` (user/pass/db: `cctv`) |
| `mailpit` | Catches every email the backend sends; sends none | http://localhost:8025 (inbox)    |

Useful pages once it's running:

- **http://localhost:8000/docs**: Swagger UI. Try every endpoint from the browser.
- **http://localhost:8000/redoc**: read-only API reference.
- **http://localhost:8000/healthz**: returns `{"ok": true}` if the API is up.
- **http://localhost:8025**: Mailpit inbox (invite and reset emails land here).

All API routes live under **`/api/v1`**.

---

## 2. Project layout

```
cctv-backend/
├── app/
│   ├── main.py              # Creates the FastAPI app and mounts every router under /api/v1
│   ├── config.py            # All settings, read from environment variables
│   ├── db.py                # Connection pool; one transaction per request
│   ├── deps.py              # Login + role checks: CurrentUser, AdminUser, AnyLoggedInUser
│   ├── security.py          # argon2 password hashing, session tokens
│   ├── password_policy.py   # Temporary password generator + new-password rules
│   ├── mailer.py            # send_email() + invite / reset email templates
│   ├── audit.py             # audit.record(): writes the audit log (used by every edit)
│   ├── rules_engine.py      # Decides if a detection is monitored and its severity
│   └── routers/
│       ├── auth.py              # login, logout, /me, change-password
│       ├── users.py             # list / create / edit users, reset-password
│       ├── locations_cameras.py # locations, cameras, health, live feed (stubbed)
│       ├── rules.py             # monitoring rules CRUD (stubbed SQL)
│       ├── settings.py          # settings + event types (stubbed SQL)
│       └── audit_log.py         # GET audit log with filters and paging
├── db/migrations/           # SQL run in filename order (001 → 004)
├── scripts/create_admin.py  # Creates the first admin account
├── tests/                   # Unit tests (no database needed)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example             # Every setting with an example value
```

---

## 3. Quick start with Docker (recommended)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + the Compose plugin)
- Free ports **8000**, **5433** and **8025**

### Step 1: Start everything

From the `cctv-backend/` folder:

```bash
docker compose up -d --build
```

This builds the API image, starts Postgres, Mailpit and the API, and **runs the migrations in `db/migrations/` automatically the first time** the database volume is created.

Check it's running:

```bash
docker compose ps
curl http://localhost:8000/healthz      # → {"ok":true}
```

### Step 2: Create the first admin

Nobody can log in until an admin exists, and only admins can create users, so the first one is made with a script:

```bash
docker compose exec api python -m scripts.create_admin \
    --name "Admin" --email admin@insight.sg --staff-id S001 --dob 1990-01-01
```

It asks for the password twice (at least 8 characters). It reads the password interactively so it never ends up in your shell history.

### Step 3: Log in and try it

1. Open http://localhost:8000/docs
2. Expand **POST /api/v1/auth/login** → *Try it out* → enter your email and password → *Execute*.
3. The response sets a session cookie in your browser, so every other request in Swagger is now logged in.
4. Try **GET /api/v1/me**. You should see your own account.

### Everyday Docker commands

```bash
docker compose logs -f api          # follow the API logs
docker compose restart api          # restart the API after changing env vars
docker compose up -d --build api    # rebuild after changing Python code
docker compose down                 # stop everything (data is kept)
docker compose down -v              # stop AND delete the database (fresh start, migrations re-run)
docker compose exec db psql -U cctv -d cctv   # open a SQL shell
```

> **Note:** the image copies the code in at build time and has no auto-reload, so after editing Python files you need `docker compose up -d --build api`. For fast iteration, run the API locally instead (section 4) and keep only `db` and `mailpit` in Docker.

---

## 4. Running without Docker

Useful for development with auto-reload.

### Step 1: Start a database (and optionally Mailpit)

Easiest: keep using Docker for just those two:

```bash
docker compose up -d db mailpit
```

Or use your own PostgreSQL 16 (pgvector isn't needed by this backend's tables yet).

### Step 2: Install Python dependencies

Python 3.12 is what the Docker image uses. 3.10+ works.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set environment variables

The app reads **real environment variables only**. It does not load a `.env` file by itself. The minimum for local development:

```bash
export DATABASE_URL=postgresql://cctv:cctv@localhost:5433/cctv
export COOKIE_SECURE=false          # needed on plain http://, otherwise the browser drops the login cookie
export SMTP_HOST=localhost          # send emails to Mailpit…
export SMTP_PORT=1025
export SMTP_STARTTLS=false
# …or leave SMTP_HOST unset to have emails printed in the terminal instead
```

See [section 11](#11-configuration-reference) for every setting.

> If you copy `.env.example` to `.env` and load it with `set -a; source .env; set +a`, put quotes around `SMTP_FROM` first (`SMTP_FROM="InSight <no-reply@insight.towera.sg>"`). Otherwise the shell treats `<` as a redirect and fails.

### Step 4: Run the migrations

Skip this if you started `db` with Docker Compose. It already ran them.

```bash
for f in db/migrations/*.sql; do psql "$DATABASE_URL" -f "$f"; done
```

### Step 5: Create the admin and start the API

```bash
python -m scripts.create_admin --name "Cayla" --email you@example.com --staff-id S001 --dob 2002-01-31
uvicorn app.main:app --reload
```

The API is now on http://localhost:8000 and restarts whenever you save a file.

---

## 5. Using the API

### Endpoints

All paths are prefixed with `/api/v1`. **Who** = who is allowed to call it.

| # | Method & path | Who | What it does | Status |
|---|---|---|---|---|
| 1 | `POST /auth/login` | anyone | Log in; sets session cookie, also returns a token | ✅ |
| 2 | `POST /auth/logout` | logged in | Ends this session only | ✅ |
| 3 | `GET /me` | logged in | The current user (check `must_change_password`) | ✅ |
| – | `POST /auth/change-password` | logged in | Set a new password | ✅ |
| 4 | `GET /users` | admin | List all users | ✅ |
| 5 | `POST /users` | admin | Add a user; emails them a temporary password | ✅ |
| 6 | `PATCH /users/{id}` | admin | Edit name, email, role, deactivate, etc. | ✅ |
| – | `POST /users/{id}/reset-password` | admin | Resend invite / reset a forgotten password | ✅ |
| 7 | `GET /locations` | staff+ | Locations with their cameras | 🚧 501 |
| 8 | `POST /locations` | admin | Add a location | 🚧 501 |
| 9 | `PATCH /locations/{id}` | admin | Rename a location | 🚧 501 |
| 10 | `GET /cameras?location_id=` | staff+ | List cameras | 🚧 501 |
| 11 | `POST /cameras` | admin | Add a camera | 🚧 501 |
| 12 | `GET /cameras/{id}` | staff+ | One camera | 🚧 501 |
| 13 | `GET /cameras/{id}/health` | staff+ | Is it recording? | 🚧 501 |
| 14 | `GET /cameras/{id}/live` | staff+ | Live feed (HLS playlist URL) | 🚧 501 |
| 35 | `GET /rules` | staff+ | List monitoring rules | 🚧 501 |
| 36 | `POST /rules` | admin | Create a rule | 🚧 501 |
| 37 | `PUT /rules` | admin | Bulk "Save changes" | 🚧 501 |
| 38 | `PATCH /rules/{id}` | admin | Edit / toggle a rule | 🚧 501 |
| 39 | `DELETE /rules/{id}` | admin | Delete a rule | 🚧 501 |
| 40 | `GET /settings` | staff+ | System settings | 🚧 501 |
| 41 | `PATCH /settings` | admin | Change settings | 🚧 501 |
| 42 | `GET /event-types` | staff+ | List event types | 🚧 501 |
| 43 | `POST /event-types` | admin | Add an event type | 🚧 501 |
| 44 | `PATCH /event-types/{id}` | admin | Rename / retire an event type | 🚧 501 |
| 45 | `GET /audit-log` | admin | Audit trail with filters and paging | ✅ |

🚧 endpoints answer **`501 Not built yet`** on purpose, so the dashboard can tell "not built" apart from "broken". Their request models and validation are already written. Each stub's docstring explains how to build it.

### Calling the API from the command line

Login returns a `token`. Scripts can send it as a Bearer header instead of using the cookie:

```bash
# Log in and keep the token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"your-password"}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# Who am I?
curl -s http://localhost:8000/api/v1/me -H "Authorization: Bearer $TOKEN"

# Add a staff member (they get an email with a temporary password)
curl -s -X POST http://localhost:8000/api/v1/users \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"Alex Tan","email":"alex@example.com","staff_id":"S002","date_of_birth":"1999-05-20","role":"staff"}'

# Deactivate a user (logs them out everywhere immediately)
curl -s -X PATCH http://localhost:8000/api/v1/users/2 \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"is_active": false}'

# Last 20 changes to user #2
curl -s "http://localhost:8000/api/v1/audit-log?entity_type=user&entity_id=2&limit=20" \
  -H "Authorization: Bearer $TOKEN"

# Log out
curl -s -X POST http://localhost:8000/api/v1/auth/logout -H "Authorization: Bearer $TOKEN"
```

### Error responses you should expect

| Code | Meaning |
|---|---|
| `401 Not logged in` / `Session expired or invalid` | No token/cookie, or the session was logged out, expired or the user was deactivated |
| `401 Incorrect email or password` | Wrong login (deliberately the same message for unknown emails) |
| `401 Your temporary password has expired…` | Invite older than `TEMP_PASSWORD_HOURS`; an admin must reset it |
| `403 Password change required` | User is still on a temporary password; only `/me`, `/auth/change-password`, `/auth/logout` work |
| `403 Admins only` | Staff calling an admin endpoint |
| `409` | Email or staff ID already in use (or duplicate name) |
| `422` | Validation failed (password rules, bad fields) |
| `501 Not built yet` | Stub endpoint |
| `502` | Email couldn't be sent, so nothing was saved |

---

## 6. How login, sessions and roles work

- **Passwords** are hashed with **argon2**. Plain passwords are never stored or logged.
- **Login** creates a random session token. Only its SHA-256 hash is stored in the `sessions` table, so a database leak doesn't expose live sessions.
- The token is sent back in two ways: an **httpOnly cookie** (`cctv_session`, what the dashboard uses) and in the JSON body (for scripts, via `Authorization: Bearer <token>`).
- **Every request** re-checks that the session isn't expired or revoked and the user is still active. That means **logout and deactivation take effect instantly**, unlike a plain JWT.
- Sessions last `SESSION_HOURS` (default 12).
- **Changing your password** logs out all your *other* sessions. **An admin resetting your password or deactivating you** logs out *all* of them.

### Roles

There are two roles: **`admin`** and **`staff`**.

Protect a route by adding one of these parameters (from `app/deps.py`):

| Parameter | Lets through |
|---|---|
| `user: AnyLoggedInUser` | Any valid session, **including** someone still on a temporary password. Only for `/me`, logout, change-password. |
| `user: CurrentUser` | Any logged-in user who has set their own password |
| `user: AdminUser` | Admins only (403 for staff) |

Safety rule: admins **can't demote or deactivate themselves**, so the last admin can't lock everyone out.

---

## 7. Adding staff (invite flow)

Admins never type or see a staff member's password.

1. **Admin adds the person:** `POST /users` with name, email, staff ID, date of birth and role (`staff` by default).
2. **The backend** generates a random temporary password (e.g. `K7m-9qXp-2Lw`, no look-alike characters), emails it, and marks the account `must_change_password = true`. It expires after `TEMP_PASSWORD_HOURS` (default 168 = 7 days).
3. **The staff member logs in** with it. Login succeeds and `user.must_change_password` is `true`. The dashboard should show a "Set your password" screen. Every other endpoint answers **403** until they do.
4. **They choose a password:** `POST /auth/change-password`

   ```json
   { "new_password": "myNewPass1", "confirm_password": "myNewPass1" }
   ```

   No current password is needed on this first change.
5. Done: full access.

**New password rules:** at least 8 characters, contains a number, and differs from the current/temporary password.

**Changing a password later (from Settings):** same endpoint, but `current_password` is required:

```json
{ "current_password": "myNewPass1", "new_password": "evenBetter2", "confirm_password": "evenBetter2" }
```

**Forgot password / invite expired:** an admin calls `POST /users/{id}/reset-password`. If the person never set a password, they get the invite email again. Otherwise they get a "your password was reset" email. Either way they're logged out everywhere and must choose a new password at next login. (There is no self-service "forgot password" link yet.)

**Date of birth** is required because it's needed later for password-locked report PDFs.

---

## 8. Email

All email goes through `app/mailer.py → send_email()`. Where it goes depends on `SMTP_HOST`:

| Setup | Behaviour |
|---|---|
| **Docker Compose (default)** | Sent to Mailpit. Read it at **http://localhost:8025**. Nothing reaches a real inbox. |
| **`SMTP_HOST` empty** | The email is printed to the API log (`docker compose logs api`). Handy for tests. |
| **Real SMTP** | Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS=true`. Amazon SES works via its SMTP interface. |

If sending fails, the request returns **502** and **the whole change is rolled back** (no half-created user with no email).

Other parts of the system (alerts, reports) should call `send_email()` too, so there is one place that sends mail.

---

## 9. Monitoring rules and the rules engine

A **rule** says: *for these cameras in this location, watch for this event type, on these days, between these times, at this severity, if the detection confidence is at least X.*

Rule fields (`app/routers/rules.py`):

| Field | Example | Notes |
|---|---|---|
| `location_id` | `1` | |
| `camera_ids` | `[3, 4]` | All must belong to the location |
| `event_type_id` | `2` | |
| `days` | `[1,2,3,4,5]` | 1 = Mon … 7 = Sun |
| `start_time`, `end_time` | `"22:00"`, `"06:00"` | See time rules below |
| `severity` | `"high"` | `low` / `medium` / `high` / `critical`, required |
| `min_confidence` | `0.6` | 0–1, required (no global default) |
| `is_active` | `true` | |

**Time windows** (in `SITE_TZ`, default Asia/Singapore):

- `start < end` (09:00–18:00): same-day window. The end time itself is excluded.
- `start > end` (22:00–06:00): overnight. The part after midnight belongs to the day it started, so a Friday 22:00–06:00 rule also covers Saturday 00:00–06:00.
- `start == end`: all day on the listed days.

### How a detection is judged

`app/rules_engine.py` turns rules into a decision. The video pipeline should call it **right after object detection and before the VLM**:

```python
from app.rules_engine import evaluate_detection

decision = evaluate_detection(conn, camera_id, event_type_id, detected_at, confidence)
if not decision.passes:
    pass  # drop it: not monitored, outside hours, or below threshold (decision.reason says which)
else:
    # send to the VLM, then create the incident with severity=decision.severity
    ...
```

The logic:

1. No active rule for this camera + event type → **not monitored**.
2. Rules exist but none is on right now → **dropped** (before the VLM).
3. Several rules on at once → **the stricter one wins on each point separately**: highest severity, lowest confidence threshold.
4. Confidence below the threshold → **ignored**.
5. Otherwise → **passes**, with `decision.severity` to copy onto the incident.

Incident code only reads `incidents.severity`; it never needs to call the engine. Deleting or editing a rule later never changes past incidents.

`decide()` is pure Python with no database, which is why it's covered by unit tests.

---

## 10. Audit trail

Every create / update / delete is written to `audit_log`, recording who did it, when, and **only the fields that changed** (old and new values). Passwords, password hashes and session tokens are never written, and bookkeeping columns (`updated_at`, etc.) are ignored.

**Reading it:** `GET /api/v1/audit-log` (admins only). Filters:

| Query param | Example |
|---|---|
| `entity_type` | `user`, `rule`, `settings` |
| `entity_id` | `2` |
| `actor_user_id` | `1` |
| `since`, `until` | `2026-11-01T00:00:00+08:00` |
| `limit` | 1–200, default 50 |
| `before_id` | paging cursor |

Results are newest first. To get the next (older) page, pass the response's `next_before_id` back as `before_id`. When `next_before_id` is `null`, there are no more pages.

**Writing it:** see [section 14](#14-extending-the-backend). The write happens in the **same transaction** as the change: if the change fails, there's no log line; if the log write fails, the change is rolled back.

There is intentionally **no HTTP endpoint for writing audit entries**. The API and the Python workers share the database and call `audit.record()` directly. An HTTP route would let a client write fake history. Workers pass `actor_id=None` (shown as "system").

---

## 11. Configuration reference

All settings are environment variables (see `app/config.py` and `.env.example`). In Docker, set them under `api: environment:` in `docker-compose.yml`, then `docker compose up -d api`.

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | `postgresql://cctv:cctv@localhost:5433/cctv` | Postgres connection string (Docker uses host `db`, port `5432`) |
| `SESSION_HOURS` | `12` | How long a login lasts |
| `COOKIE_SECURE` | `true` | Cookie only sent over HTTPS. **Set `false` for local http://** |
| `SITE_TZ` | `Asia/Singapore` | Time zone for rule days and hours |
| `TEMP_PASSWORD_HOURS` | `168` | How long an emailed temporary password works |
| `APP_NAME` | `InSight` | Product name in emails |
| `SITE_NAME` | `Tower A` | Site name in emails ("…added you to InSight for Tower A") |
| `APP_URL` | `http://localhost:8000/docs` | Sign-in link in emails. Change to the dashboard's login page |
| `SMTP_HOST` | *(empty)* | Mail server. Empty = print emails to the log |
| `SMTP_PORT` | `587` | |
| `SMTP_USER` / `SMTP_PASSWORD` | *(empty)* | Login is skipped if `SMTP_USER` is empty |
| `SMTP_FROM` | `InSight <no-reply@insight.towera.sg>` | Sender |
| `SMTP_STARTTLS` | `true` | Use STARTTLS (Mailpit needs `false`) |

### Before going to production

- `COOKIE_SECURE=true` and serve over HTTPS.
- Change the Postgres password in `docker-compose.yml` and `DATABASE_URL`, and stop publishing port 5433.
- Point `SMTP_*` at a real mail server and `APP_URL` at the dashboard.
- Remove the `mailpit` service.

---

## 12. Database and migrations

Migrations are plain SQL files in `db/migrations/`, run **in filename order**:

| File | What it does |
|---|---|
| `001_renames.sql` | Renames old tables from the earlier design: `sources → cameras`, `cameras.source_id → camera_id`, `anomaly_events → incidents`. Does nothing on a fresh DB. |
| `002_platform.sql` | Creates `users`, `sessions`, `locations`, `cameras`, `event_types`, `rules`, `rule_cameras`, `settings` (one row, id = 1), `audit_log`. Also upgrades the old schema in place. |
| `003_role_staff.sql` | Renames role `member → staff`. |
| `004_temp_passwords.sql` | Adds `must_change_password` and `temp_password_expires_at` to `users`. |

All of them are safe to run more than once.

**With Docker**, they run automatically **only when the database volume is first created**. If you add a new migration later, either:

- run it by hand: `docker compose exec -T db psql -U cctv -d cctv < db/migrations/005_xxx.sql`, or
- wipe and rebuild: `docker compose down -v && docker compose up -d --build` (**deletes all data**, so you'll need to create the admin again).

**Upgrading an older database:**

- `001` changes table names the Python workers query. Agree it with whoever owns the workers so their SQL changes at the same time.
- `002` only allows roles `admin` / `staff`. Update any other `users.role` values first or it will fail.
- The old `cameras.location` text column is left in place. Create `locations` rows from it, set `cameras.location_id`, then drop the old column and run `ALTER TABLE cameras ALTER COLUMN location_id SET NOT NULL;`.

---

## 13. Running the tests

The tests cover the rules engine, audit helper, password policy and email templates. They **don't need a database or Docker**.

From the `cctv-backend/` folder, with the requirements installed:

```bash
python -m pytest tests/
```

Use `python -m pytest`, not bare `pytest`. The project has no pytest config, so bare `pytest` can't find the `app` package and fails with `ModuleNotFoundError: No module named 'app'`. (To make bare `pytest` work, add a `pytest.ini` with `[pytest]` and `pythonpath = .`.)

The Docker image doesn't include `tests/`, so run them on your machine, not inside the container.

---

## 14. Extending the backend

### Building a stubbed endpoint

`routers/users.py` is the worked example. Every create / update / delete follows the same four steps:

```python
from .. import audit
from ..deps import AdminUser, Conn


@router.patch("/locations/{location_id}")
def rename_location(
    location_id: int, body: LocationCreate, user: AdminUser, conn: Conn
):
    # 1. Read the current row and lock it
    old = conn.execute(
        "SELECT * FROM locations WHERE location_id = %s FOR UPDATE", (location_id,)
    ).fetchone()
    if old is None:
        raise HTTPException(404, "Location not found")

    # 2. Write
    new = conn.execute(
        "UPDATE locations SET name = %s, updated_at = now() "
        "WHERE location_id = %s RETURNING *",
        (body.name, location_id),
    ).fetchone()

    # 3. Audit (same transaction)
    audit.record(
        conn,
        actor_id=user.user_id,
        action="update",
        entity_type="location",
        entity_id=location_id,
        old=old,
        new=new,
    )

    # 4. Return
    return new
```

Things you get for free:

- **One request = one transaction.** If anything raises, everything (including the audit row) rolls back. If it returns normally, it commits.
- **Rows come back as dicts** (`row["name"]`).
- `audit.record()` skips no-op updates on its own.
- Catch `psycopg.errors.UniqueViolation` and return 409 for duplicates.

Rule endpoints should call `check_cameras_in_location()` (already written in `rules.py`) before saving.

### Adding a new router

1. Create `app/routers/incidents.py` with `router = APIRouter(prefix="/incidents", tags=["Incidents"])`.
2. Protect each route with `user: CurrentUser` or `user: AdminUser`.
3. Register it in `app/main.py`:

   ```python
   from .routers import incidents

   app.include_router(incidents.router, prefix=API)
   ```

### From a Python worker

Workers can reuse the same helpers with their own connection:

```python
import psycopg
from psycopg.rows import dict_row
from app import audit
from app.config import DATABASE_URL
from app.rules_engine import evaluate_detection

with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
    decision = evaluate_detection(
        conn, camera_id, event_type_id, detected_at, confidence
    )
    ...
    audit.record(
        conn,
        actor_id=None,
        action="create",
        entity_type="incident",
        entity_id=incident_id,
        new=incident_row,
    )
```

`load_rules()` needs `dict_row`. Workers should re-read `settings` each cycle rather than caching it.

---

## 15. Build status and roadmap

**Done:** login, logout, `/me`, change-password, users list/create/edit, admin reset-password, staff invite emails, audit log (helper + GET endpoint), role checks, rules engine with tests.

**Stubbed (return 501, models written):** locations, cameras, camera health, live feed, rules CRUD + bulk save, settings, event types.

**Suggested order:**

| Week | Work |
|---|---|
| 1 | Get the stack running, create an admin, test login/me. Share `CurrentUser` / `AdminUser` and `audit.record` with the team. Agree on `001_renames.sql`. |
| 2 | Locations, cameras, event types, settings (all small: copy `users.py`). Then rules POST / PATCH / DELETE / GET. |
| 3 | Bulk rule save (`PUT /rules`, plan in its docstring), camera health, call `evaluate_detection` in the event poller before the VLM. |
| 4 | Live feed: ffmpeg writes a rolling HLS playlist per camera, the endpoint returns its URL, the dashboard plays it with hls.js (expect a 4–8 s delay; MediaMTX/WebRTC if that's too slow). |

**Deferred:** login lockout, idle timeouts, 2FA, self-service "forgot password".

---

## 16. Design decisions to confirm with the team

- **Severity levels:** `low / medium / high / critical`.
- **Overnight rules:** a Friday 22:00–06:00 rule covers Saturday until 06:00. `start = end` means all day. Times are in `SITE_TZ`.
- **New `sessions` table** (not on the Database Changes sheet). It's what makes instant logout possible.
- **`cameras.source_id` renamed to `camera_id`.** The pipeline's SQL must change at the same time.
- **Settings defaults** (pre 10 s / post 10 s / quiet 30 s, 7-day segment retention, 90-day incident retention) are placeholders.
- **Users are never deleted**, only deactivated, because audit history points at them. **Event types are never deleted**, only retired.
- **Admins can't demote or deactivate themselves.**
- Password reset is admin-driven only for now.

---

## 17. Troubleshooting

| Problem | Fix |
|---|---|
| Port 8000 / 5433 / 8025 already in use | Change the left-hand number in `ports:` in `docker-compose.yml` (e.g. `"8001:8000"`). |
| Logged in through Swagger, but the next call says `401 Not logged in` | You're on http:// with `COOKIE_SECURE=true`. Set `COOKIE_SECURE=false` locally, or send `Authorization: Bearer <token>`. |
| `relation "users" does not exist` | Migrations didn't run. With Docker: `docker compose down -v && docker compose up -d --build`. Without: run the SQL files in order. |
| A new migration file didn't apply | Docker only runs them on a brand-new volume. Run it by hand (section 12). |
| `create_admin` fails with a duplicate key | That email or staff ID already exists. Use different ones, or reset the DB. |
| Every endpoint returns `403 Password change required` | You're on a temporary password. Call `POST /auth/change-password` first. |
| `502 Couldn't send the email` when adding a user | Check the `SMTP_*` settings and that Mailpit is running (`docker compose ps`). Nothing was saved, so just retry. |
| No invite email arrived | In development it's in Mailpit (http://localhost:8025), or in `docker compose logs api` if `SMTP_HOST` is empty. |
| `ModuleNotFoundError: No module named 'app'` | Run commands from the `cctv-backend/` folder, and use `python -m pytest` / `python -m scripts.create_admin`. |
| Code changes don't show up in Docker | Rebuild: `docker compose up -d --build api`. |
| `open http://…` doesn't work | `open` is macOS only. Paste the URL into your browser. |
