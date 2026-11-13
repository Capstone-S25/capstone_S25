# Application integration audit — 6 November 2026

## Scope and source history

`feat/vision-app-integration` starts at `origin/feat/data-pipeline` commit
`18a6be0`. The existing recording, database, scheduler, decoding, migrations,
and backend tests remain. Active detection/tracking and fitted checks were
extracted from `origin/feat/stage1-integration` commit `a206896` into
`vision_runtime/tracking/`; dataset loaders and research loops were omitted.
Trained classifier inference was extracted from the prior integration runtime
and validated against its saved transfer reference. The model transfer archive
identifies its original source as `a27f881`. The upload UI builds on the local
Streamlit work in the main checkout at `3637b31`.

The branch removes the base branch's research notebooks, evaluation figures,
dataset download scripts, training/evaluation modules, research-only tests,
bundled demo video, and IDE settings. It retains focused application tests,
backend tests, license texts and deployment documentation. No research files
or local changes are removed from the original checkout/branches.

## What already existed

| Area | Existing behavior on `feat/data-pipeline` |
|---|---|
| Recording | FFmpeg records file/RTSP input in a background thread; only finalized segments become jobs. |
| Registration | PostgreSQL stores paths, recording/session identity, timestamps, status, and retention metadata; path uniqueness makes registration idempotent. |
| Segment scheduling | Oldest pending segment per source; round-robin across configured sources; transactional `FOR UPDATE SKIP LOCKED` claims. |
| Processing | PyAV decodes segments and selects frames by timestamp; status becomes processed or failed. |
| AI / UI connection | No configured detector, classifier, AI event queue, Gemma confirmation, or upload interface. |

## Added application flow

1. The backend's ordered `on_frame` callback feeds a persistent tracker for one
   camera. Upload videos use the same decoder, frame selector, and adapter.
2. YOLO26s provides person/item boxes, ByteTrack supplies short-term track IDs,
   and pose/rule checks propose event regions. Latest fitted pose action,
   reach verification, body/overlap/rider guards, and temporal crop checks are
   retained. Faster R-CNN supplies classifier contextual crop proposals; it
   is not the active camera tracker.
3. A bounded **screening queue** (16 queued events) runs the screening
   classifier. Its results are journaled before optional confirmation. The
   default is the November **SigLIP2-B + MobileCLIP2-B ensemble**
   (`ensemble-20261111`, `vision_runtime/ensemble.py`); its heads were fitted
   with whole cameras held out and its thresholds set on those held-out scores.
   Loitering and unattended windows are also compared with the session's
   reference frame. See [the F1 study](f1_study.md) for results and limits.
   The October MobileCLIP route stays available as `mobileclip2-s2`; it
   computes only the features its selected heads read, and
   `tests/test_classifier_parity.py` checks it against the original dense
   computation.
4. A separate bounded **confirmation queue** (8 queued events) calls the
   configured Gemma vision endpoint. Screening continues while this worker
   waits. Jobs waiting more than 120 seconds are marked not run. Requests have
   5-second connect and 110-second read timeouts; these are transport limits,
   not a hard bound on total job duration.
5. Gemma sees sampled images and their timestamps, task definitions, and
   proposed region/track context, while classifier predictions are not supplied
   as ground truth. Short views cover approximately 4 seconds; long views cover
   approximately 20 seconds. Output uses constrained JSON, low-variance decoding,
   frame references, evidence text and normal/anomaly/unknown decisions.
6. The event journal retains provisional scores and confirmation separately.
   Disagreement or unknown confirmation requires review. A final alert flag
   requires agreement and the existing evidence/history/ownership guards.
   There is no external alert dispatcher; upload summaries always disable it.

The CLI defaults to CPU, 5 selected frames/second, 2-second target recording
segments, and a periodic all-task scene scan every 10 seconds, in addition to
rule-triggered candidates. A final scene review runs at end-of-file/shutdown.
These periodic windows provide coverage; they are not continuous per-frame
five-class predictions. Large keyframe intervals can extend recording segments.

Temporal crop verification can delay event release. The adapter preserves the
original trigger frame and `timestamp_s`, records `available_at_s` separately,
and ends classifier evidence at the trigger time. It does not relabel later
frames as if they had been observed earlier. Camera session changes and large
gaps reset tracking/history instead of carrying identities across discontinuities.
Pending gate candidates are discarded on resets. Backward timestamps are
rejected; duplicate timestamps are skipped.

## Job states and storage

```text
PostgreSQL: pending → processing → processed / failed
AI events:  queued → screened → confirmation_pending → confirming → reviewed
Failure:    not_run / screening_error / confirmation_not_run / confirmation_error
```

The segment `processed` status covers decoding and frame callbacks, including
admission attempts to the bounded AI queue; it does not certify that AI
classification or confirmation finished. Camera queue overflow is explicit in
the AI journal instead of silently fabricating a normal decision; the live
webcam view replaces a waiting window with the newest one. Uploads have no
real-time deadline, so tracking waits for screening space instead of marking
later windows of a long video as not run. A confirmation error
preserves provisional results and records unknown confirmation.

Camera evidence and append-only event transition journals live under
`runtime/cameras/` by default. Uploads use temporary folders under
`runtime/uploads/`; clearing/replacing a review releases its folder only after
active background jobs finish. A pending review is cancelled when released;
an already running request may finish but does not publish a discarded review.
Uploads older than one hour are pruned when another upload begins. There is no
periodic janitor for abandoned browser sessions. Model weights live under
`models/`. Both model storage and generated media are ignored by Git.

Each journal write is completed before the in-process state is exposed; I/O
errors are surfaced. This is a local append journal, not an fsync/transactional
message broker. PostgreSQL persists segment jobs, but there is no automatic
lease recovery for jobs stranded `processing`. Both AI queues are in memory;
there is no automatic journal replay or exactly-once delivery after a crash.
A future durable event/outbox table and camera ownership leases are needed for
reliable unattended recovery and multiple consumers per camera.

## Validation performed

| Check | Observed outcome |
|---|---|
| Existing backend plus focused queue/evidence tests | 105 passed against a real temporary PostgreSQL instance (98 retained backend checks and 7 application checks). |
| Migrations | `alembic upgrade head` and `alembic check` passed; no schema change required. |
| Model import | 15 allowlisted model files imported with size/hash and anomaly-definition verification. |
| Classifier parity | All five scores and decisions matched the saved CPU transfer candidate; maximum absolute score drift 0.0. |
| CPU browser execution | Real image and 4.2-second video uploads analyzed and JSON reports downloaded; video produced 21 selected frames and one final scene window. Desktop and 390-pixel mobile layout checks passed with no browser JavaScript errors. |
| Full backend with real Gemma | FFmpeg → PostgreSQL → decode → tracking → MobileCLIP → Gemma completed, including `reviewed` journal state. The short video correctly left duration-dependent tasks unknown. |
| Live confirmation UI | A real Gemma-enabled image review reached zero pending/unavailable jobs; the browser downloaded a final `reviewed` report and cleared the upload without JavaScript errors. |

Local raw validation receipts, screenshots and execution logs are kept outside
the tracked branch in `.local/integration_validation_20261106/` of the original
workspace. Small representative fixtures validate wiring, not general accuracy
or operational false-alarm rates. No new model accuracy improvement is claimed.

## Practical limits and next operational work

The CPU browser fixture took about 2 seconds for a still and 15 seconds for a
4.2-second video on this host, excluding first-use loading. These are smoke-run
observations, not a latency benchmark or a two-second real-time guarantee. The
real Gemma wiring test used an existing GPU-hosted 26B-A4B model server; the
default application still runs without a GPU or Gemma. A physical RTSP camera
and sustained camera backlog/recovery load were not available for validation.

Streamlit is an upload client; live camera events currently go to the journal,
not a camera dashboard or existing external business application API. Next work
should add durable AI jobs/outbox/recovery, a camera event subscription API,
measured sustained CPU throughput, retention cleanup, and deployment-specific
authentication before exposing these local tools to other users.

## Dashboard integration (13 November 2026)

`feat/frontend` (React dashboard) and the incidents API from `ym-backend`
(commit `2414048`) are merged here and connected to the vision runtime.

```text
camera / file → FFmpeg segments → PostgreSQL jobs → decode → tracking
  → screening classifier → EventStore journal ──► IncidentRecorder
                         → optional Gemma     ──►   (PostgreSQL + evidence files)
React dashboard ◄── /api/v1/incidents, /{id}, /{id}/clip, /{id}/thumbnail ◄──┘
```

| Piece | What it does |
|---|---|
| `vision_runtime/service.py` | `EventStore(on_write=...)` calls a listener after every journaled transition. A listener failure is logged and never stops screening. |
| `data_pipeline/detection/incidents.py` | `IncidentRecorder`: a screened anomaly opens an incident, or extends the open one of the same camera and event type when it is within 30 s (the planned `t_quiet`). It saves the window's frames as an H.264 clip plus a thumbnail, and stores the screening result and any later Gemma confirmation as `vlm_results` rows. Rule-triggered events also store their region as a flagged detection. |
| `data_pipeline/api/routers/evidence.py` | `GET /api/v1/incidents/{id}/clip` (API List #17) and `/thumbnail`. Files are served only from inside the recordings root. |
| `IncidentDetail.vlm_history` | All model verdicts, oldest first. `vlm_verdict` stays the latest one. |
| `frontend/src/data/` | `api.ts` maps API incidents to the dashboard's `Incident` type; `useIncidents` polls every 5 s and falls back to demo data when the API is down; `mock.ts` recreates the demo data module the pages import (it was not committed with them). |

Decisions and limits:

- **Screening is the gate.** An incident exists only when the screening
  classifier says anomaly. Gemma's result is attached as a hint; a human still
  reviews every incident. If Gemma alone says anomaly, no incident is opened.
- **Severity** comes from a fixed default per event type until monitoring
  rules exist (unattended items critical; smoking and rough sleeping moderate;
  littering and loitering minor).
- **Confidence** shown in the dashboard is a logistic of the screening margin
  over its threshold (or the classifier's calibrated probability when it
  provides one). It is a ranking aid, not a measured precision.
- **Evidence clip** is the classifier's own input window (4 s or 20 s at the
  sampled frame rate), not the full-rate recording.
- **Recorder fix:** file inputs are read with `-fflags +genpts`. Without it an
  AVI whose packets lack timestamps never completed a segment, so nothing
  downstream ran.
- The dashboard's Accept/Dismiss/Escalate buttons, login, cameras, rules,
  hotspots and reports are not connected (their endpoints do not exist yet).

Checked on 13 November 2026: ABODA video 9 (held out, 83 s) through
`main.py` on CPU opened one unattended-item incident from the 40 s window
onward (label: unattended from 44 s) and one loitering incident (two windows);
both appeared in the dashboard with playable clips. 155 backend tests pass
against PostgreSQL 16, including the recorder and evidence endpoints.
