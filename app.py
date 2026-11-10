"""Minimal local Streamlit interface. Run: python -m streamlit run app.py"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_runtime.live import CameraManager  # noqa: E402
from vision_runtime.review import IMAGE_TYPES, LABELS, TASKS, Reviewer  # noqa: E402

st.set_page_config(page_title="Scene review", page_icon="◉", layout="wide")
st.markdown(
    """<style>
    .stApp { background: #f8faf9; }
    .block-container { max-width: 1180px; padding-top: 3rem; padding-bottom: 2rem; }
    h1 { font-size: 2.6rem !important; font-weight: 650 !important; letter-spacing: -.07rem; }
    h3 { font-size: 1.08rem !important; font-weight: 600 !important; }
    .eyebrow { color: #507166; font-size: .72rem; letter-spacing: .15em; font-weight: 650; }
    .subtitle { color: #67746f; font-size: 1rem; margin: -.35rem 0 1.6rem; }
    .signal-row { display: flex; justify-content: space-between; align-items: center;
      gap: 1rem; padding: 1rem 0; border-bottom: 1px solid #e4ebe7; }
    .signal-name { color: #213b30; font-size: .9rem; }
    .badge { border-radius: 5px; padding: .28rem .6rem; font-size: .72rem; white-space: nowrap; }
    .waiting { background: #edf0ee; color: #6b7871; }
    .normal { background: #e8f1ec; color: #326347; }
    .anomaly { background: #fff0d9; color: #925c11; }
    .unknown { background: #edf0ee; color: #6b7871; }
    .footnote { font-size: .75rem; color: #7a8880; padding-top: 1.5rem; }
    [data-testid="stFileUploader"] { margin-bottom: .5rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def reviewer():
    return Reviewer()


@st.cache_resource(show_spinner=False)
def cameras():
    return CameraManager(reviewer())


def results_view(result):
    report = result["report"] if result else None
    st.subheader("Model signals")
    st.caption("Five anomaly objectives · MobileCLIP2-S2 · CPU")
    for task in TASKS:
        value = report["summary"][task] if report else None
        decision = value["decision"] if value else "waiting"
        label = {
            "waiting": "Awaiting review",
            "normal": "No signal",
            "anomaly": "Review suggested",
            "unknown": "More evidence needed",
        }[decision]
        st.markdown(
            f'<div class="signal-row"><span class="signal-name">{LABELS[task]}</span>'
            f'<span class="badge {decision}">{label}</span></div>',
            unsafe_allow_html=True,
        )
    if not report:
        st.caption("Results appear here after you analyze a file.")
        return
    if report["kind"] == "live":
        st.caption(
            f"{len(report['events'])} recent windows · {report['observed_seconds']:.1f}s observed"
        )
        age = report.get("result_age_seconds")
        st.caption(
            f"Most recent screened window · evidence age {age:.1f}s"
            if age is not None
            else "Gathering evidence for the first scene review…"
        )
    else:
        st.caption(
            f"{len(report['events'])} evidence windows · {report['processing_seconds']:.1f}s processing"
        )
    st.caption(
        "Signals require review. No signal applies only to scored windows; it does not certify the whole recording."
    )
    if report["kind"] == "image":
        st.info(
            "A still image cannot establish littering, loitering or unattended-item duration. Upload a video for temporal evidence."
        )
    elif report["observed_seconds"] < 19.5:
        st.info(
            "Loitering and unattended items need about 20 seconds of observed history."
        )
    if any(v["unsupported_windows"] for v in report["summary"].values()):
        st.caption(
            "Some windows lacked enough evidence. Their counts and reasons are included in the details."
        )
    if report["kind"] == "live":
        if st.button("Prepare results download", width="stretch"):
            st.session_state["camera_export"] = (
                result["run_id"],
                json.dumps(report, indent=2, allow_nan=False),
            )
        export = st.session_state.get("camera_export")
        if not export or export[0] != result["run_id"]:
            return
        data = export[1]
    else:
        data = json.dumps(report, indent=2, allow_nan=False)
    st.download_button(
        "Download results",
        data,
        file_name="scene_review.json",
        mime="application/json",
        width="stretch",
        on_click="ignore",
    )


@st.fragment(run_every=1)
def live_results(result):
    if result:
        result = reviewer().refresh(result)
        st.session_state["result"] = result
    results_view(result)
    if not result:
        return
    review_states(result["report"])


def review_states(report):
    if report["screening_failed"]:
        st.warning(
            f"{report['screening_failed']} windows were not screened. See the event details."
        )
    if not report["gemma_enabled"]:
        st.caption(
            "Background Gemma review is disabled. Configure GEMMA_BASE_URL to enable it."
        )
        return
    st.subheader("Gemma confirmation")
    st.caption(
        f"{report['confirmation_pending']} pending · {report['confirmation_failed']} unavailable"
    )
    rows = []
    for event in report["events"]:
        for task, value in event.get("confirmation", {}).items():
            rows.append(
                {
                    "Time": round(event["timestamp_s"], 2),
                    "Objective": LABELS[task],
                    "Review": value["decision"],
                    "Evidence": value.get("evidence", ""),
                    "Agreement": not value.get("review_required", True),
                }
            )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Provisional signals remain visible while Gemma reviews their evidence. Review results do not dispatch external alerts."
    )


@st.fragment(run_every=0.5)
def camera_panel():
    session = st.session_state.get("camera_session")
    active = session is not None and session.active
    st.caption(
        "Uses a built-in or USB camera attached to the machine running Streamlit. "
        "Camera 0 is usually the default. A remote browser uses the server’s camera."
    )
    device, cadence = st.columns(2)
    index = device.number_input(
        "Camera index",
        min_value=0,
        max_value=15,
        value=0,
        step=1,
        disabled=active,
        help="Try 1 or 2 if the desired USB camera is not camera 0.",
    )
    seconds = cadence.number_input(
        "Scene review interval (seconds)",
        min_value=2,
        max_value=30,
        value=3,
        disabled=active,
        help="Tracking selects up to 5 frames/second. Model inference adds processing time.",
    )
    start, stop, clear, _ = st.columns([1.3, 1, 1, 3])
    if start.button("Start camera", type="primary", disabled=active, width="stretch"):
        try:
            with st.spinner("Preparing the local models…"):
                session = cameras().start(int(index), scan_seconds=int(seconds))
                st.session_state["camera_session"] = session
        except Exception as exc:
            st.error(str(exc))
    if stop.button("Stop camera", disabled=session is None, width="stretch"):
        session.stop()
    if clear.button("Clear camera", disabled=session is None, width="stretch"):
        session.release()
        st.session_state.pop("camera_session", None)
        session = None
    if session is None:
        st.info(
            "Choose your camera and select Start camera. The preview and results update automatically."
        )
        return
    try:
        result = session.snapshot()
    except Exception as exc:
        session.stop()
        st.error(f"Live results could not be read: {exc}")
        return
    state = result["state"]
    if state == "error":
        st.error(result["error"])
    elif state == "starting":
        st.info("Opening the camera and preparing tracking…")
    elif state == "stopping":
        st.info(
            "Stopping capture and finishing in-flight analysis; provisional results remain available."
        )
    elif state in {"stopped", "expired"}:
        st.info(result["stop_reason"] or "Camera stopped")
    preview, signals = st.columns([1.35, 1], gap="large")
    with preview, st.container(border=True):
        st.subheader("Live camera")
        if result["frame"]:
            st.image(result["frame"], width="stretch")
            report = result["report"]
            width, height = result["frame_size"]
            st.caption(
                f"{width} × {height} · {report['captured_frames']} captured · "
                f"{report['selected_frames']} tracked · "
                f"{report['tracking_lag_seconds']:.1f}s tracking lag"
            )
            st.caption(
                "Live preview is separate from inference. Proposed boxes appear in the evidence frames below."
            )
        else:
            st.caption("Waiting for the first camera frame…")
    with signals, st.container(border=True):
        results_view(result)
        review_states(result["report"])
    events = result["report"]["events"]
    if events:
        with st.expander("Recent camera evidence"):
            chosen = next(
                (i for i in reversed(range(len(events))) if result["previews"][i]), None
            )
            if chosen is not None:
                st.image(result["previews"][chosen], width="stretch")
            st.json(events[-1], expanded=False)
    st.caption(
        "Recent evidence is temporary; no full video is recorded. "
        "Stop releases the camera while pending analysis finishes. "
        "Closing or leaving this live view stops capture after about 20 seconds."
    )


def main():
    st.markdown(
        '<div class="eyebrow">SCENE / LOCAL VISION</div>', unsafe_allow_html=True
    )
    st.title("A closer look.")
    st.markdown(
        '<div class="subtitle">Review a file or watch a live camera.</div>',
        unsafe_allow_html=True,
    )
    source = st.radio("Input source", ["Upload a file", "Live camera"], horizontal=True)
    if source == "Live camera":
        camera_panel()
        return
    camera = st.session_state.pop("camera_session", None)
    if camera is not None:
        camera.release()
    uploaded = st.file_uploader(
        "Drop your file here",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
            "m4v",
        ],
        max_upload_size=200,
        key=f"media_{st.session_state.get('upload_version', 0)}",
        help="One image or video, up to 200 MB. Files are processed on this machine.",
    )
    data = uploaded.getvalue() if uploaded else None
    fingerprint = (
        hashlib.sha256(uploaded.name.encode() + b"\0" + data).hexdigest()
        if data
        else None
    )
    if st.session_state.get("file_id") != fingerprint:
        st.session_state["file_id"] = fingerprint
        old = st.session_state.pop("result", None)
        if old:
            reviewer().release(old["run_id"])
    run, clear, _ = st.columns([1.3, 1, 4])
    analyze = run.button(
        "Analyze file", type="primary", disabled=not data, width="stretch"
    )
    if clear.button("Clear", disabled=uploaded is None, width="stretch"):
        st.session_state["upload_version"] = (
            st.session_state.get("upload_version", 0) + 1
        )
        old = st.session_state.pop("result", None)
        if old:
            reviewer().release(old["run_id"])
        st.rerun()
    if analyze:
        old = st.session_state.pop("result", None)
        if old:
            reviewer().release(old["run_id"])
        with st.status("Loading the local model…", expanded=True) as status:
            bar = st.progress(0)

            def progress(fraction, message):
                status.update(label=message)
                if fraction is not None:
                    bar.progress(max(0.0, min(1.0, fraction)))

            try:
                st.session_state["result"] = reviewer().analyze(
                    data, uploaded.name, progress
                )
                status.update(label="Review complete", state="complete", expanded=False)
            except Exception as exc:
                logging.getLogger(__name__).exception("Upload review failed")
                status.update(
                    label="The file could not be reviewed",
                    state="error",
                    expanded=False,
                )
                st.error(
                    "Check that the file is readable and the trained model files are installed. No result was recorded for this run."
                )
                with st.expander("Error details"):
                    st.text(str(exc))
    result = st.session_state.get("result")
    preview, signals = st.columns([1.35, 1], gap="large")
    with preview, st.container(border=True):
        st.subheader("Your evidence")
        if uploaded:
            st.caption(uploaded.name)
            if Path(uploaded.name).suffix.lower() in IMAGE_TYPES:
                try:
                    from PIL import Image, ImageOps

                    with Image.open(io.BytesIO(data)) as image:
                        st.image(
                            ImageOps.exif_transpose(image).convert("RGB"),
                            width="stretch",
                        )
                except (OSError, ValueError, Image.DecompressionBombError):
                    st.error("This file could not be decoded as an image.")
            else:
                st.video(data, format=uploaded.type or "video/mp4")
                st.caption(
                    "Video preview depends on browser codec support. Analysis uses the original file."
                )
        else:
            st.markdown(
                "#### Start with a scene\n\nUpload an image for a visual check, or a video to review activity over time."
            )
            st.caption("JPG, PNG, WEBP, BMP · MP4, MOV, AVI, MKV, WEBM")
    with signals, st.container(border=True):
        live_results(result)
    if result:
        report = result["report"]
        with st.expander("Evidence & scores"):
            chosen = st.selectbox(
                "Evidence window",
                range(len(report["events"])),
                format_func=lambda i: (
                    f"{report['events'][i]['timestamp_s']:.1f}s · {report['events'][i]['kind'].replace('_', ' ')}"
                ),
            )
            event = report["events"][chosen]
            image, detail = st.columns([1.15, 1])
            image.image(result["previews"][chosen], width="stretch")
            detail.caption(
                "Boxes mark proposed regions, not confirmed anomalies or ownership."
            )
            detail.dataframe(
                [
                    {
                        "Objective": LABELS[t],
                        "Decision": p["decision"],
                        "Score": p.get("score"),
                        "Threshold": p.get("threshold"),
                    }
                    for t, p in event["predictions"].items()
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Scores are raw classifier outputs, not probabilities. A higher score than its task threshold suggests review."
            )
            st.json({"summary": report["summary"], "window": event}, expanded=False)
    st.markdown(
        '<div class="footnote">Local research prototype · Uploads use temporary storage · Automatic alerts are disabled</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
