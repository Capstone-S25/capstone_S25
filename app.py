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

from anomaly_eval.review import IMAGE_TYPES, LABELS, TASKS, Reviewer

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
    st.download_button(
        "Download results",
        json.dumps(report, indent=2, allow_nan=False),
        file_name="scene_review.json",
        mime="application/json",
        width="stretch",
    )


def main():
    st.markdown(
        '<div class="eyebrow">SCENE / LOCAL VISION</div>', unsafe_allow_html=True
    )
    st.title("A closer look.")
    st.markdown(
        '<div class="subtitle">Drop an image or video. Review the model’s signals.</div>',
        unsafe_allow_html=True,
    )
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
        st.session_state.pop("result", None)
    run, clear, _ = st.columns([1.3, 1, 4])
    analyze = run.button(
        "Analyze file", type="primary", disabled=not data, width="stretch"
    )
    if clear.button("Clear", disabled=uploaded is None, width="stretch"):
        st.session_state["upload_version"] = (
            st.session_state.get("upload_version", 0) + 1
        )
        st.session_state.pop("result", None)
        st.rerun()
    if analyze:
        st.session_state.pop("result", None)
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
        results_view(result)
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
