# Local image and video review

The Streamlit interface uses the trained MobileCLIP2-S2 CPU classifier. Videos
pass through the current YOLO26s / pose / ByteTrack gate, ten-second scene
reviews and a final scene review so short uploads are still examined. It runs
without PostgreSQL. The classifier is cached, while each video gets fresh
tracking state. A lock serializes use of the mutable trained backbone.

From a checkout with the [deployment artifacts](cpu_deployment.md) installed:

```bash
PATH_TO_CPU_PYTHON -m pip install -r requirements/ui.txt
PATH_TO_CPU_PYTHON -m streamlit run app.py
```

For an extracted bundle built with the new UI files, install using
`python3.12 scripts/deployment/cpu.py setup --with-ui`, then launch with
`.venv-cpu/bin/python -m streamlit run app.py`. Older dated archives do not
contain the UI; rebuild from the updated committed source to include it.

Open `http://localhost:8501`, drop an image or video up to 200 MB, and click
**Analyze file**. The model loads on the first run and stays resident. Review
the five signals, inspect each evidence window and its raw scores, and download
the results as JSON. **Clear** removes the upload and result from the session.
The application binds to localhost by default; remote IDE users can forward
port 8501. It does not require a GPU or Gemma.

An image can be scored for smoking and apparent public sleeping. Littering,
loitering and unattended items remain unknown without temporal evidence. Videos
need eight distinct action views for littering and around 20 seconds of history
for duration objectives. The final review uses actual past frames and timestamps;
stills are never repeated to manufacture history. A single short clip is not
forced to produce all five binary decisions.

**Review suggested** means a classifier output crossed its frozen task
threshold. **No signal** refers only to windows that were successfully scored.
Unsupported windows and their reasons remain visible in the details and JSON.
Scores are uncalibrated outputs, not confidence percentages. Proposed boxes do
not verify the responsible person or item ownership. Automated alerts remain
disabled. This interface does not establish a two-second CPU response guarantee.

Uploads and intermediate evidence use per-request temporary directories that
are removed after processing, including failures. Previews/results remain in
Streamlit session memory until cleared or the session ends. No uploaded files
or model results are added to the repository. Video playback depends on the
browser's codec support; inference decodes the original file using PyAV.

The model files retain the research-use and upstream license terms described
in [cpu_model_sources.md](cpu_model_sources.md).
