# MobileCLIP2-S2 CPU transfer bundle

This research prototype bundle contains the current runtime source, YOLO26s detection and pose
weights, ByteTrack integration, trained five-task MobileCLIP2-S2 strategies,
Faster R-CNN reference weights, frozen thresholds and a pinned offline base-model
cache. It preserves the required project-relative paths. No clone, training data,
Hugging Face login or model download is required to run it after dependencies are
installed. Python libraries and operating-system packages are not bundled.
Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement. The base and fine-tuned derivatives
retain its research-use terms; license texts and modification details are in
`docs/cpu_model_sources.md` and `docs/licenses/`.

The target is Linux x86-64, Python 3.12, one camera and 32 GB RAM. This packages
the existing implementation; **the two-second processing/alert target has not
been achieved**. Classifier-only results remain review candidates and do not
dispatch independently verified alerts. Gemma is omitted from this CPU profile.

## Transfer and install

Copy the `.tar.gz` file and its `.tar.gz.sha256` sidecar to the target. From the
directory containing them, substitute the actual archive filename:

```bash
sha256sum -c capstone-mobileclip2-s2-cpu-20261103.tar.gz.sha256
tar -xzf capstone-mobileclip2-s2-cpu-20261103.tar.gz
cd capstone-mobileclip2-s2-cpu
python3.12 scripts/deployment/cpu.py verify
```

On Ubuntu, install Python 3.12 with venv support, FFmpeg, `libgl1` and
`libglib2.0-0` using the system package manager. Then create the runtime:

```bash
python3.12 scripts/deployment/cpu.py setup
.venv-cpu/bin/python scripts/deployment/cpu.py smoke
```

For the optional image/video upload interface, use `setup --with-ui` in place
of `setup`, then run `.venv-cpu/bin/python -m streamlit run app.py`. See the
[interface guide](inference_ui.md) for evidence limits and result interpretation.

`setup` requires internet access to install Python packages. It installs the
official CPU-only PyTorch 2.8.0 / torchvision 0.23.0 wheels, then the pinned direct
dependencies in `requirements/cpu-deployment.txt`. It refuses to overwrite an
existing `.venv-cpu`. Transitive packages are resolved by pip; the direct pins
are not a complete wheel lock. To deploy to an air-gapped machine, separately
prepare a compatible Linux/Python 3.12 wheelhouse. Do not copy this workstation's
virtual environments or CUDA libraries.

`verify` checks the manifest, hashes, sizes, training-definition match and
completion receipt using only the standard library. `smoke` loads both YOLO
models, exercises tracking, and runs all five classifier tasks on synthetic
images. A successful smoke test checks installation and weight loading, not
accuracy or latency on camera footage. The outer archive checksum and internal
`SHA256SUMS` are also provided for standard tools.

## Run a video

```bash
.venv-cpu/bin/python scripts/deployment/cpu.py video /path/to/video.mp4 \
  --out results/camera_test
```

The launcher forces both detector and classifier onto the CPU, sets a
bundle-relative offline Hugging Face cache and uses the active Python
interpreter for the resident classifier worker. It does not depend on the
original `.venv-gpu` path, the current directory or the original user's home.
The direct-video path requires no database. It processes files without pacing;
queue delays from such a run are not live-camera latency measurements.

## Run RTSP with PostgreSQL scheduling

Start a PostgreSQL instance and set `DATABASE_URL` to its SQLAlchemy psycopg URL.
The database service itself is not included. For a fresh intended deployment
database, apply the included migration:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE'
(cd data-pipeline && ../.venv-cpu/bin/python -m alembic upgrade head)
.venv-cpu/bin/python scripts/deployment/cpu.py live \
  --rtsp-url 'rtsp://CAMERA/STREAM' --out results/camera_01
```

Keep credentials in the destination environment. The recorder targets two-second
segments, but source keyframes can make them longer. This segmentation delay,
the default ten-second scene-review cadence and inference work all affect alert
delay. Loitering and unattended-item checks additionally require temporal
history; missing ownership evidence must remain unknown. An actual RTSP camera
and the destination CPU must still be measured before a real-time claim.

Candidate JSONL, model-worker logs, queue/service measurements and summaries are
written beneath `--out`. Do not interpret an `unknown`, rejected or failed
request as a normal scene. Neither the CPU launcher nor the underlying
classifier changes trained thresholds or automatically sends external alerts.

## Rebuild from a development checkout

Commit the runtime changes first, then run:

```bash
python3.12 scripts/deployment/cpu.py build \
  --artifact-root /path/to/checkout-with-local-weights \
  --hf-cache "$HOME/.cache/huggingface/hub" \
  --output dist/capstone-mobileclip2-s2-cpu-20261103.tar.gz
```

The builder uses a committed runtime allowlist and only the required model
files. It dereferences the seven pinned Hugging Face snapshot files into regular
files, records SHA-256 and source commit, verifies the staged bundle, and writes
the archive via a temporary filename. Missing inputs and existing output names
fail explicitly. Datasets, environments, credentials, Git metadata, Gemma,
other model caches and research outputs are excluded. Generated archives belong
under ignored `dist/`, not in Git. See `docs/cpu_model_sources.md` for upstream
sources and license references; `bundle_manifest.json` is the exact inventory.
