# CLIP/VLM anomaly-model evaluation

Compare MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M and MiniCPM-V 4.0 on shared, reviewed task data. VadCLIP is a separately supervised offline smoking study. The current evaluation, data utilities and reporting use one CLI: `anomaly-eval`.

Use Python 3.12 and install `ffmpeg`/`ffprobe` for video preparation. From a checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The CPU wheel index is for CPU deployments. For GPU accuracy evaluation, install a CUDA-enabled PyTorch build compatible with the server driver instead. `anomaly-eval` and `python -m anomaly_eval` work from any directory after installation. Without installation, use `python /path/to/checkout/scripts/evaluate.py` with the same arguments.

```bash
anomaly-eval run --prepare-only
anomaly-eval run --mode quick
# Accuracy: GPU when available, all confirmed data, no pacing or default RAM cap:
anomaly-eval run --device auto
# Explicit GPU request fails clearly if CUDA is unavailable:
anomaly-eval run --device cuda --batch-size 32
# Separate CPU latency replay with fixed resources and batch size one:
anomaly-eval run --mode deployment --device cpu --threads 4 --max-rss-mb 24576 --camera-fps 1 --repeats 3
# Include the supervised smoking adapter when the original COCO corpus is available:
anomaly-eval run --vadclip
```

Full evaluation and quick mode run without pacing sleeps. Accuracy defaults to
available CPU threads, automatic GPU selection, GPU CLIP batches of 32 (CPU: 1),
and no host RAM cap. Generative VLMs run one input at a time. CPU uses float32;
CUDA generative models use bfloat16 when supported, otherwise float16. Override
with `--device`, `--dtype`, `--threads` and `--batch-size`. Full preparation retains
all confirmed examples; `--per-group N` explicitly caps each recording/label.
Use `--workers N` to run isolated models concurrently when RAM/VRAM allows;
`--threads` is per worker. Deployment mode requires one worker and batch size one.

This workspace also has a separate `.venv-gpu` and verified local NVIDIA libraries
for the host's driver/library mismatch. [evaluate_gpu.sh](scripts/evaluate_gpu.sh)
launches that environment without altering the system driver or CPU environment:
`scripts/evaluate_gpu.sh run --device cuda`. It checks the loaded kernel version
before selecting local driver libraries. Allow roughly 9–10 GiB of free VRAM for
MiniCPM in this configuration; its completed CUDA run peaked at 8,116 MiB reserved.
Quick mode uses at most
8 inputs per task/split and frame inputs by default. `--mode deployment` enables
camera-rate pacing on a smaller cohort (16 inputs per task/split by default).
Override these with `--limit`; `--skip-prepare` retains any earlier preparation
cap, which is recorded in `dataset.json`. See the evaluation guide for full settings and
the current five-task coverage gaps.

If prepared data was created before the source/pixel duplicate correction, run
without `--skip-prepare` to rebuild it. Older prepared schemas are rejected.

For another server, configure independent data, results and cache roots. From the
checkout, this example uses your existing `data/` and directories owned by your user:

```bash
python -m anomaly_eval --data-root "$PWD/data" \
  --results-root "$PWD/results" \
  --cache-root "$HOME/.cache/huggingface" run --threads 4 --max-rss-mb 24576
```

Point `--data-root` at the directory containing your actual `raw/` and
`additional/` data. `/srv/...`, `/mnt/...` and `/scratch/...` paths in deployment
examples are placeholders for locations already provisioned on your server.
Run inside the activated virtual environment without `sudo`.

Alternatively, copy [server.example.json](config/server.example.json) to `config/server.json`, edit its locations and use `anomaly-eval --workspace-config config/server.json run`. Relative paths in that JSON resolve against the JSON file. Environment variables `CAPSTONE_DATA_DIR`, `CAPSTONE_RESULTS_DIR`, `CAPSTONE_CACHE_DIR`, `CAPSTONE_BUNDLE_DIR` and `CAPSTONE_CONFIG_DIR` are supported. `anomaly-eval paths` shows the resolved locations.

The implementation lives in `anomaly_eval/`: shared manifests/provenance, datasets, models, resource controls, analysis and the temporal adapter. [evaluation.ipynb](notebooks/evaluation.ipynb) is the single notebook front end. New manifests and caches use portable paths; machine locations remain diagnostic metadata rather than experiment identities.

The [expanded accuracy evaluation](reports/expanded_accuracy_review_20261014.md)
adds 55,156 GPU predictions and increases unattended-item coverage from 36 to
1,770 endpoints. It also attempts littering, loitering and rough sleeping as
explicitly qualified diagnostics. The [updated slide deck](reports/slides/model_evaluation_expanded_20261014.pdf)
covers the findings and methodology without an additional latency experiment.

See [the evaluation guide](docs/evaluation.md) for data fetching, reviewed-label imports, metric definitions and all commands; [server setup and migration](docs/server.md) covers relocated datasets/caches and old commands. [The original workflow review](reports/workflow_review_20261013.md) records the preceding analysis and measurements. The historical bundle is development only, and confirmed held-out coverage for all five anomalies is still incomplete.
