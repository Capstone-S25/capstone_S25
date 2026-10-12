# CPU anomaly-model evaluation

Compare MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M and MiniCPM-V 4.0 on shared, reviewed task data. VadCLIP is a separately supervised offline smoking study. The current evaluation, data utilities and reporting use one CLI: `anomaly-eval`.

Use Python 3.12 and install `ffmpeg`/`ffprobe` for video preparation. From a checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The CPU wheel index avoids CUDA dependencies. `anomaly-eval` and `python -m anomaly_eval` work from any directory after installation. Without installation, use `python /path/to/checkout/scripts/evaluate.py` with the same arguments.

```bash
anomaly-eval run --prepare-only
anomaly-eval run --mode quick --threads 8
anomaly-eval run --threads 4 --max-rss-mb 24576 --camera-fps 1
# Include the supervised smoking adapter when the original COCO corpus is available:
anomaly-eval run --vadclip
```

Full evaluation and quick mode run without pacing sleeps. Quick mode uses at most
8 inputs per task/split and frame inputs by default. `--mode deployment` enables
camera-rate pacing on a smaller cohort (16 inputs per task/split by default).
Override these with `--limit`; see the evaluation guide for full settings and
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

See [the evaluation guide](docs/evaluation.md) for data fetching, reviewed-label imports, metric definitions and all commands; [server setup and migration](docs/server.md) covers relocated datasets/caches and old commands. [The corrected report](reports/anomaly_benchmark_report.md) documents remaining data limitations. The historical bundle is development only, and confirmed held-out coverage for all five anomalies is still incomplete.
