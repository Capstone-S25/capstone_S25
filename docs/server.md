# Server setup and codebase layout

The executable scripts have been consolidated into one CLI, with cohesive implementation modules under `anomaly_eval/`. Obsolete balanced/all-test dataset builders and the incompatible external VadCLIP wrapper are removed. Statistical calibration/bootstrap/plotting/report helpers share `analysis.py`; inventory/review/export share `datasets/tools.py`; manifests, Drive rename lookup and provenance share `data.py`. The two duplicate notebooks are replaced by `notebooks/evaluation.ipynb`.

## Independent runtime locations

Install the package in the server's Python environment as described in README. Use `anomaly-eval`, `python -m anomaly_eval`, or the checkout launcher `python /path/to/checkout/scripts/evaluate.py`. The launcher and child model workers do not require the repository as their working directory.

| Flag before the command | Environment variable | Contents |
|---|---|---|
| `--data-root` | `CAPSTONE_DATA_DIR` | Raw datasets, additional sources, prepared manifests/images |
| `--results-root` | `CAPSTONE_RESULTS_DIR` | Fresh experiment directories, reports, logs and audits |
| `--cache-root` | `CAPSTONE_CACHE_DIR` | Hugging Face `hub/`, official VadCLIP encoder weights and reusable features |
| `--bundle-root` | `CAPSTONE_BUNDLE_DIR` | Portable `test_data/` bundle; copy it or use the checkout's copy |
| `--config-root` | `CAPSTONE_CONFIG_DIR` | `anomalies.json` and `caviar_review_intervals.json` |

Defaults locate checkout data/results/bundle dynamically and use the user's Hugging Face cache. Packaged defaults live in `anomaly_eval/config/` and are included in a wheel. Configuration precedence is explicit flags, server JSON, environment, then defaults. Existing `HF_HOME`/`HF_HUB_CACHE` are respected unless a cache root is configured.

```bash
cp config/server.example.json config/server.json
# Edit paths, then:
anomaly-eval --workspace-config config/server.json paths
anomaly-eval --workspace-config config/server.json run --prepare-only
anomaly-eval --workspace-config config/server.json run --vadclip --threads 4
```

The JSON supports `data_dir`, `results_dir`, `cache_dir`, `bundle_dir` and `config_dir`. Relative locations are relative to the configuration file, not the launch directory. `~` and defined environment variables expand. `config/server.json` is ignored by Git so each server can keep its own configuration.

For a read-only shared dataset mount, place prepared data on writable storage:

```bash
anomaly-eval --data-root /mnt/datasets/capstone \
  --results-root /scratch/capstone/runs --cache-root /scratch/capstone/cache \
  run --prepared-dir /scratch/capstone/prepared --threads 4
```

`--output` chooses one new run directory, while `--results-root` chooses the parent for default timestamped runs. The retained `run --data-dir` flag overrides the data root for that invocation; `run --cache-dir` selects an explicit Hugging Face hub cache. Prefer the global roots for a consistent workspace across all commands.

## Move data and computation to another server

Copy or mount the raw data, additional sources and prepared data under the configured data root. Copy the portable bundle separately if using its historical/CAVIAR examples. Keep the same source layout, image content, reviewed labels and group IDs. Copy the cache root if you want to reuse weights and source-validated VadCLIP features; results can be copied independently for analysis. Dataset archives may be large, so transfer them separately from the code checkout.

New `dataset.json` entries name manifests relative to that file. Manifest image paths are relative to their manifest or anchored as `${DATA_ROOT}/…` and `${BUNDLE_ROOT}/…`. The loader resolves anchors from the server configuration. Source origins use dataset identifiers such as `caviar://…` and `drive://…`, rather than workstation filenames. Images retain SHA-256 validation after relocation.

Prediction identifiers and source/cache fingerprints do not include configured physical roots. Output, data, manifest and cache locations are excluded from experiment identity; model settings, labels, source bytes and code remain included. Actual resolved roots are still recorded as diagnostic provenance. Feature NPZ filenames remain relative to their cache manifest. A valid matching feature cache skips encoder computation; checkpoints/results stay separate per experiment.

Old JSON/CSV records can contain absolute paths. Apply explicit mappings when reopening them, for example:

```bash
anomaly-eval --path-map /old/server/capstone=/new/server/capstone \
  benchmark --manifest /new/server/capstone/data/manifest.csv --models mobileclip2-s2
```

Windows paths can also be mapped on Linux using a quoted `OLD_ROOT=NEW_ROOT` argument. A path map relocates only the given prefix; it does not guess which file a missing old path referred to. For old generated manifests with complex relative paths, rerun `prepare` with the new roots to generate portable manifests. Old invalid metric/cache schemas still require rerunning the corrected evaluation, as documented in the guide.

`anomaly-eval analyze --results-dir /copied/run/test --dev-dir /copied/run/dev` reads saved predictions and metrics without model weights or original images. New run summaries refer to their cohort directories relative to the run directory. Historical audit/report records are kept under `reports/archive/`; they intentionally preserve their original machine-specific evidence and are never runtime defaults.

## Command migration

| Previous entry point | Current command |
|---|---|
| `scripts/run_evaluation.py` | `anomaly-eval run` |
| `scripts/fetch_data.py` | `anomaly-eval fetch-drive` |
| `scripts/fetch_additional_data.py` | `anomaly-eval fetch` |
| `scripts/prepare_evaluation.py`, legacy manifest builders | `anomaly-eval prepare` |
| `scripts/benchmark.py` | `anomaly-eval benchmark` |
| Bootstrap, calibration and report-figure scripts | `anomaly-eval analyze` (`--dev-dir` enables calibration) |
| `scripts/init.py` | `anomaly-eval inventory` |
| `scripts/review_candidates.py` | `anomaly-eval review` |
| `scripts/make_test_data.py` | `anomaly-eval export` |
| VadCLIP module/wrapper commands | `anomaly-eval vadclip extract/train/evaluate/baselines` |
| `scripts/audit_evaluations.py` | `anomaly-eval audit` |

Use `COMMAND --help` for options. Internal Python imports now use `anomaly_eval`, rather than `scripts`. Active task definitions/review intervals moved to `anomaly_eval/config/`. The single notebook uses the installed package and can receive `CAPSTONE_WORKSPACE_CONFIG` in its kernel environment; use a kernel from the environment where the package was installed.
