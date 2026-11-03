# CLIP/VLM anomaly evaluation

Evaluate smoking, littering, loitering, apparent public sleeping and unattended items with source-grouped data. The code supports MobileCLIP2-S2, SigLIP2, SmolVLM, MiniCPM-V, supervised VADCLIP and five Gemma 4 variants. Accuracy experiments use the available GPU without pacing.

## Setup

Use Python 3.12, `ffmpeg` and `ffprobe`. Create an environment, install the PyTorch build appropriate for your CPU/GPU in that environment, then install the base requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install your selected CPU/CUDA PyTorch build here.
python -m pip install -r requirements/base.txt
python -m pip install -e . --no-deps
python -m anomaly_eval --help
```

Gemma uses a separate runtime because its Transformers version differs. Follow [Gemma setup and reproduction](docs/gemma4_evaluation.md), using `requirements/gemma4.txt`. SmolVLM adaptation dependencies are in `requirements/smol-training.txt`.

## Main workflows

```bash
# Inspect sources/licenses and the complete acquisition plan.
python -m anomaly_eval fetch-all --list
python -m anomaly_eval fetch-all --plan
# Fetch with concurrent, resumable and integrity-checked transfers.
python -m anomaly_eval fetch-all --workers 8 --per-host 3
# Prepare the shared benchmark and run accuracy evaluation.
python -m anomaly_eval run --prepare-only
python -m anomaly_eval run --device auto
# Train or use the supervised CLIP classifier bundle.
python -m anomaly_eval train-clip --help
python -m anomaly_eval predict-clip --help
# Gemma tools share one command; this launcher selects the isolated GPU environment.
bash scripts/gemma4_gpu.sh --help
bash scripts/gemma4_gpu.sh suite
bash scripts/gemma4_gpu.sh optimize
```

The CLI also works as `anomaly-eval` after installation. [evaluate_gpu.sh](scripts/evaluate_gpu.sh) selects the existing CUDA environment and matching local driver libraries. [smolvlm_gpu.sh](scripts/smolvlm_gpu.sh) selects the isolated adaptation dependencies. From another working directory, use the installed CLI or the [generic checkout launcher](scripts/evaluate.py).

[Server setup](docs/server.md) describes independent data, results and model-cache locations, environment variables, and `config/server.example.json`. `python -m anomaly_eval paths` prints the configured locations. Historical experiments may retain checkout-relative paths; their provenance and limitations are recorded in their guides.

## Workspace layout

| Location | Contents |
|---|---|
| `anomaly_eval/` | Reusable evaluation, data, training and Gemma modules; packaged CLI |
| `scripts/` | Research workflows grouped into datasets, clip, strategy, video, smolvlm, generalization, expanded and reporting |
| `requirements/` | Separate base, Gemma and SmolVLM dependency sets |
| `docs/` | Current usage, methodology, dataset links/licenses and experiment history |
| `reports/` | Concise findings, metric tables, editable paper/slides and finished publication artifacts |
| `tests/` | Regression, data-integrity and protocol tests |
| `test_data/` | Legacy example bundle and task-specific manifests; partial labels |
| `data/`, `results/` | Ignored local corpora/weights/caches and complete experiment records |

[Workspace organization](docs/workspace.md) explains cleanup boundaries and relocation. [Script index](scripts/README.md) lists current entry points; [locations.json](scripts/locations.json) maps historical paths to current files. Dated report contents and frozen raw-run implementations retain their original names.

## Results and methodology

The [completed Gemma comparison](reports/gemma4_evaluation_20261027/report.md) covers E2B, E4B, 12B, 26B-A4B and 31B, with 4,032 held-out cases per variant, source-level operating metrics and matched incumbent comparisons. The [completed second-stage comparison](reports/gemma4_optimization_20261028/report.md) covers all five variants with added detail, contrastive evidence, automatic crops and development-only selection. [Reproduction and findings](docs/gemma4_evaluation.md#second-stage-optimization) include operating-point comparisons and group uncertainty. Original results remain available.

The [experiment history](docs/experiment_history.md) links the earlier zero-shot, supervised CLIP, strategy, temporal, event, SmolVLM and generalization studies, with their papers and slides. The [evaluation guide](docs/evaluation.md), [training guide](docs/training.md), and [dataset registry](docs/dataset_sources.md) document data preparation, operational definitions, source terms and metric interpretation.

Historical holdouts have been inspected repeatedly. Weak/provisional labels, small camera/author groups and missing duration/owner evidence limit deployment conclusions. Binary AP measures ranking; clip false-positive rate does not measure false alerts per hour. Full detector/tracker event validation remains a separate workflow.

## Checks

```bash
python -m unittest discover -s tests -v
ruff check anomaly_eval scripts tests
ruff format --check anomaly_eval scripts tests
```
