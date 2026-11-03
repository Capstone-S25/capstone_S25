# Workspace organization

The repository separates reusable code, grouped research workflows, documentation/publication outputs and local experiment storage. The top-level [README](../README.md) and [script index](../scripts/README.md) are the entry points.

On 28 October, 76 research entry points were organized into eight folders. One source-pilot feature adapter was folded into the shared strategy extractor through `--data-dir`, leaving 75 research script files and avoiding mutation of another module's globals. Five Gemma script implementations moved into its reusable package, with checkpoint acquisition and runtime installation combined in `fetch.py`. The evaluation/report wrappers and duplicate all-dataset wrapper were removed. The Gemma package now exposes one dispatcher for fetch, parallel download, preflight, evaluation, suite, optimization, reporting, uncertainty and audit. Python imports, subprocess paths, GPU launchers, active guides and CI dependency paths were updated.

The dependency sets are `requirements/base.txt`, `requirements/gemma4.txt` and `requirements/smol-training.txt`. Their runtimes remain separate. Do not install the newer Gemma Transformers stack into an older experiment environment merely to reuse its launcher.

The longer chronology formerly in the root README is retained in [experiment_history.md](experiment_history.md). The early progress note is archived under `reports/archive/initial_progress_20261011.md`. Published papers/slides, tables and original raw-run snapshots remain available.

Cleanup removed the duplicate `build/` tree, source/test bytecode caches and rebuildable paper compiler sidecars/logs: 29 paths totaling about 4.6 MiB. The local receipt is `results/workspace_cleanup_20261028/cleanup.json`. This is code/workspace cleanup; the large local datasets, weights and completed experiment caches remain available for reproduction and ongoing runs.

## Git boundaries

The [.gitignore](../.gitignore) excludes local `data/`, `results/`, model/cache folders, virtual environments, credentials and server-specific configuration. It also excludes Python/build caches, notebook checkpoints, compiler intermediates, logs, temporary download/editor files and future preview/raw-report directories. Concise reports, metric/protocol/audit JSONs, editable LaTeX/slides, completed PDFs/decks, example configurations, code and dependency files remain eligible for version control.

Git ignore rules apply to untracked files. Existing tracked publication preview images and historical validation logs remain tracked where they are part of a published evidence package; changing the ignore patterns does not remove them from history. Rebuildable compiler files removed from the working tree appear as deletions for the next commit. No Git history rewrite or automated commit was performed.

`test_data/` is a legacy example bundle with partial task labels. Its manifests are retained because users and portability tests reference them. Production research acquisitions and future large generated media belong in the ignored `data/` tree.

## Existing experiments

Dated `results/` directories bind inputs and scorer implementations by hashes. Their saved implementation snapshots retain the original source paths. Moving current workflow files does not rewrite those receipts or claim that a new implementation produced old scores. [The relocation index](../scripts/locations.json) maps old script names to current code. When changing a frozen scorer or protocol, use a new output directory and document the new comparison.

Gemma's second-stage study uses `results/gemma4_optimization_20261028/` and preserves `results/gemma4_evaluation_20261027/`. Reused margins require numerical development parity before import; new routes, normalization constants and thresholds are fitted on development only. See [the study guide](gemma4_evaluation.md#second-stage-optimization).

## Validation

After reorganization, all 245 regression tests pass. Ruff checks/formatting, all four shell launchers' syntax checks, the dataset plan command, Gemma command help, 84 relocation/consolidation targets and 22 research-module imports were checked. A wheel builds successfully with the new commands/configuration and without weights, logs or bytecode. Git exclusion checks retain source, dependency, example-config and report files while excluding local/runtime output. The original five-model Gemma score replay still passes after the moves. Detailed local logs are under `results/workspace_cleanup_20261028/`.

The second-stage Gemma study also completed all five variants and passed replay. Automatic finalization checked 25 AP-summary rows, 200 operating-summary rows, 29 matched-comparison rows and 165 source/method uncertainty rows, including the exact 4,032-case all-source and 3,792-case common-source cohorts. Its local completion receipt is `results/gemma4_optimization_20261028/finalization.json`.
