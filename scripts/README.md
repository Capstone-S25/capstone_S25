# Research script index

The supported evaluation/data interface is `python -m anomaly_eval`. Gemma tools use `python -m anomaly_eval gemma4 COMMAND` or the isolated GPU launcher `bash scripts/gemma4_gpu.sh COMMAND`. `python -m anomaly_eval gemma4 --help` lists all Gemma commands.

Research scripts are grouped by workflow. Run their modules from the repository root, using the environment described in each study's guide:

| Folder | Purpose | Example |
|---|---|---|
| `datasets/` | Historical acquisition, extraction, source review and exports | `python -m scripts.datasets.export_dataset_sources --help` |
| `clip/` | CLIP preparation, fitting, reports and audits | `python -m scripts.clip.run_clip_training --help` |
| `strategy/` | Region features, alternative heads, routing and transfer | `python -m scripts.strategy.run_strategy_screen --help` |
| `video/` | Detector/tracker, temporal and event experiments | `python -m scripts.video.evaluate_video_pilot --help` |
| `smolvlm/` | SmolVLM adapter fitting, prediction and audits | `bash scripts/smolvlm_gpu.sh train --help` |
| `generalization/` | New-source and complete-stream pilot experiments | `python -m scripts.generalization.run_generalization_pilot --help` |
| `expanded/` | Earlier expanded accuracy assembly and reports | `python -m scripts.expanded.run_expanded_accuracy --help` |
| `reporting/` | Paper/deck construction and validation | `python -m scripts.reporting.build_technical_runthrough --help` |

Some historical builders run their dated build directly rather than exposing a help parser. Consult the study guide before running them. Never rebuild a dated experiment over an existing frozen output to accommodate a code change.

The root retains four environment/runtime launchers and `evaluate.py`, a small compatibility entry point that also works outside the checkout. Four redundant wrappers were removed or folded into shared commands: Gemma evaluation/reporting use the package commands, and the all-dataset fetcher uses `python -m anomaly_eval fetch-all`. Source-pilot features now use the shared strategy extractor with `--data-dir` instead of a wrapper that changed module globals. Acquisition and runtime verification are in `anomaly_eval/gemma4/fetch.py`; the parallel transfer helper, preflight and suite controller also live in that package.

[locations.json](locations.json) maps all 84 relocated or consolidated historical script paths to current files. It is an index for old notebooks, report references and manual commands, not an extra launcher. Current imports and active documentation use the new locations. Dated published reports and raw implementation snapshots retain their original references.

For the former `extract_source_pilot_features.py`, add `--data-dir data/generalization_pilot_20261020/xrf55` when using the shared extractor. The [generalization guide](../docs/generalization_pilot.md) contains the complete replacement commands.
