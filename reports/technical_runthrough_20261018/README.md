# Complete technical runthrough — 18 October 2026

The report covers the project history through the completed experiments of
17 October. It includes source discovery/acquisition, annotation and duplicate
audits, data splits, shared GPU evaluation, the historical CPU pilot, supervised
VadCLIP, expanded four-model testing, actual CLIP visual fine-tuning, the strategy
screen, causal video/event evaluation, calibration, ensembling and tiled detection.
Failures, numerical corrections, provisional labels and retrospective test reuse
are preserved alongside improvements. No new model experiment was run to write it.

- [Compiled report](model_evaluation_runthrough_20261018.pdf).
- [Main LaTeX source](model_evaluation_runthrough_20261018.tex).
- [Complete source ZIP](../model_evaluation_runthrough_20261018_source.zip).
- `chapters/`: detailed narrative and appendices.
- `generated/`: source-derived LaTeX tables, including every strategy/head/policy.
- `figures/`: four standalone scientific PDF figures.
- `numeric_exports/`: original detailed CSV exports from completed studies.
- [Source identities](source_manifest.json): full SHA-256 hashes and table counts.
- [Validation](validation.json) and [independent source compilation check](source_package_check.json).

The compiled report has **83 pages**, approximately **15,000 words of narrative**,
**38 tables** and **four scientific figures**, with 146 recorded source identities.
It has 12 main chapters and seven appendices, with an executive reading
guide, contents, table/figure lists, equations, implementation settings, all 330
strategy-development configurations, all 50 final video-pilot heads, all 38 latest
smoking policies and source-qualified event tables. The full-source ZIP includes
everything needed to compile the document without the model/data directories.

## Compile the existing source

From this folder:

```bash
python3 compile.py
```

The helper finds the workspace-local Tectonic binary, an extracted-source local
binary, or an installed Tectonic/XeLaTeX. If no compiler is available on Linux
x86_64, install the checksum-pinned portable compiler and compile with:

```bash
python3 compile.py --bootstrap
```

The portable compiler is Tectonic 0.17.0. Its official release archive digest is
pinned in `tools/bootstrap_report_latex.py`; installation and TeX package caches
remain local. Initial compilation needs network access to download missing TeX
packages. The helper uses untrusted/no-shell-escape compilation. No administrator
access or system package modification is required.

A standard XeLaTeX installation can also compile the main source three times to
resolve its contents and references. The source uses standard packages:
`geometry`, `lmodern`, `fontenc`, `microtype`, `amsmath`, `amssymb`, `booktabs`,
`longtable`, `tabularx`, `array`, `graphicx`, `float`, `pdflscape`, `xcolor`,
`enumitem`, `listings`, `fancyhdr`, `caption`, `xurl` and `hyperref`.

## Regenerate numeric tables and figures from the repository

The source ZIP already includes the generated tables and figures. Regeneration
requires the original repository's preserved study artifacts, Python environment,
NumPy and Matplotlib:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/build_technical_runthrough.py --compile
```

This reads saved JSON/CSV data and code identities and runs no model inference.
It refuses to substitute old benchmark values for corrected results. The source
registry records local evidence; acquisition URLs are recorded historical sources,
not a new access-availability survey. The CSV field `precision` denotes positive
predictive value; numeric inference precision is taken from run metadata and the
authoritative final float32 evaluation directories.

The original 13/14/17 October slide decks and prior experiment outputs remain
unchanged. The project [slide index](../slides/README.md) links those presentations.
