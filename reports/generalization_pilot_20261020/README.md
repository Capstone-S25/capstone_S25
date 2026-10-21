# Public-source generalization pilot — 20 October 2026

Read the [findings](report.md), [eight-page PDF addendum](runthrough.pdf), or
[ten-slide presentation](../slides/model_generalization_pilot_20261020.pdf).
The [editable deck](../slides/model_generalization_pilot_20261020.pptx) and
[speaker notes](../slides/model_generalization_pilot_20261020_notes.md) accompany it.
The complete 19 October paper remains the full historical account; this is its
new experimental addendum.

The study acquires 740 weak scripted-action clips, fits 36 CLIP classifier heads,
compares original/prior-LoRA SmolVLM transfer, and tests tiled detection through
the complete unattended-item tracking/alarm pipeline. It establishes transfer
regressions, rather than a model ready for promotion. No new backbone fine-tuning,
SmolVLM training or latency testing is claimed in this pilot.

- [All smoking operating points](smoking_all_operating_points.csv)
- [All candidate development cells](candidate_development.csv)
- [New-source subject metrics](new_source_subjects.csv) and [action errors](new_source_actions.csv)
- [Per-stream results](streams.csv) and [separate evidence-group aggregates](stream_aggregate.csv)
- [Reporting source/export identities](source_manifest.json) and [document validation](document_validation.json)
- [LaTeX source](runthrough.tex), [self-contained source package](runthrough_source.zip), and [isolated compilation check](source_package_check.json)
- [Rendered page review](previews/index.md)
- [Full reproduction guide](../../docs/generalization_pilot.md)

Rebuild from the repository root after the measured results and checks exist:

```bash
MPLCONFIGDIR=/tmp/capstone_matplotlib PYTHONPATH="$PWD" \
  .venv/bin/python scripts/build_generalization_pilot_report.py
XDG_CACHE_HOME="$PWD/data/tools/tectonic/cache" \
  data/tools/tectonic/tectonic --keep-logs \
  reports/generalization_pilot_20261020/runthrough.tex
```

The source package includes `runthrough.tex` and its `figures/` directory. In that
directory, run `tectonic runthrough.tex` (or XeLaTeX with the standard packages
listed in the preamble). No model or dataset downloads are needed to compile the
paper. A compiler may fetch missing TeX fonts/packages on its first run.
