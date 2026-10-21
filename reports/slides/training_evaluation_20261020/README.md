# Training and evaluation of the four requested models

This presentation focuses on MobileCLIP2-S2, SigLIP2-B/16-224,
SmolVLM-500M-Instruct and VadCLIP. It covers data, training and accuracy evaluation
through 20 October 2026, preserving previous decks and model results.

- [46-slide PDF](../model_training_evaluation_20261020.pdf) and [editable PowerPoint](../model_training_evaluation_20261020.pptx).
- [One-slide stage overview](../training_stages_summary_20261020.pdf) and [editable overview](../training_stages_summary_20261020.pptx). This compact overview uses smoking; the full deck covers all five targets.
- [Detailed speaker notes](speaker_notes.md).
- [619 unrounded metric rows](metrics.csv): AP, ROC-AUC, F1, precision, recall, FPR, accuracy, balanced accuracy, confusion counts, cutoffs, sources, stages and seeds.
- [49 dataset count rows](dataset_counts.csv), with positive/negative counts.
- [Sources and measured layout](manifest.json), [validation](validation.json), and [all rendered slides](previews/index.md).

Slides 1–11 establish stage coverage, data and metrics. Slides 12–24 cover the
original CLIP probes, actual partial visual fine-tuning, per-target results,
frozen routes, thresholds and seeds. Slides 25–28 cover expanded representation
learning and matched three-model comparisons. Slides 29–34 cover SmolVLM LoRA.
Slides 35–37 cover VadCLIP and the qualified four-model metric table. Slides
38–44 cover region/temporal learning, source-aware selection and new-data
ablations. Slides 45–46 consolidate evidence and artifacts.

The original CLIP visual study has 7,498 examples; the expanded strategy/SmolVLM
corpus has 8,616 and contains those earlier examples. These totals are not additive.
The original CLIP visual study and SmolVLM use different training corpora. The
16 October selected CLIP strategies and 19 October SmolVLM share expanded splits
and matched held-out cases. VadCLIP has a separate smoking-only session split,
with 1,056 test frames rather than the 646-frame camera-c1 holdout. Its four other
target results are unavailable. SmolVLM has no executed linear-probe stage.

The deck distinguishes frozen encoders with new classifiers, actual partial
visual updates, language-layer LoRA, saved-weight interpolation, temporal adapter
training and development-only routing. No full-parameter unfreezing is claimed.
Stage regressions and fixed-threshold failures remain visible. XRF55 data are
explicitly qualified scripted-gesture evidence. No new model inference, training
or latency benchmark was needed for this documentation build.

Rebuild and validate from the repository root:

```bash
PYTHONPATH="/tmp/capstone_slide_deps:$PWD" .venv/bin/python \
  reports/slides/build_training_evaluation_20261020.py
PYTHONPATH="/tmp/capstone_slide_deps:$PWD" .venv/bin/python \
  scripts/validate_training_evaluation_slides_20261020.py --render
```

The presentation environment needs `python-pptx`, `reportlab`, `Pillow` and NumPy;
Poppler supplies `pdfinfo`, `pdftotext`, `pdffonts` and `pdftoppm` for validation.
