# Model evaluation presentation

The latest presentation is the **expanded accuracy study finalized 14 October
2026**. It includes 55,156 new GPU predictions, larger unattended-item cohorts,
qualified diagnostics for the other three anomalies and retained smoking/VadCLIP
evidence. There was no additional latency experiment.

- [Expanded PDF deck](model_evaluation_expanded_20261014.pdf) and
  [editable PowerPoint](model_evaluation_expanded_20261014.pptx): 16 slides covering
  findings, methodology, testing, evaluation and limits.
- [Expanded one-slide summary](summary_slide_expanded_20261014.pdf) and
  [editable summary](summary_slide_expanded_20261014.pptx).
- [Expanded speaker notes](expanded_speaker_notes_20261014.md),
  [source hashes and layout](expanded_source_manifest_20261014.json) and
  [deck preview](expanded_deck_preview_20261014.png).
- [Full accuracy findings](../../results/expanded_accuracy_20261013/report.md) and
  [study overview](../expanded_accuracy_review_20261014.md).

Unattended-item labels are provisional assistant review. The three weak anomaly
cohorts report target-prediction rates rather than verified event accuracy. The
oracle pose task supplies ground-truth crops and has only four negative test
controls. These qualifications are visible in the deck and speaker notes.

To regenerate the new presentation from saved results:

```bash
PYTHONPATH="$PWD:/tmp/capstone_slide_deps" .venv/bin/python reports/slides/build_expanded_presentation.py
```

The temporary dependency directory is the presentation environment used in this
workspace. On another machine, install `requirements.txt` in a virtual environment
and include the repository root on `PYTHONPATH`. Both builders perform no model
inference and create editable vector/text slides and independently rendered PDFs
from a common measured layout. PNG previews are rendered separately.

## Earlier presentation

The original presentation uses the completed **13 October 2026** comparisons,
including the combined GPU run, separate CPU latency pilot, dataset audit and
supervised offline VadCLIP smoking study. It remains historical:

- [Editable PowerPoint deck](model_evaluation_20261013.pptx): 16 slides, with speaker notes and editable text, tables and charts.
- [PDF deck](model_evaluation_20261013.pdf): the same slide content and layout, with embedded fonts.
- [Standalone summary slide](summary_slide_20261013.pptx), also available as [PDF](summary_slide_20261013.pdf) and [PNG](summary_slide_20261013.png).
- [Deck preview](deck_preview.png) and [speaker notes](speaker_notes.md).

Slides 1–13 cover the objective, model families, dataset readiness, methodology, test settings, metrics, model findings, resource measurements, workflow improvements and next evaluation. Slides 14–16 provide development diagnostics, calibrated operating points and sources.

Smoking scores are development evidence. Unattended-item testing has very few independent source sequences. VadCLIP uses separate supervision and source groups. Littering, loitering and rough sleeping lack verified event cohorts; they remain unscored. These qualifications appear in the slides and notes.

The [source manifest](source_manifest.json) records hashes of the input artifacts and the complete layout specification. The builder reads saved results and performs no model inference. Text and geometry are checked for overflow; the PowerPoint packages and rendered PDF pages were inspected. The PDF is rendered from the common layout, rather than exported through PowerPoint.

To regenerate on this workspace, with DejaVu Sans installed:

```bash
python3 -m venv /tmp/capstone-slides
/tmp/capstone-slides/bin/pip install -r reports/slides/requirements.txt
/tmp/capstone-slides/bin/python reports/slides/build_presentation.py
```

The builder recreates the PowerPoint, PDF, speaker notes and source manifest. PNG previews are rendered separately from the PDFs.
