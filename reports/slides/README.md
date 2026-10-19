# Model evaluation presentation

The latest presentation is the **32-slide progress review finalized 17 October
2026**. It consolidates the original four-image-model comparison and separate
VadCLIP study, data expansion, six actual CLIP visual fine-tunes, the 330-strategy
screen, the 50-head causal video pilot and the latest ensemble/state/tiling work.
It retains regressions, source/label qualifications and the distinction between
ranking, fixed decisions, component recall and event alerts. No new latency
benchmarks are included.

- [Progress PDF](model_evaluation_progress_20261017.pdf) and
  [editable PowerPoint](model_evaluation_progress_20261017.pptx).
- [One-slide summary PDF](summary_slide_progress_20261017.pdf),
  [PowerPoint](summary_slide_progress_20261017.pptx) and
  [PNG](summary_slide_progress_20261017.png).
- [Speaker notes](progress_speaker_notes_20261017.md),
  [source hashes and measured layout](progress_source_manifest_20261017.json),
  [preview slides 1–16](progress_deck_preview_1_20261017.png) and
  [preview slides 17–32](progress_deck_preview_2_20261017.png).
- [Latest findings](../event_improvements_findings_20261017.md),
  [complete latest results](../../results/event_improvements_20261017/report.md)
  and [reproduction guide](../../docs/event_improvements.md).

Slides 1–10 cover scope, data, grouping, compute and metrics; 11–16 cover the image
comparison, actual fine-tuning and strategy screen; 17–23 cover unresolved targets,
error/component audits and the video pilot; 24–27 cover calibration, uncertainty
and the latest state/data/detector improvements; 28–32 cover VadCLIP, the five-target
assessment, retained regressions, next evidence and reproducibility.

The 17 October smoking/AVSS policy tests are retrospective. Three newly acquired
highway clips provide small-debris boxes but no owner-associated event labels;
three other entries are explicit ABODA aliases. The tiled detector result is
component recall on partial references, not end-to-end event accuracy. Historical
public-sleep/unattended labels still need independent human review.

To rebuild from saved completed results:

```bash
PYTHONPATH="$PWD:/tmp/capstone_slide_deps" .venv/bin/python reports/slides/build_progress_presentation.py
```

The builder checks text fit, geometry, source completion and editable-slide
counts. PDF and PowerPoint share a measured layout with embedded DejaVu Sans;
the PDF is rendered independently rather than exported by PowerPoint. Every PDF
page was rendered for layout review. Source hashes and speaker notes are saved.

## Expanded accuracy presentation — 14 October

The earlier presentation is the **expanded accuracy study finalized 14 October
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
