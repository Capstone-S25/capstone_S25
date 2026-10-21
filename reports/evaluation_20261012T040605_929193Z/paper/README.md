# Compiled experimental report

Open `paper.pdf`. Its source is `paper.tex`; numerical tables in `generated/` and
four figures in `figures/` are generated from the saved run. The paper documents
implemented methods and available evidence, without claiming a five-event
deployment winner.

The experiment has 308 prepared image inputs, 3,684/3,696 measured prediction
rows, and 19/20 completed model/cohort jobs. MiniCPM smoking timed out during its
third pass. Smoking is development-only; CAVIAR test has one positive recording;
three events and VadCLIP have no executed confirmed evaluation. The original
experiment files have been preserved.

## Rebuild the PDF

With [Tectonic](https://tectonic-typesetting.github.io/book/latest/installation/)
installed, run from this directory:

```bash
tectonic --keep-logs --keep-intermediates --untrusted paper.tex
```

The checked-in generated tables and figures are sufficient to compile. A normal
TeX Live installation can instead run `pdflatex -halt-on-error paper.tex` twice;
the bibliography is embedded, so BibTeX/Biber is unnecessary. Required LaTeX
packages are listed in the preamble (including PGF/TikZ and Latin Modern).

The initial compilation used the official portable Tectonic 0.17.0 Linux-musl
release installed under `/tmp`, with TeX packages cached under `/tmp`. These are
temporary build tools, not project dependencies or model-runtime changes.
`build_metadata.json` records the compiler/archive checksum, final PDF checksum
and PDF validation results.

## Regenerate and audit the evidence

Use Python 3.12 with the repository requirements installed:

```bash
python generate_evidence.py
```

This discovers the repository from its own location, reads the enclosing run,
recomputes metrics, verifies fingerprints and available images, and recalculates
saved bootstrap/calibration on a temporary copy. It performs no model inference
and never modifies original prediction or statistical artifacts. It requires
the matching repository source and bundled CAVIAR example images. Verification
of unavailable remote smoking pixels is explicitly limited.

`audit.json` contains all checks, recomputed quality and diagnostic partial-job
metrics, preserved original-file SHA-256 inventory, source revision and audit
runtime versions. Its expected single failing check is MiniCPM smoking's
incomplete planned cohort. Audit/runtime versions are separate from the recorded
server inference versions. `first_pass_cases.jsonl` consolidates quality evidence
without counting timing repeats as extra test examples.

If regenerated inputs differ, inspect the checks and update the interpretation
before recompiling. Retain original server prepared CSV manifests, dataset JSON,
fetch receipts and human-review notes for full preparation reproducibility;
the supplied run does not include all of these. The local prepared dataset has
a different fingerprint and cannot replace the server's preparation.

## Contents and scientific limits

The paper includes research questions, source/event definitions, shortlist
rationale with primary references, exact checkpoint revisions, annotation and
sampling rules, deduplication/group splits, endpoint/window construction,
adapter score formulas and prompts, CPU controls, calibration, uncertainty,
all completed categorical/resource comparisons, output-format errors, failure
analysis, evidence gaps and a proposed completion protocol. It distinguishes
inferred rationale from recorded decisions and future experiments from executed
tests. Bootstrap degeneracy is quantified: 571/2,000 CAVIAR draws contain no
positive recording; finite-only intervals must not establish population claims.

`../inspection.md` provides a concise assessment. The compiled paper is an
extensive screening report; untested tasks, untouched generalization and actual
deployment-host/live-stream measurements still require additional experiments.
