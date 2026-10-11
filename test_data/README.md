# Portable evaluation data

`MANIFEST.csv` retains the original 80-image bundle for provenance. Its split is now **dev**: these images and recording groups already informed model comparisons. The five folders each contain 16 historical frames. Rough-sleeping and loitering labels are proxies, and littering action labels need event review; they are not verified held-out task labels.

`EVALUATION.csv` is the portable confirmed-task manifest, with paths relative to this directory, SHA-256, task, evidence, source group and review notes. It contains 32 historical smoking/normal development frames and 72 additional CAVIAR inputs. The latter comprise 36 endpoint frames and 36 chronological past-frame contact sheets under `caviar/`.

For each unattended-item input protocol, development has 20 examples (4 positive, 16 normal) and test has 16 (4 positive, 12 normal), with three source sequences per split. Positives cover a deposited bag persisting after its carrier departs; negatives include attended placement/retrieval and walking controls. Frame/window versions of one sequence remain in the same split. `unattended_items_frame.csv`, `unattended_items_window.csv` and `smoking_frame.csv` can be read directly by the benchmark.

The CAVIAR selection is preliminary sequence-annotation review from a staged single camera. It does not establish new-camera/general deployment performance. Exact hashes verify files, not semantic correctness or independence of transformed duplicates. There is no confirmed held-out five-event test set yet. See [evaluation instructions](../docs/evaluation.md) and [interval provenance](../anomaly_eval/config/caviar_review_intervals.json).

Source: [CAVIAR Test Case Scenarios](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/), EC CAVIAR project / IST 2001 37540. CAVIAR specifies CC BY-SA and requests this acknowledgment; preserve attribution and applicable share-alike terms when redistributing these derived frames.

From the repository root, verify the bundle by running `anomaly-eval run --prepare-only`; every manifest image hash is checked. The old manifest has six CSV columns, with SHA-256 in column six; parsing CSV with Python avoids path/quoting problems.
