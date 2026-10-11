# Test data bundle — vetted head-to-head anomaly frames

The 80 vetted test frames used in the frame-level model benchmark (report Section 5, `reports/anomaly_benchmark_report.md`). One subdirectory per label: `normal/`, `littering/`, `loitering/`, `rough_sleeping/`, `smoking/` — 16 frames each. There is no `unattended_items/` directory because the corpus provides zero test support for that class (it contributes a structural zero to macro-F1).

Filenames are the original dataset basenames, which encode provenance (e.g. littering video id, loitering frame id, smoking camera + date). `MANIFEST.csv` maps every file to its label, group_id, split, absolute source path in `data/raw/datasets/`, and SHA-256 digest so the bundle can be re-verified against the original datasets:

```
sha256sum -c <(awk -F, 'NR>1 {print $7"  test_data/"$1}' MANIFEST.csv)
```

The set matches the `samples` list in `results/anomaly_headtohead/run.json` exactly (manifest_sha256 `18087c4d…`, definitions_sha256 `589882c0…`). Selection: one frame per group where a corpus provides multiple frames, plus smoking/normal paired hard negatives drawn from the same camera and date. Full provenance is documented in report Section 2; the bundle was produced by `scripts/make_test_data.py`.
