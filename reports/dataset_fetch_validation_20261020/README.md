# Dataset downloader validation — 20 October 2026

The unified downloader and source/license catalog include the later acquisitions
through the XRF55 RGB public pilot. This work changes acquisition tooling and
provenance documentation; it does not change scientific labels, splits, trained
weights or model evaluation results.

- **17 source entries; 4,520 pinned acquisition assets; 9,662,138,000 payload bytes.**
  The default research profile selects 13 sources and 4,519 assets. D2-CDNet's
  metadata is optional; the original Drive aggregation and two credential-gated
  Roboflow sources use their existing providers.
- **225 unit tests passed**, including **33 downloader regressions** covering
  interruption/resume, corrupt files, mirror fallback, retry classification,
  long host backoff, concurrency bounds, file locks, ZIP byte ranges and CRCs,
  version changes, budget guards, portable manifests and annotation discovery.
- **All 4,520 cached assets passed SHA-256 verification**, without downloading
  the full datasets. See [cache_audit.json](cache_audit.json). A subsequent
  [adoption run](cache_adoption.json) created the isolated metadata state.
- The final [cached research rerun](fast_cached_rerun.json) reused all 4,519
  selected files with zero network transfer. Timing is local acquisition-tool
  behavior, not a model latency benchmark.
- A fresh [live smoke test](live_smoke.json) downloaded and verified an ABODA
  README, an IWDD Drive annotation, one XRF55 clip and a Street Scene annotation
  selected from the 48.98 GB ZIP. Its four final file payloads total 1,566,850
  bytes; the HTTP byte counter also includes the ZIP directory/range overhead
  and excludes gdown's internal traffic. The archive was not downloaded in full.
  The [immediate rerun](live_cached_rerun.json) cached all four with no network.
- [Annotation compatibility](annotation_compatibility.json) confirms that the
  original IWDD preparation filter still finds 400 train and 100 test labels,
  while retaining the 500 historical annotation receipts. New receipts and
  resume metadata live under `DATA_ROOT/.fetch_state/`; only hidden partial
  payloads sit alongside a file during transfer.
- [Wheel validation](package_validation.json) loaded all 17 sources / 4,520 assets
  from an extracted package outside the repository, including all 14 file
  manifests and the bundled evidence. The default plan required neither local
  `data/` receipts nor network access.
- Repository-wide Ruff checks and formatting checks passed. Exact output is in
  [lint.log](lint.log), [format.log](format.log), [unit_tests.log](unit_tests.log)
  and [downloader_tests.log](downloader_tests.log).

The evidence establishes transfer correctness for the exercised public assets,
cache integrity and package portability. It does not establish that every remote
URL remains available, that gated exports are accessible, or that every source
has a complete license grant. Those limits remain explicit in the registry.

Use the [downloader guide](../../docs/dataset_fetching.md) and
[consolidated source/license inventory](../../docs/dataset_sources.md).
The [validation summary](validation.json) records the final code and artifact
hashes. Run-level JSONL journals retain the per-file outcomes.
