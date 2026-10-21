# Unified dataset acquisition

The source catalog includes the later training and transfer acquisitions as well
as the initial sources. Use Python 3.12 and the existing project requirements;
no new dependencies are needed. The [source and license registry](dataset_sources.md)
provides the inventory, exact links, recorded hashes, terms and limitations.

## Commands

```bash
# List every source and its recorded license, without contacting the network.
.venv/bin/python scripts/fetch_all_datasets.py --list

# Inspect the default public research acquisition plan without writing files.
.venv/bin/python scripts/fetch_all_datasets.py --plan

# Fetch all recorded public research acquisitions concurrently, reusing the cache.
.venv/bin/python scripts/fetch_all_datasets.py --workers 8 --per-host 3

# Choose a destination DATA ROOT or individual sources.
anomaly-eval fetch-all --output /path/to/data --sources xrf55_rgb_public iwdd

# Use the latest-source profile (the selected, versioned XRF55 pilot).
anomaly-eval fetch-all --profile latest

# Include the original Drive collection, auxiliary metadata and gated exports.
# Set ROBOFLOW_API_KEY in the environment and choose export versions if available.
anomaly-eval fetch-all --sources all

# The original Drive source can also be selected on its own; it lists live files.
anomaly-eval fetch-all --sources drive_original --workers 8 --per-host 3

# Audit every pinned public source file; no network, no per-file receipt changes.
anomaly-eval fetch-all --sources aboda caviar stanford40 iwdd street_scene \
  smoking_kaggle thermal_loitering commons_photos hmdb51 meva avss2007 \
  small_item_component xrf55_rgb_public d2_cdnet --cached-only --verify \
  --report /tmp/dataset_cache_audit.json

# Retry only the recorded failures from an earlier receipt.
anomaly-eval fetch-all --retry-failed /path/to/data/fetch_all_receipt.json

# An explicitly bounded acquisition, useful on a new machine.
anomaly-eval fetch-all --sources xrf55_rgb_public --limit 2 --max-download-gb .02
```

`python -m anomaly_eval fetch-all` and `anomaly-eval fetch --all` are equivalent
entry points. The standalone script resolves imports against its own checkout,
so it also works from another working directory. Global CLI workspace flags
precede the command; for example, `anomaly-eval --data-root /srv/data fetch-all`.
`--output` is the **data root** for this downloader; the legacy `fetch` command
without `--all` keeps its previous `data/additional` output convention.

The default `research` profile contains the 13 recorded public research
acquisitions. It excludes the separately listed Drive aggregation, gated
Roboflow exports and auxiliary D2-CDNet metadata. `--sources all` or
`--profile all` explicitly selects all 17 catalog entries. Drive uses the existing
concurrent folder downloader, which retains duplicate-name handling, resumable
transfers and its own receipt. Roboflow uses the existing credential-aware export
provider. Missing access does not stop the other acquisitions and leaves a
non-success status. Optional D2 multi-GB shards remain available through
`anomaly-eval fetch --sources d2_cdnet --d2-shards intermittentObjectMotion`.

## Speed and error handling

- Eight global workers and at most three concurrent transfers per original host
  by default. Tune both to match the network and storage. There is no intentional
  transfer rate cap. Server retry instructions are respected.
- A bounded queue holds at most twice the worker count. A failed file does not
  discard successful files or stop the remaining work.
- HTTP transfers resume from hidden `.<filename>.fetch.part` files when a pinned hash or a strong source
  validator permits it. A checked `206 Content-Range` is required; a server that
  returns `200` causes a fresh write rather than concatenation. Size and SHA-256
  are checked before atomically replacing the final file.
- Transient connection errors and HTTP 408/425/429/5xx receive bounded retries and
  backoff. Permanent 404 and authentication errors are reported without repeating
  the same request. A long `Retry-After` defers the host instead of sleeping for
  minutes in each worker. Mirrors are attempted independently where cataloged.
- Street Scene uses exact byte ranges for the ZIP directory and selected members.
  It refuses a full-archive fallback. The 48,975,796,813-byte ZIP is never fetched
  in full by this path. Each selected file gets its own size/hash and ZIP CRC check.
- First adoption of an existing file hashes its bytes against the pinned manifest.
  Subsequent normal runs use the receipt, size and modification time to skip
  unchanged files quickly. `--verify` forces SHA-256 again; use it after moving
  data or investigating corruption. Locally recorded hashes establish byte
  identity with the research acquisition, not publisher authentication.
- Independent invocations coordinate the same destination with filesystem locks.
  Partial data survives interruption. Catalog paths, sidecar collisions and
  duplicate destinations are checked before downloading.
- `--min-free-gb` reserves 0.25 GiB by default. `--max-download-gb` caps the
  conservative sum of uncached file payload sizes; it excludes retry traffic and
  remote ZIP metadata. Files with unknown size are deferred under this cap.
  A fresh research fetch needs roughly 9.7 GB for the recorded payload, plus
  partial-file space and any later extraction. Cached files are reused.

Every run produces an atomic summary receipt, defaulting to
`DATA_ROOT/fetch_all_receipt.json`, and a uniquely named JSONL journal alongside
it. Completed results are flushed individually; the summary is checkpointed as
files complete, at most once every five seconds, and on completion/interruption.
This keeps large cached reruns from repeatedly rewriting the entire manifest.
Each fetched file gets a receipt under `DATA_ROOT/.fetch_state/<path hash>/`.
Receipts, resume metadata and locks are kept outside annotation folders so that
training scripts cannot mistake them for labels. Historical `.receipt.json`, `.download.json`, experiment
splits and results are preserved. Use separate `--report` paths for concurrent
invocations if both summaries must be retained. Receipt URLs omit credentials and
signed query strings; credentials belong in provider environment variables.

Exit codes: **0** means all selected work succeeded or was cached; **2** means
at least one file/provider is missing, invalid, inaccessible, failed or deferred;
**130** means the pinned-file run was interrupted. `--limit` and `--retry-failed`
are explicitly marked `bounded_run`; their success covers only that selected
subset. A regular rerun covers files never reached before an interruption.
`--plan` and `--list` make no network requests or filesystem writes. A
`--cached-only` run writes its run-level audit/journal but no per-file receipts;
live Drive/Roboflow providers cannot be audited that way and are reported as
deferred if selected. Those providers are also deferred when a file limit or
payload budget cannot be honored.

## Extending the acquisition plan

The canonical registry is
[`dataset_sources.json`](../anomaly_eval/config/dataset_sources.json). Source IDs
are loaded from it at runtime; they are not a hard-coded CLI choice list.
Each public source points to a packaged JSONL manifest under
[`dataset_assets/`](../anomaly_eval/config/dataset_assets/). A fresh checkout
therefore does not depend on previously downloaded, ignored `data/` receipts.

Add a source row with its homepage, purpose, license evidence and manifest path.
Add its ID to the desired profile, or select it explicitly with `--sources`.
Each line in its manifest represents a file; paths are relative to the chosen
data root. For example:

```json
{"source":"new_source","path":"new_source/video.mp4","urls":["https://publisher.example/version1/video.mp4"],"sha256":"64_lowercase_hex_characters_here","size":12345}
```

Supported transfer kinds are `http` (the default), `gdrive` (add `file_id`) and
`zip_member` (add `member` and the pinned `archive_size`). `urls` can contain
mirrors **only when they provide identical bytes**; differently packaged archives
must be separate assets. Add file-level `attribution` when terms differ within a
source. Unknown licenses should remain explicit rather than being guessed.
Custom deployments can use `--catalog /path/to/catalog.json`; relative manifests
resolve against that file.

After reviewing the changed acquisition plan, update its manifest hashes and
regenerate the consolidated tables:

```bash
.venv/bin/python scripts/export_dataset_sources.py --update-manifest-hashes
.venv/bin/python scripts/fetch_all_datasets.py --plan
```

Without `--update-manifest-hashes`, the export script verifies the existing pins
and only regenerates documentation/CSV. Release versions and member hashes are
deliberately recorded: this tool is dynamic in source selection and configuration,
while reproducing the acquired research bytes. It does not silently switch old
experiments to a publisher's latest release or turn arbitrary files in `data/`
into training data.

## Acquisition versus preparation

Archives are fetched intact unless the manifest explicitly selects ZIP members.
The fetcher does not execute downloaded source code, change labels, rebuild
splits or train models. Stanford40, smoking image, HMDB, AVSS and the small-item
source therefore still need their established extraction/preparation steps on a
fresh machine. The historical scripts document their layouts; for example,
`scripts/extract_hmdb_sources.py` extracts the seven selected HMDB classes and
verifies the original archive. MEVA's original activity annotation text and source
links are packaged in `config/dataset_evidence/meva_annotations.json`.

The IWDD catalog includes the 498 actually acquired videos, 500 annotation files
and source README. Two historically inaccessible video IDs remain listed in the
source registry rather than being counted as acquired data. Commons/Flickr
includes acquired review candidates, including rejected/unused examples; its
file count is not the count of reviewed training images. A scene's frames,
an archive and its mirror, or a weak action label should not be counted as new
independent event evidence. Continue using the frozen study manifests for
training and evaluation.
