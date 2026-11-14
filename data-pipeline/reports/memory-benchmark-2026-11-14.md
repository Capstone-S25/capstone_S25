# Pipeline memory benchmark report

Generated 2026-11-14 23:49 Singapore time.

## Findings

- Four cameras with fast counting consumers peaked at **200.0 MiB summed application RSS**, including the stability run. This covers Python and recording FFmpeg processes, not Docker services or model inference.
- Above the 75.1 MiB shared Python baseline, one camera added approximately **22.2 MiB for recording** and **10.3 MiB for streaming including decoding**.
- Whole-host memory averaged **14.29 GiB** during the four-camera comparison windows. It includes Docker, OS and unrelated applications and is not the project's standalone RAM requirement.
- Direct-camera production **excludes MediaMTX and mock publishers**. Removing MediaMTX's 28.0 MiB mean working memory reduces the known-container reference budget from 90.7 to **62.7 MiB** (PostgreSQL only). This is an estimate; application-only figures are unchanged.
- With deliberately slow consumers, queued image data grew by **93.6 MiB/min**, reached **172 frames**, and application RSS peaked at **311.1 MiB**. An unbounded queue therefore has no fixed memory ceiling if this imbalance continues.
- These are **640×360, H.264, 2 selected FPS per camera** measurements. YOLO/VLM and production-resolution capacity remain unmeasured.

## Workload and method

- Host: Windows-11-10.0.22631-SP0; 15.95 GiB physical RAM; Python 3.12.3.
- Input: h264, 640×360, 25/1 FPS, RTSP over TCP.
- Selection rate: up to 2 FPS per camera; decoding still reads the full incoming stream.
- One mock stream supplies independent connections; each simulated camera owns its decoder, recording session, FFmpeg process and queue.
- Cameras share one Python process and database engine. Each camera has its own counting consumer; no YOLO or VLM model is loaded.
- Normal cases: 10s warm-up then 60s measurement; 3 fresh-process repeats per case.
- Reported means are medians of per-run steady-window means. Peaks are the highest sampled value across complete runs, including startup and draining.
- Process RSS/USS is sampled approximately every 0.2s, host memory every 1s, Docker statistics approximately every 5–8s.
- RSS is resident memory; USS is private resident memory. Summed RSS may count shared pages more than once. Short peaks between samples may be missed.
- The benchmark invokes the production worker and runner helpers. Queue instrumentation counts image bytes without changing FIFO/unbounded behavior.
- Existing services and the mock publisher remain running. Baseline means no camera workers, not an otherwise empty machine or Docker-off measurement.
- Whole-host measurements also include the benchmark coordinator and its monitoring commands; component application measurements exclude that separate coordinator.
- The in-process sampler retains measurements until the run ends. Small long-run application memory growth can include sample-storage overhead; queue-image growth is reported separately.
- Decoder memory belongs to the streaming worker. The offline recorded-segment decoder is not running and is not counted again.

The suite owns a separate looping sample-video publisher with normalized timestamps, H.264 at 25 FPS, one-second keyframes and no B-frames. Its memory is test infrastructure and is included in host/publisher totals. Existing streams were not modified.

## Component and camera measurements

Application figures include Python plus recording FFmpeg processes; they exclude Docker services.

| Scenario | Cameras | Python mean MiB | FFmpeg mean MiB | App mean RSS MiB | App peak RSS MiB | App mean USS MiB | Peak queued frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 75.1 | 0.0 | 75.1 | 75.4 | 51.1 | 0 |
| recording | 1 | 77.6 | 19.6 | 97.3 | 97.9 | 55.5 | 0 |
| streaming | 1 | 85.4 | 0.0 | 85.4 | 86.1 | 58.8 | 1 |
| full | 1 | 87.9 | 19.6 | 107.5 | 108.7 | 63.3 | 1 |
| full | 2 | 96.7 | 39.2 | 136.0 | 138.6 | 76.0 | 2 |
| full | 4 | 112.6 | 78.6 | 191.0 | 194.7 | 99.7 | 4 |

### Additional memory for one camera

These are controlled-run differences from the same imported Python baseline, not exact ownership of shared thread allocations.

| Component | Additional steady RSS MiB | Additional steady USS MiB |
| --- | ---: | ---: |
| recording | 22.2 | 4.4 |
| streaming | 10.3 | 7.7 |
| full | 32.5 | 12.2 |

## Docker and whole-machine memory

Whole-machine used memory is total RAM minus available RAM. It already includes the application, Docker/WSL, OS, publisher and other open applications; do not add component numbers to it.
WSL figures include all visible vmmem processes, not necessarily Docker alone. Docker Desktop figures are the sum of its visible Windows processes and may share pages.

| Scenario | Cameras | Host used mean GiB | Host available minimum GiB | WSL VM RSS mean MiB | Docker Desktop RSS mean MiB | Mock publisher RSS mean MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 13.97 | 0.88 | 498.6 | 403.8 | 83.2 |
| recording | 1 | 14.04 | 0.89 | 501.6 | 405.6 | 83.4 |
| streaming | 1 | 13.98 | 1.07 | 503.8 | 406.8 | 83.4 |
| full | 1 | 14.14 | 1.26 | 504.7 | 407.8 | 83.4 |
| full | 2 | 14.23 | 1.12 | 504.0 | 410.0 | 83.4 |
| full | 4 | 14.29 | 1.05 | 508.8 | 407.4 | 83.6 |

### Running containers

Docker CLI statistics exclude reclaimable cache on Linux. These container figures overlap WSL/host totals and must not be added to them.

| Container | Services-idle mean MiB | Four-camera mean MiB | Highest observed MiB |
| --- | ---: | ---: | ---: |
| capstone-postgres-dev | 55.4 | 62.7 | 64.5 |
| mock-cctv | 26.8 | 28.0 | 29.5 |

## Direct-camera production estimate (MediaMTX excluded)

For the intended production topology, real cameras provide the RTSP streams directly. The recording and streaming workers connect to each camera URL; MediaMTX is not required unless a relay is deliberately introduced.
This section is an accounting estimate using the four-camera measurements, not a new benchmark with MediaMTX stopped. Historical measurements above and raw CSV/JSON data remain unchanged.

| Component | Observed in the test setup | Direct-camera production budget |
| --- | --- | --- |
| PostgreSQL container | 62.7 MiB mean working memory | Retain 62.7 MiB as the measured reference |
| MediaMTX mock relay | 28.0 MiB mean working memory | Exclude: 0 MiB allocated to this service |
| Known container working-memory subtotal | 90.7 MiB (PostgreSQL + MediaMTX) | About 62.7 MiB (PostgreSQL only) |
| Local mock-video publishers | Test-only FFmpeg encoders | Exclude; encoding/stream publishing happens on the cameras |
| Recording FFmpeg processes | Included in application measurements | Keep; the server still needs to record the incoming streams |
| Docker runtime / VM overhead | Included in host observations | Keep an allowance while PostgreSQL or other production services use Docker |

Remove approximately 28.0 MiB from the known-container working-memory budget. This is not an exact physical host-RAM saving: container working memory, process RSS and WSL allocation are different, overlapping measurements. Do not subtract all of WSL/Docker Desktop or directly rewrite the measured whole-host totals.
The measured 200.0 MiB fast-consumer application peak and provisional 250 MiB application allowance already exclude MediaMTX, so neither is reduced again. A direct-camera deployment run is still needed to measure actual host savings and production-resolution behavior.


## Scaling estimate

The increase from one to two cameras was 28.4 MiB steady RSS. Extrapolating that increment predicts 192.8 MiB for four cameras; the actual four-camera result was 191.0 MiB (+0.9% prediction error).
Across fast-consumer four-camera runs, including any stability run, the highest sampled application RSS was 200.0 MiB. A provisional application-only allowance with 25% headroom is 250.0 MiB for this workload. This excludes extra queue backlog and models and is not a whole-machine RAM recommendation.
Shared Python overhead is counted once. Use measurements at the actual resolution/codec and include separate model memory, database growth, OS/services and queue capacity before sizing deployment hardware.

Private resident memory increased by approximately 12.7 MiB from one to two cameras. Four cameras used 99.7 MiB private resident memory on average, versus 191.0 MiB summed RSS. Shared executable/library pages explain part of the difference; neither figure should be added to host used RAM.

## Sustained operation and slower consumer

| Case | Input cameras | Measurement s | Total including drain s | App peak RSS MiB | Peak queued frames | Peak queued image MiB | Late RSS slope MiB/min | Late queue slope MiB/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_4cam_r1_soak | 4 | 600 | 610.5 | 200.0 | 4 | 2.6 | 0.26 | 0.00 |
| full_4cam_r1_slow | 4 | 60 | 102.1 | 311.1 | 172 | 113.4 | 102.53 | 93.62 |

Each queued BGR image in this workload uses 0.659 MiB, excluding frame/Python metadata and decoder reference buffers. The slow case sleeps 0.75s per consumed frame per camera, intentionally slower than selected input.
A positive queue-growth rate means memory has no fixed long-term bound. Finishing a finite stress test is not evidence that an unbounded queue is safe indefinitely. Near-zero growth during the soak establishes behavior only for that measured interval.

## Limits and deployment interpretation

- All simulated cameras read the same mock stream through independent connections. Distinct scenes and codecs may change memory behavior.
- The current stream is 640×360. These numbers are not a 1080p/4K capacity estimate; frame payload scales with pixel count, but total decoder memory is not reliably extrapolated that way.
- Multiple independent consumers are used. A future single shared YOLO consumer must sustain the aggregate selected FPS; its queue behavior requires a new benchmark.
- At four cameras and 2 selected FPS each, a shared consumer would need to handle up to 8 FPS (about 125ms per frame on average, including processing overhead) to avoid sustained backlog.
- YOLO/VLM model weights, inference workspace and GPU memory are absent. They can dominate the eventual memory budget.
- The Python pipeline runs on the Windows host, while PostgreSQL and MediaMTX run in Docker. A fully containerized/Linux deployment must be measured again.
- The mock CCTV container and local FFmpeg publisher are test infrastructure; real cameras may remove them. Their cost is included in host measurements here.
- Background Windows applications and Docker VM caching can move host totals independently of the pipeline. Host delta is not exact application attribution.
- Services were not stopped, so OS-only and Docker-off baselines were not measured. PostgreSQL contains the existing development data plus real benchmark segment metadata.

## Input timing finding

An earlier four-camera attempt against the pre-existing mock feed stopped after 5.6s with `ValueError: Missing or invalid timing for frame 19`. No memory figures from that attempt are included in this report. Evidence is retained in `recordings/benchmarks/suite_20261114T144055Z_3e4863/full_4cam_r1/worker.log`.
All reported comparisons use the separate timestamp-normalized feed. The earlier timing failure warrants a separate ingestion-reliability investigation; changing benchmark input is not a production fix.

## Artifacts

Raw runs: `recordings/benchmarks/suite_20261114T145207Z_de86ff/`. Each case includes process, host and Docker CSV samples, a JSON summary and worker log. `results.json` contains all run summaries; `aggregate.csv` contains the table aggregates.
