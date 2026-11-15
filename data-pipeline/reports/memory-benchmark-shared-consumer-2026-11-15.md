# Bounded-queue pipeline RAM benchmark

Benchmark date: 2026-11-15 (Singapore). Completed runs: 20. Excluded attempts: 4.

Consumer setup: one shared round-robin consumer across all camera queues.

## Main results

- Four cameras with fast consumption peaked at **193.5 MiB of application RAM**, including the stability run.
- Shared Python baseline: **74.6 MiB**. Recording adds **22.6 MiB** and streaming, including decoding, adds **10.4 MiB** in separate one-camera comparisons.
- Whole-machine RAM averaged **14.32 GiB** in the four-camera comparison runs. Minimum available RAM was **0.38 GiB**. These host figures include the OS, other applications and test infrastructure.
- Production container reference: **63.7 MiB** working memory, excluding **27.7 MiB** from MediaMTX. Docker/OS overhead and future models still require their own allowance.

| Recording + streaming | Typical pipeline MiB | Peak pipeline MiB |
| --- | --- | --- |
| 1 camera(s) | 106.8 | 108.2 |
| 2 camera(s) | 135.9 | 138.8 |
| 4 camera(s) | 189.9 | 193.5 |

These comparison values summarize the three short repeats per camera count. The headline four-camera peak also includes the longer stability run. All application figures include the shared Python runtime once and the recording FFmpeg children.

## Pipeline RAM and camera scaling

Application RAM is the sampled resident RAM (RSS) of Python plus its recording FFmpeg children. Shared library pages may be counted more than once when process RSS values are summed. Application figures are supporting measurements, not values to add to host RAM.

| Scenario | Cameras | Typical RAM MiB | Peak RAM MiB | Max dropped frames/run |
| --- | --- | --- | --- | --- |
| Idle baseline | 0 | 74.6 | 74.8 | 0 |
| Recording only | 1 | 97.2 | 97.6 | 0 |
| Streaming incl. decoding | 1 | 85.0 | 86.0 | 0 |
| Recording + streaming | 1 | 106.8 | 108.2 | 0 |
| Recording + streaming | 2 | 135.9 | 138.8 | 0 |
| Recording + streaming | 4 | 189.9 | 193.5 | 0 |

Typical values are medians of per-run steady-window means; peaks include startup and draining. The idle Python baseline is shared and counted once, not multiplied by camera count.

## Component estimates for one camera

| Component | Additional RAM above Python baseline (MiB) |
| --- | --- |
| Recording only | 22.6 |
| Streaming incl. decoding | 10.4 |
| Recording + streaming | 32.3 |

Recording includes FFmpeg and metadata registration. Streaming includes PyAV decoding, selection, BGR conversion, the bounded queue and a counting consumer. Decoding is not a third production worker and is not added again. Threads share Python memory, so these are comparative estimates rather than exact per-thread ownership; their increments need not add exactly to the combined result.

The observed one-to-two-camera increase was 29.1 MiB. That increment predicts 194.2 MiB at four cameras, versus 189.9 MiB measured (+2.3% prediction error). This interpolation applies only to this workload and topology.

## Whole-system RAM

Host used RAM is total physical RAM minus available RAM. It already includes the pipeline, Docker/WSL, PostgreSQL, MediaMTX, publishers, profiling and unrelated applications. Do not sum process or container values into these totals. A host change from idle is an observed difference, not exact application attribution; background activity can make it negative.

| Scenario | Cameras | Typical used GiB | Peak used GiB | Min available GiB | Change from idle MiB |
| --- | --- | --- | --- | --- | --- |
| Idle baseline | 0 | 14.22 | 15.53 | 0.42 | +0.0 |
| Recording only | 1 | 14.00 | 15.47 | 0.48 | -227.9 |
| Streaming incl. decoding | 1 | 14.87 | 15.46 | 0.49 | +660.5 |
| Recording + streaming | 1 | 14.15 | 15.90 | 0.05 | -80.3 |
| Recording + streaming | 2 | 14.25 | 15.66 | 0.29 | +25.1 |
| Recording + streaming | 4 | 14.32 | 15.57 | 0.38 | +97.4 |
| Stability | 4 | 15.21 | 15.95 | 0.00 | +1009.8 |
| Slow consumer | 4 | 15.49 | 15.57 | 0.38 | +1301.0 |

## PostgreSQL, Docker and production accounting

The container table uses four-camera steady working-memory measurements. Docker CLI statistics on Linux exclude reclaimable cache. Container values overlap Docker/WSL and host measurements.

| Container | Typical MiB | Peak MiB | Production treatment |
| --- | --- | --- | --- |
| capstone-postgres-dev | 63.7 | 66.4 | Include |
| mock-cctv | 27.7 | 29.2 | Exclude: mock relay |

| Additional layer | Four-camera typical MiB | Accounting |
| --- | --- | --- |
| WSL VM | 513.4 | Includes container allocations and VM overhead; do not add to host |
| Docker Desktop processes | 331.6 | Runtime overhead remains while production services use Docker |
| Local mock publishers | 81.3 | Test-only; real cameras encode and publish their own streams |

The estimated known production-container subtotal is 63.7 MiB, excluding MediaMTX's 27.7 MiB. It is not total server RAM and does not include models or Docker/OS overhead. Pipeline RAM already excludes MediaMTX and mock publishers, so those costs are not subtracted from application figures again. Recording FFmpeg processes remain necessary in production.

Raw host totals remain as measured. Removing a container does not guarantee an identical immediate host-RAM reduction because WSL caching and shared memory behave differently. Exact whole-host RAM without MediaMTX has not been measured; it requires direct-camera input. All visible WSL processes are included, not necessarily Docker alone.
The WSL measurement includes containers, VM infrastructure and retained cache. It cannot isolate Docker-only VM overhead exactly; the full VM allocation is shown instead. Docker Desktop process RAM is measured separately.

## Production ingestion accounting estimate

| Four-camera production accounting estimate | RAM MiB |
| --- | --- |
| Pipeline, including Python and recording FFmpeg (peak incl. stress) | 207.8 |
| Retained production containers (typical) | 63.7 |
| Docker Desktop processes (typical) | 331.6 |
| WSL remainder after all measured container working memory | 421.9 |
| Illustrative ingestion + runtime subtotal | 1025.0 |

The illustrative subtotal is **1.00 GiB before Windows/host OS, unrelated applications and models**. It retains PostgreSQL and runtime overhead while omitting MediaMTX working memory and the publishers. The WSL remainder is VM resident RAM minus all container working-memory references; it includes VM infrastructure, retained cache and potentially other WSL workloads. These different counters do not isolate ownership exactly, and the subtotal mixes a pipeline peak with typical service values. It is a rough accounting estimate, not a measured production peak, an upper bound, or a server specification. It does not modify the measured host totals.

## Queue stability and stress

| Normal scenario | Cameras | Peak waiting | Image MiB | Consumed / dropped | Max drain/cleanup s |
| --- | --- | --- | --- | --- | --- |
| Streaming incl. decoding | 1 | 3 | 1.98 | 402 / 0 | 0.1 |
| Recording + streaming | 1 | 3 | 1.98 | 404 / 0 | 0.4 |
| Recording + streaming | 2 | 6 | 3.96 | 804 / 0 | 0.6 |
| Recording + streaming | 4 | 12 | 7.91 | 1,600 / 0 | 1.4 |

Normal-run frame counts sum all repeats, including warm-up. Queue peaks are simultaneous waiting frames across cameras, tracked on every enqueue/dequeue rather than inferred from sampling. Drain/cleanup is time after the planned producer deadline; it also includes recorder finalization and worker teardown, so it is an upper bound on queue-only draining.

| Case | Measured s | Total s | Peak RAM MiB | Peak queued | Dropped | Drain/cleanup s |
| --- | --- | --- | --- | --- | --- | --- |
| Stability | 600 | 610.6 | 188.0 | 12 | 0 | 0.6 |
| Slow consumer | 60 | 94.2 | 207.8 | 32 | 413 | 24.2 |

- Stability: produced 4,688, consumed 4,688, dropped 0; peak queued image data 7.9 MiB. Late queue-image trend: -0.00 MiB/min; late application trend: -2.04 MiB/min.
- Slow consumer: produced 536, consumed 123, dropped 413; peak queued image data 21.1 MiB. Late queue-image trend: -0.21 MiB/min; late application trend: 2.25 MiB/min.

Each selected BGR image uses 0.659 MiB before metadata and decoder buffers. The configured bound is 8 waiting frames per camera, or 32 across four cameras. The frame being processed and decoder/conversion buffers are outside that queue limit. On overflow, the oldest waiting frame is dropped; the active frame is not interrupted. Stable queue memory can therefore coexist with missed live frames. Recording preserves video independently, but does not automatically recover skipped detections.

## Comparison with previous benchmark

Previous run started 2026-11-15T02:54:25.109788+08:00; this run started 2026-11-15T16:14:57.926074+08:00. The input profile, selected FPS and queue capacity match. The previous implementation used one consumer per camera; this run uses one shared round-robin consumer.

| Cameras | Previous typical MiB | Shared typical MiB | Previous peak MiB | Shared peak MiB |
| --- | --- | --- | --- | --- |
| 1 | 107.4 | 106.8 | 108.5 | 108.2 |
| 2 | 136.0 | 135.9 | 138.3 | 138.8 |
| 4 | 190.4 | 189.9 | 195.3 | 193.5 |

| Four-camera stress metric | Previous: 4 consumers | Current: 1 consumer |
| --- | --- | --- |
| Pipeline peak MiB | 215.8 | 207.8 |
| Produced frames | 536 | 536 |
| Consumed frames | 396 | 123 |
| Dropped frames | 140 | 413 |
| Peak waiting frames | 32 | 32 |
| Peak queued image MiB | 21.1 | 21.1 |
| Drain/cleanup seconds | 6.7 | 24.2 |

The stress delay is 0.75s per frame. Previously four consumers could wait independently, giving a nominal combined capacity of 5.33 FPS before overhead. One shared consumer has a nominal capacity of 1.33 FPS against up to 8 selected input FPS. Different drop counts and drain times therefore reflect a different processing bottleneck, not an equal-throughput algorithm comparison.

The shared Python baseline was 74.6 MiB previously and is 74.6 MiB now. Process/runtime changes and profiling affect RAM differences too. Whole-host results were collected at different times with other applications running; their differences cannot be attributed solely to round-robin scheduling. Neither run includes YOLO/VLM.

## Workload, method and limits

- Host: Windows-11-10.0.22631-SP0; 15.95 GiB physical RAM; Python 3.12.3.
- Input: h264, 640 x 360, 25/1 source FPS over RTSP/TCP; up to 2 selected FPS per camera. Every incoming frame is still decoded.
- Queue: 8 waiting frames per camera, drop oldest on overflow. Cameras share one Python process and database engine, with independent worker pairs and one shared round-robin consumer across all camera queues.
- Normal trials: 10s warm-up + 60s measurement, 3 fresh-process repeats per scenario. Stability and stress durations are shown above. The slow consumer sleeps 0.75s per frame across all cameras.
- Process RAM is sampled approximately every 0.2s, host RAM every 1s, and Docker approximately every 5-8s. Short peaks between samples may be missed. Peak/minimum host figures cover each full observed run, while typical host figures exclude warm-up and draining.
- The benchmark uses the production worker and runner helpers. The sampler retains timestamped rows in memory; small long-run growth may include profiling overhead. The coordinator and sampling commands are also included in host totals.
- A dedicated timestamp-normalized publisher reads the sample video, uses one-second H.264 keyframes and no B-frames, and shares the existing mock server. Existing streams/services remain running. The idle baseline is services-running, not Docker-off or OS-only.
- Simulated cameras read independent connections to one mock stream. Distinct physical camera scenes, source codecs and resolutions may alter memory. The current input is not a 1080p/4K hardware-sizing benchmark.
- No YOLO/VLM weights, inference workspaces or GPU memory are included. A shared consumer needs to sustain the aggregate selected rate, not just the per-camera rate. Model and deployment-container changes require another measurement.
- Windows background applications, available-memory pressure, shared pages and Docker/WSL caching affect RAM. A fully containerized/Linux production deployment must be measured separately. Do not interpret the current host total or component reference subtotal as a final hardware recommendation.
- This host was already busy: idle-run used RAM peaked at 15.53 GiB, and the lowest available RAM across all runs was 2 MiB. Memory pressure and unrelated activity limit causal interpretation of host deltas.
- Completed runs: 20; excluded runs: 4. Successful runs are checked for bounded queues, expected recording processes, complete draining and produced = consumed + dropped. No new test cases were added.
- Excluded attempts failed before the 10-second warm-up ended: the streaming worker rejected frame 0 because its presentation timestamp was missing or invalid. Full measurement windows from fresh replacement processes are used for estimates; failed-attempt logs are retained.
- These were benchmark-level fresh-trial retries, not camera reconnects inside production workers. No timestamps were fabricated and timing validation was not weakened. The dedicated publisher was restarted once between trials after the first failed attempt, with the same video and encoder settings.
- RAM results are conditional on successful stream startup. The observed first-frame failures remain a startup-reliability limitation to investigate separately; this benchmark does not establish production camera availability.
- The machine experienced substantial memory pressure: minimum available host RAM was 2 MiB. The stability run's lower average resident RAM does not prove lower allocations; working-set drops are consistent with paging/trimming under pressure. Peak RSS and raw time series are retained, and these results are not a memory-leak or production-capacity certification.

## Reproduction and artifacts

Run `npm run benchmark:memory:suite -- --queue-size 8` with PostgreSQL migrated and the mock RTSP server running. The suite owns only its separate publisher and camera workers; it leaves existing services and recordings intact.

Raw data: `recordings/benchmarks/suite_20261115T081449Z_256f44/`. Each case contains process.csv, host.csv, docker.csv, summary.json and worker.log. Suite files include environment.json, results.json, report_data.json, aggregate.csv and report.md. Benchmark date: 2026-11-15 Singapore time.

Excluded cases: full_1cam_r1_failed_attempt1, full_1cam_r3_failed_attempt1, full_4cam_r3_failed_attempt1, full_4cam_r3_failed_attempt2
