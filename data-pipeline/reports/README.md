# Historical memory benchmark results

The `memory-benchmark-2026-11-15` reports measured one independent consumer per
camera. They remain valid records of that implementation, not measurements of
the current shared round-robin consumer.

The benchmark harness now uses one shared consumer for all cameras. Its
0.75-second stress delay applies to every frame across the whole pipeline,
rather than independently per camera. Drop counts, draining time, and total
pipeline RAM must be measured again before publishing updated results.

The dated PDF, Markdown, CSV, and JSON files are preserved without changing
their measurements. No new full benchmark run has been published yet.
