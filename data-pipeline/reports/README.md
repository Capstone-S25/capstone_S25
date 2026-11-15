# Memory benchmark results

## Shared round-robin consumer

The latest run completed 20 accepted measurements with one shared consumer and
one eight-frame drop-oldest queue per camera:

- [PDF report](memory-benchmark-shared-consumer-2026-11-15.pdf)
- [Markdown report](memory-benchmark-shared-consumer-2026-11-15.md)
- [CSV summary](memory-benchmark-shared-consumer-2026-11-15.csv)
- [JSON summary and comparison](memory-benchmark-shared-consumer-2026-11-15.json)

Four attempts failed before warm-up because the first decoded frame lacked
valid timing. These were excluded and rerun in fresh processes; diagnostic logs
remain with the raw data. Severe host memory pressure also limits interpretation
of resident-memory decreases. The PDF documents both limitations.

These results cover 640 x 360 ingestion without YOLO/VLM. MediaMTX and mock
publishers are excluded from production accounting; raw host totals retain them.

## Historical independent consumers

The `memory-benchmark-2026-11-15` reports measured one independent consumer per
camera. They remain valid records of that implementation, not measurements of
the current shared round-robin consumer.

The new benchmark's 0.75-second stress delay applies to every frame across the
whole pipeline, rather than independently per camera. The new report compares
the different processing capacities, frame drops and drain times explicitly.

The dated PDF, Markdown, CSV, and JSON files are preserved without changing
their measurements.
