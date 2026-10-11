# Report Progress

- **Report file**: `reports/anomaly_benchmark_report.md`
- **Completed sections**: Title + Executive Summary; 1 Introduction and Scope; 2 Data; 3 Evaluation Protocol; 4 Models, Prompts, Scoring (4.1–4.5); 5 Head-to-Head Results — 5.1 Run configuration, 5.2 Accuracy/macro-F1, 5.3 Binary F1/AUC, 5.4 Per-class breakdown, 5.5 SmolVLM systematic abstention
- **Next unfinished section**: 5.6 Latency and throughput (fig04), then 5.7 Memory (fig05), 5.8 Synthesis decision matrix; afterwards Section 6 (VadCLIP: implementation, adaptation decisions, feature cache/splits, baselines, ablation suite with fig06/fig07, training dynamics, interpretation), 7 Optimizations and experiments, 8 Discussion, 9 Limitations, 10 Reproducibility, Appendices A–G
- **Relevant result paths**: `results/anomaly_headtohead/comparison.csv`, `results/anomaly_headtohead/improvements.json`, `results/anomaly_headtohead/run.json`; VadCLIP: `results/vadclip/baselines/baselines.json`, `results/vadclip/features/manifest.json`, `results/vadclip/runs/{full,no_graph,win_1,win_16,vl_128,vl_512}/eval.json`
- **Figures**: fig01 → §5.2 (placed); fig02 → §5.3 (placed); fig03 → §5.4 (placed); pending: fig04 → §5.6, fig05 → §5.7, fig06/fig07 → Section 6
