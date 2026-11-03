"""Does Stage 1 still work at the data pipeline's 2 fps? (feat/data-pipeline selects 2 frames per second.)

Perception is re-run with YOLO26s at ~2 fps (tracking depends on the frame rate, so cached 10 fps runs cannot
just be subsampled) and the unchanged rules are scored next to the existing caches:
  littering held-out clips  10 fps -> stride 5 (2 fps)
  Avenue test + train       12.5 fps (stride 2) -> stride 12 (2.08 fps)
  IWDD test                 ~10 fps -> ~2 fps
Nothing is tuned here; in-view time is recalibrated on the 2 fps Avenue tune normals the same way as at full rate.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/fps_check.py run [fps]   -> bench_cache/models/s_<fps>fps/*.pkl (default 2)
          ../.venv/bin/python exp/newcam/fps_check.py eval        -> exp/newcam/fps_check.json (every rate that has caches)
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from stage1_bench import avenue_sequences, littering_sequences, run_perception  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, ImageSequence, Stage1Gate, calibrate_inview  # noqa: E402

D = NB.parent / "data"
FULL = NB / "bench_cache" / "models" / "s"


def restride(gts, stride):
    for g in gts:
        s = g.seq
        g.seq = ImageSequence(s.paths, fps=s.fps, seq_id=s.seq_id, stride=stride)
        g.gt = {i: b for i, b in g.gt.items() if i % stride == 0}
    return gts


def cache_dir(fps):
    return NB / "bench_cache" / "models" / f"s_{fps:g}fps"


def run(fps=2.0):
    from loaders import iwdd_sequences
    out = cache_dir(fps)
    out.mkdir(parents=True, exist_ok=True)
    cfg = GateConfig(det_weights="yolo26s.pt", pose_weights="yolo26s-pose.pt", device="mps")
    jobs = {
        "litter_heldout": lambda: restride(littering_sequences(D / "littering", splits=("valid", "test")), round(10 / fps)),
        "avenue": lambda: avenue_sequences(D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" /
                                           "testing_label_mask", stride=round(25 / fps)),
        "iwdd_test": lambda: iwdd_sequences("test", target_fps=fps),
    }
    for name, build in jobs.items():
        if (out / f"{name}.pkl").exists():
            continue
        res = run_perception(Stage1Gate(cfg), build(), verbose=False)
        (out / f"{name}.pkl").write_bytes(pickle.dumps(res))
        print(f"{name} done ({len(res)} sequences)", flush=True)


def splits(cache):
    load = lambda n: pickle.loads((cache / f"{n}.pkl").read_bytes())  # noqa: E731
    avenue = load("avenue")
    num = lambda g: int(g.seq.seq_id.split("/")[-1])  # noqa: E731
    pick = lambda pre, want: [x for x in avenue if x[0].seq.seq_id.startswith(pre) and num(x[0]) % 2 == want]  # noqa: E731
    return {"littering": load("litter_heldout"), "avenue": pick("test/", 0), "normal": pick("train/", 0),
            "tune_normal": pick("train/", 1)}


def evaluate():
    from exp_harness import baseline
    from iwdd_eval import evaluate as iwdd_evaluate
    from loaders import iwdd_labels
    from stage1_bench import evaluate_run, summarize
    out = {}
    rates = sorted((float(p.name[2:-3]), p) for p in FULL.parent.glob("s_*fps") if (p / "iwdd_test.pkl").exists())
    for label, cache in [("full rate", FULL)] + [(f"{f:g} fps", p) for f, p in rates]:
        S = splits(cache)
        inview = calibrate_inview([r for _, r in S["tune_normal"]])
        row = {"inview_s": round(inview, 1)}
        for name, rules in (("rules", DEFAULT_RULES), ("rules + pose", DEFAULT_RULES + ("pose_action",))):
            m = partial(baseline, cfg=GateConfig(inview_s=inview, rules=rules))
            lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
            av = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in S["avenue"]], "a")
            fa = summarize([evaluate_run(g, r, m(r, "normal")) for g, r in S["normal"]], "n")
            iw = iwdd_evaluate(pickle.loads((cache / "iwdd_test.pkl").read_bytes()), iwdd_labels("test"), m)
            row[name] = {"littering_event_%": lit["event_recall_%"], "littering_crop_%": lit["crop_recall_%"],
                         "avenue_event_%": av["event_recall_%"], "avenue_crop_%": av["crop_recall_%"],
                         "normal_FA_per_min": fa["candidates_per_min"],
                         "iwdd_hit@10s_%": iw["hit@10s_%"], "iwdd_chance@10s_%": iw["chance@10s_%"],
                         "iwdd_neg_cand_per_min": iw["neg_cand_per_min"]}
        lat = [r.mean_latency_ms for _, r in S["normal"]]
        row["ms_per_frame"] = round(sum(lat) / len(lat), 1)
        out[label] = row
        print(label, json.dumps(row), flush=True)
    (HERE / "fps_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*map(float, sys.argv[2:]))
    else:
        evaluate()
