"""Score vlm_cands.py candidates with the VLM (Yuchuan's ClipStrategy, OpenVINO encoder + YOLO26s boxes, as the
integration test's ov-yolo worker). Asks each candidate its routed tasks plus smoking for every person rule (the live
routing never asks the pose cue about smoking), and keeps the raw scores so thresholds can be re-chosen offline.
Resumes: candidates already in the output are skipped. Untrusted bundle code -> run with -I:
  ../.venv/bin/python -I exp/integration/vlm_scores.py BUNDLE OV_DIR CANDS_DIR [WORKER_INDEX N_WORKERS]
  -> CANDS_DIR/scores_<k>.jsonl  {"event_id", "seq", "event_type", "scores": {task: score | None}}
"""
import contextlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ov_vlm  # noqa: E402

YOLO = HERE.parents[2] / "data" / "models" / "stage1_ov" / "yolo26s_openvino_model"
EXTRA = {"reach_down": ("smoking",), "pose_action": ("smoking",), "long_presence": ("smoking",)}


def main(bundle, ov_dir, cands, k=0, n=1):
    cands = Path(cands)
    rows = [json.loads(l) for f in sorted(cands.glob("candidates_*.jsonl")) for l in f.read_text().splitlines()]
    rows = [r for i, r in enumerate(rows) if i % n == k]
    out = cands / f"scores_{k}.jsonl"
    done = {json.loads(l)["event_id"] for f in cands.glob("scores_*.jsonl") for l in f.read_text().splitlines()}
    with contextlib.redirect_stdout(sys.stderr):
        eng = ov_vlm.patch_detector_yolo(ov_vlm.patch(ov_vlm.engine(bundle), ov_dir), str(YOLO))
    t0 = time.perf_counter()
    with out.open("a") as fh:
        for i, c in enumerate(rows):
            if c["event_id"] in done:
                continue
            c["tasks"] = list(dict.fromkeys(c["tasks"] + list(EXTRA.get(c["event_type"], ()))))
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    res = eng.classify(c)["predictions"]
            except Exception as exc:   # recorded, not fatal: one bad candidate should not stop the run
                res = {t: {"decision": f"error: {exc}"} for t in c["tasks"]}
            fh.write(json.dumps({"event_id": c["event_id"], "seq": c["seq"], "event_type": c["event_type"],
                                 "timestamp_s": c["timestamp_s"], "roi_box": c["roi_box"],
                                 "scores": {t: v.get("score") for t, v in res.items()},
                                 "decisions": {t: v.get("decision") for t, v in res.items()}}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(k, i, len(rows), round(time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4], *map(int, sys.argv[4:6]))
