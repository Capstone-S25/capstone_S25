"""VLM candidates for every gate alert on one ICVL half, without re-running perception: the default gate's alerts come
from the cached run (replay_rules with the cached crop-check scores, as fpr_attrib.py scores them), and each alert's
crop, context frame and short / long views are cut from the video with the same HandoffWriter settings and windows
the data pipeline's Stage1Detector uses, so vlm_scores.py can ask the VLM about exactly what it would see live.
Writes OUT/<half>/candidates.jsonl (plus the jpgs). Usage (from notebooks/):
  ../.venv/bin/python exp/integration/vlm_cands.py odd|even OUT [WORKER_INDEX N_WORKERS]
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "exp" / "icvl"), str(NB.parent / "data-pipeline")]

from data_pipeline.detection.stage1 import TASKS_BY_EVENT, VIEW_WINDOWS_S  # noqa: E402
from fpr_attrib import CFG  # noqa: E402  (also loads the cached crop-check scores)
from inventory import ICVL, video_for  # noqa: E402
from score import load_runs  # noqa: E402
from stage1_gate import HandoffWriter, replay_rules  # noqa: E402

BUFFER_S = max(VIEW_WINDOWS_S.values()) + 5.0    # as Stage1Detector


def main(half, out, k=0, n=1):
    out = Path(out) / half
    out.mkdir(parents=True, exist_ok=True)
    writer = HandoffWriter(out, buffer_s=BUFFER_S)
    with (out / f"candidates_{k}.jsonl").open("w") as fh:
        for i, (g, res) in enumerate(load_runs(half)):
            if i % n != k:
                continue
            alerts = sorted(replay_rules(res, CFG)[0], key=lambda a: a["timestamp_s"])
            if not alerts:
                continue
            stream = SimpleNamespace(seq_id=res.seq_id.replace("/", "_"), source=res.seq_id, paths=None)
            writer.start(stream)
            spans = [(a["timestamp_s"] - BUFFER_S - 1, a["timestamp_s"] + 0.01) for a in alerts]
            need = [o for o in res.observations if any(lo <= o.t <= hi + 1e-6 for lo, hi in spans)]
            cap = cv2.VideoCapture(str(video_for(ICVL / g.tag)))
            fps, pos, j = g.seq.native_fps, 0, 0
            for o in need:
                target = round(o.t * fps)
                while pos < target:
                    cap.grab()
                    pos += 1
                ok, img = cap.read()
                pos += 1
                if not ok:
                    break
                writer.observe(img, o.t, o.index)
                while j < len(alerts) and alerts[j]["timestamp_s"] <= o.t + 0.006:   # alert times are rounded to 0.01 s
                    e = dict(alerts[j])
                    j += 1
                    e["event_id"] = f"{stream.seq_id}_{e['event_id']}"
                    te = o.t
                    e["observed_until_s"] = te    # as Stage1Detector: the VLM refuses views later than this
                    writer.save(e, img, stream, t=te)
                    e["views"] = {}
                    for name, span in VIEW_WINDOWS_S.items():
                        paths, times = writer.window_frames(e, img, stream, te - span, te, tag=f"_{name}")
                        e["views"][name] = {"paths": paths, "times_s": times}
                    e["tasks"] = list(TASKS_BY_EVENT.get(e["event_type"], ()))
                    e["seq"] = res.seq_id
                    fh.write(json.dumps(e, default=float) + "\n")
            cap.release()
            writer.manifest.clear()
            fh.flush()
            print(i, res.seq_id, len(alerts), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:5]))
