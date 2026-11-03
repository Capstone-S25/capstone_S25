"""CPU time per frame for the Stage 1 gate (current defaults, rules + pose cue), broken down by part.

Runs the full gate on CPU at 5 fps on three kinds of footage and reports median / 95th percentile milliseconds for
decoding + resizing (read), the detector + tracker (det), pose, the rules (rules) and the whole frame. Real time at
5 fps needs the whole frame under 200 ms. Run once with all cores and once limited to one thread (a cheap machine).
Usage (from notebooks/): ../.venv/bin/python tools/cpu_timing.py [threads]   -> exp/cpu_timing_<threads>.json
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NB = Path(__file__).resolve().parents[1]
D = NB.parent / "data"
sys.path[:0] = [str(NB)]

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 0

from stage1_gate import DEFAULT_RULES, GateConfig, Perception, RuleEngine, VideoSequence  # noqa: E402

CLIPS = {
    "walkway, busy (Avenue, 640x360)": (D / "loitering" / "testing_videos" / "04.avi", 5),
    "indoor CCTV (ABODA, 720x480)": (D / "additional" / "aboda" / "video9.avi", 6),
    "street, 1080p (IWDD)": (D / "training_sources_20261015" / "mivia_iwdd" / "test" / "videos" / "vid0359.mp4", 6),
}
MAX_FRAMES = 150  # 30 s at 5 fps


def pct(v):
    return {"median": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}


def main():
    cfg = GateConfig(det_weights=str(NB / "yolo26s.pt"), pose_weights=str(NB / "yolo26s-pose.pt"), device="cpu",
                     inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
    perc = Perception(cfg)
    if THREADS:
        # Ultralytics resets torch's thread count on a model's first prediction, so warm both models up first.
        import numpy as _np
        import torch
        perc.reset(5.0)
        dummy = _np.zeros((480, 640, 3), _np.uint8)
        perc.det.predict(dummy, imgsz=cfg.imgsz, device="cpu", verbose=False)
        perc.pose.predict(dummy, imgsz=cfg.imgsz, device="cpu", verbose=False)
        torch.set_num_threads(THREADS)
    out = {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
           or platform.processor(), "threads": THREADS or "all", "fps": 5}
    for name, (path, stride) in CLIPS.items():
        seq = VideoSequence(path, stride=stride)
        perc.reset(seq.fps)
        eng = RuleEngine(cfg, seq.fps)
        rows = {"read": [], "det": [], "pose": [], "rules": [], "frame": []}
        it = iter(seq)
        warm = 5
        persons = []
        for k in range(MAX_FRAMES + warm):
            t0 = time.perf_counter()
            try:
                fr = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            obs = perc.observe(fr.image, fr.index, fr.t)
            t2 = time.perf_counter()
            eng.step(obs)
            t3 = time.perf_counter()
            if k < warm:
                continue
            rows["read"].append((t1 - t0) * 1000)
            rows["det"].append(obs.latency_ms.get("det_ms", 0.0))
            rows["pose"].append(obs.latency_ms.get("pose_ms", 0.0))
            rows["rules"].append((t3 - t2) * 1000)
            rows["frame"].append((t3 - t0) * 1000)
            persons.append(len(obs.persons))
        out[name] = {part: pct(v) for part, v in rows.items()}
        out[name]["people_per_frame"] = round(float(np.mean(persons)), 1)
        out[name]["frames"] = len(rows["frame"])
        print(name, json.dumps(out[name]), flush=True)
    (NB / "exp" / f"cpu_timing_{THREADS or 'all'}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
