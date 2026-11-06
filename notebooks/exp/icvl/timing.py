"""Average CPU time of each Stage 1 part on real ICVL footage (held-out half), for the compute slide.
The live gate (OpenVINO YOLO26s + YOLO26s-pose, ByteTrack, 5 fps, default threads) runs on N videos spread over the
even half; per frame: detector, pose (frames with someone in view only), tracking + bookkeeping (the rest of
Perception.observe), rules with and without the pose cue (RuleEngine.step on the same observations, crop check off).
The crop check (MobileCLIP2-S2 image encoder) is timed per crop on person crops from those videos; an alert uses 3.
Run with nothing else busy on the CPU.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/timing.py [N_VIDEOS]   -> exp/icvl/timing.json
"""
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

from inventory import ICVL  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import load_runs  # noqa: E402
from stage1_crop_check import CropCheck, crop  # noqa: E402
from stage1_gate import RuleEngine, Stage1Gate, load_sequence  # noqa: E402

OV = NB.parent / "data" / "models" / "stage1_ov"


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()


def time_rules(results, cfg):
    """ms per frame of RuleEngine.step over cached observations."""
    total, frames = 0.0, 0
    for r in results:
        eng = RuleEngine(cfg, r.fps)
        t0 = time.perf_counter()
        for o in r.observations:
            eng.step(o)
        total += time.perf_counter() - t0
        frames += len(r.observations)
    return 1000 * total / frames


def main():
    n_videos = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    runs = load_runs("even")
    picks = [g for g, _ in runs[:: max(1, len(runs) // n_videos)]][:n_videos]
    cfg = replace(CFG, det_weights=str(OV / "yolo26s_openvino_model"), pose_weights=str(OV / "yolo26s-pose_openvino_model"),
                  device="cpu", crop_check=False)
    gate = Stage1Gate(cfg)
    gate.run(load_sequence((ICVL / picks[0].tag).with_suffix(".avi"), fps=5.0, max_frames=5))   # warm-up
    results, crops = [], []
    for g in picks:
        r = gate.run(load_sequence((ICVL / g.tag).with_suffix(".avi"), fps=5.0))
        results.append(r)
        print(g.tag, len(r.observations), "frames", flush=True)
        if len(crops) < 40:
            seq = load_sequence((ICVL / g.tag).with_suffix(".avi"), fps=5.0)
            for fr, o in zip(seq, r.observations):
                if o.persons and fr.index % 10 == 0:
                    crops.append(crop(fr.image, o.persons[0]["box"]))
                if len(crops) >= 40:
                    break
    lat = [o.latency_ms for r in results for o in r.observations]
    det = [x["det_ms"] for x in lat]
    pose = [x["pose_ms"] for x in lat if "pose_ms" in x]
    track = [x["total_ms"] - x["det_ms"] - x.get("pose_ms", 0.0) for x in lat]
    total = [x["total_ms"] for x in lat]
    rules_with = time_rules(results, cfg)
    rules_without = time_rules(results, replace(cfg, rules=tuple(r for r in cfg.rules if r != "pose_action")))
    cc = CropCheck()
    cc.embed_crop(crops[0])   # load + warm-up
    ct = []
    for c in crops:
        t0 = time.perf_counter()
        cc.embed_crop(c)
        ct.append(1000 * (time.perf_counter() - t0))
    mean = lambda v: round(float(np.mean(v)), 2)
    p95 = lambda v: round(float(np.percentile(v, 95)), 1)
    out = {
        "machine": cpu_name(), "videos": len(results), "frames": len(lat), "fps": 5.0,
        "share_of_frames_with_pose": round(len(pose) / len(lat), 3),
        "ms": {
            "YOLO26s detector (per frame)": [mean(det), p95(det)],
            "ByteTrack + bookkeeping (per frame)": [mean(track), p95(track)],
            "YOLO26s-pose (per frame it runs)": [mean(pose), p95(pose)],
            "YOLO26s-pose (averaged over all frames)": [round(sum(pose) / len(lat), 2), None],
            "Rules incl. pose cue + reach verifier (per frame)": [round(rules_with, 3), None],
            "Pose cue alone (per frame)": [round(rules_with - rules_without, 3), None],
            "Crop check (per crop; 3 per alert)": [mean(ct), p95(ct)],
            "Gate total (per frame, no crop check)": [mean(total), p95(total)],
        },
        "note": "[mean, p95] ms; OpenVINO models, default threads; crop check is torch open_clip on the CPU.",
    }
    print(json.dumps(out, indent=1))
    (HERE / "timing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
