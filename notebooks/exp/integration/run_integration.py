"""YOLO + VLM integration test on recorded video, paced like live cameras, CPU only.

One process stands in for one box PC: N camera threads each play held-out ICVL videos back to back at real time
through the data pipeline's own decoder and 5 fps frame selection into a data-pipeline Stage1Detector (gate with the
crop check), and all cameras share one CandidateService (the pipeline's bounded VLM queue) with one resident VLM
worker (exp/integration/ov_worker.py: Yuchuan's MobileCLIP2-S2 classifier from his CPU bundle, optionally OpenVINO).

Recorded per frame: how late it started vs its real-time slot and how long the gate took. Per candidate (from the
pipeline's journal): queue and service seconds; end-to-end latency = (release time - alert time, the crop-check hold)
+ queue + service. Usage (from notebooks/):
  ../.venv/bin/python exp/integration/run_integration.py --cameras 1 --minutes 5 --gate ov --vlm ov --out OUT
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
REPO = NB.parent
sys.path[:0] = [str(NB), str(NB / "exp" / "icvl"), str(REPO / "data-pipeline")]

from data_pipeline.detection.reasoning import CandidateService, RecordOnly, ResidentClient  # noqa: E402
from data_pipeline.detection.stage1 import Stage1Detector, default_config, load_gate  # noqa: E402
from data_pipeline.video_ingestion.decoding.decoder import decode_segment  # noqa: E402
from data_pipeline.video_ingestion.decoding.selection import select_frames  # noqa: E402
from data_pipeline.video_ingestion.scheduling.worker import SegmentJob  # noqa: E402

from inventory import ICVL, video_for  # noqa: E402
from score import load_runs  # noqa: E402

CAMERAS = ("05", "04", "02", "07", "03", "06", "01", "10")   # busiest first
OV = REPO / "data" / "models" / "stage1_ov"


def playlists(n, minutes):
    """Per camera: held-out (even-half) ICVL videos from one ICVL camera, back to back, up to `minutes`."""
    out = []
    by_cam = defaultdict(list)
    for g, res in load_runs("even"):
        by_cam[res.seq_id.split("/")[1]].append((g, res))
    for cam in CAMERAS[:n]:
        total, chosen = 0.0, []
        for g, res in sorted(by_cam[cam], key=lambda x: x[1].seq_id):
            if total >= minutes * 60:
                break
            chosen.append({"seq": res.seq_id, "video": str(video_for(ICVL / g.tag)), "duration_s": res.duration_s})
            total += res.duration_s
        out.append({"camera": cam, "videos": chosen, "seconds": total})
    return out


def camera(k, plan, cfg, gate, service, out, start_wall, log):
    source = f"cam{k}_{plan['camera']}"
    det = Stage1Detector(source, out, fps=5.0, cfg=cfg, gate=gate, safety_scan_seconds=ARGS.safety_scan)
    offset = 0.0
    with (out / f"{source}_frames.jsonl").open("w") as fh:
        for v in plan["videos"]:
            video = Path(v["video"])
            session = uuid4()
            log.append({"source": source, "session_id": str(session), "seq": v["seq"], "stream_offset_s": offset})
            job = SegmentJob(0, source, session, video.name, 0, 0)
            first_ms = None
            with closing(decode_segment(job, video.parent)) as frames:
                for frame in select_frames(frames, target_fps=5.0):
                    first_ms = frame.timestamp_ms if first_ms is None else first_ms
                    due = start_wall + offset + (frame.timestamp_ms - first_ms) / 1000
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    began = time.perf_counter()
                    cands = det.process(frame)
                    took = time.perf_counter() - began
                    t_now = (frame.timestamp_ms - first_ms) / 1000
                    for c in cands:
                        c["integration"] = {"hold_s": max(0.0, t_now - float(c["timestamp_s"])), "late_s": began - due,
                                            "seq": v["seq"], "camera": plan["camera"]}
                        service.submit(c)
                    fh.write(json.dumps({"due": round(due - start_wall, 3), "late_s": round(began - due, 4),
                                         "gate_s": round(took, 4), "candidates": len(cands)}) + "\n")
            offset += v["duration_s"]


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", type=int, default=1)
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--gate", choices=["torch", "ov"], default="ov")
    p.add_argument("--vlm", choices=["none", "torch", "ov", "ov-yolo"], default="ov")
    p.add_argument("--safety-scan", type=float, default=0.0, help="periodic whole-scene VLM review every N s (0 = off)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--encoder", required=True, help="MobileCLIP2-S2 snapshot folder (crop check)")
    p.add_argument("--queue", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera-index", type=int, help="run only this camera of the plan (one process per camera)")
    p.add_argument("--start-at", type=float, help="wall-clock (time.time()) start shared by per-camera processes")
    ARGS = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = ARGS.out
    out.mkdir(parents=True, exist_ok=True)
    gate = load_gate()
    cfg = default_config(gate, "cpu")
    if ARGS.gate == "ov":
        cfg = replace(cfg, det_weights=str(OV / "yolo26s_openvino_model"), pose_weights=str(OV / "yolo26s-pose_openvino_model"))
    else:
        cfg = replace(cfg, det_weights=str(NB / "yolo26s.pt"), pose_weights=str(NB / "yolo26s-pose.pt"))
    cfg = replace(cfg, crop_check_encoder="local-dir:" + ARGS.encoder)
    os.environ.update(HF_HUB_CACHE=str(Path(ARGS.bundle) / "data/models/huggingface/hub"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    if ARGS.vlm == "none":
        client = RecordOnly()
    else:
        client = ResidentClient([sys.executable, "-I", str(HERE / "ov_worker.py"), ARGS.bundle,
                                 str(REPO / "data" / "models" / "vlm_ov"), ARGS.vlm],
                                ARGS.bundle, out / "worker.log", timeout=300)
    service = CandidateService(out, client, max_pending=ARGS.queue)
    t = time.perf_counter()
    service.start()
    load_s = time.perf_counter() - t
    plans = playlists(ARGS.cameras, ARGS.minutes)
    if ARGS.camera_index is not None:
        plans = [plans[ARGS.camera_index]]
    log = []
    start_wall = time.perf_counter() + (ARGS.start_at - time.time() if ARGS.start_at else 2.0)
    threads = [threading.Thread(target=camera, args=(k, plan, cfg, gate, service, out, start_wall, log))
               for k, plan in enumerate(plans)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    stream_end = time.perf_counter()
    service.close()
    drained = time.perf_counter()
    (out / "run.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(ARGS).items()}, "vlm_load_s": load_s,
                                              "plans": plans, "sessions": log, "stream_s": stream_end - start_wall,
                                              "drain_after_stream_s": drained - stream_end, "counts": service.counts}, indent=1))
    print(json.dumps({"counts": service.counts, "drain_after_stream_s": round(drained - stream_end, 1)}))


if __name__ == "__main__":
    main()
