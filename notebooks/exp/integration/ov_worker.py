"""Resident VLM worker for the integration test: Yuchuan's ClipStrategy (from his CPU bundle) with OpenVINO speed-ups,
speaking the same JSON-lines protocol as anomaly_eval.reasoning.worker (a "ready" line, then {"id", "candidate"} ->
{"id", "result"}), so the data pipeline's ResidentClient / CandidateService drive it unchanged.
  python -I ov_worker.py BUNDLE OV_DIR VARIANT
  VARIANT: torch (unchanged), ov (OpenVINO encoder + Faster R-CNN, same answers), ov-yolo (OpenVINO encoder + YOLO26s
  boxes instead of Faster R-CNN)
"""
import contextlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
YOLO = HERE.parents[2] / "data" / "models" / "stage1_ov" / "yolo26s_openvino_model"


def main():
    bundle, ov_dir, variant = sys.argv[1:4]
    protocol = sys.stdout
    send = lambda v: (protocol.write(json.dumps(v, allow_nan=False, default=str) + "\n"), protocol.flush())
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(HERE))
        import ov_vlm
        started = time.perf_counter()
        eng = ov_vlm.engine(bundle)
        if variant in ("ov", "ov-yolo"):
            ov_vlm.patch(eng, ov_dir)
            ov_vlm.patch_detector_yolo(eng, str(YOLO)) if variant == "ov-yolo" else ov_vlm.patch_detector(eng, ov_dir, bundle)
        send({"type": "ready", "metadata": {**eng.metadata, "variant": variant}, "load_seconds": time.perf_counter() - started})
        for line in sys.stdin:
            ident = None
            try:
                request = json.loads(line)
                ident = request["id"]
                if request.get("method") == "close":
                    break
                send({"id": ident, "result": eng.classify(request["candidate"])})
            except Exception as exc:
                send({"id": ident, "error": str(exc), "error_type": type(exc).__name__})


if __name__ == "__main__":
    main()
