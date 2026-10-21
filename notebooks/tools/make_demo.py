"""Build the demo videos (tune-half footage only).

Usage (from notebooks/):  ../.venv/bin/python tools/make_demo.py [n|s|m]
Writes demo/loitering_demo.mp4 and demo/littering_demo.mp4 using that model size's cached detections.
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools")]

from exp_harness import baseline, load_splits  # noqa: E402
from render_demo import open_writer, render_sequence, title_card  # noqa: E402
from stage1_bench import evaluate_run  # noqa: E402
from stage1_gate import GateConfig, calibrate_inview  # noqa: E402

SIZE = sys.argv[1] if len(sys.argv) > 1 else "n"
MODEL_NOTE = f"YOLO26{SIZE} + ByteTrack + YOLO26{SIZE}-pose"
DEMO_VIDEO = "test/09"
OUT = NB / "demo"
OUT.mkdir(exist_ok=True)


def tune_data():
    if SIZE == "n":
        return load_splits(heldout=False)
    cache = NB / "bench_cache" / "models" / SIZE
    load = lambda name: pickle.loads((cache / f"{name}.pkl").read_bytes())
    avenue = load("avenue")
    odd = lambda g: int(g.seq.seq_id.split("/")[-1]) % 2 == 1
    return {"avenue": [x for x in avenue if x[0].seq.seq_id.startswith("test/") and odd(x[0])],
            "normal": [x for x in avenue if x[0].seq.seq_id.startswith("train/") and odd(x[0])],
            "littering": load("litter_train")}


DATA = tune_data()
CFG = GateConfig(inview_s=calibrate_inview([r for _, r in DATA["normal"]]))
SYSTEM = partial(baseline, cfg=CFG)


def make_loitering():
    g, r = next((g, r) for g, r in DATA["avenue"] if g.seq.seq_id == DEMO_VIDEO)
    ev = SYSTEM(r, "avenue")
    row = evaluate_run(g, r, ev)
    labels = json.loads((NB / "tools" / "demo_annotations.json").read_text())[DEMO_VIDEO]
    w = open_writer(OUT / "loitering_demo.mp4", r.fps)
    title_card(w, [("Stage 1 YOLO gate: loitering dataset", 1.2, (255, 255, 255)),
                   (f"video {DEMO_VIDEO}: {len(labels)} labelled anomalies (the red text says what each one is)", 0.8, (200, 200, 200)),
                   ("red = true anomaly    orange = video sent to the VLM", 0.8, (90, 170, 255))], r.fps, seconds=3.0)
    render_sequence(w, g, r, ev, "Loitering dataset", f"{DEMO_VIDEO}  |  {MODEL_NOTE}", CFG, labels=labels)
    w.release()
    print(f"loitering {DEMO_VIDEO}: {row['ev_temporal']}/{row['gt_events']} anomalies reached, "
          f"{row['ev_spatial']} with the right person, {len(ev)} candidates")


def make_littering():
    picked = {}
    for g, r in DATA["littering"]:
        if g.tag in picked or len(g.seq) != 30:
            continue
        ev = SYSTEM(r, "littering")
        if evaluate_run(g, r, ev)["ev_spatial"]:
            picked[g.tag] = (g, r, ev)
    w = open_writer(OUT / "littering_demo.mp4", 10)
    title_card(w, [("Stage 1 YOLO gate: littering", 1.2, (255, 255, 255)),
                   ("one clip per littering action; the candidate fires on the person doing it", 0.8, (200, 200, 200))], 10)
    for tag in ("dropping", "dumping", "putting-down", "throwing", "tossing"):
        if tag not in picked:
            continue
        g, r, ev = picked[tag]
        title_card(w, [(f"Littering: {tag}", 1.1, (255, 255, 255))], 10, seconds=1.2)
        render_sequence(w, g, r, ev, f"Littering: {tag}", f"clip {g.seq.seq_id}  (half speed)  |  {MODEL_NOTE}", CFG,
                        repeat_frames=2, labels=[(0.0, 99.0, f"{tag} (the person in the thin red box)")])
    w.release()
    print("littering clips:", {k: v[0].seq.seq_id for k, v in picked.items()})


if __name__ == "__main__":
    make_loitering()
    make_littering()
