"""Presentation videos for Stage 1 (YOLO26s, rules + pose cue, current GateConfig defaults), from cached runs on footage
not used for tuning.

  demo/stage1_works.mp4   one example per event where Stage 1 sends the right person or bag to the VLM
  demo/stage1_misses.mp4  typical misses, each preceded by a card saying why Stage 1 misses it

Usage (from notebooks/): ../.venv/bin/python tools/make_showcase.py [works|misses]   (default: both)
Clips are picked for illustration; the benchmark numbers are in README.md.
"""
import pickle
import sys
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam")]

from exp_harness import baseline  # noqa: E402
from render_demo import open_writer, render_sequence, title_card  # noqa: E402
from stage1_bench import evaluate_run  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"
OUT = NB / "demo"
FPS = 10
CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
SYSTEM = partial(baseline, cfg=CFG)
NOTE = "YOLO26s + pose + ByteTrack, rules + pose cue"
WHITE, GREY, ORANGE, RED = (255, 255, 255), (200, 200, 200), (90, 170, 255), (110, 110, 255)


def load(name):
    return pickle.loads((CACHE / f"{name}.pkl").read_bytes())


def find(data, seq_id):
    return next((g, r) for g, r in data if r.seq_id == seq_id)


def card(w, heading, lines, seconds=3.5, colour=WHITE):
    title_card(w, [(heading, 1.15, colour)] + [(s, 0.8, GREY) for s in lines], FPS, seconds=seconds)


def littering_clips(data, want_right_person):
    """One held-out clip per littering action where the right person is (or is not) sent."""
    picked = {}
    for g, r in data:
        if g.tag in picked or len(g.seq) != 30:
            continue
        ev = SYSTEM(r, "littering")
        row = evaluate_run(g, r, ev)
        if bool(row["ev_spatial"]) == want_right_person and row["ev_temporal"]:
            picked[g.tag] = (g, r, ev)
    return picked


def works():
    w = open_writer(OUT / "stage1_works.mp4", FPS)
    card(w, "Stage 1: what it sends to the VLM", ["Thick orange box and the panel on the right = candidate sent to the VLM",
                                                  "Thin red box / red timeline = the labelled incident",
                                                  "Footage not used for tuning"], 4.5)

    lit = littering_clips(load("litter_heldout"), want_right_person=True)
    card(w, "Littering", ["Held-out clips, one per action, at half speed",
                          "The pose cue fires on the person doing it, not the bystanders"])
    for tag in ("dropping", "throwing", "tossing", "dumping"):
        if tag in lit:
            g, r, ev = lit[tag]
            render_sequence(w, g, r, ev, f"Littering: {tag}", f"held-out clip {g.seq.seq_id}  |  {NOTE}", CFG,
                            repeat_frames=2, labels=[(0.0, 99.0, f"{tag} (thin red box)")])

    iwdd = load("iwdd_test")
    card(w, "Illegal dumping, from a separate dataset", ["MIVIA IWDD test set", "Labels give only the second dumping starts"])
    for vid, onset, what in (("vid0359", 17.0, "bin dumped from a van"), ("vid0075", 16.0, "bag dumped from a car boot")):
        g, r = find(iwdd, f"iwdd/test/{vid}")
        render_sequence(w, g, r, SYSTEM(r, "littering"), f"Illegal dumping: {what}", f"IWDD test {vid}  |  {NOTE}", CFG,
                        window=(onset - 9, onset + 5), labels=[(onset, onset + 4, what)], label_prefix="Labelled: ")

    aboda = load("aboda")
    card(w, "Unattended items", ["ABODA video 10: the bag is put down, then left alone",
                                 "Pose cue at the put-down, then the unattended-bag rule"])
    g, r = find(aboda, "aboda/video10")
    render_sequence(w, g, r, SYSTEM(r, "aboda"), "Unattended item", f"ABODA video10  |  {NOTE}", CFG,
                    window=(35, 61), labels=[(40.75, 48, "bag put down, owner walks away"), (48, 61, "bag left unattended")],
                    label_prefix="Labelled: ")
    card(w, "Unattended item the detector cannot see", ["ABODA video 2: YOLO never detects this dark bag on the paving,",
                                                         "but reach-down fires as the man bends to put it down,",
                                                         "and the crop includes the area around his hand"], 4.5)
    g, r = find(aboda, "aboda/video2")
    render_sequence(w, g, r, SYSTEM(r, "aboda"), "Unattended: via the person", f"ABODA video2  |  {NOTE}",
                    CFG, window=(63, 76), labels=[(69.5, 76, "bag put down and left (top right)")],
                    label_prefix="Labelled: ")

    avenue = load("avenue")
    card(w, "Littering: throwing on a walkway", ["CUHK Avenue held-out video test/12: a man throws his bag",
                                                 "into the air three times; the thrown bag and the man are sent"])
    g, r = find(avenue, "test/12")
    render_sequence(w, g, r, SYSTEM(r, "avenue"), "Throwing on a walkway", f"Avenue held-out test/12  |  {NOTE}", CFG,
                    window=(17, 36))
    w.release()
    print("wrote", OUT / "stage1_works.mp4", "| littering clips:", {k: v[0].seq.seq_id for k, v in lit.items()})


def misses():
    w = open_writer(OUT / "stage1_misses.mp4", FPS)
    card(w, "Stage 1: where it misses, and why", ["Each clip is a labelled incident that Stage 1 did not send to the VLM",
                                                  "or sent with the wrong person"], 4.0)
    iwdd = load("iwdd_test")
    cases = [
        ("vid0012", 2.0, (0, 9), "bag thrown from a passing car",
         "No person to detect", ["The dumper never leaves the car.",
                                  "Stage 1 looks for people acting; here there is only a moving car."]),
        ("vid0112", 6.0, (1, 9), "pixelated person dumps rubbish",
         "Privacy blurring hides the person", ["Much of IWDD is news footage with the dumper pixelated.",
                                                "YOLO does not find a person inside a blur block."]),
        ("vid0431", 5.0, (0, 12), "person unloads a van next to the camera",
         "Person cut off by the frame edge", ["The dumper is half out of frame, so wrist and knee keypoints are unreliable.",
                                               "Reach-down skips people touching the edge to avoid false alarms."]),
        ("vid0406", 3.0, (0, 5.6), "bag dropped beside a car",
         "The video ends as the dumping starts", ["This clip stops 2.6 s after the bag is dropped.",
                                                   "About a quarter of IWDD dumping clips end within 3 s of the onset,",
                                                   "too soon for a rule to react."]),
    ]
    for vid, onset, window, what, heading, why in cases:
        card(w, f"Miss: {heading}", why, 4.5, RED)
        g, r = find(iwdd, f"iwdd/test/{vid}")
        render_sequence(w, g, r, SYSTEM(r, "littering"), f"Missed: {heading.lower()}", f"IWDD test {vid}  |  {NOTE}", CFG,
                        window=window, labels=[(onset, onset + 4, what)], label_prefix="Labelled: ")

    card(w, "Miss: dark bag, night footage", ["ABODA video 6: the bag is never detected in this dark corridor,",
                                               "so the bag rules cannot fire, and the put-down is not caught",
                                               "by the person rules either."], 5.0, RED)
    g, r = find(load("aboda"), "aboda/video6")
    render_sequence(w, g, r, SYSTEM(r, "aboda"), "Missed: bag never detected", f"ABODA video6  |  {NOTE}", CFG,
                    window=(108, 132), labels=[(114, 132, "bag put down and left")], label_prefix="Labelled: ")

    wrong = littering_clips(load("litter_heldout"), want_right_person=False)
    if wrong:
        tag, (g, r, ev) = next(iter(wrong.items()))
        card(w, "Miss: right moment, wrong person", ["Stage 1 fires during the littering but crops a bystander.",
                                                     "Most littering misses on the original data are this kind."], 4.5, RED)
        render_sequence(w, g, r, ev, f"Wrong person: {tag}", f"held-out clip {g.seq.seq_id}  (half speed)  |  {NOTE}", CFG,
                        repeat_frames=2, labels=[(0.0, 99.0, f"{tag} (thin red box)")])
    w.release()
    print("wrote", OUT / "stage1_misses.mp4", "| wrong-person clip:", {k: v[0].seq.seq_id for k, v in wrong.items()})


if __name__ == "__main__":
    for job in sys.argv[1:] or ("works", "misses"):
        {"works": works, "misses": misses}[job]()
