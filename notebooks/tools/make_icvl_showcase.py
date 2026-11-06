"""Presentation videos for Stage 1 on ICVL (real CCTV), current GateConfig defaults, from the cached ICVL run.

  demo/icvl_works.mp4    littering (reach-down, pose cue, item appeared), smoking and lingering caught
  demo/icvl_misses.mp4   typical misses and false alerts, each preceded by a card saying why
  demo/icvl_activities.mp4  one caught example per activity (casual littering, dumping, smoking, loitering), today's gate

The thin red box is the ICVL-labelled person during the labelled event. Clips come from the even half of ICVL (never
used for tuning) except the lingering tracking miss. Picked from exp/icvl/examples.json and the miss analysis.
Usage (from notebooks/): ../.venv/bin/python tools/make_icvl_showcase.py [works|misses|activities]   (default: works, misses)
"""
import sys
from bisect import bisect_left
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp" / "icvl")]

from make_showcase import GREY, RED, WHITE  # noqa: E402
from render_demo import open_writer, render_sequence, title_card  # noqa: E402
from score import labels_for, load_runs  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

OUT = NB / "demo"
FPS = 10
CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
NOTE = "ICVL  |  YOLO26s + pose + ByteTrack, rules + pose cue"
RUNS = {r.seq_id: (g, r) for g, r in load_runs()}


def card(w, heading, lines, seconds=4.0, colour=WHITE):
    title_card(w, [(heading, 1.15, colour)] + [(s, 0.8, GREY) for s in lines], FPS, seconds=seconds)


def clip(w, seq, pid, t0, t1, what, title, before=5.0, after=4.0, slow=2):
    """Render seq around the labelled event [t0, t1] of person pid, with that person's labelled box in thin red."""
    g, r = RUNS[f"icvl/{seq}"]
    _, people = labels_for(g.tag, g.seq.native_fps)
    tr = people[pid]
    g.gt = {}
    for o in r.observations:
        if t0 - 0.2 <= o.t <= t1 + 0.2:
            k = min(bisect_left(tr["ts"], o.t), len(tr["ts"]) - 1)
            g.gt[o.index] = [tr["boxes"][k]]
    render_sequence(w, g, r, replay_rules(r, CFG)[0], title, f"{seq}  |  {NOTE}", CFG, repeat_frames=slow,
                    window=(max(0.0, t0 - before), t1 + after), labels=[(t0, t1, what)], label_prefix="Labelled: ")


def works():
    w = open_writer(OUT / "icvl_works.mp4", FPS)
    card(w, "Stage 1 on real CCTV (ICVL)", ["Thick orange box and the panel on the right = candidate sent to the VLM",
                                            "Thin red box = the person ICVL labels as doing the event",
                                            "Videos never used for tuning, current settings"], 5.0)
    card(w, "Littering: reach-down", ["The wrist comes down to knee height while the person is slow or still,",
                                      "so the person and the area around the hand go to the VLM  (half speed)"])
    clip(w, "01/AA_014_01_20170829_01", 2, 4.6, 8.1, "littering", "Caught by reach-down")
    card(w, "Littering: pose cue", ["A small trained model on 1 s of body points scores the throwing motion",
                                    "(half speed)"])
    clip(w, "01/AA_001_01_20170829_01", 1, 4.1, 7.4, "littering", "Caught by the pose cue")
    card(w, "Littering: item appeared", ["A bag or bottle appears where nobody is standing and stays put",
                                         "(half speed)"])
    clip(w, "01/AA_052_01_20170829_01", 1, 6.3, 11.4, "littering", "Caught by 'item appeared'")
    card(w, "Smoking", ["The pose cue fires on the hand-to-face motion; the VLM decides if it is a cigarette"])
    clip(w, "05/AA_005_05_20140802_03", 1, 1.0, 15.6, "smoking", "Smoking: caught by the pose cue", before=1, after=2, slow=1)
    card(w, "Lingering (loitering stand-in)", ["Stays put: the same person within 0.6 body-heights of one spot for 20 s"])
    clip(w, "08/AA-006-08-20140524-02", 1, 7.5, 33.2, "standing still", "Lingering: caught by stays put",
         before=3, after=3, slow=1)
    w.release()
    print("wrote", OUT / "icvl_works.mp4")


class Counted:
    """VideoWriter wrapper that counts frames, so chapter times can be printed for the slide."""
    def __init__(self, w):
        self.w, self.n = w, 0

    def write(self, img):
        self.w.write(img)
        self.n += 1

    def release(self):
        self.w.release()


def activities():
    """One caught example per activity with today's gate (crop check on, cached scores): casual littering, dumping,
    smoking, loitering (in view 60 s+). Picked from exp/icvl/demo_picks.py, even half."""
    from fpr_attrib import CFG as NOW   # current defaults + pose cue; loads the cached crop-check scores
    w = Counted(open_writer(OUT / "icvl_activities.mp4", FPS))
    chapters = []

    def part(heading, lines, seq, pid, t0, t1, what, title, **kw):
        chapters.append((w.n / FPS, heading))   # no title card: each clip's own title bar names it
        g, r = RUNS[f"icvl/{seq}"]
        _, people = labels_for(g.tag, g.seq.native_fps)
        tr = people[pid]
        g.gt = {}
        for o in r.observations:
            if t0 - 0.2 <= o.t <= t1 + 0.2:
                k = min(bisect_left(tr["ts"], o.t), len(tr["ts"]) - 1)
                g.gt[o.index] = [tr["boxes"][k]]
        before, after = kw.pop("before", 5.0), kw.pop("after", 4.0)
        render_sequence(w, g, r, replay_rules(r, NOW)[0], title, f"{seq}  |  {NOTE}", NOW,
                        window=(max(0.0, t0 - before), t1 + after), labels=[(t0, t1, what)],
                        label_prefix="Labelled: ", **kw)

    part("Littering, casual: dropped while walking", ["He lets the bag fall without stopping or bending.",
                                                      "Reach-down: the hand comes down near knee height"],
         "05/AA_122_05_20170831_01", 2, 5.9, 6.7, "littering", "Casual littering: reach-down", repeat_frames=2)
    part("Littering, dumping: stops and puts it down", ["He walks up, bends and adds his bag to the pile.",
                                                        "Reach-down: one alert, as he bends"],
         "03/AA_123_03_20170901_01", 1, 13.2, 15.3, "littering", "Dumping: caught by reach-down", repeat_frames=2)
    part("Smoking", ["Both men are smoking. The pose cue fires on each one's hand-to-mouth motion;",
                     "the VLM decides if it is a cigarette  (slowed down)"],
         "10/AR-001-10-20151003-01", 1, 27.0, 31.0, "smoking", "Smoking: both caught, pose cue",
         before=1.0, after=1.0, repeat_frames=3)
    part("Loitering: in view 60 s+", ["Long in view: the same person has been around one area for 60 s,",
                                      "moving or not  (4x speed)"],
         "07/AR-001-07-20151003-01", 1, 19.7, 85.0, "in view", "Loitering: caught by long in view",
         before=1, after=6, stride=2, hold_s=6.0)
    w.release()
    print("wrote", OUT / "icvl_activities.mp4", f"{w.n / FPS:.0f} s")
    for t, h in chapters:
        print(f"{int(t // 60)}:{int(t % 60):02d}  {h}")


def misses():
    w = open_writer(OUT / "icvl_misses.mp4", FPS)
    card(w, "Stage 1 on real CCTV: misses and false alerts", ["Each clip is a labelled event Stage 1 did not send,",
                                                              "or an alert on someone doing nothing wrong"], 4.5)
    card(w, "Miss: the hand never gets low enough", ["The bag is let go at hip height beside the bushes.",
                                                     "Reach-down needs the wrist near knee height.",
                                                     "29 of the 36 held-out littering misses are this kind."], 5.0, RED)
    clip(w, "03/AA_011_03_20170828_01", 2, 8.0, 8.6, "littering", "Missed: hand not low enough")
    card(w, "Miss: dropped mid-stride", ["The person keeps walking as they drop it.",
                                         "Reach-down skips people walking faster than 0.45 body-heights a second,",
                                         "because a bag swinging from a walker's hand looks the same from above.",
                                         "17 of the 36 misses (some are both kinds)."], 5.5, RED)
    clip(w, "05/AA_038_05_20170829_02", 14, 6.2, 6.8, "littering", "Missed: dropped while walking")
    card(w, "Miss: one quick puff in a crowd", ["Most smoking misses are a single 1-3 s puff.",
                                                "The pose cue fires at most once per person every 15 s and",
                                                "about half of these never score high enough."], 5.0, RED)
    clip(w, "05/AR-004-05-20151003-01", 12, 144.7, 148.6, "smoking", "Missed: brief smoking", slow=1)
    card(w, "Miss: tracking breaks", ["Sitting backlit in front of a glass door, she is found in half the frames",
                                      "and gets 5 different track IDs, so the 20 s stays-put timer keeps restarting.",
                                      "One of only 2 of 126 people still for 20 s+ with no alert at all."], 5.5, RED)
    clip(w, "04/AA-012-04-20140524-02", 7, 28.2, 53.8, "sitting still", "Missed: tracking broke", before=3, after=2, slow=1)

    card(w, "False alerts: what is still sent", ["1.6 alerts a minute on footage with nothing labelled.",
                                                 "The remaining ones look like the real thing to body points alone."], 4.5)
    for seq, t, title, lines in (
            ("01/AA-016-01-20140502-01", 9.4, "False alert: carrying something low",
             ["A walker with something in his hand: the wrist hangs at knee height."]),
            ("03/AR-004-03-20151003-01", 3.4, "False alert: phone to the face",
             ["Raising a phone is the same motion as smoking for the pose cue."]),
            ("02/AR-021-02-20151031-01", 3.4, "False alert: pushing a bicycle",
             ["Riders are skipped, but a person walking a bike bends towards it."])):
        card(w, title, lines, 3.5)
        g, r = RUNS[f"icvl/{seq}"]
        g.gt = {}
        render_sequence(w, g, r, replay_rules(r, CFG)[0], title, f"{seq}  |  {NOTE}", CFG,
                        window=(max(0.0, t - 4), t + 3), repeat_frames=2)
    w.release()
    print("wrote", OUT / "icvl_misses.mp4")


if __name__ == "__main__":
    for job in sys.argv[1:] or ("works", "misses"):
        {"works": works, "misses": misses, "activities": activities}[job]()
