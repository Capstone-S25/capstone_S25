"""What is in ICVL (real surveillance video, 12 cameras), and how much of each event we care about?

Annotation rows (one per person per frame): Frame, Head.X, Head.Y, Feet.X, Feet.Y, ObjC, ObjId, then four layers
(ActionViewer ICVL_action_structure.int2stringLabel):
  posture   0 sitting, 1 standing, 2 lying, 3 bending
  movement  0 stationary, 1 walking, 2 running, 3 bicycling
  gesture   0 nothing, 1 texting, 2 smoking, 3 phoning, 9 others
  event     0 nothing, 1 littering, 2 falling
Per video: frames, fps, resolution, people; per event: episodes (continuous runs per person, gaps <= 1 s merged) and
seconds. Loitering proxy: a person stationary (standing or sitting) continuously for >= 20 s / >= 60 s.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/inventory.py   -> exp/icvl/inventory.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

NB = Path(__file__).resolve().parents[2]
ICVL = NB.parent / "data" / "icvl"
POSTURE = {0: "sitting", 1: "standing", 2: "lying", 3: "bending"}
MOVE = {0: "stationary", 1: "walking", 2: "running", 3: "bicycling"}
GESTURE = {0: "nothing", 1: "texting", 2: "smoking", 3: "phoning", 9: "others"}
EVENT = {0: "nothing", 1: "littering", 2: "falling"}


def read_labels(path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines()[1:]:
        p = line.split()
        if len(p) >= 11:
            try:
                rows.append([int(float(x)) for x in p[:11]])
            except ValueError:
                continue
    return rows


def video_for(txt):
    for ext in (".mkv", ".avi"):
        for cand in (txt.with_suffix(ext), *txt.parent.glob(txt.stem + "*" + ext)):
            if cand.exists():
                return cand
    return None


def runs(frames, fps, gap_s=1.0):
    """Continuous runs of frame indices (gaps up to gap_s merged) -> list of durations in seconds."""
    out, start, prev = [], None, None
    for f in sorted(frames):
        if start is None:
            start = prev = f
        elif f - prev > gap_s * fps:
            out.append((prev - start + 1) / fps)
            start = f
        prev = f
    if start is not None:
        out.append((prev - start + 1) / fps)
    return out


def main():
    videos = []
    totals = defaultdict(lambda: {"episodes": 0, "seconds": 0.0, "videos": 0})
    for txt in sorted(ICVL.glob("*/*.txt")):
        vid = video_for(txt)
        if vid is None:
            continue
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows = read_labels(txt)
        per = defaultdict(lambda: defaultdict(list))  # person -> event -> frames
        for r in rows:
            f, pid = r[0], r[6]
            post, move, gest, ev = POSTURE.get(r[7]), MOVE.get(r[8]), GESTURE.get(r[9]), EVENT.get(r[10])
            for name, on in (("smoking", gest == "smoking"), ("littering", ev == "littering"), ("lying", post == "lying"),
                             ("falling", ev == "falling"), ("stationary", move == "stationary" and post in ("standing", "sitting"))):
                if on:
                    per[pid][name].append(f)
        info = {"video": str(vid.relative_to(ICVL)), "camera": vid.parent.name, "fps": round(fps, 2), "frames": n,
                "seconds": round(n / fps, 1), "resolution": f"{w}x{h}", "people": len(per) or len({r[6] for r in rows})}
        seen = set()
        for pid, evs in per.items():
            for name, frames in evs.items():
                durs = runs(frames, fps)
                if name == "stationary":
                    for thr in (20, 60):
                        k = f"stationary {thr}s+"
                        hits = [d for d in durs if d >= thr]
                        totals[k]["episodes"] += len(hits)
                        totals[k]["seconds"] += sum(hits)
                        if hits:
                            seen.add(k)
                        info[k] = info.get(k, 0) + len(hits)
                    continue
                totals[name]["episodes"] += len(durs)
                totals[name]["seconds"] += sum(durs)
                seen.add(name)
                info[name] = info.get(name, 0) + len(durs)
        for k in seen:
            totals[k]["videos"] += 1
        videos.append(info)
    cams = defaultdict(lambda: {"videos": 0, "minutes": 0.0, "resolution": set()})
    for v in videos:
        c = cams[v["camera"]]
        c["videos"] += 1
        c["minutes"] += v["seconds"] / 60
        c["resolution"].add(v["resolution"])
    out = {"videos": len(videos), "hours": round(sum(v["seconds"] for v in videos) / 3600, 2),
           "events": {k: {**v, "seconds": round(v["seconds"])} for k, v in totals.items()},
           "cameras": {k: {**v, "minutes": round(v["minutes"], 1), "resolution": sorted(v["resolution"])}
                       for k, v in sorted(cams.items())},
           "per_video": videos}
    (Path(__file__).parent / "inventory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
