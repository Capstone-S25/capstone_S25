"""Rebuild the project's fit / test split from a downloaded copy of capstone_datasets.

Run it on the folder you got from Drive and it writes splits.json next to this script. Anyone who runs
it on the same download gets byte-for-byte the same split: the assignment below is a fixed table, not
a random draw, so there is no seed to agree on.

    python make_splits.py "path/to/capstone_datasets"            -> splits.json + a printed summary
    python make_splits.py "path/to/capstone_datasets" --full-hash  (hash whole files, slow on 26 GB)
    python make_splits.py "path/to/capstone_datasets" --check      (verify an existing splits.json)

Standard library only - no numpy, no OpenCV, nothing to install.

WHAT THE SPLIT HOLDS OUT
Test is disjoint from fit by CAMERA (ICVL, CHAD) or by SCENE (ABODA, D2-CDNet, IITB, Roboflow), never
by video. A model is allowed to have seen a camera or none of it. Three things this buys that a
per-video split does not:

 1. Unseen angles. The test number answers "does this work on a camera we never tuned on", which is
    the only question a deployment cares about.
 2. Cross-anomaly safety. ICVL cameras 02, 03 and 05 each carry more than one anomaly. Splitting each
    anomaly on its own would put camera 02 in littering-fit and loitering-test, so the scene would be
    both trained and tested on. The split is global over cameras, so a camera lands on one side for
    every anomaly at once.
 3. Duplicate safety. 28 ICVL videos are stored twice - byte-identical copies in both the loitering
    and the smoking folder, because those people are breaking both rules at once. A copy cannot
    straddle the split, because both copies carry the same camera.

    The duplication is in the FOOTAGE ONLY. The two label files are different annotations of it: for
    02/AA_008_02_20140802_03 the loitering folder names 5 loiterers and the smoking folder names 3
    smokers, and the rest of the frame is unannotated in both. So this file indexes one record per
    LABEL FILE - 949 of them, 1255 events - and marks the 28 pairs with `same_video_as`, leaving 921
    distinct video files. Deduplicating the events instead would silently delete 42 smoking events
    and 44 loitering ones.

    What that means when you run anything over this: decode each distinct video once and score it
    against the union of its label files. A rule firing on an annotated smoker in a loitering video
    is a correct detection of smoking, not a false alarm - and 53% of smoking events are the same
    person as a labelled loiterer, so scoring the two folders separately would charge you for a lot
    of correct behaviour. Group by `video_key` to get one pass per video.

HOW EACH DATASET IS KEYED
    ICVL        icvl/<camera>        from the label file's own "camera" field (11 cameras)
    CHAD        chad/<camera>        first field of the stem, 1_091_1 -> camera 1
    ABODA       aboda/<stem>         each ABODA video is a different scene
    D2-CDNet    d2cdnet/<stem>       each CDnet sequence is a different scene
    IITB        iitb/corridor        all 8 clips are the same corridor camera
    Roboflow    roboflow/<source>    clips cut from one recording share a key; see ROBOFLOW below

ROBOFLOW
These are web clips, not CCTV, so they are fit-only: a held-out number measured on them would not
say anything about CCTV performance, and ICVL camera 02 already supplies 111 littering test events.
They are still grouped by source recording, because Roboflow stored a recording once per litterer and
rebuild_littering_clips.py merged some of those into one clip ("dropping_v05_2+v06_5" holds frames
from v05 and v06). The groups are built by union-find over the vNN ids in the names, so if you make
your own dev split inside fit, group on `scene` and the two halves of a recording stay together.
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "splits.json"
VID = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
FP_BYTES = 8 << 20          # fingerprint = size + first and last 8 MB, unless --full-hash

# ---------------------------------------------------------------- the split itself
# Every scene key not listed here is fit. Chosen once and then left alone; see the notes per line.
TEST_SCENES = {
    # ICVL. 02 and 10 together hold 111 littering / 41 loitering / 44 smoking events - roughly a
    # quarter to a third of each anomaly, and 02 is the one camera carrying all three.
    "icvl/02", "icvl/10",
    # IITB Corridor is a single indoor camera and nothing was ever fitted on it, so all 8 clips are
    # test. It cannot be split by camera; there is only the one.
    "iitb/corridor",
    # Unattended items is 18 events over 14 scenes, so it is split scene-wise rather than given
    # entirely to one side - Yuchuan needs something to develop prompts against, and prompts
    # developed on a test scene are fitted on it whether or not any gradient was involved.
    "aboda/video2", "aboda/video3", "aboda/video5", "aboda/video7", "aboda/video8", "aboda/video10",
    "d2cdnet/PETS2006", "d2cdnet/abandonedBox", "d2cdnet/tramstop",
    # Rough sleeping is 4 clips on CHAD's 4 cameras: two each side. Treat either side as a demo
    # rather than a measurement, and note CHAD normal footage did inform one gate default
    # (item_rule_skip), so CHAD is not a clean held-out set in the way ICVL 02 / 10 is.
    "chad/2", "chad/4",
}


def scene_of(ds, stem, label):
    """The unit the split holds out. Returns None if the dataset is not one we know."""
    if ds == "ICVL":
        return "icvl/" + str(label.get("camera") or "?")
    if ds == "CHAD":
        return "chad/" + stem.split("_")[0]
    if ds == "ABODA":
        return "aboda/" + stem
    if ds.startswith("D2-CDNet"):
        return "d2cdnet/" + stem
    if ds.startswith("IITB"):
        return "iitb/corridor"
    if "Roboflow" in ds:
        return "roboflow/" + stem          # refined into source groups by roboflow_groups()
    return None


def roboflow_groups(stems):
    """Union-find over the vNN ids in each stem -> {stem: source-recording key}."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    keys_for = {}
    for s in stems:
        m = re.match(r"([a-z-]+)_(.*)", s)
        if not m:
            keys_for[s] = [s]
            continue
        act, rest = m.group(1), m.group(2)
        ks = [f"{act}_v{v}" for v in re.findall(r"v(\d+)", rest)] or [s]
        keys_for[s] = ks
        for k in ks[1:]:
            a, b = find(ks[0]), find(k)
            if a != b:
                par[a] = b
    return {s: "roboflow/" + find(ks[0]) for s, ks in keys_for.items()}


def fingerprint(p, full=False):
    h = hashlib.sha256()
    n = p.stat().st_size
    h.update(str(n).encode())
    with p.open("rb") as f:
        if full:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(FP_BYTES))
            if n > 2 * FP_BYTES:
                f.seek(-FP_BYTES, 2)
                h.update(f.read(FP_BYTES))
    return h.hexdigest()


def video_for(label_path, root):
    """The video a label file describes: same relative place, under videos/ instead of labels/."""
    parts = list(label_path.relative_to(root).parts)
    i = parts.index("labels")
    parts[i] = "videos"
    base = root.joinpath(*parts[:-1])
    for ext in VID:
        c = base / (label_path.stem + ext)
        if c.exists():
            return c
    hits = sorted(c for c in base.glob(label_path.stem + ".*") if c.suffix.lower() in VID)
    return hits[0] if hits else None


def build(root, full=False):
    clips, problems = [], []
    labels = sorted(root.rglob("labels/**/*.json"))
    if not labels:
        sys.exit(f"no label files under {root} - is that the capstone_datasets folder?")
    for lp in labels:
        try:
            lab = json.loads(lp.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError as e:
            problems.append(f"unreadable label {lp.relative_to(root)}: {e}")
            continue
        rel = lp.relative_to(root).parts
        ds = str(lab.get("dataset", ""))
        stem = Path(str(lab.get("video", lp.stem))).stem
        scene = scene_of(ds, stem, lab)
        if scene is None:
            problems.append(f"unknown dataset {ds!r} in {lp.relative_to(root)}")
            continue
        vp = video_for(lp, root)
        if vp is None:
            problems.append(f"no video for {lp.relative_to(root)}")
            continue
        clips.append({"anomaly": rel[0], "subset": rel[1], "dataset": ds, "scene": scene,
                      "stem": stem, "video": str(vp.relative_to(root)).replace("\\", "/"),
                      "label": str(lp.relative_to(root)).replace("\\", "/"),
                      "events": len(lab.get("events", [])),
                      "duration_s": round(float(lab.get("duration_s") or 0), 2),
                      "_path": vp})

    # Roboflow clips get their source-recording key, replacing the per-stem placeholder.
    rg = roboflow_groups([c["stem"] for c in clips if "Roboflow" in c["dataset"]])
    for c in clips:
        if "Roboflow" in c["dataset"]:
            c["scene"] = rg[c["stem"]]

    # Byte-identical footage, kept as two records because the two label files annotate different
    # people. video_key is the thing to group on for one decode pass per video.
    for c in clips:
        c["video_key"] = fingerprint(c.pop("_path"), full)
    first = {}
    for c in sorted(clips, key=lambda c: c["video"]):
        f = c["video_key"]
        if f in first:
            c["same_video_as"] = first[f]["video"]
            if c["scene"] != first[f]["scene"]:
                problems.append(f"same footage in two scenes: {c['video']} vs {first[f]['video']}")
        else:
            first[f] = c
            c["same_video_as"] = None

    for c in clips:
        c["split"] = "test" if c["scene"] in TEST_SCENES else "fit"
    clips.sort(key=lambda c: (c["anomaly"], c["subset"], c["video"]))
    return clips, problems


def digest(clips):
    """One short token over (video -> side). Two people compare this instead of diffing a file."""
    body = "\n".join(sorted(f'{c["label"]}\t{c["split"]}' for c in clips))
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def check(clips):
    """The three ways a split like this leaks. Any of them is a bug, not a warning."""
    bad = []
    for field in ("scene", "video_key", "stem"):
        sides = defaultdict(set)
        for c in clips:
            sides[c[field]].add(c["split"])
        for k, v in sorted(sides.items()):
            if len(v) > 1:
                bad.append(f"{field} {k!r} is in both fit and test")
    missing = sorted(s for s in TEST_SCENES if not any(c["scene"] == s for c in clips))
    if missing:
        bad.append("test scenes named but not found in the download: " + ", ".join(missing))
    return bad


def summarise(clips):
    rows = defaultdict(Counter)
    for c in clips:
        s = c["split"]
        rows[c["anomaly"]][s + "_events"] += c["events"]
        rows[c["anomaly"]][s + "_labelled"] += 1
    print("\n%-18s %10s %9s %10s %9s" % ("anomaly", "fit videos", "fit evts", "test videos", "test evts"))
    tot = Counter()
    for a in sorted(rows):
        r = rows[a]
        tot.update(r)
        print("%-18s %10d %9d %10d %9d" % (a, r["fit_labelled"], r["fit_events"],
                                           r["test_labelled"], r["test_events"]))
    print("%-18s %10d %9d %10d %9d" % ("TOTAL", tot["fit_labelled"], tot["fit_events"],
                                       tot["test_labelled"], tot["test_events"]))
    dups = [c for c in clips if c["same_video_as"]]
    print("\n%d label files over %d distinct videos; %d videos annotated twice "
          "(same footage, different anomaly)" % (len(clips), len({c["video_key"] for c in clips}), len(dups)))
    for c in sorted(dups, key=lambda c: c["video"])[:3]:
        print("   e.g. %s  ==  %s" % (c["video"], c["same_video_as"]))
    if len(dups) > 3:
        print("   ... and %d more, all loitering <-> smoking, all on cameras that stay on one side" % (len(dups) - 3))
    scenes = defaultdict(set)
    for c in clips:
        scenes[c["split"]].add(c["scene"])
    print("\nscenes: %d fit, %d test (disjoint)" % (len(scenes["fit"]), len(scenes["test"])))
    print("test scenes:", ", ".join(sorted(scenes["test"])))
    mins = defaultdict(float)
    seen = set()
    for c in clips:
        if c["video_key"] not in seen:
            seen.add(c["video_key"])
            mins[c["split"]] += c["duration_s"] / 60
    print("footage: %.0f min fit, %.0f min test" % (mins["fit"], mins["test"]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        sys.exit(__doc__.split("\n\n")[1].strip())
    root = Path(args[0]).resolve()
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")

    if "--check" in flags:
        if not OUT.exists():
            sys.exit(f"no {OUT.name} to check - run without --check first")
        clips = json.loads(OUT.read_text())["clips"]
    else:
        clips, problems = build(root, full="--full-hash" in flags)
        for p in problems:
            print("WARNING:", p)
        OUT.write_text(json.dumps({
            "_how": "fit / test split of capstone_datasets, held out by camera or scene. Rebuild with "
                    "make_splits.py <dataset root>; the assignment is a fixed table, so any copy of the "
                    "download yields this same file. One record per LABEL FILE; 'scene' is the unit "
                    "held out, so group on it for any further split. Records with 'same_video_as' set "
                    "share byte-identical footage with that video in another anomaly folder and carry "
                    "a different annotation of it - group on 'video_key' to decode each video once and "
                    "score it against the union of its label files.",
            "root_as_built": str(root), "test_scenes": sorted(TEST_SCENES),
            "clips": clips}, indent=1))
        print("wrote", OUT)

    bad = check(clips)
    summarise(clips)
    print("\nsplit digest:", digest(clips), "- same download, same table, same digest.")
    if bad:
        print("\nLEAK CHECK FAILED:")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("\nleak check passed: no scene, no video and no byte-identical copy appears on both sides.")


if __name__ == "__main__":
    main()
