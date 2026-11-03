"""Unseen-camera datasets for Stage 1 (never used for tuning anything).

MEVA (KF1 school/bus cameras, 1080p, 30 fps): Kitware activity annotations in data/meva_annotations
           (<clip>.activities.yml / geom.yml / types.yml, from gitlab.kitware.com/meva/meva-data-repo). GT box = the
           person actor's box during the activity (like littering, where the GT box is the actor).
CAVIAR (INRIA lobby, 384x288, 25 fps): XML ground truth; GT box = the left object ("leaving object" role) while it
           sits there. Walk1-3 have no left object and serve as normal footage from the same camera.
ABODA (11 abandoned-object videos): no annotation files; perception only (labelled separately).
IWDD (MIVIA-IWDD-500 illegal waste dumping, 400 train / 100 public test, mixed cameras 240p-4K): weakly temporal
           labels, one onset second per positive video and no boxes (iwdd_labels); gt stays empty.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml

NB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB))
from stage1_bench import GTSequence  # noqa: E402
from stage1_gate import VideoSequence  # noqa: E402

D = NB.parent / "data"
MEVA_VIDEOS = D / "strategy_sources_20261016" / "meva"
MEVA_ANN = D / "meva_annotations"
CAVIAR = D / "additional" / "caviar"
ABODA = D / "additional" / "aboda"
IWDD = D / "training_sources_20261015" / "mivia_iwdd"
_G0 = re.compile(r"g0'?: '?(\d+) (\d+) (\d+) (\d+)")
_ID1 = re.compile(r"\bid1'?: (\d+)")
_TS0 = re.compile(r"\bts0'?: (\d+)")  # two formats exist (quoted python-style and plain YAML), keys in any order


def meva_activities(stem: str) -> list[dict]:
    """[{type, start, end, persons: [id], objects: [id]}] (frames at 30 fps)."""
    types = {d["types"]["id1"]: next(iter(d["types"]["cset3"]))
             for d in yaml.safe_load((MEVA_ANN / f"{stem}.types.yml").read_text()) if "types" in d}
    out = []
    for d in yaml.safe_load((MEVA_ANN / f"{stem}.activities.yml").read_text()):
        if "act" not in d:
            continue
        a = d["act"]
        s, e = a["timespan"][0]["tsr0"]
        ids = [x["id1"] for x in a["actors"]]
        out.append({"type": next(iter(a["act2"])), "start": s, "end": e,
                    "persons": [i for i in ids if types.get(i) == "person"],
                    "objects": [i for i in ids if types.get(i) != "person"]})
    return out


def meva_boxes(stem: str, ids: set[int]) -> dict[int, dict[int, list[float]]]:
    """track id -> frame -> xyxy, only for the given ids (geom files can be ~50 MB, so parse with a regex)."""
    out = defaultdict(dict)
    with open(MEVA_ANN / f"{stem}.geom.yml") as fh:
        for line in fh:
            i = _ID1.search(line)
            if i and int(i.group(1)) in ids:
                x1, y1, x2, y2 = map(float, _G0.search(line).groups())
                out[int(i.group(1))][int(_TS0.search(line).group(1))] = [x1, y1, x2, y2]
    return out


def meva_sequences(stride: int = 3, activities=("person_puts_down_object",)) -> list[GTSequence]:
    """One GTSequence per annotated clip; GT = person boxes during the chosen activities. `g.tag` lists them."""
    out = []
    for ann in sorted(MEVA_ANN.glob("*.activities.yml")):
        stem = ann.name.replace(".activities.yml", "")
        video = MEVA_VIDEOS / f"{stem}.avi"
        if not video.exists():
            continue
        acts = [a for a in meva_activities(stem) if a["type"] in activities]
        boxes = meva_boxes(stem, {p for a in acts for p in a["persons"]})
        gt = defaultdict(list)
        for a in acts:
            for p in a["persons"]:
                for f in range(a["start"], a["end"] + 1):
                    if f in boxes[p]:
                        gt[f].append(boxes[p][f])
        out.append(GTSequence("meva", VideoSequence(video, stride=stride, seq_id=f"meva/{stem}"), dict(gt),
                              tag=",".join(sorted({a["type"] for a in acts}))))
    return out


def meva_relabel(results, activities) -> list:
    """Same cached runs, ground truth swapped to other activity types."""
    by = {g.seq.seq_id: g for g in meva_sequences(activities=activities)}
    return [(replace(g, gt=by[g.seq.seq_id].gt, tag=by[g.seq.seq_id].tag), r) for g, r in results]


def caviar_sequences(stride: int = 2) -> list[GTSequence]:
    out = []
    for xml in sorted(CAVIAR.glob("*/*.xml")):
        gt = defaultdict(list)
        for fr in ET.parse(xml).getroot().findall("frame"):
            n = int(fr.get("number"))
            ol = fr.find("objectlist")
            for o in (ol if ol is not None else []):
                h = o.find("hypothesislist/hypothesis")
                if h is not None and (h.findtext("role") or "").strip() == "leaving object":
                    b = o.find("box")
                    xc, yc, w, hh = (float(b.get(k)) for k in ("xc", "yc", "w", "h"))
                    gt[n].append([xc - w / 2, yc - hh / 2, xc + w / 2, yc + hh / 2])
        video = next(xml.parent.glob("*.mpg"))
        out.append(GTSequence("caviar", VideoSequence(video, stride=stride, seq_id=f"caviar/{xml.parent.name}"), dict(gt),
                              tag="normal" if xml.parent.name.startswith("Walk") else "left_object"))
    return out


def aboda_sequences(stride: int = 3) -> list[GTSequence]:
    vids = sorted(ABODA.glob("*.avi"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    return [GTSequence("aboda", VideoSequence(v, stride=stride, seq_id=f"aboda/{v.stem}"), {}) for v in vids]


def iwdd_labels(split: str) -> dict[str, dict]:
    """seq_id -> {positive, onset_s (None for negatives), kind: static|dynamic|normal, night, duration_s}."""
    out = {}
    for f in sorted((IWDD / split / "labels").glob("*.json")):
        d = json.loads(f.read_text())
        m = d["metadata"]
        pos = str(d["label"]).lower() == "true"
        out[f"iwdd/{split}/{f.stem}"] = {"positive": pos, "onset_s": float(d["timestamp"]) if pos else None,
                                         "kind": m["event_type"].split()[0], "night": m.get("time_of_day") == "night",
                                         "duration_s": float(m["duration"])}
    return out


def iwdd_sequences(split: str, target_fps: float = 10.0) -> list[GTSequence]:
    """Each video subsampled to ~target_fps (native rates range 6-60 fps), like the 10 fps littering clips."""
    out = []
    for sid, lab in iwdd_labels(split).items():
        video = IWDD / split / "videos" / f"{sid.rsplit('/', 1)[1]}.mp4"
        native = VideoSequence(video).native_fps
        stride = max(1, round(native / target_fps))
        out.append(GTSequence("iwdd", VideoSequence(video, stride=stride, seq_id=sid), {}, tag=lab["kind"]))
    return out


if __name__ == "__main__":
    for g in meva_sequences():
        print(g.seq.seq_id, len(g.gt), g.tag, round(g.seq.duration_s))
    for g in caviar_sequences():
        print(g.seq.seq_id, len(g.gt), g.tag, g.seq.n_frames, g.seq.native_fps)
    print(len(aboda_sequences()), "aboda videos")
