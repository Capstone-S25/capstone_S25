"""Round-4 rows: the Drive's new labelled videos as study rows.

Rebuilds the windows that round4_drive.py scored (same frames, same labels)
and writes results/f1_study_20261111/round4/rows.jsonl. `load_rows()` adds them
when F1_ROUND4=1, with split "site_fit" or "site_shift" like round 3.

ICVL cameras 02, 05, 08 and 11 are the shift side (scored, never fitted) for
every ICVL task, so no camera is fitted for one behaviour and tested for
another. They were picked to balance the video counts per task, before any
per-camera score was looked at. IITB Corridor and CHAD are test-only by their
READMEs; ABODA and D2-CDNet are too small to split and stay on the shift side.
"""

from __future__ import annotations

import collections
import hashlib
import json

from round4_drive import FRAMES, OUT, ROOT, SETS, windows

SHIFT_CAMERAS = {"02", "05", "08", "11"}


def key(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:24]


def main():
    scored = [json.loads(line) for line in (OUT / "windows.jsonl").open()]
    by_video = collections.defaultdict(dict)
    for r in scored:
        by_video[r["set"], r["video"]][round(r["t"], 3)] = r
    rows = []
    for (name, video), wanted in sorted(by_video.items()):
        task = SETS[name][0]
        frames = json.loads((FRAMES / name / video / "frames.json").read_text())
        first = next(iter(wanted.values()))
        icvl = name.startswith("icvl_")
        camera = first["camera"]
        for t, t0, views, reference in windows(task, frames, 2.0):
            r = wanted.get(round(float(t), 3))
            if r is None:
                continue
            rel = [str((ROOT / v).resolve().relative_to(ROOT)) for v in views]
            rows.append(
                {
                    "id": key("round4", name, video, f"{t:.3f}"),
                    "task": task,
                    "target": r["target"],
                    "source": name,
                    "group_id": f"icvl:{camera}" if icvl else f"{name}:{video}",
                    "site": "shift" if (not icvl or camera in SHIFT_CAMERAS) else "fit",
                    "views": rel,
                    "window_end_s": float(t),
                    "video": video,
                    "camera": camera,
                }
            )
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids")
    with open(OUT / "rows.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    count = collections.Counter((r["source"], r["site"], r["target"]) for r in rows)
    for k in sorted(count):
        print(k, count[k])


if __name__ == "__main__":
    main()
