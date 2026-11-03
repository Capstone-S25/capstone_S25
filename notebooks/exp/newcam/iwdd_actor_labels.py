"""Who dumps, in IWDD train videos: raw track ids read off actor_sheets/sheet_NN.jpg (iwdd_actor_sheets.py).

Reviewed by eye on 2026-11-03 (Claude, for Ian). Per video: the track id(s) of the dumper in [onset - 1 s, onset + 5 s].
EX = ambiguous (crowd, two people handling the same item, or a "person" box that is the rubbish itself);
NT = dumper never tracked; "only" = every tracked id near the onset is the dumper (box label hidden on the sheet).
Ids outside [onset - 1, onset + 5] are dropped with a warning. Test videos were not reviewed.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_actor_labels.py   -> exp/newcam/iwdd_actors_train.json
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX, NT, ONLY = "exclude", [], "only"

LABELS = {
    "vid0001": [8, 12], "vid0003": [36], "vid0008": [15, 24], "vid0010": [1], "vid0014": [4],
    "vid0016": [11], "vid0017": [3, 21], "vid0019": NT, "vid0022": [3], "vid0025": ONLY,
    "vid0026": [1], "vid0032": EX, "vid0033": NT, "vid0034": [1], "vid0039": [13],
    "vid0045": [2], "vid0047": ONLY, "vid0054": [4], "vid0056": NT, "vid0058": [11],
    "vid0061": [1], "vid0066": [1], "vid0070": [1, 7], "vid0071": [1], "vid0076": EX,
    "vid0078": [3], "vid0079": [7], "vid0082": [4], "vid0087": [3], "vid0091": [1],
    "vid0092": [11], "vid0093": EX, "vid0096": [1, 9, 13], "vid0097": EX, "vid0103": [1],
    "vid0109": EX, "vid0110": EX, "vid0111": [1], "vid0125": [3], "vid0126": [46],
    "vid0128": ONLY, "vid0129": [12, 2], "vid0132": [14], "vid0133": NT, "vid0136": [16],
    "vid0139": EX, "vid0144": [1], "vid0146": [36], "vid0147": [7], "vid0148": NT,
    "vid0150": [3, 6], "vid0151": EX, "vid0152": [9, 23], "vid0156": [4, 5], "vid0157": [1, 13],
    "vid0161": EX, "vid0164": NT, "vid0165": ONLY, "vid0170": EX, "vid0175": [12],
    "vid0177": ONLY, "vid0179": [1, 64], "vid0181": ONLY, "vid0187": [1], "vid0195": [19],
    "vid0196": NT, "vid0198": NT, "vid0201": [1], "vid0207": [4, 15], "vid0220": [5],
    "vid0233": EX, "vid0234": EX, "vid0238": EX, "vid0240": [9, 8], "vid0241": [12, 16],
    "vid0242": [2, 55], "vid0243": NT, "vid0246": [1], "vid0248": [1], "vid0250": [14],
    "vid0259": EX, "vid0271": [4, 24, 25], "vid0272": [1], "vid0273": [12, 38], "vid0274": NT,
    "vid0276": [10], "vid0280": NT, "vid0288": EX, "vid0290": [12], "vid0292": [1],
    "vid0299": NT, "vid0302": [12], "vid0306": [1], "vid0310": EX, "vid0311": [1],
    "vid0312": EX, "vid0313": [19, 24], "vid0314": [16, 21], "vid0316": [1, 5], "vid0317": [2],
    "vid0318": EX, "vid0319": [2], "vid0322": NT, "vid0323": NT, "vid0326": [1, 28],
    "vid0330": [20], "vid0334": [6, 9], "vid0336": EX, "vid0340": [5], "vid0344": [8, 11, 15],
    "vid0347": EX, "vid0349": [6, 8], "vid0351": [1], "vid0354": NT, "vid0355": EX,
    "vid0356": NT, "vid0358": [4, 12, 14], "vid0360": [17, 28], "vid0361": [1, 5], "vid0362": EX,
    "vid0363": EX, "vid0365": [3], "vid0366": [10], "vid0368": [1], "vid0369": EX,
    "vid0372": [9, 13], "vid0373": [2, 23, 33, 34, 41], "vid0374": [1], "vid0375": NT, "vid0376": NT,
    "vid0378": [1], "vid0381": [16], "vid0383": [1], "vid0386": [4], "vid0396": EX,
    "vid0397": EX, "vid0410": [1], "vid0418": [2, 24, 35], "vid0421": [31], "vid0429": NT,
    "vid0432": [43], "vid0438": EX, "vid0442": [18], "vid0443": [2], "vid0444": [1],
    "vid0447": [9], "vid0452": EX, "vid0465": [8], "vid0473": [3], "vid0475": [5, 6],
    "vid0477": [6], "vid0480": ONLY, "vid0483": EX, "vid0484": [3], "vid0492": [3],
    "vid0493": [2], "vid0494": NT,
}


def main():
    meta = json.loads((HERE / "actor_sheets" / "meta.json").read_text())
    assert set(meta) == set(LABELS), (set(meta) ^ set(LABELS))
    actors, unused = {}, {}
    for v, ids in LABELS.items():
        near = set(meta[v]["ids_near_onset"])
        if ids == ONLY:
            ids = sorted(near) if len(near) <= 2 else EX
        if ids == EX:
            unused[v] = "ambiguous"
            continue
        if not ids:
            unused[v] = "dumper not tracked"
            continue
        if set(ids) - near:
            print(f"warning {v}: ids {sorted(set(ids) - near)} not tracked in [onset-1, onset+5], dropped")
        ids = sorted(set(ids) & near)
        if ids:
            actors[v] = ids
        else:
            unused[v] = "dumper not tracked"
    (HERE / "iwdd_actors_train.json").write_text(json.dumps({"actors": actors, "unused": unused}, indent=1))
    print(len(actors), "videos with the dumper labelled;", dict(Counter(unused.values())))


if __name__ == "__main__":
    main()
