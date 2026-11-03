"""Score the VideoMAE action cue (tune half only) from the cached logits.

  ../../../.venv/bin/python eval_videomae.py diag     # per-person separability (uses GT for analysis only)
  ../../../.venv/bin/python eval_videomae.py sweep    # add-on / replace-reach_down rows -> cache/results.csv
"""
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import videomae as V  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine, score  # noqa: E402
from random_control import lift  # noqa: E402
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

TUNE = model_splits("s", heldout=False)
BASE = partial(baseline, cfg=GateConfig())
NOREACH = partial(baseline, cfg=GateConfig(rules=tuple(r for r in DEFAULT_RULES if r != "reach_down")))


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def person_scores(ds, group):
    """Per call: (score, is_gt_person, tag). GT is used here only to analyse separability."""
    cache = V.cache_for(ds)
    out = []
    for g, r in TUNE[ds]:
        for rec in cache.get(r.seq_id, []):
            idx = r.observations[rec["obs_i"]].index
            hit = any(box_hit(rec["box"], b) for b in g.gt.get(idx, []))
            out.append((V.group_mass(rec, group), hit, g.tag))
    return out


def diag():
    rows = []
    for ds in ("littering", "smoking", "avenue"):
        for grp in ("smoke", "throw", "kick", "pick", "litter"):
            s = person_scores(ds, grp)
            pos = [x for x, h, _ in s if h]
            neg = [x for x, h, _ in s if not h]
            normal = [V.group_mass(rec, grp) for recs in V.cache_for("normal").values() for rec in recs]
            rows.append({"dataset": ds, "group": grp, "calls": len(s), "gt_calls": len(pos),
                         "AUROC_gt_vs_other": round(auroc(pos, neg), 3),
                         "median_gt": round(float(np.median(pos)), 4) if pos else None,
                         "median_other": round(float(np.median(neg)), 4) if neg else None,
                         "p99_normal": round(float(np.percentile(normal, 99)), 4)})
            if ds == "littering" and grp in ("throw", "litter"):
                for tag in sorted({t for _, _, t in s}):
                    p = [x for x, h, t in s if h and t == tag]
                    n = [x for x, h, t in s if not h and t == tag]
                    rows.append({"dataset": f"littering:{tag}", "group": grp, "calls": len(p) + len(n),
                                 "gt_calls": len(p), "AUROC_gt_vs_other": round(auroc(p, n), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(V.CACHE / "diag.csv", index=False)
    print(df.to_string())
    # what the model actually predicts on the GT person
    for ds in ("littering", "smoking"):
        top = {}
        cache = V.cache_for(ds)
        from transformers import AutoConfig
        lab = AutoConfig.from_pretrained(V.MODEL_ID, local_files_only=True).id2label
        for g, r in TUNE[ds]:
            for rec in cache.get(r.seq_id, []):
                idx = r.observations[rec["obs_i"]].index
                if any(box_hit(rec["box"], b) for b in g.gt.get(idx, [])):
                    k = int(np.argmax(rec["logits"]))
                    top[lab[k]] = top.get(lab[k], 0) + 1
        print(ds, "top-1 on GT person:", sorted(top.items(), key=lambda x: -x[1])[:12])


def row(name, mode, method, extra):
    s = score(method, splits=TUNE, name=name)
    m, c, sd = lift(method, TUNE["littering"], seeds=10)
    out = {"variant": name, "mode": mode, **{k: v for k, v in s.items() if k != "method"},
           "littering_lift": round(m - c, 1), "control_crop_%": round(c, 1), "control_sd": round(sd, 1)}
    if extra is not None:
        n_ev = {ds: sum(len(extra(r, ds)) for _, r in TUNE[ds]) for ds in ("littering", "avenue", "smoking", "normal")}
        out.update({f"cue_events_{k}": v for k, v in n_ev.items()})
    print(out, flush=True)
    return out


def sweep():
    rows = [row("baseline (s)", "baseline", BASE, None), row("baseline - reach_down", "baseline", NOREACH, None)]
    variants = []
    for t in (0.05, 0.1, 0.2, 0.35):
        variants.append((f"litter mass>={t}", dict(groups=("litter",), thr=t)))
    for t in (0.1, 0.2, 0.35):
        variants.append((f"throw mass>={t}", dict(groups=("throw",), thr=t)))
    for t in (0.02, 0.05, 0.1, 0.2):
        variants.append((f"smoke mass>={t}", dict(groups=("smoke",), thr=t)))
    for name, params in variants:
        m = V.make_method(**params)
        rows.append(row(name, "alone", m, m))
        rows.append(row(name, "add-on", combine(BASE, m), m))
        if "smoke" not in params["groups"]:
            rows.append(row(name, "replace reach_down", combine(NOREACH, m), m))
    df = pd.DataFrame(rows)
    df.to_csv(V.CACHE / "results.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())


if __name__ == "__main__":
    {"diag": diag, "sweep": sweep}[sys.argv[1]]()
