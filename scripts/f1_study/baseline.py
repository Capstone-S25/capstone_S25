"""Score the October deployed routes with the shared F1 harness."""

from collections import defaultdict

from common import OUT, ROOT, TASKS, evaluate, load_rows, read_jsonl, save_json, table

STUDY = ROOT / "results/strategy_study_20261016"


def deployed(model):
    rows = {r["id"]: r for r in load_rows()}
    folder = STUDY / model / "transfer_evaluation"
    import json

    policies = json.loads((folder / "frozen_policies.json").read_text())
    thresholds = {t: policies[t]["selected"]["thresholds"]["dev_max_f1"] for t in TASKS}
    chosen, scores = [], []
    seen = defaultdict(int)
    for name in ["historical", "hmdb", "meva", "photo", "rough"]:
        for p in read_jsonl(folder / f"{name}_predictions.jsonl"):
            if p.get("family") != "selected":
                continue
            seen[name] += 1
            chosen.append({**rows[p["id"]], "split": "test"})
            scores.append(p["score"])
    return evaluate(chosen, scores, thresholds)


if __name__ == "__main__":
    results = {m: deployed(m) for m in ["mobileclip2-s2", "siglip2-b16-224"]}
    save_json(OUT / "baseline.json", results)
    print(table(results))
