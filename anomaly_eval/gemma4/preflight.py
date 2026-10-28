"""Check every declared visual policy on development inputs before a model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_eval.datasets.transfer import atomic_json
from anomaly_eval.gemma4.protocol import DEFAULT_OUTPUT, POLICIES, TASKS, read_rows


def diagnose_low_mass(backend, row, policy):
    captured = {}

    def capture(_module, _inputs, output):
        values, indices = output.logits[0, -1].float().log_softmax(-1).topk(10)
        captured["top_first_answer_tokens"] = [
            {"token": backend.processor.tokenizer.decode([index]), "logprob": value}
            for index, value in zip(indices.cpu().tolist(), values.cpu().tolist())
        ]

    hook = backend.model.register_forward_hook(capture)
    try:
        backend.score([row], policy)
    finally:
        hook.remove()
    return {"id": row["id"], "task": row["task"], **captured}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    directory = args.output / args.model
    spec = json.loads((directory / "checkpoint.json").read_text())
    from anomaly_eval.gemma4.backends import TransformersBackend
    from anomaly_eval.gemma4.llama_backend import ManagedLlamaBackend

    backend = (
        ManagedLlamaBackend(spec["path"], args.model, directory)
        if spec["backend"] == "llama.cpp"
        else TransformersBackend(spec["path"])
    )
    dev = [
        r
        for r in read_rows(args.output / "samples.jsonl")
        if r["cohort"] == "main" and r["split"] == "dev"
    ]
    selected = [
        max(
            (r for r in dev if r["task"] == task),
            key=lambda r: (len(r["views"]), r["id"]),
        )
        for task in TASKS
    ]
    result = {
        "model": args.model,
        "uses_test": False,
        "ids": [r["id"] for r in selected],
        "policies": {},
    }
    try:
        for policy in POLICIES:
            verification = backend.verify_batch(selected, policy)
            scores = backend.score(selected, policy)
            result["policies"][policy.name] = {
                "verification": verification,
                "scores": scores,
                "low_answer_mass_diagnostics": [
                    diagnose_low_mass(backend, row, policy)
                    for row, score in zip(selected, scores)
                    if score.get("answer_token_mass") is not None
                    and score["answer_token_mass"] < 0.5
                ],
            }
            print(args.model, policy.name, "passed", flush=True)
            atomic_json(directory / "preflight.json", result)
        result.update(status="passed", runtime=backend.metadata)
        atomic_json(directory / "preflight.json", result)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
