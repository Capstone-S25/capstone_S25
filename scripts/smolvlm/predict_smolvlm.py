"""Run a selected five-target SmolVLM binary detector on real image inputs."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.models import load_image
from anomaly_eval.training.data import TASKS
from anomaly_eval.training.predict import temporal_support
from anomaly_eval.training.smolvlm import (
    ROOT,
    digest,
    messages,
    processor_and_model,
    response_ids,
)


def prepare_views(task, images, timestamps=None):
    """Do not fabricate temporal evidence for a static deployment input."""
    if not images:
        raise ValueError("At least one image is required")
    if timestamps is not None and (
        len(timestamps) != len(images)
        or not np.isfinite(timestamps).all()
        or any(a > b for a, b in pairwise(timestamps))
    ):
        raise ValueError(
            "Timestamps must be finite, chronological and match the images"
        )
    if task in {"smoking", "rough_sleeping"}:
        return images[-1:], None
    if not temporal_support(task, timestamps, len(images)):
        return (
            None,
            "Provide eight ordered clip frames for littering, or four distinct past views over approximately 20 seconds with timestamps for duration/ownership targets",
        )
    if task == "littering":
        selected = np.linspace(0, len(images) - 1, 8).round().astype(int)
    else:
        from anomaly_eval.training.predict import temporal_indices

        selected = temporal_indices(timestamps, len(images))
    return [images[i] for i in selected], None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/smolvlm_finetuning_20261019"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--timestamps", type=float, nargs="+")
    adapter_choice = parser.add_mutually_exclusive_group()
    adapter_choice.add_argument(
        "--joint",
        action="store_true",
        help="Use the selected joint adapter instead of the development-selected target route",
    )
    adapter_choice.add_argument(
        "--original",
        action="store_true",
        help="Use the matched original model and its development thresholds",
    )
    adapter_choice.add_argument(
        "--repeated-gains",
        action="store_true",
        help="Optional retrospective route: require development improvement in both training seeds",
    )
    parser.add_argument(
        "--operating-point",
        choices=["raw", "dev_max_f1", "dev_fpr_budget"],
        default="dev_fpr_budget",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not all(p.is_file() for p in args.images):
        parser.error("Every input image must exist")
    try:
        images, unsupported = prepare_views(args.task, args.images, args.timestamps)
    except ValueError as error:
        parser.error(str(error))
    selection = json.loads(
        (args.study / f"seed_{args.seed}/selection.json").read_text()
    )
    route = (
        "original"
        if args.original
        else (
            selection["joint_adapter"]
            if args.joint
            else selection["task_routes"][args.task]
        )
    )
    if args.repeated_gains:
        if args.seed != 42:
            parser.error("The optional repeated-gain policy uses the primary seed 42")
        policy = json.loads((args.study / "repeated_gain_routes.json").read_text())
        if (
            digest(args.study / "seed_42/selection.json")
            != policy["primary_selection_sha256"]
        ):
            raise ValueError(
                "Repeated-gain policy does not match the frozen primary selection"
            )
        route = policy["task_routes"][args.task]
    result = {
        "task": args.task,
        "route": route,
        "operating_point": args.operating_point,
        "definition_scope": "Visible evidence only, no housing-status inference; temporal labels remain provisional",
        "calibration_scope": "Retrospective source-separated development set; no guaranteed deployment false-alert rate",
    }
    if unsupported:
        result.update(label="unknown", reason=unsupported, score=None)
    else:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        processor, model = processor_and_model(device=args.device)
        if route != "original":
            from peft import PeftModel

            adapter = args.study / route / "adapter"
            if (
                digest(adapter / "adapter_model.safetensors")
                != selection["adapters_sha256"][route]
            ):
                raise ValueError("Selected adapter fingerprint changed")
            model = PeftModel.from_pretrained(model, adapter).float().eval()
        text = processor.apply_chat_template(
            messages(args.task, len(images)), add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[[load_image(p, 512, "letterbox") for p in images]],
            return_tensors="pt",
        ).to(args.device)
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            logits = (
                model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            )
            ids = response_ids(processor)
            score = float(logits[ids[1]] - logits[ids[0]])
            mass = float(logits.softmax(-1)[ids].sum())
        threshold = selection["thresholds"][route][args.task][args.operating_point]
        result.update(
            label=args.task if score >= threshold else "target_absent",
            score=score,
            threshold=threshold,
            yes_no_token_mass=mass,
            used_images=[str(p) for p in images],
            probability_warning="The margin and conditional binary answer are not calibrated probabilities or native free-generation output",
        )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return result


if __name__ == "__main__":
    main()
