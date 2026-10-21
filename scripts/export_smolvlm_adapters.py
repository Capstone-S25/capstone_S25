"""Pin portable PEFT metadata without changing any selected adapter weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_eval.training.clip import save_json
from anomaly_eval.training.smolvlm import ROOT, SPEC, digest, local_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/smolvlm_finetuning_20261019"
    )
    args = parser.parse_args()
    audit_path = args.study / "portable_adapter_export.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for path in sorted(args.study.glob("seed_*/selection.json")):
        selection = json.loads(path.read_text())
        for route, expected in selection["adapters_sha256"].items():
            directory = args.study / route / "adapter"
            assert digest(directory / "adapter_model.safetensors") == expected
            config_path = directory / "adapter_config.json"
            config = json.loads(config_path.read_text())
            previous = config["base_model_name_or_path"]
            if previous not in {local_checkpoint(), SPEC.checkpoint}:
                raise ValueError("Refuse to change metadata for a different base model")
            config.update(
                base_model_name_or_path=SPEC.checkpoint, revision=SPEC.revision
            )
            save_json(config_path, config)
            audit.setdefault(route, {"original_base_reference": previous})
            audit[route].update(
                base_model=SPEC.checkpoint,
                revision=SPEC.revision,
                weights_sha256=expected,
                config_sha256=digest(config_path),
                weights_unchanged=True,
            )
    save_json(audit_path, audit)
    print("Portable adapter metadata exported:", len(audit), "checkpoints")


if __name__ == "__main__":
    main()
