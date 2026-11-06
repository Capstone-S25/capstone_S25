"""Import only verified inference weights from an existing transfer archive.

Run with the standard library: python tools/import_models.py BUNDLE.tar.gz.
No archive paths are extracted; the explicit model-file mapping is streamed.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVISION = "ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6"
MODEL = "mobileclip2-s2"


def import_models(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Model directory exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        manifests = [
            m
            for m in bundle.getmembers()
            if m.name.endswith("/bundle_manifest.json") and m.isfile()
        ]
        if len(manifests) != 1:
            raise ValueError("Expected one transfer manifest")
        prefix = manifests[0].name.rsplit("/", 1)[0] + "/"
        with bundle.extractfile(manifests[0]) as stream:
            manifest = json.load(stream)
        if manifest["model"] != MODEL or manifest["base_model"]["revision"] != REVISION:
            raise ValueError("Unexpected model or revision")
        index = {v["path"]: v for v in manifest["files"]}
        mapping = {
            f"results/strategy_study_20261016/{MODEL}/selected_heads.pt": f"{MODEL}/selected_heads.pt",
            f"results/strategy_study_20261016/{MODEL}/transfer_evaluation/frozen_policies.json": f"{MODEL}/frozen_policies.json",
            f"results/strategy_study_20261016/{MODEL}/transfer_evaluation/state.json": f"{MODEL}/state.json",
            f"results/clip_finetuning_20261015/seed_42/{MODEL}/finetuned_best.pt": f"{MODEL}/finetuned_best.pt",
            "results/clip_finetuning_20261015/provenance.json": f"{MODEL}/provenance.json",
        }
        for name in ["yolo26s.pt", "yolo26s-pose.pt"]:
            mapping["data/models/stage1/" + name] = "detectors/" + name
        detector = "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        mapping["data/models/detection/" + detector] = "detectors/" + detector
        for name in [
            "open_clip_config.json",
            "open_clip_model.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            mapping[
                f"data/models/huggingface/hub/models--timm--MobileCLIP2-S2-OpenCLIP/snapshots/{REVISION}/{name}"
            ] = f"{MODEL}/base/{name}"
        needed = sum(index[name]["bytes"] for name in mapping)
        if shutil.disk_usage(destination.parent).free < needed + 50 * 1024**2:
            raise OSError("Insufficient model storage")
        with tempfile.TemporaryDirectory(
            prefix="import-models-", dir=destination.parent
        ) as temp:
            stage = Path(temp) / "models"
            entries = []
            for source, target in sorted(
                mapping.items(),
                key=lambda item: bundle.getmember(prefix + item[0]).offset_data,
            ):
                member = bundle.getmember(prefix + source)
                info = index[source]
                if not member.isfile() or member.size != info["bytes"]:
                    raise ValueError("Invalid model member: " + source)
                output = stage / target
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with (
                    bundle.extractfile(member) as incoming,
                    output.open("wb") as stream,
                ):
                    for block in iter(lambda: incoming.read(1024**2), b""):
                        digest.update(block)
                        stream.write(block)
                if digest.hexdigest() != info["sha256"]:
                    raise ValueError("Model checksum mismatch: " + source)
                entries.append(
                    {
                        "path": target,
                        "bytes": output.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            provenance = json.loads((stage / MODEL / "provenance.json").read_text())
            definitions = ROOT / "vision_runtime/definitions.json"
            if (
                hashlib.sha256(definitions.read_bytes()).hexdigest()
                != provenance["definitions_sha256"]
            ):
                raise ValueError("Training definition mismatch")
            (stage / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_commit": manifest["source_commit"],
                        "base_revision": REVISION,
                        "files": entries,
                    },
                    indent=2,
                )
                + "\n"
            )
            stage.rename(destination)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    entries = import_models(args.archive.resolve(), args.out.resolve())
    print(f"Imported {len(entries)} verified model files to {args.out}")


if __name__ == "__main__":
    main()
