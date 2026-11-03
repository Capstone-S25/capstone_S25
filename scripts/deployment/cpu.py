"""Build, verify and run the portable MobileCLIP2-S2 CPU deployment bundle.

Build and verify use only the Python standard library. Run from any directory;
model paths are relative to this checkout/bundle, never to the original machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MODEL = "mobileclip2-s2"
REPOSITORY = "timm/MobileCLIP2-S2-OpenCLIP"
REVISION = "ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6"
CACHE_MODEL = "models--timm--MobileCLIP2-S2-OpenCLIP"
SNAPSHOT = Path("data/models/huggingface/hub") / CACHE_MODEL / "snapshots" / REVISION
BASE_FILES = (
    "open_clip_config.json",
    "open_clip_model.safetensors",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
ARTIFACTS = (
    "data/models/stage1/yolo26s.pt",
    "data/models/stage1/yolo26s-pose.pt",
    "data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    "results/clip_finetuning_20261015/provenance.json",
    f"results/clip_finetuning_20261015/seed_42/{MODEL}/finetuned_best.pt",
    f"results/strategy_study_20261016/{MODEL}/selected_heads.pt",
    f"results/strategy_study_20261016/{MODEL}/transfer_evaluation/frozen_policies.json",
    f"results/strategy_study_20261016/{MODEL}/transfer_evaluation/state.json",
)
SOURCE_FILES = {
    "notebooks/stage1_gate.py",
    "notebooks/stage1_pose_action.py",
    "notebooks/weights/pose_action_mlp.npz",
    "anomaly_eval/config/anomalies.json",
    "data-pipeline/main.py",
    "data-pipeline/stage1_offline.py",
    "data-pipeline/alembic.ini",
    "data-pipeline/.env.example",
    "data-pipeline/pyproject.toml",
    "data-pipeline/uv.lock",
    "requirements/cpu-deployment.txt",
    "scripts/deployment/cpu.py",
    "docs/cpu_deployment.md",
    "docs/cpu_model_sources.md",
    "docs/licenses/mobileclip2.txt",
    "docs/licenses/ultralytics.txt",
    "docs/licenses/torchvision.txt",
    "docs/stage1_reasoning.md",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, name):
    """Reject manifest traversal and symlinks; bundles contain regular files."""
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe bundle path: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes root: {name}")
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"Bundle path contains a symlink: {name}")
    return path


def runtime_source(name):
    return (
        name in SOURCE_FILES
        or name.startswith(("anomaly_eval/", "scripts/strategy/"))
        and name.endswith(".py")
        or name.startswith("data-pipeline/data_pipeline/")
        and name.endswith(".py")
        or name.startswith("data-pipeline/migrations/")
    )


def source_files(root):
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    selected = sorted(n for n in names if runtime_source(n))
    missing = SOURCE_FILES.difference(selected)
    if missing:
        raise ValueError(f"Commit the required source files first: {sorted(missing)}")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *selected], cwd=root, check=True
    )
    return selected


def copy_file(source, destination):
    if not source.is_file():
        raise FileNotFoundError(f"Required deployment file missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve HF cache links into real files. Never package tokens or other caches.
    shutil.copyfile(source, destination, follow_symlinks=True)


def verify(root=ROOT):
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["base_model"]["revision"] != REVISION:
        raise ValueError("Unexpected model or base revision")
    seen = set()
    for item in manifest["files"]:
        name = item["path"]
        if name in seen:
            raise ValueError(f"Duplicate manifest path: {name}")
        seen.add(name)
        path = contained(root, name)
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Missing or changed bundle file: {name}")
    required = set(ARTIFACTS) | SOURCE_FILES | {str(SNAPSHOT / n) for n in BASE_FILES}
    if not required.issubset(seen):
        raise ValueError(f"Incomplete manifest: {sorted(required - seen)}")
    provenance = json.loads((root / ARTIFACTS[3]).read_text())
    if provenance["definitions_sha256"] != sha256(
        root / "anomaly_eval/config/anomalies.json"
    ):
        raise ValueError("Anomaly definitions no longer match the training provenance")
    state = (
        root / f"results/strategy_study_20261016/{MODEL}/transfer_evaluation/state.json"
    )
    if json.loads(state.read_text())["status"] != "complete":
        raise ValueError("Selected strategy is incomplete")
    return manifest


def build(artifact_root, hf_cache, output, source_root=ROOT):
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise FileExistsError(f"Choose a new output name: {output}")
    names = source_files(source_root)
    source_snapshot = hf_cache / CACHE_MODEL / "snapshots" / REVISION
    inputs = [(source_root / n, n) for n in names]
    inputs += [(artifact_root / n, n) for n in ARTIFACTS]
    inputs += [(source_snapshot / n, str(SNAPSHOT / n)) for n in BASE_FILES]
    missing = [str(p) for p, _ in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = sum(p.stat().st_size for p, _ in inputs)
    if shutil.disk_usage(output.parent).free < 2 * payload_bytes + 100 * 1024**2:
        raise OSError("Insufficient free space for staging and the transfer archive")
    partial = output.with_name(output.name + ".partial")
    try:
        with tempfile.TemporaryDirectory(
            prefix="cpu-bundle-", dir=output.parent
        ) as tmp:
            bundle = Path(tmp) / "capstone-mobileclip2-s2-cpu"
            for source, name in inputs:
                copy_file(source, bundle / name)
            copy_file(bundle / "docs/cpu_deployment.md", bundle / "README.md")
            names = sorted(
                p.relative_to(bundle).as_posix()
                for p in bundle.rglob("*")
                if p.is_file()
            )
            manifest = {
                "schema_version": 1,
                "created_utc": datetime.now(UTC).isoformat(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=source_root
                )
                .decode()
                .strip(),
                "model": MODEL,
                "base_model": {"repository": REPOSITORY, "revision": REVISION},
                "profile": "CPU float32, YOLO26s/pose + ByteTrack + trained MobileCLIP selected strategies; no Gemma",
                "limitations": [
                    "Two-second alert delay is not achieved by this implementation.",
                    "Classifier-only output does not dispatch verified alerts.",
                ],
                "files": [
                    {
                        "path": n,
                        "bytes": (bundle / n).stat().st_size,
                        "sha256": sha256(bundle / n),
                    }
                    for n in names
                ],
            }
            (bundle / "bundle_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            verify(bundle)
            sums = [(item["sha256"], item["path"]) for item in manifest["files"]]
            sums.append(
                (sha256(bundle / "bundle_manifest.json"), "bundle_manifest.json")
            )
            (bundle / "SHA256SUMS").write_text(
                "".join(f"{digest}  {name}\n" for digest, name in sums)
            )
            with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
                archive.add(bundle, arcname=bundle.name)
        partial.replace(output)
        digest = sha256(output)
        output.with_name(output.name + ".sha256").write_text(
            f"{digest}  {output.name}\n"
        )
        return {
            "archive": str(output),
            "bytes": output.stat().st_size,
            "sha256": digest,
            "source_commit": manifest["source_commit"],
            "files": len(names),
        }
    finally:
        partial.unlink(missing_ok=True)


def environment(root=ROOT):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_CACHE": str(root / "data/models/huggingface/hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "YOLO_CONFIG_DIR": str(root / ".local/yolo"),
        "YOLO_AUTOINSTALL": "false",
        "PYTHONPATH": str(root),
    }


def command(args, root=ROOT):
    shared = [
        "--vlm",
        "clip",
        "--clip-model",
        MODEL,
        "--clip-python",
        sys.executable,
        "--vlm-device",
        "cpu",
        "--stage1-device",
        "cpu",
        "--target-fps",
        str(args.target_fps),
        "--safety-scan-seconds",
        str(args.safety_scan_seconds),
    ]
    if args.action == "video":
        return [
            sys.executable,
            str(root / "data-pipeline/stage1_offline.py"),
            str(args.video.resolve(strict=True)),
            *shared,
            "--out",
            str(args.out.resolve()),
        ]
    return [
        sys.executable,
        str(root / "data-pipeline/main.py"),
        "--source",
        "rtsp",
        "--rtsp-url",
        args.rtsp_url,
        "--stage1",
        *shared,
        "--segment-seconds",
        str(args.segment_seconds),
        "--candidate-dir",
        str(args.out.resolve()),
        "--recordings-dir",
        str(args.recordings.resolve()),
    ]


def smoke(root=ROOT):
    """Synthetic load/forward check of all five tasks; never an accuracy test."""
    os.environ.update(environment(root))
    sys.path[:0] = [str(root), str(root / "data-pipeline")]
    from uuid import uuid4

    import numpy as np
    import torch
    from data_pipeline.detection.stage1 import Stage1Detector
    from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame
    from PIL import Image

    from anomaly_eval.reasoning.backends import ClipStrategy

    with tempfile.TemporaryDirectory(prefix="cpu-smoke-") as tmp:
        folder = Path(tmp)
        paths = []
        for i in range(8):
            path = folder / f"{i}.png"
            Image.new("RGB", (320, 240), (100 + i, 110, 120)).save(path)
            paths.append(str(path))
        detector = Stage1Detector("smoke", folder, fps=5)
        session = uuid4()
        for i in range(2):
            detector.process(
                DecodedFrame(
                    0,
                    "smoke",
                    session,
                    i,
                    i * 200,
                    np.zeros((240, 320, 3), dtype=np.uint8),
                )
            )
        classifier = ClipStrategy(MODEL, "cpu")
        tasks = [
            "smoking",
            "littering",
            "loitering",
            "rough_sleeping",
            "unattended_items",
        ]
        view = {"paths": paths, "times_s": [20 * i / 7 for i in range(8)]}
        result = classifier.classify(
            {"tasks": tasks, "timestamp_s": 20, "views": {"short": view, "long": view}}
        )
        if set(result["predictions"]) != set(tasks) or any(
            not value.get("input_history_supported")
            or value.get("decision") not in {"normal", "anomaly"}
            or not math.isfinite(value.get("score", float("nan")))
            for value in result["predictions"].values()
        ):
            raise RuntimeError("Smoke test did not score all five tasks")
        return {
            "status": "passed",
            "device": "cpu",
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "synthetic_only": True,
            "result": result,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser(
        "build", help="Package committed runtime source and local weights"
    )
    p.add_argument("--artifact-root", type=Path, default=ROOT)
    p.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(
            os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
        ).expanduser(),
    )
    p.add_argument("--output", type=Path, required=True)
    sub.add_parser("verify", help="Verify every packaged file with no ML dependencies")
    sub.add_parser(
        "smoke", help="Load detectors and run synthetic five-task CPU inference"
    )
    sub.add_parser(
        "setup", help="Create .venv-cpu and install CPU wheels (internet required)"
    )
    for name in ["video", "live"]:
        p = sub.add_parser(name)
        p.add_argument("--target-fps", type=float, default=5)
        p.add_argument("--safety-scan-seconds", type=float, default=10)
        p.add_argument("--out", type=Path, default=ROOT / "results/deployment")
        if name == "video":
            p.add_argument("video", type=Path)
        else:
            p.add_argument("--rtsp-url", required=True)
            p.add_argument("--segment-seconds", type=int, default=2)
            p.add_argument(
                "--recordings", type=Path, default=ROOT / "results/recordings"
            )
    args = parser.parse_args()
    if args.action == "build":
        print(
            json.dumps(
                build(
                    args.artifact_root.resolve(),
                    args.hf_cache.resolve(),
                    args.output.resolve(),
                ),
                indent=2,
            )
        )
        return
    manifest = verify()
    if args.action == "verify":
        print(
            json.dumps(
                {
                    "status": "verified",
                    "files": len(manifest["files"]),
                    "source_commit": manifest["source_commit"],
                },
                indent=2,
            )
        )
    elif args.action == "setup":
        import venv

        target = ROOT / ".venv-cpu"
        if target.exists():
            raise FileExistsError(f"Environment already exists: {target}")
        venv.create(target, with_pip=True)
        python = target / "bin/python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "torch==2.8.0+cpu",
                "torchvision==0.23.0+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements/cpu-deployment.txt"),
            ],
            check=True,
        )
    elif args.action == "smoke":
        print(json.dumps(smoke(), indent=2))
    else:
        if args.action == "live" and not os.environ.get("DATABASE_URL"):
            parser.error(
                "Set DATABASE_URL and run the database migration first; see README.md"
            )
        subprocess.run(command(args), cwd=ROOT, env=environment(), check=True)


if __name__ == "__main__":
    main()
