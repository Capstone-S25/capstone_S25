"""Assemble one self-contained transfer folder (and optionally a .tar of it).

    python tools/build_deployment.py dist/capstone-cpu-deployment-YYYYMMDD \
        --ensemble-export results/.../bundle_v10 \
        --hf-cache ~/.cache/huggingface/hub \
        --reviewer qwen35-4b --gemma-weights <folder with the reviewer's two .gguf files> \
        --llama data/tools/llama_gemma4/b11925/llama-b11925 \
        --wheels --archive

The folder holds the application at the current commit, every model file, the
built dashboard, and optionally Python wheels (offline install) and a CPU
llama.cpp server with the review model's weights (Qwen 3.5-4B by default, or
Gemma 4 E4B). The GEMMA_* settings and the gemma/ folder keep their names for
either model. Run it from a clean checkout of the branch,
with `npm install` already done in frontend/ and the models imported.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from re import sub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import ensemble_bundle  # noqa: E402
from tools.import_models import verify_models  # noqa: E402

# Tracked paths that the target machine needs. Tests, CI files and the
# dashboard's sources stay in the repository.
INCLUDE = (
    ".env.example",
    "README.md",
    "deploy.py",
    "pyproject.toml",
    "requirements.txt",
    "data-pipeline/",
    "docs/",
    "tools/",
    "vision_runtime/",
)
EXCLUDE = ("data-pipeline/tests/", "data-pipeline/uv.lock", "docs/api_list.xlsx")
# The llama.cpp release also ships GPU, RPC and benchmark pieces; a CPU server
# needs only these.
LLAMA_SKIP = ("cuda", "rpc", "bench", "perplexity", "quantize", "cli-impl")
LLAMA_SKIP += ("completion-impl", "fit-params")
# The review model's source, recorded in gemma/gemma.json. Qwen 3.5-4B gave
# better review verdicts than Gemma 4 E4B on the same 592 held-out windows
# (macro F1 0.600 against 0.406). Both run with the same server flags; Qwen
# reads a 640x360 frame as about 220 tokens and Gemma about 120, so a Qwen
# review takes about 1.3 times as long on the CPU (docs/deployment.md).
REVIEWERS = {
    "qwen35-4b": {
        "repository": "https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF",
        "revision": "8d43d974efc48399cb58957ad49c8b26bf5a109c",
        "source_files": "Qwen3.5-4B-Q4_K_M.gguf, mmproj-Qwen3.5-4B-Q8_0.gguf",
        "license": "apache-2.0 (as stated on the model card)",
        "alias": "qwen35-4b",
    },
    "gemma4-e4b": {
        "repository": "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf",
        "revision": "4b4a2c1d584be7264f87aac328a1bc739ce81b6c",
        "license": "apache-2.0 (as stated on the model card)",
        "alias": "gemma4-e4b",
    },
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def place(source, target):
    """Copy a file; hard-link it instead when it is a large weight file on the
    same disk. Small files are always copied so that editing a bundle can
    never change the checkout it was built from."""
    source = Path(source).resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size >= 32 * 1024**2:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copyfile(source, target)
    shutil.copymode(source, target)


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ensemble-export", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--gemma-weights",
        "--reviewer-weights",
        type=Path,
        help="Folder holding the review model's weights and mmproj .gguf files",
    )
    parser.add_argument("--reviewer", choices=list(REVIEWERS), default="qwen35-4b")
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--wheels", action="store_true")
    parser.add_argument(
        "--wheels-from",
        type=Path,
        help="Reuse the wheels folder of an earlier bundle (same pinned "
        "requirements) instead of downloading them again",
    )
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if bool(args.gemma_weights) != bool(args.llama):
        parser.error("Give both --gemma-weights and --llama, or neither")
    if git("status", "--porcelain") and not args.allow_dirty:
        raise SystemExit("Commit or stash changes first; a bundle names one commit")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    components = {}

    tracked = [
        name
        for name in git("ls-files").splitlines()
        if name.startswith(INCLUDE) and not name.startswith(EXCLUDE)
    ]
    for name in tracked:
        place(ROOT / name, out / name)
    shutil.copyfile(ROOT / "tools/verify_transfer.py", out / "verify_transfer.py")
    shutil.copyfile(ROOT / "docs/START_HERE.md", out / "START_HERE.md")
    components["application"] = (
        f"{len(tracked)} files at commit {git('rev-parse', '--short', 'HEAD')}"
    )

    subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", check=True)
    shutil.copytree(ROOT / "frontend/dist", out / "frontend/dist")
    components["dashboard"] = "built React app in frontend/dist (no Node.js needed)"

    models = verify_models(ROOT / "models")
    place(ROOT / "models/manifest.json", out / "models/manifest.json")
    for item in models["files"]:
        place(ROOT / "models" / item["path"], out / "models" / item["path"])
    ensemble_bundle.build(
        args.ensemble_export, args.hf_cache, out / "models/ensemble-20261116"
    )
    ensemble_bundle.verify(out / "models/ensemble-20261116")
    components["models"] = (
        f"{len(models['files'])} detector/October files and ensemble-20261116"
    )

    if args.gemma_weights:
        for source in sorted(args.llama.iterdir()):
            keep = source.name in {"llama-server", "LICENSE"} or (
                source.name.startswith("lib") and ".so" in source.name
            )
            if keep and not any(word in source.name for word in LLAMA_SKIP):
                if source.is_symlink():
                    # Keep the release's library aliases as real files.
                    place(source.resolve(), out / "gemma/llama" / source.name)
                else:
                    place(source, out / "gemma/llama" / source.name)
        weights = sorted(args.gemma_weights.glob("*.gguf"))
        if len(weights) != 2:
            raise SystemExit("Expected the review model's weights and its mmproj file")
        source_info = REVIEWERS[args.reviewer]
        family = source_info["alias"].split("-")[0]
        if not all(family in sub("[-_.]", "", w.name.lower()) for w in weights):
            raise SystemExit(f"Those .gguf files are not {args.reviewer}'s")
        for source in weights:
            place(source, out / "gemma/models" / source.name)
        (out / "gemma/gemma.json").write_text(
            json.dumps(
                {
                    **source_info,
                    "files": {w.name: sha256(w) for w in weights},
                    "server": "llama.cpp CPU build; see gemma/llama/LICENSE",
                },
                indent=2,
            )
            + "\n"
        )
        components["gemma"] = (
            f"{source_info['alias']} 4-bit weights and a CPU llama.cpp server"
        )
    else:
        components["gemma"] = "not included; configure a server with --gemma-url"

    if args.wheels_from:
        wheels = sorted(args.wheels_from.glob("*.whl"))
        if not wheels:
            raise SystemExit(f"No wheels in {args.wheels_from}")
        for source in wheels:
            place(source, out / "wheels" / source.name)
        components["wheels"] = (
            f"{len(wheels)} wheels for Python {sys.version_info.major}.{sys.version_info.minor} "
            "x86_64 Linux; setup installs offline"
        )
    elif args.wheels:
        pip = [sys.executable, "-m", "pip", "download", "--only-binary=:all:"]
        pip += ["--dest", str(out / "wheels"), "--disable-pip-version-check"]
        subprocess.run(
            pip
            + ["torch==2.8.0+cpu", "torchvision==0.23.0+cpu"]
            + ["--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
        subprocess.run(
            pip
            + ["-r", str(ROOT / "requirements.txt"), "setuptools==78.1.0"]
            + ["wheel==0.45.1", "--find-links", str(out / "wheels")]
            + ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
        count = len(list((out / "wheels").glob("*.whl")))
        components["wheels"] = (
            f"{count} wheels for Python {sys.version_info.major}.{sys.version_info.minor} "
            "x86_64 Linux; setup installs offline"
        )
    else:
        components["wheels"] = "not included; setup needs internet access"

    files = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.is_symlink():
            raise SystemExit(f"Unexpected link in bundle: {path}")
        files.append(
            {
                "path": path.relative_to(out).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 2,
        "package": out.name,
        "created_date": datetime.date.today().isoformat(),
        "source_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "source_commit": git("rev-parse", "HEAD"),
        "components": components,
        "files": files,
    }
    (out / "DEPLOYMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(f["bytes"] for f in files)
    print(f"{out}: {len(files)} files, {total / 1024**3:.2f} GiB")
    if args.archive:
        archive = out.with_suffix(".tar")
        # Weights and wheels are already compressed; plain tar is fastest.
        with tarfile.open(archive, "w") as tar:
            tar.add(out, arcname=out.name)
        Path(str(archive) + ".sha256").write_text(
            f"{sha256(archive)}  {archive.name}\n"
        )
        print(f"{archive}: {archive.stat().st_size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
