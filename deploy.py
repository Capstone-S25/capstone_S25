"""One-command CPU setup, readiness checks and application startup (Python 3.12)."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
KNOWN_BUNDLE_SHA256 = "efabaddee92e5036918932ba0e4eb3840292124f50a8fe2b8bcb95f2786119fc"
LOCAL_DATABASE = "postgresql+psycopg://capstone:password@127.0.0.1:5432/capstone"
INSTALL_REVISION = "cpu-setup-v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    env["PIP_CACHE_DIR"] = str(ROOT / ".local/deploy/pip-cache")
    return env


def run(command, label, *, cwd=None, capture=False, timeout=None):
    if not capture:
        print(label, flush=True)
    return subprocess.run(
        [str(c) for c in command],
        cwd=cwd or ROOT,
        env=environment(),
        text=True,
        capture_output=capture,
        check=True,
        timeout=timeout,
    )


def child(python, action):
    try:
        result = run(
            [python, ROOT / "deploy.py", action], action, capture=True, timeout=30
        )
    except subprocess.CalledProcessError as exc:
        try:
            error = json.loads(exc.stdout)["error"]
        except (ValueError, KeyError):
            error = (
                "Environment check failed; rerun setup to repair missing dependencies"
            )
        raise RuntimeError(error) from None
    return json.loads(result.stdout)


@contextlib.contextmanager
def setup_lock():
    """OS lock releases even if setup crashes; no stale PID-file deletion."""
    path = ROOT / ".local/deploy/setup.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another setup is running for this checkout") from None
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def environment_python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_archive(archive, expected=None):
    archive = archive.expanduser().resolve(strict=True)
    sidecar = Path(str(archive) + ".sha256")
    if expected is None and sidecar.is_file():
        parts = sidecar.read_text().split()
        if not parts:
            raise ValueError("Archive checksum sidecar is empty")
        expected = parts[0]
    expected = expected or KNOWN_BUNDLE_SHA256
    if not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
        raise ValueError("Expected a 64-character SHA-256 checksum")
    if digest(archive) != expected.lower():
        raise ValueError(
            "Archive checksum mismatch; existing model files were not touched"
        )
    return archive


def install_dependencies(directory):
    python = environment_python(directory)
    if directory.exists() and not (directory / "pyvenv.cfg").is_file():
        raise RuntimeError(
            "The selected environment directory exists but is not a venv; choose another --venv"
        )
    if not python.is_file():
        if shutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise OSError(
                "A fresh CPU environment needs at least 3 GiB free, plus model storage"
            )
        print("Creating isolated CPU environment", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    status = child(python, "_packages")
    if status["python"][:2] != [3, 12] or Path(status["prefix"]).resolve() != directory:
        raise RuntimeError(
            "Existing venv must use Python 3.12 and belong to --venv; choose another directory"
        )
    stamp = hashlib.sha256(
        (ROOT / "requirements.txt").read_bytes()
        + (ROOT / "pyproject.toml").read_bytes()
        + INSTALL_REVISION.encode()
    ).hexdigest()
    receipt = (
        ROOT
        / ".local/deploy"
        / (hashlib.sha256(str(directory).encode()).hexdigest()[:16] + ".json")
    )
    try:
        previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    except (OSError, ValueError):
        previous = {}
    if previous.get("dependency_stamp") == stamp and status["ready"]:
        print("Dependencies already match; reusing environment", flush=True)
        return python
    run([python, "-m", "ensurepip", "--upgrade"], "Checking pip")
    pip = [python, "-m", "pip", "--disable-pip-version-check", "--no-input"]
    if not status["cpu_torch"]:
        run(
            pip
            + [
                "install",
                "torch==2.8.0+cpu",
                "torchvision==0.23.0+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            "Installing CPU-only Torch",
        )
    run(
        pip
        + [
            "install",
            "-r",
            ROOT / "requirements.txt",
            "setuptools==78.1.0",
            "wheel==0.45.1",
        ],
        "Installing application dependencies",
    )
    run(
        pip + ["install", "-e", ROOT, "--no-deps", "--no-build-isolation"],
        "Registering the application checkout",
    )
    result = child(python, "_packages")
    if not result["ready"]:
        raise RuntimeError(
            "Dependency verification failed: " + ", ".join(result["problems"])
        )
    atomic_json(
        receipt,
        {"dependency_stamp": stamp, "venv": str(directory), "python": result["python"]},
    )
    return python


def packages():
    problems = []
    try:
        from packaging.requirements import Requirement

        for line in (ROOT / "requirements.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            try:
                version = importlib.metadata.version(requirement.name)
                if version not in requirement.specifier:
                    problems.append(requirement.name + " version mismatch")
            except importlib.metadata.PackageNotFoundError:
                problems.append(requirement.name + " missing")
        distribution = importlib.metadata.distribution("capstone-vision-application")
        location = json.loads(distribution.read_text("direct_url.json") or "{}")
        if location.get("url") != ROOT.as_uri() or not location.get("dir_info", {}).get(
            "editable"
        ):
            problems.append("application is registered to a different checkout")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        problems.append("application/packaging not installed")
    try:
        cpu = all(
            importlib.metadata.version(name) == version
            for name, version in [("torch", "2.8.0+cpu"), ("torchvision", "0.23.0+cpu")]
        )
    except importlib.metadata.PackageNotFoundError:
        cpu = False
    return {
        "ready": not problems,
        "problems": problems,
        "cpu_torch": cpu,
        "python": list(sys.version_info[:3]),
        "prefix": sys.prefix,
    }


def settings():
    from dotenv import load_dotenv

    from vision_runtime.config import configure_runtime, models_root, runtime_root

    configure_runtime()
    load_dotenv(ROOT / "data-pipeline/.env", override=False)
    database = os.environ.get("DATABASE_URL")
    return {
        "models": str(models_root()),
        "runtime": str(runtime_root()),
        "database": "unset"
        if not database
        else "compose"
        if database == LOCAL_DATABASE
        else "external",
        "gemma": bool(os.environ.get("GEMMA_BASE_URL")),
        "values": {
            key: hashlib.sha256(os.environ[key].encode()).hexdigest()
            for key in ["DATABASE_URL", "GEMMA_BASE_URL", "GEMMA_MODEL"]
            if os.environ.get(key)
        },
    }


def append_setting(python, key, value, *, preserve=False):
    previous = child(python, "_settings")["values"].get(key)
    if previous:
        if not preserve and previous != hashlib.sha256(value.encode()).hexdigest():
            raise ValueError(
                key + " already configured; edit .env explicitly to change it"
            )
        return
    with (ROOT / ".env").open("a", encoding="utf-8") as stream:
        stream.write("\n" + key + "=" + json.dumps(value) + "\n")


def configure(python, args):
    path = ROOT / ".env"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write((ROOT / ".env.example").read_text())
        if os.name != "nt":
            path.chmod(0o600)
    if args.gemma_url:
        url = urlsplit(args.gemma_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not url.path.rstrip("/").endswith("/v1")
        ):
            raise ValueError(
                "--gemma-url must be an HTTP(S) endpoint ending /v1, without credentials or query parameters"
            )
        append_setting(python, "GEMMA_BASE_URL", args.gemma_url.rstrip("/"))
    if args.gemma_model or args.gemma_url:
        append_setting(
            python,
            "GEMMA_MODEL",
            args.gemma_model or "gemma4-26b-a4b",
            preserve=args.gemma_model is None,
        )
    config = child(python, "_settings")
    Path(config["runtime"]).mkdir(parents=True, exist_ok=True)
    return config


def prepare_models(config, archive):
    from tools.import_models import import_models, verify_models

    directory = Path(config["models"])
    if directory.exists():
        verify_models(directory)
        print("Existing model files verified; preserving them", flush=True)
    else:
        if archive is None:
            raise FileNotFoundError(
                "Models are not installed; rerun setup with --bundle /path/to/transfer.tar.gz"
            )
        import_models(archive, directory)
        verify_models(directory)
        print("Imported and verified 15 model files", flush=True)


def camera_setup(python):
    config = child(python, "_settings")
    if config["database"] == "unset":
        append_setting(python, "DATABASE_URL", LOCAL_DATABASE)
    if config["database"] in {"unset", "compose"}:
        if not shutil.which("docker"):
            raise FileNotFoundError(
                "Install Docker Compose for the local database, or configure an existing DATABASE_URL"
            )
        run(["docker", "compose", "version"], "Checking Docker Compose")
        run(
            [
                "docker",
                "compose",
                "-f",
                ROOT / "data-pipeline/infra/compose.yaml",
                "up",
                "-d",
                "--wait",
                "--wait-timeout",
                "90",
                "postgres",
            ],
            "Starting the local database",
        )
    run(
        [python, "-m", "alembic", "upgrade", "head"],
        "Applying database migrations",
        cwd=ROOT / "data-pipeline",
    )
    child(python, "_database")
    print("Camera database schema is ready", flush=True)


def database_check():
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine

    settings()
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("Camera mode needs DATABASE_URL; run setup --with-camera")
    engine = create_engine(
        os.environ["DATABASE_URL"], connect_args={"connect_timeout": 5}
    )
    try:
        with engine.connect() as connection:
            actual = set(MigrationContext.configure(connection).get_current_heads())
        expected = set(
            ScriptDirectory.from_config(
                Config(str(ROOT / "data-pipeline/alembic.ini"))
            ).get_heads()
        )
        if actual != expected:
            raise RuntimeError(
                "Database migrations are not current; run setup --with-camera"
            )
    finally:
        engine.dispose()
    return {"database_ready": True}


def smoke():
    import math
    from uuid import uuid4

    import numpy as np
    import torch
    from PIL import Image

    from vision_runtime.classifier import ENSEMBLE, load_classifier
    from vision_runtime.config import models_root

    settings()
    sys.path.insert(0, str(ROOT / "data-pipeline"))
    from data_pipeline.detection.stage1 import Stage1Detector
    from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

    if torch.version.cuda is not None:
        raise RuntimeError("Expected the pinned CPU-only Torch build")
    with tempfile.TemporaryDirectory(
        prefix="smoke-", dir=Path(settings()["runtime"])
    ) as temp:
        folder = Path(temp)
        path = folder / "frame.png"
        Image.new("RGB", (96, 96), (110, 120, 130)).save(path)
        tasks = [
            "smoking",
            "littering",
            "loitering",
            "rough_sleeping",
            "unattended_items",
        ]
        view = {"paths": [str(path)], "times_s": [0.0]}
        # The ensemble is the default classifier when its folder is installed.
        name = (
            ENSEMBLE
            if (models_root() / ENSEMBLE / "ensemble.json").is_file()
            else "mobileclip2-s2"
        )
        print("Smoke classifier:", name, flush=True)
        result = load_classifier(name, "cpu").classify(
            {"tasks": tasks, "timestamp_s": 0, "views": {"short": view, "long": view}}
        )
        for task in tasks:
            value = result["predictions"][task]
            if task in {"littering", "loitering", "unattended_items"}:
                if value["decision"] != "unknown":
                    raise RuntimeError("Temporal support check failed")
            elif not math.isfinite(value["score"]):
                raise RuntimeError("Classifier returned a nonfinite score")
        detector = Stage1Detector("setup", folder, fps=5)
        try:
            session = uuid4()
            for index in range(2):
                detector.process(
                    DecodedFrame(
                        0,
                        "setup",
                        session,
                        index,
                        index * 200,
                        np.full((128, 128, 3), 110, np.uint8),
                    )
                )
            if not detector.finish():
                raise RuntimeError("Tracker did not produce a final scene review")
        finally:
            detector._close_pool()
    print(
        "CPU smoke passed: classifier, five tasks, temporal guards, YOLO/pose, ByteTrack and fitted checks",
        flush=True,
    )


def gemma_check():
    import requests

    settings()
    url = os.environ.get("GEMMA_BASE_URL")
    if not url:
        raise RuntimeError("GEMMA_BASE_URL is not configured")
    key = os.environ.get("GEMMA_API_KEY")
    response = requests.get(
        url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + key} if key else {},
        timeout=(5, 10),
    )
    response.raise_for_status()
    models = {item["id"] for item in response.json()["data"]}
    if os.environ.get("GEMMA_MODEL", "gemma4-26b-a4b") not in models:
        raise RuntimeError("Configured Gemma alias was not advertised by the server")
    return {
        "gemma_endpoint_ready": True,
        "scope": "Model listing only, not image inference",
    }


def check(python, *, camera=False, full_hashes=True):
    if not python.is_file():
        raise FileNotFoundError(
            "Environment is missing; run python3.12 deploy.py setup --bundle ..."
        )
    status = child(python, "_packages")
    if not status["ready"] or status["python"][:2] != [3, 12]:
        raise RuntimeError("Environment needs setup: " + ", ".join(status["problems"]))
    config = child(python, "_settings")
    from tools.import_models import verify_models

    verify_models(Path(config["models"]), hashes=full_hashes)
    if camera:
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("Camera recording needs ffmpeg on PATH")
        child(python, "_database")
    return config


def startup_command(python, args):
    if args.camera:
        if not args.rtsp_url and not args.video:
            raise ValueError("Camera mode requires --rtsp-url or --video")
        command = [
            python,
            ROOT / "data-pipeline/main.py",
            "--device",
            "cpu",
            "--source-id",
            args.source_id,
        ]
        return command + (
            ["--source", "rtsp", "--rtsp-url", args.rtsp_url]
            if args.rtsp_url
            else ["--video", str(args.video.resolve(strict=True))]
        )
    if args.rtsp_url or args.video:
        raise ValueError(
            "Use --camera for a recording backend source; uploads use the UI"
        )
    return [
        python,
        "-m",
        "streamlit",
        "run",
        ROOT / "app.py",
        "--server.headless",
        "true",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]


def main():
    # Private subprocess actions use the selected venv without importing ML into
    # the installer process. They never print .env values or database URLs.
    if len(sys.argv) == 2 and sys.argv[1].startswith("_"):
        actions = {
            "_packages": packages,
            "_settings": settings,
            "_database": database_check,
            "_gemma": gemma_check,
            "_smoke": smoke,
        }
        try:
            value = actions[sys.argv[1]]()
        except Exception as exc:
            message = str(exc)
            for key in ["DATABASE_URL", "GEMMA_BASE_URL", "GEMMA_API_KEY"]:
                if os.environ.get(key):
                    message = message.replace(os.environ[key], "<configured service>")
            print(json.dumps({"error": message}))
            sys.exit(1)
        if value is not None:
            print(json.dumps(value))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ["setup", "check", "start"]:
        p = sub.add_parser(name)
        p.add_argument(
            "--venv",
            type=Path,
            default=ROOT / ".venv",
            help="Environment directory; default: checkout/.venv",
        )
        if name == "setup":
            p.add_argument("--bundle", type=Path)
            p.add_argument(
                "--sha256",
                help="Expected transfer archive checksum; otherwise use sidecar or the pinned 3 November checksum",
            )
            p.add_argument("--with-camera", action="store_true")
            p.add_argument("--gemma-url")
            p.add_argument("--gemma-model")
        elif name == "check":
            p.add_argument("--camera", action="store_true")
            p.add_argument("--smoke", action="store_true")
            p.add_argument(
                "--gemma",
                action="store_true",
                help="Probe configured /models endpoint; does not send images",
            )
        else:
            p.add_argument("--camera", action="store_true")
            source = p.add_mutually_exclusive_group()
            source.add_argument("--rtsp-url")
            source.add_argument("--video", type=Path)
            p.add_argument("--source-id", default="camera_01")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    args.venv = args.venv.expanduser().resolve()
    python = environment_python(args.venv)
    if args.action == "setup":
        if (
            sys.version_info[:2] != (3, 12)
            or platform.machine().lower() not in {"x86_64", "amd64"}
            or sys.platform not in {"linux", "win32"}
        ):
            raise RuntimeError(
                "The pinned CPU wheels require Python 3.12 on x86_64 Linux/Windows; Linux is the validated platform"
            )
        if args.sha256 and not args.bundle:
            raise ValueError("--sha256 requires --bundle")
        if args.with_camera and not shutil.which("ffmpeg"):
            raise FileNotFoundError(
                "Install ffmpeg before camera setup; uploads do not require its executable"
            )
        with setup_lock():
            archive = verify_archive(args.bundle, args.sha256) if args.bundle else None
            python = install_dependencies(args.venv)
            config = configure(python, args)
            prepare_models(config, archive)
            run(
                [python, ROOT / "deploy.py", "_smoke"],
                "Testing real CPU inference and tracking",
                timeout=180,
            )
            if args.with_camera:
                camera_setup(python)
            print(
                "Setup complete. Start with: python deploy.py start"
                + (" --venv " + str(args.venv) if args.venv != ROOT / ".venv" else "")
            )
    elif args.action == "check":
        config = check(python, camera=args.camera)
        if args.smoke:
            run(
                [python, ROOT / "deploy.py", "_smoke"],
                "Testing CPU inference",
                timeout=180,
            )
        if args.gemma:
            child(python, "_gemma")
        print(
            "Ready: CPU application and verified model files; Gemma "
            + ("configured" if config["gemma"] else "disabled")
        )
    else:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        command = startup_command(python, args)
        check(python, camera=args.camera, full_hashes=False)
        print(
            "Starting camera backend"
            if args.camera
            else f"Upload UI: http://{args.host}:{args.port}",
            flush=True,
        )
        os.chdir(ROOT)
        os.execve(str(python), [str(c) for c in command], environment())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted. Existing models/configuration are preserved; rerun setup to resume.",
            file=sys.stderr,
        )
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(
            f"Deployment step failed (exit {exc.returncode}). Fix the reported prerequisite and rerun the same command.",
            file=sys.stderr,
        )
        if exc.stderr:
            print(
                "Readiness check failed; rerun setup or check the configured service.",
                file=sys.stderr,
            )
        sys.exit(1)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        sys.exit(1)
