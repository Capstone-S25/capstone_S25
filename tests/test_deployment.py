"""Deployment contracts: integrity, safe reruns, configuration and service routing."""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import deploy
from tools import import_models as models


@pytest.fixture
def archive(tmp_path):
    payloads = {}
    for source, target in models.model_mapping().items():
        data = target.encode()
        if target.endswith("provenance.json"):
            data = json.dumps(
                {
                    "definitions_sha256": hashlib.sha256(
                        (models.ROOT / "vision_runtime/definitions.json").read_bytes()
                    ).hexdigest()
                }
            ).encode()
        elif target.endswith("state.json"):
            data = b'{"status":"complete"}'
        payloads[source] = data
    manifest = {
        "model": models.MODEL,
        "base_model": {"revision": models.REVISION},
        "source_commit": "fixture",
        "files": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in payloads.items()
        ],
    }
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in {
            **payloads,
            "bundle_manifest.json": json.dumps(manifest).encode(),
        }.items():
            info = tarfile.TarInfo("transfer/" + name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return path


def test_import_verify_and_reuse_without_overwrite(tmp_path, archive):
    destination = tmp_path / "model store"
    entries = models.import_models(archive, destination)
    assert len(entries) == 15
    assert len(models.verify_models(destination)["files"]) == 15
    before = (destination / "manifest.json").read_bytes()
    deploy.prepare_models({"models": str(destination)}, archive)
    assert (destination / "manifest.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        models.import_models(archive, destination)


def test_corruption_and_manifest_escape_fail_closed(tmp_path, archive):
    destination = tmp_path / "models"
    models.import_models(archive, destination)
    target = destination / models.MODEL / "selected_heads.pt"
    target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(ValueError, match="checksum"):
        models.verify_models(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["files"][0]["path"] = "../outside"
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete required file set"):
        models.verify_models(destination)


def test_archive_checksum_required_to_match(archive):
    expected = deploy.digest(archive)
    Path(str(archive) + ".sha256").write_text(expected + "  fixture.tar.gz\n")
    assert deploy.verify_archive(archive) == archive
    with pytest.raises(ValueError, match="checksum mismatch"):
        deploy.verify_archive(archive, "0" * 64)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    (root / "requirements.txt").write_text("fixture==1\n")
    (root / "pyproject.toml").write_text("# fixture\n")
    directory = root / ".venv"
    python = deploy.environment_python(directory)
    python.parent.mkdir(parents=True)
    python.write_text("fixture")
    (directory / "pyvenv.cfg").write_text("# fixture")
    monkeypatch.setattr(deploy, "ROOT", root)
    return root, directory


def test_dependency_resume_and_changed_requirements_reinstall(environment, monkeypatch):
    root, directory = environment
    status = {
        "ready": True,
        "cpu_torch": True,
        "python": [3, 12, 3],
        "prefix": str(directory),
    }
    monkeypatch.setattr(deploy, "child", lambda *_: status)
    calls = []
    monkeypatch.setattr(deploy, "run", lambda *a, **k: calls.append((a, k)))
    deploy.install_dependencies(directory)
    assert calls
    calls.clear()
    deploy.install_dependencies(directory)
    assert not calls
    (root / "requirements.txt").write_text("fixture==2\n")
    deploy.install_dependencies(directory)
    assert calls


def test_dependency_receipt_does_not_hide_environment_drift(environment, monkeypatch):
    _, directory = environment
    status = {
        "ready": True,
        "cpu_torch": True,
        "python": [3, 12, 3],
        "prefix": str(directory),
    }
    monkeypatch.setattr(deploy, "child", lambda *_: status)
    calls = []
    monkeypatch.setattr(deploy, "run", lambda *a, **k: calls.append(a))
    deploy.install_dependencies(directory)
    calls.clear()
    statuses = iter([{**status, "ready": False}, status])
    monkeypatch.setattr(deploy, "child", lambda *_: next(statuses))
    deploy.install_dependencies(directory)
    assert calls


def test_unmanaged_directory_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    directory = tmp_path / "existing files"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("do not remove")
    with pytest.raises(RuntimeError, match="not a venv"):
        deploy.install_dependencies(directory)
    assert marker.read_text() == "do not remove"


@pytest.fixture
def config_checkout(tmp_path, monkeypatch):
    original = deploy.ROOT
    root = tmp_path / "config checkout with spaces"
    (root / "vision_runtime").mkdir(parents=True)
    (root / "vision_runtime/__init__.py").touch()
    shutil.copyfile(
        original / "vision_runtime/config.py", root / "vision_runtime/config.py"
    )
    shutil.copyfile(original / "deploy.py", root / "deploy.py")
    shutil.copyfile(original / ".env.example", root / ".env.example")
    monkeypatch.setattr(deploy, "ROOT", root)
    for name in [
        "GEMMA_BASE_URL",
        "GEMMA_MODEL",
        "DATABASE_URL",
        "VISION_MODELS_DIR",
        "VISION_RUNTIME_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)
    return root


def test_real_dotenv_roundtrip_preserves_existing_configuration(config_checkout):
    root = config_checkout
    content = 'VISION_MODELS_DIR="a model folder"\n# Keep this comment\nAPI_TOKEN="private fixture"\n'
    (root / ".env").write_text(content)
    args = SimpleNamespace(gemma_url=None, gemma_model=None)
    settings = deploy.configure(Path(sys.executable), args)
    assert settings["models"] == str(root / "a model folder")
    assert (root / ".env").read_text() == content
    args.gemma_url = "http://127.0.0.1:8080/v1"
    args.gemma_model = "local-model"
    deploy.configure(Path(sys.executable), args)
    before = (root / ".env").read_bytes()
    deploy.configure(Path(sys.executable), args)
    assert (root / ".env").read_bytes() == before
    args.gemma_url = "http://127.0.0.1:9000/v1"
    with pytest.raises(ValueError, match="already configured"):
        deploy.configure(Path(sys.executable), args)
    assert (root / ".env").read_bytes() == before


def test_setup_lock_cannot_be_taken_twice_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    with deploy.setup_lock():
        with pytest.raises(RuntimeError, match="Another setup"):
            with deploy.setup_lock():
                pass
    with deploy.setup_lock():
        pass


def test_external_database_does_not_start_or_reset_local_compose(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    monkeypatch.setattr(
        deploy,
        "child",
        lambda _, action: (
            {"database": "external"}
            if action == "_settings"
            else {"database_ready": True}
        ),
    )
    calls = []
    monkeypatch.setattr(
        deploy, "run", lambda command, *a, **kw: calls.append((command, kw))
    )
    deploy.camera_setup(Path(sys.executable))
    assert len(calls) == 1
    assert calls[0][0][1:] == ["-m", "alembic", "upgrade", "head"]
    assert calls[0][1]["cwd"] == tmp_path / "data-pipeline"


def test_child_failure_has_actionable_message_without_traceback(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            ["private action"],
            output='{"error":"Database migrations are not current; run setup --with-camera"}',
        )

    monkeypatch.setattr(deploy, "run", fail)
    with pytest.raises(RuntimeError, match="run setup --with-camera"):
        deploy.child(Path(sys.executable), "_database")


def test_gemma_healthcheck_lists_models_without_sending_evidence(monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            body = b'{"data":[{"id":"fixture-model"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv(
            "GEMMA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
        )
        monkeypatch.setenv("GEMMA_MODEL", "fixture-model")
        monkeypatch.setenv("GEMMA_API_KEY", "fixture-token")
        try:
            assert deploy.gemma_check()["gemma_endpoint_ready"]
            monkeypatch.setenv("GEMMA_MODEL", "not-served")
            with pytest.raises(RuntimeError, match="not advertised"):
                deploy.gemma_check()
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert calls == [("/v1/models", "Bearer fixture-token")] * 2


def start_args(**changes):
    values = dict(
        camera=False,
        dashboard=False,
        video=None,
        rtsp_url=None,
        source_id="camera_01",
        host="127.0.0.1",
        port=8501,
        api_port=8000,
    )
    return SimpleNamespace(**{**values, **changes})


def test_full_start_plans_gemma_api_and_camera_in_stop_order(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    monkeypatch.setattr(
        deploy, "local_gemma", lambda: (Path("server"), Path("w.gguf"), Path("p.gguf"))
    )
    local = {"port": 8090, "model": "gemma4-e4b", "threads": 4, "image_tokens": 280}
    planned = deploy.services(
        Path("python"),
        start_args(camera=True, dashboard=True, video=video),
        {"gemma_local": local},
    )
    assert [(name, role) for name, role, *_ in planned] == [
        ("Gemma (CPU)", "helper"),
        ("Dashboard and API", "server"),
        ("Camera backend", "job"),
    ]
    gemma = [str(part) for part in planned[0][2]]
    assert gemma[gemma.index("--alias") + 1] == "gemma4-e4b"
    assert gemma[gemma.index("--n-gpu-layers") + 1] == "0"
    assert planned[1][4]["DASHBOARD_DIR"].endswith("frontend/dist")
    # A live camera never ends by itself.
    live = deploy.services(
        Path("python"), start_args(camera=True, rtsp_url="rtsp://x/y"), {}
    )
    assert [(name, role) for name, role, *_ in live] == [("Camera backend", "server")]


def test_dashboard_alone_starts_no_model_process(monkeypatch):
    monkeypatch.setattr(
        deploy, "local_gemma", lambda: (Path("s"), Path("w"), Path("p"))
    )
    planned = deploy.services(
        Path("python"), start_args(dashboard=True), {"gemma_local": {"port": 8090}}
    )
    assert [name for name, *_ in planned] == ["Dashboard and API"]
    # The API package lives in data-pipeline/, so it must start there.
    assert planned[0][3] == deploy.ROOT / "data-pipeline"
    upload = deploy.services(Path("python"), start_args(), {})
    assert [name for name, *_ in upload] == ["Upload interface"]


def test_bundled_gemma_needs_server_weights_and_projector(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "GEMMA", tmp_path)
    assert deploy.local_gemma() is None
    (tmp_path / "llama").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "llama/llama-server").write_bytes(b"")
    (tmp_path / "models/gemma-it.gguf").write_bytes(b"")
    assert deploy.local_gemma() is None
    (tmp_path / "models/gemma-mmproj.gguf").write_bytes(b"")
    server, weights, projector = deploy.local_gemma()
    assert (weights.name, projector.name) == ("gemma-it.gguf", "gemma-mmproj.gguf")


@pytest.mark.parametrize("reachable", [True, False])
def test_local_database_is_used_when_running_else_instructions(
    tmp_path, monkeypatch, reachable
):
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    answers = {
        "_settings": {"database": "compose"},
        "_database_ping": {"reachable": reachable},
        "_database": {"database_ready": True},
    }
    monkeypatch.setattr(deploy, "child", lambda _, action: answers[action])
    monkeypatch.setattr(deploy.shutil, "which", lambda name: None)  # no Docker
    calls = []
    monkeypatch.setattr(deploy, "run", lambda command, *a, **kw: calls.append(command))
    if reachable:
        deploy.camera_setup(Path(sys.executable))
        assert [c[1:] for c in calls] == [["-m", "alembic", "upgrade", "head"]]
    else:
        with pytest.raises(FileNotFoundError, match="sudo apt install postgresql"):
            deploy.camera_setup(Path(sys.executable))
        assert not calls
