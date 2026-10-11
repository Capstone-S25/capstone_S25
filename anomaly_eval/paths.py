"""Runtime locations and portable artifact paths for local or remote workspaces."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

PACKAGE_ROOT = Path(__file__).resolve().parent
_CHECKOUT = PACKAGE_ROOT.parent
PROJECT_ROOT = (
    _CHECKOUT
    if (_CHECKOUT / "test_data").is_dir()
    else Path.home() / ".local" / "share" / "capstone-anomaly-evaluation"
)
ENV_PREFIX = "CAPSTONE_"
ROOT_FIELDS = {
    "DATA_ROOT": "data_dir",
    "RESULTS_ROOT": "results_dir",
    "CACHE_ROOT": "cache_dir",
    "BUNDLE_ROOT": "bundle_dir",
    "CONFIG_ROOT": "config_dir",
}


def expand_location(value: str | Path, base: Path | None = None) -> Path:
    text = os.path.expandvars(os.fspath(value))
    if re.search(r"\$\{[^}]+\}", text):
        raise ValueError(f"Undefined environment variable in path: {value}")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else (base or Path.cwd()) / path).resolve()


@dataclass(frozen=True)
class Workspace:
    data_dir: Path
    results_dir: Path
    cache_dir: Path
    bundle_dir: Path
    config_dir: Path

    @classmethod
    def current(cls) -> Workspace:
        defaults = {
            "data_dir": PROJECT_ROOT / "data",
            "results_dir": PROJECT_ROOT / "results",
            "cache_dir": Path(
                os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
            ),
            "bundle_dir": PROJECT_ROOT / "test_data",
            "config_dir": PACKAGE_ROOT / "config",
        }
        return cls(
            **{
                key: expand_location(os.environ.get(ENV_PREFIX + key.upper(), value))
                for key, value in defaults.items()
            }
        )

    @property
    def model_cache(self) -> Path:
        if "CAPSTONE_CACHE_DIR" not in os.environ and os.environ.get("HF_HUB_CACHE"):
            return expand_location(os.environ["HF_HUB_CACHE"])
        return self.cache_dir / "hub"

    @property
    def definitions(self) -> Path:
        return self.config_dir / "anomalies.json"

    def as_dict(self) -> dict[str, str]:
        return {key: str(getattr(self, key)) for key in ROOT_FIELDS.values()}

    def configure_cache(self) -> None:
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.model_cache)


def configure(
    config_file: Path | None = None,
    overrides: dict | None = None,
    path_maps: list[str] | None = None,
) -> Workspace:
    """Precedence: explicit flags > server JSON > existing environment > defaults."""
    values = {}
    if config_file:
        config_file = expand_location(config_file)
        values = json.loads(config_file.read_text())
        if not isinstance(values, dict) or set(values) - set(ROOT_FIELDS.values()):
            raise ValueError(
                "Workspace JSON supports data_dir,results_dir,cache_dir,bundle_dir,config_dir only"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in values.values()
        ):
            raise ValueError("Workspace locations must be nonempty strings")
        values = {
            key: expand_location(value, config_file.parent)
            for key, value in values.items()
        }
    values.update(
        {k: expand_location(v) for k, v in (overrides or {}).items() if v is not None}
    )
    for key, value in values.items():
        if key not in ROOT_FIELDS.values():
            raise ValueError(f"Unknown workspace location: {key}")
        os.environ[ENV_PREFIX + key.upper()] = str(value)
    if path_maps:
        maps = json.loads(os.environ.get("CAPSTONE_PATH_MAPS", "{}"))
        for spec in path_maps:
            old, separator, new = spec.partition("=")
            if not separator or not old or not new:
                raise ValueError("Path maps require OLD_ROOT=NEW_ROOT")
            maps[old.replace("\\", "/").rstrip("/")] = str(expand_location(new))
        os.environ["CAPSTONE_PATH_MAPS"] = json.dumps(maps)
    workspace = Workspace.current()
    workspace.configure_cache()
    return workspace


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    """Read relative/anchored paths, or explicitly remap old absolute artifacts."""
    text = os.fspath(value).replace("\\", "/")
    workspace = Workspace.current()
    for token, field in ROOT_FIELDS.items():
        prefix = "${" + token + "}"
        if text == prefix or text.startswith(prefix + "/"):
            return (
                getattr(workspace, field) / text[len(prefix) :].lstrip("/")
            ).resolve()
    maps = json.loads(os.environ.get("CAPSTONE_PATH_MAPS", "{}"))
    windows = PureWindowsPath(text).is_absolute() and not Path(text).is_absolute()
    for old, new in sorted(maps.items(), key=lambda pair: -len(pair[0])):
        left, right = (text.lower(), old.lower()) if windows else (text, old)
        if left == right or left.startswith(right + "/"):
            return (Path(new) / text[len(old) :].lstrip("/")).resolve()
    if windows:
        raise ValueError(
            "Windows absolute path needs --path-map OLD_ROOT=NEW_ROOT on this server"
        )
    result = expand_location(text, base)
    for old, new in sorted(maps.items(), key=lambda pair: -len(pair[0])):
        absolute = result.as_posix()
        if absolute == old or absolute.startswith(old + "/"):
            return (Path(new) / absolute[len(old) :].lstrip("/")).resolve()
    return result


def portable_path(path: Path, base: Path | None = None) -> str:
    """Prefer owned-artifact relative paths, otherwise named workspace roots."""
    path = path.resolve()
    if base and path.is_relative_to(base.resolve()):
        return path.relative_to(base.resolve()).as_posix()
    workspace = Workspace.current()
    roots = sorted(
        ROOT_FIELDS.items(), key=lambda pair: -len(str(getattr(workspace, pair[1])))
    )
    for token, field in roots:
        root = getattr(workspace, field)
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            return "${" + token + "}" + ("/" + relative if relative != "." else "")
    if base:
        try:
            return Path(os.path.relpath(path, base.resolve())).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def process_environment() -> dict[str, str]:
    """Children can import an editable checkout even when launched elsewhere."""
    env = os.environ.copy()
    parent = str(PACKAGE_ROOT.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = parent + (os.pathsep + existing if existing else "")
    return env


def resolve_arguments(args):
    """Normalize CLI Path values before operations; relative paths use caller cwd."""
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, resolve_path(value))
    return args
