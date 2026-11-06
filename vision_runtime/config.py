"""Portable application paths; weights and generated media remain outside Git."""

import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def models_root():
    path = Path(os.environ.get("VISION_MODELS_DIR", "models")).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def runtime_root():
    path = Path(os.environ.get("VISION_RUNTIME_DIR", "runtime")).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def model_paths(model="mobileclip2-s2"):
    if model not in {"mobileclip2-s2", "siglip2-b16-224"}:
        raise ValueError("Unsupported classifier")
    directory = models_root() / model
    return SimpleNamespace(directory=directory, base=directory / "base")


def configure_runtime():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_root() / "yolo"))


def resolve_device(device):
    import torch

    if device == "cpu":
        return device
    if device.startswith("cuda") and torch.cuda.is_available():
        return str(torch.device(device))
    raise ValueError("Choose cpu or an available CUDA device")
