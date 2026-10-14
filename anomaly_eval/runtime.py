"""Explicit inference placement; accuracy runs never hide available GPUs."""

from __future__ import annotations

import os


def available_threads() -> int:
    return (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )


def resolve_device(requested="auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is unavailable. Check the NVIDIA driver and install CUDA-enabled PyTorch."
            )
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {requested}")
        return str(device)
    raise ValueError("device must be auto, cpu, cuda, or cuda:N")


def resolve_dtype(requested, device, generative=False):
    import torch

    if requested == "auto":
        if device.startswith("cuda") and generative:
            with torch.cuda.device(device):
                return (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        return torch.float32
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested not in values:
        raise ValueError("Unsupported inference dtype")
    if device == "cpu" and requested == "float16":
        raise ValueError("CPU inference requires float32 or bfloat16")
    return values[requested]


def adapter_options(adapter, options, generative=False):
    options = dict(options)
    adapter.device = resolve_device(options.pop("_device", "cpu"))
    adapter.dtype = resolve_dtype(
        options.pop("_dtype", "float32"), adapter.device, generative
    )
    return options


def move_inputs(inputs, device, dtype):
    return {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
        for key, value in inputs.items()
    }


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def add_device_argument(parser, default="auto"):
    parser.add_argument(
        "--device",
        default=default,
        help="auto (GPU when available), cpu, cuda, or cuda:N",
    )
