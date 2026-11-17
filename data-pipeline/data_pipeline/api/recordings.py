from pathlib import Path

from fastapi import Request


def get_recordings_dir(request: Request) -> Path:
    """Return the recordings root that evidence clip paths are relative to."""
    return request.app.state.storage_root
