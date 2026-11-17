from pathlib import Path

from fastapi import Request


def get_recordings_dir(request: Request) -> Path:
    """Return the folder that evidence clip paths are relative to."""
    return request.app.state.recordings_dir
