from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    USB = "usb"
    RTSP = "rtsp"
    FILE = "file"


@dataclass(frozen=True)
class VideoSource:
    """Describe where video source comes from.

    Attributes:
        source_id   : Identifier used in recording folder names, e.g. "camera_01".
        name        : Human-readable label, e.g. "Entrance camera".
        source_type : SourceType.USB, SourceType.RTSP, or SourceType.FILE.
        location    : Device identifier (e.g. "0"), RTSP URL, or video file path,
                      depending on source_type.
        enabled     : Whether this source may start a session; defaults to True.
    """

    source_id: str
    name: str
    source_type: SourceType
    location: str
    enabled: bool = True
