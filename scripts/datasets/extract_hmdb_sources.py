"""Safely extract selected action classes from the author-hosted RAR release."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/strategy_sources_20261016/hmdb51"
CLASSES = {"smoke", "drink", "eat", "chew", "talk", "sit", "stand"}


def extract(archive, destination, accept):
    lib = ctypes.CDLL(ctypes.util.find_library("archive"))
    lib.archive_read_new.restype = ctypes.c_void_p
    for name in [
        "archive_read_support_filter_all",
        "archive_read_support_format_all",
        "archive_read_data_skip",
        "archive_read_free",
    ]:
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.archive_read_open_filename.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.archive_read_next_header.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    lib.archive_entry_pathname.restype = ctypes.c_char_p
    lib.archive_entry_size.argtypes = [ctypes.c_void_p]
    lib.archive_entry_size.restype = ctypes.c_longlong
    lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.archive_read_data.restype = ctypes.c_ssize_t
    lib.archive_error_string.argtypes = [ctypes.c_void_p]
    lib.archive_error_string.restype = ctypes.c_char_p
    handle = lib.archive_read_new()
    extracted = []
    try:
        lib.archive_read_support_filter_all(handle)
        lib.archive_read_support_format_all(handle)
        if lib.archive_read_open_filename(handle, str(archive).encode(), 1048576) != 0:
            raise ValueError(lib.archive_error_string(handle))
        entry = ctypes.c_void_p()
        while True:
            status = lib.archive_read_next_header(handle, ctypes.byref(entry))
            if status == 1:
                break
            if status < 0:
                raise ValueError(lib.archive_error_string(handle))
            name = lib.archive_entry_pathname(entry).decode()
            size = lib.archive_entry_size(entry)
            if not accept(name):
                lib.archive_read_data_skip(handle)
                continue
            target = (destination / name).resolve()
            if (
                not target.is_relative_to(destination.resolve())
                or not 0 < size <= 536870912
            ):
                raise ValueError("Unsafe or oversized archive entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            buffer = ctypes.create_string_buffer(1048576)
            count = 0
            with target.open("wb") as stream:
                while True:
                    got = lib.archive_read_data(handle, buffer, len(buffer))
                    if got == 0:
                        break
                    if got < 0:
                        raise ValueError(lib.archive_error_string(handle))
                    count += got
                    if count > size:
                        raise ValueError("Archive entry exceeded declared size")
                    stream.write(buffer.raw[:got])
            assert count == size
            extracted.append(
                {
                    "path": str(target.relative_to(ROOT)),
                    "bytes": count,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    finally:
        lib.archive_read_free(handle)
    return extracted


def main():
    archive = SOURCE / "hmdb51_org.rar"
    expected = "9e714a0d8b76104d76e932764a7ca636f929fff66279cda3f2e326fa912a328e"
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Author-hosted archive hash mismatch")
    outer = extract(
        archive,
        SOURCE / "class_archives",
        lambda p: Path(p).stem in CLASSES and p.endswith(".rar"),
    )
    rows = []
    for entry in outer:
        source = ROOT / entry["path"]
        rows.extend(extract(source, SOURCE / "videos", lambda p: p.endswith(".avi")))
        print("extracted", source.name, len(rows), flush=True)
    (SOURCE / "extraction_receipt.json").write_text(
        json.dumps(
            {"archive_sha256": actual, "classes": sorted(CLASSES), "clips": rows},
            indent=2,
        )
        + "\n"
    )
    print("HMDB EXTRACTION COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
