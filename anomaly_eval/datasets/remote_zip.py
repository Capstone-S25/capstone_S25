"""Fetch selected ZIP members with validated byte ranges, never a full fallback."""

from __future__ import annotations

import io
import stat
import struct
import threading
import zipfile
import zlib

from .transfer import TransferError


def byte_range(client, url, start, end, size):
    if not 0 <= start <= end < size:
        raise TransferError("invalid", "ZIP byte range lies outside the archive")
    with client.request(
        url, headers={"Range": f"bytes={start}-{end}"}, stream=True
    ) as response:
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
        ):
            raise TransferError(
                "failed",
                "Archive host does not honor exact byte ranges; full download refused",
            )
        blocks = []
        count = 0
        for block in response.iter_content(1024**2):
            client.check_cancel()
            count += len(block)
            if count > end - start + 1:
                raise TransferError(
                    "integrity_error", "Archive range exceeds requested bytes"
                )
            blocks.append(block)
            with client.lock:
                client.bytes_received += len(block)
        if count != end - start + 1:
            raise TransferError("failed", "Incomplete ZIP range", retryable=True)
    return b"".join(blocks)


class RangeFile(io.RawIOBase):
    def __init__(self, client, url, size):
        self.client, self.url, self.size = client, url, size
        self.position = 0
        self.block_start, self.block = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if self.position < 0:
            raise ValueError("Negative ZIP seek")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        if end - self.position > 64 * 1024**2:
            raise TransferError(
                "invalid", "ZIP directory read exceeds the 64 MiB metadata bound"
            )
        if self.block_start <= self.position and end <= self.block_start + len(
            self.block
        ):
            value = self.block[
                self.position - self.block_start : end - self.block_start
            ]
        else:
            # Central-directory reads get a modest read-ahead, never a 49 GB body.
            stop = min(self.size, max(end, self.position + 256 * 1024))
            self.block_start = self.position
            self.block = byte_range(
                self.client, self.url, self.position, stop - 1, self.size
            )
            value = self.block[: end - self.position]
        self.position = end
        return value


def fetch_member(client, asset, url, partial):
    if not asset.member or not asset.archive_size:
        raise TransferError(
            "invalid", "ZIP member needs a name and pinned archive size"
        )
    key = (url, asset.archive_size)
    with client.lock:
        entry = client.remote_archives.setdefault(
            key, {"lock": threading.Lock(), "members": None}
        )
    with entry["lock"]:
        if entry["members"] is None:
            with zipfile.ZipFile(RangeFile(client, url, asset.archive_size)) as archive:
                entry["members"] = {info.filename: info for info in archive.infolist()}
    info = entry["members"].get(asset.member)
    if info is None:
        raise TransferError("not_found", "Pinned ZIP member is absent")
    if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
        raise TransferError("invalid", "ZIP member is not an unencrypted regular file")
    if asset.size is not None and info.file_size != asset.size:
        raise TransferError(
            "integrity_error", "ZIP member size differs from the pinned manifest"
        )
    if info.file_size > 512 * 1024**2 or info.compress_size > 512 * 1024**2:
        raise TransferError(
            "invalid", "Selected ZIP member exceeds the per-member memory bound"
        )
    header = byte_range(
        client, url, info.header_offset, info.header_offset + 29, asset.archive_size
    )
    if header[:4] != b"PK\x03\x04":
        raise TransferError("integrity_error", "Invalid ZIP local header")
    namesize, extra = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + 30 + namesize + extra
    data = (
        byte_range(
            client, url, start, start + info.compress_size - 1, asset.archive_size
        )
        if info.compress_size
        else b""
    )
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        value = decoder.decompress(data, info.file_size + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise TransferError(
                "integrity_error",
                "ZIP member decompression exceeded its declared bounds",
            )
    elif info.compress_type == zipfile.ZIP_STORED:
        value = data
    else:
        raise TransferError("invalid", "Unsupported selected ZIP compression method")
    if len(value) != info.file_size or zlib.crc32(value) != info.CRC:
        raise TransferError(
            "integrity_error", "ZIP member checksum mismatch", retryable=True
        )
    partial.write_bytes(value)
