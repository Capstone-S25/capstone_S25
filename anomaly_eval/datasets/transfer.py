"""Bounded, resumable dataset transfers with atomic publication and receipts."""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def public_url(url):
    """Keep version selectors but omit credentials and signed query parameters."""
    parts = urlsplit(url)
    public = {"datasetVersionNumber", "download", "raw", "revision"}
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k in public])
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, query, ""))


class TransferError(RuntimeError):
    def __init__(self, status, message, *, retryable=False, delay=0):
        super().__init__(message)
        self.status, self.retryable, self.delay = status, retryable, delay


@dataclass(frozen=True)
class Asset:
    source: str
    path: str
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    size: int | None = None
    kind: str = "http"
    member: str | None = None
    archive_size: int | None = None
    file_id: str | None = None
    attribution: dict = field(default_factory=dict, compare=False)

    @property
    def key(self):
        return hashlib.sha256((self.source + ":" + self.path).encode()).hexdigest()[:24]

    @property
    def fingerprint(self):
        identity = asdict(self)
        identity.pop("attribution")
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass
class Options:
    workers: int = 8
    per_host: int = 3
    retries: int = 3
    connect_timeout: float = 15
    read_timeout: float = 60
    min_free_bytes: int = 256 * 1024**2
    verify: bool = False
    cached_only: bool = False
    max_retry_delay: float = 30
    max_download_bytes: int | None = None


class Transfers:
    def __init__(self, root, options=None, cancel=None):
        self.root = Path(root).resolve()
        self.options = options or Options()
        self.cancel = cancel or threading.Event()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.hosts, self.cooldowns = {}, {}
        self.bytes_received = 0
        self.remote_archives = {}
        self.sessions = []
        self.reserved_bytes = 0

    def close(self):
        for session in self.sessions:
            session.close()

    @property
    def session(self):
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "CapstoneDatasetFetcher/2.0",
                    "Accept-Encoding": "identity",
                }
            )
            adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=4)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        return self.local.session

    def target(self, asset):
        path = (self.root / asset.path).resolve()
        if (
            Path(asset.path).is_absolute()
            or "\\" in asset.path
            or ".." in Path(asset.path).parts
            or not Path(asset.path).parts
            or Path(asset.path).parts[0] == ".fetch_state"
            or not path.is_relative_to(self.root)
        ):
            raise TransferError("invalid", "Destination escapes the data root")
        return path

    def state_paths(self, asset):
        """Keep JSON metadata outside dataset scanners; partial bytes stay atomic."""
        target = self.target(asset)
        identity = hashlib.sha256(
            str(target.relative_to(self.root)).encode()
        ).hexdigest()
        state = self.root / ".fetch_state" / identity[:2] / identity[2:]
        if not state.resolve().is_relative_to(self.root):
            raise TransferError("invalid", "Transfer state escapes the data root")
        return {
            "receipt": state / "receipt.json",
            "metadata": state / "partial.json",
            "lock": state / "lock",
            "partial": target.with_name("." + target.name + ".fetch.part"),
        }

    def check_cancel(self):
        if self.cancel.is_set():
            raise TransferError(
                "cancelled", "Transfer cancelled; partial data retained"
            )

    def _acquire_host(self, url):
        self.check_cancel()
        host = urlsplit(url).hostname
        with self.lock:
            semaphore = self.hosts.setdefault(
                host, threading.BoundedSemaphore(self.options.per_host)
            )
            deferred = self.cooldowns.get(host, 0) - time.time()
        if deferred > 0:
            raise TransferError(
                "deferred", "Host retry window is still active", delay=deferred
            )
        while not semaphore.acquire(timeout=0.1):
            self.check_cancel()
        with self.lock:
            deferred = self.cooldowns.get(host, 0) - time.time()
        if deferred > 0:
            semaphore.release()
            raise TransferError(
                "deferred", "Host retry window is still active", delay=deferred
            )
        if self.cancel.is_set():
            semaphore.release()
            self.check_cancel()
        return semaphore

    def request(self, url, **kwargs):
        semaphore = self._acquire_host(url)
        host = urlsplit(url).hostname
        try:
            response = self.session.get(
                url,
                timeout=(self.options.connect_timeout, self.options.read_timeout),
                **kwargs,
            )
        except BaseException:
            semaphore.release()
            raise
        original_close = response.close

        def close():
            if not getattr(response, "_fetch_closed", False):
                response._fetch_closed = True
                try:
                    original_close()
                finally:
                    semaphore.release()

        response.close = close
        code = response.status_code
        if code >= 400 and code != 416:
            retry_after = response.headers.get("Retry-After")
            delay = 0
            if retry_after:
                try:
                    delay = max(0, float(retry_after))
                except ValueError:
                    try:
                        delay = max(
                            0,
                            email.utils.parsedate_to_datetime(retry_after).timestamp()
                            - time.time(),
                        )
                    except (ValueError, TypeError, OverflowError):
                        delay = 0
            response.close()
            retryable = code in {408, 425, 429} or 500 <= code < 600
            status = (
                "needs_credentials"
                if code in {401, 403}
                else "not_found"
                if code == 404
                else "failed"
            )
            if delay > self.options.max_retry_delay:
                with self.lock:
                    self.cooldowns[host] = time.time() + delay
                raise TransferError(
                    "deferred",
                    f"HTTP {code}: host requested a later retry",
                    delay=delay,
                )
            raise TransferError(
                status, f"HTTP {code}", retryable=retryable, delay=delay
            )
        return response

    def _cached(self, asset, target, receipt_path):
        if not target.is_file():
            return None
        previous = {}
        if receipt_path.is_file():
            try:
                previous = json.loads(receipt_path.read_text())
            except (ValueError, OSError):
                pass
        stat = target.stat()
        if asset.size is not None and stat.st_size != asset.size:
            return None
        expected = asset.sha256 or previous.get("sha256")
        if not expected:
            return None
        if not asset.sha256 and previous.get("fingerprint") != asset.fingerprint:
            return None
        unchanged = (
            previous.get("fingerprint") == asset.fingerprint
            and previous.get("sha256") == expected
            and previous.get("bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
        )
        if (self.options.verify or not unchanged) and digest(target) != expected:
            return None
        value = {
            "source": asset.source,
            "path": asset.path,
            "status": "cached",
            "sha256": expected,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "fingerprint": asset.fingerprint,
            "verification": "sha256"
            if self.options.verify or not unchanged
            else "unchanged_receipt_stat",
        }
        if not self.options.cached_only and (not unchanged or self.options.verify):
            atomic_json(receipt_path, value)
        return value

    def _publish(self, asset, partial, target, receipt_path, *, sha=None, url=None):
        size = partial.stat().st_size
        if asset.size is not None and size != asset.size:
            raise TransferError(
                "integrity_error", "Source size mismatch", retryable=True
            )
        sha = sha or digest(partial)
        if asset.sha256 and sha != asset.sha256:
            raise TransferError(
                "integrity_error", "Source SHA-256 mismatch", retryable=True
            )
        partial.replace(target)
        result = {
            "source": asset.source,
            "path": asset.path,
            "status": "downloaded",
            "sha256": sha,
            "bytes": size,
            "mtime_ns": target.stat().st_mtime_ns,
            "fingerprint": asset.fingerprint,
            "verification": "sha256" if asset.sha256 else "transport_and_local_sha256",
        }
        if url:
            result["url"] = public_url(url)
        atomic_json(receipt_path, result)
        receipt_path.with_name("partial.json").unlink(missing_ok=True)
        return result

    def _http(self, asset, url, partial, target, receipt_path):
        metadata_path = receipt_path.with_name("partial.json")
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except (ValueError, OSError):
                pass
        url_identity = hashlib.sha256(url.encode()).hexdigest()
        same = metadata.get("fingerprint") == asset.fingerprint and (
            asset.sha256 or metadata.get("url_identity") == url_identity
        )
        offset = partial.stat().st_size if partial.exists() and same else 0
        if offset and asset.size == offset and asset.sha256:
            return self._publish(asset, partial, target, receipt_path, url=url)
        validator = metadata.get("etag") or metadata.get("last_modified")
        resume = bool(offset and (asset.sha256 or validator))
        headers = {"Range": f"bytes={offset}-"} if resume else {}
        if resume and validator:
            headers["If-Range"] = validator
        with self.request(url, stream=True, headers=headers) as response:
            if response.status_code == 416:
                raise TransferError(
                    "integrity_error",
                    "Server rejected saved partial range",
                    retryable=True,
                )
            if response.status_code == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", ""),
                )
                if (
                    not resume
                    or not match
                    or int(match[1]) != offset
                    or int(match[2]) < offset
                ):
                    raise TransferError(
                        "integrity_error",
                        "Invalid partial Content-Range",
                        retryable=True,
                    )
                total = int(match[3])
                if asset.size is not None and total != asset.size:
                    raise TransferError(
                        "integrity_error",
                        "Range total differs from pinned size",
                        retryable=True,
                    )
                if (
                    not asset.sha256
                    and validator
                    and validator
                    != (
                        response.headers.get("ETag")
                        or response.headers.get("Last-Modified")
                    )
                ):
                    raise TransferError(
                        "integrity_error",
                        "Partial response validator changed",
                        retryable=True,
                    )
            else:
                offset = 0
                total = int(response.headers.get("Content-Length", 0)) or None
                if (
                    total is not None
                    and asset.size is not None
                    and not response.headers.get("Content-Encoding")
                    and total != asset.size
                ):
                    raise TransferError(
                        "integrity_error",
                        "Response length differs from pinned source size",
                        retryable=True,
                    )
            if "text/html" in response.headers.get(
                "Content-Type", ""
            ).lower() and target.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
                raise TransferError(
                    "failed", "Received HTML instead of the requested dataset file"
                )
            etag = response.headers.get("ETag", "")
            if etag.startswith("W/"):
                etag = ""
            atomic_json(
                metadata_path,
                {
                    "fingerprint": asset.fingerprint,
                    "etag": etag,
                    "last_modified": response.headers.get("Last-Modified"),
                    "total": total,
                    "url_identity": url_identity,
                },
            )
            hasher = hashlib.sha256()
            if offset:
                with partial.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024**2), b""):
                        hasher.update(block)
            with partial.open("ab" if offset else "wb") as stream:
                for block in response.iter_content(1024**2):
                    self.check_cancel()
                    if block:
                        stream.write(block)
                        hasher.update(block)
                        with self.lock:
                            self.bytes_received += len(block)
                        if asset.size is not None and stream.tell() > asset.size:
                            raise TransferError(
                                "integrity_error",
                                "Response exceeds pinned source size",
                                retryable=True,
                            )
            if (
                total is not None
                and not response.headers.get("Content-Encoding")
                and partial.stat().st_size != total
            ):
                raise TransferError("failed", "Incomplete response", retryable=True)
        return self._publish(
            asset, partial, target, receipt_path, sha=hasher.hexdigest(), url=url
        )

    def one(self, asset):
        target = self.target(asset)
        state = self.state_paths(asset)
        receipt_path = state["receipt"]
        self.check_cancel()
        cached = self._cached(asset, target, receipt_path)
        if cached:
            return cached
        if self.options.cached_only:
            return {
                "source": asset.source,
                "path": asset.path,
                "status": "missing_or_invalid",
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        state["lock"].parent.mkdir(parents=True, exist_ok=True)
        # flock coordinates independent invocations without stale lock ownership.
        import fcntl

        with state["lock"].open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.cancel.wait(0.1):
                        self.check_cancel()
            cached = self._cached(asset, target, receipt_path)
            if cached:
                return cached
            return self._transfer(asset, target, receipt_path)

    def _transfer(self, asset, target, receipt_path):
        if self.options.max_download_bytes is not None:
            with self.lock:
                if (
                    asset.size is None
                    or self.reserved_bytes + asset.size
                    > self.options.max_download_bytes
                ):
                    raise TransferError(
                        "deferred_budget", "File exceeds the remaining payload budget"
                    )
                self.reserved_bytes += asset.size
        partial = self.state_paths(asset)["partial"]
        needed = max(
            0, (asset.size or 0) - (partial.stat().st_size if partial.exists() else 0)
        )
        if shutil.disk_usage(target.parent).free < needed + self.options.min_free_bytes:
            raise TransferError(
                "disk_full", "Insufficient free space for this file plus reserved space"
            )
        last_error = None
        for url in asset.urls or ("",):
            for attempt in range(self.options.retries + 1):
                self.check_cancel()
                try:
                    if asset.kind == "http":
                        return self._http(asset, url, partial, target, receipt_path)
                    if asset.kind == "gdrive":
                        import gdown

                        semaphore = self._acquire_host("https://drive.google.com")
                        try:
                            result = gdown.download(
                                id=asset.file_id,
                                output=str(partial),
                                resume=True,
                                use_cookies=False,
                                quiet=True,
                                cancel=self.cancel,
                                timeout=(
                                    self.options.connect_timeout,
                                    self.options.read_timeout,
                                ),
                            )
                        except gdown.exceptions.FileURLRetrievalError:
                            raise TransferError(
                                "needs_credentials_or_quota",
                                "Drive file is inaccessible or quota limited",
                            ) from None
                        except gdown.exceptions.DownloadCancelled:
                            raise TransferError(
                                "cancelled",
                                "Drive transfer cancelled; partial data retained",
                            ) from None
                        finally:
                            semaphore.release()
                        if not result:
                            raise TransferError("failed", "Drive returned no file")
                        return self._publish(
                            asset, partial, target, receipt_path, url=url
                        )
                    if asset.kind == "zip_member":
                        from .remote_zip import fetch_member

                        fetch_member(self, asset, url, partial)
                        return self._publish(
                            asset, partial, target, receipt_path, url=url
                        )
                    raise TransferError("invalid", "Unknown asset transfer kind")
                except TransferError as exc:
                    last_error = exc
                    if exc.status == "integrity_error":
                        partial.unlink(missing_ok=True)
                        receipt_path.with_name("partial.json").unlink(missing_ok=True)
                    if not exc.retryable or attempt == self.options.retries:
                        break
                    if self.cancel.wait(
                        max(
                            exc.delay,
                            min(
                                self.options.max_retry_delay,
                                2**attempt + random.random(),
                            ),
                        )
                    ):
                        self.check_cancel()
                except (requests.RequestException, OSError) as exc:
                    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
                        raise TransferError(
                            "disk_full", "Storage filled; partial retained"
                        ) from None
                    last_error = TransferError(
                        "failed",
                        type(exc).__name__ + ": transport failed",
                        retryable=True,
                    )
                    if attempt == self.options.retries:
                        break
                    if self.cancel.wait(
                        min(self.options.max_retry_delay, 2**attempt + random.random())
                    ):
                        self.check_cancel()
        raise last_error or TransferError("failed", "No usable source URL")
