"""Fetch new datasets from the project's Google Drive folder, skipping what exists.

`anomaly-eval fetch-drive` stops on the first Google Doc whose export link is
not public (gdown resolves every Doc while listing), and Drive throttles
gdown's per-file page requests after a few hundred files. This script lists
the folder with gdown but records such Docs instead of stopping, then
downloads from Drive's direct download host with curl.

* Files already on disk (at their Drive path under data/raw/datasets) are skipped.
* `--exclude` globs leave folders out (default: the "Reject" folders, and
  the Roboflow littering clips, which are the existing `littering/Littering`
  images rebuilt as video).
* Videos are re-encoded to at most 640 px wide at their original frame rate,
  so label frame numbers still apply, and the original is deleted. Each video
  is recorded in data/raw/datasets/drive_<date>_videos.json with its original
  size, frame counts before and after, and the box scale factor.

    python scripts/datasets/fetch_drive_videos.py --list-only
    python scripts/datasets/fetch_drive_videos.py --workers 4 --min-free-gb 1.5

Needs gdown 6.4.1 (the project's .venv), curl, ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/datasets"
VIDEO = (".avi", ".mp4", ".mkv")
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
DEFAULT_EXCLUDE = ["*Reject*", "littering/Roboflow littering clips/videos/*"]
ORDER = ["unattended items", "rough sleeping", "smoking", "loitering", "littering"]
stop = threading.Event()


def listing(skipped):
    """[{id, path}] for every file gdown can list; unexportable Docs go to `skipped`."""
    import gdown
    from gdown.download_folder import GoogleDriveFileToDownload

    folder = sys.modules[
        "gdown.download_folder"
    ]  # gdown re-exports the function by this name
    real = folder.download

    def tolerant(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        except gdown.exceptions.FileURLRetrievalError:
            file_id = kwargs["url"].rsplit("=", 1)[-1]
            skipped.append(file_id)
            return GoogleDriveFileToDownload(id=file_id, path="", local_path="")

    folder.download = tolerant
    try:
        from anomaly_eval.datasets.drive import DRIVE_URL

        entries = gdown.download_folder(
            url=DRIVE_URL,
            output="/nonexistent/",
            skip_download=True,
            quiet=True,
            use_cookies=False,
        )
    finally:
        folder.download = real
    # gdown joins a skipped Doc's empty name onto its folder: "folder/".
    return [
        {"id": e.id, "path": e.path}
        for e in entries
        if e.path and not e.path.endswith("/")
    ]


def small_path(path):
    parts = list(Path(path).parts)
    parts[parts.index("videos")] = "videos_640"
    return Path(*parts).with_suffix(".mp4")


def probe(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_packets,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {
        "codec": s["codec_name"],
        "width": s["width"],
        "height": s["height"],
        "fps": num / den,
        "frames": int(s["nb_read_packets"]),
    }


def download(file_id, target):
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        if stop.is_set():
            raise RuntimeError("stopped")
        r = subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--fail",
                "-C",
                "-",
                "-o",
                str(part),
                URL.format(file_id),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and part.exists() and part.stat().st_size > 0:
            head = part.open("rb").read(64).lstrip()
            if head.startswith((b"<!DOCTYPE", b"<html")):
                part.unlink()
                raise RuntimeError("Drive returned an HTML page (quota or permission)")
            part.rename(target)
            return target.stat().st_size
        time.sleep(min(5 * 2**attempt, 120))
    raise RuntimeError(f"curl failed: {r.stderr.strip()[:200]}")


def fetch(entry, min_free):
    rel = entry["path"]
    original = RAW / rel
    if not rel.endswith(VIDEO) or "/videos/" not in f"/{rel}":
        if original.is_file():
            return "skip", rel, None
        download(entry["id"], original)
        return "got", rel, None
    small = RAW / small_path(rel)
    if small.is_file() or original.is_file():
        return "skip", rel, None
    free = shutil.disk_usage(RAW).free / 1e9
    if free < min_free:
        stop.set()
        raise RuntimeError(f"free disk {free:.1f} GB below {min_free} GB")
    tmp = small.with_name(small.stem + ".orig" + Path(rel).suffix)
    small.parent.mkdir(parents=True, exist_ok=True)
    size = download(entry["id"], tmp)
    before = probe(tmp)
    out = small.with_name(small.name + ".part.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(tmp),
            "-map",
            "0:v:0",
            "-vf",
            "scale='min(640,iw)':-2",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    after = probe(out)
    out.rename(small)
    tmp.unlink()
    return (
        "got",
        rel,
        {
            "id": entry["id"],
            "drive_path": rel,
            "local_path": str(small.relative_to(RAW)),
            "original_bytes": size,
            "bytes": small.stat().st_size,
            "original": before,
            "encoded": after,
            "scale": after["width"] / before["width"],
            # A few ICVL files end in a packet with no picture: off by 1-3 frames.
            "frames_match": before["frames"] == after["frames"],
        },
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="globs on the Drive path"
    )
    ap.add_argument(
        "--list-only", action="store_true", help="print what is new; download nothing"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=RAW / f"drive_{date.today():%Y%m%d}_videos.json",
    )
    a = ap.parse_args()

    skipped = []
    entries = listing(skipped)
    print(
        f"Drive lists {len(entries)} files; {len(skipped)} Google Docs could not be exported "
        f"(open them in a browser): {skipped}",
        flush=True,
    )
    entries = [
        e
        for e in entries
        if not any(fnmatch.fnmatchcase(e["path"], g) for g in a.exclude)
    ]
    new = [
        e
        for e in entries
        if not (RAW / e["path"]).is_file()
        and not (
            e["path"].endswith(VIDEO)
            and "/videos/" in f"/{e['path']}"
            and (RAW / small_path(e["path"])).is_file()
        )
    ]
    by_folder = {}
    for e in new:
        key = "/".join(e["path"].split("/")[:2])
        by_folder[key] = by_folder.get(key, 0) + 1
    print(f"{len(new)} of {len(entries)} files are not on disk:", flush=True)
    for k, v in sorted(by_folder.items()):
        print(f"  {v:6d}  {k}")
    if a.list_only or not new:
        return 0
    rank = lambda e: next(
        (i for i, p in enumerate(ORDER) if e["path"].startswith(p)), len(ORDER)
    )
    new.sort(key=lambda e: (e["path"].endswith(VIDEO), rank(e), e["path"]))
    records = json.loads(a.manifest.read_text()) if a.manifest.exists() else []
    counts, lock = {"got": 0, "skip": 0, "fail": 0}, threading.Lock()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(fetch, e, a.min_free_gb): e for e in new}
        for n, fut in enumerate(as_completed(futures), 1):
            e = futures[fut]
            try:
                status, rel, record = fut.result()
                counts[status] += 1
                if record:
                    with lock:
                        records.append(record)
                        a.manifest.write_text(json.dumps(records, indent=1))
                    print(
                        f"[{n}/{len(new)}] {rel}: {record['original_bytes'] / 1e6:.0f} MB -> "
                        f"{record['bytes'] / 1e6:.1f} MB",
                        flush=True,
                    )
                elif status == "got":
                    print(f"[{n}/{len(new)}] {rel}", flush=True)
            except (
                OSError,
                RuntimeError,
                subprocess.CalledProcessError,
                KeyError,
                ValueError,
            ) as exc:
                counts["fail"] += 1
                print(
                    f"[{n}/{len(new)}] FAIL {e['path']}: {str(exc)[:200]}", flush=True
                )
    print(
        "SUMMARY",
        counts,
        f"free {shutil.disk_usage(RAW).free / 1e9:.1f} GB",
        flush=True,
    )
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
