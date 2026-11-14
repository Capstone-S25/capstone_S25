"""Build a small update for a machine that already has an earlier bundle.

    python tools/build_update.py dist/capstone-cpu-deployment-20261113 \
        dist/capstone-cpu-deployment-20261114 dist/capstone-update-20261113-to-20261114

Compares the two bundles' DEPLOYMENT_MANIFEST.json files. The update carries
only the files whose content is new; a file the old bundle already has under
another path (the image encoders, which each classifier folder repeats) is
copied on the target instead of being transferred again. `apply_update.py`
in the update (Python 3.12 standard library only) applies it to the old
bundle's folder and verifies every file of the new one. Files only the old
bundle has are left in place, so the earlier classifier stays selectable.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

APPLY = r'''"""Apply this update to an installed bundle (Python 3.12 standard library).

    python3.12 apply_update.py /path/to/capstone-cpu-deployment-folder

Stop the application first. Nothing is downloaded. Every new file is checked
before anything is changed, files are replaced one at a time through a
temporary file, and the whole updated bundle is verified at the end. Your
.env, the Python environment, recordings and the database are not touched.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if len(sys.argv) != 2:
        raise ValueError("Give the folder of the installed bundle (it contains deploy.py)")
    target = Path(sys.argv[1]).expanduser().resolve()
    plan = json.loads((HERE / "update.json").read_text())
    manifest_path = target / "DEPLOYMENT_MANIFEST.json"
    if not (target / "deploy.py").is_file() or not manifest_path.is_file():
        raise ValueError(f"{target} is not a deployment bundle folder")
    installed = json.loads(manifest_path.read_text())
    if installed["source_commit"] == plan["to_commit"]:
        print("This bundle already has the update; verifying it again.")
    elif installed["source_commit"] != plan["from_commit"]:
        raise ValueError(
            f"This update is for {plan['from_package']} (commit "
            f"{plan['from_commit'][:7]}); the folder holds {installed['package']} "
            f"(commit {installed['source_commit'][:7]})"
        )
    # Check everything before changing anything.
    for item in plan["files"]:
        source = (
            HERE / "files" / item["path"]
            if item["action"] == "payload"
            else target / item["source"]
        )
        current = target / item["path"]
        if current.is_file() and current.stat().st_size == item["bytes"]:
            if sha256(current) == item["sha256"]:
                item["action"] = "present"
                continue
        if not source.is_file() or sha256(source) != item["sha256"]:
            raise ValueError(
                "Missing or damaged "
                + ("update file: " if item["action"] == "payload" else "installed file: ")
                + str(source)
            )
        item["from"] = source
    written = 0
    for item in plan["files"]:
        if item["action"] == "present":
            continue
        destination = target / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".update-part")
        shutil.copyfile(item["from"], temporary)
        os.chmod(temporary, item["mode"])
        os.replace(temporary, destination)
        written += 1
    # The updated folder must now be exactly the new bundle.
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = target / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing file or wrong size after update: " + item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Checksum mismatch after update: " + item["path"])
    print(
        f"Updated {plan['from_package']} to {plan['to_package']}: {written} files "
        f"written, {len(manifest['files'])} files verified."
    )
    for line in plan["notes"]:
        print("  " + line)
    print("Next, in that folder: python3.12 deploy.py check --camera --dashboard --smoke")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print("Update failed, nothing further was changed: " + str(exc), file=sys.stderr)
        sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--zip", action="store_true", help="Also write output.zip")
    args = parser.parse_args()
    old = json.loads((args.old / "DEPLOYMENT_MANIFEST.json").read_text())
    new = json.loads((args.new / "DEPLOYMENT_MANIFEST.json").read_text())
    old_by_path = {f["path"]: f for f in old["files"]}
    old_by_hash = {}
    for f in old["files"]:
        old_by_hash.setdefault((f["sha256"], f["bytes"]), f["path"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files, payload_bytes, copied_bytes = [], 0, 0
    for f in new["files"]:
        before = old_by_path.get(f["path"])
        if before and before["sha256"] == f["sha256"]:
            continue
        source = args.new / f["path"]
        entry = {**f, "mode": source.stat().st_mode & 0o777}
        elsewhere = old_by_hash.get((f["sha256"], f["bytes"]))
        if elsewhere:
            entry |= {"action": "copy", "source": elsewhere}
            copied_bytes += f["bytes"]
        else:
            entry["action"] = "payload"
            target = out / "files" / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            payload_bytes += f["bytes"]
        files.append(entry)
    # The manifest does not list itself; it goes last, once the files it
    # describes are in place.
    name = "DEPLOYMENT_MANIFEST.json"
    shutil.copyfile(args.new / name, out / "files" / name)
    files.append(
        {
            "path": name,
            "bytes": (args.new / name).stat().st_size,
            "sha256": hashlib.sha256((args.new / name).read_bytes()).hexdigest(),
            "mode": 0o644,
            "action": "payload",
        }
    )
    kept = sorted(set(old_by_path) - {f["path"] for f in new["files"]})
    plan = {
        "from_package": old["package"],
        "from_commit": old["source_commit"],
        "to_package": new["package"],
        "to_commit": new["source_commit"],
        "files": files,
        "left_in_place": kept,
        "notes": args.note,
    }
    (out / "update.json").write_text(json.dumps(plan, indent=1) + "\n")
    (out / "apply_update.py").write_text(APPLY)
    (out / "README.txt").write_text(
        f"Update {old['package']} -> {new['package']}\n\n"
        "1. Stop the application (Ctrl+C in its terminal).\n"
        "2. Copy this whole folder to the machine and run, from inside it:\n\n"
        "       python3.12 apply_update.py /path/to/the/deployment/folder\n\n"
        "3. In the deployment folder: python3.12 deploy.py check --camera --dashboard --smoke\n"
        "4. Start it as before.\n\n" + "\n".join(args.note) + "\n"
    )
    count = sum(f["action"] == "payload" for f in files)
    print(
        f"{out}: {count} files to transfer ({payload_bytes / 1024**2:.1f} MiB), "
        f"{len(files) - count} copied on the target ({copied_bytes / 1024**3:.2f} GiB), "
        f"{len(kept)} old files left in place"
    )
    if args.zip:
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(p for p in out.rglob("*") if p.is_file()):
                bundle.write(path, Path(out.name) / path.relative_to(out))
        print(f"{archive}: {archive.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
