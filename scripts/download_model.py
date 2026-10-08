#!/usr/bin/env python3
"""Stdlib-only, immutable Hugging Face downloads for TensorFold EXL3 4.05.

Files use SHA-256 for LFS payloads, or Git blob SHA-1 (including the blob
header) for ordinary Git files. No credentials, mutable branch resolution,
automatic restarts, or overwrites of existing finals/full invalid partials.
"""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "exl3-405"
DEFAULT_MANIFEST_PATH = ROOT / "manifests" / "exl3-405.json"
MANIFESTS = {name: ROOT / "manifests" / f"{name}.json" for name in ("exl3-405", "vision-bf16")}
CHUNK_BYTES = 8 * 1024 * 1024
MIN_RESERVE_BYTES = 20 * 1024 ** 3


class DownloadError(RuntimeError):
    """A failed contract. Existing files are preserved for inspection."""


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def validate_manifest(record):
    """Validate every identifier before any filesystem changes or HTTP."""
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise DownloadError("manifest must have schema_version=1")
    repo = record.get("repo")
    if not isinstance(repo, str) or not _matches(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo) or ".." in repo:
        raise DownloadError("invalid Hugging Face owner/repository identifier")
    if not _matches(r"[0-9a-f]{40}", record.get("revision")):
        raise DownloadError("revision must be an immutable lowercase 40-character commit SHA")
    if not _integer(record.get("reserve_bytes"), MIN_RESERVE_BYTES):
        raise DownloadError("reserve_bytes must be at least 20 GiB")
    if not _integer(record.get("total_bytes")):
        raise DownloadError("invalid total_bytes")
    if "branch" in record and not _matches(r"[A-Za-z0-9_.-]+", record["branch"]):
        raise DownloadError("invalid informational branch name")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise DownloadError("files must be a nonempty list")
    occupied = set()
    for item in files:
        if not isinstance(item, dict):
            raise DownloadError("file entries must be objects")
        name = item.get("path")
        if not isinstance(name, str) or not name or any(
            part in ("", ".", "..") or not _matches(r"[A-Za-z0-9_.-]+", part)
            for part in name.split("/")
        ):
            raise DownloadError("invalid relative file path")
        if not _integer(item.get("bytes")):
            raise DownloadError(f"invalid byte count: {name}")
        algorithm = item.get("algorithm")
        if algorithm not in ("sha256", "git-blob-sha1"):
            raise DownloadError(f"unsupported digest algorithm: {name}")
        if not _matches(r"[0-9a-f]{64}" if algorithm == "sha256" else r"[0-9a-f]{40}", item.get("digest")):
            raise DownloadError(f"invalid digest: {name}")
        if "sha256" in item and (algorithm != "sha256" or item["sha256"] != item["digest"]):
            raise DownloadError(f"sha256 field disagrees with algorithm/digest: {name}")
        # Reserve both final and partial namespaces; case-folding also protects
        # case-insensitive filesystems. File/directory ancestor collisions fail.
        for candidate in (name.casefold(), (name + ".partial").casefold()):
            if any(candidate == other or candidate.startswith(other + "/") or other.startswith(candidate + "/") for other in occupied):
                raise DownloadError(f"duplicate or overlapping file/partial path: {name}")
            occupied.add(candidate)
    if sum(item["bytes"] for item in files) != record["total_bytes"]:
        raise DownloadError("manifest total_bytes disagrees with files")
    return record


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DownloadError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path):
    path = MANIFESTS.get(str(path), Path(path).expanduser())
    try:
        return validate_manifest(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"cannot read manifest: {exc}") from exc


def absolute_path(path):
    path = Path(path).expanduser()
    if ".." in path.parts:
        raise DownloadError("destination must not contain '..' components")
    return path.absolute()  # Do not resolve away symlinks.


@contextmanager
def directory_fd(path, create=False):
    """Walk directories without following links, including destination ancestors.

    Dirfd-relative leaf operations remain anchored if a directory is renamed.
    This is intentionally POSIX, matching the recipe's Linux platform.
    """
    path = absolute_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                next_fd = os.open(part, flags, dir_fd=fd)
            except OSError as exc:
                if isinstance(exc, FileNotFoundError):
                    raise
                raise DownloadError("unsafe or non-directory destination component") from exc
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _regular(info, name):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise DownloadError(f"refusing symlink, non-regular or multiply-linked file: {name}")
    return info


def inspect_file(path):
    try:
        with directory_fd(path.parent) as fd:
            return _regular(os.stat(path.name, dir_fd=fd, follow_symlinks=False), path.name)
    except FileNotFoundError:
        return None


def verify(path, item):
    """Read only; never create a missing parent or write a verification receipt."""
    path = absolute_path(path)
    info = inspect_file(path)
    if info is None:
        return False, "missing"
    if info.st_size != item["bytes"]:
        return False, f"size={info.st_size} expected={item['bytes']}"
    digest = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-blob-sha1":
        digest.update(f"blob {item['bytes']}\0".encode("ascii"))
    with directory_fd(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            before = _regular(os.fstat(handle.fileno()), path.name)
            if (before.st_dev, before.st_ino, before.st_size) != (info.st_dev, info.st_ino, item["bytes"]):
                raise DownloadError(f"file changed before verification: {path.name}")
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise DownloadError(f"file changed during verification: {path.name}")
    actual = digest.hexdigest()
    return (actual == item["digest"], actual if actual == item["digest"] else f"{item['algorithm']} mismatch")


def build_url(record, item):
    quoted = urllib.parse.quote(item["path"], safe="/")
    return f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{quoted}?download=true"


def check_disk(path, required):
    """For preflight, inspect the nearest existing destination ancestor."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < required:
        raise DownloadError(f"insufficient disk: free={free} required={required} (includes reserve)")


def download(url, partial, item, reserve_bytes):
    info = inspect_file(partial)
    offset = info.st_size if info else 0
    if offset >= item["bytes"]:
        raise DownloadError(f"refusing to overwrite full/oversized partial: {partial.name}")
    headers = {"User-Agent": "tensorfold-exl3-downloader/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.status
            if status not in (200, 206) or (offset and status != 206):
                raise DownloadError("server did not honor Range; partial preserved")
            if status == 206:
                expected = f"bytes {offset}-{item['bytes'] - 1}/{item['bytes']}"
                if response.headers.get("Content-Range") != expected:
                    raise DownloadError("invalid Content-Range; partial preserved")
            elif response.headers.get("Content-Range"):
                raise DownloadError("unexpected Content-Range")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise DownloadError("encoded HTTP payload cannot be verified/resumed safely")
            length = response.headers.get("Content-Length")
            if length is not None and (not re.fullmatch(r"[0-9]+", length) or int(length) != item["bytes"] - offset):
                raise DownloadError("Content-Length disagrees with manifest")
            with directory_fd(partial.parent, create=True) as parent:
                flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
                if info is None:
                    flags |= os.O_CREAT | os.O_EXCL
                fd = os.open(partial.name, flags, 0o644, dir_fd=parent)
                with os.fdopen(fd, "ab", buffering=0) as handle:
                    actual = _regular(os.fstat(handle.fileno()), partial.name)
                    if actual.st_size != offset or (info and (actual.st_dev, actual.st_ino) != (info.st_dev, info.st_ino)):
                        raise DownloadError("partial changed before resume")
                    while True:
                        chunk = response.read(min(CHUNK_BYTES, item["bytes"] - offset + 1))
                        if not chunk:
                            break
                        if offset + len(chunk) > item["bytes"]:
                            raise DownloadError("HTTP payload exceeds manifest; partial preserved")
                        check_disk(partial.parent, reserve_bytes + len(chunk))
                        # Handle a rare short filesystem write without discarding bytes.
                        view = memoryview(chunk)
                        while view:
                            written = handle.write(view)
                            if not written:
                                raise DownloadError("short filesystem write")
                            view = view[written:]
                        offset += len(chunk)
                    os.fsync(handle.fileno())
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        raise DownloadError(f"download failed; partial preserved: {exc}") from exc
    if offset != item["bytes"]:
        raise DownloadError(f"incomplete HTTP payload: {offset} expected={item['bytes']}; resume by rerunning")


def publish(partial, final, item):
    ok, detail = verify(partial, item)
    if not ok:
        raise DownloadError(f"partial verification failed: {item['path']}: {detail}; preserved")
    with directory_fd(final.parent) as parent:
        # link() is atomic and fails if final already exists, unlike replace().
        before = _regular(os.stat(partial.name, dir_fd=parent, follow_symlinks=False), partial.name)
        os.link(partial.name, final.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        linked = os.stat(final.name, dir_fd=parent, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino, linked.st_size) != (before.st_dev, before.st_ino, item["bytes"]):
            raise DownloadError("partial changed during publication; files preserved")
        os.unlink(partial.name, dir_fd=parent)
        os.fsync(parent)
    ok, detail = verify(final, item)
    if not ok:
        raise DownloadError(f"post-publication verification failed: {item['path']}: {detail}")


def run(record, destination, *, verify_only=False, dry_run=False):
    record = validate_manifest(record)
    destination = absolute_path(destination)
    try:
        with directory_fd(destination):
            pass
    except FileNotFoundError:
        pass
    # Check all manifest and partial paths before dry-run or any download.
    for item in record["files"]:
        final = destination / item["path"]
        inspect_file(final)
        inspect_file(final.with_name(final.name + ".partial"))
    if dry_run:
        print(json.dumps({"destination": str(destination), **record}, indent=2))
        return 0
    pending = []
    remaining = 0
    for item in record["files"]:
        final = destination / item["path"]
        ok, detail = verify(final, item)
        if ok:
            continue
        if verify_only:
            raise DownloadError(f"verification failed: {item['path']}: {detail}")
        if inspect_file(final) is not None:
            raise DownloadError(f"existing invalid final: {item['path']}: {detail}; refusing overwrite")
        partial = final.with_name(final.name + ".partial")
        info = inspect_file(partial)
        offset = info.st_size if info else 0
        if offset > item["bytes"]:
            raise DownloadError(f"oversized partial: {item['path']}; refusing overwrite")
        if info is not None and offset == item["bytes"]:
            ok, detail = verify(partial, item)
            if not ok:
                raise DownloadError(f"full unverified partial: {item['path']}: {detail}; refusing overwrite")
        pending.append((item, final, partial, offset))
        remaining += item["bytes"] - offset
    if pending:
        check_disk(destination, remaining + record["reserve_bytes"])
    for item, final, partial, offset in pending:
        if offset < item["bytes"]:
            download(build_url(record, item), partial, item, record["reserve_bytes"])
        elif inspect_file(partial) is None:  # Empty file, with no existing partial.
            with directory_fd(partial.parent, create=True) as parent:
                fd = os.open(partial.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=parent)
                os.close(fd)
        publish(partial, final, item)
    for item in record["files"]:
        print(f"verified {item['path']} {item['bytes']} bytes {item['algorithm']}:{item['digest']}")
    print(json.dumps({"status": "verified", "repo": record["repo"], "revision": record["revision"],
                      "total_bytes": record["total_bytes"], "files": len(record["files"])}, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify pinned TensorFold EXL3 4.05 weights or BF16 vision tower")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="exl3-405 (default), vision-bf16, or a manifest JSON path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="read only; no download, directories, or receipts")
    mode.add_argument("--dry-run", action="store_true", help="validate and show manifest; no network or destination writes")
    args = parser.parse_args(argv)
    try:
        return run(load_manifest(args.manifest), args.destination, verify_only=args.verify_only, dry_run=args.dry_run)
    except (DownloadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
