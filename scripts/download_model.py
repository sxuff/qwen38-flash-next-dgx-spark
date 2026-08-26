#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "manifests" / "q1-iq1s.json"
CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int, expected_sha256: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size != expected_bytes:
        return False, f"size={size} expected={expected_bytes}"
    actual = sha256_file(path)
    if actual != expected_sha256:
        return False, f"sha256={actual} expected={expected_sha256}"
    return True, actual


def download(url: str, partial: Path, expected_bytes: int, token: str | None) -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise RuntimeError(f"partial file is larger than manifest: {partial}")
    headers = {"User-Agent": "qwen38-flash-next-dgx-spark/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch for {partial}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify a pinned GGUF profile")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if sum(item["bytes"] for item in manifest["files"]) != manifest["total_bytes"]:
        raise SystemExit("invalid manifest total")

    destination = args.destination.expanduser().resolve()
    repo = manifest["repo"]
    revision = manifest["revision"]
    token = os.environ.get("HF_TOKEN")

    if args.dry_run:
        print(json.dumps({
            "destination": str(destination),
            "repo": repo,
            "revision": revision,
            "total_bytes": manifest["total_bytes"],
            "files": manifest["files"],
        }, indent=2))
        return 0

    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)
        remaining = 0
        for item in manifest["files"]:
            final = destination / item["path"]
            partial = final.with_name(final.name + ".partial")
            final_ok, _ = verify(final, item["bytes"], item["sha256"])
            if final_ok:
                continue
            partial_bytes = partial.stat().st_size if partial.exists() else 0
            remaining += item["bytes"] - min(partial_bytes, item["bytes"])
        free = shutil.disk_usage(destination).free
        required = remaining + manifest["reserve_bytes"]
        if free < required:
            raise SystemExit(f"insufficient disk: free={free} required={required}")

    verified = []
    for item in manifest["files"]:
        final = destination / item["path"]
        ok, detail = verify(final, item["bytes"], item["sha256"])
        if not ok and args.verify_only:
            raise SystemExit(f"verification failed: {final}: {detail}")
        if not ok:
            quoted_path = urllib.parse.quote(item["path"], safe="/")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{quoted_path}?download=true"
            partial = final.with_name(final.name + ".partial")
            download(url, partial, item["bytes"], token)
            ok, detail = verify(partial, item["bytes"], item["sha256"])
            if not ok:
                raise SystemExit(f"verification failed: {partial}: {detail}")
            os.replace(partial, final)
            ok, detail = verify(final, item["bytes"], item["sha256"])
            if not ok:
                raise SystemExit(f"post-rename verification failed: {final}: {detail}")
        verified.append({"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]})
        print(f"verified {item['path']} {item['bytes']} bytes {item['sha256']}")

    print(json.dumps({
        "status": "verified",
        "repo": repo,
        "revision": revision,
        "total_bytes": sum(item["bytes"] for item in verified),
        "files": len(verified),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
