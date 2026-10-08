"""Offline contract tests. HTTP payloads are small, synthetic fixtures, not weights."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("download_model", ROOT / "scripts/download_model.py")
assert SPEC is not None and SPEC.loader is not None
dm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dm)


def entry(payload=b"fixture-data", path="nested/weight.safetensors", algorithm="sha256"):
    raw = payload if algorithm == "sha256" else b"blob " + str(len(payload)).encode() + b"\0" + payload
    digest = hashlib.new("sha256" if algorithm == "sha256" else "sha1", raw).hexdigest()
    result = {"path": path, "bytes": len(payload), "algorithm": algorithm, "digest": digest}
    if algorithm == "sha256":
        result["sha256"] = digest
    return result


def manifest(items=None):
    items = items if items is not None else [entry()]
    return {"schema_version": 1, "repo": "fixture/model", "revision": "a" * 40,
            "reserve_bytes": 20 * 1024 ** 3, "total_bytes": sum(x["bytes"] for x in items), "files": items}


class FixtureHTTP:
    def __init__(self, payload=b"fixture-data", mode="normal"):
        self.payload, self.mode, self.requests = payload, mode, []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                offset = 0
                requested_range = self.headers.get("Range")
                owner.requests.append(requested_range)
                if owner.mode == "error":
                    self.send_error(503)
                    return
                if requested_range:
                    offset = int(requested_range.removeprefix("bytes=").removesuffix("-"))
                status = 206 if offset else 200
                if owner.mode == "ignore-range":
                    status, offset = 200, 0
                body = owner.payload[offset:]
                if owner.mode == "overlong":
                    body += b"extra"
                self.send_response(status)
                if status == 206:
                    start = offset + (1 if owner.mode == "bad-range" else 0)
                    self.send_header("Content-Range", f"bytes {start}-{len(owner.payload)-1}/{len(owner.payload)}")
                if owner.mode != "overlong":
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if owner.mode == "short":
                    body = body[:-2]
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/fixture"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-download-", dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "destination"
        self.item = entry()
        self.record = manifest()
        self.disk = mock.patch.object(dm.shutil, "disk_usage", return_value=SimpleNamespace(free=10**15))
        self.disk.start()
        self.addCleanup(self.disk.stop)

    def run_download(self, server, record=None, **kwargs):
        with mock.patch.object(dm, "build_url", return_value=server.url), contextlib.redirect_stdout(io.StringIO()):
            return dm.run(record or self.record, self.destination, **kwargs)

    def partial(self, payload):
        path = self.destination / (self.item["path"] + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_real_http_fresh_download_and_verify_only(self):
        with FixtureHTTP() as server:
            self.run_download(server)
            self.assertEqual(server.requests, [None])
        final = self.destination / self.item["path"]
        self.assertEqual(final.read_bytes(), b"fixture-data")
        self.assertFalse(final.with_name(final.name + ".partial").exists())
        before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in self.destination.rglob("*")}
        with mock.patch.object(dm.urllib.request, "urlopen", side_effect=AssertionError("unexpected HTTP")):
            with contextlib.redirect_stdout(io.StringIO()):
                dm.run(self.record, self.destination, verify_only=True)
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in self.destination.rglob("*")})

    def test_real_http_resume_and_git_blob(self):
        partial = self.partial(b"fixt")
        with FixtureHTTP() as server:
            self.run_download(server)
            self.assertEqual(server.requests, ["bytes=4-"])
        self.assertFalse(partial.exists())
        item = entry(b"git fixture", "config.json", "git-blob-sha1")
        with FixtureHTTP(b"git fixture") as server:
            self.run_download(server, manifest([item]))
        self.assertEqual((self.destination / "config.json").read_bytes(), b"git fixture")

    def test_multi_chunk_disk_guard_preserves_already_written_prefix(self):
        dm.shutil.disk_usage.side_effect = [SimpleNamespace(free=10**15), SimpleNamespace(free=10**15), SimpleNamespace(free=self.record["reserve_bytes"])]
        with FixtureHTTP() as server, mock.patch.object(dm, "CHUNK_BYTES", 4):
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
        partial = self.destination / (self.item["path"] + ".partial")
        self.assertEqual(partial.read_bytes(), b"fixt")
        self.assertFalse((self.destination / self.item["path"]).exists())

    def test_verified_final_skipped_and_empty_files_supported(self):
        final = self.destination / self.item["path"]
        final.parent.mkdir(parents=True)
        final.write_bytes(b"fixture-data")
        old_mtime = final.stat().st_mtime_ns
        empty = entry(b"", "empty.json", "git-blob-sha1")
        with FixtureHTTP() as server:
            self.run_download(server, manifest([self.item, empty]))
            self.assertEqual(server.requests, [])
        self.assertEqual((self.destination / "empty.json").read_bytes(), b"")
        self.assertEqual(final.stat().st_mtime_ns, old_mtime)

    def test_publish_cannot_overwrite_final_created_after_preflight(self):
        final = self.destination / self.item["path"]
        real_link = dm.os.link

        def racing_link(*args, **kwargs):
            final.write_bytes(b"concurrent-file")
            return real_link(*args, **kwargs)

        with FixtureHTTP() as server, mock.patch.object(dm.os, "link", side_effect=racing_link):
            with self.assertRaises((dm.DownloadError, OSError)):
                self.run_download(server)
        self.assertEqual(final.read_bytes(), b"concurrent-file")
        self.assertEqual(final.with_name(final.name + ".partial").read_bytes(), b"fixture-data")

    def test_verify_only_never_opens_for_write_or_changes_filesystem(self):
        final = self.destination / self.item["path"]
        final.parent.mkdir(parents=True)
        final.write_bytes(b"fixture-data")
        real_open = dm.os.open

        def read_only_open(path, flags, *args, **kwargs):
            self.assertEqual(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND), 0)
            return real_open(path, flags, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(dm.os, "open", side_effect=read_only_open))
            for method in ("mkdir", "link", "unlink", "fsync"):
                stack.enter_context(mock.patch.object(dm.os, method, side_effect=AssertionError("unexpected filesystem mutation")))
            stack.enter_context(mock.patch.object(dm.urllib.request, "urlopen", side_effect=AssertionError("unexpected HTTP")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            dm.run(self.record, self.destination, verify_only=True)

    def test_range_ignored_or_bad_preserves_partial(self):
        for mode in ("ignore-range", "bad-range"):
            with self.subTest(mode=mode):
                partial = self.partial(b"fixt")
                with FixtureHTTP(mode=mode) as server:
                    with self.assertRaises(dm.DownloadError):
                        self.run_download(server)
                self.assertEqual(partial.read_bytes(), b"fixt")
                self.assertFalse((self.destination / self.item["path"]).exists())

    def test_short_http_response_is_resumable(self):
        with FixtureHTTP(mode="short") as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
        partial = self.destination / (self.item["path"] + ".partial")
        self.assertEqual(partial.read_bytes(), b"fixture-da")
        with FixtureHTTP() as server:
            self.run_download(server)
            self.assertEqual(server.requests, ["bytes=10-"])

    def test_overlong_http_response_never_publishes(self):
        with FixtureHTTP(mode="overlong") as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
        self.assertFalse((self.destination / self.item["path"]).exists())
        partial = self.destination / (self.item["path"] + ".partial")
        self.assertLessEqual(partial.stat().st_size, self.item["bytes"])

    def test_http_failure_preserves_partial(self):
        partial = self.partial(b"fixt")
        with FixtureHTTP(mode="error") as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
        self.assertEqual(partial.read_bytes(), b"fixt")

    def test_wrong_digest_preserves_full_partial_and_refuses_retry(self):
        with FixtureHTTP(b"corrupt-data") as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
        partial = self.destination / (self.item["path"] + ".partial")
        self.assertEqual(partial.read_bytes(), b"corrupt-data")
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
            self.assertEqual(server.requests, [])
        self.assertEqual(partial.read_bytes(), b"corrupt-data")

    def test_valid_full_partial_promotes_without_http(self):
        partial = self.partial(b"fixture-data")
        with FixtureHTTP() as server:
            self.run_download(server)
            self.assertEqual(server.requests, [])
        self.assertFalse(partial.exists())
        self.assertEqual((self.destination / self.item["path"]).read_bytes(), b"fixture-data")

    def test_existing_invalid_final_never_clobbered(self):
        final = self.destination / self.item["path"]
        final.parent.mkdir(parents=True)
        final.write_bytes(b"unrelated")
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
            self.assertEqual(server.requests, [])
        self.assertEqual(final.read_bytes(), b"unrelated")

    def test_oversize_partial_never_clobbered(self):
        partial = self.partial(b"too-large-for-fixture")
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
            self.assertEqual(server.requests, [])
        self.assertEqual(partial.read_bytes(), b"too-large-for-fixture")

    def test_preflight_checks_all_files_before_downloading(self):
        second = entry(b"second", "config.json")
        self.destination.mkdir()
        (self.destination / "config.json").write_bytes(b"invalid")
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server, manifest([self.item, second]))
            self.assertEqual(server.requests, [])
        self.assertFalse((self.destination / "nested").exists())

    def test_verify_only_missing_and_dry_run_make_no_destination(self):
        with self.assertRaises(dm.DownloadError):
            dm.run(self.record, self.destination, verify_only=True)
        self.assertFalse(self.destination.exists())
        with mock.patch.object(dm.urllib.request, "urlopen", side_effect=AssertionError("unexpected HTTP")):
            with contextlib.redirect_stdout(io.StringIO()):
                dm.run(self.record, self.destination, dry_run=True)
        self.assertFalse(self.destination.exists())

    def test_disk_preflight_and_ongoing_chunk_guard(self):
        dm.shutil.disk_usage.return_value = SimpleNamespace(free=self.record["reserve_bytes"])
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
            self.assertEqual(server.requests, [])
        self.assertFalse(self.destination.exists())
        dm.shutil.disk_usage.side_effect = [SimpleNamespace(free=10**15), SimpleNamespace(free=self.record["reserve_bytes"])]
        with FixtureHTTP() as server:
            with self.assertRaises(dm.DownloadError):
                self.run_download(server)
            self.assertEqual(server.requests, [None])
        partial = self.destination / (self.item["path"] + ".partial")
        self.assertEqual(partial.read_bytes(), b"")

    def test_disk_budget_counts_only_missing_bytes(self):
        self.partial(b"fixt")
        dm.shutil.disk_usage.return_value = SimpleNamespace(free=self.record["reserve_bytes"] + 8)
        with FixtureHTTP() as server:
            self.run_download(server)
        self.assertEqual((self.destination / self.item["path"]).read_bytes(), b"fixture-data")

    def test_symlink_destination_parent_final_partial_and_broken_link_rejected(self):
        target = self.root / "outside"
        target.mkdir()
        for location in (self.destination, self.destination / "nested", self.destination / self.item["path"], self.destination / (self.item["path"] + ".partial")):
            with self.subTest(location=location.name):
                location.parent.mkdir(parents=True, exist_ok=True)
                location.symlink_to(target)
                with self.assertRaises(dm.DownloadError):
                    dm.run(self.record, self.destination, dry_run=True)
                location.unlink()
        broken = self.destination / self.item["path"]
        broken.symlink_to(target / "missing")
        with self.assertRaises(dm.DownloadError):
            dm.run(self.record, self.destination, verify_only=True)
        self.assertTrue(broken.is_symlink())

    def test_hardlinked_partial_rejected(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"fixt")
        partial = self.destination / (self.item["path"] + ".partial")
        partial.parent.mkdir(parents=True)
        os.link(outside, partial)
        with self.assertRaises(dm.DownloadError):
            dm.run(self.record, self.destination)
        self.assertEqual(outside.read_bytes(), b"fixt")

    def test_cli_required_destination_default_alias_and_verify_failure(self):
        script = str(ROOT / "scripts/download_model.py")
        proc = subprocess.run([sys.executable, "-B", script, "--dry-run"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        proc = subprocess.run([sys.executable, "-B", script, "--destination", str(self.destination), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["repo"], "turboderp/Qwen3.8-Flash-Next-exl3")
        self.assertFalse(self.destination.exists())
        proc = subprocess.run([sys.executable, "-B", script, "--destination", str(self.destination), "--manifest", "vision-bf16", "--verify-only"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.destination.exists())


class ManifestTests(unittest.TestCase):
    def test_public_manifests_are_exact_pinned_inventory(self):
        weights = dm.load_manifest(ROOT / "manifests/exl3-405.json")
        tower = dm.load_manifest(ROOT / "manifests/vision-bf16.json")
        self.assertEqual(weights["repo"], "turboderp/Qwen3.8-Flash-Next-exl3")
        self.assertEqual(weights["revision"], "55a732e0c4c3d4614bc42b68493bb930d9b02c0a")
        self.assertEqual(weights["branch"], "4.05bpw_h6_ng6")
        self.assertEqual(len(weights["files"]), 27)
        self.assertEqual(weights["total_bytes"], 107463600896)
        self.assertEqual(sum(x["algorithm"] == "sha256" for x in weights["files"]), 14)
        self.assertEqual(sum(x["algorithm"] == "git-blob-sha1" for x in weights["files"]), 13)
        self.assertEqual(tower["repo"], "doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw")
        self.assertEqual(tower["revision"], "b5a234522364b10cdc4869844597bf77eeb942a7")
        self.assertEqual(tower["files"], [dict(path="vision_tower_bf16.safetensors", bytes=897899504, algorithm="sha256", digest="cdd69998e52e34badced49ef8ec09824b8b9a52e488094f0cadeadc7ebfc641e", sha256="cdd69998e52e34badced49ef8ec09824b8b9a52e488094f0cadeadc7ebfc641e")])
        for record in (weights, tower):
            self.assertEqual(record["reserve_bytes"], 20 * 1024 ** 3)
            self.assertEqual(record["total_bytes"], sum(x["bytes"] for x in record["files"]))
            self.assertNotIn("/opt/", json.dumps(record))

    def test_malformed_revisions_repos_paths_sizes_and_hashes(self):
        mutations = []
        for value in ("main", "a" * 39, "A" * 40, "a" * 40 + "/file"):
            mutations.append(("revision", value))
        for value in ("../model", "user/model/extra", "https://example.invalid/model", "user/model?x", "user/..", "user/a..b"):
            mutations.append(("repo", value))
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                bad = manifest()
                bad[key] = value
                with self.assertRaises(dm.DownloadError):
                    dm.validate_manifest(bad)
        for path in ("../weight", "/weight", "a/../weight", "a//weight", "a/./weight", "a\\weight", "a/%2e%2e/weight", "a?x", "a#x", "a/", "C:weight", "a\nweight"):
            with self.subTest(path=path):
                with self.assertRaises(dm.DownloadError):
                    dm.validate_manifest(manifest([entry(path=path)]))
        for key, value in (("bytes", True), ("bytes", -1), ("bytes", 1.5), ("digest", "g" * 64), ("algorithm", "sha1"), ("sha256", "b" * 64)):
            with self.subTest(key=key):
                bad = manifest()
                bad["files"][0][key] = value
                with self.assertRaises(dm.DownloadError):
                    dm.validate_manifest(bad)
        for key, value in (("reserve_bytes", 0), ("reserve_bytes", True), ("total_bytes", 0), ("schema_version", True)):
            bad = manifest()
            bad[key] = value
            with self.assertRaises(dm.DownloadError):
                dm.validate_manifest(bad)

    def test_duplicate_names_casefold_ancestors_and_partial_collisions(self):
        for paths in (("same", "same"), ("File", "file"), ("a", "a/b"), ("a", "a.partial"), ("a.partial/x", "a")):
            with self.subTest(paths=paths):
                with self.assertRaises(dm.DownloadError):
                    dm.validate_manifest(manifest([entry(path=x) for x in paths]))

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"repo":"fixture/model","repo":"other/model"}')
            with self.assertRaises(dm.DownloadError):
                dm.load_manifest(path)

    def test_build_url_uses_immutable_revision(self):
        record = manifest()
        url = dm.build_url(record, entry(path="nested/config.json"))
        self.assertEqual(url, "https://huggingface.co/fixture/model/resolve/" + "a" * 40 + "/nested/config.json?download=true")


if __name__ == "__main__":
    unittest.main()
