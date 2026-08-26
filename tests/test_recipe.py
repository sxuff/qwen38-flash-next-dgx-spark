#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249"
LLAMA_COMMIT = "bea3b12daee45876b0129a3602dc8f534ce30bf0"

manifest = json.loads((ROOT / "manifests/q1-iq1s.json").read_text())
assert manifest["revision"] == MODEL_REVISION
assert len(manifest["files"]) == 3
assert sum(item["bytes"] for item in manifest["files"]) == manifest["total_bytes"] == 72546461344
assert all(len(item["sha256"]) == 64 for item in manifest["files"])

results = json.loads((ROOT / "results/q1-iq1s.json").read_text())
assert results["model"]["revision"] == MODEL_REVISION
assert results["runtime"]["commit"] == LLAMA_COMMIT
assert results["sweep"]["status"] == "passed"
assert results["sweep"]["batches"] == 32
assert results["sweep"]["request_rows"] == 120
assert results["sweep"]["cold_warm_pairs"] == 60
assert results["safety"]["safety_incident"] is False
assert results["safety"]["maximum_service_swap_bytes"] == 0

readme = (ROOT / "README.md").read_text()
build = (ROOT / "scripts/build_llama.sh").read_text()
server = (ROOT / "scripts/run_server.sh").read_text()
service = (ROOT / "systemd/qwen38-flash-next-llama.service").read_text()
all_public_text = "\n".join(
    path.read_text(errors="replace")
    for path in ROOT.rglob("*")
    if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
)

for required in (MODEL_REVISION, LLAMA_COMMIT, "121a-real", "UD-IQ1_S"):
    assert required in readme or required in build
for required_flag in ("--no-kv-unified", "-ub 64", "-ngl 99", "--no-context-shift", "--host 127.0.0.1"):
    assert required_flag in server
assert "MemorySwapMax=0" in service
assert "MemoryMax=110G" in service
for forbidden in (
    "/home/" + "sxuf",
    "gx10" + "-fe09",
    "benchmark" + "-contract",
    "super" + "seded",
):
    assert forbidden not in all_public_text

print("recipe tests passed")
