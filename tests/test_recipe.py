#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249"
VALIDATED_LLAMA_COMMIT = "bea3b12daee45876b0129a3602dc8f534ce30bf0"
LLAMA_COMMIT = "b8bdf73bb9baf044caadd33be2a51be70156ec57"

manifest = json.loads((ROOT / "manifests/q1-iq1s.json").read_text())
assert manifest["revision"] == MODEL_REVISION
assert len(manifest["files"]) == 3
assert sum(item["bytes"] for item in manifest["files"]) == manifest["total_bytes"] == 72546461344
assert all(len(item["sha256"]) == 64 for item in manifest["files"])

q3_revision = "8bdc666649440e9bdc97e16f3f75782c98478ff5"
q3_manifest = json.loads((ROOT / "manifests/q3-q3kxl.json").read_text())
assert q3_manifest["revision"] == q3_revision
assert q3_manifest["quantization"] == "UD-Q3_K_XL"
assert len(q3_manifest["files"]) == 3
assert sum(item["bytes"] for item in q3_manifest["files"]) == q3_manifest["total_bytes"] == 89986353824
assert all(len(item["sha256"]) == 64 for item in q3_manifest["files"])

results = json.loads((ROOT / "results/q1-iq1s.json").read_text())
assert results["model"]["revision"] == MODEL_REVISION
assert results["runtime"]["commit"] == VALIDATED_LLAMA_COMMIT
assert results["sweep"]["status"] == "passed"
assert results["sweep"]["batches"] == 32
assert results["sweep"]["request_rows"] == 120
assert results["sweep"]["cold_warm_pairs"] == 60
assert results["safety"]["safety_incident"] is False
assert results["safety"]["maximum_service_swap_bytes"] == 0

q3_results = json.loads((ROOT / "results/q3-q3kxl.json").read_text())
assert q3_results["model"]["revision"] == q3_revision
assert q3_results["runtime"]["commit"] == VALIDATED_LLAMA_COMMIT
assert q3_results["validation"]["status"] == "passed"
assert q3_results["validation"]["request_rows"] == 10
assert q3_results["paired_quality"]["passed"] == 4
assert q3_results["paired_quality"]["exact_q1_output_matches"] == 4
assert q3_results["safety"]["safety_incident"] is False
assert q3_results["safety"]["maximum_service_swap_bytes"] == 0

ngram_results = json.loads((ROOT / "results/q3-q3kxl-ngram-mod.json").read_text())
assert ngram_results["runtime"]["commit"] == LLAMA_COMMIT
assert ngram_results["design"]["paired_cases"] == 4
assert ngram_results["design"]["repetitions"] == 1
assert ngram_results["aggregate"]["completion_tokens_per_arm"] == 3496
assert ngram_results["aggregate"]["exact_output_matches"] == 4
assert ngram_results["aggregate"]["accepted_tokens"] == 2992
assert ngram_results["readiness_and_safety"]["post_deployment_service_swap_bytes"] == 0

readme = (ROOT / "README.md").read_text()
build = (ROOT / "scripts/build_llama.sh").read_text()
server = (ROOT / "scripts/run_server.sh").read_text()
installer = (ROOT / "scripts/install_service.sh").read_text()
service = (ROOT / "systemd/qwen38-flash-next-llama.service").read_text()
all_public_text = "\n".join(
    path.read_text(errors="replace")
    for path in ROOT.rglob("*")
    if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
)

for required in (MODEL_REVISION, q3_revision, VALIDATED_LLAMA_COMMIT, LLAMA_COMMIT, "SM121", "UD-IQ1_S", "UD-Q3_K_XL"):
    assert required in readme or required in build
for required_flag in ("--no-kv-unified", "-ngl 99", "--no-context-shift", "--host", "127.0.0.1", "--spec-type", "ngram-mod"):
    assert required_flag in server
for required_profile_value in ("ctx_size=65536", "parallel=1", "batch_size=512", "ngram_mod=1"):
    assert required_profile_value in installer
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
