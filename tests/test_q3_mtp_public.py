#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/q3-q3kxl-mtp4.json"


class Q3Mtp4PublicTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads(RESULT_PATH.read_text())

    def test_result_identity_and_decision(self):
        self.assertEqual(self.result["status"], "completed-split-run-operating-sweep")
        decision = self.result["decision"]
        self.assertEqual(decision["promoted_profile"], "q3-q3kxl-mtp4")
        self.assertEqual(decision["batch_size"], 2048)
        self.assertEqual(decision["micro_batch_size"], 512)
        self.assertEqual(decision["parallel_slots"], 1)
        self.assertEqual(decision["context_tokens"], 65536)
        self.assertEqual(decision["rejected_profile"], "q3-q3kxl-ngram-mod-np2")

    def test_prefill_batch_ab(self):
        ab = self.result["prefill_batch_ab"]
        self.assertEqual(ab["profile"], "MTP-3")
        self.assertEqual(ab["prompt_tokens"], 32000)
        self.assertEqual(ab["completion_tokens"], 256)
        self.assertAlmostEqual(ab["control"]["ttft_seconds"], 82.43501904699951)
        self.assertAlmostEqual(ab["candidate"]["ttft_seconds"], 46.803955713054165)
        self.assertAlmostEqual(ab["control"]["prefill_tokens_per_second"], 388.2562670931339)
        self.assertAlmostEqual(ab["candidate"]["prefill_tokens_per_second"], 683.9123427840473)
        self.assertAlmostEqual(ab["candidate_delta"]["ttft_reduction_fraction"], 0.43223212)
        self.assertAlmostEqual(ab["candidate_delta"]["prefill_increase_fraction"], 0.76149724)
        self.assertTrue(ab["output_sha256_match"])
        self.assertTrue(ab["cold_cache_verified"])
        self.assertEqual(ab["control"]["maximum_service_swap_bytes"], 0)
        self.assertEqual(ab["candidate"]["maximum_service_swap_bytes"], 0)

    def test_mtp4_promotion(self):
        comparison = self.result["mtp3_vs_mtp4"]
        suite = comparison["four_task_suite"]
        long32 = comparison["long_32k"]
        self.assertAlmostEqual(suite["mtp3_weighted_decode_tokens_per_second"], 58.620155163460936)
        self.assertAlmostEqual(suite["mtp4_weighted_decode_tokens_per_second"], 63.10577878694844)
        self.assertAlmostEqual(suite["weighted_decode_increase_fraction"], 0.07652015950792768)
        self.assertAlmostEqual(suite["wall_reduction_fraction"], 0.06049613333937143)
        self.assertAlmostEqual(long32["decode_increase_fraction"], 0.1002405742963497)
        self.assertEqual(comparison["exact_output_matches"], "5/5")
        self.assertTrue(comparison["validators_passed"])
        self.assertTrue(comparison["promotion_gate_passed"])

    def test_cache_boundary_and_rejected_concurrency_are_qualified(self):
        cache = self.result["warm_32k_mtp3"]
        boundary = self.result["context_boundary_mtp3"]
        rejected = self.result["rejected_two_slot_ngram_mod"]
        self.assertEqual(cache["exact_prefix_repeat"]["cached_tokens"], 31996)
        self.assertEqual(cache["exact_prefix_repeat"]["evaluated_prompt_tokens"], 4)
        self.assertTrue(cache["output_sha256_match"])
        self.assertIn("not rerun with MTP-4", boundary["qualification"])
        self.assertEqual(boundary["prompt_tokens"], 65000)
        self.assertEqual(boundary["completion_tokens"], 256)
        self.assertEqual(boundary["headroom_tokens"], 280)
        self.assertFalse(rejected["promoted"])
        self.assertLess(rejected["aggregate_change_fraction"], 0)
        self.assertEqual(rejected["exact_output_matches"], "4/4")
        safety = self.result["safety"]
        self.assertEqual(safety["mtp3_minimum_mem_available_bytes"], 50507980800)
        self.assertEqual(safety["mtp3_maximum_host_swap_growth_bytes"], 315711488)
        self.assertEqual(safety["mtp4_minimum_mem_available_bytes"], 51397378048)
        self.assertEqual(safety["mtp4_maximum_host_swap_growth_bytes"], 72650752)
        self.assertEqual(safety["np2_minimum_mem_available_bytes"], 54021926912)
        self.assertEqual(safety["np2_maximum_host_swap_growth_bytes"], 241737728)
        self.assertEqual(safety["maximum_service_swap_bytes_in_every_valid_stage"], 0)
        split = rejected["split_run_qualification"]
        self.assertTrue(split["parent_startup_nullified_before_primary_request"])
        self.assertEqual(split["parent_primary_rows_for_treatment"], 0)
        self.assertEqual(split["amendment_rows"], 4)
        self.assertEqual(rejected["safety"]["minimum_mem_available_bytes"], 54021926912)
        self.assertEqual(rejected["safety"]["maximum_host_swap_growth_bytes"], 241737728)
        source = self.result["prefill_batch_ab"]["source_receipt"]
        self.assertEqual(source["run_id"], "q3-prefill-test2-corrected-onepass")
        self.assertEqual(source["scientific_status"], "decision")
        self.assertTrue(source["restoration_verified"])
        rows = self.result["sanitized_rows"]
        self.assertEqual(len(rows["mtp3_and_mtp4_parent"]), 12)
        self.assertEqual(len(rows["two_slot_missing_cell_amendment"]), 4)
        for row in rows["mtp3_and_mtp4_parent"] + rows["two_slot_missing_cell_amendment"]:
            self.assertTrue(row["task_validation"]["passed"])
            self.assertFalse(row["output_included"])
            self.assertFalse(row["request_bytes_included"])
            self.assertNotIn("output", row)
            self.assertNotIn("request_bytes", row)

    def test_public_recipe_and_obsolete_artifacts(self):
        readme = (ROOT / "README.md").read_text()
        server = (ROOT / "scripts/run_server.sh").read_text()
        installer = (ROOT / "scripts/install_service.sh").read_text()
        build = (ROOT / "scripts/build_llama_mtp.sh").read_text()
        card = ROOT / "results/q3-q3kxl-mtp4-card.png"
        for value in (
            "q3-q3kxl-mtp4",
            "-b 2048",
            "-ub 512",
            "58.62 to 63.11 tok/s",
            "results/q3-q3kxl-mtp4-card.png",
            "not a claim that every workload decodes at 63.11 tok/s",
            "2,741 of 3,440 completion tokens",
            "deterministic repeated-token microbenchmark",
            "Long-form creative prose was not part of the sealed suite",
            "MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark",
            "Weschera/Qwen3.8-Flash-Next-1x-DGX-Spark",
            "not rerun under MTP-4",
            "sealed missing-cell amendment",
            "50.31 GiB",
        ):
            self.assertIn(value, readme)
        self.assertTrue(card.is_file())
        self.assertEqual(card.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for value in ("mtp-2", "mtp-3", "mtp-4", "draft-mtp", "MTP_DRAFT_PATH", "DRY_RUN"):
            self.assertIn(value, server)
        self.assertNotIn("mtp-5", server)
        for value in ("q3-q3kxl-mtp3", "q3-q3kxl-mtp4", "build-gb10-mtp", "FIT_MODE", "DRY_RUN"):
            self.assertIn(value, installer)
        self.assertIn("d1a92352cbd417fd840b4e765c0b82f5fe3d1d89", build)
        for obsolete in (
            "q3-q3kxl-mtp.json",
            "q3-q3kxl-mtp-posthoc.md",
            "q3-q3kxl-mtp-publication.json",
        ):
            self.assertFalse((ROOT / "results" / obsolete).exists())

    def test_installer_rejects_unverified_supplied_artifact_paths(self):
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            fixture_root = temp / "recipe"
            shutil.copytree(ROOT / "scripts", fixture_root / "scripts")
            shutil.copytree(ROOT / "manifests", fixture_root / "manifests")

            def make_zero_byte_manifest(name):
                path = fixture_root / "manifests" / name
                manifest = json.loads(path.read_text())
                for entry in manifest["files"]:
                    entry["bytes"] = 0
                    entry["sha256"] = empty_sha256
                manifest["total_bytes"] = 0
                manifest["reserve_bytes"] = 0
                path.write_text(json.dumps(manifest))
                return manifest

            target_manifest = make_zero_byte_manifest("q3-q3kxl.json")
            draft_manifest = make_zero_byte_manifest("q3-mtp-shared-q8.json")
            projector_manifest = make_zero_byte_manifest("q3-mmproj-f16.json")

            target_root = temp / "target"
            draft_root = temp / "draft"
            llama_root = temp / "llama"
            for entry in target_manifest["files"]:
                path = target_root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
            canonical_draft = draft_root / draft_manifest["files"][0]["path"]
            canonical_draft.parent.mkdir(parents=True, exist_ok=True)
            canonical_draft.write_bytes(b"")
            canonical_projector = target_root / projector_manifest["files"][0]["path"]
            canonical_projector.parent.mkdir(parents=True, exist_ok=True)
            canonical_projector.write_bytes(b"")
            unverified_draft = canonical_draft.parent / "UNVERIFIED.gguf"
            unverified_projector = canonical_projector.parent / "UNVERIFIED-mmproj.gguf"
            unverified_draft.write_bytes(b"not manifest verified")
            unverified_projector.write_bytes(b"not manifest verified")
            binary = llama_root / "build-gb10-mtp/bin/llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("#!/usr/bin/env bash\nexit 0\n")
            binary.chmod(0o755)

            base_command = [
                "bash",
                str(fixture_root / "scripts/install_service.sh"),
                str(target_root),
                str(llama_root),
                "q3-q3kxl-mtp4",
            ]
            env = {**os.environ, "DRY_RUN": "1"}
            canonical = subprocess.run(
                base_command + [str(canonical_draft), str(canonical_projector)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(canonical.returncode, 0, canonical.stderr)
            self.assertIn("BATCH_SIZE=2048", canonical.stdout)
            self.assertIn("UBATCH_SIZE=512", canonical.stdout)
            self.assertIn("SPEC_MODE=mtp-4", canonical.stdout)

            bad_draft = subprocess.run(
                base_command + [str(unverified_draft), str(canonical_projector)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_draft.returncode, 0)
            self.assertIn("not the manifest-verified artifact", bad_draft.stderr)

            bad_projector = subprocess.run(
                base_command + [str(canonical_draft), str(unverified_projector)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_projector.returncode, 0)
            self.assertIn("not the manifest-verified artifact", bad_projector.stderr)

            missing_projector = subprocess.run(
                base_command + [str(canonical_draft), str(target_root / "MISSING-mmproj.gguf")],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(missing_projector.returncode, 0)
            self.assertIn("supplied MMPROJ does not exist", missing_projector.stderr)

    def test_no_private_host_identifiers(self):
        paths = [
            ROOT / "README.md",
            RESULT_PATH,
            ROOT / "scripts/build_llama_mtp.sh",
            ROOT / "scripts/install_service.sh",
            ROOT / "scripts/run_server.sh",
        ]
        text = "\n".join(path.read_text(errors="replace") for path in paths)
        for forbidden in ("/home/" + "sxuf", "gx10" + "-fe09", "172.17." + "0.1"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
