#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "ngram-mod", "mtp-2", "mtp-3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Q3MtpPublicTests(unittest.TestCase):
    def setUp(self):
        self.result_path = ROOT / "results/q3-q3kxl-mtp.json"
        self.result = json.loads(self.result_path.read_text())

    def test_result_identity_and_decision(self):
        self.assertEqual(set(self.result["arms"]), set(ARMS))
        self.assertEqual(self.result["winner_by_frozen_total_wall_rule"], "ngram-mod")
        self.assertEqual(
            self.result["operator_recommendation"],
            "keep ngram-mod for copy-heavy work and add mtp-3 as a novel-generation profile",
        )
        self.assertTrue(self.result["request_hashes_match_across_arms"])
        self.assertTrue(self.result["response_models_match_across_arms"])

    def test_every_arm_is_complete_valid_and_safe(self):
        for arm in ARMS:
            entry = self.result["arms"][arm]
            self.assertTrue(entry["eligibility"]["passed"])
            self.assertEqual(entry["aggregate"]["task_passes"], 4)
            self.assertEqual(entry["aggregate"]["exact_output_matches_vs_baseline"], 4)
            self.assertEqual(entry["aggregate"]["maximum_service_swap_bytes"], 0)
            self.assertLessEqual(entry["aggregate"]["host_swap_growth_bytes"], 512 * 1024**2)
            self.assertGreaterEqual(entry["aggregate"]["minimum_mem_available_bytes"], 6 * 1024**3)
            self.assertTrue(entry["runtime"]["vision_canary"]["passed"])
            self.assertTrue(all("content" not in row for row in entry["rows"]))

    def test_exact_measured_rates(self):
        expected = {
            "baseline": (25.021236809443785, 28.066047258657505, 139.041887757),
            "ngram-mod": (61.58648367038954, 83.95967232801179, 56.489667743),
            "mtp-2": (41.30299276273237, 50.70789400642167, 84.231184408),
            "mtp-3": (45.85941101536825, 57.8499785269992, 75.862291359),
        }
        for arm, (whole, decode, wall) in expected.items():
            aggregate = self.result["arms"][arm]["aggregate"]
            self.assertAlmostEqual(aggregate["aggregate_whole_request_completion_tokens_per_second"], whole)
            self.assertAlmostEqual(aggregate["weighted_server_decode_tokens_per_second"], decode)
            self.assertAlmostEqual(aggregate["wall_seconds"], wall)

    def test_publication_receipt_and_artifact_hash(self):
        receipt = json.loads((ROOT / "results/q3-q3kxl-mtp-publication.json").read_text())
        self.assertEqual(receipt["corrected_public_analysis_sha256"], sha256(self.result_path))
        self.assertEqual(receipt["corrected_public_analysis_bytes"], self.result_path.stat().st_size)
        self.assertTrue(receipt["installer_dry_run"]["passed"])

    def test_public_profile_and_prior_art(self):
        readme = (ROOT / "README.md").read_text()
        server = (ROOT / "scripts/run_server.sh").read_text()
        installer = (ROOT / "scripts/install_service.sh").read_text()
        build = (ROOT / "scripts/build_llama_mtp.sh").read_text()
        manifest = json.loads((ROOT / "manifests/q3-mtp-shared-q8.json").read_text())
        for value in (
            "MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark",
            "Weschera/Qwen3.8-Flash-Next-1x-DGX-Spark",
            "MTP depth 3",
            "42.8% reduction",
        ):
            self.assertIn(value, readme)
        for value in ("mtp-2", "mtp-3", "draft-mtp", "MTP_DRAFT_PATH", "DRY_RUN"):
            self.assertIn(value, server)
        for value in ("q3-q3kxl-mtp3", "build-gb10-mtp", "FIT_MODE", "DRY_RUN"):
            self.assertIn(value, installer)
        self.assertIn("d1a92352cbd417fd840b4e765c0b82f5fe3d1d89", build)
        self.assertEqual(manifest["files"][0]["sha256"], "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6")

    def test_no_private_host_identifiers(self):
        paths = [
            ROOT / "README.md",
            ROOT / "results/q3-q3kxl-mtp.json",
            ROOT / "results/q3-q3kxl-mtp-posthoc.md",
            ROOT / "results/q3-q3kxl-mtp-publication.json",
            ROOT / "scripts/build_llama_mtp.sh",
            ROOT / "scripts/install_service.sh",
            ROOT / "scripts/run_server.sh",
        ]
        text = "\n".join(path.read_text(errors="replace") for path in paths)
        for forbidden in ("/home/" + "sxuf", "gx10" + "-fe09", "172.17." + "0.1"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
