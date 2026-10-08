"""Public-safe benchmark receipt, README values and supplied card identity."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TensorFoldResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ROOT / "results/tensorfold-exl3-405.json").read_text()
        cls.d = json.loads(cls.raw)
        cls.readme = (ROOT / "README.md").read_text()

    def test_supplied_card_matches_receipt_and_readme(self):
        card = self.d["result_card"]
        data = (ROOT / card["path"]).read_bytes()
        self.assertEqual(len(data), card["bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), card["sha256"])
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", data[16:24]), (card["width"], card["height"]))
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", self.readme)
        self.assertEqual(images, [card["path"]])
        self.assertFalse((ROOT / "results/gsq-iq3xxs-card.png").exists())

    def test_complete_fixed_denominator(self):
        self.assertEqual({k: v["rows"] for k, v in self.d["arms"].items()}, {"A": 89, "B": 85, "C": 85})
        self.assertEqual(sum(a["rows"] for a in self.d["arms"].values()), 259)
        for key, arm in self.d["arms"].items():
            self.assertTrue(arm["eligible"])
            self.assertEqual(len(arm["per_prompt"]), 4)
            self.assertEqual(self.d["verified_rows"][key]["actual_rows"], arm["rows"])
            self.assertEqual(self.d["verified_rows"][key]["expected_rows"], arm["rows"])
            self.assertTrue(all(v["repeats_received"] == 3 for v in arm["per_prompt"].values()))
            for dataset, expected in [("gsm8k", 50), ("humaneval", 20)]:
                self.assertEqual(arm["quality"][dataset]["scored"], expected)
                self.assertIn(f"{arm['quality'][dataset]['passed']}/{expected}", self.readme)

    def test_aggregate_and_readme_rounding(self):
        for arm in self.d["arms"].values():
            self.assertAlmostEqual(arm["decode_tps"], statistics.mean(v["decode_median_tps"] for v in arm["per_prompt"].values()), places=10)
            self.assertAlmostEqual(arm["ttft_seconds"], statistics.mean(v["ttft_median_seconds"] for v in arm["per_prompt"].values()), places=10)
            for value, fmt in [(arm["decode_tps"], ".2f"), (arm["ttft_seconds"], ".3f"), (arm["whole_host_peak_unavailable_GiB"], ".2f")]:
                self.assertIn(format(value, fmt), self.readme)
            for value in arm["prefill_request_wall_proxy_tps"].values():
                self.assertIn(format(value, ".2f"), self.readme)

    def test_comparison_boundaries(self):
        a, b = self.d["arms"]["A"], self.d["arms"]["B"]
        gain = (a["decode_tps"] / b["decode_tps"] - 1) * 100
        self.assertAlmostEqual(gain, self.d["derived"]["decode_gain_A_over_B_percent"])
        self.assertEqual(f"{gain:.1f}%", "12.0%")
        self.assertIn("12.0%", self.readme)
        self.assertTrue(self.d["C_reused_earlier_same_prompt_sweep"])
        self.assertTrue(self.d["A_B_same_block"])
        self.assertTrue(self.d["derived"]["A_B_quality_outcomes_equal"])
        self.assertEqual(self.d["benchmark_context_A_B"], 40960)
        self.assertFalse(self.d["vision_in_benchmark"])
        self.assertEqual(self.d["subsequent_serving_verification"]["context_tokens"], 262144)
        self.assertTrue(self.d["subsequent_serving_verification"]["native_image_input_verified"])
        self.assertEqual(self.d["A_drafted_vs_serial_exactness"]["passed_prompts"], 4)
        for qualifier in ["recipe-deployment comparison", "40,960-token context, text-only, one stream", "earlier same-prompts sweep", "not isolated kernel throughput", "not** a 262K or vision-enabled benchmark", "does not install TensorFold"]:
            self.assertIn(qualifier, self.readme)

    def test_readme_local_links_exist(self):
        for target in re.findall(r"\]\(([^)]+)\)", self.readme):
            if "://" not in target and not target.startswith("#"):
                self.assertTrue((ROOT / target.split("#")[0]).exists(), target)

    def test_public_sanitization(self):
        for forbidden in ["/home/", "/opt/data/", "172.17.", "tail63c80f", "RESTORE_CHECK_OK", "output_text", "reasoning_content"]:
            self.assertNotIn(forbidden, self.raw)
        for name in ["private_summary_sha256", "private_prefill_diagnostic_sha256", "private_preregistration_sha256"]:
            self.assertRegex(self.d["source_hashes"][name], r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
