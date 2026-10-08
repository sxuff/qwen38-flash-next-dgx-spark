#!/usr/bin/env python3
"""Public contract for the retained GSQ-RCO receipt and llama.cpp recipe."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GSQPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / "results/gsq-iq3xxs.json").read_text())
        cls.manifest = json.loads((ROOT / "manifests/gsq-iq3xxs.json").read_text())
        cls.readme = (ROOT / "README.md").read_text()

    def test_replaced_card_is_absent_and_receipt_is_retained(self):
        card = self.result["result_card"]
        # Retain historical image metadata without publishing the replaced asset.
        self.assertEqual(card["width"], 1600)
        self.assertEqual(card["height"], 1150)
        self.assertFalse((ROOT / card["path"]).exists())
        self.assertNotIn(f"]({card['path']})", self.readme)
        self.assertIn("Earlier GSQ-RCO target-quant comparison", self.readme)
        self.assertIn("results/tensorfold-exl3-405-card.png", self.readme)
        self.assertFalse((ROOT / "results/q3-q3kxl-next-card.png").exists())
        self.assertNotIn("q3-q3kxl-next-card.png", self.readme)

    def test_model_manifest_and_metric_rounding(self):
        before = self.result["arms"]["UD-Q3_K_XL"]
        after = self.result["arms"]["GSQ-RCO IQ3_XXS"]
        self.assertEqual(self.manifest["total_bytes"], sum(f["bytes"] for f in self.manifest["files"]))
        self.assertEqual(self.manifest["total_bytes"], after["weight_bytes"])
        self.assertEqual(before["weight_bytes"], json.loads((ROOT / "manifests/q3-q3kxl.json").read_text())["total_bytes"])
        for field, baseline, winner in (
            ("long32_decode_tokens_per_second", "62.18", "68.40"),
            ("long65_decode_tokens_per_second", "61.66", "71.11"),
            ("six_row_wall_seconds", "204.63", "193.05"),
            ("real30_prefill_tokens_per_second", "539.42", "529.74"),
            ("wikitext2_perplexity", "4.0486", "4.0824"),
            ("source_code_perplexity", "1.3748", "1.3861"),
        ):
            fmt = ".4f" if "perplexity" in field else ".2f"
            self.assertEqual(format(before[field], fmt), baseline)
            self.assertEqual(format(after[field], fmt), winner)
            self.assertIn(baseline, self.readme)
            self.assertIn(winner, self.readme)
        changes = {
            "10.0%": (after["long32_decode_tokens_per_second"] / before["long32_decode_tokens_per_second"] - 1) * 100,
            "15.3%": (after["long65_decode_tokens_per_second"] / before["long65_decode_tokens_per_second"] - 1) * 100,
            "5.7%": (1 - after["six_row_wall_seconds"] / before["six_row_wall_seconds"]) * 100,
            "15.7%": (1 - after["weight_bytes"] / before["weight_bytes"]) * 100,
            "1.8%": (1 - after["real30_prefill_tokens_per_second"] / before["real30_prefill_tokens_per_second"]) * 100,
        }
        for label, measured in changes.items():
            self.assertEqual(f"{measured:.1f}%", label)
            self.assertIn(label, self.readme)
        self.assertLessEqual(after["wikitext2_perplexity"] / before["wikitext2_perplexity"], 1.02)
        self.assertLessEqual(after["source_code_perplexity"] / before["source_code_perplexity"], 1.02)
        self.assertEqual(after["four_task_passes"], after["four_task_total"])
        self.assertTrue(all(after["functional"].values()))

    def test_current_reproduction_recipe(self):
        installer = (ROOT / "scripts/install_service.sh").read_text()
        server = (ROOT / "scripts/run_server.sh").read_text()
        downloader = (ROOT / "scripts/download_model.py").read_text()
        self.assertIn("manifest_profile=gsq-iq3xxs", installer)
        self.assertIn("model_alias=qwen38-flash-next-gsq-iq3xxs", installer)
        self.assertIn("MODEL_ALIAS", server)
        self.assertIn('"gsq-iq3xxs.json"', downloader)
        for field in ("block_topk_patch_sha256", "context_gated_mtp_patch_sha256"):
            self.assertEqual(len(self.result["runtime"][field]), 64)
        self.assertNotIn("/home/" + "sxuf", (ROOT / "results/gsq-iq3xxs.json").read_text())


if __name__ == "__main__":
    unittest.main()
