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
RESULT = ROOT / "results/q3-q3kxl-next.json"
BLOCK_SHA = "1c8c953bae42a9b9765330b802a3d8ce38d0f15dd09b0f12242103010c152814"
GATE_SHA = "68f6e9e3ea7999aabeabcc253dd78faa985a63f91299256a95bdda01945dada7"


class Q3NextPublicTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads(RESULT.read_text())

    def test_result_identity_and_decision(self):
        self.assertEqual(self.result["status"], "completed-one-pass-deployment-comparison")
        decision = self.result["decision"]
        self.assertEqual(decision["deep_context_profile"], "block-top-k-context-gated-mtp46-q4")
        self.assertEqual(decision["deep_context_threshold_tokens"], 49152)
        self.assertEqual(decision["configured_deployment_context_tokens"], 262144)
        self.assertEqual(decision["benchmark_context_tokens"], 65536)
        self.assertEqual(decision["parallel_slots"], 1)

    def test_metrics_and_qualifications(self):
        before = self.result["before"]
        after = self.result["after"]
        delta = self.result["before_to_after"]
        self.assertAlmostEqual(before["long65_decode_tokens_per_second"], 49.700052418024036)
        self.assertAlmostEqual(after["long65_decode_tokens_per_second"], 63.54564861242123)
        self.assertAlmostEqual(delta["long65_decode_increase_fraction"], 0.27858313061609575)
        self.assertAlmostEqual(delta["long32_decode_increase_fraction"], 0.06075628038412595)
        self.assertAlmostEqual(delta["long65_prefill_increase_fraction"], 0.21299010967234654)
        self.assertAlmostEqual(delta["six_row_wall_reduction_fraction"], 0.10576901338902655)
        self.assertEqual(delta["exact_output_matches"], "6/6")
        self.assertIn("Deployment-to-deployment", self.result["claim_scope"])
        self.assertIn("not a full 256K stress test", self.result["deployment_256k_validation"]["qualification"])
        self.assertEqual(self.result["safety"]["maximum_service_swap_bytes"], 0)

    def test_patches_and_artifacts_are_exact(self):
        block = ROOT / "patches/block-topk.patch"
        gate = ROOT / "patches/context-gated-mtp46.patch"
        self.assertEqual(hashlib.sha256(block.read_bytes()).hexdigest(), BLOCK_SHA)
        self.assertEqual(hashlib.sha256(gate.read_bytes()).hexdigest(), GATE_SHA)
        manifest = json.loads((ROOT / "manifests/q3-mtp-shared-q4.json").read_text())
        self.assertEqual(manifest["revision"], "38bb39ee97821de2c9009abb7e93950eec396e66")
        self.assertEqual(manifest["variant"], "shared-Q4_K_M")
        self.assertEqual(manifest["total_bytes"], 1907151936)
        self.assertEqual(manifest["files"][0]["sha256"], "f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc")
        card = ROOT / "results/q3-q3kxl-next-card.png"
        self.assertTrue(card.is_file())
        self.assertEqual(hashlib.sha256(card.read_bytes()).hexdigest(), "d69596d4e34710338c08276d60c9685f2019c44020f8b1f4b1409eb85f818029")
        self.assertFalse((ROOT / "results/q3-q3kxl-mtp4-card.png").exists())
        self.assertEqual(self.result["result_card"]["width"], 1472)
        self.assertEqual(self.result["result_card"]["height"], 1312)

    def test_recipe_surface_matches_new_architecture(self):
        readme = (ROOT / "README.md").read_text()
        build = (ROOT / "scripts/build_llama_mtp.sh").read_text()
        server = (ROOT / "scripts/run_server.sh").read_text()
        installer = (ROOT / "scripts/install_service.sh").read_text()
        for value in (
            "49.70 tok/s",
            "63.55 tok/s",
            "27.86% higher",
            "deployment-to-deployment comparison",
            "q3-q3kxl-next",
            "not a complete 256K stress test",
            "results/q3-q3kxl-next.json",
            "results/q3-q3kxl-next-card.png",
        ):
            self.assertIn(value, readme)
        for value in (
            "797da982b488254b11f844718c30e0c41f34d718",
            BLOCK_SHA,
            GATE_SHA,
            "patches/block-topk.patch",
            "patches/context-gated-mtp46.patch",
            "test-arg-parser",
        ):
            self.assertIn(value, build)
        for value in (
            "mtp-46",
            "--spec-draft-n-max",
            "--spec-draft-p-min",
            "--spec-draft-mtp-context-threshold",
            "MTP_CONTEXT_THRESHOLD",
        ):
            self.assertIn(value, server)
        for value in (
            "q3-q3kxl-next",
            "ctx_size=262144",
            "spec_mode=mtp-46",
            "spec_p_min=0.75",
            "mtp_context_threshold=49152",
            "build-gb10-next",
            "q3-mtp-shared-q4.json",
        ):
            self.assertIn(value, installer)

    def test_installer_accepts_only_manifest_artifacts(self):
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            fixture = temp / "recipe"
            shutil.copytree(ROOT / "scripts", fixture / "scripts")
            shutil.copytree(ROOT / "manifests", fixture / "manifests")

            def zero_manifest(name):
                path = fixture / "manifests" / name
                manifest = json.loads(path.read_text())
                for entry in manifest["files"]:
                    entry["bytes"] = 0
                    entry["sha256"] = empty_sha256
                manifest["total_bytes"] = 0
                manifest["reserve_bytes"] = 0
                path.write_text(json.dumps(manifest))
                return manifest

            target_manifest = zero_manifest("q3-q3kxl.json")
            draft_manifest = zero_manifest("q3-mtp-shared-q4.json")
            projector_manifest = zero_manifest("q3-mmproj-f16.json")
            target_root = temp / "target"
            draft_root = temp / "draft"
            llama_root = temp / "llama"
            for entry in target_manifest["files"]:
                path = target_root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
            draft = draft_root / draft_manifest["files"][0]["path"]
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_bytes(b"")
            projector = target_root / projector_manifest["files"][0]["path"]
            projector.parent.mkdir(parents=True, exist_ok=True)
            projector.write_bytes(b"")
            binary = llama_root / "build-gb10-next/bin/llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("#!/usr/bin/env bash\nexit 0\n")
            binary.chmod(0o755)
            env = {**os.environ, "DRY_RUN": "1"}
            command = ["bash", str(fixture / "scripts/install_service.sh"), str(target_root), str(llama_root), "q3-q3kxl-next", str(draft), str(projector)]
            proc = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for value in ("CTX_SIZE=262144", "SPEC_MODE=mtp-46", "SPEC_P_MIN=0.75", "MTP_CONTEXT_THRESHOLD=49152"):
                self.assertIn(value, proc.stdout)
            bad = draft.parent / "UNVERIFIED.gguf"
            bad.write_bytes(b"bad")
            proc = subprocess.run(command[:-2] + [str(bad), str(projector)], env=env, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not the manifest-verified artifact", proc.stderr)

    def test_no_private_identifiers(self):
        paths = [ROOT / "README.md", RESULT, ROOT / "scripts/build_llama_mtp.sh", ROOT / "scripts/install_service.sh", ROOT / "scripts/run_server.sh"]
        text = "\n".join(path.read_text(errors="replace") for path in paths)
        for forbidden in ("/home/" + "sxuf", "gx10" + "-fe09", "172.17." + "0.1", "100.65." + "244.79"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
