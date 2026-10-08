"""CPU contracts for the public TensorFold launcher and independent safety guards."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name+".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServingRecipe(unittest.TestCase):
    def test_default_engine_matches_operational_recipe(self):
        module = load("serve")
        self.assertEqual(module.engine_options(), dict(depth=6, confidence=0.70, max_len=262144,
            context_explicit=True, tp=1, streams=2, prefetch=False, graphs=True, kv_dtype="bf16", vision=True))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(module.main(["--model", "/model", "--vision-weights", "/vision/tower", "--print-config"]), 0)
        config = json.loads(out.getvalue())
        self.assertEqual(config["output_default_tokens"], 32768)
        self.assertTrue(config["thinking_default"])
        self.assertEqual(config["model_id"], "qwen38-flash-next-tf405")
        receipt = json.loads((ROOT/"results/tensorfold-exl3-405.json").read_text())
        self.assertIs(receipt["public_recipe"]["thinking_default"], True)
        self.assertIn("Thinking default: on", (ROOT/"README.md").read_text())

    def test_actual_app_constructor_enables_thinking(self):
        tree = ast.parse((ROOT/"scripts/serve.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "App"]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
        self.assertIs(kwargs["default_thinking"], True)
        self.assertEqual(kwargs["max_tokens"], 32768)

    def test_smoke_checks_thinking_without_override(self):
        module = load("smoke")
        payloads = []
        def request(base, path, payload=None):
            if path == "/health": return {"context_length": 262144}
            if path == "/v1/models": return {"data": [{"id": module.MODEL}]}
            assert payload is not None
            payloads.append(dict(payload))
            if len(payloads) == 1:
                return {"choices": [{"message": {"reasoning_content": "fixture", "content": "x = 4"}, "finish_reason": "stop"}]}
            return {"choices": [{"message": {"content": "TEXT_OK" if len(payloads) == 2 else "red, blue"}, "finish_reason": "stop"}]}
        with patch.object(module, "request_json", side_effect=request), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(module.main([]), 0)
        self.assertNotIn("reasoning_effort", payloads[0])
        self.assertNotIn("chat_template_kwargs", payloads[0])
        self.assertEqual(payloads[0]["max_tokens"], 2048)
        self.assertTrue(json.loads(out.getvalue())["thinking_default_pass"])

    def test_smoke_rejects_missing_reasoning(self):
        module = load("smoke")
        replies = [{"context_length": 262144}, {"data": [{"id": module.MODEL}]},
                   {"choices": [{"message": {"content": "x = 4"}, "finish_reason": "stop"}]}]
        with patch.object(module, "request_json", side_effect=replies):
            with self.assertRaisesRegex(SystemExit, "Thinking-default"):
                module.main([])

    def test_reserve_swap_disk_boundaries(self):
        guard = load("supervise")
        baseline = dict(mem_available=100*guard.GIB, swap_used=2*guard.GIB, disk_free=25*guard.GIB)
        self.assertIsNone(guard.breach(baseline, baseline))
        self.assertIsNone(guard.breach({**baseline, "mem_available": 7*guard.GIB, "swap_used": 3*guard.GIB, "disk_free": 20*guard.GIB}, baseline))
        for changed, swap in [({"mem_available": 7*guard.GIB-1}, 0), ({"swap_used": 3*guard.GIB+1}, 0), ({"disk_free": 20*guard.GIB-1}, 0), ({}, 1)]:
            self.assertIsNotNone(guard.breach({**baseline, **changed}, baseline, swap))

    def test_docker_command_has_safe_defaults_and_exact_runtime(self):
        guard = load("supervise")
        args = guard.docker_command(Path("/weights with spaces"), Path("/tower"), Path("/cache"), "unit-test-container", 8001, Path("/cid"))
        for value in ["--gpus", "all", "--memory", "112g", "--memory-swap", "--read-only", "127.0.0.1:8001:8001", "qwen38-flash-next-tensorfold:609ca419", "TENSORFOLD_MEMORY_RESERVE_GIB=7", "/weights with spaces:/model:ro", "--vision-weights"]:
            self.assertIn(value, args)
        self.assertNotIn("--privileged", args)
        self.assertNotIn("--network=host", args)

    def test_dry_run_does_not_create_model_or_runtime_dirs(self):
        guard = load("supervise")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as out:
            model, tower = Path(temp)/"missing-model", Path(temp)/"vision_tower_bf16.safetensors"
            self.assertEqual(guard.main([str(model), str(tower), "--dry-run"]), 0)
            self.assertFalse(model.exists())
            self.assertFalse(tower.exists())
            self.assertIsInstance(json.loads(out.getvalue()), list)

    def test_service_render_uses_only_new_launcher_and_no_implicit_restart(self):
        installer = load("install_service")
        unit = installer.render(Path("/model with spaces"), Path("/tower"))
        self.assertIn("scripts/supervise.py", unit)
        self.assertIn('"/model with spaces"', unit)
        self.assertIn("Restart=no", unit)
        self.assertIn("MemorySwapMax=0", unit)
        self.assertNotIn("llama", unit.lower())
        for bad in ["line\nbreak", "null\x00"]:
            with self.assertRaises(ValueError):
                installer.systemd_quote(bad)

    def test_smoke_fixture_is_a_valid_png(self):
        module = load("smoke")
        data = module.two_colors()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(data), 100)

    def test_readme_installs_only_tensorfold(self):
        text = (ROOT/"README.md").read_text()
        for required in ["build_tensorfold.sh", "install_service.sh", "qwen38-flash-next-tensorfold.service", "manifests/exl3-405.json", "manifests/vision-bf16.json", "262,144-token context", "MTP6", "32,768 tokens"]:
            self.assertIn(required, text)
        for old in ["llama.cpp", "GGUF", "Earlier GSQ", "Recommended architecture", "build_llama", "MTP sidecar", "Q4_K_M", "q1-iq1s", "71.11 tok/s"]:
            self.assertNotIn(old, text)
        for old in ["scripts/build_llama.sh", "scripts/build_llama_mtp.sh", "systemd/qwen38-flash-next-llama.service", "results/gsq-iq3xxs.json", "patches/block-topk.patch"]:
            self.assertFalse((ROOT/old).exists())


if __name__ == "__main__":
    unittest.main()
