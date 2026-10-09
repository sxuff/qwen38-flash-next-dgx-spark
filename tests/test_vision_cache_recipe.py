"""Public image-cache contracts and smoke negative-path tests, no GPU needed."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(answer="red, blue", cached=0, prompt=100, finish="stop"):
    return {"choices": [{"message": {"content": answer}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": cached}}}


class VisionCacheRecipe(unittest.TestCase):
    def test_launcher_print_and_constructor_set_64_images(self):
        module = load("serve")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            module.main(["--model", "/model", "--vision-weights", "/tower", "--print-config"])
        config = json.loads(out.getvalue())
        self.assertEqual(config["vision_max_images"], 64)
        self.assertEqual(config["vision_prefix_cache"], "same-complete-media-history-v1")
        tree = ast.parse((ROOT / "scripts/serve.py").read_text())
        app = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "App")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in app.keywords}
        self.assertEqual(kwargs["vision_max_images"], 64)

    def test_patch_hash_applied_before_runtime_install(self):
        receipt = json.loads((ROOT / "results/vision-prefix-cache.json").read_text())
        patch_path = ROOT / receipt["patch"]["path"]
        sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        self.assertEqual(sha, receipt["patch"]["sha256"])
        docker = (ROOT / "docker/Dockerfile").read_text()
        self.assertIn(sha, docker)
        self.assertIn(receipt["runtime_revision"], docker)
        self.assertIn("apply --check", docker)
        self.assertLess(docker.index("apply /opt/recipe/patches/vision-prefix-cache.patch"), docker.index("pip install --no-deps --no-build-isolation /opt/TensorFold"))
        source = patch_path.read_text()
        self.assertIn("src/tensorfold/families/qwen4_exp/cuda/vision_prefix.py", source)
        self.assertIn("tests/test_flashnext_vision_prefix.py", source)

    def test_receipt_records_actual_hits_and_conservative_misses(self):
        receipt = json.loads((ROOT / "results/vision-prefix-cache.json").read_text())
        cases = {r["label"]: r for r in receipt["live_functional_checks"]["image_cases"]}
        self.assertEqual(cases["vision_cold"]["cached_tokens"], 0)
        self.assertEqual(cases["vision_warm"]["cached_tokens"], 4164)
        self.assertEqual(cases["vision_text_extension"]["cached_tokens"], 4164)
        self.assertEqual(cases["changed_same_size_image"]["cached_tokens"], 0)
        self.assertEqual(cases["added_image_history"]["cached_tokens"], 0)
        branches = receipt["live_functional_checks"]["concurrent_branches"]
        self.assertEqual(len(branches), 2)
        self.assertEqual(branches[0]["answer"], branches[1]["answer"])
        self.assertEqual(min(r["cached_tokens"] for r in branches), 0)
        self.assertGreater(max(r["cached_tokens"] for r in branches), 0)
        self.assertEqual(receipt["image_history"]["history_image_count"], 5)

    def test_smoke_runs_unchanged_changed_appended_and_five_image_history(self):
        module = load("smoke_vision_cache")
        replies = [reply(), reply(cached=99), reply(cached=99),
                   reply("blue, red"), reply("blue, red", cached=99), reply("blue, red")]
        with patch.object(module, "request_json", side_effect=replies) as request, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(module.main(["--base-url", "http://localhost:8001/v1"]), 0)
        report = json.loads(out.getvalue())
        self.assertTrue(report["vision_cache_pass"])
        self.assertEqual(len(report["rows"]), 6)
        self.assertEqual(report["history_image_count"], 5)
        payloads = [call.args[2] for call in request.call_args_list]
        self.assertEqual(request.call_args_list[0].args[0], "http://localhost:8001")
        self.assertEqual(payloads[0], payloads[1])
        self.assertNotEqual(payloads[0]["messages"][1], payloads[3]["messages"][1])
        images = sum(1 for m in payloads[-1]["messages"] if isinstance(m["content"], list)
                     for c in m["content"] if c["type"] == "image_url")
        self.assertEqual(images, 5)
        self.assertTrue(all(p["reasoning_effort"] == "none" for p in payloads))

    def test_smoke_rejects_cache_miss_when_hit_required(self):
        module = load("smoke_vision_cache")
        with self.assertRaisesRegex(SystemExit, "expected cache hit"):
            module.checked(reply(), "fixture", ("red", "blue"), "hit")

    def test_smoke_rejects_stale_hit_after_changed_media(self):
        module = load("smoke_vision_cache")
        with self.assertRaisesRegex(SystemExit, "expected cache miss"):
            module.checked(reply("blue, red", cached=99), "fixture", ("blue", "red"), "miss")

    def test_smoke_rejects_missing_usage_and_invalid_counts(self):
        module = load("smoke_vision_cache")
        data = reply()
        data.pop("usage")
        for bad in [data, reply(cached=-1), reply(cached=101)]:
            with self.assertRaisesRegex(SystemExit, "invalid cached-token usage"):
                module.checked(bad, "fixture", ("red", "blue"), "miss")

    def test_smoke_rejects_wrong_image_answer_or_incomplete_finish(self):
        module = load("smoke_vision_cache")
        for bad in [reply("blue, red"), reply("red"), reply(finish="length")]:
            with self.assertRaisesRegex(SystemExit, "correctly ordered image answer"):
                module.checked(bad, "fixture", ("red", "blue"), "miss")

    def test_smoke_rejects_changed_cold_warm_answer(self):
        module = load("smoke_vision_cache")
        with patch.object(module, "request_json", side_effect=[reply(), reply("red blue", cached=99)]):
            with self.assertRaisesRegex(SystemExit, "Cold/warm answers differ"):
                module.main([])


if __name__ == "__main__":
    unittest.main()
