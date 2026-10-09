#!/usr/bin/env python3
"""Bounded native-image-history/cache check. Run on an idle server, not a benchmark."""
import argparse
import base64
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import MODEL, request_json, two_colors


def image(reverse=False):
    url = "data:image/png;base64," + base64.b64encode(two_colors(reverse)).decode()
    return {"role": "user", "content": [
        {"type": "text", "text": "Name the two dominant colors in the current image, from left to right. Reply with only their names."},
        {"type": "image_url", "image_url": {"url": url}}]}


def checked(result, label, order, cache):
    choice = result["choices"][0]
    answer = choice["message"].get("content", "").strip()
    value = answer.lower()
    if choice["finish_reason"] != "stop" or any(c not in value for c in order) or value.index(order[0]) >= value.index(order[1]):
        raise SystemExit(label + ": expected a complete, correctly ordered image answer")
    usage = result.get("usage", {})
    hit = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    prompt = usage.get("prompt_tokens")
    if not isinstance(hit, int) or not isinstance(prompt, int) or not 0 <= hit <= prompt:
        raise SystemExit(label + ": missing or invalid cached-token usage")
    if (cache == "hit" and hit <= 0) or (cache == "miss" and hit != 0):
        raise SystemExit(label + ": expected cache " + cache + ", received " + str(hit))
    return {"label": label, "answer": answer, "prompt_tokens": prompt,
            "cached_tokens": hit, "finish_reason": choice["finish_reason"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    # Unique prefix makes the first request cold even if this smoke ran before.
    system = {"role": "system", "content": "Verification session " + uuid.uuid4().hex + ". " +
              "Read the current image, not earlier images. Reply with only two color names, left to right.\n" +
              "Stable image-history prefix. " * 120}
    rows = []
    def call(label, messages, order, cache):
        result = request_json(base, "/v1/chat/completions", {
            "model": MODEL, "messages": messages, "max_tokens": 64,
            "temperature": 0, "reasoning_effort": "none", "stream": False})
        row = checked(result, label, order, cache)
        rows.append(row)
        return row["answer"]
    original = [system, image()]
    cold = call("image_cold", original, ("red", "blue"), "miss")
    warm = call("image_warm", original, ("red", "blue"), "hit")
    if cold != warm:
        raise SystemExit("Cold/warm answers differ")
    extended = original + [{"role": "assistant", "content": cold},
                           {"role": "user", "content": "Repeat the two colors from left to right. Reply with only their names."}]
    call("appended_text", extended, ("red", "blue"), "hit")
    changed = [system, image(True)]
    changed_cold = call("changed_image", changed, ("blue", "red"), "miss")
    changed_warm = call("changed_image_warm", changed, ("blue", "red"), "hit")
    if changed_cold != changed_warm:
        raise SystemExit("Changed-image cold/warm answers differ")
    history = [system]
    for _ in range(4):
        history.extend([image(), {"role": "assistant", "content": cold}])
    history.append(image(True))
    call("five_image_history", history, ("blue", "red"), "miss")
    print(json.dumps({"model": MODEL, "vision_cache_pass": True, "history_image_count": 5,
                      "rows": rows, "scope": "Functional check, not a 64-image capacity test or performance benchmark."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
