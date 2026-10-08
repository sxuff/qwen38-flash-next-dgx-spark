#!/usr/bin/env python3
"""Verify model identity, context, default thinking, text and native image input."""
import argparse
import base64
import json
import os
import struct
import urllib.request
import zlib

MODEL = "qwen38-flash-next-tf405"


def two_colors():
    width, height = 256, 128
    raw = b"".join(b"\0" + b"\xff\0\0"*(width//2) + b"\0\0\xff"*(width//2) for _ in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def request_json(base, path, payload=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("QWEN_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base+path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    a = p.parse_args(argv)
    base = a.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health = request_json(base, "/health")
    if health.get("context_length") != 262144:
        raise SystemExit("Expected 262144-token serving context")
    models = request_json(base, "/v1/models")
    if MODEL not in [m["id"] for m in models.get("data", [])]:
        raise SystemExit("Served model identity mismatch")
    # No reasoning override: this must exercise the server's default.
    thinking = request_json(base, "/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "Solve 2x + 3 = 11. Give the value of x briefly."}],
        "max_tokens": 2048, "temperature": 0, "stream": False})
    choice = thinking["choices"][0]
    message = choice["message"]
    if not message.get("reasoning_content", "").strip() or "4" not in message.get("content", "") or choice["finish_reason"] != "stop":
        raise SystemExit("Thinking-default generation check failed")
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with exactly TEXT_OK."}],
               "max_tokens": 128, "temperature": 0, "reasoning_effort": "none", "stream": False}
    text = request_json(base, "/v1/chat/completions", payload)
    if text["choices"][0]["message"].get("content", "").strip() != "TEXT_OK" or text["choices"][0]["finish_reason"] != "stop":
        raise SystemExit("Text generation check failed")
    image = "data:image/png;base64," + base64.b64encode(two_colors()).decode()
    payload["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "Name the two dominant colors in this image, from left to right. Reply with just the color names."},
        {"type": "image_url", "image_url": {"url": image}}]}]
    result = request_json(base, "/v1/chat/completions", payload)
    colors = result["choices"][0]["message"].get("content", "").lower()
    if "red" not in colors or "blue" not in colors or colors.index("red") > colors.index("blue") or result["choices"][0]["finish_reason"] != "stop":
        raise SystemExit("Native image generation check failed")
    print(json.dumps({"model": MODEL, "context_tokens": health["context_length"], "thinking_default_pass": True,
                      "thinking_characters": len(message["reasoning_content"]), "text_pass": True,
                      "native_vision_pass": True, "text_response": text["choices"][0]["message"]["content"], "vision_response": colors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
