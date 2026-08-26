#!/usr/bin/env python3
import argparse
import json
import urllib.error
import urllib.request


def request_json(url: str, payload: dict | None = None, timeout: int = 60) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health_status, health = request_json(f"{base}/health", timeout=10)
    if health_status != 200:
        raise SystemExit(f"health failed: HTTP {health_status}")

    payload = {
        "messages": [{"role": "user", "content": "Reply with exactly: GB10 READY"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    status, result = request_json(f"{base}/v1/chat/completions", payload, timeout=300)
    choices = result.get("choices") or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    if status != 200 or not content.strip():
        raise SystemExit("generation failed or returned empty content")
    usage = result.get("usage", {})
    print(json.dumps({
        "health_http_status": health_status,
        "health": health,
        "generation_http_status": status,
        "nonempty_output": True,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage,
        "output": content,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
