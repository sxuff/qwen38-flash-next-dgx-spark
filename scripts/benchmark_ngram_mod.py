#!/usr/bin/env python3
"""Paired ngram-mod benchmark against an authenticated llama.cpp server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fixture_python() -> str:
    rows = [
        "from dataclasses import dataclass",
        "",
        "@dataclass(frozen=True)",
        "class Event:",
        "    sequence: int",
        "    topic: str",
        "    payload: str",
        "",
        "EVENTS = [",
    ]
    for i in range(48):
        rows.append(f'    Event(sequence={i}, topic="build.step.{i % 6}", payload="artifact-{i:03d}"),')
    rows.extend([
        "]",
        "",
        "def by_topic(events: list[Event]) -> dict[str, list[Event]]:",
        "    result: dict[str, list[Event]] = {}",
        "    for event in events:",
        "        result.setdefault(event.topic, []).append(event)",
        "    return result",
    ])
    return "\n".join(rows)


def fixture_json() -> str:
    rows = []
    for i in range(36):
        rows.append({
            "id": f"task-{i:03d}",
            "owner": f"worker-{i % 4}",
            "state": ["queued", "running", "passed"][i % 3],
            "attempt": i % 2 + 1,
        })
    return json.dumps(rows, indent=2, separators=(",", ": "))


def cases() -> list[dict]:
    py = fixture_python()
    js = fixture_json()
    return [
        {
            "id": "copy_python",
            "max_tokens": 1800,
            "messages": [{"role": "user", "content": (
                "Return exactly the text between BEGIN and END. Do not add fences, commentary, or the markers.\n"
                f"BEGIN\n{py}\nEND"
            )}],
        },
        {
            "id": "copy_json",
            "max_tokens": 2200,
            "messages": [{"role": "user", "content": (
                "Return exactly the JSON between BEGIN and END. Do not add fences, commentary, or the markers.\n"
                f"BEGIN\n{js}\nEND"
            )}],
        },
        {
            "id": "structured_transform",
            "max_tokens": 700,
            "messages": [{"role": "user", "content": (
                "Using the task records below, return a JSON array containing only records whose state is passed. "
                "Preserve id, owner, state, and attempt exactly. Return JSON only.\n" + js
            )}],
        },
        {
            "id": "novel_code",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": (
                "Write a Python function schedule_jobs(durations, workers) that assigns each positive integer "
                "duration to the currently least-loaded worker. Preserve input order for ties, reject workers below "
                "one, and do not mutate durations. Return only valid Python source with no code fence."
            )}],
        },
    ]


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            try:
                out[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return out


class Client:
    def __init__(self, base_url: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def raw(self, path: str, payload: dict | None = None, timeout: int = 900) -> bytes:
        data = None if payload is None else canonical(payload)
        request = urllib.request.Request(self.base + path, data=data, headers=self.headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}")
            return response.read()

    def json(self, path: str, payload: dict | None = None, timeout: int = 900) -> dict | list:
        return json.loads(self.raw(path, payload, timeout))

    def metrics(self) -> dict[str, float]:
        return parse_metrics(self.raw("/metrics", timeout=30).decode())


def delta(after: dict[str, float], before: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--arm", required=True, choices=("baseline", "ngram-mod"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    api_key = os.environ.get("QWEN38_GX10_API_KEY", "")
    client = Client(args.base_url, api_key)

    model_body = client.json("/v1/models", timeout=30)
    if not isinstance(model_body, dict):
        raise RuntimeError("/v1/models did not return an object")
    model = model_body["data"][0]["id"]
    slots_before = client.json("/slots", timeout=30)
    rows = []

    for case in cases():
        payload = {
            "model": model,
            "messages": case["messages"],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": case["max_tokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        before = client.metrics()
        started = time.monotonic_ns()
        raw = client.raw("/v1/chat/completions", payload, timeout=1200)
        elapsed_ns = time.monotonic_ns() - started
        after = client.metrics()
        response = json.loads(raw)
        choice = response["choices"][0]
        content = choice["message"].get("content") or ""
        usage = response.get("usage") or {}
        predicted_tokens = delta(after, before, "llamacpp:tokens_predicted_total")
        predicted_seconds = delta(after, before, "llamacpp:tokens_predicted_seconds_total")
        draft_tokens = delta(after, before, "llamacpp:spec_decode_num_draft_tokens_total")
        accepted_tokens = delta(after, before, "llamacpp:spec_decode_num_accepted_tokens_total")
        rows.append({
            "id": case["id"],
            "request_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "elapsed_seconds": elapsed_ns / 1e9,
            "server_predicted_tokens": predicted_tokens,
            "server_predicted_seconds": predicted_seconds,
            "server_decode_tokens_per_second": predicted_tokens / predicted_seconds if predicted_seconds > 0 else None,
            "spec_draft_tokens": draft_tokens,
            "spec_accepted_tokens": accepted_tokens,
            "spec_acceptance_rate": accepted_tokens / draft_tokens if draft_tokens > 0 else None,
            "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        })

    slots_after = client.json("/slots", timeout=30)
    result = {
        "schema_version": 1,
        "arm": args.arm,
        "model": model_body,
        "slots_before": slots_before,
        "slots_after": slots_after,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    print(json.dumps({
        "arm": args.arm,
        "row_count": len(rows),
        "total_completion_tokens": sum(int(r["usage"].get("completion_tokens", 0)) for r in rows),
        "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
        "total_draft_tokens": sum(r["spec_draft_tokens"] for r in rows),
        "total_accepted_tokens": sum(r["spec_accepted_tokens"] for r in rows),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
