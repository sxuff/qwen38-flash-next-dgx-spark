#!/usr/bin/env python3
"""The pinned FlashNext engine and native OpenAI server, without benchmark imports."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading

MODEL_ID = "qwen38-flash-next-tf405"
RUNTIME_SHA = "609ca419abecebdc5a059498a613680bd3aa847f"


def engine_options():
    return dict(depth=6, confidence=0.70, max_len=262144, context_explicit=True,
                tp=1, streams=2, prefetch=False, graphs=True, kv_dtype="bf16", vision=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TensorFold EXL3 4.05, 262K context, native vision")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vision-weights", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--print-config", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.port <= 65535:
        p.error("port outside 1..65535")
    return a


def main(argv=None):
    a = parse_args(argv)
    config = dict(model_id=MODEL_ID, runtime_revision=RUNTIME_SHA, engine=engine_options(),
                  host=a.host, port=a.port, output_default_tokens=32768, thinking_default=False)
    if a.print_config:
        print(json.dumps(config, indent=2))
        return 0
    if not a.model.is_dir() or not a.vision_weights.is_file():
        raise SystemExit("Verified model directory and BF16 vision tower are required")
    raw = json.loads((a.model / "config.json").read_text())
    if raw.get("model_type") != "qwen4_exp" or raw.get("quantization_config", {}).get("quant_method") != "exl3":
        raise SystemExit("Expected the pinned qwen4_exp EXL3 checkpoint")
    os.environ["TENSORFOLD_VISION_WEIGHTS"] = str(a.vision_weights.resolve())
    os.environ.setdefault("TENSORFOLD_MEMORY_RESERVE_GIB", "7")
    os.environ.setdefault("TENSORFOLD_NO_UPDATE_CHECK", "1")
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.cuda.server import App, Server, make_handler
    engine = FlashNextEngine(a.model, **engine_options())
    server = None
    try:
        app = App(engine, a.model, MODEL_ID, default_thinking=False, max_tokens=32768,
                  context_window=262144, sampling={"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0})
        server = Server((a.host, a.port), make_handler(app))
        def stop(_signum, _frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        print(json.dumps(config), flush=True)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
