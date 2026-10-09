#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -m)" == aarch64 ]] || { printf 'GB10 aarch64 host required\n' >&2; exit 1; }
python3 -c 'import shutil; assert shutil.disk_usage(".").free >= 20*1024**3, "20 GiB free disk required"'
docker build --platform linux/arm64 -f "$ROOT/docker/Dockerfile" -t qwen38-flash-next-tensorfold:609ca419 "$ROOT"
docker run --rm --network none --entrypoint python3 qwen38-flash-next-tensorfold:609ca419 -c 'import torch,triton,tensorfold; from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine; from tensorfold.families.qwen4_exp.cuda import vision_prefix; from tensorfold.cuda.server import App,Server,make_handler; from transformers import Qwen2VLImageProcessor; assert vision_prefix.reusable(None,None); print("Patched TensorFold runtime imports passed",torch.__version__,triton.__version__,tensorfold.__version__)'
