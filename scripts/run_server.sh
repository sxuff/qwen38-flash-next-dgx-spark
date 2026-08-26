#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${LLAMA_ROOT:?LLAMA_ROOT is required}"

binary="$LLAMA_ROOT/build-gb10-pr27742/bin/llama-server"
model="$MODEL_ROOT/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"

[[ "$(uname -m)" == "aarch64" ]] || { printf 'aarch64 required\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null
nvidia-smi >/dev/null
[[ -x "$binary" ]] || { printf 'missing llama-server: %s\n' "$binary" >&2; exit 1; }
[[ -f "$model" ]] || { printf 'missing model entry shard: %s\n' "$model" >&2; exit 1; }

mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( mem_available_kib >= 6291456 )) || { printf 'less than 6 GiB MemAvailable\n' >&2; exit 1; }

exec "$binary" \
  --model "$model" \
  -c 262144 \
  -np 8 \
  --no-kv-unified \
  -b 2048 \
  -ub 64 \
  -ngl 99 \
  -ot per_layer_token_embd=CPU \
  --cache-prompt \
  --cache-reuse 0 \
  --slot-prompt-similarity 0.10 \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --no-context-shift \
  --load-mode mmap \
  --metrics \
  --slots \
  --host 127.0.0.1 \
  --port 8001 \
  --no-webui
