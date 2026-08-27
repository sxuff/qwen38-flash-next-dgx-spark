#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${LLAMA_ROOT:?LLAMA_ROOT is required}"

binary="$LLAMA_ROOT/build-gb10-pr27742/bin/llama-server"
model="$MODEL_ROOT/${MODEL_ENTRY:-UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf}"

[[ "$(uname -m)" == "aarch64" ]] || { printf 'aarch64 required\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null
nvidia-smi >/dev/null
[[ -x "$binary" ]] || { printf 'missing llama-server: %s\n' "$binary" >&2; exit 1; }
[[ -f "$model" ]] || { printf 'missing model entry shard: %s\n' "$model" >&2; exit 1; }

mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( mem_available_kib >= 6291456 )) || { printf 'less than 6 GiB MemAvailable\n' >&2; exit 1; }

ctx_size="${CTX_SIZE:-262144}"
parallel="${PARALLEL:-8}"
batch_size="${BATCH_SIZE:-2048}"
ubatch_size="${UBATCH_SIZE:-64}"
ngram_mod="${NGRAM_MOD:-0}"
mmproj_path="${MMPROJ_PATH:-}"

args=(
  --model "$model"
  -c "$ctx_size"
  -np "$parallel"
  --no-kv-unified
  -b "$batch_size"
  -ub "$ubatch_size"
  -ngl 99
  -ot per_layer_token_embd=CPU
  --cache-prompt
  --cache-reuse 0
  --slot-prompt-similarity 0.10
  --cache-ram 0
  --no-cache-idle-slots
  --no-context-shift
  --load-mode mmap
  --metrics
  --slots
  --host 127.0.0.1
  --port 8001
  --no-webui
)

if [[ -n "$mmproj_path" ]]; then
  [[ -f "$mmproj_path" ]] || { printf 'missing multimodal projector: %s\n' "$mmproj_path" >&2; exit 1; }
  args+=(--mmproj "$mmproj_path")
fi

if [[ "$ngram_mod" == "1" ]]; then
  args+=(
    --spec-type ngram-mod
    --spec-ngram-mod-n-match 24
    --spec-ngram-mod-n-min 48
    --spec-ngram-mod-n-max 64
  )
elif [[ "$ngram_mod" != "0" ]]; then
  printf 'NGRAM_MOD must be 0 or 1\n' >&2
  exit 1
fi

exec "$binary" "${args[@]}"
