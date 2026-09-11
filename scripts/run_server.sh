#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${LLAMA_ROOT:?LLAMA_ROOT is required}"

binary_rel="${BINARY_REL:-build-gb10-pr27742/bin/llama-server}"
binary="$LLAMA_ROOT/$binary_rel"
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
spec_mode="${SPEC_MODE:-}"
mtp_draft_path="${MTP_DRAFT_PATH:-}"
fit_mode="${FIT_MODE:-default}"
mmproj_path="${MMPROJ_PATH:-}"

if [[ -z "$spec_mode" ]]; then
  if [[ "$ngram_mod" == "1" ]]; then
    spec_mode="ngram-mod"
  elif [[ "$ngram_mod" == "0" ]]; then
    spec_mode="none"
  else
    printf 'NGRAM_MOD must be 0 or 1\n' >&2
    exit 1
  fi
fi

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

case "$fit_mode" in
  default) ;;
  off) args+=(--fit off) ;;
  *) printf 'FIT_MODE must be default or off\n' >&2; exit 1 ;;
esac

case "$spec_mode" in
  none) ;;
  ngram-mod)
    args+=(
      --spec-type ngram-mod
      --spec-ngram-mod-n-match 24
      --spec-ngram-mod-n-min 48
      --spec-ngram-mod-n-max 64
    )
    ;;
  mtp-2|mtp-3|mtp-4)
    [[ -f "$mtp_draft_path" ]] || { printf 'missing MTP sidecar: %s\n' "$mtp_draft_path" >&2; exit 1; }
    depth="${spec_mode#mtp-}"
    args+=(
      -md "$mtp_draft_path"
      --spec-type draft-mtp
      --spec-draft-n-max "$depth"
      --spec-draft-n-min 0
    )
    ;;
  *) printf 'SPEC_MODE must be none, ngram-mod, mtp-2, mtp-3, or mtp-4\n' >&2; exit 1 ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'COMMAND='
  printf '%q ' "$binary" "${args[@]}"
  printf '\n'
  exit 0
fi

exec "$binary" "${args[@]}"
