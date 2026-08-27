#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-}"
llama_root="${2:-}"
profile="${3:-q1-iq1s}"
[[ -n "$model_root" && -n "$llama_root" ]] || {
  printf 'usage: %s MODEL_ROOT LLAMA_ROOT [q1-iq1s|q3-q3kxl]\n' "$0" >&2
  exit 2
}
model_root="$(realpath "$model_root")"
llama_root="$(realpath "$llama_root")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$profile" in
  q1-iq1s)
    ctx_size=262144
    parallel=8
    batch_size=2048
    ngram_mod=0
    ;;
  q3-q3kxl)
    ctx_size=65536
    parallel=1
    batch_size=512
    ngram_mod=1
    ;;
  *) printf 'error: unknown profile: %s\n' "$profile" >&2; exit 2 ;;
esac
manifest="$repo_root/manifests/$profile.json"
python3 "$repo_root/scripts/download_model.py" --manifest "$manifest" --destination "$model_root" --verify-only
model_entry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' "$manifest")"
[[ -x "$llama_root/build-gb10-pr27742/bin/llama-server" ]] || {
  printf 'error: pinned llama-server binary is missing\n' >&2
  exit 1
}

install -d -m 0700 "$HOME/.config/qwen38-flash-next"
install -d -m 0755 "$HOME/.config/systemd/user" "$HOME/.local/lib/qwen38-flash-next"
install -m 0755 "$repo_root/scripts/run_server.sh" "$HOME/.local/lib/qwen38-flash-next/run_server.sh"
install -m 0644 "$repo_root/systemd/qwen38-flash-next-llama.service" "$HOME/.config/systemd/user/qwen38-flash-next-llama.service"

{
  printf 'MODEL_ROOT=%q\n' "$model_root"
  printf 'LLAMA_ROOT=%q\n' "$llama_root"
  printf 'MODEL_ENTRY=%q\n' "$model_entry"
  printf 'CTX_SIZE=%q\n' "$ctx_size"
  printf 'PARALLEL=%q\n' "$parallel"
  printf 'BATCH_SIZE=%q\n' "$batch_size"
  printf 'UBATCH_SIZE=64\n'
  printf 'NGRAM_MOD=%q\n' "$ngram_mod"
} > "$HOME/.config/qwen38-flash-next/server.env"
chmod 0600 "$HOME/.config/qwen38-flash-next/server.env"

systemctl --user daemon-reload
systemctl --user cat qwen38-flash-next-llama.service >/dev/null
printf 'installed. Start with: systemctl --user enable --now qwen38-flash-next-llama.service\n'
