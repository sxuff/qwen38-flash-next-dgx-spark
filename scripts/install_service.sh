#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-}"
llama_root="${2:-}"
[[ -n "$model_root" && -n "$llama_root" ]] || {
  printf 'usage: %s MODEL_ROOT LLAMA_ROOT\n' "$0" >&2
  exit 2
}
model_root="$(realpath "$model_root")"
llama_root="$(realpath "$llama_root")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$repo_root/scripts/download_model.py" --destination "$model_root" --verify-only
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
} > "$HOME/.config/qwen38-flash-next/server.env"
chmod 0600 "$HOME/.config/qwen38-flash-next/server.env"

systemctl --user daemon-reload
systemctl --user cat qwen38-flash-next-llama.service >/dev/null
printf 'installed. Start with: systemctl --user enable --now qwen38-flash-next-llama.service\n'
