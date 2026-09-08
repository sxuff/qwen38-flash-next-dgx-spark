#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-}"
llama_root="${2:-}"
profile="${3:-q1-iq1s}"
mtp_draft="${4:-}"
projector_arg="${5:-}"
[[ -n "$model_root" && -n "$llama_root" ]] || {
  printf 'usage: %s MODEL_ROOT LLAMA_ROOT [q1-iq1s|q3-q3kxl|q3-q3kxl-mtp3] [MTP_DRAFT] [MMPROJ]\n' "$0" >&2
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
    spec_mode=none
    binary_rel=build-gb10-pr27742/bin/llama-server
    fit_mode=default
    manifest_profile=q1-iq1s
    ;;
  q3-q3kxl)
    ctx_size=65536
    parallel=1
    batch_size=512
    ngram_mod=1
    spec_mode=ngram-mod
    binary_rel=build-gb10-pr27742/bin/llama-server
    fit_mode=default
    manifest_profile=q3-q3kxl
    ;;
  q3-q3kxl-mtp3)
    ctx_size=65536
    parallel=1
    batch_size=512
    ngram_mod=0
    spec_mode=mtp-3
    binary_rel=build-gb10-mtp/bin/llama-server
    fit_mode=off
    manifest_profile=q3-q3kxl
    ;;
  *) printf 'error: unknown profile: %s\n' "$profile" >&2; exit 2 ;;
esac
manifest="$repo_root/manifests/$manifest_profile.json"
python3 "$repo_root/scripts/download_model.py" --manifest "$manifest" --destination "$model_root" --verify-only
model_entry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' "$manifest")"
mtp_draft_path=""
if [[ "$profile" == "q3-q3kxl-mtp3" ]]; then
  [[ -n "$mtp_draft" ]] || { printf 'error: MTP_DRAFT is required for q3-q3kxl-mtp3\n' >&2; exit 2; }
  mtp_draft_path="$(realpath "$mtp_draft")"
  mtp_root="$(dirname "$(dirname "$mtp_draft_path")")"
  python3 "$repo_root/scripts/download_model.py" \
    --manifest "$repo_root/manifests/q3-mtp-shared-q8.json" \
    --destination "$mtp_root" \
    --verify-only
fi
mmproj_path=""
if [[ "$profile" == "q3-q3kxl" || "$profile" == "q3-q3kxl-mtp3" ]]; then
  projector="${projector_arg:-$model_root/mmproj-F16.gguf}"
  if [[ -f "$projector" ]]; then
    projector_manifest="$repo_root/manifests/q3-mmproj-f16.json"
    python3 "$repo_root/scripts/download_model.py" \
      --manifest "$projector_manifest" \
      --destination "$(dirname "$projector")" \
      --verify-only
    mmproj_path="$(realpath "$projector")"
  else
    printf 'warning: %s is absent; Q3 will run text-only\n' "$projector" >&2
  fi
fi
[[ -x "$llama_root/$binary_rel" ]] || {
  printf 'error: pinned llama-server binary is missing\n' >&2
  exit 1
}

render_env() {
  printf 'MODEL_ROOT=%q\n' "$model_root"
  printf 'LLAMA_ROOT=%q\n' "$llama_root"
  printf 'MODEL_ENTRY=%q\n' "$model_entry"
  printf 'CTX_SIZE=%q\n' "$ctx_size"
  printf 'PARALLEL=%q\n' "$parallel"
  printf 'BATCH_SIZE=%q\n' "$batch_size"
  printf 'UBATCH_SIZE=64\n'
  printf 'NGRAM_MOD=%q\n' "$ngram_mod"
  printf 'SPEC_MODE=%q\n' "$spec_mode"
  printf 'BINARY_REL=%q\n' "$binary_rel"
  printf 'FIT_MODE=%q\n' "$fit_mode"
  printf 'MTP_DRAFT_PATH=%q\n' "$mtp_draft_path"
  printf 'MMPROJ_PATH=%q\n' "$mmproj_path"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  render_env
  exit 0
fi

install -d -m 0700 "$HOME/.config/qwen38-flash-next"
install -d -m 0755 "$HOME/.config/systemd/user" "$HOME/.local/lib/qwen38-flash-next"
install -m 0755 "$repo_root/scripts/run_server.sh" "$HOME/.local/lib/qwen38-flash-next/run_server.sh"
install -m 0644 "$repo_root/systemd/qwen38-flash-next-llama.service" "$HOME/.config/systemd/user/qwen38-flash-next-llama.service"
render_env > "$HOME/.config/qwen38-flash-next/server.env"
chmod 0600 "$HOME/.config/qwen38-flash-next/server.env"

systemctl --user daemon-reload
systemctl --user cat qwen38-flash-next-llama.service >/dev/null
printf 'installed. Start with: systemctl --user enable --now qwen38-flash-next-llama.service\n'
