#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-}"
llama_root="${2:-}"
profile="${3:-q1-iq1s}"
mtp_draft="${4:-}"
projector_arg="${5:-}"
[[ -n "$model_root" && -n "$llama_root" ]] || {
  printf 'usage: %s MODEL_ROOT LLAMA_ROOT [q1-iq1s|q3-q3kxl|q3-q3kxl-mtp3|q3-q3kxl-mtp4] [MTP_DRAFT] [MMPROJ]\n' "$0" >&2
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
    batch_size=2048
    ubatch_size=512
    ngram_mod=0
    spec_mode=mtp-3
    binary_rel=build-gb10-mtp/bin/llama-server
    fit_mode=off
    manifest_profile=q3-q3kxl
    ;;
  q3-q3kxl-mtp4)
    ctx_size=65536
    parallel=1
    batch_size=2048
    ubatch_size=512
    ngram_mod=0
    spec_mode=mtp-4
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
if [[ "$profile" == "q3-q3kxl-mtp3" || "$profile" == "q3-q3kxl-mtp4" ]]; then
  [[ -n "$mtp_draft" ]] || { printf 'error: MTP_DRAFT is required for %s\n' "$profile" >&2; exit 2; }
  supplied_mtp_path="$(realpath "$mtp_draft")"
  mtp_root="$(dirname "$(dirname "$supplied_mtp_path")")"
  mtp_manifest="$repo_root/manifests/q3-mtp-shared-q8.json"
  python3 "$repo_root/scripts/download_model.py" \
    --manifest "$mtp_manifest" \
    --destination "$mtp_root" \
    --verify-only
  mtp_entry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' "$mtp_manifest")"
  verified_mtp_path="$(realpath "$mtp_root/$mtp_entry")"
  [[ "$supplied_mtp_path" == "$verified_mtp_path" ]] || {
    printf 'error: supplied MTP_DRAFT is not the manifest-verified artifact: %s\n' "$supplied_mtp_path" >&2
    exit 2
  }
  mtp_draft_path="$verified_mtp_path"
fi
mmproj_path=""
if [[ "$profile" == "q3-q3kxl" || "$profile" == "q3-q3kxl-mtp3" || "$profile" == "q3-q3kxl-mtp4" ]]; then
  projector="${projector_arg:-$model_root/mmproj-F16.gguf}"
  if [[ -n "$projector_arg" && ! -f "$projector" ]]; then
    printf 'error: supplied MMPROJ does not exist: %s\n' "$projector" >&2
    exit 2
  fi
  if [[ -f "$projector" ]]; then
    supplied_projector_path="$(realpath "$projector")"
    projector_root="$(dirname "$supplied_projector_path")"
    projector_manifest="$repo_root/manifests/q3-mmproj-f16.json"
    python3 "$repo_root/scripts/download_model.py" \
      --manifest "$projector_manifest" \
      --destination "$projector_root" \
      --verify-only
    projector_entry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' "$projector_manifest")"
    verified_projector_path="$(realpath "$projector_root/$projector_entry")"
    [[ "$supplied_projector_path" == "$verified_projector_path" ]] || {
      printf 'error: supplied MMPROJ is not the manifest-verified artifact: %s\n' "$supplied_projector_path" >&2
      exit 2
    }
    mmproj_path="$verified_projector_path"
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
  printf 'UBATCH_SIZE=%q\n' "${ubatch_size:-64}"
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
