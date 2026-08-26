#!/usr/bin/env bash
set -euo pipefail

commit="bea3b12daee45876b0129a3602dc8f534ce30bf0"
pr_ref="pull/27742/head"
destination="${1:-$HOME/src/llama.cpp-qwen38-flash-next}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  printf 'error: this recipe requires aarch64 GB10\n' >&2
  exit 1
fi
command -v nvidia-smi >/dev/null
command -v cmake >/dev/null
command -v git >/dev/null
nvidia-smi >/dev/null

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init
  git -C "$destination" remote add origin https://github.com/ggml-org/llama.cpp.git
fi

remote="$(git -C "$destination" remote get-url origin)"
[[ "$remote" == "https://github.com/ggml-org/llama.cpp.git" ]] || {
  printf 'error: unexpected origin: %s\n' "$remote" >&2
  exit 1
}

git -C "$destination" fetch --depth=1 origin "$pr_ref"
actual="$(git -C "$destination" rev-parse FETCH_HEAD)"
[[ "$actual" == "$commit" ]] || {
  printf 'error: PR head is %s, expected %s\n' "$actual" "$commit" >&2
  exit 1
}
git -C "$destination" checkout --detach "$commit"

cmake -S "$destination" -B "$destination/build-gb10-pr27742" \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DGGML_CURL=ON \
  -DGGML_RPC=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build "$destination/build-gb10-pr27742" --config Release --target llama-server -j"$(nproc)"

binary="$destination/build-gb10-pr27742/bin/llama-server"
[[ -x "$binary" ]]
ldd "$binary" | grep -E 'ggml-cuda|libcuda|libcudart' >/dev/null
"$binary" --list-devices
printf 'verified llama-server commit=%s binary=%s\n' "$commit" "$binary"
