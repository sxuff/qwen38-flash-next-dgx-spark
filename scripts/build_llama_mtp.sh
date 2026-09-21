#!/usr/bin/env bash
set -euo pipefail

commit="797da982b488254b11f844718c30e0c41f34d718"
destination="${1:-$HOME/src/llama.cpp-qwen38-flash-next-next}"
jobs="${JOBS:-2}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
block_patch="$repo_root/patches/block-topk.patch"
gate_patch="$repo_root/patches/context-gated-mtp46.patch"
block_sha="1c8c953bae42a9b9765330b802a3d8ce38d0f15dd09b0f12242103010c152814"
gate_sha="68f6e9e3ea7999aabeabcc253dd78faa985a63f91299256a95bdda01945dada7"

[[ "$(uname -m)" == "aarch64" ]] || { printf 'error: this recipe requires aarch64 GB10\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null
command -v cmake >/dev/null
command -v git >/dev/null
command -v sha256sum >/dev/null
nvidia-smi >/dev/null
printf '%s  %s\n' "$block_sha" "$block_patch" | sha256sum -c -
printf '%s  %s\n' "$gate_sha" "$gate_patch" | sha256sum -c -

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

git -C "$destination" fetch --depth=1 origin "$commit"
actual="$(git -C "$destination" rev-parse FETCH_HEAD)"
[[ "$actual" == "$commit" ]] || { printf 'error: fetched %s, expected %s\n' "$actual" "$commit" >&2; exit 1; }
git -C "$destination" checkout --detach "$commit"

if git -C "$destination" apply --reverse --check "$gate_patch" >/dev/null 2>&1; then
  printf 'patch set already applied\n'
else
  [[ -z "$(git -C "$destination" status --porcelain)" ]] || {
    printf 'error: source tree has changes not recognized as the complete recipe patch set\n' >&2
    exit 1
  }
  git -C "$destination" apply --check "$block_patch"
  git -C "$destination" apply "$block_patch"
  git -C "$destination" apply --check "$gate_patch"
  git -C "$destination" apply "$gate_patch"
fi

git -C "$destination" diff --check
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
build="$destination/build-gb10-next"
cmake -S "$destination" -B "$build" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build "$build" --config Release --target llama-server test-arg-parser -j"$jobs"

binary="$build/bin/llama-server"
[[ -x "$binary" ]]
"$build/bin/test-arg-parser"
"$binary" --version
"$binary" --list-devices
printf 'base_commit=%s\nblock_topk_patch_sha256=%s\ncontext_gated_mtp_patch_sha256=%s\n' \
  "$commit" "$block_sha" "$gate_sha" > "$build/recipe-patchset.txt"
printf 'verified llama-server base=%s binary=%s\n' "$commit" "$binary"
