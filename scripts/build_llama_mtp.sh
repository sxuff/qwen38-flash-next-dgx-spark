#!/usr/bin/env bash
set -euo pipefail

commit="d1a92352cbd417fd840b4e765c0b82f5fe3d1d89"
destination="${1:-$HOME/src/llama.cpp-qwen38-flash-next-mtp}"
jobs="${JOBS:-2}"

[[ "$(uname -m)" == "aarch64" ]] || { printf 'error: this recipe requires aarch64 GB10\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null
command -v cmake >/dev/null
command -v git >/dev/null
nvidia-smi >/dev/null

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init
  git -C "$destination" remote add origin https://github.com/danielhanchen/llama.cpp.git
fi
remote="$(git -C "$destination" remote get-url origin)"
[[ "$remote" == "https://github.com/danielhanchen/llama.cpp.git" ]] || {
  printf 'error: unexpected origin: %s\n' "$remote" >&2
  exit 1
}

git -C "$destination" fetch --depth=1 origin "$commit"
actual="$(git -C "$destination" rev-parse FETCH_HEAD)"
[[ "$actual" == "$commit" ]] || { printf 'error: fetched %s, expected %s\n' "$actual" "$commit" >&2; exit 1; }
git -C "$destination" checkout --detach "$commit"

export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cmake -S "$destination" -B "$destination/build-gb10-mtp" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build "$destination/build-gb10-mtp" --config Release --target llama-server -j"$jobs"

binary="$destination/build-gb10-mtp/bin/llama-server"
[[ -x "$binary" ]]
"$binary" --version
"$binary" --list-devices
printf 'verified llama-server commit=%s binary=%s\n' "$commit" "$binary"
