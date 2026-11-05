#!/usr/bin/env bash
set -euo pipefail
# Use the host toolkit; never install a different driver or assume Blackwell.
COMMIT=cd959bae99ff4bed22bef2604f45241c09bbb8c8
BASE="${1:-$HOME/.cache/kaggle-tpu-lab}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) required; driver alone is insufficient.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version
mkdir -p "$BASE"
SOURCE="$BASE/llama.cpp"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "$SOURCE"
fi
if [[ -n "$(git -C "$SOURCE" status --porcelain)" ]]; then
  echo 'Refusing to overwrite a modified llama.cpp checkout.' >&2; exit 1
fi
if ! git -C "$SOURCE" cat-file -e "$COMMIT^{commit}"; then
  git -C "$SOURCE" fetch origin "$COMMIT"
fi
git -C "$SOURCE" checkout --detach "$COMMIT"
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$COMMIT" ]]
cmake -S "$SOURCE" -B "$SOURCE/build-cuda" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$SOURCE/build-cuda" --parallel "$JOBS" --target llama-server llama-cli llama-bench
"$SOURCE/build-cuda/bin/llama-server" --version
"$SOURCE/build-cuda/bin/llama-server" --list-devices
