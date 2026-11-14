#!/usr/bin/env bash
# Pod setup: toolchain + edge0-strata engine (sm_86) + server venv. Logs to /results/setup.log.
# Pinned: Edge0 repo commit, ggml (llama.cpp) commit from edge0-strata/EDGE0.md.
set -euo pipefail
EDGE0_COMMIT=da9cc526747759227714ddde0728763d95306c04
GGML_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
SRC=/opt/edge0
mkdir -p /results

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential python3 python3-venv python3-pip \
  curl ca-certificates fio pciutils numactl procps psmisc >/dev/null

if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/Edge0-AI/Edge0.git "$SRC"
fi
git -C "$SRC" fetch -q --depth 1 origin "$EDGE0_COMMIT" || true
git -C "$SRC" checkout -q "$EDGE0_COMMIT"

cd "$SRC/edge0-strata"
if [ ! -f third_party/llama.cpp/CMakeLists.txt ]; then
  mkdir -p third_party/llama.cpp
  curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/${GGML_COMMIT}.tar.gz" \
    | tar -xz --strip-components=1 -C third_party/llama.cpp
fi

export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build --target strata -j "$(nproc)"
ls -la build/strata

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

{
  echo "edge0_commit=$EDGE0_COMMIT"
  echo "ggml_commit=$GGML_COMMIT"
  echo "nvcc=$(nvcc --version | tail -1)"
  echo "strata_bin=$SRC/edge0-strata/build/strata"
} > /results/versions.txt
echo SETUP_DONE
