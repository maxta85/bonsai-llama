#!/usr/bin/env bash
# Clone and prepare the PrismML-Eng/llama.cpp fork (prism branch) which adds
# Q1_0 (1-bit) and Q2_0 (ternary) g128 kernels for CPU, Metal, CUDA, Vulkan.
#
# This does NOT build it (that can take a while and depends on your backend);
# it just clones + configures CMake so you can `cmake --build` next.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMACPP_DIR="$ROOT/third_party/llama.cpp"
BUILD_DIR="$ROOT/build/llamacpp"

echo "Cloning PrismML-Eng/llama.cpp (prism branch) into $LLAMACPP_DIR"
mkdir -p "$ROOT/third_party"
if [ -d "$LLAMACPP_DIR/.git" ]; then
    echo "  already cloned; pulling latest"
    git -C "$LLAMACPP_DIR" fetch --depth 1 origin prism
    git -C "$LLAMACPP_DIR" checkout prism
    git -C "$LLAMACPP_DIR" reset --hard origin/prism
else
    git clone --depth 1 --branch prism \
        https://github.com/PrismML-Eng/llama.cpp "$LLAMACPP_DIR"
fi

echo "Configuring CMake in $BUILD_DIR"
# Pick a backend via env vars (defaults to CPU):
#   BONSAI_BACKEND=cuda   -> -DGGML_CUDA=ON
#   BONSAI_BACKEND=metal  -> -DGGML_METAL=ON   (macOS)
#   BONSAI_BACKEND=vulkan -> -DGGML_VULKAN=ON
#   BONSAI_BACKEND=rocm   -> -DGGML_HIP=ON
BACKEND_FLAG=""
case "${BONSAI_BACKEND:-cpu}" in
    cuda)   BACKEND_FLAG="-DGGML_CUDA=ON" ;;
    metal)  BACKEND_FLAG="-DGGML_METAL=ON" ;;
    vulkan) BACKEND_FLAG="-DGGML_VULKAN=ON" ;;
    rocm)   BACKEND_FLAG="-DGGML_HIP=ON" ;;
    cpu)    BACKEND_FLAG="" ;;
    *) echo "Unknown BONSAI_BACKEND=$BONSAI_BACKEND"; exit 1 ;;
esac

cmake -B "$BUILD_DIR" -S "$LLAMACPP_DIR" \
    -DCMAKE_BUILD_TYPE=Release $BACKEND_FLAG

echo
echo "Done. Now build with:"
echo "  cmake --build $BUILD_DIR -j --target llama-cli llama-server"
echo
echo "Then run a Bonsai model:"
echo "  $BUILD_DIR/bin/llama-cli -hf prism-ml/Ternary-Bonsai-8B-gguf:Q2_0 -p 'Hello'"
